from __future__ import annotations
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import warnings
import itertools
warnings.filterwarnings("ignore")
from typing import Tuple, List, Dict, Union, Optional, Callable, Any


from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split



def feature_engineering(TRAIN_START_DATE:str = '2009-01-01',
                        TRAIN_END_DATE:str = '2020-07-01',
                        TEST_START_DATE:str = '2020-07-01',
                        TEST_END_DATE:str = '2021-10-31',
                        TICKER_LIST:list[str] = ['IVV', 'IEF'],
                        user_defined_features_df:pd.DataFrame = None
                        ) -> pd.DataFrame|pd.DataFrame|dict:

    """
    This is the main function for feature engineering.

    Parameters
    ----------
    TRAIN_START_DATE (str, optional): start date of training
    TRAIN_END_DATE (str, optional): end date of training
    TEST_START_DATE (str, optional): start date of testing
    TEST_END_DATE (str, optional): end date of testing
    TICKER_LIST (list[str], optional): list of ticker symbols
    user_defined_features_df (pd.DataFrame, optional): dataframe with user defined features

    Returns
    -------
    pd.DataFrame: training data
    pd.DataFrame: testing data
    dict: information dictionary
    """
    equity_df = YahooDownloader(start_date = TRAIN_START_DATE, end_date = TEST_END_DATE, ticker_list = TICKER_LIST).fetch_data()
    fe = FeatureEngineer(use_technical_indicator=False)
    equity_df = fe.preprocess_data(equity_df)
    feature_df = equity_df.merge(user_defined_features_df, on='date', how='left').dropna()

    # re-arrange the dataframe
    list_ticker = feature_df["tic"].unique().tolist()
    list_date = list(pd.date_range(feature_df['date'].min(),feature_df['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))
    feature_df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(feature_df,on=["date","tic"],how="left")
    feature_df_full = feature_df_full[feature_df_full['date'].isin(feature_df['date'])]
    feature_df_full = feature_df_full.sort_values(['date','tic'])
    feature_df_full = feature_df_full.fillna(0)

    train = data_split(feature_df_full, TRAIN_START_DATE,TRAIN_END_DATE)
    test = data_split(feature_df_full, TEST_START_DATE,TEST_END_DATE)

    INDICATORS = user_defined_features_df.columns[1:]
    STOCK_DIMENSION = len(train.tic.unique())
    STATE_SPACE = 1 + 2 * STOCK_DIMENSION + len(INDICATORS) * 2

    info_dict = {'indicator_list': INDICATORS,
                 'stock_dimension': STOCK_DIMENSION,
                 'state_space': STATE_SPACE
                }
    
    return train, test, info_dict


def get_monthly_date_format(date_string:str, kwargs: dict = {}) -> pd.Period:
    """
    Get the monthly date format for a given date string.

    Parameters
    ----------
    date_string : str
        The date string to parse.
    kwargs : dict, optional
        A dictionary of keyword arguments to pass to the function.
        
    Returns
    -------
    pd.Period
        The monthly date format.
    """
    monthly_date = pd.to_datetime(date_string, **kwargs).to_period('M')
    return monthly_date

def get_quarterly_date_format(date_string:str, kwargs: dict = {}) -> pd.Period:
    """
    Get the quarterly date format for a given date string.

    Parameters
    ----------
    date_string : str
        The date string to parse.
    kwargs : dict, optional
        A dictionary of keyword arguments to pass to the function.
        
    Returns
    -------
    pd.Period
        The quarterly date format.
    """
    monthly_date = pd.to_datetime(date_string, **kwargs).to_period('M')
    quarterly_end_date = pd.Period(str(monthly_date.year) + '-' + str(3 * monthly_date.month), freq='M')
    return quarterly_end_date

def get_econ_predictors(data_freq = 'monthly',
                        START_DATE:str = '1947-01',
                        END_DATE:str = '2005-04') -> pd.DataFrame:
    """
    This function returns the economic predictions for the given date range.
    The data is available for monthly and quarterly frequency.
    The data is available from 1947-01 to 2005-04.

    Parameters
    ----------
    data_freq (str, optional): monthly or quarterly
    START_DATE (str, optional): start date of the data
    END_DATE (str, optional): end date of the data

    Returns
    -------
    pd.DataFrame: economic predictors
    """

    date_freq_to_data_func_map = {'monthly': ('../../data/econ_predictors_monthly_2021_Amit_Goyal.csv',
                                            get_monthly_date_format), 
                                'quarterly': ('../../data/econ_predictors_quarterly_2021_Amit_Goyal.csv',
                                            get_quarterly_date_format)}
    data_path, date_format_func = date_freq_to_data_func_map[data_freq]

    data = pd.read_csv(data_path, index_col=0)
    data.index = [date_format_func(str(x), {'format':'%Y%m'}) for x in data.index]
    econ_data = data[START_DATE:END_DATE]

    equity_price = econ_data['Index'].apply(lambda x: re.sub(r'[^\w\s|.]', '', x))
    equity_price = equity_price.astype(float)
    equity_return = equity_price.pct_change()
    equity_premium = equity_return - econ_data['Rfree']
    
    dividend_price_ratio = np.log(econ_data['D12']) - np.log(equity_price)

    dividend_yield = np.log(econ_data['D12']) - np.log(equity_price.shift(1))

    earnings_price_ratio = np.log(econ_data['E12']) - np.log(equity_price)

    earnings_payout_ratio = np.log(econ_data['D12']) - np.log(econ_data['E12'])

    stock_variance = econ_data['svar']

    book_to_market = econ_data['b/m']
    
    net_equity_expansion = econ_data['ntis']

    treasury_bill = econ_data['tbl']

    long_term_yield = econ_data['lty']

    long_term_return = econ_data['ltr']

    term_spread = long_term_yield - treasury_bill

    default_yield_spread = econ_data['BAA'] - econ_data['AAA']

    default_return_spread = econ_data['corpr'] - econ_data['ltr']

    inflation = econ_data['infl']

    econ_predictors = pd.concat([equity_premium, 
                            dividend_price_ratio, 
                            dividend_yield,
                            earnings_price_ratio, 
                            earnings_payout_ratio,
                            stock_variance,
                            book_to_market,
                            net_equity_expansion,
                            treasury_bill,
                            long_term_yield,
                            long_term_return,
                            term_spread,
                            default_yield_spread,
                            default_return_spread, 
                            inflation], axis=1).dropna()
                            
    econ_predictors.columns = ['Equity Premium',
                            'Dividend Price Ratio', 
                            'Dividend Yield', 
                            'Earnings Price Ratio',
                            'Earnings Payout Ratio',
                            'Stock Variance',
                            'Book To Market',
                            'Net Equity Expansion',
                            'Treasury Bill',
                            'Long Term Yield',
                            'Long Term Return',
                            'Term Spread',
                            'Default Yield Spread',
                            'Default Return Spread',
                            'Inflation']

    if data_freq == 'quarterly':
        econ_predictors['Invest Capital Ratio'] = econ_data['ik']
        econ_predictors.dropna(inplace=True)
        
    return econ_predictors

def get_equities_returns(data_freq: str = 'monthly') -> pd.DataFrame:
    """
    Load stock market data from a CSV file and calculate the monthly or quarterly returns
    for the equity index and the risk-free rate.

    Parameters:
    -----------
    data_freq : str, optional
        The frequency of the data in the CSV file. Valid values are 'monthly' and 'quarterly'.
        Default is 'monthly'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the equity returns and the risk-free rate.

    Raises:
    -------
    KeyError
        If an invalid value is passed for the `data_freq` argument.

    """
    date_freq_to_data_func_map = {'monthly': ('../../data/econ_predictors_monthly_2021_Amit_Goyal.csv',
                                              get_monthly_date_format),
                                  'quarterly': ('../../data/econ_predictors_quarterly_2021_Amit_Goyal.csv',
                                                get_quarterly_date_format)}
    try:
        data_path, date_format_func = date_freq_to_data_func_map[data_freq]
    except KeyError:
        raise KeyError("Invalid value for data_freq. Valid values are 'monthly' and 'quarterly'.")
    data = pd.read_csv(data_path, index_col=0)
    data.index = [date_format_func(str(x), {'format': '%Y%m'}) for x in data.index]
    econ_data = data
    equity_price = econ_data['Index'].apply(lambda x: re.sub(r'[^\w\s|.]', '', x)).astype(float).pct_change().dropna()
    rf = econ_data['Rfree'].dropna()
    
    return pd.concat([equity_price, rf], axis=1)


def get_equities_returns_volatility(data_freq: str = 'quarterly') -> pd.DataFrame:
    """
    Load stock market data from a CSV file, calculate the monthly or quarterly returns for the
    equity index and the risk-free rate, and calculate the volatility (standard deviation) of the
    returns for each quarter.

    Parameters:
    -----------
    data_freq : str, optional
        The frequency of the data in the CSV file. Valid values are 'monthly' and 'quarterly'.
        Default is 'quarterly'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the volatility of the equity returns and the risk-free rate for each quarter.

    Raises:
    -------
    KeyError
        If an invalid value is passed for the `data_freq` argument.

    """
    data = get_equities_returns(data_freq = 'monthly')
    data_vol = data.resample('Q').std()
    data_vol.index = data_vol.index.asfreq('M', 'end')
    
    return data_vol


def get_env_data(start_date: str, end_date: str, data_freq: str = 'quarterly') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieve economic predictors, equities returns, and equities returns volatility data within the specified dates.

    Parameters:
    -----------
    start_date : str
        The start date in 'YYYY' format.
    end_date : str
        The end date in 'YYYY' format.
    data_freq : str
        The frequency of the data, either 'quarterly' or 'monthly'.

    Returns:
    --------
    A tuple of the following:
        - A data frame containing the economic predictors within the specified dates
        - A series of equities returns within the specified dates
        - A series of equities returns volatility within the specified dates
    """
    data = get_econ_predictors(data_freq=data_freq, START_DATE=start_date, END_DATE=end_date)
    data.pop('Equity Premium')
    equities_returns = get_equities_returns(data_freq=data_freq)
    equities_returns_vol = get_equities_returns_volatility(data_freq=data_freq)

    data = data.loc[start_date:end_date]
    index = data.index
    equities_returns = equities_returns.loc[index]
    equities_returns_vol = equities_returns_vol.loc[index]

    return data, equities_returns, equities_returns_vol

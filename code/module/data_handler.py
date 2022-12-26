from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import warnings
import itertools
warnings.filterwarnings("ignore")

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

    Args:
    TRAIN_START_DATE (str, optional): start date of training
    TRAIN_END_DATE (str, optional): end date of training
    TEST_START_DATE (str, optional): start date of testing
    TEST_END_DATE (str, optional): end date of testing
    TICKER_LIST (list[str], optional): list of ticker symbols
    user_defined_features_df (pd.DataFrame, optional): dataframe with user defined features
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


def get_monthly_date_format(date_string:str) -> pd.Period:
    """
    Get the monthly date format for a given date string.

    Parameters
    ----------
    date_string : str
        The date string to parse.
        
    Returns
    -------
    pd.Period
        The monthly date format.
    """
    monthly_date = pd.to_datetime(date_string).to_period('M')
    return monthly_date
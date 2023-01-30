import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
sys.path.append('../module')
from data_handler import get_econ_predictors

def get_return_forecast_performance(y_hat:pd.DataFrame, y:pd.DataFrame, forecast_name:str='forecast performance') -> tuple:
    '''Evaluate the return forecast in terms of following measurement:
    1. Hit ratio (HR)
    2. Root-Mean-Squared Forecast Error (RMSFE)
    3. Cross-Sectional Standard Deviation
    4. Time Series Volatility
    5. The First-order Autocorrelation

    Parameters:
    ----------
    y_hat: pd.DataFrame
        The predicted return
    y: pd.DataFrame
        The actual return
    forecast_name: str
        The name of the forecast
    
    Returns:
    -------
    performance_df: pd.DataFrame
        A dataframe containing the performance of the forecast
    '''

    N_c = (np.sign(y) == np.sign(y_hat)).sum().sum()
    N = y.count().sum()
    HR = N_c / N
    HR_percentage = HR * 100

    RMSFE = np.sqrt(np.square(y_hat - y).sum().mean())
    RMSFE_percentage = RMSFE * 100

    sigma_i = y_hat.std(axis=1).mean()
    sigma_i_percentage = sigma_i * 100

    sigma_t = y_hat.std(axis=0).mean()
    sigma_t_percentage = sigma_t * 100

    rho_1_vector = y_hat.apply(lambda x: sm.tsa.acf(x, nlags=1)[1], axis=0)
    rho_1 = rho_1_vector.mean()
    rho_1_percentage = rho_1 * 100

    ss_res = ((y - y_hat) ** 2).values.sum()
    ss_tot = ((y - y.mean()) ** 2).values.sum()
    R_2 = 1 - ss_res / ss_tot

    performance_df = pd.DataFrame([HR_percentage, RMSFE_percentage, sigma_i_percentage, sigma_t_percentage, rho_1_percentage, R_2], 
                                  index=['HR', ' RMSFE', 'sigma_i', 'sigma_t', 'rho_1', 'R^2'],
                                  columns=[forecast_name])

    return(performance_df)
def get_period_return(return_series:pd.DataFrame) -> float:
    """
    Calculate the period return for a given time series.

    Parameters
    ----------
    return_series : pd.DataFrame
        A time series of daily returns.
    
    Returns
    -------
    period_return : float
        The period return for the given time series.
    """
    period_return = (1 + return_series).product() - 1

    return period_return

def get_oos_r_square(y_hat: np.ndarray, y: np.ndarray, y_bar: np.ndarray) -> float:
    """
    This function calculates the out-of-sample R square for a prediction.

    Parameters
    ----------
    y_hat : np.ndarray
        Prediction values.
    y : np.ndarray
        True values.
    y_bar : np.ndarray
        Historical average.

    Returns
    -------
    float
        Out-of-sample R square.
    """
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_bar) ** 2)
    R_2 = 1 - ss_res / ss_tot
    R_2_percentage = R_2 * 100

    return R_2_percentage

def get_p_value_of_MSPE_adjusted_test(y:np.ndarray, y_bar:np.ndarray, y_hat:np.ndarray) -> float:
    """
    
    Parameters
    ----------
    y : np.ndarray (n_samples, 1)
    y_bar : np.ndarray (n_samples, 1)
    y_hat : np.ndarray (n_samples, 1)

    Returns
    -------
    p_value_of_MSPE_adjusted : float
    """
    F = (y - y_bar) ** 2 - ((y - y_hat) ** 2 - (y_bar - y_hat) ** 2)
    dummy = np.ones_like(F)
    lm_result = sm.OLS(F, dummy).fit()
    p_value = lm_result.pvalues.values[0]

    return p_value

def get_significance_of_p_value(p_value:float) -> str:
    """
    This function returns the combine the significant code with a p-value into a string.

    Parameters
    ----------
    p_value : float
        The p-value of a statistical test.

    Returns
    -------
    significance : str 
        The significance of the p-value.
    """

    p_value = round(p_value, ndigits=3)
    if p_value >= 0.1:
        significance = str(p_value) + ' '
    elif p_value > 0.05:
        significance = str(p_value) +' *'
    elif p_value > 0.01:
        significance = str(p_value) +' **'
    elif p_value <= 0.01:
        significance = str(p_value) +' ***'

    return significance

def get_significance_of_MSPE_adjusted_test(y:np.ndarray, y_bar:np.ndarray, y_hat:np.ndarray) -> str:
    """
    
    Parameters
    ----------
    y : np.ndarray (n_samples, 1)
    y_bar : np.ndarray (n_samples, 1)
    y_hat : np.ndarray (n_samples, 1)

    Returns
    -------
    significance_of_MSPE_adjusted : str
    """

    p_value = get_p_value_of_MSPE_adjusted_test(y=y, y_hat=y_hat, y_bar=y_bar)
    p_value = round(p_value, ndigits=3)
    if p_value >= 0.1:
        significance = str(p_value) + ' '
    elif p_value > 0.05:
        significance = str(p_value) +' *'
    elif p_value > 0.01:
        significance = str(p_value) +' **'
    elif p_value <= 0.01:
        significance = str(p_value) +' ***'

    return significance

def get_utility_gain_from_prediction(START_DATE: str,
                                     END_DATE: str,
                                     prediction: pd.DataFrame,
                                     historical_average: pd.DataFrame,
                                     equity_return: pd.DataFrame = None,
                                     rolling_window_size: int = 5, # number in year
                                     data_frequency: int = 12, # number of observations per year
                                     gamma: int = 3) -> float:
    """
    Get utility gain from prediction.
    TODO:
    -----
    1. the calculation of utility gain requires the true return of interested.
    2. we need to replace spy equity premium with the true equity return of interested.

    Parameters
    ----------
    START_DATE : str
        Start date of the utility gain curve.
        Format: YYYY-MM
    END_DATE : str
        End date of the utility gain curve.
        Format: YYYY-MM
    prediction : pd.DataFrame
        Prediction.
    historical_average : pd.DataFrame
        Historical average.
    equity_return : pd.DataFrame, optional
        Equity return starting N years before the start date. (N = rolling_window_size)
    rolling_window_size : int, optional
        Rolling window size.
        Default: 10
    data_frequency : int, optional
        Data frequency.
        Default: 12
    gamma : int, optional
        Gamma.
        Default: 3

    Returns
    -------
    utility_gain : float
        Utility gain.
    """

    START_DATE = datetime.strptime(START_DATE, '%Y-%m')
    START_DATE = str(START_DATE.year - rolling_window_size) + '-' + str(START_DATE.month)
    data_frequency_dict = {12: 'monthly', 4: 'quarterly', 1: 'yearly'}
    econ_predictors = get_econ_predictors(START_DATE=START_DATE, END_DATE=END_DATE, data_freq=data_frequency_dict[data_frequency])
    risk_free_bond = econ_predictors['Treasury Bill'] / 100
    if equity_return is None:
        stock_return = econ_predictors['Equity Premium']
    else:
        stock_return = equity_return[START_DATE:END_DATE]
    portfolio_df = pd.concat([stock_return, risk_free_bond], axis=1).dropna()

    sample_varince = portfolio_df.iloc[:, 0].rolling(rolling_window_size * data_frequency - 1).var().dropna()
    varince_estimation = sample_varince.shift(1).dropna()

    stock_weight_0 = (1 / gamma) * (historical_average / varince_estimation)
    stock_weight_0 = stock_weight_0.clip(0, 1.5)
    portfolio_weight_0 = pd.concat([stock_weight_0, 1 - stock_weight_0], axis = 1)
    w_0 = portfolio_weight_0.values.reshape(-1, 1, 2)

    stock_weight_1 = (1 / gamma) * (prediction / varince_estimation)
    stock_weight_1 = stock_weight_1.clip(0, 1.5)
    portfolio_weight_1 = pd.concat([stock_weight_1, 1 - stock_weight_1], axis = 1)
    w_1 = portfolio_weight_1.values.reshape(-1, 1, 2)

    return_df = portfolio_df.loc[portfolio_weight_0.index] # need to change
    returns = return_df.values.reshape(-1, 2, 1)

    portfolio_return_0 = (w_0 @ returns).flatten()
    portfolio_return_1 = (w_1 @ returns).flatten()

    # annualize the return according to Rapach (2010)
    mu_0 = np.mean(portfolio_return_0) * data_frequency
    sigma_0 = np.var(portfolio_return_0) * data_frequency
    uitility_0 = mu_0 - 0.5 * gamma * sigma_0

    mu_1 = np.mean(portfolio_return_1) * data_frequency
    sigma_1 = np.var(portfolio_return_1) * data_frequency
    uitility_1 = mu_1 - 0.5 * gamma * sigma_1

    utility_gain = uitility_1 - uitility_0
    utility_gain_percentage = utility_gain * 100

    return utility_gain_percentage
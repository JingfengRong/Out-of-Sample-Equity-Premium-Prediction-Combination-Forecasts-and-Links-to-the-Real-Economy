import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

def get_return_forecast_performance(y_hat:pd.DataFrame, y:pd.DataFrame, forecast_name:str='forecast performance') -> tuple:
    '''Evaluate the return forecast in terms of following measurement:
    1. Hit ratio (HR)
    2. Root-Mean-Squared Forecast Error (RMSFE)
    3. Cross-Sectional Standard Deviation
    4. Time Series Volatility
    5. The First-order Autocorrelation

    ----------
    Args:
    y_hat: return forecast
    y: the true value

    ----------
    Returns:
    a dataframe of five measurement
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
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import sys
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression



def get_linear_prediction(X_train:np.array, y_train:np.array, X_test:np.array) -> np.array:
    """
    Fit a linear regression model on the training data and get predictions on the test data

    Parameters
    ----------
    X_train: np.array
        Training data
    y_train: np.array
        Training label
    X_test: np.array
        Test data
    
    Returns
    -------
    prediction: np.array
        Prediction on the test data
    """
    model = LinearRegression(positive=True).fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    prediction = model.predict(X_test.reshape(-1,1))

    return prediction


def get_ts_predictions(X:np.ndarray, y:np.ndarray, X_test_size:int) -> np.ndarray:
    """
    Build linear models on training data in expanding window way.
    Get a series of predictions from the test data using the fit models.

    Parameters
    ----------
    X : np.array
        Whole sample of feature data.
        shape = (n_samples, 1)
    y : np.array
        Whole sample of labels.
        shape = (n_samples, 1)
    X_test_size : int
        Size of the test sample.

    Returns
    -------
    np.array
        Time series predictions.
        shape = (n_samples, 1)
    """

    expanding_window_idx_generator = TimeSeriesSplit(n_splits = X_test_size, test_size=1)
    prediction = [get_linear_prediction(X_train=X[train_index], 
                                        y_train=y[train_index], 
                                        X_test=X[test_index])
                                        for _, (train_index, test_index) in enumerate(expanding_window_idx_generator.split(X))]
    prediction = np.array(prediction).reshape(-1)
    return prediction


def get_ml_ret_prediction(model,
                          param_dict:dict,
                          cv_generator,
                          selection_criterion:str,
                          X_train:np.array, 
                          y_train:np.array, 
                          X_test:np.array, 
                          y_test:np.array = None
                          ) -> np.array|dict:
    """
    This function is used to get the expected return prediction from ML model on the test data.

    Parameters
    ----------
    model : object
        ML model from SKlearn package
    param_dict : dict
        Dictionary containing the model parameters
    X_train : np.array
        Training data
    y_train : np.array
        Training labels
    X_test : np.array
        Test data
    y_test : np.array
        Test labels. Not needed. Because the OOS performance is NOT evaluated for the single prediction.
    
    Returns
    -------
    np.array
        Expected return prediction
    dict
        In sample performance (e.g. R square)
    """

    param_num = [np.count_nonzero(param_vector) for _, param_vector in param_dict.items()]
    cv_sample_size = 0.5 * np.prod(param_num)
    grid_search_generator = RandomizedSearchCV(estimator = model, 
                                     param_distributions = param_dict,
                                     cv=cv_generator, 
                                     scoring=selection_criterion,
                                     n_jobs=-1,
                                     n_iter=cv_sample_size)
    grid_search_result = grid_search_generator.fit(X_train, y_train)
    performance_in_sample = grid_search_result.best_score_
    best_model = grid_search_result.best_estimator_
    pred = best_model.predict(X_test)

    return pred, {selection_criterion:performance_in_sample}


def get_shrunk_covariance_matrix(data: np.ndarray|pd.DataFrame) -> np.ndarray:
    """
    Returns the shrunk covariance matrix of the data.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Data for which the covariance matrix is calculated.
    
    Returns
    -------
    np.ndarray
    """
    cov = LedoitWolf().fit(data).covariance_.tolist()
    return cov

def get_benchmark_of_equity_premium_prediction(equity_premium:np.ndarray, 
                                               prediction_size:int, 
                                               holdout_size:int,
                                               prediction_index: pd.core.indexes) -> pd.DataFrame:
    """
    This function returns the historical average of the equity premium.
    The historical average is calculated by dividing the cumulative sum of the equity premium by the number of observations.
    The historical excludes holdout period.
    
    Parameters
    ----------
    equity_premium : np.ndarray
    prediction_size : int
    holdout_size : int
    prediction_index : pd.core.indexes

    Returns
    -------
    pd.DataFrame
    """

    historical_average = np.cumsum(equity_premium) / np.arange(1, len(equity_premium) + 1)
    historical_average = historical_average[-prediction_size+holdout_size-1:-1]
    benchmark = np.vstack([historical_average, equity_premium[-prediction_size+holdout_size:]]).T
    benchmark_df = pd.DataFrame(benchmark, 
                                columns = ['Historical Average', 'Equity Premium'], 
                                index = prediction_index[-prediction_size+holdout_size:])
    
    return benchmark_df


def get_combined_prediction(true_values: np.ndarray, prediction: np.ndarray, prediction_index: pd.core.indexes, holdout_size: int) -> pd.DataFrame:
    """
    This function retruns four combined predictions from each individual economic model
    
    Parameters
    ----------
    true_values : np.ndarray
        The actual equity return for the prediction period.
    
    prediction : np.ndarray
        The predicted equity return for the prediction period.
    
    prediction_index : pd.core.indexes
        The index of the prediction period.
    
    holdout_size : int
        The number of holdout observations.
    
    Returns
    -------
    np.ndarray
        The combined predictions.
    """
    mean = prediction.mean(axis=1)

    median = np.median(prediction, axis=1)
    
    prediction_copy = np.copy(prediction)
    prediction_copy.sort(axis=1)
    prediction_trimmed = prediction_copy[:,1:-1]
    trimmed_mean = prediction_trimmed.mean(axis=1)

    theta_1 = 1
    theta_2 = 0.9
    discount_power = np.arange(len(true_values), 0, -1) - 1
    discounted_theta_1 = np.power(theta_1, discount_power).reshape(-1,1)
    discounted_theta_2 = np.power(theta_2, discount_power).reshape(-1,1)
    squared_prediction_error = np.square(prediction - true_values)
    phi_1 = np.cumsum(squared_prediction_error * discounted_theta_1, axis=0)
    phi_2 = np.cumsum(squared_prediction_error * discounted_theta_2, axis=0)
    weight_1 = (1 / phi_1) / np.sum(1 / phi_1, axis=1, keepdims=True)
    weight_2 = (1 / phi_2) / np.sum(1 / phi_2, axis=1, keepdims=True)
    DMSPE_prediction_1 = np.sum(weight_1 * prediction, axis=1)
    DMSPE_prediction_2 = np.sum(weight_2 * prediction, axis=1)

    prediction_mat = np.vstack([mean, median, trimmed_mean, DMSPE_prediction_1, DMSPE_prediction_2]).T
    prediction_df = pd.DataFrame(prediction_mat, columns=['Mean', 'Median', 'Trimmed mean', 'DMSPE theta 1', 'DMSPE theta 0.9'], index=prediction_index)
    prediction_df = prediction_df.iloc[holdout_size:, :]
    
    return prediction_df
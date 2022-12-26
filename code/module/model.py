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
    """
    cov = LedoitWolf().fit(data).covariance_.tolist()
    return cov
    
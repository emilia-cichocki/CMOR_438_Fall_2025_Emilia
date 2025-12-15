
"""
    Postprocessing utilities for regression (NumPy)
    
    This module calculates a comprehensive set of postprocessing and evaluation 
    metrics for regression algorithms, with support for NumPy arrays.

    Functions
    ---------
    mae
        Computes the mean absolute error
    mse
        Computes the mean squared error
    rmse
        Computes the root mean squared error
    r2
        Computes the R2 score
    adjusted_r2
        Computes the R2 score adjusted for number of features
    print_model_metrics
        Prints accuracy, MAE, MSE, RMSE, R2, and adjusted R2
"""

import numpy as np
from typing import *
import matplotlib.pyplot as plt
from rice_ml.preprocess.datatype import *
from rice_ml.supervised_learning.distances import _ensure_numeric

__all__ = [
    'mae',
    'mse',
    'rmse',
    'r2',
    'adjusted_r2',
    'print_model_metrics'
]

def _validate_vector_match(predicted_scores: np.ndarray, true_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Validation of input vectors
    
    Converts both inputs to 1D numeric arrays and checks that they have the same length

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted target scores
    true_scores: np.ndarray
        Array of true target scores

    Returns
    -------
    pred_score: np.ndarray
        1D array of predicted target scores
    true_score: np.ndarray
        1D array of true target scores

    Raises
    ------
    ValueError
        If predicted and true score arrays are of different lengths
    """

    pred_score = _ensure_numeric(predicted_scores)
    true_score = _ensure_numeric(true_scores)
    if pred_score.shape[0] != true_score.shape[0]:
        raise ValueError('Predicted and true labels must be of the same length')
    
    return pred_score, true_score

def mae(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:

    """
    Calculates mean absolute error between predicted and true scores

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted target scores
    true_scores: np.ndarray
        Array of true target scores

    Returns
    -------
    mae: float
        Mean absolute error

    Raises
    ------
    ValueError
        If predicted and true score arrays are non-numeric or have different lengths

    Examples
    --------
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> mae(y_pred, y_true)
    0.5
    """

    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)
    mae = float(np.mean(abs(pred_scores - true_scores)))

    return mae

def mse(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:

    """
    Calculates mean squared error between predicted and true scores

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted target scores
    true_scores: np.ndarray
        Array of true target scores

    Returns
    -------
    mse: float
        Mean squared error

    Raises
    ------
    ValueError
        If predicted and true score arrays are non-numeric or have different lengths

    Examples
    --------
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> mse(y_pred, y_true)
    0.375
    """

    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)
    mse = float(np.mean((pred_scores - true_scores) ** 2))

    return mse

def rmse(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:

    """
    Calculates root mean squared error between predicted and true scores

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted target scores
    true_scores: np.ndarray
        Array of true target scores

    Returns
    -------
    rmse: float
        Root mean squared error

    Raises
    ------
    ValueError
        If predicted and true score arrays are non-numeric or have different lengths

    Examples
    --------
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> rmse(y_pred, y_true)
    0.6123724356957945
    """

    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)
    mse_score = mse(pred_scores, true_scores)
    rmse = float(np.sqrt(mse_score))

    return rmse

def r2(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:

    """
    Calculates R2 score between predicted and true scores

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted target scores
    true_scores: np.ndarray
        Array of true target scores

    Returns
    -------
    r2: float
        R2 score

    Raises
    ------
    ValueError
        If predicted and true score arrays are non-numeric or have different lengths

    Examples
    --------
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> r2(y_pred, y_true)
    0.9486081370449679
    """
        
    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)

    rss = np.sum((pred_scores - true_scores) ** 2)
    actual_mean = np.mean(true_scores)
    tss = np.sum((true_scores - actual_mean) ** 2)

    if tss == 0:
        if rss == 0:
            return 1.0
        else:
            raise ValueError('R2 score is undefined for constant true targets, unless predictions are perfect')

    r2_score = float(1 - (rss / tss))

    return r2_score

def adjusted_r2(predicted_scores: np.ndarray, true_scores: np.ndarray, n_features: int = 1) -> float:

    """
    Calculates adjusted R2 score between predicted and true scores, accounting
    for the number of input features

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted target scores
    true_scores: np.ndarray
        Array of true target scores
    n_features: int, default = 1
        Number of input features

    Returns
    -------
    adj_r2: float
        Adjusted R2 score

    Raises
    ------
    TypeError
        If the number of features is not an integer
    ValueError
        If predicted and true score arrays are non-numeric or have different lengths, or
        if the number of features is not a positive value

    Examples
    --------
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> adjusted_r2(y_pred, y_true, n_features=1)
    0.9229122055674519
    """

    if n_features is not None and not isinstance(n_features, int):
        raise TypeError("Number of features must be an integer")
    if n_features is not None and n_features <= 0:
        raise ValueError("Number of features must be greater than zero")
    
    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)

    n_samples = true_scores.shape[0]

    if n_samples <= n_features + 1:
        raise ValueError("Number of samples must be greater than number of features + 1 for adjusted R2")

    r2_score = r2(pred_scores, true_scores)

    adjusted_r2_score = 1 - ((1 - r2_score) * (n_samples - 1) / (n_samples - n_features - 1))

    return adjusted_r2_score


def print_model_metrics(predicted_scores: np.ndarray, true_scores: np.ndarray, n_features: int = 1) -> None:
    
    """
    Prints the set of model evaluation metrics for regression

    Includes MAE, MSE, RMSE, R2, and adjusted R2

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted target scores
    true_scores: np.ndarray
        Array of true target scores
    n_features: int, default = 1
        Number of input features

    Raises
    ------
    ValueError
        If the dimensions of `predicted_classes` and `true_classes` do not match
    """
   
    pred_score, true_score = _validate_vector_match(predicted_scores, true_scores)

    print(f"Model Metrics \n\
{'-' *13} \n\
MAE: {mae(pred_score, true_score):.2f} \n\
MSE: {mse(pred_score, true_score):.2f} \n\
RMSE: {rmse(pred_score, true_score):.2f} \n\
R2: {r2(pred_score, true_score):.2f} \n\
Adjusted R2: {adjusted_r2(pred_score, true_score, n_features):.2f}")
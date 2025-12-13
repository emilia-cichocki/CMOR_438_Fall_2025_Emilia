
"""
    Postprocessing utilities for regression (Numpy)
    # TODO: finish this!!

"""

import numpy as np
from typing import *
import matplotlib.pyplot as plt
from rice_ml.preprocess.datatype import *
from scipy import stats # TODO: finish this!
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
    
    pred_score = _ensure_numeric(predicted_scores)
    true_score = _ensure_numeric(true_scores)
    if pred_score.shape[0] != true_score.shape[0]:
        raise ValueError('Predicted and true labels must be of the same length')
    
    return pred_score, true_score

def mae(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:

    # TODO: type hints, docstrings

    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)
    mae = float(np.mean(abs(pred_scores - true_scores)))

    return mae

def mse(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:

    # TODO: type hints, docstrings

    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)
    mse = float(np.mean((pred_scores - true_scores) ** 2))

    return mse

def rmse(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:

    # TODO: type hints, docstrings

    pred_scores, true_scores = _validate_vector_match(predicted_scores, true_scores)
    mse_score = mse(pred_scores, true_scores)
    rmse = float(np.sqrt(mse_score))

    return rmse

def r2(predicted_scores: np.ndarray, true_scores: np.ndarray) -> float:
        
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
    
    pred_score, true_score = _validate_vector_match(predicted_scores, true_scores)

    print(f"Model Metrics \n\
{'-' *13} \n\
MAE: {mae(pred_score, true_score):.2f} \n\
MSE: {mse(pred_score, true_score):.2f} \n\
RMSE: {rmse(pred_score, true_score):.2f} \n\
R2: {r2(pred_score, true_score):.2f} \n\
Adjusted R2: {adjusted_r2(pred_score, true_score, n_features):.2f}")
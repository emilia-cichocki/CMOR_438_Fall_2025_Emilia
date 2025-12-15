
"""
    Standardization for data sets (NumPy and Pandas)

    This module performs common standardization processes on data sets intended for 
    use in a variety of machine learning algorithms. It primarily uses NumPy (and converts
    to NumPy arrays internally), with error handling and docstrings included.

    Functions
    ---------
    z_score_standardize
        Standardizes using z-scores (feature-wise)
    min_max_standardize
        Standardizes to specified range using min-max scaling (feature-wise)
    max_abs_standardize
        Standardizes using maximum absolute value scaling (feature-wise)
    l2_standardize
        Normalizes by the L2 norm (row-wise)
"""

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *

__all__ = [
    'z_score_standardize',
    'min_max_standardize',
    'max_abs_standardize',
    'l2_standardize',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def z_score_standardize(data_array: ArrayLike, return_params: bool = False, ddof: int = 0) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    
    """
    Standardizes a data array using z-score normalization

    Each feature (columns in the array) is centered by the mean
    of the feature and then divided by its standard deviation

    Parameters
    ----------
    data_array: ArrayLike
        Input array-like object shape (n_samples, n_features)
    return_params: bool, default = False
        Whether the mean and scale should be returned
    ddof: int, default = 0
        Delta degrees of freedom used to calculate scale (standard deviation)

    Returns
    -------
    standardized_array: np.ndarray
        2D array with columns standardized using z-scores
    dict (if `return_params` is True)
        Dictionary with mean and scale values

    Raises
    ------
    TypeError
        If `return_params` is not a boolean or `ddof` is not an integer
    ValueError
        If `ddof` is negative or `data_array` contains non-numeric values
    """

    array = _2D_numeric(data_array)

    if not isinstance(ddof, int):
        raise TypeError(f"ddof parameter must be an integer, got {type(ddof).__name__}")
    
    if not isinstance(return_params, bool):
        raise TypeError(f"return_params must be a boolean, got {type(return_params).__name__}")
    
    if ddof < 0:
        raise ValueError(f"ddof parameter must be greater than or equal to zero")

    columnwise_mean = array.mean(axis = 0)
    scale = array.std(axis = 0, ddof = ddof)
    scale[scale == 0.0] = 1.0
    standardized_array = (array - columnwise_mean) / scale

    if return_params:
        return standardized_array, {'mean': columnwise_mean, 'scale': scale}

    return standardized_array

def min_max_standardize(data_array: ArrayLike, *, feature_range: Tuple[float, float] = (0.0, 1.0), return_params: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:

    """
    Standardizes a data array using min-max normalization

    Each feature (columns in the array) is scaled to a given range
    using the respective minimum and maximum values

    Parameters
    ----------
    data_array: ArrayLike
        Input array-like object shape (n_samples, n_features)
    feature_range: tuple, default = (0.0, 1.0)
        Range (min, max) of the transformed data
    return_params: bool, default = False
        Whether the minimum, maximum, scale, and feature range should be returned

    Returns
    -------
    standardized_array: np.ndarray
        2D array with columns standardized using min-max normalization
    dict (if `return_params` is True)
        Dictionary with minimum, maximum, scale, and feature range

    Raises
    ------
    TypeError
        If `feature_range` is not a tuple with float or integer elements,
        or if `return_params` is not a boolean
    ValueError
        If `data_array` contains non-numeric values, or if the desired minimum
        is greater than the desired maximum
    """

    array = _2D_numeric(data_array)
    
    if not (isinstance(feature_range, Tuple) and len(feature_range) == 2 and all(isinstance(element, (int, float)) for element in feature_range)):
        raise TypeError(f"Feature range must be a tuple of length 2 (min, max) with float or integer elements")
    
    if not isinstance(return_params, bool):
        raise TypeError(f"return_params must be a boolean, got {type(return_params).__name__}")
    
    feature_min, feature_max = feature_range[0], feature_range[1]
    if feature_min >= feature_max:
        raise ValueError(f"Minimum of feature range must be less than maximum")
    
    array_maximums = array.max(axis = 0)
    array_minimums = array.min(axis = 0)
    scale = array_maximums - array_minimums
    scale[scale == 0.0] = 1.0
    standardized_array = feature_min + ((array - array_minimums) / scale) * (feature_max - feature_min)

    if return_params:
        return standardized_array, {'minimum': array_minimums, 'maximum': array_maximums, 'scale': scale, 'feature_range': feature_range}
    return standardized_array

def max_abs_standardize(data_array: ArrayLike, return_params: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    
    """
    Standardizes a data array using maximum absolute values

    Each feature (columns in the array) is scaled using the maximum
    absolute value from the feature, resulting in data between -1 and 1

    Parameters
    ----------
    data_array: ArrayLike
        Input array-like object shape (n_samples, n_features)
    return_params: bool, default = False
        Whether the scale should be returned

    Returns
    -------
    standardized_array: np.ndarray
        2D array with columns standardized using max-absolute normalization
    dict (if `return_params` is True)
        Dictionary with scale

    Raises
    ------
    TypeError
        If `return_params` is not a boolean
    ValueError
        If `data_array` contains non-numeric values
    """

    array = _2D_numeric(data_array)

    if not isinstance(return_params, bool):
        raise TypeError(f"return_params must be a boolean, got {type(return_params).__name__}")
    
    absolute_value_array = abs(array)
    column_maximums = absolute_value_array.max(axis = 0)
    column_maximums[column_maximums == 0.0] = 1.0
    standardized_array = array / column_maximums

    if return_params:
        return standardized_array, {'scale': column_maximums}
    return standardized_array

def l2_standardize(data_array: ArrayLike, epsilon: float = 1e-15) -> np.ndarray:

    """
    Standardizes a data array using L2 norms

    Each row in the array is scaled such that the L2 norm is equal to 1

    Parameters
    ----------
    data_array: ArrayLike
        Input array-like object shape (n_samples, n_features)
    epsilon: float, default = 1e-15
        Small scale value used to prevent zero division

    Returns
    -------
    standardized_array: np.ndarray
        2D array with rows standardized using L2 normalization

    Raises
    ------
    TypeError
        If `epsilon` is not an integer or float
    ValueError
        If `data_array` contains non-numeric values, or if `epsilon`
        is not positive
    """

    array = _2D_numeric(data_array)

    if not isinstance(epsilon, (float, int)):
        raise TypeError(f"Floor value must be a float, got {type(epsilon).__name__}")
    
    if epsilon <= 0:
        raise ValueError("Floor value must be greater than zero")
    
    l2_norms = np.linalg.norm(array, axis = 1)
    l2_norms[l2_norms <= epsilon] = epsilon
    l2_norms = l2_norms[:, None]

    standardized_array = array / l2_norms

    return standardized_array
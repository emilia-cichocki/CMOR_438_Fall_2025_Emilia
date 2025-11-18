
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
    
    # TODO: add type hints/docstrings/examples

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

    # TODO: add type hints/docstrings/examples

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
    
    # TODO: add type hints/docstrings/examples

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

    # TODO: add type hints/docstrings/examples

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
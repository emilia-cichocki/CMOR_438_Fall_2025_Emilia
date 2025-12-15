
"""
    Data cleaning utilities for data sets (NumPy and Pandas)

    This module performs standard data cleaning processes (identification of missing 
    data, outliers, and duplicates) on numeric NumPy arrays, with flexible methods of
    accounting for anomalous data.

    Functions
    ---------
    missing_data
        Locates missing data, with the option to remove or estimate values
    outlier_identify
        Locates univariate outliers, with the option to remove
    duplicate_identify
        Locates duplicate rows, with the option to remove
"""

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *
from scipy import stats
from rice_ml.preprocess.standardize import z_score_standardize

__all__ = [
    'missing_data',
    'outlier_identify',
    'duplicate_identify',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def missing_data(data_array: ArrayLike, strategy: Literal['drop', 'mean', 'median', 'mode']) -> np.array:

    """
    Identifies and handles missing data in a 2D array

    Accounts for missing values (NaN) in the array using the specified strategies,
    which can be one of:
    - 'drop': Removes rows containing NaN
    - 'mean' Replaces missing values with the mean of the column
    - 'median': Replaces missing values with the median of the column
    - 'mode': Replaces missing values with the mode of the column

    Parameters
    ----------
    data_array : ArrayLike
        Array-like object with shape (n_samples, n_features)
    strategy : {'drop', 'mean', 'median', 'mode'}
        Strategy to handle missing data

    Returns
    -------
    cleaned_array: np.ndarray
        2D array with missing values accounted for using the given strategy

    Raises
    ------
    ValueError
        If `strategy` is not one of {'drop', 'mean', 'median', 'mode'}, or
        if the data array is not 2D or numeric
    """

    array = _2D_numeric(data_array)
    possible_strategies = {'drop', 'mean', 'median', 'mode'}
    if strategy not in possible_strategies:
        raise ValueError(f"Strategy must be one of {possible_strategies}, got '{strategy}'")
    
    if strategy == 'drop':
        cleaned_array = array[~np.any(np.isnan(array), axis = 1)]
    else:
        cleaned_array = array.copy()

        for column in range(array.shape[1]):
            empty_elements = np.isnan(array[:, column])

            if strategy == 'mean':
                column_mean = np.nanmean(array[:, column])
                cleaned_array[empty_elements, column] = column_mean

            if strategy == 'median':
                column_median = np.nanmedian(array[:, column])
                cleaned_array[empty_elements, column] = column_median

            if strategy == 'mode':
                column_elements = array[:, column][~np.isnan(array[:, column])]
                column_mode = stats.mode(column_elements).mode
                cleaned_array[empty_elements, column] = column_mode

    return cleaned_array

def outlier_identify(data_array: ArrayLike, method: Literal['IQR', 'zscore'], *, drop: bool = False, threshold: float = 3) -> np.array:

    """
    Identifies and handles outliers in a 2D array

    Finds outliers in each column from an array using a given strategy, 
    which can be one of
    - 'IQR': outliers identified using interquartile range
    - 'zscore': outliers identified through z-scoring thresholds

    Parameters
    ----------
    data_array: ArrayLike
        Array-like object with shape (n_samples, n_features)
    method: {'IQR', 'zscore'}
        Strategy to calculate outliers
    drop: bool, default = False
        Whether rows with outliers should be removed
    threshold: float, default = 3
        Z-score threshold value to be considered an outlier

    Returns
    -------
    cleaned_array: np.ndarray
        2D array with outliers either removed or calculated

    Raises
    ------
    TypeError
        If `drop` is not a boolean or `threshold` is not a float or 
        integer value
    ValueError
        if `threshold` is less than zero or `method` is not one of
        {'IQR', 'zscore}, or if the data array is not 2D or numeric
    """
    
    array = _2D_numeric(data_array)

    if not isinstance(drop, bool):
        raise TypeError(f"Drop parameter must be a boolean, got {type(drop).__name__}")
    if not isinstance(threshold, (float, int)) or isinstance(threshold, bool):
        raise TypeError(f"Threshold must be a float or integer, got {type(threshold).__name__}")
    if threshold < 0:
        raise ValueError(f"Threshold must be greater than zero")
    possible_methods = {'IQR', 'zscore'}
    if method not in possible_methods:
        raise ValueError(f"Method of outlier detection must be one of {possible_methods}, got '{method}'")
    
    if method == 'IQR':
        outlier_indices = set()
        for column in range(array.shape[1]):
            q3 = np.percentile(array[:, column], 75)
            q1 = np.percentile(array[:, column], 25)
            iqr = q3 - q1
            indices = (np.where((array[:, column] > q3 + 1.5 * iqr) | (array[:, column] < q1 - 1.5 * iqr))[0]).tolist()
            outlier_indices.update(indices)

    if method == 'zscore':
        outlier_indices = set()
        z_scores = z_score_standardize(array, False, 0)
        indices = (np.where((abs(z_scores) > threshold))[0]).tolist()
        outlier_indices.update(indices)

    if drop:
        cleaned_array = np.delete(array, list(outlier_indices), axis = 0)
    else:
        cleaned_array = array.copy()

    return cleaned_array

def duplicate_identify(data_array: ArrayLike, drop: bool = False) -> np.array:

    """
    Identifies and handles duplicate rows in a 2D array

    Finds rows that are duplicates of at least one other row
    in the array, and optionally drops them

    Parameters
    ----------
    data_array: ArrayLike
        Array-like object with shape (n_samples, n_features)
    drop: bool, default = False
        Whether duplicate rows should be removed

    Returns
    -------
    cleaned_array: np.ndarray
        2D array with duplicate rows either removed or calculated

    Raises
    ------
    TypeError
        If `drop` is not a boolean
    ValueError
        If the data array is not 2D or numeric
    """

    array = _2D_numeric(data_array)
    if not isinstance(drop, bool):
            raise TypeError(f"Drop parameter must be a boolean, got {type(drop).__name__}")

    indices = []
    for row in range(array.shape[0]):
        for comparison_row in range(row + 1, array.shape[0]):
            if np.array_equal(array[row], array[comparison_row]) and row != comparison_row:
                indices.append(comparison_row)
    indices = list(set(indices))

    if drop:
        cleaned_array = np.delete(array, indices, axis = 0)
    else:
        cleaned_array = array.copy()

    return cleaned_array
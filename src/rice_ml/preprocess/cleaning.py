
"""
    Data cleaning utilities for data sets (NumPy)

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

# TODO: fix the general formatting to make it consistent across functions (newlines, etc.), account for all nan columns (in unit tests as well)

def missing_data(data_array: ArrayLike, strategy: str) -> np.array:

    # TODO: type hints, docstring, explanation of strategies, add workaround for all unique values in mode (and add to unit tests)

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

def outlier_identify(data_array: ArrayLike, method: str, *, drop: bool = False, threshold: float = 3) -> np.array:

    # TODO: type hints, docstring, explanation of strategies, option to print outliers/indicate (?)

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

    # TODO: type hints, docstring, explanation of strategies, option to print duplicate rows/indicate (?)

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
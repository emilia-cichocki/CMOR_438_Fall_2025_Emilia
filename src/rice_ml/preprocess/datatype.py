
"""
    Data type and shape-checking utilities (NumPy and Pandas)

    This module ensures that the data set is in the form of a numeric NumPy array such that 
    different preprocessing and data analysis functions can be applied, with internal
    conversions applied if necessary.

    Functions
    ---------
    _2D_numeric
        Checks that the data is a 2D numeric NumPy array
    _1D_numeric
        Checks that the data is a 1D numeric NumPy array/vector
    _shape_match
        Checks that the shape of two arrays/vectors match
"""

import numpy as np
from typing import *
import pandas as pd

__all__ = [
    '_2D_numeric',
    '_1D_vectorized',
    '_shape_match',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _2D_numeric(data_array: ArrayLike, name: str = 'Data') -> np.ndarray:

    """
    Converts an input to a 2D numeric array

    Ensures that the data is array-like, 2D, non-empty, and 
    contains only numeric values or non-numeric values that can
    be converted to a float

    Parameters
    ----------
    data_array: ArrayLike
        Array-like object with shape (n_samples, n_features)
    name: str, default = 'Data'
        Name of the input array for use in error messages

    Returns
    -------
    array: np.ndarray
        2D array with float entries

    Raises
    ------
    TypeError
        If `data_array` is not array-like or contains non-numeric values, or
        if `name` is not a string
    ValueError
        If `data_array` is not 2D or is empty
    """

    if not isinstance(data_array, (np.ndarray, tuple, list, pd.DataFrame, pd.Series)):
        raise TypeError(f'Data array must be ArrayLike, got {type(data_array).__name__}')
    
    if not isinstance(name, str):
        raise TypeError(f'Name must be str, got {type(name).__name__}')
    
    array = np.asarray(data_array)

    if array.ndim != 2:
        raise ValueError(f'{name} must be a 2D array; got {array.ndim}D instead.')
    if array.size == 0:
        raise ValueError(f'{name} must be non-empty')

    if not np.issubdtype(array.dtype, np.number):
        try:
            array = array.astype(float, copy = False)
        except (TypeError, ValueError) as e:
            raise TypeError(f'All entries in {name} must be numeric') from e
    else:
        array = array.astype(float, copy = False)

    return array

def _1D_vectorized(data_vector: Optional[ArrayLike], name: str = 'Data') -> Optional[np.ndarray]:
    
    """
    Converts an input to a 1D array

    Ensures that the data is array-like, 1D, and non-empty, but
    allows for non-numeric values

    Parameters
    ----------
    data_vector: optional, ArrayLike
        Array-like object with shape (n_samples, n_features)
    name: str, default = 'Data'
        Name of the input array for use in error messages

    Returns
    -------
    array: np.ndarray
        1D vector with non-empty entries

    Raises
    ------
    TypeError
        If `data_vector` is not array-like or if `name` is not a 
        string
    ValueError
        If `data_vector` is not 1D or is empty
    """

    if (data_vector is not None) and (not isinstance(data_vector, (np.ndarray, tuple, list, pd.DataFrame, pd.Series))):
        raise TypeError(f'Data vector must be an array, got {type(data_vector).__name__}')
    
    if not isinstance(name, str):
        raise TypeError(f'Name must be str, got {type(name).__name__}')
    
    if data_vector is None:
        return None
    
    vector = np.asarray(data_vector)

    if vector.ndim != 1:
        raise ValueError(f'{name} must be a 1D array; got {vector.ndim}D instead')
    if vector.size == 0:
        raise ValueError(f'{name} must be non-empty')

    return vector

def _shape_match(data_array: np.ndarray, data_vector: Optional[np.ndarray]) -> None:

    """
    Checks the shapes of two input arrays

    Ensures that the input arrays have the same first dimension
    (typically representing number of samples)

    Parameters
    ----------
    data_array: np.ndarray
        Array with each row representing a sample
    data_vector: optional, np.ndarray
        Array with each row representing a sample

    Raises
    ------
    TypeError
        If `data_vector` or `data_array` is not an array
    ValueError
        If `data_vector` and `data_array` do not have the same first
        dimension
    """

    if not isinstance(data_array, np.ndarray):
        raise TypeError(f'Data array must be an array, got {type(data_array).__name__}')
    
    if (data_vector is not None) and (not isinstance(data_vector, np.ndarray)):
        raise TypeError(f'Data vector must be an array, got {type(data_vector).__name__}')

    if data_vector is not None:
        if data_array.shape[0] != data_vector.shape[0]:
            raise ValueError(f'Both arrays must have the same first dimension;'
                             f'got data_array.shape[0] = {data_array.shape[0]} '
                             f'and data_vector.shape[0] = {data_vector.shape[0]} instead')
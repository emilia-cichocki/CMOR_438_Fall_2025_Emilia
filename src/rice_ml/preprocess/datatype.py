
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
    '_1D_numeric',
    '_shape_match',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

# TODO: fix the general formatting to make it consistent across functions (newlines, etc.)

def _2D_numeric(data_array: ArrayLike, name: str = 'Data') -> np.ndarray:

    # TODO: add type hints and docstrings, change the name Data

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

def _1D_numeric(data_vector: Optional[ArrayLike], name: str = 'Data') -> Optional[np.ndarray]:
    
    # TODO: add type hints and docstrings, rename the function

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

    # TODO: add type hints and docstrings

    if not isinstance(data_array, np.ndarray):
        raise TypeError(f'Data array must be an array, got {type(data_array).__name__}')
    
    if (data_vector is not None) and (not isinstance(data_vector, np.ndarray)):
        raise TypeError(f'Data vector must be an array, got {type(data_vector).__name__}')

    if data_vector is not None:
        if data_array.shape[0] != data_vector.shape[0]:
            raise ValueError(f'Both arrays must have the same first dimension;'
                             f'got data_array.shape[0] = {data_array.shape[0]} '
                             f'and data_vector.shape[0] = {data_vector.shape[0]} instead')
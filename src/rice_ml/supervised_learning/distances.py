
"""
    Distance calculation utilities (NumPy and Pandas)

    This module calculates common distance metrics (Euclidean, Manhattan, and Minkowski) 
    for one-dimensional numerical vectors. It allows both NumPy vectors or Pandas Series,
    with internal conversions applied for efficient NumPy implementation.

    Functions
    ---------
    euclidean_distance
        Calculates the Euclidean distance for two vectors
    manhattan_distance
        Calculates the Manhattan distance for two vectors
    minkowski_distance
        Calculates the Minkowski distance for two vectors

"""

import numpy as np
import pandas as pd
from typing import *
import math
from rice_ml.preprocess.datatype import *

__all__ = [
    'euclidean_distance',
    'manhattan_distance',
    'minkowski_distance'
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _ensure_numeric(data_vector: ArrayLike, name: str = 'Data') -> np.ndarray:

    """
    Converts a data vector to a 1D array and ensures
    all entries are numeric
    
    Parameters
    ----------
    data_vector: ArrayLike
        1D array-like input vector
    name: str, default = 'Data'
        Name of the data vector, used for error messages

    Returns
    -------
    vector: np.ndarray
        1D NumPy array of only numeric data

    Raises
    ------
    TypeError
        If the vector contains values that are not numeric
    ValueError
        If the vector contains missing data (NaN values)
    """

    vector = _1D_vectorized(data_vector, name)
    
    if not np.issubdtype(vector.dtype, np.number):
        try:
            vector = vector.astype(float, copy = False)
        except (TypeError, ValueError) as e:
            raise TypeError(f'All entries in {name} must be numeric') from e
    else:
        vector = vector.astype(float, copy = False)

    if np.isnan(vector).any():
        raise ValueError(f'{name} contains missing data (NaN values)')
    
    return vector

def euclidean_distance(data_vector_1: ArrayLike, 
                       data_vector_2: ArrayLike, 
                       name_1: str = 'data_vector_1', 
                       name_2: str = 'data_vector_2') -> float:
    
    """
    Computes the Euclidean distance between two vectors

    Parameters
    ----------
    data_vector_1: ArrayLike
        1D array-like object containing numeric values
    data_vector_2: ArrayLike
        1D array-like object containing numeric values
    name_1: str, default = 'data_vector_1'
        Name of first vector, used for error messages
    name_2 : str, default='data_vector_2'
        Name of second vector, used for error messages

    Returns
    -------
    distance: float
        Euclidean distance between the two vectors

    Raises
    ------
    TypeError
        If either vector has data that is not numeric
    ValueError
        If the shapes of the vectors do not match
    
    Examples
    --------
    >>> euclidean_distance([0, 0], [3, 4])
    5.0
    """

    vector_1 = _ensure_numeric(data_vector_1, name_1)
    vector_2 = _ensure_numeric(data_vector_2, name_2)

    _shape_match(vector_1, vector_2)

    distance = float(np.linalg.norm((vector_2 - vector_1)))

    return distance

def manhattan_distance(data_vector_1: ArrayLike, 
                       data_vector_2: ArrayLike, 
                       name_1: str = 'data_vector_1', 
                       name_2: str = 'data_vector_2') -> float:
    
    """
    Computes the Manhattan distance between two vectors

    Parameters
    ----------
    data_vector_1: ArrayLike
        1D array-like object containing numeric values
    data_vector_2: ArrayLike
        1D array-like object containing numeric values
    name_1: str, default = 'data_vector_1'
        Name of first vector, used for error messages
    name_2 : str, default='data_vector_2'
        Name of second vector, used for error messages

    Returns
    -------
    distance: float
        Manhattan distance between the two vectors

    Raises
    ------
    TypeError
        If either vector has data that is not numeric
    ValueError
        If the shapes of the vectors do not match
    
    Examples
    --------
    >>> manhattan_distance([0, 0], [3, 4])
    7.0
    """

    vector_1 = _ensure_numeric(data_vector_1, name_1)
    vector_2 = _ensure_numeric(data_vector_2, name_2)

    _shape_match(vector_1, vector_2)

    distance = float(np.sum(np.abs(vector_2 - vector_1)))

    return distance

def minkowski_distance(data_vector_1: ArrayLike, 
                       data_vector_2: ArrayLike, 
                       p: int,
                       name_1: str = 'data_vector_1', 
                       name_2: str = 'data_vector_2') -> float:
    
    """
    Computes the Minkowski distance between two vectors

    Parameters
    ----------
    data_vector_1: ArrayLike
        1D array-like object containing numeric values
    data_vector_2: ArrayLike
        1D array-like object containing numeric values
    p: int
        Order used for the calculation
    name_1: str, default = 'data_vector_1'
        Name of first vector, used for error messages
    name_2 : str, default='data_vector_2'
        Name of second vector, used for error messages

    Returns
    -------
    distance: float
        Minkowski distance between the two vectors

    Raises
    ------
    TypeError
        If either vector has data that is not numeric
    ValueError
        If the shapes of the vectors do not match
    
    Examples
    --------
    >>> minkowski_distance([0, 0], [3, 4], p = 1)
    7.0
    """

    if not isinstance(p, int):
        raise TypeError('p parameter must be an integer')
    if p <= 0:
        raise ValueError('p parameter must be greater than zero')
    
    vector_1 = _ensure_numeric(data_vector_1, name_1)
    vector_2 = _ensure_numeric(data_vector_2, name_2)

    _shape_match(vector_1, vector_2)

    if p > 25:
        distance = float(np.exp(np.log(np.sum((np.abs(vector_2 - vector_1)) ** p)) / p))
    else:
        distance = float((np.sum((np.abs(vector_2 - vector_1)) ** p)) ** (1 / p))

    return distance
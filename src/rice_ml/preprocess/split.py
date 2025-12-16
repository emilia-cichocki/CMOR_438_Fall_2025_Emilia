"""
    Data splitting utilities (NumPy and Pandas)

    This module splits NumPy arrays into training, testing, and (optionally) validation 
    data subsets for use in machine learning.

    Functions
    ---------
    _bounded_count
        Used to bound index counts
    _random_number
        Creates an instance of a random number generator with a specified random state
    _stratified_indices
        Uses proportional sampling to find training and testing indices, with the option of a validation set
    train_test
        Splits data into training and testing sets, with the option of a validation set
    
"""

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *

__all__ = [
    'train_test',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _bounded_count(length: int, proportion: float) -> int:
    
    """
    Calculates a bounded count for a length using a 
    specified proportion

    Ensures that the count is within appropriate bounds to prevent
    empty splits

    Parameters
    ----------
    length: int
        Number of elements
    proportion: float
        Proportion to be selected

    Returns
    -------
    bound_count: int
        A bounded integer count based on the given proportion

    Raises
    ------
    TypeError
        If `length` is not an integer or `proportion` is not a
        float or integer
    """

    if not isinstance(length, int):
        raise TypeError(f"First value must be an integer, got {type(length).__name__}")
    if not isinstance(proportion, (float, int)):
        raise TypeError(f"Second number must be a float, got {type(proportion).__name__}")

    count = int(round(proportion * length))

    if length <= 1:
        if count < length:
            return 0
        else:
            return 1
    
    bound_count = min(max(1, count), length - 1)

    return bound_count

def _random_number(random_state: Optional[int]) -> np.random.Generator:
    
    """
    Creates a random number generator using NumPy

    Parameters
    ----------
    random_state: int, optional
        Random state for reproducibility; if none specified,
        a random seed will be used

    Returns
    -------
    np.random.Generator
        NumPy random number generator instance using the specified
        random state

    Raises
    ------
    TypeError
        If `random_state` is not an integer or None
    """

    if random_state is not None and not isinstance(random_state, int):
            raise TypeError(f"Random state must be an integer")
    if random_state is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(int(random_state))

def _stratified_indices(data: np.ndarray, 
                        test_size: float, 
                        rng: np.random.Generator, 
                        *, 
                        validation: bool = False,
                        val_size: Optional[float] = None) -> Union[Tuple[np.array, np.array], Tuple[np.array, np.array, np.array]]:

    """
    Generates stratified indices for data given a specified test size

    Splits indices to maintaining proportions across the results, which
    can then be used for train/test data splits that require the preservation
    of relative proportions for each class

    Parameters
    ----------
    data: np.ndarray
        1D array of class labels
    test_size: float
        Proportion of data that should be placed in the test set
    rng: np.random.Generator
        Random number generator used for shuffling indices
    validation: bool, default = False
        Whether a validation split should be generated
    val_size: float, optional
        Proportion of the remaining data assigned to validation, required
        when `validation` is True

    Returns
    -------
    testing_indices: np.ndarray
        1D array containing indices of test data
    training_indices: np.ndarray
        1D array containing indices of training data
    val_indices: np.ndarray (if `validation` is True)
        1D array containing indices of validation data

    Raises
    ------
    TypeError
        If `test_size` is not a float, `validation` is not a boolean, 
        or `rng` is not a random generator instance
    ValueError
        If proportions are not in the range 0 to 1, or are not compatible
        with one another
    """

    if not isinstance(test_size, float):
        raise TypeError(f"Test proportion must be a float, got {type(test_size).__name__}")
    if not isinstance(validation, bool):
        raise TypeError(f"Validation parameter must be a boolean, got {type(validation).__name__}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a random generator, got {type(rng).__name__}")

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"Test proportion must be between 0 and 1, got {test_size}")
    
    data = _1D_vectorized(data)

    if not validation:
        classes, label_index = np.unique(data, return_inverse = True)
        testing_indices = []
        training_indices = []
        for class_number in range(len(classes)):
            class_index = np.flatnonzero(label_index == class_number)
            rng.shuffle(class_index)

            if len(classes) > 1:
                test_number = max(1, int(round(test_size * len(class_index))))
            else:
                test_number = int(round(test_size * len(class_index)))

            if len(class_index) > 1:
                test_number = min(test_number, len(class_index) - 1)

            testing_indices.append(class_index[0:test_number])
            training_indices.append(class_index[test_number:])
        
        training_indices = np.concatenate(training_indices)
        testing_indices = np.concatenate(testing_indices)

        return testing_indices, training_indices

    elif validation:
        if not isinstance(val_size, float):
            raise TypeError(f"Validation set proportion must be a float, got {type(val_size).__name__}")
        if not (0.0 < val_size < 1.0):
            raise ValueError(f"Validation set proportion must be between 0 and 1, got {val_size}")
        if val_size + test_size >= 1.0:
            raise ValueError("Combined validation and test set proportions must be less than 1.")
        
        val_size_remaining = val_size / (1.0 - test_size)

        classes, label_index = np.unique(data, return_inverse = True)
        testing_indices = []
        training_indices = []
        val_indices = []

        for class_number in range(len(classes)):
            class_index = np.flatnonzero(label_index == class_number)
            rng.shuffle(class_index)

            if len(classes) > 1:
                test_number = max(1, int(round(test_size * len(class_index))))
            else:
                test_number = int(round(test_size * len(class_index)))

            if len(class_index) <= 1:
                if test_number < len(class_index):
                    test_number = 0
                else:
                    test_number = 1
            else:
                 test_number = min(test_number, len(class_index) - 1)

            testing_indices_initial = class_index[0:test_number]
            remaining_indices = class_index[test_number:]

            val_number = int(round(val_size_remaining * len(remaining_indices)))
            if len(remaining_indices) <= 1:
                if val_number < len(remaining_indices):
                    val_number = 0
                else:
                    val_number = 1
            else:
                val_number = min(val_number, len(remaining_indices) - 1)
            
            val_indices_initial = remaining_indices[0:val_number]
            training_indices_initial = remaining_indices[val_number:]

            testing_indices.append(testing_indices_initial)
            training_indices.append(training_indices_initial)
            val_indices.append(val_indices_initial)

        training_indices = np.concatenate(training_indices)
        testing_indices = np.concatenate(testing_indices)
        val_indices = np.concatenate(val_indices)
        
        return testing_indices, training_indices, val_indices

def train_test(data_array: ArrayLike, 
               data_vector: Optional[ArrayLike] = None, 
               test_size: float = 0.3,
               validation: bool = False, 
               val_size: Optional[float] = 0.1,  
               shuffle: bool = True, 
               stratify: Optional[ArrayLike] = None, 
               random_state: Optional[int] = None
               ) -> Union[
                        Union
                            [Tuple[np.ndarray, np.ndarray], 
                             Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], 
                        Union[
                            Tuple[np.ndarray, np.ndarray, np.ndarray],
                            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                        ]]:
    
    """
    Splits data into training, testing, and validation (optional) sets

    Divides data and label vectors based on the given proportion sizes, with 
    support for shuffling and stratification based on separate labels

    Parameters
    ----------
    data_array: np.ndarray
        2D array of size (n_samples, n_features)
    data_vector: ArrayLike, optional
        Optional target or label vector of size (n_samples,)
    test_size: float, default = 0.3
        Proportion of data assigned to the test set
    validation: boolean, default = False
        Whether data is split into a validation set
    val_size : float, default = 0.1
        Proportion of data assigned to the validation set
    shuffle: boolean, default = True
        Whether data should be shuffled before splitting
    stratify: ArrayLike, optional
        Labels used for stratified splitting, if applicable
    random_state: int, optional
        Random state used in shuffling

    Returns
    -------
    training_array: np.ndarray
        2D array containing training data
    testing_array: np.ndarray
        2D array containing testing data
    val_array: np.ndarray (if `validation` is True)
        2D array containing validation data
    training_data_vector: np.ndarray
        1D array containing training labels
    testing_data_vector: np.ndarray
        1D array containing testing labels
    val_data_vector: np.ndarray (if `validation` is True)
        1D array containing testing labels

    Raises
    ------
    TypeError
        If `test_size` is not a float, or `validation` or `shuffle` is not 
        a boolean
    ValueError
        If proportions are not in the range 0 to 1 or are incompatible, or
        if shapes of arrays do not match
    """

    array = _2D_numeric(data_array,'data array')

    if not isinstance(test_size, float):
        raise TypeError(f"Test proportion must be a float, got {type(test_size).__name__}")
    if not isinstance(validation, bool):
        raise TypeError(f"Validation parameter must be a boolean, got {type(validation).__name__}")
    if not isinstance(shuffle, bool):
        raise TypeError(f"Shuffle parameter must be a boolean, got {type(shuffle).__name__}")
    
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"Test proportion must be between 0 and 1, got {test_size}")

    rng = _random_number(random_state)

    if not validation:
        if not stratify is None:
            stratify_array = _1D_vectorized(stratify, 'stratify')
            if len(stratify_array) != array.shape[0]:
                raise ValueError('Stratify must have the same length as the data array')
            testing_indices, training_indices = _stratified_indices(stratify_array, test_size, rng)
        else:
            indices = np.arange(array.shape[0])
            if shuffle:
                rng.shuffle(indices)
            test_number = _bounded_count(len(indices), test_size)
            testing_indices = indices[0:test_number]
            training_indices = indices[test_number:]

        training_array = array[training_indices]
        testing_array = array[testing_indices]

        if data_vector is None:
            return training_array, testing_array
        else:
            vector = _1D_vectorized(data_vector, 'label')
            _shape_match(array, vector)
            training_data_vector = vector[training_indices]
            testing_data_vector = vector[testing_indices]
            return training_array, testing_array, training_data_vector, testing_data_vector

    elif validation:
        if not isinstance(val_size, float):
            raise TypeError(f"Validation set proportion must be a float, got {type(val_size).__name__}")
        if not (0.0 < val_size < 1.0):
            raise ValueError(f"Validation set proportion must be between 0 and 1, got {val_size}")
        if val_size + test_size >= 1.0:
            raise ValueError("Combined validation and test set proportions must be less than 1.")
                
        if not stratify is None:
            stratify_array = _1D_vectorized(stratify, 'stratify')
            if len(stratify_array) != array.shape[0]:
                raise ValueError('Stratify must have the same length as the data array')
            testing_indices, training_indices, val_indices = _stratified_indices(stratify_array, test_size, rng, validation = True, val_size = val_size)
        else:
            val_prop_remaining = val_size / (1.0 - test_size)
            indices = np.arange(array.shape[0])
            if shuffle:
                rng.shuffle(indices)

            test_number = _bounded_count(len(indices), test_size)
            testing_indices = indices[0:test_number]
            remaining_indices = indices[test_number:]

            val_number = _bounded_count(len(remaining_indices), val_prop_remaining)
            val_indices = remaining_indices[0:val_number]
            training_indices = remaining_indices[val_number:]

        training_array = array[training_indices]
        testing_array = array[testing_indices]
        val_array = array[val_indices]

        if data_vector is None:
            return training_array, testing_array, val_array
        else:
            vector = _1D_vectorized(data_vector, 'label')
            _shape_match(array, vector)
            training_data_vector = vector[training_indices]
            testing_data_vector = vector[testing_indices]
            val_data_vector = vector[val_indices]
            return training_array, testing_array, val_array, training_data_vector, testing_data_vector, val_data_vector
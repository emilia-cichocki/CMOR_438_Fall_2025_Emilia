"""
    Regression algorithms (NumPy)

    This module implements several regression algorithms (ordinary least squares linear and logistic) on 
    numeric NumPy arrays. It supports both gradient descent and direct calculation through the normal equation (linear regression).

    Functions
    ---------
    

    Classes
    ---------
   
"""

__all__ = [
    'linear_regression',
]

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]


def _validate_parameters(method: Literal['normal', 'gradient_descent'],
                         learning_rate: Optional[float] = None,
                         epochs: Optional[int] = None,
                         fit_intercept: bool = True, 
                         ) -> None:

    # TODO: add docstrings

    if method not in ('normal', 'gradient_descent'):
        raise ValueError(f"Regression method must be one of {['normal', 'gradient_descent']}")
    if learning_rate is not None and not isinstance(learning_rate, (int, float)):
        raise TypeError('Learning rate must be a float')
    if learning_rate is not None and learning_rate <= 0:
        warnings.warn(f"For model to learn properly, learning rate should be greater than zero", UserWarning)
    if epochs is not None and not isinstance(epochs, int):
        raise TypeError('Maximum epochs must be a float')
    if epochs is not None and epochs <= 0:
        raise ValueError('Maximum epochs must be greater than zero')
    if not isinstance(fit_intercept, bool):
        raise TypeError('fit_intercept parameter must be a boolean')  
    if method == 'gradient_descent' and epochs is None:
        raise ValueError('Number of epochs is required for linear descent')
    if method == 'gradient_descent' and learning_rate is None:
        raise ValueError('Learning rate is required for linear descent')
    
def _validate_arrays(data_array: Optional[ArrayLike] = None,
                     target_vector: Optional[ArrayLike] = None
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    # TODO: add docstrings

    if data_array is not None:
        array = _2D_numeric(data_array, 'data_array')

        if np.isnan(array).any():
            raise ValueError('Data array contains missing data (NaN values)')

    if target_vector is not None:
        target_vector = np.array(target_vector)
        if target_vector.ndim == 2 and (target_vector.shape[1] == 1 or target_vector.shape[0] == 1):
            vector = target_vector.reshape(-1)
            vector = _ensure_numeric(vector, 'target_vector')
        else:
            vector = _ensure_numeric(target_vector, 'target_vector')
        if np.isnan(vector).any():
            raise ValueError('Target vector contains missing data (NaN values)')

    if data_array is not None and target_vector is not None:
        _shape_match(array, vector)
        return array, vector
    elif data_array is not None:
        return array
    elif target_vector is not None:
        return vector


class linear_regression:

    # TODO: docstrings, explanation

    def __init__(self,
                 method: Literal['normal', 'gradient_descent'],
                 fit_intercept: bool = True,
                 *, 
                 learning_rate: Optional[float] = None,
                 epochs: Optional[int] = None) -> None:
        
        # TODO: docstrings

        _validate_parameters(method, learning_rate, epochs, fit_intercept)

        self.method = method
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self._training_array: Optional[np.ndarray] = None
        self._training_targets: Optional[np.ndarray] = None

    
    def fit(self, training_array: np.ndarray, training_targets: np.ndarray, random_state: Optional[int] = None, shuffle: bool = True) -> 'linear_regression':
        
        # TODO: docstrings/comments 

        if not isinstance(shuffle, bool):
            raise TypeError('Shuffle must be a boolean')
        train_array, train_targets = _validate_arrays(training_array, training_targets)
        
        self._training_array = train_array
        self._training_targets = train_targets
        
        if self.fit_intercept:
            train_array = np.hstack([np.ones((train_array.shape[0], 1)), train_array])

        if self.method == 'normal':
            first_matrix = np.matmul(train_array.T, train_array)
            second_matrix = np.matmul(train_array.T, train_targets)
            try:
                weights = np.linalg.solve(first_matrix, second_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is singular and normal equation cannot be solved; try stochastic gradient descent instead")

        elif self.method == 'gradient_descent':
            rng = _random_number(random_state)
            if self.fit_intercept:
                weights = rng.standard_normal(training_array.shape[1] + 1).reshape(-1)
            else:
                weights = rng.standard_normal(training_array.shape[1]).reshape(-1)

            for iteration in range(self.epochs):
                if shuffle:
                    indices = rng.permutation(train_array.shape[0])
                else:
                    indices = np.arange(train_array.shape[0])
                for entry in indices:
                    error = np.matmul(train_array[entry], weights) - train_targets[entry]
                    weights -= self.learning_rate * error * train_array[entry]
            
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = None
            self.coef_ = weights

        return self

    def _verify_fit(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")
        
        return self._training_array, self._training_targets
    
    def prediction(self, testing_array: np.ndarray) -> np.ndarray:
        
        # TODO: doctrings/comments

        self._verify_fit()

        test_array = _validate_arrays(testing_array)
        coef_array = _ensure_numeric(self.coef_)

        if test_array.shape[1] != len(coef_array):
            raise ValueError('Test array must have the same number of input features as coefficients')
        
        intercept = self.intercept_

        prediction = np.matmul(test_array, coef_array)

        if intercept is not None:
            prediction += intercept
            return prediction
        else:
            return prediction
    
    def scoring(self, testing_array: ArrayLike, actual_targets: ArrayLike) -> float:

        # TODO: docstrings

        predicted_target_array = self.prediction(testing_array)
        actual_target_array = _ensure_numeric(actual_targets)

        rss = np.sum((predicted_target_array - actual_target_array) ** 2)
        actual_mean = np.mean(actual_target_array)
        tss = np.sum((actual_target_array - actual_mean) ** 2)
        
        training_data_array, _ = self._verify_fit()
        testing_data_array = _2D_numeric(testing_array)

        if tss == 0:
            if np.array_equal(testing_data_array, training_data_array) and rss == 0:
                return 1.0
            else:
                raise ValueError('R2 score is undefined for constant true targets, unless evaluated on training data with a perfect fit')

        r2_score = 1 - (rss / tss)

        return r2_score
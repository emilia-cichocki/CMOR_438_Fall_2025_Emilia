"""
    Principal Component Analysis (NumPy)


    Functions
    ---------
    

    Classes
    ---------
   
"""

# TODO: finish above! and check below for appropriate imports

__all__ = [
    'PCA'
]

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.standardize import z_score_standardize
from rice_ml.supervised_learning.distances import _ensure_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

class PCA():
    
    def __init__(self,
                 n_components: int) -> None:
        
        if not isinstance(n_components, int):
            raise TypeError('Number of retained components must be an integer')
        if n_components <= 0:
            raise ValueError('Number of retained components must be greater than zero')

        self.n_components = n_components
        self.components: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.variance: Optional[np.ndarray] = None
    
    def fit(self, data_input_array: ArrayLike) -> np.ndarray:

        data_array = z_score_standardize(data_input_array)

        if self.n_components > data_array.shape[1]:
            raise ValueError('Number of retained components cannot exceed number of features')

        observation_number = data_array.shape[0]
        means = np.mean(data_array, axis = 0)

        covariance_matrix = np.dot((data_array - means).T, (data_array - means)) / (observation_number - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.eigenvalues = sorted_eigenvalues[:self.n_components]
        self.variance = sorted_eigenvalues[:self.n_components]/(np.sum(eigenvalues))
        self.components = sorted_eigenvectors[:, :self.n_components]

        return self

    def _verify_fit(self) -> 'PCA':
        if self.components is None or self.variance is None:
            raise RuntimeError("Model is not fitted; call fit(data_input_array)")

        return self

    def transform(self, data_input_array: ArrayLike) -> np.ndarray:

        self._verify_fit()
        data_array = z_score_standardize(data_input_array)
        transformed_data = np.dot(data_array, self.components)

        return transformed_data
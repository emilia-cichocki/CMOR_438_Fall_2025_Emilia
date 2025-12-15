
"""
    Principal Component Analysis (NumPy)

    This module implements Principal Component Analysis (PCA) by finding the eigenvectors
    and eigenvalues of the covariance matrix. It supports numeric NumPy arrays and
    user-specified numbers of components. 
    
    Classes
    ---------
    PCA:
        Implements PCA and returns the principal components with associated variance,
        as well as transformed data

"""

__all__ = [
    'PCA'
]

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.standardize import z_score_standardize

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

class PCA():
    
    """
    Class for implementing the PCA algorithm using eigendecomposition
    on the covariance matrix

    Attributes
    ----------
    n_components: int
        Number of principal components
    components: np.ndarray
        Array of feature loadings on each component
    eigenvalues: np.ndarray
        Array of eigenvalues for each component
    variance: np.ndarray
        Amount of explained variance from each component

    Methods
    -------
    fit(data_input_array):
        Rescales the data and implements PCA
    transform(data_input_array):
        Transforms the original data into the space described by principal components

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9]])
    >>> pca = PCA(n_components=1)
    >>> _ = pca.fit(X)
    >>> pca.transform(X)
    array([[ 0.91595659],
           [-1.96650334],
           [ 1.05054674]])
    """

    def __init__(self,
                 n_components: int) -> None:
        
        """
        Creates associated attributes for PCA with
        validated parameters

        Parameters
        ----------
        n_components: int
            Number of principal components
        """

        if not isinstance(n_components, int):
            raise TypeError('Number of retained components must be an integer')
        if n_components <= 0:
            raise ValueError('Number of retained components must be greater than zero')

        self.n_components = n_components
        self.components: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.variance: Optional[np.ndarray] = None
    
    def fit(self, data_input_array: ArrayLike) -> 'PCA':

        """
        Fits the PCA model on the input data

        Standardizes the data array and calculates the covariance matrix,
        eigenvalues, eigenvectors, variance, and component values

        Parameters
        ----------
        data_input_array: ArrayLike
            2D array-like object containing feature values with size
            (n_samples, n_features)

        Returns
        -------
        PCA
            Fitted PCA model

        Raises
        ------
        ValueError
            If the number of components is greater than the number of features
        """

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

        """
        Verifies that the model has been fitted
        """

        if self.components is None or self.variance is None:
            raise RuntimeError("Model is not fitted; call fit(data_input_array)")

        return self

    def transform(self, data_input_array: ArrayLike) -> np.ndarray:

        """
        Transforms a data array into the space described by the principal
        components

        Parameters
        ----------
        data_input_array: ArrayLike
            2D array-like object containing feature values with size
            (n_samples, n_features)

        Returns
        -------
        transformed_data: np.ndarray
            2D array containing new values for each sample on the principal 
            components

        Raises
        ------
        RuntimeError
            If the model has not been fit
        """

        self._verify_fit()
        data_array = z_score_standardize(data_input_array)
        transformed_data = np.dot(data_array, self.components)

        return transformed_data
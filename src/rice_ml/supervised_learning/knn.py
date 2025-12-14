
"""
    K-nearest neighbors algorithm (NumPy)

    This module implements the k-nearest neighbors (KNN) algorithm on numeric NumPy arrays. It
    supports classification, regression, various distance metrics ('euclidean', 'manhattan', 'minkowski'),
    and weightings. 

    Functions
    ---------
    _validate_parameters
        Ensures that all parameters are accepted
    _validate_arrays
        Ensures that provided arrays are of appropriate data type and dimension
    _distance_calculations
        Calculates pairwise distance between array rows
    _neighbor_finding
        Finds the distance and indices of the nearest neighbors to each query
    _weighting_by_distance
        Applies the specified weighting (uniform or based on distance)

    Classes
    ---------
    _knn_foundation
        Provides the base structure for KNN (fitting the model and implementing the algorithm)
    knn_classification
        Runs the KNN algorithm for classification
    knn_regressor
        Runs the KNN algorithm for regression
"""

# TODO: finish editing the above description, add examples

import numpy as np
import pandas as pd
from typing import *
import math
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import *
from rice_ml.supervised_learning.distances import _ensure_numeric, euclidean_distance, manhattan_distance, minkowski_distance

__all__ = [
    'knn_classification',
    'knn_regressor',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters(k: int, 
                         metric: Literal['euclidean', 'manhattan', 'minkowski'], 
                         weight: Literal['uniform', 'distance']
                         ) -> None:

    # TODO: add docstrings

    if not isinstance(k, (int, np.integer)):
        raise TypeError('k must be an integer')
    if k <= 0:
        raise ValueError('k must be greater than zero')
    if metric not in ('euclidean', 'manhattan', 'minkowski'):
        raise ValueError(f"Distance metric must be one of {['euclidean', 'manhattan', 'minkowski']}")
    if weight not in ('uniform', 'distance'):
        raise ValueError(f"Weighting parameter must be one of {['uniform', 'distance']}")
    
def _validate_arrays(data_array: ArrayLike, 
                     label_vector: Optional[ArrayLike] = None, 
                     regression: bool = False
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    # TODO: add docstrings

    if not isinstance(regression, bool):
        raise TypeError(f'regression parameter must be a boolean, got {type(regression).__name__}')

    array = _2D_numeric(data_array, 'data_array')

    if np.isnan(array).any():
        raise ValueError('Data array contains missing data (NaN values)')
    
    if label_vector is not None:
        vector = _1D_vectorized(label_vector, 'label_vector')
        _shape_match(array, vector)
        if regression:
            vector = _ensure_numeric(vector, 'label_vector')
            if np.isnan(vector).any():
                raise ValueError('Data vector contains missing data (NaN values)')

        return array, vector

    return array

def _distance_calculations(training_array: np.ndarray, 
                           query_array: np.ndarray, 
                           metric: str, 
                           p: Optional[int] = 3
                           ) -> np.ndarray:

    # TODO: docstrings, examples (query is row, training is column)

    query_array = _2D_numeric(query_array)
    training_array = _2D_numeric(training_array)

    distance_matrix = np.full((query_array.shape[0], training_array.shape[0]), np.nan)
    for index_1, point_1 in enumerate(query_array):
        for index_2, point_2 in enumerate(training_array):
            if metric == 'euclidean':
                distance = euclidean_distance(point_1, point_2)
            elif metric == 'manhattan':
                distance = manhattan_distance(point_1, point_2)
            elif metric == 'minkowski':
                distance = minkowski_distance(point_1, point_2, p = p)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            distance_matrix[index_1, index_2] = distance

    return distance_matrix

def _neighbor_finding(training_array: np.ndarray, 
                      query_array: np.ndarray, 
                      k: int, 
                      metric: str, 
                      p: Optional[int] = 3
                      ) -> Tuple[np.ndarray, np.ndarray]:
    
    # TODO: docstrings and examples, potentially add further checks for other inputs

    if k > training_array.shape[0]:
        raise ValueError(f'Number of neighbors (k = {k}) cannot be greater than number of training samples ({training_array.shape[0]})')
    
    distance_matrix = _distance_calculations(training_array, query_array, metric = metric, p = p)
    indices = np.argpartition(distance_matrix, kth = k - 1, axis = 1)[:, 0:k]

    query_indices = np.arange(distance_matrix.shape[0])[:, None]
    neighbor_distances = distance_matrix[query_indices, indices]
    ordering = np.argsort(neighbor_distances, axis = 1)
    sorted_indices = indices[query_indices, ordering]
    distances_sorted = neighbor_distances[query_indices, ordering]

    return distances_sorted, sorted_indices

def _weighting_by_distance(distance_array: np.ndarray, 
                           weight: str, 
                           eps: float = 1e-10
                           ) -> np.ndarray:

    # TODO: docstrings and examples

    distance_array = _validate_arrays(distance_array)

    if weight == 'uniform':
        return np.ones_like(distance_array, dtype=float)

    elif weight == 'distance':
        zeros = (distance_array < eps)
        weight_matrix = np.empty_like(distance_array, dtype = float)
        exact_neighbor = zeros.any(axis = 1)
        if np.any(exact_neighbor):
            weight_matrix[exact_neighbor] = zeros[exact_neighbor].astype(float)
        if np.any(~exact_neighbor):
            weight_matrix[~exact_neighbor] = 1.0 / np.maximum(distance_array[~exact_neighbor], eps)
        
        return weight_matrix
    



class _knn_foundation:

    # TODO: explanation/comments on the code

    def __init__(self, 
                 k: int = 3, 
                 *, 
                 metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean', 
                 weight: Literal['uniform', 'distance'] = 'uniform',
                 p: Optional[int] = 3
                 ) -> None:

        # TODO: explanation or additional comments if necessary
        
        _validate_parameters(k, metric, weight)

        self.n_neighbors = k
        self.metric = metric
        self.weight = weight
        self.p = p
        self._training: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, training_array: np.ndarray, training_labels: np.ndarray, regression: bool):

        # TODO: account for regression in the line below, fix the dimension of training_labels
        training_array, training_labels = _validate_arrays(training_array, training_labels, regression = regression)

        if self.n_neighbors > training_array.shape[0]:
            raise ValueError(f'Number of neighbors (k = {self.n_neighbors}) cannot be greater than number of training samples ({training_array.shape[0]})')
        self._training = training_array

        if regression:
            self._labels = training_labels.astype(float, copy=False)
        else:
            self._labels = training_labels

        return self
    
    def _verify_fit(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._training is None or self._labels is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_labels)")

        return self._training, self._labels
    
    def knn_implement(self, query_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        query_array = _2D_numeric(query_data)
        training_array, training_labels = self._verify_fit()

        if query_array.shape[1] != training_array.shape[1]:
            raise ValueError(f"Query data must have {training_array.shape[1]} features, got {query_array.shape[1]} features")
        
        distances, indices = _neighbor_finding(training_array, query_array, k = self.n_neighbors, metric = self.metric, p = self.p)

        return distances, indices
    

class knn_classification(_knn_foundation):

    # TODO: explanation/comments on the code

    def __init__(self, 
                 k: int = 3, 
                 *, 
                 metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean', 
                 weight: Literal['uniform', 'distance'] = 'uniform',
                 p: Optional[int] = 3
                 ) -> None:
        
        # TODO: docstrings

        super().__init__(k = k, metric = metric, weight = weight, p = p)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, training_array: np.ndarray, training_labels: np.ndarray) -> 'knn_classification':

        # TODO: docstrings

        super().fit(training_array, training_labels, regression = False)
        self.classes_ = np.unique(self._labels)

        return self

    def probabilities(self, query_data: np.ndarray) -> np.ndarray:
        
        # TODO: docstrings
        
        distances, indices = self.knn_implement(query_data)
        weighting = self.weight
        weight_matrix = _weighting_by_distance(distances, weight = weighting, eps = 1e-10)

        query_numbers = _2D_numeric(query_data).shape[0]
        class_numbers = self.classes_.shape[0]
        label_position = self._labels
        probability_matrix = np.full((query_numbers, class_numbers), np.nan)

        for query_number in range(query_numbers):
            n_neighbor_labels = label_position[indices[query_number]]
            n_neighbor_numeric = np.searchsorted(self.classes_, n_neighbor_labels)
            n_neighbor_counts = np.bincount(n_neighbor_numeric, weights = weight_matrix[query_number], minlength = class_numbers)
            total_counts = n_neighbor_counts.sum()
            if total_counts == 0:
                probability_matrix[query_number] = 1 / class_numbers
            else:
                probability_matrix[query_number] = n_neighbor_counts / total_counts
        
        return probability_matrix
    
    def prediction(self, query_data: np.ndarray) -> np.ndarray:

        # TODO: docstrings

        probability_matrix = self.probabilities(query_data)

        max_prob_location = np.argmax(probability_matrix, axis = 1)
        
        return self.classes_[max_prob_location]
    
    def scoring(self, query_data: ArrayLike, actual_labels: ArrayLike) -> float:

        # TODO: docstrings

        predicted_labels_array = self.prediction(query_data)
        actual_labels_array = _1D_vectorized(actual_labels)

        if predicted_labels_array.shape[0] != actual_labels_array.shape[0]:
            raise ValueError("Predicted labels and actual labels must have the same length")

        overlap = (predicted_labels_array == actual_labels_array).astype(float)
        mean_accuracy = float(np.mean(overlap))

        return mean_accuracy
    

class knn_regressor(_knn_foundation):

    # TODO: explanation/comments on the code

    def __init__(self, 
                 k: int = 3, 
                 *, 
                 metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean', 
                 weight: Literal['uniform', 'distance'] = 'uniform',
                 p: Optional[int] = 3
                 ) -> None:
        
        # TODO: docstrings

        super().__init__(k = k, metric = metric, weight = weight, p = p)

    def fit(self, training_array: np.ndarray, training_labels: np.ndarray) -> 'knn_regressor':

        # TODO: docstrings

        super().fit(training_array, training_labels, regression = True)
    
        return self
    
    def prediction(self, query_data: np.ndarray) -> np.ndarray:
        
        # TODO: docstrings
        
        distances, indices = self.knn_implement(query_data)
        weighting = self.weight
        weight_matrix = _weighting_by_distance(distances, weight = weighting, eps = 1e-10)
        weight_sums = weight_matrix.sum(axis = 1)

        n_neighbor_targets = self._labels[indices]
        weighted_targets = n_neighbor_targets * weight_matrix

        if np.any(weight_sums == 0).astype(bool):
            zero_weight_mask = (weight_sums == 0)
            predicted_targets = np.divide(np.sum(weighted_targets, axis=1), weight_sums, where = ~zero_weight_mask)
            predicted_targets[zero_weight_mask] = n_neighbor_targets[zero_weight_mask].mean(axis=1)
        else:
            predicted_targets = np.mean(weighted_targets, axis = 1)
    
        return predicted_targets
    
    def scoring(self, query_data: ArrayLike, actual_targets: ArrayLike) -> float:

        # TODO: docstrings

        predicted_target_array = self.prediction(query_data)
        actual_target_array = _ensure_numeric(actual_targets)

        rss = np.sum((predicted_target_array - actual_target_array) ** 2)
        actual_mean = np.mean(actual_target_array)
        tss = np.sum((actual_target_array - actual_mean) ** 2)
        
        training_data_array, _ = self._verify_fit()
        query_data_array = _2D_numeric(query_data)

        if tss == 0:
            if np.array_equal(query_data_array, training_data_array) and rss == 0:
                return 1.0
            else:
                raise ValueError('R2 score is undefined for constant true targets, unless evaluated on training data with a perfect fit')

        r2_score = 1 - (rss / tss)

        return r2_score


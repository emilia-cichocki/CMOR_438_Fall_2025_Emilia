"""
    Clustering algorithms (NumPy)

    This module implements several clustering algorithms (k-means clustering and DBSCAN)

    # TODO: finish this!

    Functions
    ---------
    

    Classes
    ---------
   
"""

# TODO: finish above! and check below for redundant imports

__all__ = [
    'k_means',
    'dbscan'
]

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric, euclidean_distance

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters_k_means(n_clusters: int, max_iterations: int, tol: float, random_state: Optional[int] = None) -> None:

    if not isinstance(n_clusters, int):
        raise TypeError('Number of clusters must be an integer')
    if n_clusters <= 0:
        raise ValueError('Number of clusters must be greater than zero')
    if not isinstance(max_iterations, int):
        raise TypeError('Maximum number of iterations must be an integer')
    if max_iterations <= 0:
        raise ValueError('Maximum number of iterations must be greater than zero')
    if not isinstance(tol, float):
        raise TypeError('Tolerance must be a float')
    if tol <= 0:
        raise ValueError('Tolerance must be greater than zero')
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError('Random state must be an integer')

class k_means():
    
    def __init__(self,
                n_clusters: int,
                max_iterations: int,
                tol: float = 1e-6,
                random_state: Optional[int] = None) -> None:
        
        _validate_parameters_k_means(n_clusters, max_iterations, tol, random_state)

        self.n_clusters = n_clusters
        self.max_iter = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.cluster_labels: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_features_: Optional[int] = None
        

    def _initial_centroids(self, training_array: np.ndarray) -> np.ndarray:

        train_array = _2D_numeric(training_array) # TODO: move this so it isn't redundant
        rng = _random_number(self.random_state)
        centroid_indices = rng.choice(train_array.shape[0], self.n_clusters, replace = False)
        initial_centroids = train_array[centroid_indices]
        
        return initial_centroids
    
    def _distance_calc(self, training_array: np.ndarray, centroid_array: np.ndarray) -> np.ndarray:

        distance_array = np.full((training_array.shape[0], self.n_clusters), np.nan)

        for sample in range(training_array.shape[0]):
            data_vector = training_array[sample]
            for centroid in range(centroid_array.shape[0]):
                centroid_vector = centroid_array[centroid]
                distance_array[sample, centroid] = euclidean_distance(data_vector, centroid_vector)

        if any(value is np.nan or value is None for value in distance_array):
            raise ValueError('Distance was not computed for all points')
        
        return distance_array
    
    def _clustering(self, training_array: np.ndarray, centroid_array: np.ndarray) -> np.ndarray:

        centroids = _2D_numeric(centroid_array)
        distance_array = self._distance_calc(training_array, centroids)
        cluster_indices = np.argmin(distance_array, axis = 1)

        return cluster_indices
    
    def _updated_centroids(self, training_array: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:

        updated_centroids = np.full((self.n_clusters, training_array.shape[1]), np.nan)

        rng = _random_number(self.random_state)
        for cluster in range(self.n_clusters):
            training_array_cluster = training_array[cluster_labels == cluster]
            if len(training_array_cluster) > 0:
                updated_centroids[cluster] = training_array_cluster.mean(axis = 0)
            else:
                updated_centroids[cluster] = training_array[rng.choice(training_array.shape[0])]
      
        if any(value is np.nan or value is None for value in updated_centroids):
            raise ValueError('Not all centroids were updated')
        
        return updated_centroids
    
    def _inertia(self, training_array: np.ndarray, centroid_array: np.ndarray, cluster_indices_array: np.ndarray) -> float:

        cluster_indices = _ensure_numeric(cluster_indices_array).astype(int)
        distance_array = self._distance_calc(training_array, centroid_array)

        actual_distances = distance_array[np.arange(training_array.shape[0]), cluster_indices]
        squared_distances = actual_distances ** 2
        final_distance = np.sum(squared_distances)

        return final_distance
    
    def fit(self, training_array: np.ndarray) -> 'k_means':

        train_array = _2D_numeric(training_array)

        self.n_features_ = train_array.shape[1]

        centroids = self._initial_centroids(train_array)

        for iteration in range(self.max_iter):
            cluster_labels = self._clustering(train_array, centroids)
            updated_centroids = self._updated_centroids(train_array, cluster_labels)

            shift = np.sum(np.linalg.norm(updated_centroids - centroids, axis = 1))

            centroids = updated_centroids

            if shift < self.tol:
                break

        inertia = self._inertia(train_array, updated_centroids, cluster_labels)

        self.centroids_ = centroids
        self.cluster_labels = cluster_labels
        self.inertia_ = inertia

        return self

    def _verify_fit(self):
        if self.centroids_ is None or self.cluster_labels is None or self.inertia_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array)")

        return self

    def prediction(self, testing_array: ArrayLike) -> np.ndarray:

        test_array = _2D_numeric(testing_array)

        if test_array.shape[1] != self.n_features_:
            raise ValueError('Test data must have the same number of features as training data')
        
        cluster_predictions = self._clustering(test_array, self.centroids_)

        return cluster_predictions
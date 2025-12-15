
"""
    Clustering algorithms (NumPy)

    This module implements several clustering algorithms (k-means clustering and DBSCAN)
    on numeric NumPy arrays. It supports Euclidean distance calculations for unsupervised
    cluster detection.

    Classes
    ---------
    k_means
        Implements the k-means algorithm using Euclidean distance
    dbscan
        Implements the DBSCAN algorithm using Euclidean distance
"""

__all__ = [
    'k_means',
    'dbscan'
]

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric, euclidean_distance

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters_k_means(n_clusters: int, max_iterations: int, tol: float, random_state: Optional[int] = None) -> None:
    
    """
    Validates hyperparameters for k-means clustering

    Parameters
    ----------
    n_clusters: int
        Number of clusters
    max_iterations: int
        Maximum number of iterations
    tol: float
        Tolerance for centroid shifts
    random_state: int, optional
        Random state, used when selecting initial centroids

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

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

    """
    Class implementing k-means clustering with Euclidean distance
    
    Covers centroid initialization, distance calculations and clustering,
    inertia scoring, and centroid updates

    Attributes
    ----------
    n_clusters: int
        Number of clusters
    max_iter: int
        Maximum number of iterations
    tol: float
        Tolerance for centroid shifts
    random_state: int or None
        Random state, used when selecting initial centroids
    cluster_labels: np.ndarray, optional
        Array of determined cluster labels
    centroids_: np.ndarray, optional
        Array denoting cluster centroids
    inertia_: float, optional
        Inertia calculation for final clustering
    n_features_: int, optional
        Number of features in the data set

    Methods
    -------
    fit(training_array):
        Fits the k-means model based on numeric data
    prediction(testing_array):
        Assigns cluster predictions to previously unseen data (not common
        in practice, but included for functionality)

    Examples
    --------
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> model = k_means(n_clusters=2, max_iterations=100, random_state=42)
    >>> _ = model.fit(X)
    >>> model.centroids_
    array([[ 1.,  2.],
           [10.,  2.]])
    >>> model.prediction(np.array([[0, 0], [12, 3]]))
    array([0, 1])
    >>> model.inertia_ > 0
    True
    """

    def __init__(self,
                n_clusters: int,
                max_iterations: int,
                tol: float = 1e-6,
                random_state: Optional[int] = None) -> None:
        
        """
        Creates associated attributes for the k-means model with
        validated parameters

        Parameters
        ----------
        n_clusters: int
            Number of clusters
        max_iter: int
            Maximum number of iterations
        tol: float
            Tolerance for centroid shifts
        random_state: int, optional
            Random state, used when selecting initial centroids
        """

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

        """
        Selecting the initial centroids out of the input data
        """

        train_array = _2D_numeric(training_array)
        rng = _random_number(self.random_state)
        centroid_indices = rng.choice(train_array.shape[0], self.n_clusters, replace = False)
        initial_centroids = train_array[centroid_indices]
        
        return initial_centroids
    
    def _distance_calc(self, training_array: np.ndarray, centroid_array: np.ndarray) -> np.ndarray:

        """
        Computing pairwise distances between input data points and the cluster
        centroids
        """

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

        """
        Finds the cluster that is closest to each input sample
        and assigns associated labels
        """
        
        centroids = _2D_numeric(centroid_array)
        distance_array = self._distance_calc(training_array, centroids)
        cluster_indices = np.argmin(distance_array, axis = 1)

        return cluster_indices
    
    def _updated_centroids(self, training_array: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:

        """
        Updates centroids by calculating the average of all points in
        a cluster
        """
        
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

        """
        Calculates inertia of the clustering scheme as the sum of squared distance
        """
        
        cluster_indices = _ensure_numeric(cluster_indices_array).astype(int)
        distance_array = self._distance_calc(training_array, centroid_array)

        actual_distances = distance_array[np.arange(training_array.shape[0]), cluster_indices]
        squared_distances = actual_distances ** 2
        final_distance = np.sum(squared_distances)

        return final_distance
    
    def fit(self, training_array: np.ndarray) -> 'k_means':

        """
        Fits the k-means clustering model on given input data

        Parameters
        ----------
        training_array: np.ndarray
            2D array-like object containing training data with size
            (n_samples, n_features)

        Returns
        -------
        k_means
            Fitted k-means clustering model

        Raises
        ------
        ValueError
            If the training array contains non-numeric data
        """

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

        """
        Verifies that the model has been fitted
        """

        if self.centroids_ is None or self.cluster_labels is None or self.inertia_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array)")

        return self

    def prediction(self, testing_array: ArrayLike) -> np.ndarray:

        """
        Assigns input samples to an existing cluster

        Parameters
        ----------
        testing_array: ArrayLike
            2D array-like object with testing samples

        Returns
        -------
        cluster_predictions: np.ndarray
            Array of cluster for each sample
        """

        self._verify_fit()

        test_array = _2D_numeric(testing_array)

        if test_array.shape[1] != self.n_features_:
            raise ValueError('Test data must have the same number of features as training data')
        
        cluster_predictions = self._clustering(test_array, self.centroids_)

        return cluster_predictions
    

def _validate_parameters_dbscan(epsilon: float, core_point_min: int) -> None:
    
    """
    Validates hyperparameters for DBSCAN

    Parameters
    ----------
    epsilon: float
        Value used to define neighborhoods
    core_point_min: int
        Minimum number of neighbors required to be considered a core point

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

    if not isinstance(epsilon, (float, int)):
        raise TypeError('Epsilon must be a float or integer')
    if epsilon < 0:
        raise ValueError('Epsilon must be non-negative')
    if not isinstance(core_point_min, int):
        raise TypeError('Minimum number of neighboring samples to be considered a core point must be an integer')
    if core_point_min < 0:
        raise ValueError('Minimum number of neighboring samples to be considered a core point must be non-negative')
    
class dbscan():

    """
    Class implementing DBSCAN with Euclidean distance
    
    Covers core point calculations, identification of noise points,
    and cluster assignments

    Attributes
    ----------
    epsilon: float
        Value used to define neighborhoods
    min_points: int
        Minimum number of neighbors required to be considered a core point
    cluster_labels: np.ndarray, optional
        Array of determined cluster labels
    core_point_indices: np.ndarray, optional
        Indices of core points

    Methods
    -------
    fit(training_array):
        Fits the DBSCAN model based on numeric data

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [1, 3],
    ...               [10, 10], [10, 11], [11, 10]])
    >>> model = dbscan(epsilon = 1.5, core_point_min = 2)
    >>> _ = model.fit(X)
    >>> model.cluster_labels
    array([0, 0, 0, 1, 1, 1])
    >>> model.core_sample_indices_
    array([0, 1, 2, 3, 4, 5])
    """

    def __init__(self,
                 epsilon: float,
                 core_point_min: int) -> None:
        
        """
        Creates associated attributes for the DBSCAN model with
        validated parameters

        Parameters
        ----------
        epsilon: float
            Value used to define neighborhoods
        core_point_min: int
            Minimum number of neighbors required to be considered a core point
        """

        _validate_parameters_dbscan(epsilon, core_point_min)

        self.epsilon = epsilon
        self.min_points = core_point_min
        self.cluster_labels: Optional[np.ndarray] = None
        self.core_point_indices: Optional[np.ndarray] = None

    def _distance_calc(self, training_array: np.ndarray) -> np.ndarray:

        """
        Calculates pairwise distances between points in the input data
        """

        n_features = training_array.shape[0]
        distance_array = np.full((n_features, n_features), np.nan)

        for feature_1 in range(n_features):
            for feature_2 in range(n_features):
                distance_array[feature_1, feature_2] = euclidean_distance(training_array[feature_1], training_array[feature_2])

        if any(value is np.nan or value is None for value in distance_array):
            raise ValueError('Distance was not computed for all points')
        
        return distance_array
    
    def _find_neighbors(self, point_index: int, distance_array: np.ndarray) -> list:
        
        """
        Finds neighboring points for all points in the input data 
        """

        distances = _2D_numeric(distance_array)
        if not isinstance(point_index, int):
            raise TypeError('Point index must be an integer')
        if point_index < 0:
            raise ValueError('Point index must be non-negative')
        if point_index >= distances.shape[0]:
            raise ValueError('Point index is out of range')
        neighbors = np.where(distances[point_index] <= self.epsilon)[0]
        neighbors = neighbors.tolist()

        return neighbors

    def _expand_region(self, 
                       cluster_labels: np.ndarray, 
                       point_index: int, 
                       neighbor_list: list,
                       cluster_id: int,
                       distance_array: np.ndarray) -> None:
        
        """
        Recursively expands a clustering region; begins with a given point
        and recruits additional neighboring points
        """

        cluster_labels[point_index] = cluster_id

        i = 0
        while i < len(neighbor_list):
            neighbor_point_index = neighbor_list[i]

            if cluster_labels[neighbor_point_index] == -1:
                cluster_labels[neighbor_point_index] = cluster_id

            new_neighbors = self._find_neighbors(neighbor_point_index, distance_array)

            if len(new_neighbors) >= self.min_points:
                for neighbor in new_neighbors:
                    if neighbor not in neighbor_list:
                        neighbor_list.append(neighbor)
            
            i += 1

    def fit(self, training_array: ArrayLike) -> 'dbscan':

        """
        Fits the DBSCAN model on given input data

        Assigns integer cluster labels to each points, and gives noise points
        a label of -1

        Parameters
        ----------
        training_array: np.ndarray
            2D array-like object containing training data with size
            (n_samples, n_features)

        Returns
        -------
        dbscan
            Fitted DBSCAN model

        Raises
        ------
        ValueError
            If the training array contains non-numeric data
        """

        train_array = _2D_numeric(training_array)
        n_samples = train_array.shape[0]

        cluster_labels = np.full(n_samples, -1, dtype = int)
        cluster_id = 0

        distance_array = self._distance_calc(train_array)

        for point_index in range(n_samples):
            if cluster_labels[point_index] != -1:
                continue
            
            neighbor_list = self._find_neighbors(point_index, distance_array)

            if len(neighbor_list) < self.min_points:
                cluster_labels[point_index] = -1
            else:
                self._expand_region(cluster_labels, point_index, neighbor_list, cluster_id, distance_array)
                cluster_id += 1

        self.cluster_labels = cluster_labels
        self.core_sample_indices_ = np.where(np.array([len(self._find_neighbors(point, distance_array)) for point in range(n_samples)]) >= self.min_points)[0]

        return self
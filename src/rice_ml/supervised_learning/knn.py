
"""
    K-nearest neighbors algorithm (NumPy)

    This module implements the k-nearest neighbors (KNN) algorithm on numeric NumPy arrays. It
    supports classification, regression, various distance metrics ('euclidean', 'manhattan', 'minkowski'),
    and weightings. 

    Classes
    ---------
    knn_classification
        Runs the KNN algorithm for classification
    knn_regressor
        Runs the KNN algorithm for regression
"""

import numpy as np
import pandas as pd
from typing import *
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

    """
    Validates hyperparameters for KNN

    Parameters
    ----------
    k: int, optional
        Number of nearest neighbors
    metric: {'euclidean', 'manhattan', 'minkowski'}
        Method of calculating distance
    weight: {'uniform', 'distance'}
        Method of weighting nearest neighbors

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

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

    """
    Ensures that data arrays are 2D and contain only numeric data,
    ensures that label vectors are 1D, and checks that the shape of 
    feature data and labels match

    Parameters
    ----------
    data_array: ArrayLike
        2D data array containing numeric values
    label_vector: ArrayLike, optional
        1D label vector (must contain numeric values if `regression` is True)
    regression: bool, default = False
        Whether `label_vector` should be treated as numeric targets

    Returns
    -------
    array: np.ndarray
        2D validated feature array
    vector: np.ndarray (if `label_vector` is not None)
        1D validated label vector

    Raises
    ------
    TypeError
        If regression is not a boolean
    ValueError
        If either input array contains NaN values or shapes of the two 
        input arrays do not match

    Examples
    --------
    >>> _validate_arrays([[1, 2], [3, 4]])
    array([[1., 2.],
           [3., 4.]])
    """

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

    """
    Computes the pairwise distances between a set of query points and
    the training points

    Parameters
    ----------
    training_array: np.ndarray
        2D array of training data with size (n_samples, n_features)
    query_array: np.ndarray
        2D array of query data with size (n_queries, n_features)
    metric: {'euclidean', 'manhattan', 'minkowski'}
        Metric for calculating distance
    p: int, optional, default = 3
        Order used for Minkowski distance

    Returns
    -------
    distance_matrix np.ndarray
        2D distance matrix with size (query_samples, training_samples)

    Examples
    --------
    >>> _distance_calculations([[0,0],[1,1]], [[1,0]], metric = 'euclidean')
    array([[1., 1.]])
    """

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
    
    """
    Determines the k-nearest neighbors for each query point

    Parameters
    ----------
    training_array: np.ndarray
        2D array of training data with size (n_samples, n_features)
    query_array: np.ndarray
        2D array of query data with size (n_queries, n_features)
    k: int
        Number of nearest neighbors
    metric: {'euclidean', 'manhattan', 'minkowski'}
        Metric for calculating distance
    p: int, optional, default = 3
        Order used for Minkowski distance

    Returns
    -------
    distances_sorted: np.ndarray
        Sorted distances of nearest neighbors for each query point
    sorted_indices: np.ndarray
        Indices of the k-nearest neighbors to a query point in the 
        training array

    Raises
    ------
    ValueError
        If k is larger than the number of training samples

    Examples
    --------
    >>> _neighbor_finding(np.array([[0,0],[1,1]]), np.array([[0,1]]), k=1, metric='euclidean')
    (array([[1.]]), array([[0]]))
    """

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

    """
    Computes weights for a neighbor to a query based on distance

    Parameters
    ----------
    distance_array: np.ndarray
        2D array of distances for neighbors to queries
    weight: {'uniform', 'distance'}
        Method used for weighting (uniform weight or inversely proportional
        to distance)
    eps: float, default = 1e-10
        Small value to prevent zero division

    Returns
    -------
    weight_matrix: np.ndarray
        2D matrix with same size as distance_array

    Examples
    --------
    >>> _weighting_by_distance([[0.5, 1.5]], weight='uniform')
    array([[1., 1.]])
    """

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

    """
    Foundation class for use in KNN classification and regression
    
    Covers k-nearest neighbor calculation, application of distance metrics, 
    and weighting method

    Attributes
    ----------
    k: int
        Number of nearest neighbors
    metric: {'euclidean', 'manhattan', 'minkowski'}, default = 'euclidean'
        Distance metric used for nearest neighbor calculations
    weight: {'uniform', 'distance'}, default = 'uniform'
        Method for weighting neighbors
    p: int, default = 3
        Order for calculating Minkowski distance
    _training: np.ndarray
        Training data array, set after fitting
    _labels: np.ndarray
        Training label array, set after fitting

    Methods
    -------
    fit(training_array, training_labels):
        Fits the KNN based on training values and numeric data
    knn_implement(query_data):
        Finds the distances and indices of the nearest neighbors to a query
    """

    def __init__(self, 
                 k: int = 3, 
                 *, 
                 metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean', 
                 weight: Literal['uniform', 'distance'] = 'uniform',
                 p: Optional[int] = 3
                 ) -> None:

        """
        Creates associated attributes for a base KNN with
        validated parameters

        Parameters
        ----------
        k: int
        Number of nearest neighbors
        metric: {'euclidean', 'manhattan', 'minkowski'}, default = 'euclidean'
            Distance metric used for nearest neighbor calculations
        weight: {'uniform', 'distance'}, default = 'uniform'
            Method for weighting neighbors
        p: int, default = 3
            Order for calculating Minkowski distance
        """
        
        _validate_parameters(k, metric, weight)

        self.n_neighbors = k
        self.metric = metric
        self.weight = weight
        self.p = p
        self._training: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, training_array: np.ndarray, training_labels: np.ndarray, regression: bool) -> "_knn_foundation":

        """
        Fits the KNN on given input data

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_labels: ArrayLike
            1D array-like object containing training labels

        Returns
        -------
        _knn_foundation
            Fitted model

        Raises
        ------
        ValueError
            If number of neighbors is greater than number of training samples
        """
        
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

        """
        Verifies that the KNN model has been fitted
        """

        if self._training is None or self._labels is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_labels)")

        return self._training, self._labels
    
    def knn_implement(self, query_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Computes the k-nearest neighbors for each query point

        Parameters
        ----------
        query_data: np.ndarray
            2D array of query points with size (n_queries, n_features)
        Returns
        -------
        distances: np.ndarray
            2D array of distances from each query point to its nearest neighbors
        indices : np.ndarray
            2D of indices of the nearest neighbors for each query point

        Raises
        ------
        ValueError
            If query data has a different number of features than training data
        RuntimeError
            If the model has not been fitted
        """
    
        query_array = _2D_numeric(query_data)
        training_array, training_labels = self._verify_fit()

        if query_array.shape[1] != training_array.shape[1]:
            raise ValueError(f"Query data must have {training_array.shape[1]} features, got {query_array.shape[1]} features")
        
        distances, indices = _neighbor_finding(training_array, query_array, k = self.n_neighbors, metric = self.metric, p = self.p)

        return distances, indices
    
class knn_classification(_knn_foundation):

    """
    Implements the KNN classification algorithm using the foundational KNN
    class

    Attributes
    ----------
    k: int
        Number of nearest neighbors
    metric: {'euclidean', 'manhattan', 'minkowski'}, default = 'euclidean'
        Distance metric used for nearest neighbor calculations
    weight: {'uniform', 'distance'}, default = 'uniform'
        Method for weighting neighbors
    p: int, default = 3
        Order for calculating Minkowski distance
    _training: np.ndarray
        Training data array, set after fitting
    _labels: np.ndarray
        Training label array, set after fitting
    classes_: np.ndarray
        Classes for the data

    Methods
    -------
    fit(training_array, training_labels):
        Fits the KNN classifier based on training values and numeric data
    probabilities(query_data):
        Computes the probability of each label for a query sample
    prediction(query_data):
        Predicts the class label for a query sample
    scoring(query_data, actual_labels)
        Scores the classifier using accuracy
    
    Examples
    --------
    >>> knn = knn_classification(k=3)
    >>> _ = knn.fit([[0,0],[1,1],[2,2]], [0,1,1])
    >>> knn.prediction([[1,0]])
    array([1])
    """

    def __init__(self, 
                 k: int = 3, 
                 *, 
                 metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean', 
                 weight: Literal['uniform', 'distance'] = 'uniform',
                 p: Optional[int] = 3
                 ) -> None:
        
        """
        Creates associated attributes for a KNN classifier with
        validated parameters

        Parameters
        ----------
        k: int
        Number of nearest neighbors
        metric: {'euclidean', 'manhattan', 'minkowski'}, default = 'euclidean'
            Distance metric used for nearest neighbor calculations
        weight: {'uniform', 'distance'}, default = 'uniform'
            Method for weighting neighbors
        p: int, default = 3
            Order for calculating Minkowski distance
        """

        super().__init__(k = k, metric = metric, weight = weight, p = p)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, training_array: np.ndarray, training_labels: np.ndarray) -> 'knn_classification':

        """
        Fits the KNN on given input data

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_labels: ArrayLike
            1D array-like object containing training labels

        Returns
        -------
        knn_classification
            Fitted KNN classification model

        Raises
        ------
        ValueError
            If number of neighbors is greater than number of training samples
        """

        super().fit(training_array, training_labels, regression = False)
        self.classes_ = np.unique(self._labels)

        return self

    def probabilities(self, query_data: np.ndarray) -> np.ndarray:
        
        """
        Computes probabilities that a query sample belongs to a given class

        Applies the given weighting metric in computing probabilities
        based on neighbor distance, if applicable

        Parameters
        ----------
        query_data: np.ndarray
            2D data array with size (n_queries, n_features)

        Returns
        -------
        probability_matrix: np.ndarray
            2D array of size (n_queries, n_classes) containing probabilites that
            a sample belongs to a class

        Raises
        ------
        ValueError
            If the number of features in `query_data` does not match the number of 
            training features, or if `query_data` contains non-numeric values
        """
        
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

        """
        Predicts the target class for given query samples

        Parameters
        ----------
        query_data: ArrayLike
            2D array-like object of size (n_queries, n_features)

        Returns
        -------
        pred: np.ndarray
            Array of predicted target class for each sample
        """

        probability_matrix = self.probabilities(query_data)

        max_prob_location = np.argmax(probability_matrix, axis = 1)
        
        pred = self.classes_[max_prob_location]

        return pred
    
    def scoring(self, query_data: ArrayLike, actual_labels: ArrayLike) -> float:

        """
        Calculates accuracy score for classification on query data

        Parameters
        ----------
        query_data: ArrayLike
            2D array-like object of size (n_queries, n_features)
        actual_labels: ArrayLike
            1D array-like object of true labels with size (n_queries,)

        Returns
        -------
        mean_accuracy: float
            Mean accuracy of the predictions
        """

        predicted_labels_array = self.prediction(query_data)
        actual_labels_array = _1D_vectorized(actual_labels)

        if predicted_labels_array.shape[0] != actual_labels_array.shape[0]:
            raise ValueError("Predicted labels and actual labels must have the same length")

        overlap = (predicted_labels_array == actual_labels_array).astype(float)
        mean_accuracy = float(np.mean(overlap))

        return mean_accuracy
    

class knn_regressor(_knn_foundation):

    """
    Implements the KNN regression algorithm using the foundational KNN
    class

    Attributes
    ----------
    k: int
        Number of nearest neighbors
    metric: {'euclidean', 'manhattan', 'minkowski'}, default = 'euclidean'
        Distance metric used for nearest neighbor calculations
    weight: {'uniform', 'distance'}, default = 'uniform'
        Method for weighting neighbors
    p: int, default = 3
        Order for calculating Minkowski distance
    _training: np.ndarray
        Training data array, set after fitting
    _labels: np.ndarray
        Training label array, set after fitting

    Methods
    -------
    fit(training_array, training_labels):
        Fits the KNN classifier based on training values and numeric data
    prediction(query_data):
        Predicts the class label for a query sample
    scoring(query_data, actual_targets)
        Scores the classifier using accuracy
    
    Examples
    --------
    >>> import numpy as np
    >>> X_train = np.array([[1], [2], [3], [4]])
    >>> y_train = np.array([1.0, 2.0, 1.5, 3.5])
    >>> knn = knn_regressor(k=2)
    >>> _ = knn.fit(X_train, y_train)
    >>> X_test = np.array([[2.5]])
    >>> knn.prediction(X_test)
    array([1.75])
    """

    def __init__(self, 
                 k: int = 3, 
                 *, 
                 metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean', 
                 weight: Literal['uniform', 'distance'] = 'uniform',
                 p: Optional[int] = 3
                 ) -> None:
        
        """
        Creates associated attributes for a KNN regressor with
        validated parameters

        Parameters
        ----------
        k: int
            Number of nearest neighbors
        metric: {'euclidean', 'manhattan', 'minkowski'}, default = 'euclidean'
            Distance metric used for nearest neighbor calculations
        weight: {'uniform', 'distance'}, default = 'uniform'
            Method for weighting neighbors
        p: int, default = 3
            Order for calculating Minkowski distance
        """

        super().__init__(k = k, metric = metric, weight = weight, p = p)

    def fit(self, training_array: np.ndarray, training_labels: np.ndarray) -> 'knn_regressor':
        
        """
        Fits the KNN on given input data

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_labels: ArrayLike
            1D array-like object containing training values

        Returns
        -------
        knn_regressor
            Fitted model

        Raises
        ------
        ValueError
            If number of neighbors is greater than number of training samples
        """

        super().fit(training_array, training_labels, regression = True)
    
        return self
    
    def prediction(self, query_data: np.ndarray) -> np.ndarray:
        
        """
        Predicts the target values for given query samples, applying
        a given weighting method

        Parameters
        ----------
        query_data: ArrayLike
            2D array-like object of size (n_queries, n_features)

        Returns
        -------
        predicted_targets: np.ndarray
            Array of predicted target values for each sample
        """
        
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

        
        """
        Calculates R2 score for regression on query data

        Parameters
        ----------
        query_data: ArrayLike
            2D array-like object of size (n_queries, n_features)
        actual_targets: ArrayLike
            1D array-like object of true target values with size (n_queries,)

        Returns
        -------
        r2_score: float
            R2 score for the predictions
        """

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


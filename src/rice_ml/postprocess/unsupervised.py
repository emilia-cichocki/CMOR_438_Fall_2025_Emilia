
"""
    Postprocessing utilities for unsupervised learning (NumPy)
    
    This module calculates a comprehensive set of postprocessing and evaluation 
    metrics for unsupervised learning algorithms, with support for NumPy arrays.

    Functions
    ---------
    silhouette_score
        Computes the silhouette score for cluster analysis
    evaluate_clusters
        Calculates and prints the number of clusters, noise points, and 
        points per cluster
"""

import numpy as np
from typing import *
from rice_ml.preprocess.datatype import *
from rice_ml.supervised_learning.distances import _ensure_numeric
import collections
from collections import Counter

__all__ = [
    'silhouette_score',
    'evaluate_clusters'
]

def _validate_array_match(data_array: np.ndarray, label_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Validation of input vectors
    
    Converts the data array to a 2D numeric array, the label array to a 1D
    numeric array, and checks that they have the same number of samples

    Parameters
    ----------
    data_array: np.ndarray
        Array of shape (n_samples, n_features)
    label_array: np.ndarray
        Array of labels with shape (n_samples,)

    Returns
    -------
    data_array: np.ndarray
        2D array of features for each sample
    label_array: np.ndarray
        1D array of labels

    Raises
    ------
    ValueError
        If data and label arrays have a different first dimension
    """

    data_array = _2D_numeric(data_array)
    label_array = _ensure_numeric(label_array)
    if data_array.shape[0] != label_array.shape[0]:
        raise ValueError('Data and label array must have the same number of samples')
    
    return data_array, label_array

def silhouette_score(data_array: np.ndarray, label_array: np.ndarray, ignore_noise: bool = False) -> float:

    """
    Computes the mean silhouette score across all samples

    The silhouette score describes how close a sample is to other samples
    in the same cluster compared to those in other clusters

    Parameters
    ----------
    data_array: np.ndarray
        Array of shape (n_samples, n_features)
    label_array: np.ndarray
        Array of cluster labels with shape (n_samples,)
    ignore_noise: bool, default = False
        Whether noise (samples labeled with -1) is ignored

    Returns
    -------
    mean_score: float
        Mean silhouette score over all samples

    Raises
    ------
    TypeError
        If the `ignore_noise` parameter is not a boolean
    ValueError
        If labels describe less than two clusters

    Examples
    --------
    >>> X = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
    >>> labels_with_noise = np.array([0, 0, 1, -1])
    >>> silhouette_score(X, labels_with_noise, ignore_noise = True)
    0.9008016272913615
    """

    data_array, label_array = _validate_array_match(data_array, label_array)
    if not isinstance(ignore_noise, bool):
        raise TypeError("ignore_noise parameter must be a boolean")
    
    unique_labels = np.unique(label_array)

    if ignore_noise:

        mask = (label_array != -1)
        data_array = data_array[mask]
        label_array = label_array[mask]
        unique_labels = unique_labels[unique_labels != -1]
    
    if len(unique_labels) <= 1:
        raise ValueError("Silhouette score requires at least two clusters")
    
    n_samples = data_array.shape[0]
    score = np.zeros(n_samples)

    for i in range(n_samples):
        self_cluster = label_array[i]
        self_mask = (label_array == self_cluster)

        if np.sum(self_mask) > 1:
            same_mask = (label_array == label_array[i])
            same_mask[i] = False
            if np.any(same_mask):
                same_dist = np.mean(np.linalg.norm(data_array[i] - data_array[same_mask], axis=1))
            else:
                same_dist = 0
        else:
            same_dist = 0
        
        other_dist = np.inf
        
        for label in unique_labels:
            if label == self_cluster:
                continue
            cluster_mask = (label_array == label)

            mean_distance = np.mean(np.linalg.norm(data_array[i] - data_array[cluster_mask], axis = 1))

            if mean_distance < other_dist:
                other_dist = mean_distance
    
        denom = max(same_dist, other_dist)
        if denom != 0:
            score[i] = (other_dist - same_dist) / denom
        else:
            score[i] = 0
        
    mean_score = np.mean(score)

    return mean_score

def evaluate_clusters(label_array: np.ndarray, print_eval: bool = True) -> Tuple[int, Union[int, str], collections.Counter]:

    """
    Calculates and optionally prints the set of cluster evaluation metrics

    Includes number of clusters, number of noise points, and number of samples
    in each cluster

    Parameters
    ----------
    data_array: np.ndarray
        Array of shape (n_samples, n_features)
    label_array: np.ndarray
        Array of cluster labels with shape (n_samples,)
    print_eval: bool, default = True
        Whether to print the cluster evaluation metrics

    Returns
    -------
    n_clusters: int
        Number of clusters, excluding noise
    n_noise: int or str
        Number of noise points (if present, otherwise the string
        'No noise points')
    cluster_counts: collections.Counter
        Counter mapping cluster labels to number of samples

    Raises
    ------
    TypeError
        If the `print_eval` parameter is not a boolean

    """

    label_array = _ensure_numeric(label_array)
    if not isinstance(print_eval, bool):
        raise TypeError("print_eval parameter must be a boolean")
    
    n_clusters = len(set(label_array)) - (1 if -1 in label_array else 0)

    if -1 in label_array:
        n_noise = np.sum(label_array == -1)
    else:
        n_noise = 'No noise points'

    cluster_counts = Counter(label_array)

    if print_eval:
        print(f"Cluster Evaluation Metrics:")
        print(f"{'-' * 27}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print("Cluster counts (including noise as -1):")
        for cluster_label, count in cluster_counts.items():
            print(f"  Cluster {cluster_label}: {count} points")
    
    return n_clusters, n_noise, cluster_counts

"""
    Postprocessing utilities for unsupervised learning (Numpy)
    # TODO: finish this!!

"""

import numpy as np
from typing import *
from rice_ml.preprocess.datatype import *
from rice_ml.supervised_learning.distances import _ensure_numeric # TODO: finish this!!
import collections
from collections import Counter

__all__ = [
    'silhouette_score',
    'evaluate_clusters'
]

def _validate_array_match(data_array: np.ndarray, label_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    data_array = _2D_numeric(data_array)
    label_array = _ensure_numeric(label_array)
    if data_array.shape[0] != label_array.shape[0]:
        raise ValueError('Data and label array must have the same number of samples')
    
    return data_array, label_array

def silhouette_score(data_array: np.ndarray, label_array: np.ndarray, ignore_noise: bool = False) -> float:

    # TODO: type hints, docstrings

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
            same_mask[i] = False  # remove self
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

def evaluate_clusters(label_array: np.ndarray, print_eval: bool = True) -> Tuple[float, Union[float, str], collections.Counter]:

    # TODO: type hints, docstrings

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
# TODO: unit tests
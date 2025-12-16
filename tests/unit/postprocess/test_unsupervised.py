
import numpy as np
from typing import *
import pytest
from rice_ml.postprocess.unsupervised import _validate_array_match, silhouette_score, evaluate_clusters
from collections import Counter

def test_validate_array_match_basic():
    test_data_array = np.array([[1, 1], [0, 1]])
    test_label_array = np.array([1, 1])
    data, label = _validate_array_match(test_data_array, test_label_array)
    assert isinstance(data, np.ndarray)
    assert isinstance(label, np.ndarray)
    assert np.allclose(data, test_data_array)
    assert np.allclose(label, test_label_array)

def test_validate_array_match_lists():
    test_data_array = [[1, 1], [0, 1]]
    test_label_array = [1, 1]
    data, label = _validate_array_match(test_data_array, test_label_array)
    assert isinstance(data, np.ndarray)
    assert isinstance(label, np.ndarray)

def test_validate_array_match_non_num():
    test_data_array = np.array([[1, 'A'], [0, 1]])
    test_label_array = np.array([1, 1])
    with pytest.raises(TypeError):
        _validate_array_match(test_data_array, test_label_array)

def test_validate_array_match_non_num_labels():
    test_data_array = np.array([[1, 1], [0, 1]])
    test_label_array = np.array([1, 'A'])
    with pytest.raises(TypeError):
        _validate_array_match(test_data_array, test_label_array)

def test_validate_array_match_dimension_data():
    test_data_array = np.array([[[1, 1], [0, 1]]])
    test_label_array = np.array([1, 1])
    with pytest.raises(ValueError):
        _validate_array_match(test_data_array, test_label_array)

def test_validate_array_match_dimension_data():
    test_data_array = np.array([[1, 1], [0, 1]])
    test_label_array = np.array([[1, 1]])
    with pytest.raises(ValueError):
        _validate_array_match(test_data_array, test_label_array)

def test_silhouette_score_basic():
    test_array = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
    test_labels = np.array([0, 0, 1, 1])
    score = silhouette_score(test_array, test_labels)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_silhouette_score_noise_keep():
    test_array = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
    test_labels = np.array([0, 0, 1, -1])
    score = silhouette_score(test_array, test_labels, False)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_silhouette_score_noise_ignore():
    test_array = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
    test_labels = np.array([0, 0, 1, -1])
    score = silhouette_score(test_array, test_labels, True)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_silhouette_score_noise_ignore_type():
    test_array = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
    test_labels = np.array([0, 0, 1, -1])
    with pytest.raises(TypeError):
        silhouette_score(test_array, test_labels, 'True')

def test_silhouette_score_labels():
    test_array = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
    test_labels = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError):
        silhouette_score(test_array, test_labels)

def test_silhouette_score_shape_mismatch():
    test_array = np.array([[0, 0], [0, 1], [5, 5]])
    test_labels = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        silhouette_score(test_array, test_labels)

def test_evaluate_clusters_basic():
    test_labels = np.array([0, 1, 1, 0])
    n, n_noise, c = evaluate_clusters(test_labels, print_eval = False)
    assert isinstance(n, int)
    assert isinstance(n_noise, str)
    assert isinstance(c, Counter)
    assert n == 2

def test_evaluate_clusters_noise():
    test_labels = np.array([0, 1, 1, -1])
    n, n_noise, c = evaluate_clusters(test_labels, print_eval = False)
    assert isinstance(n, int)
    assert isinstance(n_noise, np.int64)
    assert isinstance(c, Counter)
    assert n == 2
    assert n_noise == 1

def test_evaluate_clusters_non_num():
    test_labels = np.array([0, 1, 'A', -1])
    with pytest.raises(TypeError):
        evaluate_clusters(test_labels, print_eval = False)

def test_evaluate_clusters_eval_type():
    test_labels = np.array([0, 1, 1, -1])
    with pytest.raises(TypeError):
        evaluate_clusters(test_labels, print_eval = 'False')
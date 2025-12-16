
import numpy as np
import pandas as pd
import pytest
from rice_ml.unsupervised_learning.clustering import _validate_parameters_k_means, _validate_parameters_dbscan, k_means, dbscan

def test_kmeans_fit_basic_array():
    test_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(test_array)
    assert kmeans.n_features_ is not None
    assert isinstance(kmeans.n_features_, int)
    assert kmeans.n_features_ == 2
    assert kmeans.centroids_ is not None
    assert isinstance(kmeans.centroids_, np.ndarray)
    assert np.allclose(kmeans.centroids_, [[0.5, 0.5], [10.5, 10.5]])
    assert kmeans.cluster_labels is not None
    assert isinstance(kmeans.cluster_labels, np.ndarray)
    assert np.allclose(kmeans.cluster_labels, [0, 0, 1, 1])
    assert kmeans.inertia_ is not None
    assert isinstance(kmeans.inertia_, float)

def test_kmeans_fit_basic_list():
    test_array = [[0, 0],[1, 1],[10, 10],[11, 11]]
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(test_array)
    assert kmeans.n_features_ is not None
    assert isinstance(kmeans.n_features_, int)
    assert kmeans.n_features_ == 2
    assert kmeans.centroids_ is not None
    assert isinstance(kmeans.centroids_, np.ndarray)
    assert np.allclose(kmeans.centroids_, [[0.5, 0.5], [10.5, 10.5]])
    assert kmeans.cluster_labels is not None
    assert isinstance(kmeans.cluster_labels, np.ndarray)
    assert np.allclose(kmeans.cluster_labels, [0, 0, 1, 1])
    assert kmeans.inertia_ is not None
    assert isinstance(kmeans.inertia_, float)

def test_kmeans_fit_basic_tuple():
    test_array = ([0, 0],[1, 1],[10, 10],[11, 11])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(test_array)
    assert kmeans.n_features_ is not None
    assert isinstance(kmeans.n_features_, int)
    assert kmeans.n_features_ == 2
    assert kmeans.centroids_ is not None
    assert isinstance(kmeans.centroids_, np.ndarray)
    assert np.allclose(kmeans.centroids_, [[0.5, 0.5], [10.5, 10.5]])
    assert kmeans.cluster_labels is not None
    assert isinstance(kmeans.cluster_labels, np.ndarray)
    assert np.allclose(kmeans.cluster_labels, [0, 0, 1, 1])
    assert kmeans.inertia_ is not None
    assert isinstance(kmeans.inertia_, float)

def test_kmeans_fit_tolerance():
    test_array = np.array([[0,0], [0.1,0], [0,0.1], [5,5], [5.1,5], [5,5.1],[10,10], [10.1,10], [10,10.1]])
    kmeans_1 = k_means(2, 50, 1e-3, 42)
    kmeans_1.fit(test_array)
    kmeans_2 = k_means(2, 1, 1e-3, 42)
    kmeans_2.fit(test_array)
    assert not np.allclose(kmeans_1.centroids_, kmeans_2.centroids_)

def test_kmeans_fit_random_state():
    test_array = np.random.rand(20, 2)
    kmeans_1 = k_means(2, 50, 1e-3, 42)
    kmeans_1.fit(test_array)
    kmeans_2 = k_means(2, 50, 1e-3, 42)
    kmeans_2.fit(test_array)
    assert np.allclose(kmeans_1.centroids_, kmeans_2.centroids_)
    assert np.all(kmeans_1.cluster_labels == kmeans_2.cluster_labels)
    assert np.isclose(kmeans_1.inertia_, kmeans_2.inertia_)

def test_kmeans_fit_mutation():
    test_array = np.random.rand(20, 2)
    test_array_copy = test_array.copy()
    kmeans = k_means(2, 50, 1e-3, 42)
    kmeans.fit(test_array)
    assert np.allclose(test_array, test_array_copy)

def test_kmeans_fit_type_training_data():
    test_array = np.array([['0', '0'], ['1', '1'], ['10', '10'], ['11', '11']])
    kmeans = k_means(2, 50, 1e-3, 42)
    kmeans.fit(test_array)
    assert kmeans.n_features_ is not None
    assert isinstance(kmeans.n_features_, int)
    assert kmeans.n_features_ == 2
    assert kmeans.centroids_ is not None
    assert isinstance(kmeans.centroids_, np.ndarray)
    assert np.allclose(kmeans.centroids_, [[0.5, 0.5], [10.5, 10.5]])
    assert kmeans.cluster_labels is not None
    assert isinstance(kmeans.cluster_labels, np.ndarray)
    assert np.allclose(kmeans.cluster_labels, [0, 0, 1, 1])
    assert kmeans.inertia_ is not None
    assert isinstance(kmeans.inertia_, float)

def test_kmeans_fit_type_training():
    test_array = 'np.array([[0, 0],[1, 1],[10, 10],[11, 11]])'
    kmeans = k_means(2, 50, 1e-3, 42)
    with pytest.raises(TypeError):
        kmeans.fit(test_array)

def test_kmeans_fit_dimension_training_2D():
    test_array = np.array([0, 0])
    kmeans = k_means(2, 50, 1e-3, 42)
    with pytest.raises(ValueError):
        kmeans.fit(test_array)

def test_kmeans_fit_dimension_training_3D():
    test_array = np.array([[[0, 0],[1, 1],[10, 10],[11, 11]]])
    kmeans = k_means(2, 50, 1e-3, 42)
    with pytest.raises(ValueError):
        kmeans.fit(test_array)

def test_kmeans_prediction_basic():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = np.array([[0.5, 0.2], [10.2, 10.5]])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    predictions = kmeans.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_kmeans_prediction_centroids():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = np.array([[0.5, 0.5], [10.5, 10.5]])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    predictions = kmeans.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_kmeans_prediction_testing_string_numeric():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = np.array([['0.5', '0.2'], ['10.2', '10.5']])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    predictions = kmeans.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_kmeans_prediction_testing_string_nonnumeric():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = np.array([['A', 'A'], ['B', 'B']])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    with pytest.raises(TypeError):
        kmeans.prediction(test_array)

def test_kmeans_prediction_type_testing():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = 'np.array([[0.5, 0.2], [10.2, 10.5]])'
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    with pytest.raises(TypeError):
        kmeans.prediction(test_array)

def test_kmeans_prediction_dimension_testing_1D():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = np.array([0.5, 0.2])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    with pytest.raises(ValueError):
        kmeans.prediction(test_array)

def test_kmeans_prediction_dimension_testing_3D():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = np.array([[[0.5, 0.2]]])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    with pytest.raises(ValueError):
        kmeans.prediction(test_array)

def test_kmeans_prediction_shape_mismatch():
    train_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    test_array = np.array([[0.5, 0.2, 0.1], [10.2, 10.5, 10.1]])
    kmeans = k_means(2, 50, 1e-10, 42)
    kmeans.fit(train_array)
    with pytest.raises(ValueError):
        kmeans.prediction(test_array)

def test_dbscan_fit_basic_array():
    test_array = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    dbs = dbscan(0.5, 2)
    dbs.fit(test_array)
    cluster_ids = set(dbs.cluster_labels[dbs.cluster_labels != -1])
    assert isinstance(dbs.cluster_labels, np.ndarray)
    assert dbs.cluster_labels.shape == (4,)
    assert np.allclose(dbs.cluster_labels, [0, 0, 1, 1])
    assert isinstance(dbs.core_sample_indices_, np.ndarray)
    assert np.allclose(dbs.core_sample_indices_, [0, 1, 2, 3])
    assert cluster_ids == set(range(len(cluster_ids)))

def test_dbscan_fit_basic_list():
    test_array = [[0, 0], [0.1, 0], [5, 5], [5.1, 5]]
    dbs = dbscan(0.5, 2)
    dbs.fit(test_array)
    assert isinstance(dbs.cluster_labels, np.ndarray)
    assert dbs.cluster_labels.shape == (4,)
    assert np.allclose(dbs.cluster_labels, [0, 0, 1, 1])
    assert isinstance(dbs.core_sample_indices_, np.ndarray)
    assert np.allclose(dbs.core_sample_indices_, [0, 1, 2, 3])

def test_dbscan_fit_basic_tuple():
    test_array = ([0, 0], [0.1, 0], [5, 5], [5.1, 5])
    dbs = dbscan(0.5, 2)
    dbs.fit(test_array)
    assert isinstance(dbs.cluster_labels, np.ndarray)
    assert dbs.cluster_labels.shape == (4,)
    assert np.allclose(dbs.cluster_labels, [0, 0, 1, 1])
    assert isinstance(dbs.core_sample_indices_, np.ndarray)
    assert np.allclose(dbs.core_sample_indices_, [0, 1, 2, 3])

def test_dbscan_fit_noisy():
    test_array = np.array([[0, 0], [10, 10]])
    dbs = dbscan(0.5, 2)
    dbs.fit(test_array)
    assert all(dbs.cluster_labels == -1)

def test_dbscan_fit_multiple_runs():
    test_array = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    dbs_1 = dbscan(0.5, 2)
    dbs_1.fit(test_array)
    dbs_2 = dbscan(0.5, 2)
    dbs_2.fit(test_array)
    assert np.allclose(dbs_1.cluster_labels, dbs_2.cluster_labels)

def test_dbscan_fit_type_training_data_string_numeric():
    test_array = np.array([['0', '0'], [0.1, 0], [5, 5], [5.1, 5]])
    dbs = dbscan(0.5, 2)
    dbs.fit(test_array)
    assert isinstance(dbs.cluster_labels, np.ndarray)
    assert dbs.cluster_labels.shape == (4,)
    assert np.allclose(dbs.cluster_labels, [0, 0, 1, 1])
    assert isinstance(dbs.core_sample_indices_, np.ndarray)
    assert np.allclose(dbs.core_sample_indices_, [0, 1, 2, 3])

def test_dbscan_fit_type_training_data_string_nonnumeric():
    test_array = np.array([['A', 'A'], [0.1, 0], [5, 5], [5.1, 5]])
    dbs = dbscan(0.5, 2)
    with pytest.raises(TypeError):
        dbs.fit(test_array)

def test_dbscan_fit_dimension_training_data_1D():
    test_array = np.array([0, 0])
    dbs = dbscan(0.5, 2)
    with pytest.raises(ValueError):
        dbs.fit(test_array)

def test_dbscan_fit_dimension_training_data_3D():
    test_array = np.array([[[0, 0]]])
    dbs = dbscan(0.5, 2)
    with pytest.raises(ValueError):
        dbs.fit(test_array)
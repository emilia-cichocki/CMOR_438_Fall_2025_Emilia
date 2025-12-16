
import numpy as np
import pandas as pd
import pytest
from rice_ml.unsupervised_learning.clustering import _validate_parameters_k_means, _validate_parameters_dbscan, k_means, dbscan

def test_validate_parameters_kmeans_basic():
    _validate_parameters_k_means(n_clusters = 3, max_iterations = 50, tol = 1e-10)

def test_validate_parameters_kmeans_type_clusters():
    with pytest.raises(TypeError):
        _validate_parameters_k_means(n_clusters = '3', max_iterations = 50, tol = 1e-10)

def test_validate_parameters_kmeans_type_iter():
    with pytest.raises(TypeError):
        _validate_parameters_k_means(n_clusters = 3, max_iterations = '50', tol = 1e-10)

def test_validate_parameters_kmeans_type_tol():
    with pytest.raises(TypeError):
        _validate_parameters_k_means(n_clusters = 3, max_iterations = 50, tol = '1e-10')

def test_validate_parameters_kmeans_type_random():
    with pytest.raises(TypeError):
        _validate_parameters_k_means(n_clusters = 3, max_iterations = 50, tol = 1e-10, random_state = '42')

def test_validate_parameters_kmeans_cluster_value():
    with pytest.raises(ValueError):
        _validate_parameters_k_means(n_clusters = -1, max_iterations = 50, tol = 1e-10, random_state = 42)

def test_validate_parameters_kmeans_iter_value():
    with pytest.raises(ValueError):
        _validate_parameters_k_means(n_clusters = 3, max_iterations = -1, tol = 1e-10, random_state = 42)

def test_validate_parameters_kmeans_tol_value():
    with pytest.raises(ValueError):
        _validate_parameters_k_means(n_clusters = 3, max_iterations = 50, tol = -1.0, random_state = 42)

def test_validate_parameters_dbscan_basic():
    _validate_parameters_dbscan(0.1, 2)

def test_validate_parameters_dbscan_type_epsilon():
    with pytest.raises(TypeError):
        _validate_parameters_dbscan(epsilon = '0.1', core_point_min = 2)

def test_validate_parameters_dbscan_type_core():
    with pytest.raises(TypeError):
        _validate_parameters_dbscan(epsilon = 0.1, core_point_min = '2')

def test_validate_parameters_dbscan_value_epsilon():
    with pytest.raises(ValueError):
        _validate_parameters_dbscan(epsilon = -0.1, core_point_min = 2)

def test_validate_parameters_dbscan_value_core():
    with pytest.raises(ValueError):
        _validate_parameters_dbscan(epsilon = 0.1, core_point_min = -2)

def test_k_means_init_basic():
    kmeans = k_means(3, 50)
    assert kmeans.n_clusters == 3
    assert kmeans.max_iter == 50
    assert kmeans.tol == 1e-6
    assert kmeans.random_state is None
    assert kmeans.cluster_labels is None
    assert kmeans.centroids_ is None
    assert kmeans.inertia_ is None
    assert kmeans.n_features_ is None

def test_k_means_init_mixed():
    kmeans = k_means(3, 50, 1e-10, 42)
    assert kmeans.n_clusters == 3
    assert kmeans.max_iter == 50
    assert kmeans.tol == 1e-10
    assert kmeans.random_state == 42
    assert kmeans.cluster_labels is None
    assert kmeans.centroids_ is None
    assert kmeans.inertia_ is None
    assert kmeans.n_features_ is None

def test_k_means_init_type_inputs():
    with pytest.raises(TypeError):
        k_means('3', 50, 1e-10, 42)
    with pytest.raises(TypeError):
        k_means(3, '50', 1e-10, 42)
    with pytest.raises(TypeError):
        k_means(3, 50, '1e-10', 42)
    with pytest.raises(TypeError):
        k_means(3, 50, 1e-10, '42')

def test_k_means_init_input_values():
    with pytest.raises(ValueError):
        k_means(-1, 50, 1e-10, 42)
    with pytest.raises(ValueError):
        k_means(1, -50, 1e-10, 42)
    with pytest.raises(ValueError):
        k_means(1, 50, -1.0, 42)

def test_initial_centroids_basic():
    test_array = np.random.rand(10, 3)
    kmeans = k_means(2, 50, 1e-10, 42)
    test_centroids = kmeans._initial_centroids(test_array)
    assert isinstance(test_centroids, np.ndarray)
    assert test_centroids.shape == (2, 3)
    for row in test_centroids:
        assert any(np.all(row == test_row) for test_row in test_array)

def test_initial_centroids_type_input():
    test_array = 'np.ndarray([[10, 10, 1, 0], [0, 1, 3, 2], [1, 10, 2, 5]])'
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(TypeError):
        kmeans._initial_centroids(test_array)

def test_initial_centroids_dimensions_1D():
    test_array = np.random.rand(10)
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(ValueError):
        kmeans._initial_centroids(test_array)

def test_initial_centroids_dimensions_3D():
    test_array = np.random.rand(10, 10, 10)
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(ValueError):
        kmeans._initial_centroids(test_array)

def test_initial_centroids_duplicates():
    test_array = np.random.rand(10, 3)
    kmeans = k_means(4, 50, 1e-10, 42)
    test_centroids = kmeans._initial_centroids(test_array)
    assert len(np.unique(test_centroids, axis = 0)) == 4

def test_initial_centroids_random_state():
    test_array = np.random.rand(10, 3)
    kmeans_1 = k_means(4, 50, 1e-10, 42)
    kmeans_2 = k_means(4, 50, 1e-10, 42)
    test_centroids_1 = kmeans_1._initial_centroids(test_array)
    test_centroids_2 = kmeans_2._initial_centroids(test_array)
    assert np.allclose(test_centroids_1, test_centroids_2)

def test_distance_calc_basic():
    test_array = np.array([[0, 0],[1, 1],[2, 2]])
    test_centroids = np.array([[0, 0],[2, 2]])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_distance = kmeans._distance_calc(test_array, test_centroids)
    expected_distance = np.array([[0, np.sqrt(8)], [np.sqrt(2), np.sqrt(2)], [np.sqrt(8), 0]])
    assert isinstance(test_distance, np.ndarray)
    assert test_distance.shape == (3, 2)
    assert np.allclose(test_distance, expected_distance)

def test_distance_calc_negatives():
    test_array = np.random.rand(10, 3)
    test_centroids = np.random.rand(2, 3)
    kmeans = k_means(2, 50, 1e-10, 42)
    test_distance = kmeans._distance_calc(test_array, test_centroids)
    assert np.all(test_distance >= 0)

def test_distance_calc_nan():
    test_array = np.random.rand(10, 3)
    test_centroids = np.random.rand(2, 3)
    kmeans = k_means(2, 50, 1e-10, 42)
    test_distance = kmeans._distance_calc(test_array, test_centroids)
    assert not np.isnan(test_distance).any()

def test_distance_calc_one_centroid():
    test_array = np.array([[0, 0]])
    test_centroids = np.array([[2, 2]])
    kmeans = k_means(1, 50, 1e-10, 42)
    test_distance = kmeans._distance_calc(test_array, test_centroids)
    expected_distance = np.array([[np.sqrt(8)]])
    assert isinstance(test_distance, np.ndarray)
    assert test_distance.shape == (1, 1)
    assert np.allclose(test_distance, expected_distance)

def test_clustering_basic():
    test_array = np.array([[0, 0],[1, 1],[2, 2]])
    test_centroids = np.array([[0, 0],[2, 2]])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_clusters = kmeans._clustering(test_array, test_centroids)
    assert isinstance(test_clusters, np.ndarray)
    assert test_clusters.shape == (3,)
    assert np.all((test_clusters >= 0) & (test_clusters < kmeans.n_clusters))
    assert test_clusters[0] == 0 and test_clusters[2] == 1

def test_clustering_centroid_type():
    test_array = np.array([[0, 0],[1, 1],[2, 2]])
    test_centroids = 'np.array([[0, 0],[2, 2]])'
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(TypeError):
        kmeans._clustering(test_array, test_centroids)

def test_clustering_centroid_dimension_1D():
    test_array = np.array([[0, 0],[1, 1],[2, 2]])
    test_centroids = np.array([0, 0])
    kmeans = k_means(1, 50, 1e-10, 42)
    with pytest.raises(ValueError):
        kmeans._clustering(test_array, test_centroids)

def test_clustering_centroid_dimension_3D():
    test_array = np.array([[0, 0],[1, 1],[2, 2]])
    test_centroids = np.array([[[0, 0], [2, 2]]])
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(ValueError):
        kmeans._clustering(test_array, test_centroids)

def test_clustering_one_sample():
    test_array = np.array([[0, 0]])
    test_centroids = np.array([[0, 0], [2, 2]])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_clusters = kmeans._clustering(test_array, test_centroids)
    assert isinstance(test_clusters, np.ndarray)
    assert test_clusters.shape == (1,)
    assert test_clusters[0] == 0
    
def test_updated_centroids_basic():
    test_array = np.array([[0, 0], [1, 1], [2, 2]])
    test_labels = np.array([0, 1, 1])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_updated = kmeans._updated_centroids(test_array, test_labels)
    assert isinstance(test_updated, np.ndarray)
    assert test_updated.shape == (2, 2)
    assert np.allclose(test_updated[0], [0, 0])
    assert np.allclose(test_updated[1], [[1.5, 1.5]])

def test_updated_centroids_nan():
    test_array = np.random.rand(5,3)
    test_labels = np.array([0, 0, 1, 1, 2])
    kmeans = k_means(3, 50, 1e-10, 42)
    test_updated = kmeans._updated_centroids(test_array, test_labels)
    assert isinstance(test_updated, np.ndarray)
    assert test_updated.shape == (3, 3)
    assert not np.isnan(test_updated).any()

def test_inertia_basic():
    test_array = np.array([[0, 0], [1, 0], [2, 2]])
    test_centroids = np.array([[0, 0], [2, 2]])
    test_labels = np.array([0, 0, 1])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_distance = kmeans._inertia(test_array, test_centroids, test_labels)
    assert isinstance(test_distance, float)
    assert np.isclose(test_distance, 1)

def test_inertia_zeros():
    test_array = np.array([[0, 0], [2, 2]])
    test_centroids = np.array([[0, 0], [2, 2]])
    test_labels = np.array([0, 1])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_distance = kmeans._inertia(test_array, test_centroids, test_labels)
    assert isinstance(test_distance, float)
    assert np.isclose(test_distance, 0)

def test_inertia_nonnegatives():
    test_array = np.random.rand(5,3)
    test_centroids = np.random.rand(2,3)
    test_labels = np.array([0,1,0,1,0])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_distance = kmeans._inertia(test_array, test_centroids, test_labels)
    assert isinstance(test_distance, float)
    assert test_distance >= 0

def test_inertia_type_string_numerals():
    test_array = np.array([[0, 0], [1, 0], [2, 2]])
    test_centroids = np.array([[0, 0], [2, 2]])
    test_labels = np.array(['0', '0', '1'])
    kmeans = k_means(2, 50, 1e-10, 42)
    test_distance = kmeans._inertia(test_array, test_centroids, test_labels)
    assert isinstance(test_distance, float)
    assert np.isclose(test_distance, 1)

def test_inertia_type_string_nonnumeric():
    test_array = np.array([[0, 0], [1, 0], [2, 2]])
    test_centroids = np.array([[0, 0], [2, 2]])
    test_labels = np.array(['A', 'A', 'B'])
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(TypeError):
        kmeans._inertia(test_array, test_centroids, test_labels)

def test_inertia_type_input_labels():
    test_array = np.array([[0, 0], [1, 0], [2, 2]])
    test_centroids = np.array([[0, 0], [2, 2]])
    test_labels = 'np.array([0, 0, 1])'
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(TypeError):
        kmeans._inertia(test_array, test_centroids, test_labels)

def test_inertia_dimension_labels():
    test_array = np.array([[0, 0], [1, 0], [2, 2]])
    test_centroids = np.array([[0, 0], [2, 2]])
    test_labels = np.array([[0, 0, 1]])
    kmeans = k_means(2, 50, 1e-10, 42)
    with pytest.raises(ValueError):
        kmeans._inertia(test_array, test_centroids, test_labels)

def test_kmeans_verify_fit_basic():
    test_array = np.array([[0, 0],[1, 1],[10, 10],[11, 11]])
    kmeans = k_means(2, 50, 1e-3, 42)
    kmeans.fit(test_array)
    kmeans = kmeans._verify_fit()
    assert kmeans is kmeans

def test_verify_fit_unfit():
    kmeans = k_means(2, 50, 1e-3, 42)
    with pytest.raises(RuntimeError):
        kmeans._verify_fit()

def test_dbscan_init_basic():
    dbs = dbscan(0.1, 2)
    assert dbs.epsilon == 0.1
    assert dbs.min_points == 2
    assert dbs.cluster_labels is None
    assert dbs.core_point_indices is None

def test_dbscan_init_type_inputs():
    with pytest.raises(TypeError):
        dbscan('0.1', 2)
    with pytest.raises(TypeError):
        dbscan(0.1, '2')

def test_dbscan_init_input_values():
    with pytest.raises(ValueError):
        dbscan(-0.1, 2)
    with pytest.raises(ValueError):
        dbscan(0.1, -2)

def test_dbscan_distance_calc_basic():
    test_array = np.array([[0, 0], [1, 1], [2, 2]])
    dbs = dbscan(0.1, 2)
    test_distance = dbs._distance_calc(test_array)
    expected_distance = np.array([[0, np.sqrt(2), np.sqrt(8)], [np.sqrt(2), 0, np.sqrt(2)], [np.sqrt(8), np.sqrt(2), 0]])
    assert isinstance(test_distance, np.ndarray)
    assert test_distance.shape == (3, 3)
    assert np.allclose(test_distance, expected_distance)

def test_dbscan_distance_calc_negatives():
    test_array = np.random.rand(10, 3)
    dbs = dbscan(0.1, 2)
    test_distance = dbs._distance_calc(test_array)
    assert isinstance(test_distance, np.ndarray)
    assert test_distance.shape == (10, 10)
    assert np.all(test_distance >= 0)

def test_dbscan_distance_calc_nan():
    test_array = np.random.rand(10, 3)
    dbs = dbscan(0.1, 2)
    test_distance = dbs._distance_calc(test_array)
    assert not np.isnan(test_distance).any()

def test_dbscan_distance_calc_one_point():
    test_array = np.array([[1, 1]])
    dbs = dbscan(0.1, 2)
    test_distance = dbs._distance_calc(test_array)
    assert test_distance.shape == (1, 1)
    assert np.allclose(test_distance, [0])

def test_find_neighbors_basic():
    test_distances = np.array([[0.0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])
    dbs = dbscan(1, 2)
    test_neighbors = dbs._find_neighbors(0, test_distances)
    assert isinstance(test_neighbors, list)
    assert test_neighbors == [0, 1]

def test_find_neighbors_self():
    test_distances = np.array([[0.0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])
    dbs = dbscan(0.25, 2)
    test_neighbors = dbs._find_neighbors(0, test_distances)
    assert isinstance(test_neighbors, list)
    assert test_neighbors == [0]

def test_find_neighbors_type_distances_string_numeric():
    test_distances = np.array([['0.0', '0.5', '2.0'], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])
    dbs = dbscan(1, 2)
    test_neighbors = dbs._find_neighbors(0, test_distances)
    assert isinstance(test_neighbors, list)
    assert test_neighbors == [0, 1]

def test_find_neighbors_type_distances_string_nonnumeric():
    test_distances = np.array([['zero', 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])
    dbs = dbscan(1, 2)
    with pytest.raises(TypeError):
        dbs._find_neighbors(0, test_distances)

def test_find_neighbors_type_distances():
    test_distances = 'np.array([[0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])'
    dbs = dbscan(1, 2)
    with pytest.raises(TypeError):
        dbs._find_neighbors(0, test_distances)

def test_find_neighbors_dimension_distances():
    test_distances = np.array([[[0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]]])
    dbs = dbscan(1, 2)
    with pytest.raises(ValueError):
        dbs._find_neighbors(0, test_distances)

def test_find_neighbors_type_point_index():
    test_distances = np.array([[0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])
    dbs = dbscan(1, 2)
    with pytest.raises(TypeError):
        dbs._find_neighbors('0', test_distances)

def test_find_neighbors_point_index_value():
    test_distances = np.array([[0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])
    dbs = dbscan(1, 2)
    with pytest.raises(ValueError):
        dbs._find_neighbors(-1, test_distances)

def test_find_neighbors_point_index_value():
    test_distances = np.array([[0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]])
    dbs = dbscan(1, 2)
    with pytest.raises(ValueError):
        dbs._find_neighbors(-1, test_distances)
    with pytest.raises(ValueError):
        dbs._find_neighbors(3, test_distances)

def test_expand_region_basic():
    test_cluster_labels = np.array([-1, -1])
    test_distances = np.array([[0, 0.5],[0.5, 0]])
    test_point_index = 0
    dbs = dbscan(1, 2)
    test_neighbor_list = dbs._find_neighbors(test_point_index, test_distances)
    dbs._expand_region(test_cluster_labels, test_point_index, test_neighbor_list, 2, test_distances)
    assert test_cluster_labels[0] == 2

def test_expand_region_labeling():
    test_cluster_labels = np.array([-1, -1])
    test_distances = np.array([[0, 0.5],[0.5, 0]])
    test_point_index = 0
    dbs = dbscan(1, 1)
    test_neighbor_list = dbs._find_neighbors(test_point_index, test_distances)
    dbs._expand_region(test_cluster_labels, test_point_index, test_neighbor_list, 2, test_distances)
    assert test_cluster_labels[1] == 2

def test_expand_region_neighbor_update():
    test_cluster_labels = np.array([-1, -1, -1])
    test_distances = np.array([[0, 0.2, 0.2], [0.2, 0, 0.2], [0.2, 0.2, 0]])
    test_point_index = 0
    dbs = dbscan(0.5, 2)
    test_neighbor_list = dbs._find_neighbors(test_point_index, test_distances)
    dbs._expand_region(test_cluster_labels, test_point_index, test_neighbor_list, 2, test_distances)
    assert set(test_neighbor_list) == {0, 1, 2}
    assert all(test_cluster_labels == 2)

def test_expand_region_neighbor_no_update():
    test_cluster_labels = np.array([-1, -1, -1])
    test_distances = np.array([[0, 0.2, 1.0], [0.2, 0, 1.0], [1.0, 1.0, 0]])
    test_point_index = 0
    dbs = dbscan(0.5, 2)
    test_neighbor_list = dbs._find_neighbors(test_point_index, test_distances)
    dbs._expand_region(test_cluster_labels, test_point_index, test_neighbor_list, 2, test_distances)
    assert set(test_neighbor_list) == {0, 1}
    assert np.allclose(test_cluster_labels, [2, 2, -1])
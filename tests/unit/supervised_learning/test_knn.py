
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from rice_ml.supervised_learning.knn import _validate_parameters, _validate_arrays, _distance_calculations, _neighbor_finding, _weighting_by_distance, _knn_foundation, knn_classification, knn_regressor

def assert_array_output(test_array, test_labels, result_array, result_labels):
    assert isinstance(result_array, np.ndarray)
    assert isinstance(result_labels, np.ndarray)
    assert result_array.shape == np.array(test_array).shape
    assert result_labels.shape == np.array(test_labels).shape
    assert np.array_equal(result_array, np.array(test_array))
    assert np.array_equal(result_labels, np.array(test_labels))

def assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k):
    assert isinstance(distances, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert distances.shape == (query_array.shape[0], k)
    assert indices.shape == (query_array.shape[0], k)
    assert np.allclose(distances, expected_distances)
    assert np.array_equal(indices, expected_indices)

def assert_probabilities(prob_matrix, n_queries, n_classes):
    assert prob_matrix.shape == (n_queries, n_classes)
    for row in prob_matrix:
        assert np.isfinite(row).all()
        assert abs(row.sum() - 1.0) < 1e-10

def array_initializations_classification(weight: str = 'uniform'):
    train_array = np.array([[0, 0], [1, 1], [1, 2]])
    train_labels = np.array([1, 2, 2])
    knn = knn_classification(k = 2, weight = weight)
    knn.fit(train_array, train_labels)
    return knn

def test_validate_parameters_basic():
    k = 3
    metric = 'euclidean'
    weight = 'uniform'
    _validate_parameters(k, metric, weight)

def test_validate_parameters_k_npint():
    k = np.int64(3)
    metric = 'euclidean'
    weight = 'uniform'
    _validate_parameters(k, metric, weight)

def test_validate_parameters_type_k_float():
    k = 3.0
    metric = 'euclidean'
    weight = 'uniform'
    with pytest.raises(TypeError):
        _validate_parameters(k, metric, weight)

def test_validate_parameters_type_k_other():
    k = '3'
    metric = 'euclidean'
    weight = 'uniform'
    with pytest.raises(TypeError):
        _validate_parameters(k, metric, weight)

def test_validate_parameters_k_zero():
    k = 0
    metric = 'euclidean'
    weight = 'uniform'
    with pytest.raises(ValueError):
        _validate_parameters(k, metric, weight)

def test_validate_parameters_k_neg():
    k = -3
    metric = 'euclidean'
    weight = 'uniform'
    with pytest.raises(ValueError):
        _validate_parameters(k, metric, weight)

def test_validate_parameters_type_metric():
    k = 3
    metric = ['euclidean']
    weight = 'uniform'
    with pytest.raises(ValueError):
        _validate_parameters(k, metric, weight)

def test_validate_parameters_metric_value():
    k = 3
    metric = 'cosine similarity'
    weight = 'uniform'
    with pytest.raises(ValueError):
        _validate_parameters(k, metric, weight)

def test_validate_parameters_type_weight():
    k = 3
    metric = 'euclidean'
    weight = ['uniform']
    with pytest.raises(ValueError):
        _validate_parameters(k, metric, weight)

def test_validate_parameters_weight_value():
    k = 3
    metric = 'euclidean'
    weight = 'negative'
    with pytest.raises(ValueError):
        _validate_parameters(k, metric, weight)

def test_validate_arrays_basic_array():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = np.array([0, 1])
    result_array, result_labels = _validate_arrays(test_array, test_labels)
    assert_array_output(test_array, test_labels, result_array, result_labels)

def test_validate_arrays_basic_df():
    test_array = pd.DataFrame({
        'A': [1, 3],
        'B': [2, 4]
    })
    test_labels = pd.Series([0, 1])
    result_array, result_labels = _validate_arrays(test_array, test_labels)
    assert_array_output(test_array, test_labels, result_array, result_labels)

def test_validate_arrays_basic_lists():
    test_array = [[1, 2], [3, 4]]
    test_labels =[0, 1]
    result_array, result_labels = _validate_arrays(test_array, test_labels)
    assert_array_output(test_array, test_labels, result_array, result_labels)

def test_validate_arrays_basic_tuples():
    test_array = ([1, 2], [3, 4])
    test_labels =([0, 1])
    result_array, result_labels = _validate_arrays(test_array, test_labels)
    assert_array_output(test_array, test_labels, result_array, result_labels)

def test_validate_arrays_basic_mixed():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = [0, 1]
    result_array, result_labels = _validate_arrays(test_array, test_labels)
    assert_array_output(test_array, test_labels, result_array, result_labels)

def test_validate_arrays_basic_strings():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = np.array(['a', 'b'])
    result_array, result_labels = _validate_arrays(test_array, test_labels)
    assert_array_output(test_array, test_labels, result_array, result_labels)

def test_validate_arrays_basic_no_labels():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = None
    result_array = _validate_arrays(test_array, test_labels)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)
    assert np.array_equal(np.array(test_array), result_array)

def test_validate_arrays_basic_regression():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = np.array([0, 1])
    result_array, result_labels = _validate_arrays(test_array, test_labels, regression = True)
    assert_array_output(test_array, test_labels, result_array, result_labels)
    assert np.issubdtype(result_labels.dtype, np.number)

def test_validate_arrays_floats():
    test_array = np.array([[1.1, 2], [3, 4]])
    test_labels = np.array([0, 1.1])
    result_array, result_labels = _validate_arrays(test_array, test_labels, regression = True)
    assert_array_output(test_array, test_labels, result_array, result_labels)
    assert np.issubdtype(result_labels.dtype, np.number)

def test_validate_arrays_strings():
    test_array = np.array([['1', 2], [3, 4]])
    test_labels = np.array(['0', '1'])
    result_array, result_labels = _validate_arrays(test_array, test_labels, regression = True)
    assert isinstance(result_array, np.ndarray)
    assert isinstance(result_labels, np.ndarray)
    assert result_array.shape == (2, 2)
    assert result_labels.shape == (2,)
    assert np.array_equal(np.array(test_array).astype(float), result_array)
    assert np.array_equal(np.array(test_labels).astype(float), result_labels)
    assert np.issubdtype(result_array.dtype, np.number)
    assert np.issubdtype(result_labels.dtype, np.number)

def test_validate_arrays_empty():
    test_array = np.array([[]])
    test_labels = np.array([])
    with pytest.raises(ValueError):
        _validate_arrays(test_array, test_labels)

def test_validate_arrays_nan():
    test_array = np.array([[1, np.nan], [3, 4]])
    test_labels = np.array([0, 1])
    with pytest.raises(ValueError):
        _validate_arrays(test_array, test_labels)

def test_validate_arrays_dimensions():
    test_array = np.array([[1, 2], [3, 4]])
    test_array_1D = np.array([1, 2])
    test_labels = np.array([0, 1])
    test_labels_2D = np.array([[0, 1]])
    with pytest.raises(ValueError):
        _validate_arrays(test_array_1D, test_labels)
    with pytest.raises(ValueError):
        _validate_arrays(test_array, test_labels_2D)

def test_validate_arrays_matching_shape():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        _validate_arrays(test_array, test_labels)

def test_validate_arrays_type_input_array_data():
    test_array = np.array([['a', 2], [3, 4]])
    test_labels = np.array([0, 1, 0])
    with pytest.raises(TypeError):
        _validate_arrays(test_array, test_labels)

def test_validate_arrays_type_input_array():
    test_array = 'test_array'
    test_labels = np.array([0, 1, 0])
    with pytest.raises(TypeError):
        _validate_arrays(test_array, test_labels)

def test_validate_arrays_type_input_vector():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = 'test_labels'
    with pytest.raises(TypeError):
        _validate_arrays(test_array, test_labels)

def test_validate_arrays_type_regression():
    test_array = np.array([[1, 2], [3, 4]])
    test_labels = np.array([0, 1, 0])
    with pytest.raises(TypeError):
        _validate_arrays(test_array, test_labels, regression = 'True')

def test_distance_calculations_basic_euclidean():
    train_array = np.array([[1, 1], [0, 0]])
    test_array = np.array([[1, 0]])
    expected = np.array([[1.0, 1.0]])
    result_array = _distance_calculations(train_array, test_array, metric = 'euclidean')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 2)
    assert np.allclose(result_array, expected)

def test_distance_calculations_basic_manhattan():
    train_array = np.array([[1, 1], [0, 0]])
    test_array = np.array([[1, 0]])
    expected = np.array([[1.0, 1.0]])
    result_array = _distance_calculations(train_array, test_array, metric = 'manhattan')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 2)
    assert np.allclose(result_array, expected)

def test_distance_calculations_basic_minkowski():
    train_array = np.array([[1, 1], [0, 0]])
    test_array = np.array([[1, 0]])
    expected = np.array([[1.0, 1.0]])
    result_array = _distance_calculations(train_array, test_array, metric = 'minkowski', p = 3)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 2)
    assert np.allclose(result_array, expected)

def test_distance_calculations_basic_floats():
    train_array = np.array([[1.0, 1.0], [0.0, 0.0]])
    test_array = np.array([[1.0, 0.0]])
    expected = np.array([[1.0, 1.0]])
    result_array = _distance_calculations(train_array, test_array, metric = 'euclidean')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 2)
    assert np.allclose(result_array, expected)

def test_distance_calculations_basic_strings():
    train_array = np.array([['1.0', 1.0], [0.0, 0.0]])
    test_array = np.array([[1.0, '0.0']])
    expected = np.array([[1.0, 1.0]])
    result_array = _distance_calculations(train_array, test_array, metric = 'euclidean')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 2)
    assert np.allclose(result_array, expected)

def test_distance_calculations_dimensions():
    train_array = np.array([[1, 1], [0, 0]])
    test_array_1D = np.array([1, 0])
    with pytest.raises(ValueError):
        _distance_calculations(train_array, test_array_1D, metric = 'euclidean')

def test_distance_calculations_dif_size():
    train_array = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    test_array = np.array([[1.0, 0.0], [0.0, 0.0]])
    expected = np.array([[1.0, 1.0, 1.0], [np.sqrt(2.0), 0.0, np.sqrt(2.0)]])
    result_array = _distance_calculations(train_array, test_array, metric = 'euclidean')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 3)
    assert np.allclose(result_array, expected)

def test_distance_calculations_data_type_input():
    train_array = np.array([[1.0, 'a'], [0.0, 0.0]])
    test_array = np.array([[1.0, 'b']])
    with pytest.raises(TypeError):
        _distance_calculations(train_array, test_array, metric = 'euclidean')

def test_distance_calculations_type_input():
    train_array = 'np.array([[1.0, 1.0], [0.0, 0.0]])'
    test_array = np.array([[1.0, 0.0]])
    with pytest.raises(TypeError):
        _distance_calculations(train_array, test_array, metric = 'euclidean')

def test_distance_calculations_type_metric():
    train_array = np.array([[1, 1], [0, 0]])
    test_array = np.array([[1, 0]])
    with pytest.raises(ValueError):
        _distance_calculations(train_array, test_array, metric = 'cosine similarity')

def test_distance_calculations_type_p():
    train_array = np.array([[1, 1], [0, 0]])
    test_array = np.array([[1, 0]])
    with pytest.raises(TypeError):
        _distance_calculations(train_array, test_array, metric = 'minkowski', p = '3')

def test_distance_calculations_p_value():
    train_array = np.array([[1, 1], [0, 0]])
    test_array = np.array([[1, 0]])
    with pytest.raises(ValueError):
        _distance_calculations(train_array, test_array, metric = 'minkowski', p = 0)

def test_distance_calculations_empty_training():
    train_array = np.array([[]])
    test_array = np.array([[1, 0]])
    with pytest.raises(ValueError):
        _distance_calculations(train_array, test_array, metric = 'euclidean')

def test_distance_calculations_empty_testing():
    train_array =np.array([[1, 1], [0, 0]])
    test_array = np.array([[]])
    with pytest.raises(ValueError):
        _distance_calculations(train_array, test_array, metric = 'euclidean')

def test_distance_calculations_one_element():
    train_array = np.array([[1]])
    test_array = np.array([[1]])
    expected = np.array([[0]])
    result_array = _distance_calculations(train_array, test_array, metric = 'euclidean')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 1)
    assert np.allclose(result_array, expected)

def test_distance_calculations_nan():
    train_array = np.array([[1, np.nan], [0, 0]])
    test_array = np.array([[1, 0]])
    with pytest.raises(ValueError):
        _distance_calculations(train_array, test_array, metric = 'euclidean')

def test_neighbor_finding_basic_euclidean():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2
    distances, indices = _neighbor_finding(train_array, query_array, k, metric = 'euclidean')
    expected_distances = np.array([[1, 1]])
    expected_indices = np.array([[0, 1]])
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_neighbor_finding_basic_manhattan():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2
    distances, indices = _neighbor_finding(train_array, query_array, k, metric = 'manhattan')
    expected_distances = np.array([[1, 1]])
    expected_indices = np.array([[0, 1]])
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_neighbor_finding_basic_minkowski():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2
    distances, indices = _neighbor_finding(train_array, query_array, k, metric = 'minkowski', p = 3)
    expected_distances = np.array([[1, 1]])
    expected_indices = np.array([[0, 1]])
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_neighbor_finding_higher_dim():
    train_array = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
    query_array = np.array([[1, 0, 0], [1, 0, 0]])
    k = 2
    distances, indices = _neighbor_finding(train_array, query_array, k, metric = 'euclidean')
    expected_distances = np.array([[1, 1], [1, 1]])
    expected_indices = np.array([[0, 1], [0, 1]])
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_neighbor_finding_ties():
    train_array = np.array([[1, 0], [-1, -1], [-1, -1]])
    query_array = np.array([[1, 0]])
    k = 2
    distances, indices = _neighbor_finding(train_array, query_array, k, metric = 'euclidean')
    expected_distances = np.array([[0, np.sqrt(5)]])
    expected_indices = np.array([[0, 1]])
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_neighbor_finding_all_ties():
    train_array = np.array([[1, 1], [1, 1], [1, 1]])
    query_array = np.array([[1, 0]])
    k = 2
    distances, indices = _neighbor_finding(train_array, query_array, k, metric = 'euclidean')
    expected_distances = np.array([[1, 1]])
    expected_indices = np.array([[0, 1]])
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_neighbor_finding_dif_k():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 1
    distances, indices = _neighbor_finding(train_array, query_array, k, metric = 'euclidean')
    expected_distances = np.array([[1]])
    expected_indices = np.array([[0]])
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_neighbor_finding_empty():
    train_array = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
    query_array = np.array([[]])
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k = 2, metric = 'euclidean')

def test_neighbor_finding_nan():
    train_array = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
    query_array = np.array([[1, np.nan]])
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k = 2, metric = 'euclidean')

def test_neighbor_finding_query_dimensions():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([1, 0])
    k = 2
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_training_dimensions():
    train_array = np.array([[[0, 0], [1, 1], [2, 2]]])
    query_array = np.array([[1, 0]])
    k = 2
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_matching_shape():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0, 0]])
    k = 2
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_k_max():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 4
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_type_k_float():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2.0
    with pytest.raises(TypeError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_type_k_other():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = '2'
    with pytest.raises(TypeError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_type_array_input():
    train_array = 'np.array([[0, 0], [1, 1], [2, 2]])'
    query_array = np.array([[1, 0]])
    k = 2
    with pytest.raises(AttributeError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_data_type_input():
    train_array = np.array([[0, 'a'], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2
    with pytest.raises(TypeError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_type_query_input():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = 'np.array([[1, 0]])'
    k = 2
    with pytest.raises(TypeError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_data_query_input():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 'a']])
    k = 2
    with pytest.raises(TypeError):
        _neighbor_finding(train_array, query_array, k, metric = 'euclidean')

def test_neighbor_finding_type_metric():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k, metric = 'cosine similarity')

def test_neighbor_finding_type_p():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2
    with pytest.raises(TypeError):
        _neighbor_finding(train_array, query_array, k, metric = 'minkowski', p = '3')

def test_neighbor_finding_p_value():
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    query_array = np.array([[1, 0]])
    k = 2
    with pytest.raises(ValueError):
        _neighbor_finding(train_array, query_array, k, metric = 'minkowski', p = 0)

def test_weighting_by_distance_uniform():
    test_array = np.array([[1, 1], [2, 1]])
    result_array = _weighting_by_distance(test_array, 'uniform')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == test_array.shape
    assert np.allclose(result_array, 1.0)

def test_weighting_by_distance_distance():
    test_array = np.array([[1, 1], [2, 4]])
    result_array = _weighting_by_distance(test_array, 'distance')
    expected_array = np.array([[1, 1], [0.5, 0.25]])
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == test_array.shape
    assert np.allclose(result_array, expected_array)

def test_weighting_by_distance_zero_distance():
    test_array = np.array([[0, 1], [2, 0]])
    result_array = _weighting_by_distance(test_array, 'distance')
    expected_array = np.array([[1, 0], [0, 1]])
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == test_array.shape
    assert np.allclose(result_array, expected_array)

def test_weighting_by_distance_input_floats():
    test_array = np.array([[1.0, 1.0], [2.0, 4.0]])
    result_array = _weighting_by_distance(test_array, 'distance')
    expected_array = np.array([[1, 1], [0.5, 0.25]])
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == test_array.shape
    assert np.allclose(result_array, expected_array)

def test_weighting_by_distance_small_distance():
    eps = 1e-10
    test_array = np.array([[eps, (eps * 2)]])
    result_array = _weighting_by_distance(test_array, 'distance')
    expected_array = np.array([[1/eps, 1/(eps * 2)]])
    assert np.allclose(result_array, expected_array)

def test_weighting_by_distance_data_type_input():
    test_array = np.array([['a', 1], [2, 4]])
    with pytest.raises(TypeError):
        _weighting_by_distance(test_array, 'distance')
        
def test_weighting_by_distance_nan():
    test_array = np.array([[np.nan, 1], [2, 4]])
    with pytest.raises(ValueError):
        _weighting_by_distance(test_array, 'distance')

def test_weighting_by_distance_dimensions():
    test_array = np.array([[[1, 1], [2, 4]]])
    with pytest.raises(ValueError):
        _weighting_by_distance(test_array, 'distance')

def test_weighting_by_distance_type_inputs():
    test_array = 'np.array([[1, 1], [2, 4]])'
    with pytest.raises(TypeError):
        _weighting_by_distance(test_array, 'distance')

def test_knn_foundation_init_basic():
    knn = _knn_foundation(k = 3)
    assert knn.n_neighbors == 3
    assert knn.metric == 'euclidean'
    assert knn.weight == 'uniform'
    assert knn.p == 3
    assert knn._training is None
    assert knn._labels is None

def test_knn_foundation_init_basic_mixed():
    knn = _knn_foundation(k = 5, metric = 'minkowski', weight = 'distance', p = 5)
    assert knn.n_neighbors == 5
    assert knn.metric == 'minkowski'
    assert knn.weight == 'distance'
    assert knn.p == 5
    assert knn._training is None
    assert knn._labels is None

def test_knn_foundation_init_type_inputs():
    with pytest.raises(TypeError):
        _knn_foundation(k = '0')
    with pytest.raises(ValueError):
        _knn_foundation(k = 3, metric = 0.0)
    with pytest.raises(ValueError):
        _knn_foundation(k = 3, weight = 1)

def test_knn_foundation_init_input_values():
    with pytest.raises(ValueError):
        _knn_foundation(k = -2)
    with pytest.raises(ValueError):
        _knn_foundation(k = 3, metric = 'cosine distance')
    with pytest.raises(ValueError):
        _knn_foundation(k = 3, weight = 'weighted')

def test_verify_fit_basic():
    knn = _knn_foundation(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    train_array_fit, train_label_fit = knn._verify_fit()
    assert np.array_equal(train_array_fit, train_array)
    assert np.array_equal(train_label_fit, train_label)

def test_verify_fit_unfit():
    knn = _knn_foundation(k = 3)
    with pytest.raises(RuntimeError):
        knn._verify_fit()

def test_knn_implement_basic_euclidean():
    k = 2
    knn = _knn_foundation(k = k, metric = 'euclidean')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([[1, 0]])
    expected_distances = np.array([[1, 1]])
    expected_indices = np.array([[0, 1]])
    distances, indices = knn.knn_implement(query_array)
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_knn_implement_basic_manhattan():
    k = 2
    knn = _knn_foundation(k = k, metric = 'manhattan')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([[1, 0]])
    expected_distances = np.array([[1, 1]])
    expected_indices = np.array([[0, 1]])
    distances, indices = knn.knn_implement(query_array)
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_knn_implement_basic_minkowski():
    k = 2
    knn = _knn_foundation(k = k, metric = 'minkowski', p = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([[1, 0]])
    expected_distances = np.array([[1, 1]])
    expected_indices = np.array([[0, 1]])
    distances, indices = knn.knn_implement(query_array)
    assert_neighbor_output(distances, indices, query_array, expected_distances, expected_indices, k)

def test_knn_implement_dimensions():
    k = 2
    knn = _knn_foundation(k = k, metric = 'euclidean')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        knn.knn_implement(query_array)
        
def test_knn_implement_query_shape():
    k = 2
    knn = _knn_foundation(k = k, metric = 'euclidean')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([[1, 0, 0]])
    with pytest.raises(ValueError):
        knn.knn_implement(query_array)

def test_knn_implement_unfit():
    knn = _knn_foundation(k = 2, metric = 'euclidean')
    query_array = np.array([[1, 0, 0]])
    with pytest.raises(RuntimeError):
        knn.knn_implement(query_array)

def test_knn_implement_type_query():
    k = 2
    knn = _knn_foundation(k = k, metric = 'euclidean')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = 'np.array([[1, 0]])'
    with pytest.raises(TypeError):
        knn.knn_implement(query_array)

def test_knn_implement_nan():
    k = 2
    knn = _knn_foundation(k = k, metric = 'euclidean')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([[np.nan, 0]])
    with pytest.raises(ValueError):
        knn.knn_implement(query_array)

def test_knn_implement_empty():
    k = 2
    knn = _knn_foundation(k = k, metric = 'euclidean')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([[]])
    with pytest.raises(ValueError):
        knn.knn_implement(query_array)

def test_knn_implement_data_type_query():
    k = 2
    knn = _knn_foundation(k = k, metric = 'euclidean')
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    query_array = np.array([['a', 0]])
    with pytest.raises(TypeError):
        knn.knn_implement(query_array)

def test_knn_implement_duplicates():
    train_array = np.array([[0, 0], [0, 0], [1, 1]])
    train_label = np.array([0, 0, 1])
    knn = _knn_foundation(k = 2, metric = 'euclidean')
    knn.fit(train_array, train_label, regression=False)
    query_array = np.array([[0, 0]])
    distances, indices = knn.knn_implement(query_array)
    assert distances.shape == (1, 2)
    assert indices.shape == (1, 2)
    assert set(indices[0]) <= {0, 1}

def test_knn_classification_init_basic():
    knn = knn_classification(k = 3)
    assert knn.n_neighbors == 3
    assert knn.metric == 'euclidean'
    assert knn.weight == 'uniform'
    assert knn.p == 3
    assert knn._training is None
    assert knn._labels is None
    assert knn.classes_ is None

def test_knn_classification_probabilities_basic_uniform():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    probability_array = knn.probabilities(query_array)
    assert_probabilities(probability_array, n_queries = 2, n_classes = 2)
    assert np.array_equal(probability_array, np.array([[0.5, 0.5], [0.0, 1.0]]))

def test_knn_classification_probabilities_basic_distance():
    knn = array_initializations_classification('distance')
    query_array = np.array([[0.5, 1.0]])
    probability_array = knn.probabilities(query_array)
    expected_probability = np.array([[0.3090169944, 0.6909830056]])
    assert_probabilities(probability_array, n_queries = 1, n_classes = 2)
    assert np.allclose(probability_array, expected_probability)

def test_knn_classification_probabilities_strings():
    train_array = np.array([[0, 1], [1, 1], [1, 2]])
    train_labels = np.array(['a', 'b', 'b'])
    knn = knn_classification(k = 2, weight = 'uniform')
    knn.fit(train_array, train_labels)
    query_array = np.array([[0, 0], [2, 2]])
    probability_array = knn.probabilities(query_array)
    assert_probabilities(probability_array, n_queries = 2, n_classes = 2)

def test_knn_classification_probabilities_identical():
    train_array = np.array([[0, 0], [1, 1], [1, 1]])
    train_labels = np.array([1, 2, 2])
    knn = knn_classification(k = 2, weight = 'distance')
    knn.fit(train_array, train_labels)
    query_array = np.array([[1, 1]])
    probability_array = knn.probabilities(query_array)
    assert_probabilities(probability_array, n_queries = 1, n_classes = 2)
    assert np.array_equal(probability_array[0], np.array([0.0, 1.0]))

def test_knn_classification_probabilities_type_input():
    knn = array_initializations_classification('uniform')
    query_array = 'np.array([[1, 1]])'
    with pytest.raises(TypeError):
        knn.probabilities(query_array)

def test_knn_classification_probabilities_data_type_input():
    knn = array_initializations_classification('uniform')
    query_array = np.array([['a', 1]])
    with pytest.raises(TypeError):
        knn.probabilities(query_array)

def test_knn_classification_probabilities_empty():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[]])
    with pytest.raises(ValueError):
        knn.probabilities(query_array)

def test_knn_classification_probabilities_nan():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[1, np.nan]])
    with pytest.raises(ValueError):
        knn.probabilities(query_array)

def test_knn_classification_probabilities_dimensions():
    knn = array_initializations_classification('uniform')
    query_array = np.array([1, 1])
    with pytest.raises(ValueError):
        knn.probabilities(query_array)

def test_knn_classification_probabilities_zeros():
    train_array = np.array([[0, 0], [1, 1], [10, 10]])
    train_labels = np.array([0, 1, 2])
    knn = knn_classification(k = 2, weight = 'distance')
    knn.fit(train_array, train_labels)
    query_array = np.array([[0.2, 0.2]])
    normal_probability = knn.probabilities(query_array)
    expected_probability = np.array([[1/3, 1/3, 1/3]])
    with patch('rice_ml.supervised_learning.knn._weighting_by_distance', return_value = np.zeros((1, knn.n_neighbors))):
        fallback_probability = knn.probabilities(query_array)
    assert not np.allclose(normal_probability, fallback_probability)
    assert np.allclose(fallback_probability, expected_probability)

def test_knn_regressor_init_basic():
    knn = knn_regressor(k = 3)
    assert knn.n_neighbors == 3
    assert knn.metric == 'euclidean'
    assert knn.weight == 'uniform'
    assert knn.p == 3
    assert knn._training is None
    assert knn._labels is None

def array_initializations_regressor(weight: str = 'uniform'):
    train_array = np.array([[0, 0], [1, 1], [1, 2]])
    train_labels = np.array([1, 2, 2])
    knn = knn_regressor(k = 2, weight = weight)
    knn.fit(train_array, train_labels)
    return knn

def test_knn_regressor_prediction_basic_uniform():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    prediction = knn.prediction(query_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert prediction[0] == 1.5
    assert prediction[1] == 2

def test_knn_regressor_prediction_basic_distance():
    knn = array_initializations_regressor('distance')
    query_array = np.array([[0.0, 0.5]])
    prediction = knn.prediction(query_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert prediction[0] > 1
    assert prediction[0] < 2

def test_knn_regressor_prediction_unfit():
    knn = knn_regressor(k = 2)
    query_array = np.array([[1, 1]])
    with pytest.raises(RuntimeError):
        knn.prediction(query_array)

def test_knn_regressor_probabilities_zeros():
    train_array = np.array([[0, 0], [1, 1], [10, 10]])
    train_labels = np.array([0, 1, 2])
    knn = knn_regressor(k = 2, weight = 'distance')
    knn.fit(train_array, train_labels)
    query_array = np.array([[0.2, 0.2]])
    normal_prediction = knn.prediction(query_array)
    with patch('rice_ml.supervised_learning.knn._weighting_by_distance', return_value = np.zeros((1, knn.n_neighbors))):
        fallback_prediction = knn.prediction(query_array)
    assert isinstance(fallback_prediction, np.ndarray)
    assert not np.allclose(normal_prediction, fallback_prediction)
    assert fallback_prediction[0] == 0.5
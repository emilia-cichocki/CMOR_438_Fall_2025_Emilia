
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from rice_ml.supervised_learning.knn import _knn_foundation, knn_classification, knn_regressor

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

def test_knn_foundation_fit_basic_class():
    knn = _knn_foundation(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, False)
    assert np.array_equal(knn._training, train_array)
    assert np.array_equal(knn._labels, train_label)

def test_knn_foundation_fit_data_type_input_class():
    knn = _knn_foundation(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array(['a', 'b', 'c'])
    knn = knn.fit(train_array, train_label, False)
    assert np.array_equal(knn._training, train_array)
    assert np.array_equal(knn._labels, train_label)

def test_knn_foundation_fit_basic_reg():
    knn = _knn_foundation(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label, True)
    assert knn._labels.dtype == float
    assert np.array_equal(knn._training, train_array.astype(float, copy = False))
    assert np.array_equal(knn._labels, train_label)

def test_knn_foundation_fit_data_type_input_reg():
    knn = _knn_foundation(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array(['a', 'b', 'c'])
    with pytest.raises(TypeError): 
        knn.fit(train_array, train_label, True)

def test_knn_foundation_fit_dimensions():
    knn = _knn_foundation(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([[1, 2, 1]])
    with pytest.raises(ValueError):
        knn.fit(train_array, train_label, False)

def test_knn_foundation_fit_match_shape():
    knn = _knn_foundation(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2])
    with pytest.raises(ValueError):
        knn.fit(train_array, train_label, False)

def test_knn_foundation_fit_k_value():
    knn = _knn_foundation(k = 5)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([[1, 2, 1]])
    with pytest.raises(ValueError):
        knn.fit(train_array, train_label, False)

def test_knn_classification_fit_basic():
    knn = knn_classification(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label)
    assert np.array_equal(knn._training, train_array)
    assert np.array_equal(knn._labels, train_label)
    assert np.array_equal(knn.classes_, np.array([1, 2]))

def test_knn_classification_fit_strings():
    knn = knn_classification(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array(['a', 'b', 'a'])
    knn = knn.fit(train_array, train_label)
    assert np.array_equal(knn._training, train_array)
    assert np.array_equal(knn._labels, train_label)
    assert np.array_equal(knn.classes_, np.array(['a', 'b']))

def test_knn_classification_prediction_basic_uniform():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    prediction = knn.prediction(query_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert prediction[0] == 1
    assert prediction[1] == 2

def test_knn_classification_prediction_basic_distance():
    knn = array_initializations_classification('distance')
    query_array = np.array([[0.5, 1.0]])
    prediction = knn.prediction(query_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert prediction[0] == 2

def test_knn_classification_prediction_ties():
    knn = array_initializations_classification('distance')
    query_array = np.array([[1, 0]])
    prediction = knn.prediction(query_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert prediction[0] == 1

def test_knn_classification_prediction_strings():
    train_array = np.array([[0, 1], [1, 1], [1, 2]])
    train_labels = np.array(['a', 'b', 'b'])
    knn = knn_classification(k = 2, weight = 'uniform')
    knn.fit(train_array, train_labels)
    query_array = np.array([[0, 0], [2, 2]])
    prediction = knn.prediction(query_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert prediction[0] == 'a'
    assert prediction[1] == 'b'

def test_knn_classification_prediction_unsorted():
    train_array = np.array([[0, 1], [1, 1], [1, 2]])
    train_labels = np.array([2, 2, 1])
    knn = knn_classification(k = 2, weight = 'uniform')
    knn.fit(train_array, train_labels)
    query_array = np.array([[0, 0], [2, 2]])
    prediction = knn.prediction(query_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert prediction[0] == 2
    assert prediction[1] == 1

def test_knn_classification_prediction_unfit():
    knn = knn_classification(k = 2)
    query_array = np.array([[1, 1]])
    with pytest.raises(RuntimeError):
        knn.prediction(query_array)

def test_knn_classification_scoring_basic_uniform():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1, 1])
    score = knn.scoring(query_array, actual_labels)
    assert isinstance(score, float)
    assert score == 0.5

def test_knn_classification_scoring_basic_distance():
    knn = array_initializations_classification('distance')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1, 1])
    score = knn.scoring(query_array, actual_labels)
    assert isinstance(score, float)
    assert score == 0.5

def test_knn_classification_scoring_correct():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1, 2])
    score = knn.scoring(query_array, actual_labels)
    assert isinstance(score, float)
    assert score == 1.0

def test_knn_classification_scoring_incorrect():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([2, 1])
    score = knn.scoring(query_array, actual_labels)
    assert isinstance(score, float)
    assert score == 0.0

def test_knn_classification_scoring_strings():
    train_array = np.array([[0, 1], [1, 1], [1, 2]])
    train_labels = np.array(['a', 'b', 'b'])
    knn = knn_classification(k = 2, weight = 'uniform')
    knn.fit(train_array, train_labels)
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array(['a', 'a'])
    score = knn.scoring(query_array, actual_labels)
    assert isinstance(score, float)
    assert score == 0.5

def test_knn_classification_scoring_type_labels():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = 'np.array([1, 1])'
    with pytest.raises(TypeError):
        knn.scoring(query_array, actual_labels)

def test_knn_classification_scoring_match_shape():
    knn = array_initializations_classification('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        knn.scoring(query_array, actual_labels)

def test_knn_classification_scoring_unfit():
    knn = knn_classification(k = 2)
    with pytest.raises(RuntimeError):
        knn.scoring(np.array([[1, 1]]), np.array([1]))

def array_initializations_regressor(weight: str = 'uniform'):
    train_array = np.array([[0, 0], [1, 1], [1, 2]])
    train_labels = np.array([1, 2, 2])
    knn = knn_regressor(k = 2, weight = weight)
    knn.fit(train_array, train_labels)
    return knn

def test_knn_regressor_fit_basic():
    knn = knn_regressor(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array([1, 2, 1])
    knn = knn.fit(train_array, train_label)
    assert np.array_equal(knn._training, train_array)
    assert np.array_equal(knn._labels, train_label)

def test_knn_regressor_fit_strings():
    knn = knn_regressor(k = 3)
    train_array = np.array([[0, 0], [1, 1], [2, 2]])
    train_label = np.array(['a', 'b', 'a'])
    with pytest.raises(TypeError):
        knn.fit(train_array, train_label)

def test_knn_regressor_scoring_basic():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0.5, 0.5], [1, 1.5]])
    actual_labels = np.array([1.4, 2.1])
    score = knn.scoring(query_array, actual_labels)
    expected_score = 0.918367
    assert isinstance(score, float)
    assert np.isclose(score, expected_score)
    
def test_knn_regressor_scoring_zeros():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1.75, 2.25])
    score = knn.scoring(query_array, actual_labels)
    expected_score = 0
    assert isinstance(score, float)
    assert np.isclose(score, expected_score)

def test_knn_regressor_scoring_correct():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1.5, 2])
    score = knn.scoring(query_array, actual_labels)
    assert isinstance(score, float)
    assert score == 1.0

def test_knn_regressor_scoring_training_data():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [1, 1], [1, 2]])
    actual_labels = np.array([1.5, 2, 2])
    score = knn.scoring(query_array, actual_labels)
    assert isinstance(score, float)
    assert score == 1.0

def test_knn_regressor_scoring_uniform_actual():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [0, 1], [1, 2]])
    actual_labels = np.array([2, 2, 2])
    with pytest.raises(ValueError):
        knn.scoring(query_array, actual_labels)

def test_knn_regressor_scoring_type_labels():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = 'np.array([1, 1])'
    with pytest.raises(TypeError):
        knn.scoring(query_array, actual_labels)

def test_knn_regressor_scoring_match_shape():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1, 1, 2])
    with pytest.raises(ValueError):
        knn.scoring(query_array, actual_labels)

def test_knn_regressor_scoring_strings():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array(['a', 'a'])
    with pytest.raises(TypeError):
        knn.scoring(query_array, actual_labels)

def test_knn_regressor_scoring_nan():
    knn = array_initializations_regressor('uniform')
    query_array = np.array([[0, 0], [2, 2]])
    actual_labels = np.array([1, np.nan])
    with pytest.raises(ValueError):
        knn.scoring(query_array, actual_labels)

def test_knn_regressor_scoring_unfit():
    knn = knn_regressor(k = 2)
    with pytest.raises(RuntimeError):
        knn.scoring(np.array([[1, 1]]), np.array([1]))
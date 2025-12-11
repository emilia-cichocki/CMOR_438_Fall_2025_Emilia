
import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch
from rice_ml.supervised_learning.perceptron import _validate_parameters, _validate_arrays_perceptron, Perceptron, multilayer_Perceptron

# TODO: rename 'test_arrays' for lists/df, fix the formatting and spacing, add comments to indicate functions being tested


def test_validate_parameters_basic():
    learning_rate = 0.001
    epochs = 1000
    _validate_parameters(learning_rate, epochs)

def test_validate_parameters_unspecified_epochs():
    learning_rate = 0.001
    with pytest.raises(TypeError):
        _validate_parameters(learning_rate = learning_rate)

def test_validate_parameters_unspecified_learning_rate():
    epochs = 1000
    with pytest.raises(TypeError):
        _validate_parameters(epochs = epochs)

def test_validate_parameters_type_learning_rate_int():
    learning_rate = 1
    epochs = 1000
    _validate_parameters(learning_rate, epochs)

def test_validate_parameters_type_learning_rate():
    learning_rate = '1'
    epochs = 1000
    with pytest.raises(TypeError):
        _validate_parameters(learning_rate, epochs)

def test_validate_parameters_learning_rate_value():
    learning_rate = -0.001
    epochs = 1000
    with pytest.warns(UserWarning, match = "learning rate"):
        _validate_parameters(learning_rate, epochs)

def test_validate_parameters_type_epochs_float():
    learning_rate = 0.001
    epochs = 1000.1
    with pytest.raises(TypeError):
        _validate_parameters(learning_rate, epochs)

def test_validate_parameters_type_epochs():
    learning_rate = 0.001
    epochs = '1000'
    fit_intercept = True
    with pytest.raises(TypeError):
        _validate_parameters(learning_rate, epochs)

def test_validate_parameters_epochs_value():
    learning_rate = 0.001
    epochs = -1
    with pytest.raises(ValueError):
        _validate_parameters(learning_rate, epochs)


def test_validate_arrays_perceptron_basic_array():
    test_array = np.array([[1, 2], [3, 4]])
    result_array = _validate_arrays_perceptron(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)
    assert np.array_equal(test_array, result_array)

def test_validate_arrays_perceptron_basic_array_df():
    test_array = pd.DataFrame({
        'A': [1, 3],
        'B': [2, 4]})
    result_array = _validate_arrays_perceptron(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)
    
def test_validate_arrays_perceptron_basic_array_list():
    test_array = [[1, 2], [3, 4]]
    result_array = _validate_arrays_perceptron(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)

def test_validate_arrays_perceptron_basic_array_tuple():
    test_array = ([1, 2], [3, 4])
    result_array = _validate_arrays_perceptron(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)

def test_validate_arrays_perceptron_array_dimension():
    test_array = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(data_array = test_array)

def test_validate_arrays_perceptron_basic_vector():
    test_array = np.array([1, 2])
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(result_array, np.array([-1, 1]))

def test_validate_arrays_perceptron_basic_vector_df():
    test_array = pd.Series([1, 2])
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)

def test_validate_arrays_perceptron_basic_vector_list():
    test_array = [1, 2]
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)

def test_validate_arrays_perceptron_basic_vector_tuple():
    test_array = (1, 2)
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)

def test_validate_arrays_perceptron_vector_2D_hor():
    test_array = np.array([[1, 2]])
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(result_array, np.array([-1, 1]))

def test_validate_arrays_perceptron_vector_2D_ver():
    test_array = np.array([[1], [2]])
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(result_array, np.array([-1, 1]))

def test_validate_arrays_perceptron_vector_2D_dimensions():
    test_array = np.array([[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(target_vector = test_array)

def test_validate_arrays_perceptron_vector_dimension():
    test_array = np.array([[[1, 2]]])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(data_array = test_array)

def test_validate_arrays_perceptron_array_nan():
    test_array = np.array([[1, np.nan], [1, 2]])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(data_array = test_array)

def test_validate_arrays_perceptron_array_data_type_input():
    test_array = np.array([['a', 'a'], [1, 1]])
    with pytest.raises(TypeError):
        _validate_arrays_perceptron(data_array = test_array)

def test_validate_arrays_perceptron_array_type_input():
    test_array = 'np.array([[1, 2], [3, 4]])'
    with pytest.raises(TypeError):
        _validate_arrays_perceptron(data_array = test_array)

def test_validate_arrays_perceptron_vector_nan():
    test_array = np.array([1, np.nan])
    with pytest.raises(TypeError):
        _validate_arrays_perceptron(target_vector = test_array)

def test_validate_arrays_perceptron_vector_data_type_input():
    test_array = np.array([1, 'a'])
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(result_array, np.array([-1, 1]))

def test_validate_arrays_perceptron_vector_type_input():
    test_array = 'np.array([1, 2])'
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(target_vector = test_array)

def test_validate_arrays_perceptron_data_type_vector():
    test_array = np.array([[1, 'a']])
    result_array = _validate_arrays_perceptron(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(result_array, np.array([-1, 1]))

def test_validate_arrays_perceptron_combined():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([1, 2])
    result_array, result_vector = _validate_arrays_perceptron(data_array = test_array, target_vector = test_vector)
    assert isinstance(result_array, np.ndarray)
    assert isinstance(result_vector, np.ndarray)
    assert result_array.shape == (2, 2)
    assert result_vector.shape == (2,)

def test_validate_arrays_perceptron_combined_2D():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([[1, 2]])
    result_array, result_vector = _validate_arrays_perceptron(data_array = test_array, target_vector = test_vector)
    assert isinstance(result_array, np.ndarray)
    assert isinstance(result_vector, np.ndarray)
    assert result_array.shape == (2, 2)
    assert result_vector.shape == (2,)

def test_validate_arrays_perceptron_match_shapes():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([[1, 1, -1]])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(data_array = test_array, target_vector = test_vector)

def test_validate_arrays_perceptron_empty_array():
    test_array = np.array([[]])
    test_vector = np.array([-1, 1])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(data_array = test_array, target_vector = test_vector)

def test_validate_arrays_perceptron_empty_vector():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(data_array = test_array, target_vector = test_vector)

def test_validate_arrays_perceptron_nonbinary():
    test_array = np.array([[1, 2], [3, 4], [1, 4]])
    test_vector = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        _validate_arrays_perceptron(data_array = test_array, target_vector = test_vector)

def test_perceptron_init_basic():
    perceptron = Perceptron(1000, 0.01)
    assert perceptron.epochs == 1000
    assert perceptron.learning_rate == 0.01
    assert perceptron.coef_ is None
    assert perceptron.bias_ is None
    assert perceptron.class_mapping_ is None

def test_perceptron_init_type_inputs():
    with pytest.raises(TypeError):
        Perceptron('1000', 0.01)
    with pytest.raises(TypeError):
        Perceptron(1000, '0.01')

def test_perceptron_init_input_values():
    with pytest.warns(UserWarning, match = "learning rate"):
        Perceptron(epochs = 1000, learning_rate = -0.01)
    with pytest.raises(ValueError):
        Perceptron(epochs = -1000, learning_rate = 0.01)

def test_perceptron_fit_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    perceptron.fit(train_array, train_targets)
    assert isinstance(perceptron.coef_, np.ndarray)
    assert np.isscalar(perceptron.bias_)
    assert perceptron.coef_.shape == (1,)
    assert perceptron.coef_[0] > 0
    assert perceptron.bias_ < 0

def test_perceptron_fit_nonzero():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, -1, -1, -1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    perceptron.fit(train_array, train_targets)
    assert isinstance(perceptron.coef_, np.ndarray)
    assert np.isscalar(perceptron.bias_)
    assert perceptron.coef_.shape == (1,)
    assert perceptron.coef_[0] < 0
    assert perceptron.bias_ > 0

def test_perceptron_fit_mult_feat():
    train_array = np.array([
        [1, 1],
        [2, 0],
        [0, 3],
        [4, 1],
        [3, 2]
    ])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    assert isinstance(perceptron.coef_, np.ndarray)
    assert np.isscalar(perceptron.bias_)
    assert perceptron.coef_.shape == (2,)

def test_perceptron_random_state_reproducibility():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron1 = Perceptron(epochs = 10000, learning_rate = 0.01)
    perceptron1.fit(train_array, train_targets, random_state = 72, shuffle = True)
    perceptron2 = Perceptron(epochs = 10000, learning_rate = 0.01)
    perceptron2.fit(train_array, train_targets, random_state = 72, shuffle = True)
    assert np.allclose(perceptron1.coef_, perceptron2.coef_, rtol = 1e-3, atol = 1e-3)
    assert np.isclose(perceptron1.bias_, perceptron2.bias_, rtol = 1e-3, atol = 1e-3)

def test_perceptron_fit_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array(['a', 'a', 'b', 'b', 'b'])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    perceptron.fit(train_array, train_targets)
    assert isinstance(perceptron.coef_, np.ndarray)
    assert np.isscalar(perceptron.bias_)
    assert perceptron.coef_.shape == (1,)
    assert perceptron.coef_[0] > 0
    assert perceptron.bias_ < 0

def test_perceptron_fit_mixed():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 'b', 'b', 'b'])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    perceptron.fit(train_array, train_targets)
    assert isinstance(perceptron.coef_, np.ndarray)
    assert np.isscalar(perceptron.bias_)
    assert perceptron.coef_.shape == (1,)
    assert perceptron.coef_[0] > 0
    assert perceptron.bias_ < 0

def test_perceptron_fit_2D_hor():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[-1, -1, 1, 1, 1]])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    perceptron.fit(train_array, train_targets)
    assert isinstance(perceptron.coef_, np.ndarray)
    assert np.isscalar(perceptron.bias_)
    assert perceptron.coef_.shape == (1,)
    assert perceptron.coef_[0] > 0
    assert perceptron.bias_ < 0

def test_perceptron_fit_2D_ver():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    perceptron.fit(train_array, train_targets)
    assert isinstance(perceptron.coef_, np.ndarray)
    assert np.isscalar(perceptron.bias_)
    assert perceptron.coef_.shape == (1,)
    assert perceptron.coef_[0] > 0
    assert perceptron.bias_ < 0

def test_perceptron_fit_nonbinary():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 2])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_nan_array():
    train_array = np.array([[0], [np.nan], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_nan_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, np.nan, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_empty_array():
    train_array = np.array([[]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_empty_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_data_type_array():
    train_array = np.array([['a'], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_type_array():
    train_array = 'np.array([[0], [1], [2], [3], [4]])'
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_type_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = 'np.array([-1, -1, 1, 1, 1])'
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_match_shape():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_internal_shape():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_array_dimensions():
    train_array = np.array([0, 1, 2, 3, 4])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_vector_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[[-1, -1, 1, 1, 1]]])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)

def test_perceptron_fit_data_type_random_state():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        perceptron.fit(train_array, train_targets, random_state = '1')

def test_perceptron_fit_data_type_shuffle():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        perceptron.fit(train_array, train_targets, shuffle = 'True')

def test_perceptron_fit_data_type_constant():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([1, 1, 1, 1, 1])
    perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        perceptron.fit(train_array, train_targets)
    
def test_perceptron_verify_fit_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    perceptron._verify_fit()

def test_perceptron_verify_fit_unfit():
    perceptron = Perceptron(1000, 0.01)
    with pytest.raises(RuntimeError):
        perceptron._verify_fit()

def test_perceptron_prediction_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    classification = perceptron.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array([1, 1]))

def test_perceptron_prediction_neg():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, -1, -1, -1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    classification = perceptron.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array([-1, -1]))

def test_perceptron_prediction_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array(['a', 'a', 'b', 'b', 'b'])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[0.5], [3.5]])
    classification = perceptron.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array(['a', 'b']))

def test_perceptron_prediction_mult_feat():
    train_array = np.array([
                           [1, 1],
                           [2, 0],
                           [0, 3],
                           [4, 1],
                           [3, 2]
    ])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[0.5, 0.5], [3.5, 3.5]])
    classification = perceptron.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array([-1, 1]))

def test_perceptron_prediction_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([0, 1])
    with pytest.raises(ValueError):
        perceptron.prediction(test_array)
    
def test_perceptron_prediction_mismatch():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[0, 1]])
    with pytest.raises(ValueError):
        perceptron.prediction(test_array)

def test_perceptron_prediction_data_type_array():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([['a'], [1]])
    with pytest.raises(TypeError):
        perceptron.prediction(test_array)

def test_perceptron_prediction_data_type_unfit():
    perceptron = Perceptron(1000, 0.01)
    test_array = np.array([[0], [2]])
    with pytest.raises(RuntimeError):
        perceptron.prediction(test_array)

def test_perceptron_scoring_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([-1, 1])
    score = perceptron.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 0.5)

def test_perceptron_scoring_correct():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([1, 1])
    score = perceptron.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 1)

def test_perceptron_scoring_incorrect():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([-1, -1])
    score = perceptron.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 0)

def test_perceptron_scoring_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 'a', 'a', 'a'])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array(['a', 'a'])
    score = perceptron.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 1)

def test_perceptron_scoring_type_labels():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = 'np.array([1, 1])'
    with pytest.raises(TypeError):
        perceptron.scoring(test_array, actual_array)

def test_perceptron_scoring_lengths():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        perceptron.scoring(test_array, actual_array)

def test_perceptron_scoring_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([-1, -1, 1, 1, 1])
    perceptron = Perceptron(1000, 0.01)
    perceptron.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([[1, 1]])
    with pytest.raises(ValueError):
        perceptron.scoring(test_array, actual_array)

def test_perceptron_scoring_unfit():
    perceptron = Perceptron(1000, 0.01)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([[1, 1]])
    with pytest.raises(RuntimeError):
        perceptron.scoring(test_array, actual_array)


def test_mlp_init_basic():
    mlp = multilayer_Perceptron([2, 2, 1], 1000, 0.01)
    assert mlp.layers == [2, 2, 1]
    assert mlp.epochs == 1000
    assert mlp.learning_rate == 0.01
    assert mlp.coef_ is None
    assert mlp.bias_ is None

def test_mlp_init_type_inputs():
    with pytest.raises(TypeError):
        multilayer_Perceptron((2, 2, 1), 1000, 0.01)
    with pytest.raises(TypeError):
        multilayer_Perceptron([2.5, 2, 1], 1000, 0.01)
    with pytest.raises(TypeError):
        multilayer_Perceptron(['a', 2, 1], 1000, 0.01)
    with pytest.raises(TypeError):
        multilayer_Perceptron([2, 2, 1], '1000', 0.01)
    with pytest.raises(TypeError):
        multilayer_Perceptron([2, 2, 1], 1000, '0.01')

def test_mlp_init_input_values():
    with pytest.raises(ValueError):
        multilayer_Perceptron(layers = [2, 2, -1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        multilayer_Perceptron(layers = [2, 2, 0], epochs = 1000, learning_rate = 0.01)
    with pytest.warns(UserWarning, match = "learning rate"):
        multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = -0.01)
    with pytest.raises(ValueError):
        multilayer_Perceptron(layers = [2, 2, 1], epochs = -1000, learning_rate = 0.01)

def test_weight_initialization_basic():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_weights, test_bias = mlp._weight_initialization()
    assert isinstance(test_weights, list)
    assert isinstance(test_bias, list)
    assert len(test_weights) == 2
    for element in test_weights:
        assert isinstance(element, np.ndarray)
    assert test_weights[0].shape == (2, 2)
    assert test_weights[1].shape == (2, 1)
    assert len(test_bias) == 2
    for element in test_bias:
        assert isinstance(element, np.ndarray)
        assert np.all(element == 0)
    assert test_bias[0].shape == (2,)
    assert test_bias[1].shape == (1,)

def test_weight_initialization_random_state():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_weights_1, test_bias_1 = mlp._weight_initialization(random_state = 42)
    test_weights_2, test_bias_2 = mlp._weight_initialization(random_state = 42)
    assert len(test_weights_1) == len(test_weights_2)
    assert len(test_bias_1) == len(test_bias_2)
    for i in range(len(test_weights_1)):
        assert np.allclose(test_weights_1[i], test_weights_2[i])

def test_weight_initialization_type_input():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp._weight_initialization(random_state = '42')

def test_forward_layer_basic():

#     def _forward_layer(self, training_array: np.ndarray, weights: list, bias: list) -> Tuple[list, list]:
        
#         train_array = _validate_arrays_perceptron(training_array)

#         z = []
#         a = [train_array]
#         for i in range(len(weights)):
#             z_layer = np.matmul(a[-1], weights[i]) + bias[i]
#             z.append(z_layer)
#             a_layer = _sigmoid(z_layer)
#             a.append(a_layer)
            
#         return z, a

#     def _back_propagation(self, z: list, a: list, weights: list, training_targets: np.ndarray) -> Tuple[list, list]:

#         train_targets = _validate_arrays_perceptron(training_targets)
        
#         L = len(self.layers) - 1
#         learning_rate = self.learning_rate
#         delta = dict()
#         delta[L] = (a[-1] - train_targets) * derivative_sigmoid(z[-1])
#         d_weights = []
#         d_bias = []

#         for i in range(L - 1, 0, -1):
#             delta[i] = (np.matmul(delta[i + 1], weights[i].T)) * derivative_sigmoid(z[i - 1])
    
#         for j in range(1, L + 1):
#             d_weights_layer = learning_rate * np.matmul(a[j - 1].T, delta[j])
#             d_bias_layer = learning_rate * np.mean(delta[j], axis = 0)
#             d_weights.append(d_weights_layer)
#             d_bias.append(d_bias_layer)

#         return d_weights, d_bias

#     def _weight_update(self, weights: list, bias: list, d_weights: list, d_bias: list) -> Tuple[list, list]:
        
#         for i in range(len(weights)):
#                 weights[i] -= d_weights[i]
#                 bias[i] -= d_bias[i]

#         return weights, bias

#     def fit(self, training_array: np.ndarray, training_targets: np.ndarray, random_state: Optional[int] = None) -> 'multilayer_Perceptron':
        
#         train_array = _validate_arrays_perceptron(training_array)
#         train_targets = _validate_arrays_perceptron(training_targets)

#         weights, bias = self._weight_initialization(random_state = random_state)

#         for _ in range(self.epochs):
#             z, a = self._forward_layer(train_array, weights, bias)
#             d_weights, d_bias = self._back_propagation(z, a, weights, train_targets)
#             weights, bias = self._weight_update(weights, bias, d_weights, d_bias)

#         self.coef_ = weights
#         self.bias_ = bias

#         return self
    
#     def _verify_fit(self) -> 'multilayer_Perceptron':
#         if self.coef_ is None or self.bias_ is None:
#             raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

#         return self
    
#     def predict(self, testing_array: np.ndarray):
        
#         # TODO: doctrings/comments

#         self._verify_fit()

#         test_array = _validate_arrays_perceptron(testing_array)

#         if test_array.shape[1] != self.coef_[0].shape[0]:
#             raise ValueError('Test array must have the same number of input features as coefficients')

#         _, a = self._forward_layer(testing_array, self.coef_, self.bias_)

#         prediction = a[-1]

#         predicted_labels = np.where(prediction > 0.5, 1, 0)

#         return prediction, predicted_labels

# def test_perceptron_fit_basic():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     perceptron.fit(train_array, train_targets)
#     assert isinstance(perceptron.coef_, np.ndarray)
#     assert np.isscalar(perceptron.bias_)
#     assert perceptron.coef_.shape == (1,)
#     assert perceptron.coef_[0] > 0
#     assert perceptron.bias_ < 0

# def test_perceptron_fit_nonzero():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([0, 0, -1, -1, -1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     perceptron.fit(train_array, train_targets)
#     assert isinstance(perceptron.coef_, np.ndarray)
#     assert np.isscalar(perceptron.bias_)
#     assert perceptron.coef_.shape == (1,)
#     assert perceptron.coef_[0] < 0
#     assert perceptron.bias_ > 0

# def test_perceptron_fit_mult_feat():
#     train_array = np.array([
#         [1, 1],
#         [2, 0],
#         [0, 3],
#         [4, 1],
#         [3, 2]
#     ])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     assert isinstance(perceptron.coef_, np.ndarray)
#     assert np.isscalar(perceptron.bias_)
#     assert perceptron.coef_.shape == (2,)

# def test_perceptron_random_state_reproducibility():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron1 = Perceptron(epochs = 10000, learning_rate = 0.01)
#     perceptron1.fit(train_array, train_targets, random_state = 72, shuffle = True)
#     perceptron2 = Perceptron(epochs = 10000, learning_rate = 0.01)
#     perceptron2.fit(train_array, train_targets, random_state = 72, shuffle = True)
#     assert np.allclose(perceptron1.coef_, perceptron2.coef_, rtol = 1e-3, atol = 1e-3)
#     assert np.isclose(perceptron1.bias_, perceptron2.bias_, rtol = 1e-3, atol = 1e-3)

# def test_perceptron_fit_strings():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array(['a', 'a', 'b', 'b', 'b'])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     perceptron.fit(train_array, train_targets)
#     assert isinstance(perceptron.coef_, np.ndarray)
#     assert np.isscalar(perceptron.bias_)
#     assert perceptron.coef_.shape == (1,)
#     assert perceptron.coef_[0] > 0
#     assert perceptron.bias_ < 0

# def test_perceptron_fit_mixed():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 'b', 'b', 'b'])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     perceptron.fit(train_array, train_targets)
#     assert isinstance(perceptron.coef_, np.ndarray)
#     assert np.isscalar(perceptron.bias_)
#     assert perceptron.coef_.shape == (1,)
#     assert perceptron.coef_[0] > 0
#     assert perceptron.bias_ < 0

# def test_perceptron_fit_2D_hor():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([[-1, -1, 1, 1, 1]])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     perceptron.fit(train_array, train_targets)
#     assert isinstance(perceptron.coef_, np.ndarray)
#     assert np.isscalar(perceptron.bias_)
#     assert perceptron.coef_.shape == (1,)
#     assert perceptron.coef_[0] > 0
#     assert perceptron.bias_ < 0

# def test_perceptron_fit_2D_ver():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([[-1], [-1], [1], [1], [1]])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     perceptron.fit(train_array, train_targets)
#     assert isinstance(perceptron.coef_, np.ndarray)
#     assert np.isscalar(perceptron.bias_)
#     assert perceptron.coef_.shape == (1,)
#     assert perceptron.coef_[0] > 0
#     assert perceptron.bias_ < 0

# def test_perceptron_fit_nonbinary():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 2])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_nan_array():
#     train_array = np.array([[0], [np.nan], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_nan_vector():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([0, np.nan, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_empty_array():
#     train_array = np.array([[]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_empty_vector():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_data_type_array():
#     train_array = np.array([['a'], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(TypeError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_type_array():
#     train_array = 'np.array([[0], [1], [2], [3], [4]])'
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(TypeError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_type_vector():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = 'np.array([-1, -1, 1, 1, 1])'
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_match_shape():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_internal_shape():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_array_dimensions():
#     train_array = np.array([0, 1, 2, 3, 4])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_vector_dimensions():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([[[-1, -1, 1, 1, 1]]])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)

# def test_perceptron_fit_data_type_random_state():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(TypeError):
#         perceptron.fit(train_array, train_targets, random_state = '1')

# def test_perceptron_fit_data_type_shuffle():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(TypeError):
#         perceptron.fit(train_array, train_targets, shuffle = 'True')

# def test_perceptron_fit_data_type_constant():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([1, 1, 1, 1, 1])
#     perceptron = Perceptron(epochs = 1000, learning_rate = 0.01)
#     with pytest.raises(ValueError):
#         perceptron.fit(train_array, train_targets)
    
# def test_perceptron_verify_fit_basic():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     perceptron._verify_fit()

# def test_perceptron_verify_fit_unfit():
#     perceptron = Perceptron(1000, 0.01)
#     with pytest.raises(RuntimeError):
#         perceptron._verify_fit()

# def test_perceptron_prediction_basic():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     classification = perceptron.prediction(test_array)
#     assert isinstance(classification, np.ndarray) 
#     assert classification.shape == (2,)
#     assert np.array_equal(classification, np.array([1, 1]))

# def test_perceptron_prediction_neg():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([0, 0, -1, -1, -1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     classification = perceptron.prediction(test_array)
#     assert isinstance(classification, np.ndarray) 
#     assert classification.shape == (2,)
#     assert np.array_equal(classification, np.array([-1, -1]))

# def test_perceptron_prediction_strings():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array(['a', 'a', 'b', 'b', 'b'])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[0.5], [3.5]])
#     classification = perceptron.prediction(test_array)
#     assert isinstance(classification, np.ndarray) 
#     assert classification.shape == (2,)
#     assert np.array_equal(classification, np.array(['a', 'b']))

# def test_perceptron_prediction_mult_feat():
#     train_array = np.array([
#                            [1, 1],
#                            [2, 0],
#                            [0, 3],
#                            [4, 1],
#                            [3, 2]
#     ])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[0.5, 0.5], [3.5, 3.5]])
#     classification = perceptron.prediction(test_array)
#     assert isinstance(classification, np.ndarray) 
#     assert classification.shape == (2,)
#     assert np.array_equal(classification, np.array([-1, 1]))

# def test_perceptron_prediction_dimensions():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([0, 1])
#     with pytest.raises(ValueError):
#         perceptron.prediction(test_array)
    
# def test_perceptron_prediction_mismatch():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[0, 1]])
#     with pytest.raises(ValueError):
#         perceptron.prediction(test_array)

# def test_perceptron_prediction_data_type_array():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([['a'], [1]])
#     with pytest.raises(TypeError):
#         perceptron.prediction(test_array)

# def test_perceptron_prediction_data_type_unfit():
#     perceptron = Perceptron(1000, 0.01)
#     test_array = np.array([[0], [2]])
#     with pytest.raises(RuntimeError):
#         perceptron.prediction(test_array)

# def test_perceptron_scoring_basic():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = np.array([-1, 1])
#     score = perceptron.scoring(test_array, actual_array)
#     assert isinstance(score, float)
#     assert np.isclose(score, 0.5)

# def test_perceptron_scoring_correct():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = np.array([1, 1])
#     score = perceptron.scoring(test_array, actual_array)
#     assert isinstance(score, float)
#     assert np.isclose(score, 1)

# def test_perceptron_scoring_incorrect():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = np.array([-1, -1])
#     score = perceptron.scoring(test_array, actual_array)
#     assert isinstance(score, float)
#     assert np.isclose(score, 0)

# def test_perceptron_scoring_strings():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 'a', 'a', 'a'])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = np.array(['a', 'a'])
#     score = perceptron.scoring(test_array, actual_array)
#     assert isinstance(score, float)
#     assert np.isclose(score, 1)

# def test_perceptron_scoring_type_labels():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = 'np.array([1, 1])'
#     with pytest.raises(TypeError):
#         perceptron.scoring(test_array, actual_array)

# def test_perceptron_scoring_lengths():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = np.array([1, 1, 1])
#     with pytest.raises(ValueError):
#         perceptron.scoring(test_array, actual_array)

# def test_perceptron_scoring_dimensions():
#     train_array = np.array([[0], [1], [2], [3], [4]])
#     train_targets = np.array([-1, -1, 1, 1, 1])
#     perceptron = Perceptron(1000, 0.01)
#     perceptron.fit(train_array, train_targets)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = np.array([[1, 1]])
#     with pytest.raises(ValueError):
#         perceptron.scoring(test_array, actual_array)

# def test_perceptron_scoring_unfit():
#     perceptron = Perceptron(1000, 0.01)
#     test_array = np.array([[2.5], [3.5]])
#     actual_array = np.array([[1, 1]])
#     with pytest.raises(RuntimeError):
#         perceptron.scoring(test_array, actual_array)
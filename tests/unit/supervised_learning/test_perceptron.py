
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
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1], [0, 1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert isinstance(test_z, list)
    assert isinstance(test_a, list)
    assert len(test_z) == len(test_weights)
    assert len(test_a) == len(test_weights) + 1
    assert test_z[0].shape == (test_array.shape[0], 2)
    assert test_z[1].shape == (test_array.shape[0], 1)
    assert test_a[0].shape == (test_array.shape[0], 2)
    assert test_a[1].shape == (test_array.shape[0], 2)
    assert test_a[2].shape == (test_array.shape[0], 1)

def test_forward_layer_basic_range():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1], [0, 1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    for a in test_a:
        assert np.all(a >= 0)
        assert np.all(a <= 1)

def test_forward_layer_basic_multilayer():
    mlp = multilayer_Perceptron(layers = [3, 2, 2], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert isinstance(test_z, list)
    assert isinstance(test_a, list)
    assert len(test_z) == len(test_weights)
    assert len(test_a) == len(test_weights) + 1
    assert test_z[0].shape == (test_array.shape[0], 2)
    assert test_z[1].shape == (test_array.shape[0], 2)
    assert test_a[0].shape == (test_array.shape[0], 3)
    assert test_a[1].shape == (test_array.shape[0], 2)
    assert test_a[2].shape == (test_array.shape[0], 2)

def test_forward_layer_basic_random_state():
    mlp = multilayer_Perceptron(layers = [3, 2, 2], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    test_weights_1, test_bias_1= mlp._weight_initialization(42)
    test_z_1, test_a_1 = mlp._forward_layer(test_array, test_weights_1, test_bias_1)
    test_weights_2, test_bias_2= mlp._weight_initialization(42)
    test_z_2, test_a_2 = mlp._forward_layer(test_array, test_weights_2, test_bias_2)
    for (test_1, test_2) in zip(test_z_1, test_z_2):
        assert np.allclose(test_1, test_2)
    for (test_1, test_2) in zip(test_a_1, test_a_2):
        assert np.allclose(test_1, test_2)

def test_forward_layer_zero_input():
    mlp = multilayer_Perceptron(layers = [3, 2, 2], epochs = 1000, learning_rate = 0.01)
    test_array = np.zeros((3, 3))
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert np.all(test_z[0] == 0)
    assert np.all(test_a[1] == 0.5)

def test_forward_layer_single_input():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert isinstance(test_z, list)
    assert isinstance(test_a, list)
    assert len(test_z) == len(test_weights)
    assert len(test_a) == len(test_weights) + 1
    assert test_z[0].shape == (test_array.shape[0], 2)
    assert test_z[1].shape == (test_array.shape[0], 1)
    assert test_a[0].shape == (test_array.shape[0], 2)
    assert test_a[1].shape == (test_array.shape[0], 2)
    assert test_a[2].shape == (test_array.shape[0], 1)

def test_forward_layer_single_layer():
    mlp = multilayer_Perceptron(layers = [2, 1, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert isinstance(test_z, list)
    assert isinstance(test_a, list)
    assert len(test_z) == len(test_weights)
    assert len(test_a) == len(test_weights) + 1
    assert test_z[0].shape == (test_array.shape[0], 1)
    assert test_z[1].shape == (test_array.shape[0], 1)
    assert test_a[0].shape == (test_array.shape[0], 2)
    assert test_a[1].shape == (test_array.shape[0], 1)
    assert test_a[2].shape == (test_array.shape[0], 1)

def test_forward_layer_single_set():
    mlp = multilayer_Perceptron(layers = [2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert isinstance(test_z, list)
    assert isinstance(test_a, list)
    assert len(test_z) == len(test_weights)
    assert len(test_a) == len(test_weights) + 1
    assert test_z[0].shape == (test_array.shape[0], 1)
    assert test_a[0].shape == (test_array.shape[0], 2)
    assert test_a[1].shape == (test_array.shape[0], 1)

def test_forward_layer_expansion():
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0], [1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert isinstance(test_z, list)
    assert isinstance(test_a, list)
    assert len(test_z) == len(test_weights)
    assert len(test_a) == len(test_weights) + 1
    assert test_z[0].shape == (test_array.shape[0], 3)
    assert test_z[1].shape == (test_array.shape[0], 1)
    assert test_a[0].shape == (test_array.shape[0], 1)
    assert test_a[1].shape == (test_array.shape[0], 3)
    assert test_a[2].shape == (test_array.shape[0], 1)

def test_forward_layer_dimension_input_array():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[[0, 0], [1, 1]]])
    test_weights, test_bias= mlp._weight_initialization(42)
    with pytest.raises(ValueError):
        mlp._forward_layer(test_array, test_weights, test_bias)

def test_forward_layer_feature_mismatch():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0, 0], [1, 1, 0]])
    test_weights, test_bias= mlp._weight_initialization(42)
    with pytest.raises(ValueError):
        mlp._forward_layer(test_array, test_weights, test_bias)

def test_forward_layer_type_input():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([['A', 'A'], ['B', 'B']])
    test_weights, test_bias= mlp._weight_initialization(42)
    with pytest.raises(TypeError):
        mlp._forward_layer(test_array, test_weights, test_bias)

def test_forward_layer_type_input_strings():
    mlp = multilayer_Perceptron(layers = [3, 2, 2], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([['0', '0', '0'], ['1', '1', '1'], ['0', '1', '0']])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    assert isinstance(test_z, list)
    assert isinstance(test_a, list)
    assert len(test_z) == len(test_weights)
    assert len(test_a) == len(test_weights) + 1
    assert test_z[0].shape == (test_array.shape[0], 2)
    assert test_z[1].shape == (test_array.shape[0], 2)
    assert test_a[0].shape == (test_array.shape[0], 3)
    assert test_a[1].shape == (test_array.shape[0], 2)
    assert test_a[2].shape == (test_array.shape[0], 2)

def test_back_propagation_basic():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1]])
    test_targets = np.array([[0], [1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    test_d_weights, test_d_bias = mlp._back_propagation(test_z, test_a, test_weights, test_targets)
    assert isinstance(test_d_weights, list)
    assert isinstance(test_d_bias, list)
    assert len(test_d_weights) == 2
    assert len(test_d_bias) == 2
    assert test_d_weights[0].shape == test_weights[0].shape
    assert test_d_bias[0].shape == test_bias[0].shape
    assert test_d_weights[1].shape == test_weights[1].shape
    assert test_d_bias[1].shape == test_bias[1].shape

def test_back_propagation_nan():
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1]])
    test_targets = np.array([[0], [1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    test_d_weights, test_d_bias = mlp._back_propagation(test_z, test_a, test_weights, test_targets)
    for d_weight, d_bias in zip(test_d_weights, test_d_bias):
        assert not np.isnan(d_weight).any()
        assert not np.isnan(d_bias).any()

def test_back_propagation_zero_input():
    mlp = multilayer_Perceptron(layers = [1, 1, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0], [0]])
    test_targets = np.array([[0], [1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    test_d_weights, test_d_bias = mlp._back_propagation(test_z, test_a, test_weights, test_targets)
    assert np.all(test_array == 0) and np.all(test_weights[0] == test_weights[0])
    assert np.all(test_d_weights[0] != 0) or np.all(test_d_bias[0] != 0)

def test_back_propagation_multineuron():
    mlp = multilayer_Perceptron(layers = [2, 2, 2], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1]])
    test_targets = np.array([[0, 0], [1, 0]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    test_d_weights, test_d_bias = mlp._back_propagation(test_z, test_a, test_weights, test_targets)
    assert isinstance(test_d_weights, list)
    assert isinstance(test_d_bias, list)
    assert len(test_d_weights) == 2
    assert len(test_d_bias) == 2

def test_back_propagation_multilayer():
    mlp = multilayer_Perceptron(layers = [2, 2, 3, 5, 1, 2], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0, 0], [1, 1]])
    test_targets = np.array([[0, 0], [1, 0]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    test_d_weights, test_d_bias = mlp._back_propagation(test_z, test_a, test_weights, test_targets)
    assert isinstance(test_d_weights, list)
    assert isinstance(test_d_bias, list)
    assert len(test_d_weights) == 5
    assert len(test_d_bias) == 5
    for array_1, array_2 in zip(test_d_weights, test_weights):
        assert array_1.shape[0] == array_2.shape[0]
    for array_1, array_2 in zip(test_d_bias, test_bias):
        assert array_1.shape[0] == array_2.shape[0] 

def test_back_propagation_expansion():
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0], [1]])
    test_targets = np.array([[0], [1]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    test_d_weights, test_d_bias = mlp._back_propagation(test_z, test_a, test_weights, test_targets)
    assert isinstance(test_d_weights, list)
    assert isinstance(test_d_bias, list)
    assert len(test_d_weights) == 2
    assert len(test_d_bias) == 2

def test_back_propagation_dimension_input_targets():
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0], [1]])
    test_targets = np.array([[[0], [1]]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    with pytest.raises(ValueError):
        mlp._back_propagation(test_z, test_a, test_weights, test_targets)

def test_back_propagation_shape_mismatch():
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0], [1]])
    test_targets = np.array([[0], [1], [0]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    with pytest.raises(ValueError):
        mlp._back_propagation(test_z, test_a, test_weights, test_targets)

def test_back_propagation_target_mismatch():
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0], [1]])
    test_targets = np.array([[0, 0], [1, 0]])
    test_weights, test_bias= mlp._weight_initialization(42)
    test_z, test_a = mlp._forward_layer(test_array, test_weights, test_bias)
    with pytest.raises(ValueError):
        mlp._back_propagation(test_z, test_a, test_weights, test_targets)

def test_weight_update_basic():
    test_weights = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5], [0.6]])]
    test_bias = [np.array([0.1, 0.2]), np.array([0.3])]
    test_d_weights = [np.array([[0.01, 0.02], [0.03, 0.04]]), np.array([[0.05], [0.06]])]
    test_d_bias = [np.array([0.01, 0.02]), np.array([0.03])]
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    update_weights, update_bias = mlp._weight_update(test_weights, test_bias, test_d_weights, test_d_bias)
    assert isinstance(update_weights, list)
    assert isinstance(update_bias, list)
    assert len(update_weights) == len(test_weights)
    assert len(update_bias) == len(test_bias)
    for weight, update in zip(test_weights, update_weights):
        assert weight.shape == update.shape
    for bias, update in zip(test_bias, update_bias):
        assert bias.shape == update.shape
    assert np.allclose(update_weights[0], np.array([[0.09, 0.18], [0.27, 0.36]]))
    assert np.allclose(update_weights[1], np.array([[0.45], [0.54]]))
    assert np.allclose(update_bias[0], np.array([0.09, 0.18]))
    assert np.allclose(update_bias[1], np.array([[0.27]]))

def test_weight_update_basic():
    test_weights = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5], [0.6]])]
    test_bias = [np.array([0.1, 0.2]), np.array([0.3])]
    test_d_weights = [np.array([[0.01, 0.02], [0.03, 0.04]]), np.array([[0.05], [0.06]])]
    test_d_bias = [np.array([0.01, 0.02]), np.array([0.03])]
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    update_weights, update_bias = mlp._weight_update(test_weights, test_bias, test_d_weights, test_d_bias)
    assert isinstance(update_weights, list)
    assert isinstance(update_bias, list)
    assert len(update_weights) == len(test_weights)
    assert len(update_bias) == len(test_bias)
    for weight, update in zip(test_weights, update_weights):
        assert weight.shape == update.shape
    for bias, update in zip(test_bias, update_bias):
        assert bias.shape == update.shape

def test_weight_update_zero_weight():
    test_weights = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5], [0.6]])]
    test_bias = [np.array([0.1, 0.2]), np.array([0.3])]
    test_d_weights = [np.array([[0, 0], [0, 0]]), np.array([[0], [0]])]
    test_d_bias = [np.array([0., 0.]), np.array([0])]
    mlp = multilayer_Perceptron(layers = [1, 3, 1], epochs = 1000, learning_rate = 0.01)
    update_weights, update_bias = mlp._weight_update(test_weights, test_bias, test_d_weights, test_d_bias)
    assert isinstance(update_weights, list)
    assert isinstance(update_bias, list)
    assert len(update_weights) == len(test_weights)
    assert len(update_bias) == len(test_bias)
    for weight, update in zip(test_weights, update_weights):
        assert weight.shape == update.shape
        assert np.allclose(weight, update)
    for bias, update in zip(test_bias, update_bias):
        assert bias.shape == update.shape
        assert np.allclose(bias, update)

def test_mlp_fit_basic():
    train_array = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    assert isinstance(mlp.coef_, list)
    assert isinstance(mlp.bias_, list)
    for i, (weight, bias) in enumerate(zip(mlp.coef_, mlp.bias_)):
        assert weight.shape[0] == mlp.layers[i]
        assert weight.shape[1] == mlp.layers[i + 1]
        assert bias.shape[0] == mlp.layers[i + 1]

def test_mlp_fit_nonzero():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [-1], [-1], [-1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    assert isinstance(mlp.coef_, list)
    assert isinstance(mlp.bias_, list)
    for i, (weight, bias) in enumerate(zip(mlp.coef_, mlp.bias_)):
        assert weight.shape[0] == mlp.layers[i]
        assert weight.shape[1] == mlp.layers[i + 1]
        assert bias.shape[0] == mlp.layers[i + 1]

def test_mlp_random_state_reproducibility():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [-1], [-1], [-1]])
    mlp_1 = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp_1.fit(train_array, train_targets, random_state = 72)
    mlp_2 = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp_2.fit(train_array, train_targets, random_state = 72)
    for coef_1, coef_2 in zip(mlp_1.coef_, mlp_2.coef_):
        assert np.allclose(coef_1, coef_2, rtol = 1e-3, atol = 1e-3)
    for bias_1, bias_2 in zip(mlp_1.bias_, mlp_2.bias_):
        assert np.allclose(bias_1, bias_2, rtol = 1e-3, atol = 1e-3)

def test_mlp_fit_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([['A'], ['A'], ['B'], ['B'], ['B']])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_mixed():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], ['B'], ['B'], ['B']])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_nonbinary():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[-1], [-1], [1], [1], [2]])
    mlp = multilayer_Perceptron(layers = [1, 2, 3], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    assert isinstance(mlp.coef_, list)
    assert isinstance(mlp.bias_, list)
    for i, (weight, bias) in enumerate(zip(mlp.coef_, mlp.bias_)):
        assert weight.shape[0] == mlp.layers[i]
        assert weight.shape[1] == mlp.layers[i + 1]
        assert bias.shape[0] == mlp.layers[i + 1]

def test_mlp_fit_nan_array():
    train_array = np.array([[0], [np.nan], [2], [3], [4]])
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_nan_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[-1], [np.nan], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_empty_array():
    train_array = np.array([[]])
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_empty_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_data_type_array():
    train_array = np.array([['a'], [1], [2], [3], [4]])
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_type_array():
    train_array = 'np.array([[0], [1], [2], [3], [4]])'
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_type_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = 'np.array([-1, -1, 1, 1, 1])'
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_match_shape():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[-1], [-1], [1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_internal_shape():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[-1, -1], [-1, 1], [1, 0], [1, 1], [1, 1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_array_dimensions():
    train_array = np.array([0, 1, 2, 3, 4])
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_vector_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[[-1], [-1], [1], [1], [1]]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        mlp.fit(train_array, train_targets)

def test_mlp_fit_data_type_random_state():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[-1], [-1], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp.fit(train_array, train_targets, random_state = '1')

def test_mlp_verify_fit_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    mlp._verify_fit()

def test_mlp_verify_fit_unfit():
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(RuntimeError):
        mlp._verify_fit()
    
def test_mlp_prediction_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    raw_pred, class_pred = mlp.prediction(test_array)
    assert isinstance(raw_pred, np.ndarray) 
    assert isinstance(class_pred, np.ndarray) 
    assert raw_pred.shape == (2, 2)
    assert class_pred.shape == (2, 1)
    assert np.array_equal(class_pred, np.array([[1], [1]]))

def test_mlp_prediction_multi_output():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])
    mlp = multilayer_Perceptron(layers = [1, 2, 2], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets, random_state = 40)
    test_array = np.array([[0.5], [3.5]])
    raw_pred, class_pred = mlp.prediction(test_array)
    assert isinstance(raw_pred, np.ndarray) 
    assert isinstance(class_pred, np.ndarray) 
    assert raw_pred.shape == (2, 2)
    assert class_pred.shape == (2, 2)
    assert np.array_equal(class_pred, np.array([[0, 1], [1, 0]]))

def test_mlp_prediction_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([['a'], ['a'], ['b'], ['b'], ['b']])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        mlp.fit(train_array, train_targets)

def test_mlp_prediction_mult_feat():
    train_array = np.array([
                           [1, 1],
                           [2, 0],
                           [0, 3],
                           [4, 1],
                           [3, 2]
    ])
    train_targets = np.array([[0], [0], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [2, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets, random_state = 72)
    test_array = np.array([[0.5, 0.5], [3.5, 3.5]])
    raw_pred, class_pred = mlp.prediction(test_array)
    assert isinstance(raw_pred, np.ndarray) 
    assert isinstance(class_pred, np.ndarray) 
    assert raw_pred.shape == (2, 2)
    assert class_pred.shape == (2, 1)
    assert np.array_equal(class_pred, np.array([[0], [1]]))

def test_mlp_prediction_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    test_array = np.array([0, 1])
    with pytest.raises(ValueError):
        mlp.prediction(test_array)

def test_mlp_prediction_mismatch():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    test_array = np.array([[0, 1]])
    with pytest.raises(ValueError):
        mlp.prediction(test_array)

def test_mlp_prediction_data_type_array():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [1], [1], [1]])
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    mlp.fit(train_array, train_targets)
    test_array = np.array([['a'], [1]])
    with pytest.raises(TypeError):
        mlp.prediction(test_array)

def test_mlp_prediction_data_type_unfit():
    mlp = multilayer_Perceptron(layers = [1, 2, 1], epochs = 1000, learning_rate = 0.01)
    test_array = np.array([[0], [2]])
    with pytest.raises(RuntimeError):
        mlp.prediction(test_array)

import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch
from rice_ml.supervised_learning.perceptron import _validate_parameters, _validate_arrays_perceptron, Perceptron, multilayer_Perceptron

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
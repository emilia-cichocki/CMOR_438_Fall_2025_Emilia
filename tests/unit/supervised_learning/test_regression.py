
import numpy as np
import pandas as pd
import pytest
from rice_ml.supervised_learning.regression import _validate_parameters, _validate_arrays, _sigmoid, linear_regression, logistic_regression

def test_validate_parameters_basic():
    method = 'gradient_descent'
    learning_rate = 0.001
    epochs = 1000
    fit_intercept = True
    _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_no_int():
    method = 'gradient_descent'
    learning_rate = 0.001
    epochs = 1000
    fit_intercept = False
    _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_opt():
    method = 'normal'
    _validate_parameters(method)

def test_validate_parameters_gd_unspecified_epochs():
    method = 'gradient_descent'
    learning_rate = 0.001
    with pytest.raises(ValueError):
        _validate_parameters(method, learning_rate)

def test_validate_parameters_gd_unspecified_learning_rate():
    method = 'gradient_descent'
    epochs = 1000
    with pytest.raises(ValueError):
        _validate_parameters(method, epochs)

def test_validate_parameters_type_method():
    method = 'sgd'
    learning_rate = 0.001
    epochs = 1000
    fit_intercept = True
    with pytest.raises(ValueError):
        _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_type_learning_rate_int():
    method = 'gradient_descent'
    learning_rate = 1
    epochs = 1000
    fit_intercept = True
    _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_type_learning_rate():
    method = 'gradient_descent'
    learning_rate = '1'
    epochs = 1000
    fit_intercept = True
    with pytest.raises(TypeError):
        _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_learning_rate_value():
    method = 'gradient_descent'
    learning_rate = -0.001
    epochs = 1000
    fit_intercept = True
    with pytest.warns(UserWarning, match = "learning rate"):
        _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_type_epochs_float():
    method = 'gradient_descent'
    learning_rate = 0.001
    epochs = 1000.1
    fit_intercept = True
    with pytest.raises(TypeError):
        _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_type_epochs():
    method = 'gradient_descent'
    learning_rate = 0.001
    epochs = '1000'
    fit_intercept = True
    with pytest.raises(TypeError):
        _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_epochs_value():
    method = 'gradient_descent'
    learning_rate = 0.001
    epochs = -1
    fit_intercept = True
    with pytest.raises(ValueError):
        _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_parameters_type_fit_intercept():
    method = 'gradient_descent'
    learning_rate = 0.001
    epochs = 1000
    fit_intercept = 'True'
    with pytest.raises(TypeError):
        _validate_parameters(method, learning_rate, epochs, fit_intercept)

def test_validate_arrays_basic_array():
    test_array = np.array([[1, 2], [3, 4]])
    result_array = _validate_arrays(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)
    assert np.array_equal(test_array, result_array)

def test_validate_arrays_basic_array_df():
    test_array = pd.DataFrame({
        'A': [1, 3],
        'B': [2, 4]})
    result_array = _validate_arrays(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)
    
def test_validate_arrays_basic_array_list():
    test_array = [[1, 2], [3, 4]]
    result_array = _validate_arrays(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)

def test_validate_arrays_basic_array_tuple():
    test_array = ([1, 2], [3, 4])
    result_array = _validate_arrays(data_array = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 2)

def test_validate_arrays_array_dimension():
    test_array = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError):
        _validate_arrays(data_array = test_array)

def test_validate_arrays_basic_vector():
    test_array = np.array([1, 2])
    result_array = _validate_arrays(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(test_array, result_array)

def test_validate_arrays_basic_vector_df():
    test_array = pd.Series([1, 2])
    result_array = _validate_arrays(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)

def test_validate_arrays_basic_vector_list():
    test_array = [1, 2]
    result_array = _validate_arrays(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)

def test_validate_arrays_basic_vector_tuple():
    test_array = (1, 2)
    result_array = _validate_arrays(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)

def test_validate_arrays_vector_2D_hor():
    test_array = np.array([[1, 2]])
    result_array = _validate_arrays(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(result_array, np.array([1, 2]))

def test_validate_arrays_vector_2D_ver():
    test_array = np.array([[1], [2]])
    result_array = _validate_arrays(target_vector = test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2,)
    assert np.array_equal(result_array, np.array([1, 2]))

def test_validate_arrays_vector_2D_dimensions():
    test_array = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _validate_arrays(target_vector = test_array)

def test_validate_arrays_vector_dimension():
    test_array = np.array([[[1, 2]]])
    with pytest.raises(ValueError):
        _validate_arrays(data_array = test_array)

def test_validate_arrays_array_nan():
    test_array = np.array([[1, np.nan], [3, 4]])
    with pytest.raises(ValueError):
        _validate_arrays(data_array = test_array)

def test_validate_arrays_array_data_type_input():
    test_array = np.array([[1, 'a'], [3, 4]])
    with pytest.raises(TypeError):
        _validate_arrays(data_array = test_array)

def test_validate_arrays_array_type_input():
    test_array = 'np.array([[1, 2], [3, 4]])'
    with pytest.raises(TypeError):
        _validate_arrays(data_array = test_array)

def test_validate_arrays_vector_nan():
    test_array = np.array([1, np.nan])
    with pytest.raises(ValueError):
        _validate_arrays(target_vector = test_array)

def test_validate_arrays_vector_data_type_input():
    test_array = np.array([1, 'a'])
    with pytest.raises(TypeError):
        _validate_arrays(target_vector = test_array)

def test_validate_arrays_vector_type_input():
    test_array = 'np.array([1, 2])'
    with pytest.raises(ValueError):
        _validate_arrays(target_vector = test_array)

def test_validate_arrays_data_type_vector():
    test_array = np.array([[1, 'a']])
    with pytest.raises(TypeError):
        _validate_arrays(target_vector = test_array)

def test_validate_arrays_combined():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([1, 2])
    result_array, result_vector = _validate_arrays(data_array = test_array, target_vector = test_vector)
    assert isinstance(result_array, np.ndarray)
    assert isinstance(result_vector, np.ndarray)
    assert result_array.shape == (2, 2)
    assert result_vector.shape == (2,)

def test_validate_arrays_combined_2D():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([[1, 2]])
    result_array, result_vector = _validate_arrays(data_array = test_array, target_vector = test_vector)
    assert isinstance(result_array, np.ndarray)
    assert isinstance(result_vector, np.ndarray)
    assert result_array.shape == (2, 2)
    assert result_vector.shape == (2,)

def test_validate_arrays_match_shapes():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        _validate_arrays(data_array = test_array, target_vector = test_vector)

def test_validate_arrays_empty_array():
    test_array = np.array([[]])
    test_vector = np.array([1, 2])
    with pytest.raises(ValueError):
        _validate_arrays(data_array = test_array, target_vector = test_vector)

def test_validate_arrays_empty_vector():
    test_array = np.array([[1, 2], [3, 4]])
    test_vector = np.array([])
    with pytest.raises(ValueError):
        _validate_arrays(data_array = test_array, target_vector = test_vector)

def test_linear_regression_init_basic_normal():
    linreg = linear_regression('normal', True)
    assert linreg.method == 'normal'
    assert linreg.fit_intercept == True
    assert linreg.learning_rate is None
    assert linreg.epochs is None
    assert linreg.coef_ is None
    assert linreg.intercept_ is None
    assert linreg._training_array is None
    assert linreg._training_targets is None

def test_linear_regression_init_basic_sgd():
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.001, epochs = 1000)
    assert linreg.method == 'gradient_descent'
    assert linreg.fit_intercept == True
    assert linreg.learning_rate == 0.001
    assert linreg.epochs == 1000
    assert linreg.coef_ is None
    assert linreg.intercept_ is None
    assert linreg._training_array is None
    assert linreg._training_targets is None

def test_linear_regression_init_type_inputs():
    with pytest.raises(ValueError):
        linear_regression('sgd')
    with pytest.raises(TypeError):
        linear_regression('normal', 'True')
    with pytest.raises(TypeError):
        linear_regression('gradient_descent', learning_rate = '0.01', epochs = 1000)
    with pytest.raises(TypeError):
        linear_regression('gradient_descent', learning_rate = 0.001, epochs = '1000')

def test_linear_regression_init_input_values():
    with pytest.warns(UserWarning, match = "learning rate"):
        linear_regression('gradient_descent', learning_rate = -0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linear_regression('gradient_descent', learning_rate = 0.01, epochs = -1000)

def test_linear_regression_verify_fit_basic():
    linreg = linear_regression('normal', True)
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array([1, 5, 8, 11, 14])
    linreg.fit(train_array, train_targets, False)
    training, targets = linreg._verify_fit()
    assert np.array_equal(training, train_array)
    assert np.array_equal(targets, train_targets)

def test_linear_regression_verify_fit_unfit():
    linreg = linear_regression('normal', True)
    with pytest.raises(RuntimeError):
        linreg._verify_fit()

def test_logistic_regression_init_basic():
    logreg = logistic_regression(1000, 0.01)
    assert logreg.epochs == 1000
    assert logreg.learning_rate == 0.01
    assert logreg.coef_ is None
    assert logreg.bias_ is None
    assert logreg.class_mapping_ is None

def test_logistic_regression_init_type_inputs():
    with pytest.raises(TypeError):
        logistic_regression('1000', 0.01)
    with pytest.raises(TypeError):
        logistic_regression(1000, '0.01')

def test_logistic_regression_init_input_values():
    with pytest.warns(UserWarning, match = "learning rate"):
        logistic_regression(epochs = 1000, learning_rate = -0.01)
    with pytest.raises(ValueError):
        logistic_regression(epochs = -1000, learning_rate = 0.01)

def test_logistic_regression_verify_fit_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    logreg._verify_fit()

def test_logistic_regression_verify_fit_unfit():
    logreg = logistic_regression(1000, 0.01)
    with pytest.raises(RuntimeError):
        logreg._verify_fit()
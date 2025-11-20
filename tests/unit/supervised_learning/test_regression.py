
import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch
from rice_ml.supervised_learning.regression import _validate_parameters, _validate_arrays, linear_regression

# TODO: rename 'test_arrays' for lists/df, fix the formatting and spacing, add comments to indicate functions being tested


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




def test_linear_regression_fit_basic_normal():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([2, 5, 8, 11, 14])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.array_equal(linreg.coef_, np.array([3]))
    assert linreg.intercept_ == 2
    assert np.array_equal(linreg._training_array, train_array)
    assert np.array_equal(linreg._training_targets, train_targets)

def test_linear_regression_fit_basic_sgd():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([2, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([3]))
    assert np.isclose(linreg.intercept_, 2)

def test_linear_regression_fit_negative():
    train_array = np.array([[0],[2],[4],[6]])
    train_targets =  np.array([10, 2, -6, -14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([-4]))
    assert np.isclose(linreg.intercept_, 10)

def test_linear_regression_fit_zero_int():
    train_array = np.array([[0],[2],[4],[6]])
    train_targets =  np.array([0, 4, 8, 12])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([2]))
    assert np.isclose(linreg.intercept_, 0, rtol = 1e-5, atol = 1e-5)

def test_linear_regression_fit_normal_no_int():
    train_array = np.array([[0], [1], [2], [3]])
    train_targets =  np.array([3, 5, 7, 9])
    linreg = linear_regression('normal', False)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert linreg.intercept_ is None
    assert linreg.coef_.shape == (1,)
    assert not np.allclose(linreg.coef_, np.array([2]))
    assert np.isclose(linreg.coef_[0], 3.2857)

def test_linear_regression_fit_sgd_no_int():
    train_array = np.array([[0], [1], [2], [3]])
    train_targets =  np.array([3, 5, 7, 9])
    linreg = linear_regression('gradient_descent', False, learning_rate = 0.001, epochs = 10000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert linreg.intercept_ is None
    assert linreg.coef_.shape == (1,)
    assert not np.allclose(linreg.coef_, np.array([2]))
    assert np.isclose(linreg.coef_[0], 3.2857, rtol = 1e-3, atol = 1e-3)

def test_linear_regression_fit_mult_feat():
    train_array = np.array([
                           [1, 1],
                           [2, 0],
                           [0, 3],
                           [4, 1],
                           [3, 2]
    ])
    train_targets = np.array([4, 9, -4, 10, 5])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 10000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert np.isclose(linreg.intercept_, 5)
    assert linreg.coef_.shape == (2,)
    assert np.allclose(linreg.coef_, np.array([2, -3]))

def test_linear_regression_fit_noise_normal():
    rng = np.random.default_rng(0)
    train_array = rng.uniform(-5, 5, size = (50, 1))
    noise = rng.normal(0, 0.5, size=50)
    train_targets = 4 * train_array[:, 0] + 1 + noise
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert np.isclose(linreg.intercept_, 1.009, rtol = 1e-3, atol = 1e-3)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([4.026]), rtol = 1e-3, atol = 1e-3)

def test_linear_regression_fit_noise_sgd():
    rng = np.random.default_rng(0)
    train_array = rng.uniform(-5, 5, size = (50, 1))
    noise = rng.normal(0, 0.5, size=50)
    train_targets = 4 * train_array[:, 0] + 1 + noise
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.001, epochs = 1000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert np.isclose(linreg.intercept_, 1.009, rtol = 1e-3, atol = 1e-3)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([4.026]), rtol = 1e-3, atol = 1e-3)

def test_linear_regression_fit_shuffle():
    train_array = np.array([[1], [2], [3], [4]])
    train_targets = np.array([2, 4, 6, 8])
    linreg1 = linear_regression('gradient_descent', fit_intercept=True, learning_rate = 0.01, epochs = 1000)
    linreg1.fit(train_array, train_targets, random_state = 42, shuffle = True)
    assert linreg1.coef_.shape == (1,)
    assert np.isscalar(linreg1.intercept_)
    linreg2 = linear_regression('gradient_descent', fit_intercept = True, learning_rate = 0.01, epochs = 1000)
    linreg2.fit(train_array, train_targets, random_state = 42, shuffle = False)
    assert linreg2.coef_.shape == (1,)
    assert np.isscalar(linreg2.intercept_)
    assert np.allclose(linreg1.coef_, linreg2.coef_, rtol = 1e-3, atol = 1e-3)
    assert np.isclose(linreg1.intercept_, linreg2.intercept_, rtol = 1e-3, atol = 1e-3)

def test_gradient_descent_random_state_reproducibility():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([2, 4, 6])
    linreg1 = linear_regression('gradient_descent', fit_intercept = True, learning_rate = 0.1, epochs = 100)
    linreg1.fit(train_array, train_targets, random_state = 72, shuffle = True)
    linreg2 = linear_regression('gradient_descent', fit_intercept = True, learning_rate = 0.1, epochs = 100)
    linreg2.fit(train_array, train_targets, random_state = 72, shuffle = True)
    assert np.allclose(linreg1.coef_, linreg2.coef_, atol = 1e-5)
    assert np.isclose(linreg1.intercept_, linreg2.intercept_, atol = 1e-5)

def test_linear_regression_fit_normal_2D_hor():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([[2, 5, 8, 11, 14]])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.array_equal(linreg.coef_, np.array([3]))
    assert linreg.intercept_ == 2

def test_linear_regression_fit_normal_2D_ver():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([[2], [5], [8], [11], [14]])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([3]))
    assert np.isclose(linreg.intercept_, 2)

def test_linear_regression_fit_sgd_2D_hor():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([[2, 5, 8, 11, 14]])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([3]))
    assert np.isclose(linreg.intercept_, 2)

def test_linear_regression_fit_sgd_ver():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([[2], [5], [8], [11], [14]])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (1,)
    assert np.allclose(linreg.coef_, np.array([3]))
    assert np.isclose(linreg.intercept_, 2)

def test_linear_regression_fit_nan_array():
    train_array = np.array([[np.nan],[1],[2],[3],[4]])
    train_targets = np.array([[2], [5], [8], [11], [14]])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_nan_vector():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array([np.nan, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_empty_array():
    train_array = np.array([[]])
    train_targets = np.array([1, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_empty_vector():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array([])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_data_type_array():
    train_array = np.array([['a'],[1],[2],[3],[4]])
    train_targets = np.array([1, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(TypeError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_data_type_vector():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array(['a', 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(TypeError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_type_array():
    train_array = 'np.array([[1],[1],[2],[3],[4]])'
    train_targets = np.array([1, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(TypeError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_type_vector():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = 'np.array([1, 5, 8, 11, 14])'
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_match_shape():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array([1, 5, 8, 11, 14, 10])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_internal_shape():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array([[1, 2], [5, 6], [8, 9], [11, 12], [14, 15]])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_array_dimensions():
    train_array = np.array([1, 1, 2, 3, 4])
    train_targets = np.array([1, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_vector_dimensions():
    train_array = np.array([[1, 1, 2, 3, 4]])
    train_targets = np.array([[[1, 5, 8, 11, 14]]])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_vector_dimensions():
    train_array = np.array([[1, 1, 2, 3, 4]])
    train_targets = np.array([[[1, 5, 8, 11, 14]]])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, None)

def test_linear_regression_fit_singular_normal():
    train_array = np.array([[1, 2], [2, 4], [3, 6]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', fit_intercept = True)
    with pytest.raises(ValueError):
        linreg.fit(train_array, train_targets, random_state=None)

def test_linear_regression_fit_singular_sgd():
    train_array = np.array([[1, 2], [2, 4], [3, 6]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('gradient_descent', fit_intercept = True, learning_rate = 0.01, epochs = 1000)
    linreg.fit(train_array, train_targets, None)
    assert isinstance(linreg.coef_, np.ndarray)
    assert np.isscalar(linreg.intercept_)
    assert linreg.coef_.shape == (2,)

def test_linear_regression_fit_data_type_random_state():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array([1, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(TypeError):
        linreg.fit(train_array, train_targets, '1')

def test_linear_regression_fit_data_type_shuffle():
    train_array = np.array([[1],[1],[2],[3],[4]])
    train_targets = np.array([1, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    with pytest.raises(TypeError):
        linreg.fit(train_array, train_targets, shuffle = 'True')

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


def test_linear_regression_prediction_basic_normal():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([2, 5, 8, 11, 14])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    test_array = np.array([[3], [4]])
    prediction = linreg.prediction(test_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert np.allclose(prediction, np.array([11, 14]))

def test_linear_regression_prediction_basic_sgd():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([2, 5, 8, 11, 14])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 1000)
    linreg.fit(train_array, train_targets, 42)
    test_array = np.array([[3], [4]])
    prediction = linreg.prediction(test_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert np.allclose(prediction, np.array([11, 14]))

def test_linear_regression_prediction_no_int():
    train_array = np.array([[0], [1], [2], [3]])
    train_targets =  np.array([3, 5, 7, 9])
    linreg = linear_regression('normal', False)
    linreg.fit(train_array, train_targets, None)
    test_array = np.array([[3], [4]])
    prediction = linreg.prediction(test_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert np.allclose(prediction, np.array([9.857, 13.143]), rtol = 1e-3, atol = 1e-3)

def test_linear_regression_prediction_mult_feat():
    train_array = np.array([
                           [1, 1],
                           [2, 0],
                           [0, 3],
                           [4, 1],
                           [3, 2]
    ])
    train_targets = np.array([4, 9, -4, 10, 5])
    linreg = linear_regression('gradient_descent', True, learning_rate = 0.01, epochs = 10000)
    linreg.fit(train_array, train_targets, None)
    test_array = np.array([[3, 1], [4, 1]])
    prediction = linreg.prediction(test_array)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert np.allclose(prediction, np.array([8, 10]))

def test_linear_regression_prediction_dimensions():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([2, 5, 8, 11, 14])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    test_array = np.array([3, 4])
    with pytest.raises(ValueError):
        linreg.prediction(test_array)

def test_linear_regression_prediction_feature_mismatch():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([2, 5, 8, 11, 14])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    test_array = np.array([[3, 4]])
    with pytest.raises(ValueError):
        linreg.prediction(test_array)

def test_linear_regression_prediction_data_type_array():
    train_array = np.array([[0],[1],[2],[3],[4]])
    train_targets = np.array([2, 5, 8, 11, 14])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets, None)
    test_array = np.array([['a'], [4]])
    with pytest.raises(TypeError):
        linreg.prediction(test_array)

def test_linear_regression_prediction_unfit():
    test_array = np.array([[3], [4]])
    linreg = linear_regression('normal', True)
    with pytest.raises(RuntimeError):
        linreg.prediction(test_array)



def test_linear_regression_scoring_basic():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = np.array([4, 0])
    score = linreg.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, -2.125)

def test_linear_regression_scoring_correct():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = np.array([4, 5])
    score = linreg.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 1)

def test_linear_regression_scoring_training_data():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    actual_array = np.array([1, 2, 3])
    score = linreg.scoring(train_array, actual_array)
    assert isinstance(score, float)
    assert score == 1.0

def test_linear_regression_scoring_uniform_actual():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[2], [4], [6]])
    actual_array = np.array([2, 2, 2])
    with pytest.raises(ValueError):
        linreg.scoring(test_array, actual_array)

def test_linear_regression_scoring_type_labels():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = 'np.array([2, 2])'
    with pytest.raises(TypeError):
        linreg.scoring(test_array, actual_array)

def test_linear_regression_scoring_strings():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        linreg.scoring(test_array, actual_array)

def test_linear_regression_scoring_strings():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = np.array(['a', 2])
    with pytest.raises(TypeError):
        linreg.scoring(test_array, actual_array)

def test_linear_regression_scoring_nan():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = np.array([1, np.nan])
    with pytest.raises(ValueError):
        linreg.scoring(test_array, actual_array)

def test_linear_regression_scoring_unfit():
    linreg = linear_regression('normal', True)
    test_array = np.array([[4], [5]])
    actual_array = np.array([1, 2])
    with pytest.raises(RuntimeError):
        linreg.scoring(test_array, actual_array)
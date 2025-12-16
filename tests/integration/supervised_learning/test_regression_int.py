
import numpy as np
import pandas as pd
import pytest
from rice_ml.supervised_learning.regression import linear_regression, logistic_regression

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

def test_linear_regression_scoring_lengths():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        linreg.scoring(test_array, actual_array)

def test_linear_regression_scoring_dimensions():
    train_array = np.array([[1], [2], [3]])
    train_targets = np.array([1, 2, 3])
    linreg = linear_regression('normal', True)
    linreg.fit(train_array, train_targets)
    test_array = np.array([[4], [5]])
    actual_array = np.array([[1, 1]])
    with pytest.raises(ValueError):
        linreg.scoring(test_array, actual_array)

def test_linear_regression_scoring_unfit():
    linreg = linear_regression('normal', True)
    test_array = np.array([[4], [5]])
    actual_array = np.array([1, 2])
    with pytest.raises(RuntimeError):
        linreg.scoring(test_array, actual_array)


def test_logistic_regression_fit_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (1,)
    assert logreg.coef_[0] > 0
    assert logreg.bias_ < 0

def test_logistic_regression_fit_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (1,)
    assert logreg.coef_[0] > 0
    assert logreg.bias_ < 0

def test_logistic_regression_fit_nonzero():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, -1, -1, -1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (1,)
    assert logreg.coef_[0] < 0
    assert logreg.bias_ > 0

def test_logistic_regression_fit_mult_feat():
    train_array = np.array([
        [1, 1],
        [2, 0],
        [0, 3],
        [4, 1],
        [3, 2]
    ])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (2,)
    
def test_logistic_regression_fit_shuffle():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg1 = logistic_regression(epochs = 10000, learning_rate = 0.1)
    logreg1.fit(train_array, train_targets, shuffle = True)
    assert isinstance(logreg1.coef_, np.ndarray)
    assert np.isscalar(logreg1.bias_)
    assert logreg1.coef_.shape == (1,)
    logreg2 = logistic_regression(epochs = 10000, learning_rate = 0.1)
    logreg2.fit(train_array, train_targets, shuffle = True)
    assert isinstance(logreg2.coef_, np.ndarray)
    assert np.isscalar(logreg2.bias_)
    assert logreg1.coef_.shape == (1,)
    assert np.allclose(logreg1.coef_, logreg2.coef_, rtol = 1e-2, atol = 1e-2)
    assert np.isclose(logreg1.bias_, logreg2.bias_, rtol = 1e-2, atol = 1e-2)

def test_logistic_regression_random_state_reproducibility():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg1 = logistic_regression(epochs = 10000, learning_rate = 0.01)
    logreg1.fit(train_array, train_targets, random_state = 72, shuffle = True)
    logreg2 = logistic_regression(epochs = 10000, learning_rate = 0.01)
    logreg2.fit(train_array, train_targets, random_state = 72, shuffle = True)
    assert np.allclose(logreg1.coef_, logreg2.coef_, rtol = 1e-3, atol = 1e-3)
    assert np.isclose(logreg1.bias_, logreg2.bias_, rtol = 1e-3, atol = 1e-3)

def test_logistic_regression_fit_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array(['a', 'a', 'b', 'b', 'b'])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (1,)
    assert logreg.coef_[0] > 0
    assert logreg.bias_ < 0

def test_logistic_regression_fit_mixed():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 'b', 'b', 'b'])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (1,)
    assert logreg.coef_[0] > 0
    assert logreg.bias_ < 0

def test_logistic_regression_fit_2D_hor():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0, 0, 1, 1, 1]])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (1,)
    assert logreg.coef_[0] > 0
    assert logreg.bias_ < 0

def test_logistic_regression_fit_2D_ver():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0], [0], [1], [1], [1]])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    logreg.fit(train_array, train_targets)
    assert isinstance(logreg.coef_, np.ndarray)
    assert np.isscalar(logreg.bias_)
    assert logreg.coef_.shape == (1,)
    assert logreg.coef_[0] > 0
    assert logreg.bias_ < 0

def test_logistic_regression_fit_nonbinary():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 2])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_nan_array():
    train_array = np.array([[0], [np.nan], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_nan_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, np.nan, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_empty_array():
    train_array = np.array([[]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_empty_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_data_type_array():
    train_array = np.array([['a'], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_type_array():
    train_array = 'np.array([[0], [1], [2], [3], [4]])'
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_type_vector():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = 'np.array([0, 0, 1, 1, 1])'
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_match_shape():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_internal_shape():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_array_dimensions():
    train_array = np.array([0, 1, 2, 3, 4])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_vector_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([[[0, 0, 1, 1, 1]]])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(ValueError):
        logreg.fit(train_array, train_targets)

def test_logistic_regression_fit_data_type_random_state():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        logreg.fit(train_array, train_targets, random_state = '1')

def test_logistic_regression_fit_data_type_shuffle():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(epochs = 1000, learning_rate = 0.01)
    with pytest.raises(TypeError):
        logreg.fit(train_array, train_targets, shuffle = 'True')

def test_logistic_regression_prediction_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    classification = logreg.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array([1, 1]))

def test_logistic_regression_prediction_neg():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, -1, -1, -1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    classification = logreg.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array([-1, -1]))

def test_logistic_regression_prediction_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array(['a', 'a', 'b', 'b', 'b'])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[0.5], [3.5]])
    classification = logreg.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array(['a', 'b']))

def test_logistic_regression_prediction_mult_feat():
    train_array = np.array([
                           [1, 1],
                           [2, 0],
                           [0, 3],
                           [4, 1],
                           [3, 2]
    ])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[0.5, 0.5], [3.5, 3.5]])
    classification = logreg.prediction(test_array)
    assert isinstance(classification, np.ndarray) 
    assert classification.shape == (2,)
    assert np.array_equal(classification, np.array([0, 1]))

def test_logistic_regression_prediction_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([0, 1])
    with pytest.raises(ValueError):
        logreg.prediction(test_array)
    
def test_logistic_regression_prediction_mismatch():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[0, 1]])
    with pytest.raises(ValueError):
        logreg.prediction(test_array)

def test_logistic_regression_prediction_data_type_array():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([['a'], [1]])
    with pytest.raises(TypeError):
        logreg.prediction(test_array)

def test_logistic_regression_prediction_data_type_unfit():
    logreg = logistic_regression(1000, 0.01)
    test_array = np.array([[0], [2]])
    with pytest.raises(RuntimeError):
        logreg.prediction(test_array)

def test_logistic_regression_prediction_threshold():
    logreg = logistic_regression(1000, 0.01)
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[1], [1]])
    with pytest.raises(TypeError):
        logreg.prediction(test_array, '0.5')

def test_logistic_regression_scoring_basic():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([0, 1])
    score = logreg.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 0.5)

def test_logistic_regression_scoring_correct():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([1, 1])
    score = logreg.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 1)

def test_logistic_regression_scoring_incorrect():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([0, 0])
    score = logreg.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 0)

def test_logistic_regression_scoring_strings():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 'a', 'a', 'a'])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array(['a', 'a'])
    score = logreg.scoring(test_array, actual_array)
    assert isinstance(score, float)
    assert np.isclose(score, 1)

def test_logistic_regression_scoring_type_labels():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = 'np.array([1, 1])'
    with pytest.raises(TypeError):
        logreg.scoring(test_array, actual_array)

def test_logistic_regression_scoring_lengths():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        logreg.scoring(test_array, actual_array)

def test_logistic_regression_scoring_dimensions():
    train_array = np.array([[0], [1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 1, 1])
    logreg = logistic_regression(1000, 0.01)
    logreg.fit(train_array, train_targets)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([[1, 1]])
    with pytest.raises(ValueError):
        logreg.scoring(test_array, actual_array)

def test_logistic_regression_scoring_unfit():
    logreg = logistic_regression(1000, 0.01)
    test_array = np.array([[2.5], [3.5]])
    actual_array = np.array([[1, 1]])
    with pytest.raises(RuntimeError):
        logreg.scoring(test_array, actual_array)
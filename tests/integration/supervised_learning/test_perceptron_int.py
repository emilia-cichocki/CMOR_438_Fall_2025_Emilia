
import numpy as np
import pandas as pd
import pytest
from rice_ml.supervised_learning.perceptron import Perceptron, multilayer_Perceptron

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
    assert isinstance(perceptron.error_, list)

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
    assert isinstance(mlp.error_, list)

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
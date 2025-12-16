
import numpy as np
import pytest
from rice_ml.supervised_learning.randomforest import random_forest

def test_fit_classification_basic():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array([0, 0, 1, 1])
    rf = random_forest(100, 'classification')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert rf.n_features_ == 2
    assert len(rf.trees) == 100
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 2 for feature in features)

def test_fit_regression_basic():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array([0, 0, 1, 1])
    rf = random_forest(100, 'regression')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert len(rf.trees) == 100
    assert rf.n_features_ == 2
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 2 for feature in features)

def test_fit_type_input_array_string_numeric():
    test_array = np.array([['1', '2'], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array([0, 0, 1, 1])
    rf = random_forest(100, 'regression')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert rf.n_features_ == 2
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 2 for feature in features)

def test_fit_type_input_array_string_nonnumeric():
    test_array = np.array([['A', 'B'], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array([0, 0, 1, 1])
    rf = random_forest(100, 'regression')
    with pytest.raises(TypeError):
        rf.fit(test_array, test_targets)

def test_fit_type_input_targets_string_numeric():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array(['0', '0', 1, 1])
    rf = random_forest(100, 'regression')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert rf.n_features_ == 2
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 2 for feature in features)

def test_fit_classification_type_input_targets_string_nonnumeric():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array(['A', 'A', 'B', 'B'])
    rf = random_forest(100, 'classification')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert rf.n_features_ == 2
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 2 for feature in features)

def test_fit_regression_type_input_targets_string_nonnumeric():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array(['A', 'A', 'B', 'B'])
    rf = random_forest(100, 'regression')
    with pytest.raises(TypeError):
        rf.fit(test_array, test_targets)

def test_fit_random_state():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array([0, 0, 1, 1])
    rf_1 = random_forest(100, 'classification', random_state = 42)
    rf_1.fit(test_array, test_targets)
    rf_2 = random_forest(100, 'classification', random_state = 42)
    rf_2.fit(test_array, test_targets)
    for (rf_1_tree, rf_1_feature),  (rf_2_tree, rf_2_feature) in zip(rf_1.trees, rf_2.trees):
        assert np.allclose(rf_1_feature, rf_2_feature)

def test_fit_dimension_test_array():
    test_array = np.array([[[1, 2], [2, 1], [3, 4], [4, 3]]])
    test_targets = np.array([0, 0, 1, 1])
    rf = random_forest(100, 'regression')
    with pytest.raises(ValueError):
        rf.fit(test_array, test_targets)

def test_fit_dimension_test_targets():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array([[0, 0, 1, 1]])
    rf = random_forest(100, 'regression')
    with pytest.raises(ValueError):
        rf.fit(test_array, test_targets)

def test_fit_shape_mismatch():
    test_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    test_targets = np.array([[0, 0, 1, 1, 0]])
    rf = random_forest(100, 'regression')
    with pytest.raises(ValueError):
        rf.fit(test_array, test_targets)

def test_fit_one_sample():
    test_array = np.array([[1, 2]])
    test_targets = np.array([0])
    rf = random_forest(100, 'classification')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert rf.n_features_ == 2
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 2 for feature in features)

def test_fit_one_feature():
    test_array = np.array([[1], [2]])
    test_targets = np.array([0, 1])
    rf = random_forest(100, 'classification')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert rf.n_features_ == 1
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 1 for feature in features)

def test_fit_one_target():
    test_array = np.array([[1], [2]])
    test_targets = np.array([0, 0])
    rf = random_forest(100, 'classification')
    rf.fit(test_array, test_targets)
    assert rf.trees is not None
    assert isinstance(rf.trees, list)
    assert rf.n_features_ == 1
    for tree, features in rf.trees:
        assert hasattr(tree, 'tree')
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert all(feature < 1 for feature in features)
        assert tree.tree.value == 0


def test_prediction_classification_basic():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1, 0.5], [3, 4.5]])
    rf = random_forest(100, 'classification', random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_prediction_regression_basic():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1, 0.5], [3, 4.5]])
    rf = random_forest(100, 'regression', random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_prediction_basic_max_depth():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1, 0.5], [3, 4.5]])
    rf = random_forest(100, 'classification', max_depth = 5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_prediction_basic_min_samples():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1, 0.5], [3, 4.5]])
    rf = random_forest(100, 'classification', min_samples_split = 1, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_prediction_basic_max_features():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1, 0.5], [3, 4.5]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 1])

def test_prediction_mult_class():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[1, 0.5], [4, 3.5]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert np.allclose(predictions, [0, 2])

def test_prediction_one_sample():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[1, 0.5]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1,)
    assert np.allclose(predictions, [0])

def test_prediction_one_feature():
    train_array = np.array([[1], [2], [3], [4]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[1]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1,)
    assert np.allclose(predictions, [0])

def test_prediction_one_sample():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[1, 0.5]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1,)
    assert np.allclose(predictions, [0])

def test_prediction_type_input_array_string_numeric():
    train_array = np.array([['1', '2'], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[1, 0.5]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1,)
    assert np.allclose(predictions, [0])

def test_prediction_type_test_array_string_numeric():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([['1', '0.5']])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1,)
    assert np.allclose(predictions, [0])

def test_prediction_type_test_array_string_nonnumeric():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([['A', 'B']])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    with pytest.raises(TypeError):
        rf.prediction(test_array)

def test_prediction_dimension_test_array():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[[0, 1]]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    with pytest.raises(ValueError):
        rf.prediction(test_array)

def test_prediction_shape_mismatch():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[0, 1, 1]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    with pytest.raises(ValueError):
        rf.prediction(test_array)

def test_prediction_empty():
    train_array = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[]])
    rf = random_forest(100, 'classification', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    with pytest.raises(ValueError):
        rf.prediction(test_array)

def test_prediction_regression_mean():
    train_array = np.array([[1], [1], [1]])
    train_targets = np.array([10, 10, 10])
    test_array = np.array([[1], [1]])
    rf = random_forest(100, 'regression', max_features = 0.5, random_state = 42)
    rf.fit(train_array, train_targets)
    predictions = rf.prediction(test_array)
    assert np.all(predictions == 10)


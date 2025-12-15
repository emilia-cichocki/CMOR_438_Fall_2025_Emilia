
import numpy as np
import pytest
from rice_ml.supervised_learning.randomforest import _validate_parameters_rf, random_forest

# TODO: formatting

def test_validate_parameters_basic():
    _validate_parameters_rf(100, 'classification', 10, 2, 'sqrt', 42)
    _validate_parameters_rf(100, 'classification', None, None, None, None)

def test_validate_parameters_type_inputs():
    with pytest.raises(TypeError):
        _validate_parameters_rf(100.5, 'classification', 10, 2, 'sqrt', 42)
    with pytest.raises(TypeError):
        _validate_parameters_rf(100, 'classification', '10', 2, 'sqrt', 42)
    with pytest.raises(TypeError):
        _validate_parameters_rf(100, 'classification', 10, 2.5, 'sqrt', 42)
    with pytest.raises(TypeError):
        _validate_parameters_rf(100, 'classification', 10, 2, 'square', 42)
    with pytest.raises(TypeError):
        _validate_parameters_rf(100, 'classification', 10, 2, 'sqrt', 42.5)

def test_validate_parameters_input_values():
    with pytest.raises(ValueError):
        _validate_parameters_rf(-1, 'classification', 10, 2, 'sqrt', 42)
    with pytest.raises(ValueError):
        _validate_parameters_rf(100, 'classify', 10, 2, 'sqrt', 42)
    with pytest.raises(ValueError):
        _validate_parameters_rf(100, 'classification', -10, 2, 'sqrt', 42)
    with pytest.raises(ValueError):
        _validate_parameters_rf(100, 'classification', 10, -2, 'sqrt', 42)
    with pytest.raises(ValueError):
        _validate_parameters_rf(100, 'classification', 10, 2, -1, 42)

def test_random_forest_init_basic():
    rf = random_forest(100, 'classification', 20, 2, 'sqrt', 42)
    assert rf.n_trees == 100
    assert rf.task == 'classification'
    assert rf.max_depth == 20
    assert rf.min_samples_split == 2
    assert rf.max_features == 'sqrt'
    assert rf.random_state == 42
    assert rf.trees == []
    assert rf.n_features_ == None

def test_random_forest_init_mixed():
    rf = random_forest(100, 'regression', 20, 2, 5, None)
    assert rf.n_trees == 100
    assert rf.task == 'regression'
    assert rf.max_depth == 20
    assert rf.min_samples_split == 2
    assert rf.max_features == 5
    assert rf.random_state == None
    assert rf.trees == []
    assert rf.n_features_ == None

def test_random_forest_type_inputs():
    with pytest.raises(TypeError):
        random_forest('100', 'classification', 20, 2, 'sqrt', 42)
    with pytest.raises(TypeError):
        random_forest(100, 'classification', '20', 2, 'sqrt', 42)
    with pytest.raises(TypeError):
        random_forest(100, 'classification', 20, '2', 'sqrt', 42)
    with pytest.raises(TypeError):
        random_forest(100, 'classification', 20, 2, 'square', 42)
    with pytest.raises(TypeError):
        random_forest(100, 'classification', 20, 2, 'sqrt', 42.5)

def test_random_forest_input_values():
    with pytest.raises(ValueError):
        random_forest(-100, 'classification', 20, 2, 'sqrt', 42)
    with pytest.raises(ValueError):
        random_forest(100, 'classify', 20, 2, 'sqrt', 42)
    with pytest.raises(ValueError):
        random_forest(100, 'classification', -20, 2, 'sqrt', 42)
    with pytest.raises(ValueError):
        random_forest(100, 'classification', 20, -2, 'sqrt', 42)
    with pytest.raises(ValueError):
        random_forest(100, 'classification', 20, -2, -1, 42)

def test_bootstrap_data_basic():
    test_array = np.array([[1], [2], [3], [4]])
    test_targets = np.array([10, 20, 30, 40])
    rf = random_forest(100, 'classification', random_state = 42)
    bootstrap_array, bootstrap_targets = rf._bootstrap_data(test_array, test_targets)
    assert isinstance(bootstrap_array, np.ndarray)
    assert isinstance(bootstrap_targets, np.ndarray)
    assert bootstrap_array.shape == test_array.shape
    assert bootstrap_targets.shape == test_targets.shape

def test_bootstrap_data_replacement():
    test_array = np.array([[1], [2], [3]])
    test_targets = np.array([10, 20, 30])
    rf = random_forest(100, 'classification', random_state = 72)
    bootstrap_array, bootstrap_targets = rf._bootstrap_data(test_array, test_targets)
    assert len(set(bootstrap_array.flatten())) < 3
    assert len(set(bootstrap_targets.flatten())) < 3

def test_bootstrap_data_random_state():
    test_array = np.array([[1], [2], [3]])
    test_targets = np.array([10, 20, 30])
    rf_1 = random_forest(100, 'classification', random_state = 42)
    bootstrap_array_1, bootstrap_targets_1 = rf_1._bootstrap_data(test_array, test_targets)
    rf_2 = random_forest(100, 'classification', random_state = 42)
    bootstrap_array_2, bootstrap_targets_2 = rf_2._bootstrap_data(test_array, test_targets)
    assert np.allclose(bootstrap_array_1, bootstrap_array_2)
    assert np.allclose(bootstrap_targets_1, bootstrap_targets_2)

def test_bootstrap_data_string_targets():
    test_array = np.array([[1], [2], [3]])
    test_targets = np.array(['A', 'B', 'C'])
    rf = random_forest(100, 'classification', random_state = 42)
    bootstrap_array, bootstrap_targets = rf._bootstrap_data(test_array, test_targets)
    assert set(bootstrap_targets).issubset({'A', 'B', 'C'})

def test_feature_selection_basic_int():
    rf = random_forest(100, 'classification', max_features = 2)
    test_features = rf._feature_selection(10)
    assert isinstance(test_features, np.ndarray)
    assert test_features.shape == (2,)

def test_feature_selection_basic_float():
    rf = random_forest(100, 'classification', max_features = 0.2)
    test_features = rf._feature_selection(10)
    assert isinstance(test_features, np.ndarray)
    assert test_features.shape == (2,)

def test_feature_selection_basic_sqrt():
    rf = random_forest(100, 'classification', max_features = 'sqrt')
    test_features = rf._feature_selection(10)
    assert isinstance(test_features, np.ndarray)
    assert test_features.shape == (3,)

def test_feature_selection_basic_log2():
    rf = random_forest(100, 'classification', max_features = 'log2')
    test_features = rf._feature_selection(10)
    assert isinstance(test_features, np.ndarray)
    assert test_features.shape == (3,)

def test_feature_selection_type_input():
    rf = random_forest(100, 'classification', max_features = 'log2')
    with pytest.raises(TypeError):
        rf._feature_selection(10.5)

def test_feature_selection_input_values():
    rf = random_forest(100, 'classification', max_features = 5)
    with pytest.raises(ValueError):
        rf._feature_selection(4)
    rf = random_forest(100, 'classification', max_features = 1.2)
    with pytest.raises(ValueError):
        rf._feature_selection(4)

def test_feature_selection_random_state():
    rf_1 = random_forest(100, 'classification', max_features = 2, random_state = 42)
    test_features_1 = rf_1._feature_selection(10)
    rf_2 = random_forest(100, 'classification', max_features = 2, random_state = 42)
    test_features_2 = rf_2._feature_selection(10)
    assert np.allclose(test_features_1, test_features_2)

def test_feature_selection_small_prop():
    rf = random_forest(100, 'classification', max_features = 0.01, random_state = 42)
    test_features = rf._feature_selection(10)
    assert test_features.shape[0] > 0

def test_feature_selection_one_feature():
    rf = random_forest(100, 'classification', max_features = 0.6, random_state = 42)
    test_features = rf._feature_selection(1)
    assert test_features.shape[0] > 0

def test_feature_selection_none():
    rf = random_forest(100, 'classification', max_features = None, random_state = 42)
    test_features = rf._feature_selection(10)
    assert test_features.shape[0] == 10

def test_feature_selection_index_values():
    rf = random_forest(100, 'classification', max_features = None, random_state = 42)
    test_features = rf._feature_selection(10)
    assert all(0 <= feature < 10 for feature in test_features)

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

def test_verify_fit_basic():
    test_array = np.array([[1], [2]])
    test_targets = np.array([0, 0])
    rf = random_forest(100, 'classification')
    rf.fit(test_array, test_targets)
    rf._verify_fit()

def test_verify_fit_unfit():
    rf = random_forest(100, 'classification')
    with pytest.raises(RuntimeError):
        rf._verify_fit()

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


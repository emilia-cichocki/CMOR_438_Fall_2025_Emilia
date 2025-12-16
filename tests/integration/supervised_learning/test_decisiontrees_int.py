
import numpy as np
import pandas as pd
import pytest
from rice_ml.supervised_learning.decisiontrees import decision_tree, regression_tree, Node

def test_decision_tree_fit_basic():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1])
    decision = decision_tree(100, 2)
    decision.fit(test_array, test_targets)
    assert decision.tree is not None
    assert isinstance(decision.tree, Node)
    assert isinstance(decision._class_mappings, dict)
    assert decision._class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._reverse_class_mappings, dict)
    assert decision._reverse_class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._n_features, int)
    assert decision._n_features == 2

def test_decision_tree_fit_string_targets():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array(['A', 'B', 'A', 'B'])
    decision = decision_tree(100, 2)
    decision.fit(test_array, test_targets)
    assert decision.tree is not None
    assert isinstance(decision.tree, Node)
    assert isinstance(decision._class_mappings, dict)
    assert decision._class_mappings == {'A': 0, 'B': 1}
    assert isinstance(decision._reverse_class_mappings, dict)
    assert decision._reverse_class_mappings == {0: 'A', 1: 'B'}
    assert isinstance(decision._n_features, int)
    assert decision._n_features == 2

def test_decision_tree_fit_basic_list():
    test_array = [[0, 1], [1, 0], [0, 0], [1, 1]]
    test_targets = np.array([0, 1, 0, 1])
    decision = decision_tree(100, 2)
    decision.fit(test_array, test_targets)
    assert decision.tree is not None
    assert isinstance(decision.tree, Node)
    assert isinstance(decision._class_mappings, dict)
    assert decision._class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._reverse_class_mappings, dict)
    assert decision._reverse_class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._n_features, int)
    assert decision._n_features == 2

def test_decision_tree_fit_basic_tuple():
    test_array = ([0, 1], [1, 0], [0, 0], [1, 1])
    test_targets = np.array([0, 1, 0, 1])
    decision = decision_tree(100, 2)
    decision.fit(test_array, test_targets)
    assert decision.tree is not None
    assert isinstance(decision.tree, Node)
    assert isinstance(decision._class_mappings, dict)
    assert decision._class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._reverse_class_mappings, dict)
    assert decision._reverse_class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._n_features, int)
    assert decision._n_features == 2

def test_decision_tree_fit_basic_targets_list():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = [0, 1, 0, 1]
    decision = decision_tree(100, 2)
    decision.fit(test_array, test_targets)
    assert decision.tree is not None
    assert isinstance(decision.tree, Node)
    assert isinstance(decision._class_mappings, dict)
    assert decision._class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._reverse_class_mappings, dict)
    assert decision._reverse_class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._n_features, int)
    assert decision._n_features == 2

def test_decision_tree_fit_basic_targets_tuple():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = ([0, 1, 0, 1])
    decision = decision_tree(100, 2)
    decision.fit(test_array, test_targets)
    assert decision.tree is not None
    assert isinstance(decision.tree, Node)
    assert isinstance(decision._class_mappings, dict)
    assert decision._class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._reverse_class_mappings, dict)
    assert decision._reverse_class_mappings == {0: 0, 1: 1}
    assert isinstance(decision._n_features, int)
    assert decision._n_features == 2

def test_decision_tree_fit_testing_nan():
    test_array = ([0, 1], [1, 0], [0, 0], [1, 1])
    test_targets = np.array([0, np.nan, 0, np.nan])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision.fit(test_array, test_targets)

def test_decision_tree_fit_dimension_input_array():
    test_array = np.array([[[0, 1], [1, 0], [0, 0], [1, 1]]])
    test_targets = np.array([0, 1, 0, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision.fit(test_array, test_targets)

def test_decision_tree_fit_dimension_input_targets():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([[0, 1, 0, 1]])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision.fit(test_array, test_targets)

def test_decision_tree_fit_shape_mismatch():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1, 0])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision.fit(test_array, test_targets)

def test_verify_fit_basic():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1])
    decision = decision_tree(100, 2)
    decision.fit(test_array, test_targets)
    decision._verify_fit()

def test_verify_fit_unfit():
    decision = decision_tree(100, 2)
    with pytest.raises(RuntimeError):
        decision._verify_fit()

def test_predict_basic():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1], [7], [3]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array([0, 1, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)
    assert all(value is not None for value in test_pred)

def test_predict_mult_feature():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [11, 12]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1, 0], [7, 10], [3, 3]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array([0, 1, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_predict_mult_class():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [15, 20]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[1, 0], [7, 10], [16, 21]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array([0, 1, 2])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_predict_one_class():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [15, 20]])
    train_targets = np.array([0, 0, 0, 0])
    test_array = np.array([[1, 0], [7, 10], [16, 21]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array([0, 0, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_predict_one_sample():
    train_array = np.array([[0]])
    train_targets = np.array([0])
    test_array = np.array([[2], [1]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array([0, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (2,)
    assert np.array_equal(test_pred, expected_pred)

def test_predict_type_targets_string_numeric():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [11, 12]])
    train_targets = np.array(['0', '0', '1', '1'])
    test_array = np.array([[1, 0], [7, 10], [3, 3]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array(['0', '1', '0'])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_predict_type_targets_string_nonnumeric():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [11, 12]])
    train_targets = np.array(['A', 'A', 'B', 'B'])
    test_array = np.array([[1, 0], [7, 10], [3, 3]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array(['A', 'B', 'A'])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_predict_type_input():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([['1'], ['7'], ['3']])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    test_pred = decision.predict(test_array)
    expected_pred = np.array([0, 1, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_predict_type_input_string():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([['A'], ['B'], ['C']])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    with pytest.raises(TypeError):
        decision.predict(test_array)

def test_predict_dimension_input():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([0])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    with pytest.raises(ValueError):
        decision.predict(test_array)

def test_predict_shape_mismatch():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[0, 1]])
    decision = decision_tree(100, 2)
    decision.fit(train_array, train_targets)
    with pytest.raises(ValueError):
        decision.predict(test_array)

def test_predict_unfit():
    test_array = np.array([[0, 1]])
    decision = decision_tree(100, 2)
    with pytest.raises(RuntimeError):
        decision.predict(test_array)


def test_regression_tree_fit_basic():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1])
    regression = regression_tree(100, 2)
    regression.fit(test_array, test_targets)
    assert regression.tree is not None
    assert isinstance(regression.tree, Node)
    assert isinstance(regression._n_features, int)
    assert regression._n_features == 2

def test_regression_tree_fit_basic_list():
    test_array = [[0, 1], [1, 0], [0, 0], [1, 1]]
    test_targets = np.array([0, 1, 0, 1])
    regression = regression_tree(100, 2)
    regression.fit(test_array, test_targets)
    assert regression.tree is not None
    assert isinstance(regression.tree, Node)
    assert isinstance(regression._n_features, int)
    assert regression._n_features == 2

def test_regression_tree_fit_basic_tuple():
    test_array = ([0, 1], [1, 0], [0, 0], [1, 1])
    test_targets = np.array([0, 1, 0, 1])
    regression = regression_tree(100, 2)
    regression.fit(test_array, test_targets)
    assert regression.tree is not None
    assert isinstance(regression.tree, Node)
    assert isinstance(regression._n_features, int)
    assert regression._n_features == 2

def test_regression_tree_fit_string_targets():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array(['A', 'B', 'A', 'B'])
    regression = regression_tree(100, 2)
    with pytest.raises(TypeError):
        regression.fit(test_array, test_targets)

def test_regression_tree_fit_basic_targets_list():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = [0, 1, 0, 1]
    regression = regression_tree(100, 2)
    regression.fit(test_array, test_targets)
    assert regression.tree is not None
    assert isinstance(regression.tree, Node)
    assert isinstance(regression._n_features, int)
    assert regression._n_features == 2

def test_regression_tree_fit_basic_targets_tuple():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = ([0, 1, 0, 1])
    regression = regression_tree(100, 2)
    regression.fit(test_array, test_targets)
    assert regression.tree is not None
    assert isinstance(regression.tree, Node)
    assert isinstance(regression._n_features, int)
    assert regression._n_features == 2

def test_regression_tree_fit_testing_nan():
    test_array = ([0, 1], [1, 0], [0, 0], [1, 1])
    test_targets = np.array([0, np.nan, 0, np.nan])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression.fit(test_array, test_targets)

def test_regression_tree_fit_dimension_input_array():
    test_array = np.array([[[0, 1], [1, 0], [0, 0], [1, 1]]])
    test_targets = np.array([0, 1, 0, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression.fit(test_array, test_targets)

def test_regression_tree_fit_dimension_input_targets():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([[0, 1, 0, 1]])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression.fit(test_array, test_targets)

def test_regression_tree_fit_shape_mismatch():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1, 0])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression.fit(test_array, test_targets)


def test_regression_predict_recursive_basic_leaf():
    test_array = np.array([0, 0, 1])
    leaf = Node()
    leaf.value = 42
    regression = regression_tree(100, 2)
    regression._predict_recursive(test_array, leaf)
    assert isinstance(leaf.value, int)
    assert leaf.value == 42

def test_regression_predict_recursive_basic_split():
    test_array_left = np.array([0])
    test_array_right = np.array([2])
    left = Node()
    left.value = 0
    right = Node()
    right.value = 1
    node = Node(threshold_value = 0.5, left = left, right = right)
    regression = regression_tree(100, 2)
    test_left = regression._predict_recursive(test_array_left, node)
    test_right = regression._predict_recursive(test_array_right, node)
    assert isinstance(test_left, int)
    assert isinstance(test_left, int)
    assert test_left == 0
    assert test_right == 1

def test_regression_predict_recursive_basic_type_input():
    test_array_left = np.array([0])
    test_array_right = np.array([2])
    left = Node()
    left.value = 0
    right = Node()
    right.value = 1.5
    node = Node(threshold_value = 0.5, left = left, right = right)
    regression = regression_tree(100, 2)
    test_left = regression._predict_recursive(test_array_left, node)
    test_right = regression._predict_recursive(test_array_right, node)
    assert isinstance(test_left, int)
    assert isinstance(test_left, int)
    assert test_left == 0
    assert test_right == 1.5

def test_regression_predict_basic():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1], [7], [3]])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    test_pred = regression.predict(test_array)
    expected_pred = np.array([0, 1, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)
    assert all(value is not None for value in test_pred)

def test_regression_predict_mult_feature():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [11, 12]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[1, 0], [7, 10], [3, 3]])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    test_pred = regression.predict(test_array)
    expected_pred = np.array([0, 1, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_regression_predict_mult_class():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [15, 20]])
    train_targets = np.array([0, 0, 1, 2])
    test_array = np.array([[1, 0], [7, 10], [16, 21]])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    test_pred = regression.predict(test_array)
    expected_pred = np.array([0, 1, 2])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_regression_predict_one_class():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [15, 20]])
    train_targets = np.array([0, 0, 0, 0])
    test_array = np.array([[1, 0], [7, 10], [16, 21]])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    test_pred = regression.predict(test_array)
    expected_pred = np.array([0, 0, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_regression_predict_one_sample():
    train_array = np.array([[0]])
    train_targets = np.array([0])
    test_array = np.array([[2], [1]])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    test_pred = regression.predict(test_array)
    expected_pred = np.array([0, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (2,)
    assert np.array_equal(test_pred, expected_pred)

def test_regression_predict_type_targets_string_numeric():
    train_array = np.array([[0, 1], [1, 0], [10, 11], [11, 12]])
    train_targets = np.array(['0', '0', '1', '1'])
    test_array = np.array([[1, 0], [7, 10], [3, 3]])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    test_pred = regression.predict(test_array)
    expected_pred = np.array([0, 1, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_regression_predict_type_input():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([['1'], ['7'], ['3']])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    test_pred = regression.predict(test_array)
    expected_pred = np.array([0, 1, 0])
    assert isinstance(test_pred, np.ndarray)
    assert test_pred.shape == (3,)
    assert np.array_equal(test_pred, expected_pred)

def test_regression_predict_type_input_string():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([['A'], ['B'], ['C']])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    with pytest.raises(TypeError):
        regression.predict(test_array)

def test_regression_predict_dimension_input():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([0])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    with pytest.raises(ValueError):
        regression.predict(test_array)

def test_regression_predict_shape_mismatch():
    train_array = np.array([[1], [2], [10], [11]])
    train_targets = np.array([0, 0, 1, 1])
    test_array = np.array([[0, 1]])
    regression = regression_tree(100, 2)
    regression.fit(train_array, train_targets)
    with pytest.raises(ValueError):
        regression.predict(test_array)

def test_regression_predict_unfit():
    test_array = np.array([[0, 1]])
    regression = regression_tree(100, 2)
    with pytest.raises(RuntimeError):
        regression.predict(test_array)
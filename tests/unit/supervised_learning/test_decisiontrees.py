
import numpy as np
import pandas as pd
import pytest
from rice_ml.supervised_learning.decisiontrees import _validate_parameters, _validate_parameters_node, _entropy, Node, decision_tree, regression_tree

def test_validate_parameters_basic():
    _validate_parameters(100, 2)

def test_validate_parameters_none():
    _validate_parameters(100, None)
    _validate_parameters(None, 2)
    _validate_parameters(None, None)

def test_validate_parameters_type_inputs():
    with pytest.raises(TypeError):
        _validate_parameters('100', 2)
    with pytest.raises(TypeError):
        _validate_parameters(100, '2')

def test_validate_parameters_input_values():
    with pytest.raises(ValueError):
        _validate_parameters(-1, 2)
    with pytest.raises(ValueError):
        _validate_parameters(100, -2)

def test_validate_parameters_node_basic():
    left = Node(2, 0.5)
    right = Node(2, 0.5)
    _validate_parameters_node(2, 1, None, None)
    _validate_parameters_node(2, 1.0, None, None)
    _validate_parameters_node(2, 1, left, right)

def test_validate_parameters_node_type_inputs():
    with pytest.raises(TypeError):
        _validate_parameters_node('2', 1, None, None)
    with pytest.raises(TypeError):
        _validate_parameters_node(2, '1', None, None)
    with pytest.raises(TypeError):
        _validate_parameters_node(2, 1, 'None', None)
    with pytest.raises(TypeError):
        _validate_parameters_node(2, 1, None, 'None')

def test_validate_parameters_node_input_values():
    with pytest.raises(ValueError):
        _validate_parameters_node(-1, 1, None, None)

def test_entropy_basic():
    test_array = np.array([0, 0, 1, 1])
    entropy = _entropy(test_array)
    assert isinstance(entropy, float)
    assert np.isclose(entropy, 1)

def test_entropy_zero():
    test_array = np.array([0, 0, 0, 0])
    entropy = _entropy(test_array)
    assert isinstance(entropy, float)
    assert np.isclose(entropy, 0)

def test_entropy_one_value():
    test_array = np.array([0])
    entropy = _entropy(test_array)
    assert isinstance(entropy, float)
    assert np.isclose(entropy, 0)

def test_entropy_no_values():
    test_array = np.array([])
    with pytest.raises(ValueError):
        _entropy(test_array)

def test_entropy_data_type():
    test_array = np.array(['A', 'A', 'B', 'B'])
    entropy = _entropy(test_array)
    assert isinstance(entropy, float)
    assert np.isclose(entropy, 1)

def test_node_init_basic():
    node = Node(2, 0.5)
    assert node.feature_index == 2
    assert node.threshold == 0.5
    assert node.left is None
    assert node.right is None
    assert node.value is None

def test_node_init_basic_nodes():
    left = Node(2, 0.5)
    right = Node(2, 0.5)
    node = Node(2, 0.5, left, right)
    assert node.feature_index == 2
    assert node.threshold == 0.5
    assert node.left == left
    assert node.right == right
    assert node.value is None

def test_node_init_type_inputs():
    with pytest.raises(TypeError):
        Node('2', 0.5, None, None)
    with pytest.raises(TypeError):
        Node(2, '0.5', None, None)
    with pytest.raises(TypeError):
        Node(2, 0.5, 'None', None)
    with pytest.raises(TypeError):
        Node(2, 0.5, None, 'None')

def test_node_init_input_values():
    with pytest.raises(ValueError):
        Node(-2, 0.5, None, None)

def test_node_is_leaf_basic_false():
    left = Node(2, 0.5)
    right = Node(2, 0.5)
    node = Node(2, 0.5, left, right)
    assert node.is_leaf() is False

def test_node_is_leaf_basic_true():
    left = Node(2, 0.5)
    right = Node(2, 0.5)
    node = Node(2, 0.5, left, right, 1)
    assert node.is_leaf() is True

def test_decision_tree_init_basic():
    decision = decision_tree(100, 2)
    assert decision.max_depth == 100
    assert decision.min_samples_split == 2
    assert decision.tree is None
    assert decision._class_mappings is None
    assert decision._reverse_class_mappings is None
    assert decision._n_features is None

def test_decision_tree_init_type_inputs():
    with pytest.raises(TypeError):
        decision_tree('100', 2)
    with pytest.raises(TypeError):
        decision_tree(100, '2')

def test_decision_tree_init_input_values():
    with pytest.raises(ValueError):
        decision_tree(-1, 2)

def test_information_gain_basic():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([0, 0])
    test_right = np.array([1, 1])
    decision = decision_tree(100, 2)
    test_gain = decision._information_gain(test_parent, test_left, test_right)
    assert isinstance(test_gain, float)
    assert np.isclose(test_gain, 1.0)

def test_information_gain_basic_poor_gain():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([1, 0])
    test_right = np.array([0, 1])
    decision = decision_tree(100, 2)
    test_gain = decision._information_gain(test_parent, test_left, test_right)
    assert isinstance(test_gain, float)
    assert np.isclose(test_gain, 0.0)

def test_information_gain_basic_multiple_class():
    test_parent = np.array([0, 0, 1, 1, 2])
    test_left = np.array([0, 0])
    test_right = np.array([1, 1, 2])
    decision = decision_tree(100, 2)
    test_gain = decision._information_gain(test_parent, test_left, test_right)
    assert isinstance(test_gain, float)
    assert np.isclose(test_gain, 0.9709505944546685)

def test_information_gain_basic_empty():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([])
    test_right = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._information_gain(test_parent, test_left, test_right)

def test_information_gain_dimension_parent():
    test_parent = np.array([[0, 0, 1, 1]])
    test_left = np.array([0, 0])
    test_right = np.array([1, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._information_gain(test_parent, test_left, test_right)

def test_information_gain_dimension_left():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([[0, 0]])
    test_right = np.array([1, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._information_gain(test_parent, test_left, test_right)

def test_information_gain_dimension_right():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([0, 0])
    test_right = np.array([[1, 1]])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._information_gain(test_parent, test_left, test_right)

def test_information_gain_type_input_strings():
    test_parent = np.array(['A', 'A', 'B', 'B'])
    test_left = np.array(['A', 'A'])
    test_right = np.array(['B', 'B'])
    decision = decision_tree(100, 2)
    test_gain = decision._information_gain(test_parent, test_left, test_right)
    assert isinstance(test_gain, float)
    assert np.isclose(test_gain, 1)

def test_information_gain_split_count():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([0, 0, 1])
    test_right = np.array([1, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._information_gain(test_parent, test_left, test_right)

def test_best_split_basic():
    test_array = np.array([[0], [0], [1], [1]])
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    feature, threshold = decision._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature == 0
    assert 0 <= threshold <= 1

def test_best_split_mult_features():
    test_array = np.array([[0, 10], [0, 20], [1, 10], [1, 20]])
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    feature, threshold = decision._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature in [0, 1]
    assert threshold == 0.5 or threshold == 15

def test_best_split_one_feature():
    test_array = np.array([[0], [0], [0], [0]])
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    feature, threshold = decision._best_split(test_array, test_targets)
    assert feature is None
    assert threshold is None

def test_best_split_one_sample():
    test_array = np.array([[0]])
    test_targets = np.array([0])
    decision = decision_tree(100, 2)
    feature, threshold = decision._best_split(test_array, test_targets)
    assert feature is None
    assert threshold is None

def test_best_split_type_input_array():
    test_array = np.array([['0', '0'], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    feature, threshold = decision._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature == 0
    assert 0 <= threshold <= 1

def test_best_split_type_input_array_nonnumeric():
    test_array = np.array([['A', 'A'], ['A', 'B'], ['B', 'B'], ['B', 'C']])
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(TypeError):
        decision._best_split(test_array, test_targets)

def test_best_split_type_input_array_nan():
    test_array = np.array([[np.nan, np.nan], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._best_split(test_array, test_targets)

def test_best_split_type_input_targets():
    test_array = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array(['0', '0', 1, 1])
    decision = decision_tree(100, 2)
    feature, threshold = decision._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature == 0
    assert 0 <= threshold <= 1

def test_best_split_type_input_targets_nonnumeric():
    test_array = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array(['A', 'A', 'B', 'B'])
    decision = decision_tree(100, 2)
    feature, threshold = decision._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature == 0
    assert 0 <= threshold <= 1

def test_best_split_dimension_input_array():
    test_array = np.array([[[0, 0], [0, 1], [1, 1], [1, 2]]])
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._best_split(test_array, test_targets)

def test_best_split_dimension_input_targets():
    test_array = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array([[0, 0, 1, 1]])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._best_split(test_array, test_targets)

def test_leaf_value_basic():
    test_targets = np.array([1, 1, 2, 1, 3])
    decision = decision_tree(100, 2)
    test_class = decision._leaf_value(test_targets)
    assert isinstance(test_class, np.int64)
    assert test_class == 1

def test_leaf_value_basic_float():
    test_targets = np.array([1.5, 1.5, 2.5, 1.5, 3.5])
    decision = decision_tree(100, 2)
    test_class = decision._leaf_value(test_targets)
    assert isinstance(test_class, float)
    assert test_class == 1.5

def test_leaf_value_basic_strings():
    test_targets = np.array(['A', 'A', 'B', 'A', 'C'])
    decision = decision_tree(100, 2)
    test_class = decision._leaf_value(test_targets)
    assert isinstance(test_class, str)
    assert test_class == 'A'

def test_leaf_value_basic_single():
    test_targets = np.array(['A'])
    decision = decision_tree(100, 2)
    test_class = decision._leaf_value(test_targets)
    assert isinstance(test_class, str)
    assert test_class == 'A'

def test_leaf_value_ties():
    test_targets = np.array([0, 0, 1, 1])
    decision = decision_tree(100, 2)
    test_class = decision._leaf_value(test_targets)
    assert isinstance(test_class, np.int64)
    assert test_class in [0, 1]

def test_leaf_value_dimensions():
    test_targets = np.array([[0, 0, 1, 1]])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._leaf_value(test_targets)

def test_build_tree_basic():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1])
    decision = decision_tree(100, 2)
    root = decision._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert isinstance(root.left, Node)
    assert root.right is not None
    assert isinstance(root.right, Node)

def test_build_tree_leaf():
    test_array = np.array([[1],[2]])
    test_targets = np.array([0, 1])
    decision = decision_tree(100, 3)
    root = decision._build_tree(test_array, test_targets, 0)
    assert root.value in test_targets
    assert root.left is None and root.right is None

def test_build_tree_maximum_depth():
    test_array = np.array([[1],[2]])
    test_targets = np.array([0, 1])
    decision = decision_tree(1, 3)
    root = decision._build_tree(test_array, test_targets, 2)
    assert root.value in test_targets
    assert root.left is None and root.right is None

def test_build_tree_leaf_pure():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array([0, 0, 0])
    decision = decision_tree(100, 2)
    root = decision._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.value == 0
    assert root.left is None and root.right is None

def test_build_tree_leaf_type_input_array():
    test_array = np.array([['0'], ['1'], ['3']])
    test_targets = np.array([0, 0, 1])
    decision = decision_tree(100, 2)
    root = decision._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert isinstance(root.left, Node)
    assert root.right is not None
    assert isinstance(root.right, Node)

def test_build_tree_leaf_type_input_array_nonnumeric():
    test_array = np.array([['A'], ['B'], ['C']])
    test_targets = np.array([0, 0, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(TypeError):
        decision._build_tree(test_array, test_targets, 0)

def test_build_tree_leaf_type_input_targets():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array(['0', '0', '1'])
    decision = decision_tree(100, 2)
    root = decision._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert isinstance(root.left, Node)
    assert root.right is not None
    assert isinstance(root.right, Node)

def test_build_tree_leaf_type_input_targets_nonnumeric():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array(['A', 'A', 'B'])
    decision = decision_tree(100, 2)
    root = decision._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert isinstance(root.left, Node)
    assert root.right is not None
    assert isinstance(root.right, Node)

def test_build_tree_leaf_dimension_input_array():
    test_array = np.array([[[0], [1], [3]]])
    test_targets = np.array([0, 0, 1])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._build_tree(test_array, test_targets, 0)

def test_build_tree_leaf_dimension_input_target():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array([[0, 0, 1]])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._build_tree(test_array, test_targets, 0)

def test_build_tree_leaf_shape_mismatch():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array([[0, 0, 1, 0]])
    decision = decision_tree(100, 2)
    with pytest.raises(ValueError):
        decision._build_tree(test_array, test_targets, 0)
    
def test_predict_recursive_basic_leaf():
    test_array = np.array([0, 0, 1])
    leaf = Node()
    leaf.value = 42
    decision = decision_tree(100, 2)
    decision._predict_recursive(test_array, leaf)
    assert isinstance(leaf.value, int)
    assert leaf.value == 42

def test_predict_recursive_basic_split():
    test_array_left = np.array([0])
    test_array_right = np.array([2])
    left = Node()
    left.value = 0
    right = Node()
    right.value = 1
    node = Node(threshold_value = 0.5, left = left, right = right)
    decision = decision_tree(100, 2)
    test_left = decision._predict_recursive(test_array_left, node)
    test_right = decision._predict_recursive(test_array_right, node)
    assert isinstance(test_left, int)
    assert isinstance(test_left, int)
    assert test_left == 0
    assert test_right == 1

def test_predict_recursive_basic_type_input():
    test_array_left = np.array([0])
    test_array_right = np.array([2])
    left = Node()
    left.value = 0
    right = Node()
    right.value = 1.5
    node = Node(threshold_value = 0.5, left = left, right = right)
    decision = decision_tree(100, 2)
    test_left = decision._predict_recursive(test_array_left, node)
    test_right = decision._predict_recursive(test_array_right, node)
    assert isinstance(test_left, int)
    assert isinstance(test_left, int)
    assert test_left == 0
    assert test_right == 1.5

def test_predict_recursive_basic_type_input_string():
    test_array_left = np.array([0])
    test_array_right = np.array([2])
    left = Node()
    left.value = 'B'
    right = Node()
    right.value = 'A'
    node = Node(threshold_value = 0.5, left = left, right = right)
    decision = decision_tree(100, 2)
    test_left = decision._predict_recursive(test_array_left, node)
    test_right = decision._predict_recursive(test_array_right, node)
    assert isinstance(test_left, str)
    assert isinstance(test_left, str)
    assert test_left == 'B'
    assert test_right == 'A'

def test_regression_tree_init_basic():
    regression = regression_tree(100, 2)
    assert regression.max_depth == 100
    assert regression.min_samples_split == 2
    assert regression.tree is None
    assert regression._n_features is None

def test_regression_tree_init_type_inputs():
    with pytest.raises(TypeError):
        regression_tree('100', 2)
    with pytest.raises(TypeError):
        regression_tree(100, '2')

def test_regression_tree_init_input_values():
    with pytest.raises(ValueError):
        regression_tree(-1, 2)

def test_variance_reduction_basic():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([0, 0])
    test_right = np.array([1, 1])
    regression = regression_tree(100, 2)
    test_var_red = regression._variance_reduction(test_parent, test_left, test_right)
    assert isinstance(test_var_red, float)
    assert np.isclose(test_var_red, 0.25)

def test_variance_reduction_basic_poor():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([1, 0])
    test_right = np.array([0, 1])
    regression = regression_tree(100, 2)
    test_var_red = regression._variance_reduction(test_parent, test_left, test_right)
    assert isinstance(test_var_red, float)
    assert np.isclose(test_var_red, 0.0)

def test_variance_reduction_basic_multiple_class():
    test_parent = np.array([0, 0, 1, 1, 2])
    test_left = np.array([0, 0])
    test_right = np.array([1, 1, 2])
    regression = regression_tree(100, 2)
    test_var_red = regression._variance_reduction(test_parent, test_left, test_right)
    assert isinstance(test_var_red, float)
    assert np.isclose(test_var_red, 0.42666666666666675)

def test_variance_reduction_basic_empty():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([])
    test_right = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._variance_reduction(test_parent, test_left, test_right)

def test_variance_reduction_type_input_strings():
    test_parent = np.array(['A', 'A', 'B', 'B'])
    test_left = np.array(['A', 'A'])
    test_right = np.array(['B', 'B'])
    regression = regression_tree(100, 2)
    with pytest.raises(TypeError):
        regression._variance_reduction(test_parent, test_left, test_right)

def test_variance_reduction_dimension_parent():
    test_parent = np.array([[0, 0, 1, 1]])
    test_left = np.array([0, 0])
    test_right = np.array([1, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._variance_reduction(test_parent, test_left, test_right)

def test_variance_reduction_dimension_left():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([[0, 0]])
    test_right = np.array([1, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._variance_reduction(test_parent, test_left, test_right)

def test_variance_reduction_dimension_right():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([0, 0])
    test_right = np.array([[1, 1]])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._variance_reduction(test_parent, test_left, test_right)

def test_variance_reduction_split_count():
    test_parent = np.array([0, 0, 1, 1])
    test_left = np.array([0, 0, 1])
    test_right = np.array([1, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._variance_reduction(test_parent, test_left, test_right)

def test_regression_best_split_basic():
    test_array = np.array([[0], [0], [1], [1]])
    test_targets = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    feature, threshold = regression._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature == 0
    assert 0 <= threshold <= 1

def test_regression_best_split_mult_features():
    test_array = np.array([[0, 10], [0, 20], [1, 10], [1, 20]])
    test_targets = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    feature, threshold = regression._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature in [0, 1]
    assert threshold == 0.5 or threshold == 15

def test_regression_best_split_one_feature():
    test_array = np.array([[0], [0], [0], [0]])
    test_targets = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    feature, threshold = regression._best_split(test_array, test_targets)
    assert feature is None
    assert threshold is None

def test_regression_best_split_one_sample():
    test_array = np.array([[0]])
    test_targets = np.array([0])
    regression = regression_tree(100, 2)
    feature, threshold = regression._best_split(test_array, test_targets)
    assert feature is None
    assert threshold is None

def test_regression_best_split_type_input_array():
    test_array = np.array([['0', '0'], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    feature, threshold = regression._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature == 0
    assert 0 <= threshold <= 1

def test_regression_best_split_type_input_array_nonnumeric():
    test_array = np.array([['A', 'A'], ['A', 'B'], ['B', 'B'], ['B', 'C']])
    test_targets = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(TypeError):
        regression._best_split(test_array, test_targets)

def test_regression_best_split_type_input_array_nan():
    test_array = np.array([[np.nan, np.nan], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._best_split(test_array, test_targets)

def test_regression_best_split_type_input_targets():
    test_array = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array(['0', '0', 1, 1])
    regression = regression_tree(100, 2)
    feature, threshold = regression._best_split(test_array, test_targets)
    assert isinstance(feature, int)
    assert isinstance(threshold, float)
    assert feature == 0
    assert 0 <= threshold <= 1

def test_regression_best_split_type_input_targets_nonnumeric():
    test_array = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array(['A', 'A', 'B', 'B'])
    regression = regression_tree(100, 2)
    with pytest.raises(TypeError):
        regression._best_split(test_array, test_targets)

def test_regression_best_split_dimension_input_array():
    test_array = np.array([[[0, 0], [0, 1], [1, 1], [1, 2]]])
    test_targets = np.array([0, 0, 1, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._best_split(test_array, test_targets)

def test_regression_best_split_dimension_input_targets():
    test_array = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
    test_targets = np.array([[0, 0, 1, 1]])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._best_split(test_array, test_targets)
        
def test_regression_leaf_value_basic():
    test_targets = np.array([1, 1, 2, 1, 3])
    regression = regression_tree(100, 2)
    test_value = regression._leaf_value(test_targets)
    assert isinstance(test_value, float)
    assert test_value == 1.6

def test_regression_leaf_value_basic_float():
    test_targets = np.array([1.5, 1.5, 2.5, 1.5, 3.5])
    regression = regression_tree(100, 2)
    test_value = regression._leaf_value(test_targets)
    assert isinstance(test_value, float)
    assert test_value == 2.1

def test_regression_leaf_value_basic_strings():
    test_targets = np.array(['A', 'A', 'B', 'A', 'C'])
    regression = regression_tree(100, 2)
    with pytest.raises(TypeError):
        regression._leaf_value(test_targets)

def test_regression_leaf_value_basic_single():
    test_targets = np.array([1])
    regression = regression_tree(100, 2)
    test_value = regression._leaf_value(test_targets)
    assert isinstance(test_value, float)
    assert test_value == 1

def test_regression_leaf_value_dimensions():
    test_targets = np.array([[0, 0, 1, 1]])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._leaf_value(test_targets)

def test_regression_build_tree_basic():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1])
    regression = regression_tree(100, 2)
    root = regression._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert isinstance(root.left, Node)
    assert root.right is not None
    assert isinstance(root.right, Node)

def test_regression_build_tree_leaf():
    test_array = np.array([[1],[2]])
    test_targets = np.array([0, 1])
    regression = regression_tree(100, 3)
    root = regression._build_tree(test_array, test_targets, 0)
    assert root.value == 0.5
    assert root.left is None and root.right is None

def test_regression_build_tree_maximum_depth():
    test_array = np.array([[1],[2]])
    test_targets = np.array([0, 1])
    regression = regression_tree(1, 2)
    root = regression._build_tree(test_array, test_targets, 2)
    assert root.value == 0.5
    assert root.left is None and root.right is None

def test_regression_build_tree_leaf_pure():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array([0, 0, 0])
    regression = regression_tree(100, 2)
    root = regression._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.value == 0
    assert root.left is None and root.right is None

def test_regression_build_tree_leaf_type_input_array():
    test_array = np.array([['0'], ['1'], ['3']])
    test_targets = np.array([0, 0, 1])
    regression = regression_tree(100, 2)
    root = regression._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert isinstance(root.left, Node)
    assert root.right is not None
    assert isinstance(root.right, Node)

def test_regression_build_tree_leaf_type_input_array_nonnumeric():
    test_array = np.array([['A'], ['B'], ['C']])
    test_targets = np.array([0, 0, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(TypeError):
        regression._build_tree(test_array, test_targets, 0)

def test_regression_build_tree_leaf_type_input_targets():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array(['0', '0', '1'])
    regression = regression_tree(100, 2)
    root = regression._build_tree(test_array, test_targets, 0)
    assert isinstance(root, Node)
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert isinstance(root.left, Node)
    assert root.right is not None
    assert isinstance(root.right, Node)

def test_regression_build_tree_leaf_type_input_targets_nonnumeric():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array(['A', 'A', 'B'])
    regression = regression_tree(100, 2)
    with pytest.raises(TypeError):
        regression._build_tree(test_array, test_targets, 0)

def test_regression_build_tree_leaf_dimension_input_array():
    test_array = np.array([[[0], [1], [3]]])
    test_targets = np.array([0, 0, 1])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._build_tree(test_array, test_targets, 0)

def test_regression_build_tree_leaf_dimension_input_target():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array([[0, 0, 1]])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._build_tree(test_array, test_targets, 0)

def test_regression_build_tree_leaf_shape_mismatch():
    test_array = np.array([[0], [1], [3]])
    test_targets = np.array([[0, 0, 1, 0]])
    regression = regression_tree(100, 2)
    with pytest.raises(ValueError):
        regression._build_tree(test_array, test_targets, 0)
        
def test_regression_verify_fit_basic():
    test_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    test_targets = np.array([0, 1, 0, 1])
    regression = regression_tree(100, 2)
    regression.fit(test_array, test_targets)
    regression._verify_fit()

def test_regression_verify_fit_unfit():
    regression = regression_tree(100, 2)
    with pytest.raises(RuntimeError):
        regression._verify_fit()
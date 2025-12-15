
"""
    Decision and regression tree algorithms (NumPy)

    This module contains the classes implementing a decision tree and regression
    tree algorithm. It supports numeric input features for data, with either 
    categorical (decision trees) or numeric (decision trees, regression trees) target
    values. 

    It utilizes information gain and entropy for decision trees, as well
    as variance reduction for regression trees

    Classes
    ---------
    Node
        Class for a node in a tree
    decision_tree
        Decision tree classifier, based on entropy
    regression_tree
        Regression tree, based on variance reduction
"""

__all__ = [
    'decision_tree',
    'regression_tree'
]

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *
from rice_ml.supervised_learning.distances import _ensure_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters(max_depth: Optional[int], min_samples_split: Optional[int]) -> None:

    """
    Validates hyperparameters for a tree

    Parameters
    ----------
    max_depth: int, optional
        Maximum depth of the tree
    min_samples_split: int, optional
        Minimum number of samples required to split a node

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

    if max_depth is not None and not isinstance(max_depth, int):
        raise TypeError('Maximum depth must be an integer')
    if max_depth is not None and max_depth <= 0:
        raise ValueError('Maximum depth must be greater than zero')
    if min_samples_split is not None and not isinstance(min_samples_split, int):
        raise TypeError('Minimum samples required to split node must be an integer')
    if min_samples_split is not None and min_samples_split <= 0:
        raise ValueError('Minimum samples required to split node must be greater than zero')
    
def _validate_parameters_node(feature_index: Optional[int], 
                              threshold_value: Optional[float], 
                              left: Optional["Node"] = None, 
                              right: Optional["Node"] = None) -> None:

    """
    Validates hyperparameters for a node

    Parameters
    ----------
    feature_index: int, optional
        Index of feature used to split the node
    threshold_value: float, optional
        Value of feature threshold used for splits
    left: Node, optional
        Left child node
    right: Node, optional
        Right child node

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

    if feature_index is not None and not isinstance(feature_index, int):
        raise TypeError('Feature index must be an integer')
    if feature_index is not None and feature_index < 0:
        raise ValueError('Feature index must be greater than or equal to zero')
    if threshold_value is not None and not isinstance(threshold_value, (float, int)):
        raise TypeError('Threshold value must be a float')
    if left is not None and not isinstance(left, Node):
        raise TypeError('Left node must be an instance of the Node class')
    if right is not None and not isinstance(right, Node):
        raise TypeError('Right node must be an instance of the Node class')

def _entropy(train_targets: np.ndarray) -> float:

    """
    Calculates entropy for a target vector

    Parameters
    ----------
    train_targets: np.ndarray
        1D vector of targets

    Returns
    -------
    entropy: float
        Calculated entropy value
    """

    train_targets = _1D_vectorized(train_targets)
    _, counts = np.unique(train_targets, return_counts = True)
    probabilities = counts / np.sum(counts)
    probabilities_filtered = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities_filtered * np.log2(probabilities_filtered))

    return entropy


class Node():

    """
    Representation of a decision or regression tree node

    Nodes are either decision nodes used to create further splits in the data,
    or leaf nodes that have an associated prediction

    Attributes
    ----------
    feature_index: int, optional
        Index of feature used to split the node
    threshold_value: float, optional
        Value of feature threshold used for splits
    left: Node, optional
        Left child node
    right: Node, optional
        Right child node
    value: Any, optional
        Value of a leaf node

    Methods
    -------
    is_leaf():
        Returns the boolean for a node value
    """

    def __init__(self,
                 feature_index: Optional[int] = None,
                 threshold_value: Optional[float] = None,
                 left: "Node" = None,
                 right: "Node" = None,
                 value: Optional[Any] = None) -> None:
        
        """
        Creates associated attributes for a node with
        validated parameters

        Parameters
        ----------
        feature_index: int, optional
            Index of feature used to split the node
        threshold_value: float, optional
            Value of feature threshold used for splits
        left: Node, optional
            Left child node
        right: Node, optional
            Right child node
        value: Any, optional
            Value of a leaf node
        """

        _validate_parameters_node(feature_index, threshold_value, left, right)

        self.feature_index = feature_index
        self.threshold = threshold_value
        self.left = left
        self.right = right
        self.value: Optional[Any] = value

    def is_leaf(self) -> bool:
        
        """
        Returns the boolean of a node value

        Returns
        -------
        bool
            Whether a value is associated with the node
        """
        
        return self.value is not None

class decision_tree():
    
    """
    Decision tree classifier using entropy and information gain

    Attributes
    ----------
    max_depth: int, optional
        Maximum depth of the tree
    min_samples_split: int, optional, default = 2
        Minimum number of samples required to split a node
    tree: Node
        Created decision tree
    _class_mappings: dict, optional
        Dictionary containing class mappings to numeric values
    _reverse_class_mapping: dict, optional
        Dictionary containing numeric values to class mappings
    _n_features: int, optional
        Number of features

    Methods
    -------
    fit(training_array, training_targets):
        Fits the decision tree based on training labels and numeric data
    predict(testing_array):
        Predicts the labels for a set of testing data
    print_tree(node, depth):
        Prints a visual representation of the tree
    """

    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = 2) -> None:

        """
        Creates associated attributes for a decision tree with
        validated parameters

        Parameters
        ----------
        max_depth: int, optional
            Maximum depth of the tree
        min_samples_split: int, optional, default = 2
            Minimum number of samples required to split a node
        """

        _validate_parameters(max_depth, min_samples_split)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[Node] = None
        self._class_mappings: Optional[dict] = None
        self._reverse_class_mappings: Optional[dict] = None
        self._n_features: Optional[int] = None

    def _information_gain(self,
                          parent_class: np.ndarray,
                          left_class: np.ndarray,
                          right_class: np.ndarray) -> float:
        
        """
        Calculates information gain for a given split using entropy
        """

        parent_entropy = _entropy(parent_class)
        left_entropy = _entropy(left_class)
        right_entropy = _entropy(right_class)

        if len(parent_class) != len(left_class) + len(right_class):
            raise ValueError('Summed number of split samples must equal total number of samples')
        
        left_weight = len(left_class) / len(parent_class)
        right_weight = len(right_class) / len(parent_class)

        final_gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

        return final_gain
    
    def _best_split(self,
                    training_array: ArrayLike,
                    training_targets: ArrayLike) -> Tuple[int, float]:
        
        """
        Finds the optimal split for a set of training data
        and targets, returning the associated feature and threshold
        values
        """

        train_array = _2D_numeric(training_array)
        train_targets = _1D_vectorized(training_targets)

        _, n_features = train_array.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            feature_values = np.sort(np.unique(train_array[:, feature]))
            possible_thresholds = (feature_values[:-1] + feature_values[1:]) / 2
            for threshold in possible_thresholds:
                left_indices = train_array[:, feature] <= threshold
                right_indices = train_array[:, feature] > threshold
                
                left_classes = train_targets[left_indices]
                right_classes = train_targets[right_indices]

                if len(left_classes) == 0 or len(right_classes) == 0:
                    continue
                
                gain = self._information_gain(train_targets, left_classes, right_classes)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
                
    def _leaf_value(self, training_targets: np.ndarray) -> Union[int, float, str]:

        """
        Determines the leaf value for a node based on class counts
        """

        train_targets = _1D_vectorized(training_targets)
        classes, counts = np.unique(train_targets, return_counts = True)
        classification = classes[np.argmax(counts)]

        return classification
    
    def _build_tree(self, training_array: ArrayLike, training_targets: ArrayLike, depth: int) -> Node:

        """
        Recursively builds the decision tree from a set of training
        data and targets
        """

        train_array = _2D_numeric(training_array)
        train_targets = _1D_vectorized(training_targets)

        _shape_match(train_array, train_targets)

        n_samples, n_features = train_array.shape

        if n_samples < self.min_samples_split:
            return Node(value = self._leaf_value(train_targets))
        if self.max_depth is not None and self.max_depth <= depth:
            return Node(value = self._leaf_value(train_targets))
        if len(np.unique(train_targets)) == 1:
            return Node(value = self._leaf_value(train_targets))
        
        best_feature, best_threshold = self._best_split(train_array, train_targets)

        if best_feature is None or best_threshold is None:
            return Node(value = self._leaf_value(train_targets))
        
        left_indices = train_array[:, best_feature] <= best_threshold
        right_indices = train_array[:, best_feature] > best_threshold
        
        left_classes = train_targets[left_indices]
        right_classes = train_targets[right_indices]

        left_child = self._build_tree(train_array[left_indices, :], left_classes, depth + 1)
        right_child = self._build_tree(train_array[right_indices, :], right_classes, depth + 1)

        return Node(feature_index = best_feature, threshold_value = best_threshold, left = left_child, right = right_child)
    
    def fit(self, training_array: ArrayLike, training_targets: ArrayLike) -> "decision_tree":

        """
        Fits the decision tree on given input data

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_targets: ArrayLike
            1D array-like object containing training labels

        Returns
        -------
        decision_tree
            Fitted model

        Raises
        ------
        ValueError
            If targets have missing data, or input data
            is not numeric
        """

        training_targets = _1D_vectorized(training_targets)
        if pd.isna(training_targets).any():
            raise ValueError('Target array contains NaN values')
    
        train_array = _2D_numeric(training_array)

        unique_classes = list(dict.fromkeys(training_targets))
        self._class_mappings = {cls: i for i, cls in enumerate(unique_classes)}

        self._reverse_class_mappings = {
                                    i: (cls.item() if isinstance(cls, np.generic) else cls)
                                    for cls, i in self._class_mappings.items()
                                }
        
        train_targets = np.array([self._class_mappings[item] for item in training_targets])
        
        self._n_features = train_array.shape[1]

        final_tree = self._build_tree(train_array, train_targets, 0)
        self.tree = final_tree

        return self
    
    def _verify_fit(self) -> "decision_tree":

        """
        Verifies that the decision tree has been created
        """

        if self.tree is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def _predict_recursive(self, testing_row: np.ndarray, node: Node) -> Union[int, float, str]:
        
        """
        Recursively predicts the target label for an input sample
        """

        if node.value is not None:
            return node.value
        
        if testing_row[node.feature_index] <= node.threshold:
            return self._predict_recursive(testing_row, node.left)
        else:
            return self._predict_recursive(testing_row, node.right)
        
    def predict(self, testing_array: ArrayLike) -> np.ndarray:
        
        """
        Predicts the class labels for given test samples

        Parameters
        ----------
        testing_array: ArrayLike
            2D array-like object of size (n_samples, n_features)

        Returns
        -------
        predictions: np.ndarray
            Array of predicted class labels for each sample
        
        Raises
        ------
        ValueError
            If the number of features in the testing data does not
            match the number of features in training data, or if some
            values are not numeric
        """

        self._verify_fit()

        test_array = _2D_numeric(testing_array)

        if test_array.shape[1] != self._n_features:
            raise ValueError("Number of features in testing data must match number of features in training data")
        
        prediction_array = np.full((test_array.shape[0],), np.nan, dtype = object)
        for sample in range(test_array.shape[0]):
            prediction = self._predict_recursive(test_array[sample, :], self.tree)
            prediction_array[sample] = prediction
        
        if any(value is np.nan or value is None for value in prediction_array):
            raise ValueError("Predictions were not made for all samples")
        
        predictions = np.array([self._reverse_class_mappings[p] for p in prediction_array], dtype=object)
        
        return predictions
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        
        """
        Prints a visual representation of the decision tree

        The tree is displayed recursively with indentation indicating depth; each
        node is given with the feature index and threshold value until a leaf
        node is reached, at which point the associated label is displayed

        Parameters
        ----------
        node: Node, optional
            The current node to print; begins from the root of the fitted
            decision tree if None
        depth: int, default = 0
            Current depth of the tree

        Raises
        ------
        RuntimeError
            If the model has not been fitted prior to use
        """

        if node is None:
            self._verify_fit()
            node = self.tree

        if node.is_leaf():
            print("\t" * depth + f"Predict: {node.value}")
        else:
            print("\t" * depth + f"Feature {node.feature_index} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

class regression_tree():

    """
    Regression tree using variance reduction

    Attributes
    ----------
    max_depth: int, optional
        Maximum depth of the tree
    min_samples_split: int, optional, default = 2
        Minimum number of samples required to split a node
    tree: Node
        Created regression tree
    _n_features: int, optional
        Number of features

    Methods
    -------
    fit(training_array, training_targets):
        Fits the regression tree based on training values and numeric data
    predict(testing_array):
        Predicts the labels for a set of testing data
    print_tree(node, depth):
        Prints a visual representation of the tree
    """

    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = 2) -> None:
        
        """
        Creates associated attributes for a regression tree with
        validated parameters

        Parameters
        ----------
        max_depth: int, optional
            Maximum depth of the tree
        min_samples_split: int, optional, default = 2
            Minimum number of samples required to split a node
        """

        _validate_parameters(max_depth, min_samples_split)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[Node] = None
        self._n_features: Optional[int] = None

    def _variance_reduction(self,
                          parent_class: np.ndarray,
                          left_class: np.ndarray,
                          right_class: np.ndarray) -> float:
        
        """
        Calculates variance reduction for a given split
        """

        parent_class = _ensure_numeric(parent_class)
        left_class = _ensure_numeric(left_class)
        right_class = _ensure_numeric(right_class)

        if len(left_class) == 0 or len(right_class) == 0:
            raise ValueError("Length of right or left child is empty")

        parent_variance = np.var(parent_class)
        left_variance = np.var(left_class)
        right_variance = np.var(right_class)

        if len(parent_class) != len(left_class) + len(right_class):
            raise ValueError('Summed number of split samples must equal total number of samples')
        
        left_weight = len(left_class) / len(parent_class)
        right_weight = len(right_class) / len(parent_class)

        final_var_reduction = parent_variance - (left_weight * left_variance + right_weight * right_variance)

        return final_var_reduction
    
    def _best_split(self,
                    training_array: ArrayLike,
                    training_targets: ArrayLike) -> Tuple[int, float]:
        
        """
        Finds the optimal split for a set of training data
        and targets, returning the associated feature and threshold
        values
        """

        train_array = _2D_numeric(training_array)
        train_targets = _ensure_numeric(training_targets)

        _, n_features = train_array.shape

        best_var = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            feature_values = np.sort(np.unique(train_array[:, feature]))
            possible_thresholds = (feature_values[:-1] + feature_values[1:]) / 2
            for threshold in possible_thresholds:
                left_indices = train_array[:, feature] <= threshold
                right_indices = train_array[:, feature] > threshold
                
                left_classes = train_targets[left_indices]
                right_classes = train_targets[right_indices]

                if len(left_classes) == 0 or len(right_classes) == 0:
                    continue
                
                var = self._variance_reduction(train_targets, left_classes, right_classes)

                if var > best_var:
                    best_var = var
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
                
    def _leaf_value(self, training_targets: np.ndarray) -> Union[int, float, str]:

        """
        Calculates the value at a leaf node through averaging
        """

        train_targets = _ensure_numeric(training_targets)
        regression = np.mean(train_targets)

        return regression
    
    def _build_tree(self, training_array: ArrayLike, training_targets: ArrayLike, depth: int) -> Node:
        
        """
        Recursively builds the regression tree from a set of training
        data and target values
        """

        train_array = _2D_numeric(training_array)
        train_targets = _ensure_numeric(training_targets)

        _shape_match(train_array, train_targets)

        n_samples, n_features = train_array.shape

        if n_samples < self.min_samples_split:
            return Node(value = self._leaf_value(train_targets))
        if self.max_depth is not None and self.max_depth <= depth:
            return Node(value = self._leaf_value(train_targets))
        if np.var(train_targets) == 0:
            return Node(value = self._leaf_value(train_targets))
        
        best_feature, best_threshold = self._best_split(train_array, train_targets)

        if best_feature is None or best_threshold is None:
            return Node(value = self._leaf_value(train_targets))
        
        left_indices = train_array[:, best_feature] <= best_threshold
        right_indices = train_array[:, best_feature] > best_threshold
        
        left_classes = train_targets[left_indices]
        right_classes = train_targets[right_indices]

        left_child = self._build_tree(train_array[left_indices, :], left_classes, depth + 1)
        right_child = self._build_tree(train_array[right_indices, :], right_classes, depth + 1)

        return Node(feature_index = best_feature, threshold_value = best_threshold, left = left_child, right = right_child)
    
    def fit(self, training_array: ArrayLike, training_targets: ArrayLike) -> "regression_tree":

        """
        Fits the regression tree on given input data

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_targets: ArrayLike
            1D array-like object containing training values

        Returns
        -------
        regression_tree
            Fitted model

        Raises
        ------
        ValueError
            If targets have missing data, or data is not numeric
        """

        train_targets = _ensure_numeric(training_targets)
        train_array = _2D_numeric(training_array)
        self._n_features = train_array.shape[1]

        final_tree = self._build_tree(train_array, train_targets, 0)
        self.tree = final_tree

        return self
    
    def _verify_fit(self) -> "regression_tree":

        """
        Verifies that the regression tree has been fit
        """

        if self.tree is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self

    def _predict_recursive(self, testing_row: np.ndarray, node: Node) -> Union[int, float, str]:
        
        """
        Recursively predicts the value of a testing sample
        """

        if node.value is not None:
            return node.value
        
        if testing_row[node.feature_index] <= node.threshold:
            return self._predict_recursive(testing_row, node.left)
        else:
            return self._predict_recursive(testing_row, node.right)
        
    def predict(self, testing_array: ArrayLike) -> np.ndarray:

        """
        Predicts the target values for given test samples

        Parameters
        ----------
        testing_array: ArrayLike
            2D array-like object of size (n_samples, n_features)

        Returns
        -------
        prediction_array: np.ndarray
            Array of predicted target values for each sample
        
        Raises
        ------
        ValueError
            If the number of features in the testing data does not
            match the number of features in training data, or if some
            values are not numeric
        """

        self._verify_fit()

        test_array = _2D_numeric(testing_array)

        if test_array.shape[1] != self._n_features:
            raise ValueError("Number of features in testing data must match number of features in training data")

        prediction_array = np.full((test_array.shape[0],), np.nan, dtype = float)
        for sample in range(test_array.shape[0]):
            prediction = self._predict_recursive(test_array[sample, :], self.tree)
            prediction_array[sample] = prediction
        
        if any(value is np.nan or value is None for value in prediction_array):
            raise ValueError("Predictions were not made for all samples")
        
        return prediction_array

    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        
        """
        Prints a visual representation of the regression tree

        The tree is displayed recursively with indentation indicating depth; each
        node is given with the feature index and threshold value until a leaf
        node is reached, at which point the associated value is displayed

        Parameters
        ----------
        node: Node, optional
            The current node to print; begins from the root of the fitted
            regression tree if None
        depth: int, default = 0
            Current depth of the tree

        Raises
        ------
        RuntimeError
            If the model has not been fitted prior to use
        """

        if node is None:
            self._verify_fit()
            node = self.tree

        if node.is_leaf():
            print("\t" * depth + f"Predict: {node.value}")
        else:
            print("\t" * depth + f"Feature {node.feature_index} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
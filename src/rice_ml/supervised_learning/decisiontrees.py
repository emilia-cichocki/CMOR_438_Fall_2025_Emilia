"""
    Decision and regression tree algorithms (NumPy)


    # TODO: update this here, calculated using entropy - clarify that it only allows for numeric inputs

    Functions
    ---------
    

    Classes
    ---------
   
"""

# TODO: finish above!

__all__ = [
    'decision_tree',
    'regression_tree'
]



# TODO: components of decision tree - node - goes through all possible splits to find what maximizes entropy
# for each node, do that again lol

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]


def _validate_parameters(max_depth: Optional[int], min_samples_split: Optional[int]) -> None:

    # TODO: type hints, docstrings

    if max_depth is not None and not isinstance(max_depth, int):
        raise TypeError('Maximum depth must be an integer')
    if max_depth is not None and max_depth <= 0:
        raise ValueError('Maximum depth must be greater than zero')
    if not isinstance(min_samples_split, int):
        raise TypeError('Minimum samples required to split node must be an integer')
    if min_samples_split <= 0:
        raise TypeError('Minimum samples required to split node must be greater than zero')
    
def _validate_parameters_node(feature_index: Optional[int], 
                              threshold_value: Optional[float], 
                              left: Optional["Node"] = None, 
                              right: Optional["Node"] = None) -> None:

    # TODO: type hints, docstrings

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

def _entropy(train_targets: np.ndarray):

    train_targets = _1D_vectorized(train_targets)

    _, counts = np.unique(train_targets, return_counts = True)
    probabilities = counts / np.sum(counts)
    probabilities_filtered = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities_filtered * np.log2(probabilities_filtered))

    return entropy


class Node():

    def __init__(self,
                 feature_index: Optional[int] = None,
                 threshold_value: Optional[float] = None,
                 left: "Node" = None,
                 right: "Node" = None,
                 value: Optional[Any] = None) -> None:
        
        _validate_parameters_node(feature_index, threshold_value, left, right)

        self.feature_index = feature_index
        self.threshold = threshold_value
        self.left = left
        self.right = right
        self.value: Optional[Any] = value

    def is_leaf(self) -> bool:
        return self.value is not None

class decision_tree():

    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = 2) -> None:
        
        _validate_parameters(max_depth, min_samples_split)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[Node] = None
        self._class_mappings: Optional[dict] = None # TODO: flip the _ so it's class_mappings_
        self._reverse_class_mappings: Optional[dict] = None
        self._n_features: Optional[int] = None

    def _information_gain(self,
                          parent_class: np.ndarray,
                          left_class: np.ndarray,
                          right_class: np.ndarray) -> float:
        
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
        
        train_array = _2D_numeric(training_array)
        train_targets = _1D_vectorized(training_targets)

        _, n_features = train_array.shape

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            possible_thresholds = np.unique(train_array[:, feature])
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

        train_targets = _1D_vectorized(training_targets)
        classes, counts = np.unique(train_targets, return_counts = True)
        classification = classes[np.argmax(counts)]

        return classification
    
    def _build_tree(self, training_array: ArrayLike, training_targets: ArrayLike, depth: int) -> Node:

        train_array = _2D_numeric(training_array)
        train_targets = _1D_vectorized(training_targets)

        n_samples, n_features = train_array.shape

        if n_samples < self.min_samples_split:
            return Node(value = self._leaf_value(train_targets))
        if self.max_depth is not None and self.max_depth <= depth:
            return Node(value = self._leaf_value(train_targets))
        if len(np.unique(train_targets)) == 1:
            return Node(value = self._leaf_value(train_targets))
        
        best_feature, best_threshold = self._best_split(train_array, train_targets)

        left_indices = train_array[:, best_feature] <= best_threshold
        right_indices = train_array[:, best_feature] > best_threshold
        
        left_classes = train_targets[left_indices]
        right_classes = train_targets[right_indices]

        left_child = self._build_tree(train_array[left_indices, :], left_classes, depth + 1)
        right_child = self._build_tree(train_array[right_indices, :], right_classes, depth + 1)

        return Node(feature_index = best_feature, threshold_value = best_threshold, left = left_child, right = right_child)
    
    def fit(self, training_array: ArrayLike, training_targets: ArrayLike) -> "decision_tree":
        
        unique_classes = list(dict.fromkeys(training_targets))
        self._class_mappings = {cls: i for i, cls in enumerate(unique_classes)}

        self._reverse_class_mappings = {
                                    i: (cls.item() if isinstance(cls, np.generic) else cls)
                                    for cls, i in self._class_mappings.items()
                                }
        
        train_targets = np.array([self._class_mappings[item] for item in training_targets])
        
        train_array = _2D_numeric(training_array)
        self._n_features = train_array.shape[1]

        final_tree = self._build_tree(train_array, train_targets, 0)
        self.tree = final_tree

        return self
    
    def _verify_fit(self) -> "decision_tree":
        if self.tree is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def predict(self, testing_array: ArrayLike) -> np.ndarray:

        self._verify_fit()

        test_array = _2D_numeric(testing_array)

        if test_array.shape[1] != self._n_features:
            raise ValueError("Number of features in testing data must match number of features in training data")
        
        # TODO: add something to test that test_array has the same number of features

        prediction_array = np.full((test_array.shape[0],), np.nan, dtype = object)
        for sample in range(test_array.shape[0]):
            prediction = self._predict_recursive(test_array[sample, :], self.tree)
            prediction_array[sample] = prediction
        
        if any(value is np.nan or value is None for value in prediction_array):
            raise ValueError("Predictions were not made for all samples")
        
        predictions = np.array([self._reverse_class_mappings[p] for p in prediction_array], dtype=object)
        
        return predictions

    def _predict_recursive(self, testing_row: np.ndarray, node: Node) -> Union[int, float, str]:
        
        if node.value is not None:
            return node.value
        
        if testing_row[node.feature_index] <= node.threshold:
            return self._predict_recursive(testing_row, node.left)
        else:
            return self._predict_recursive(testing_row, node.right)
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        
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

    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = 2) -> None:
        
        _validate_parameters(max_depth, min_samples_split)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[Node] = None
        self._n_features: Optional[int] = None

    def _variance_reduction(self,
                          parent_class: np.ndarray,
                          left_class: np.ndarray,
                          right_class: np.ndarray) -> float:
        
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

        train_targets = _ensure_numeric(training_targets)
        regression = np.mean(train_targets)

        return regression
    
    def _build_tree(self, training_array: ArrayLike, training_targets: ArrayLike, depth: int) -> Node:

        train_array = _2D_numeric(training_array)
        train_targets = _ensure_numeric(training_targets)

        n_samples, n_features = train_array.shape

        if n_samples < self.min_samples_split:
            return Node(value = self._leaf_value(train_targets))
        if self.max_depth is not None and self.max_depth <= depth:
            return Node(value = self._leaf_value(train_targets))
        if np.var(train_targets) == 0:
            return Node(value = self._leaf_value(train_targets))
        
        best_feature, best_threshold = self._best_split(train_array, train_targets)

        left_indices = train_array[:, best_feature] <= best_threshold
        right_indices = train_array[:, best_feature] > best_threshold
        
        left_classes = train_targets[left_indices]
        right_classes = train_targets[right_indices]

        left_child = self._build_tree(train_array[left_indices, :], left_classes, depth + 1)
        right_child = self._build_tree(train_array[right_indices, :], right_classes, depth + 1)

        return Node(feature_index = best_feature, threshold_value = best_threshold, left = left_child, right = right_child)
    
    def fit(self, training_array: ArrayLike, training_targets: ArrayLike) -> "regression_tree":
        
        train_targets = _ensure_numeric(training_targets)
        train_array = _2D_numeric(training_array)
        self._n_features = train_array.shape[1]

        final_tree = self._build_tree(train_array, train_targets, 0)
        self.tree = final_tree

        return self
    
    def _verify_fit(self) -> "regression_tree":
        if self.tree is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def predict(self, testing_array: ArrayLike) -> np.ndarray:

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

    def _predict_recursive(self, testing_row: np.ndarray, node: Node) -> Union[int, float, str]:
        
        if node.value is not None:
            return node.value
        
        if testing_row[node.feature_index] <= node.threshold:
            return self._predict_recursive(testing_row, node.left)
        else:
            return self._predict_recursive(testing_row, node.right)
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        
        if node is None:
            self._verify_fit()
            node = self.tree

        if node.is_leaf():
            print("\t" * depth + f"Predict: {node.value}")
        else:
            print("\t" * depth + f"Feature {node.feature_index} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
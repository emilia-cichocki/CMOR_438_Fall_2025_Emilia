"""
    Random forest algorithms (NumPy)

    # TODO: update this!

    Functions
    ---------
    

    Classes
    ---------
   
"""

# TODO: finish above!

__all__ = [
    'random_forest',
]

# TODO: arraylike?

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric
from rice_ml.supervised_learning.decisiontrees import *

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters_rf(n_trees: int, 
                         task: Literal['classification', 'regression'], 
                         max_depth: Optional[int],
                         min_samples_split: Optional[int],
                         max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]],
                         random_state: Optional[int]) -> None:
    
    # TODO: type hints, docstrings

    if not isinstance(n_trees, int):
        raise TypeError('Number of trees must be an integer')
    if n_trees <= 0:
        raise ValueError('Number of trees must be greater than zero')
    if task not in ['classification', 'regression']:
        raise ValueError(f"Task type must be one of {['classification', 'regression']}")
    if max_depth is not None and not isinstance(max_depth, int):
        raise TypeError('Maximum depth must be an integer')
    if max_depth is not None and max_depth <= 0:
        raise ValueError('Maximum depth must be greater than zero')
    if min_samples_split is not None and not isinstance(min_samples_split, int):
        raise TypeError('Minimum samples required to split node must be an integer')
    if min_samples_split is not None and min_samples_split <= 0:
        raise ValueError('Minimum samples required to split node must be greater than zero')
    if max_features is not None:
        if not isinstance(max_features, (int, float)) and max_features not in ["sqrt", "log2"]:
            raise TypeError(f"Maximum number of features must be an integer, float, or in {['sqrt', 'log2']}")
        if isinstance(max_features, (int, float)) and max_features <= 0:
            raise ValueError("Maximum number of features must be greater than zero")
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError("Random state parameter must be an integer")
    
class random_forest():

    def __init__(self,
                 n_trees: int,
                 task: Literal['classification', 'regression'],
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = 2,
                 max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = "sqrt",
                 random_state: Optional[int] = None
                 ) -> None:
        
        _validate_parameters_rf(n_trees, task, max_depth, min_samples_split, max_features, random_state)

        self.n_trees = n_trees
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list = []
        self.n_features_: Optional[int] = None

    def _bootstrap_data(self, training_array: np.ndarray, training_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        n_samples = training_array.shape[0]
        rng = _random_number(self.random_state)
        index_choices = rng.choice(n_samples, size = n_samples, replace = True)
        bootstrapped_train_array = training_array[index_choices]
        bootstrapped_train_targets = training_targets[index_choices]

        return bootstrapped_train_array, bootstrapped_train_targets
    
    def _feature_selection(self, total_features: int) -> np.ndarray:
        
        selection_type = self.max_features

        if not isinstance(total_features, int):
            raise TypeError('Total features must be an integer')
        
        if selection_type is None:
            return np.arange(total_features)
        if isinstance(selection_type, int):
            if selection_type > total_features:
                raise ValueError('Number of selected features cannot exceed total features')
            n_select = selection_type
        elif isinstance(selection_type, float):
            if selection_type > 1:
                raise ValueError('Proportion of selected features cannot exceed 1')
            n_select = max(1, int(selection_type * total_features))
        elif selection_type == 'sqrt':
            n_select = max(1, int(np.sqrt(total_features)))
        elif selection_type == 'log2':
            n_select = max(1, int(np.log2(total_features)))
        
        rng = _random_number(self.random_state)

        features = rng.choice(total_features, size = n_select, replace = False)

        return features

    def fit(self, training_array: ArrayLike, training_targets: ArrayLike) -> 'random_forest':
        
        # TODO: type hints/docstrings

        train_array = _2D_numeric(training_array)
        train_targets = _1D_vectorized(training_targets)

        _shape_match(train_array, train_targets)

        self.n_features_ = train_array.shape[1]

        for _ in range(self.n_trees):
            sampled_train_array, sampled_train_targets = self._bootstrap_data(train_array, train_targets)
            features = self._feature_selection(self.n_features_)
            subset_train_array = sampled_train_array[:, features]

            if self.task == 'classification':
                tree = decision_tree(self.max_depth, self.min_samples_split)
            elif self.task == 'regression':
                tree = regression_tree(self.max_depth, self.min_samples_split)

            tree.fit(subset_train_array, sampled_train_targets)

            self.trees.append((tree, features))
        
        return self

    def _verify_fit(self) -> "random_forest":
        if len(self.trees) == 0:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def prediction(self, testing_array: ArrayLike) -> np.ndarray:

        # TODO: type hints, docstrings

        self._verify_fit()

        test_array = _2D_numeric(testing_array)
        if test_array.shape[1] != self.n_features_:
            raise ValueError("Number of features in testing data must match number of features in training data")

        predictions = []
        for tree, features in self.trees:
            subset_test_array = test_array[:, features]
            prediction_value = tree.predict(subset_test_array)
            predictions.append(prediction_value)

        prediction_array = np.array(predictions)

        if self.task == 'classification':
            final_prediction = []
            for i in range(prediction_array.shape[1]):
                values, counts = np.unique(prediction_array[:, i], return_counts = True)
                final_prediction.append(values[np.argmax(counts)])
            
            final_prediction_array = np.array(final_prediction)
        elif self.task == 'regression':
            final_prediction_array = np.mean(prediction_array, axis = 0)
        
        return final_prediction_array

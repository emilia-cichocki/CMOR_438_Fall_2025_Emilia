
"""
    Random forest algorithms (NumPy)

    This module contains the classes implementing random forest algorithms for
    classification and regression tasks. It uses combinations of decision or
    regression trees, and supports numeric input features for categorical (classification)
    or numeric (regression) targets

    Classes
    ---------
    random_forest
        Implements the random forest algorithm for either classification or regression
"""

__all__ = [
    'random_forest',
]

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.decisiontrees import *

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters_rf(n_trees: int, 
                         task: Literal['classification', 'regression'], 
                         max_depth: Optional[int],
                         min_samples_split: Optional[int],
                         max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]],
                         random_state: Optional[int]) -> None:
    
    """
    Validates hyperparameters for a random forest model

    Parameters
    ----------
    task: {'classification', 'regression'}
        Specifies whether the task is classification (decision trees) or
        regression (regression trees)
    max_depth: int, optional
        Maximum depth of the tree
    min_samples_split: int, optional
        Minimum number of samples required to split a node
    max_features: {int, float, 'sqrt', 'log2'}, optional
        Specifies how to calculate the maximum number of features considered
        in each tree
    random_state: int, optional
        The random state for the random generator

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

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

    """
    Random forest classifier, with component trees using entropy and 
    information gain for classification, and variance reduction for regression

    Attributes
    ----------
    n_trees: int
        Number of trees in the random forest
    task: {'classification', 'regression'}
        Type of task performed
    max_depth: int
        Maximum depth of each tree
    min_samples_split: int, default = 2
        Minimum number of samples required to split a node
    max_features: {int, float, 'sqrt', 'log2'}
        Specifies how to calculate the maximum number of features considered
        in each tree
        - int: a defined number of features
        - float: a proportion of total features
        - 'sqrt': the integer value of the square root of total number of features
        - 'log2': the integer value of the log2 of total number of features
    random_state: int
        Random state for bootstrapping and feature selection; if None,
        a randomized seed is used
    trees: list
        List of trees in the random forest
    n_features_: int
        Number of features in the data

    Methods
    -------
    fit(training_array, training_targets):
        Fits the random forest based on training labels and numeric data
    prediction(testing_array):
        Predicts the labels for a set of testing data
    
    Examples
    --------
    >>> X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    >>> y = np.array([0, 1, 1, 0])
    >>> model = random_forest(n_trees=3, task='classification', max_depth=2, random_state=42)
    >>> _ = model.fit(X, y)
    >>> preds = model.prediction(X)
    >>> preds.shape
    (4,)
    >>> set(preds.astype(int).tolist())
    {0, 1}
    """

    def __init__(self,
                 n_trees: int,
                 task: Literal['classification', 'regression'],
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = 2,
                 max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = "sqrt",
                 random_state: Optional[int] = None
                 ) -> None:
        
        """
        Creates associated attributes for a random forest tree with
        validated parameters

        Parameters
        ----------
        n_trees: int
            Number of trees in the random forest
        task: {'classification', 'regression'}
            Type of task performed
        max_depth: int
            Maximum depth of each tree
        min_samples_split: int, default = 2
            Minimum number of samples required to split a node
        max_features: {int, float, 'sqrt', 'log2'}
            Specifies how to calculate the maximum number of features considered
            in each tree
            - int: a defined number of features
            - float: a proportion of total features
            - 'sqrt': the integer value of the square root of total number of features
            - 'log2': the integer value of the log2 of total number of features
        random_state: int
            Random state for bootstrapping and feature selection; if None,
            a randomized seed is used
        """
         
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

        """
        Generates samples of bootstrapped data using random selection with
        replacement from a data array
        """

        n_samples = training_array.shape[0]
        rng = _random_number(self.random_state)
        index_choices = rng.choice(n_samples, size = n_samples, replace = True)
        bootstrapped_train_array = training_array[index_choices]
        bootstrapped_train_targets = training_targets[index_choices]

        return bootstrapped_train_array, bootstrapped_train_targets
    
    def _feature_selection(self, total_features: int) -> np.ndarray:
        
        """
        Selects the features to be used in each component tree
        based on the given method
        """

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
        
        """
        Fits the random forest on given input data

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_targets: ArrayLike
            1D array-like object containing training values

        Returns
        -------
        random_forest
            Fitted random forest model
            - For classification, each component tree is a decision tree with
            majority voting
            - For regression, each component tree is a regression tree with
            averaging

        Raises
        ------
        ValueError
            If targets have missing data, or data is not numeric
        """

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

        """
        Verifies that the model has been fitted
        """    

        if len(self.trees) == 0:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def prediction(self, testing_array: ArrayLike) -> np.ndarray:

        """
        Predicts the target values for given input samples

        Parameters
        ----------
        testing_array: ArrayLike
            2D array-like object of size (n_samples, n_features)

        Returns
        -------
        final_prediction_array: np.ndarray
            Array of predicted target values for each sample
            - For classification, this corresponds to the class label
            - For regression, this corresponds to the target feature
        
        Raises
        ------
        ValueError
            If the number of features in the input data does not
            match the number of features in training data, or if some
            values are not numeric
        """

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

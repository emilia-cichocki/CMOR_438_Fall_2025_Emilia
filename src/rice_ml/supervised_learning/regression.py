"""
    Regression algorithms (NumPy)

    This module implements several regression algorithms (linear and logistic) on 
    numeric NumPy arrays. It supports both gradient descent with mean-squared error (linear)
    or binary cross-entropy loss with sigmoid activation (logistic), and direct calculation 
    through the normal equation (linear).

    Classes
    ---------
    linear_regression
        Implements linear regression using either the normal equation or stochastic gradient
        descent with mean-squared error
    logistic_regression
        Implements logistic regression using stochastic gradient descent with a
        sigmoid activation function
"""

__all__ = [
    'linear_regression',
    'logistic_regression'
]

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters(method: Literal['normal', 'gradient_descent'] = None,
                         learning_rate: Optional[float] = None,
                         epochs: Optional[int] = None,
                         fit_intercept: bool = True, 
                         ) -> None:

    """
    Validates hyperparameters for regression

    Parameters
    ----------
    method: {'normal', 'gradient_descent'}
        Selection of method for linear regression
    learning_rate: float, optional
        Learning rate for the model
    epochs: int, optional
        Maximum number of epochs
    fit_intercept: boolean, default = True
        Whether to fit a model intercept

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

    if method is not None and method not in ('normal', 'gradient_descent'):
        raise ValueError(f"Regression method must be one of {['normal', 'gradient_descent']}")
    if learning_rate is not None and not isinstance(learning_rate, (int, float)):
        raise TypeError('Learning rate must be a float')
    if learning_rate is not None and learning_rate <= 0:
        warnings.warn(f"For model to learn properly, learning rate should be greater than zero", UserWarning)
    if epochs is not None and not isinstance(epochs, int):
        raise TypeError('Maximum epochs must be an integer')
    if epochs is not None and epochs <= 0:
        raise ValueError('Maximum epochs must be greater than zero')
    if not isinstance(fit_intercept, bool):
        raise TypeError('fit_intercept parameter must be a boolean')  
    if method == 'gradient_descent' and epochs is None:
        raise ValueError('Number of epochs is required for linear descent')
    if method == 'gradient_descent' and learning_rate is None:
        raise ValueError('Learning rate is required for linear descent')
    
def _validate_arrays(data_array: Optional[ArrayLike] = None,
                     target_vector: Optional[ArrayLike] = None
                     ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    """
    Ensures that data array is 2D and contain only numeric data,
    target vectors are 1D (or can be reshaped to 1D), and that the 
    shape of feature data and target values match for linear regression

    Parameters
    ----------
    data_array: ArrayLike, optional
        2D data array containing numeric feature values
    target_vector: ArrayLike, optional
        1D vector containing target values

    Returns
    -------
    array: np.ndarray (if `data_array` is not None)
        2D validated feature array
    vector: np.ndarray (if `target_vector` is not None)
        1D validated target value vector

    Raises
    ------
    ValueError
        If either input array contains NaN values, or the shapes of the two 
        input arrays do not match

    Examples
    --------
    >>> _validate_arrays([[1, 2], [3, 4]])
    array([[1., 2.],
           [3., 4.]])
    """

    if data_array is not None:
        array = _2D_numeric(data_array, 'data_array')

        if np.isnan(array).any():
            raise ValueError('Data array contains missing data (NaN values)')

    if target_vector is not None:
        target_vector = np.array(target_vector)
        if target_vector.ndim == 2 and (target_vector.shape[1] == 1 or target_vector.shape[0] == 1):
            vector = target_vector.reshape(-1)
            vector = _ensure_numeric(vector, 'target_vector')
        else:
            vector = _ensure_numeric(target_vector, 'target_vector')
        if np.isnan(vector).any():
            raise ValueError('Target vector contains missing data (NaN values)')

    if data_array is not None and target_vector is not None:
        _shape_match(array, vector)
        return array, vector
    elif data_array is not None:
        return array
    elif target_vector is not None:
        return vector
    
def _validate_arrays_logistic(data_array: Optional[ArrayLike] = None,
                              target_vector: Optional[ArrayLike] = None
                              ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    """
    Ensures that data array is 2D and contain only numeric data,
    target vectors are 1D (or can be reshaped to 1D) and only contains two
    targets, and that the shape of feature data and target values match
    for logistic regression in binary classification

    Parameters
    ----------
    data_array: ArrayLike, optional
        2D data array containing numeric feature values
    target_vector: ArrayLike, optional
        1D vector containing class labels

    Returns
    -------
    array: np.ndarray (if `data_array` is not None)
        2D validated feature array
    vector: np.ndarray (if `target_vector` is not None)
        1D validated target class labels

    Raises
    ------
    ValueError
        If either input array contains NaN values, the shapes of the two 
        input arrays do not match, or the target labels contain more than
        two classes

    Examples
    --------
    >>> _validate_arrays_logistic([[1, 2], [3, 4]])
    array([[1., 2.],
           [3., 4.]])
    """

    if data_array is not None:
        array = _2D_numeric(data_array, 'data_array')

        if np.isnan(array).any():
            raise ValueError('Data array contains missing data (NaN values)')

    if target_vector is not None:
        target_vector = np.array(target_vector)
        if target_vector.ndim == 2 and (target_vector.shape[1] == 1 or target_vector.shape[0] == 1):
            vector = target_vector.reshape(-1)
            vector = _1D_vectorized(vector, 'target_vector')
        else:
            vector = _1D_vectorized(target_vector, 'target_vector')
        classes = np.unique(target_vector)
        if len(classes) != 2:
            raise ValueError("Logistic regression only supports binary targets")
        mapping = {classes[0]: 0, classes[1]: 1}
        vector = (np.vectorize(mapping.get)(target_vector)).reshape(-1)
        if np.isnan(vector).any():
            raise ValueError('Target vector contains missing data (NaN values)')

    if data_array is not None and target_vector is not None:
        _shape_match(array, vector)
        return array, vector
    elif data_array is not None:
        return array
    elif target_vector is not None:
        return vector
    
def _sigmoid(z):
    
    """
    Calculates the sigmoid activation function
    """

    return 1.0/(1.0 + np.exp(-z))


class linear_regression:

    """
    Implements linear regression using either the normal equation
    or stochastic gradient descent

    Attributes
    ----------
    method: {'normal, 'gradient_descent'}
        Method for performing linear regression
    fit_intercept: bool
        Whether the model fits an intercept
    learning_rate: float, optional
        Learning rate for the model
    epochs: int, optional
        Maximum number of epochs
    coef_: np.ndarray, optional
        Weight matrix corresponding to each feature, representing
        coefficients in the model
    intercept_: np.ndarray
        Intercept term for the model
    error_: list
        Stores the error for the model over time
    _training_array: np.ndarray
        Stores the training data array
    _training_targets: np.ndarray
        Stores the training target values

    Methods
    -------
    fit(training_array, training_targets, random_state, shuffle):
        Fits the linear regression based on training targets and numeric data
    prediction(testing_array):
        Predicts the target value for a testing sample
    scoring(testing_array, actual_targets)
        Scores the linear regression using R2
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([2, 4, 6, 8])
    >>> model = linear_regression(method='normal')
    >>> _ = model.fit(X, y)
    >>> preds = model.prediction(np.array([[5], [6]]))
    >>> preds
    array([10., 12.])
    >>> bool(model.scoring(X, y) == 1.0)
    True
    """

    def __init__(self,
                 method: Literal['normal', 'gradient_descent'],
                 fit_intercept: bool = True,
                 *, 
                 learning_rate: Optional[float] = None,
                 epochs: Optional[int] = None) -> None:
        
        """
        Creates associated attributes for a linear regression with
        validated parameters

        Parameters
        ----------
        method: {'normal, 'gradient_descent'}
            Method for performing linear regression
        fit_intercept: bool
            Whether the model fits an intercept
        learning_rate: float
            Learning rate for the model
        epochs: int
            Maximum number of epochs
        """

        _validate_parameters(method, learning_rate, epochs, fit_intercept)

        self.method = method
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.error_: Optional[list] = None
        self._training_array: Optional[np.ndarray] = None
        self._training_targets: Optional[np.ndarray] = None

    
    def fit(self, 
            training_array: np.ndarray, 
            training_targets: np.ndarray, 
            random_state: Optional[int] = None, 
            shuffle: bool = True) -> 'linear_regression':
        
        """
        Fits the linear regression on given input data using either the
        normal equation or stochastic gradient descent

        Parameters
        ----------
        training_array: np.ndarray
            2D array containing training data
        training_targets: np.ndarray
            1D array containing target values
        random_state: int, optional
            Random state for shuffling data using a random generator; if
            None, a randomized seed is used (gradient descent)
        shuffle: boolean, default = 'True'
            Whether to shuffle the data on each iteration (gradient descent)

        Returns
        -------
        linear_regression
            Fitted linear regression model

        Raises
        ------
        TypeError
            If `shuffle` is not a boolean
        ValueError
            If the input data does not match in size, or contains NaN or non-numeric values
            (for features); for normal equation, if the matrix is singular
        """

        if not isinstance(shuffle, bool):
            raise TypeError('Shuffle must be a boolean')
        train_array, train_targets = _validate_arrays(training_array, training_targets)
        
        self._training_array = train_array
        self._training_targets = train_targets

        if self.fit_intercept:
            train_array = np.hstack([np.ones((train_array.shape[0], 1)), train_array])

        if self.method == 'normal':
            first_matrix = np.matmul(train_array.T, train_array)
            second_matrix = np.matmul(train_array.T, train_targets)
            try:
                weights = np.linalg.solve(first_matrix, second_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is singular and normal equation cannot be solved; try stochastic gradient descent instead")

        elif self.method == 'gradient_descent':
            self.error_ = []
            rng = _random_number(random_state)
            if self.fit_intercept:
                weights = rng.standard_normal(training_array.shape[1] + 1).reshape(-1)
            else:
                weights = rng.standard_normal(training_array.shape[1]).reshape(-1)

            for iteration in range(self.epochs):
                if shuffle:
                    indices = rng.permutation(train_array.shape[0])
                else:
                    indices = np.arange(train_array.shape[0])
                for entry in indices:
                    error = np.matmul(train_array[entry], weights) - train_targets[entry]
                    weights -= self.learning_rate * error * train_array[entry]
                
                predictions = np.matmul(train_array, weights)
                mse = np.mean((predictions - train_targets) ** 2)
                
                self.error_.append(mse)

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = None
            self.coef_ = weights

        return self

    def _verify_fit(self) -> Tuple[np.ndarray, np.ndarray]:

        """
        Verifies that the model is fitted
        """

        if self.coef_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")
        
        return self._training_array, self._training_targets
    
    def prediction(self, testing_array: np.ndarray) -> np.ndarray:
        
        """
        Predicts the target value for given input samples

        Parameters
        ----------
        testing_array: np.ndarray
            2D array of size (n_samples, n_features)

        Returns
        -------
        prediction: np.ndarray
            Array of predicted target values for each sample
        """

        self._verify_fit()

        test_array = _validate_arrays(testing_array)
        coef_array = _ensure_numeric(self.coef_)

        if test_array.shape[1] != len(coef_array):
            raise ValueError('Test array must have the same number of input features as coefficients')
        
        intercept = self.intercept_

        prediction = np.matmul(test_array, coef_array)

        if intercept is not None:
            prediction += intercept
            return prediction
        else:
            return prediction
    
    def scoring(self, testing_array: ArrayLike, actual_targets: ArrayLike) -> float:

        """
        Calculates R2 score for regression on input data

        Parameters
        ----------
        testing_array: ArrayLike
            2D array-like object of size (n_samples, n_features)
        actual_targets: ArrayLike
            1D array-like object of true target values with size (n_sample,)

        Returns
        -------
        r2_score: float
            R2 score for the predictions compared to true values
        """

        predicted_target_array = self.prediction(testing_array)
        actual_target_array = _ensure_numeric(actual_targets)

        if predicted_target_array.shape != actual_target_array.shape:
            raise ValueError("Shapes of predicted and actual targets must match")

        rss = np.sum((predicted_target_array - actual_target_array) ** 2)
        actual_mean = np.mean(actual_target_array)
        tss = np.sum((actual_target_array - actual_mean) ** 2)
        
        training_data_array, _ = self._verify_fit()
        testing_data_array = _2D_numeric(testing_array)

        if tss == 0:
            if np.array_equal(testing_data_array, training_data_array) and rss == 0:
                return 1.0
            else:
                raise ValueError('R2 score is undefined for constant true targets, unless evaluated on training data with a perfect fit')

        r2_score = 1 - (rss / tss)

        return r2_score
    
class logistic_regression():
    
    """
    Implements logistic regression using stochastic gradient descent
    with sigmoid activation

    Attributes
    ----------
    epochs: int
        Maximum number of epochs
    learning_rate: float
        Learning rate for the model
    coef_: np.ndarray, optional
        Weight matrix corresponding to each feature
    bias_: np.ndarray
        Bias term for the model
    class_mapping_: dict
        Dictionary mapping classes to discrete numeric values
    loss_: list
        Stores the loss for the model over time
    _training_array: np.ndarray
        Stores the training data array
    _training_targets: np.ndarray
        Stores the training target values
    
    Methods
    -------
    fit(training_array, training_targets, random_state, shuffle):
        Fits the logistic regression based on training targets and class labels
    prediction(testing_array, threshold):
        Predicts the class for a testing sample
    predict_proba(testing_array):
        Predicts the probability that a sample is in the positive class
    scoring(testing_array, actual_targets, threshold)
        Scores the logistic regression using accuracy
    
    Examples
    --------
    >>> X = np.array([[0], [1], [2], [3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> model = logistic_regression(epochs = 100, learning_rate = 0.1)
    >>> _ = model.fit(X, y, random_state=42)
    >>> model.prediction(np.array([[1.5], [2.5]]))
    array([1, 1])
    >>> model.predict_proba(np.array([[1.5], [2.5]]))
    array([0.59869776, 0.9333179 ])
    >>> bool(model.scoring(X, y) == 1.0)
    True
    """

    def __init__(self,
                 epochs: int = 1000,
                 learning_rate: float = 0.01
                 ) -> None:
        
        """
        Creates associated attributes for a logistic regression with
        validated parameters

        Parameters
        ----------
        epochs: int, default = 1000
            Maximum number of epochs
        learning_rate: float = 0.01
            Learning rate for the model
        """

        _validate_parameters(learning_rate = learning_rate, epochs = epochs)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.class_mapping_: Optional[dict] = None
        self.loss_: Optional[list] = None
    
    def fit(self, 
            training_array: np.ndarray, 
            training_targets: np.ndarray, 
            random_state: Optional[int] = None, shuffle: bool = True) -> 'logistic_regression':
        
        """
        Fits the logistic regression on given input data using stochastic
        gradient descent with encoded vectors for class

        Parameters
        ----------
        training_array: np.ndarray
            2D array containing training data
        training_targets: np.ndarray
            2D array containing training labels
        random_state: int, optional
            Random state for shuffling data using a random generator; if
            None, a randomized seed is used

        Returns
        -------
        logistic_regression
            Fitted logistic regression model

        Raises
        ------
        TypeError
            If `shuffle` is not a boolean
        ValueError
            If the input data does not match in size, or contains missing values
        """

        if not isinstance(shuffle, bool):
            raise TypeError('Shuffle must be a boolean')
        
        rng = _random_number(random_state)

        train_array, train_targets = _validate_arrays_logistic(training_array, training_targets)
        
        classes = np.unique(training_targets)

        self.class_mapping_ = {0: classes[0], 1: classes[1]}
        
        train_array = np.hstack([np.ones((train_array.shape[0], 1)), train_array])
        weights = rng.standard_normal(train_array.shape[1]).reshape(-1)
        
        self.loss_ = []

        for iteration in range(self.epochs):
            if shuffle:
                indices = rng.permutation(train_array.shape[0])
            else:
                indices = np.arange(train_array.shape[0])
            for entry in indices:
                x = train_array[entry]
                y = train_targets[entry]
                learn_rate = self.learning_rate
                z = _sigmoid(np.matmul(x, weights))
                error = z - y
                weights -= learn_rate * error * x

            pred = _sigmoid(np.matmul(train_array, weights))
            loss = -np.mean(train_targets * np.log(pred + 1e-15) + (1 - train_targets) * np.log(1 - pred + 1e-15))
            self.loss_.append(loss)
        
        self.bias_ = weights[0]
        self.coef_ = weights[1:]

        return self
    
    def _verify_fit(self) -> Tuple[np.ndarray, np.ndarray]:

        """
        Verifies that the model has been fitted
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def prediction(self, testing_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        
        """
        Predicts the target class for given input samples

        Parameters
        ----------
        testing_array: np.ndarray
            2D array of size (n_samples, n_features)
        threshold: float, default = 0.5
            Threshold value for assigning a sample to the positive class

        Returns
        -------
        classification: np.ndarray
            Array containing the predicted class for each sample
        
        Raises
        ------
        TypeError
            If `threshold` is not a float
        """

        if not isinstance(threshold, float):
            raise TypeError('Threshold value must be a float')
        
        self._verify_fit()

        test_array = _validate_arrays_logistic(testing_array)
        
        coef_array = _1D_vectorized(self.coef_)

        if test_array.shape[1] != len(coef_array):
            raise ValueError('Test array must have the same number of input features as coefficients')
        
        bias = self.bias_
        prediction_value = _sigmoid(np.matmul(test_array, coef_array) + bias)

        prediction_prob = np.array([1 if x > threshold else 0 for x in prediction_value])

        classification = np.array([self.class_mapping_[prediction] for prediction in prediction_prob])

        return classification
    
    def predict_proba(self, testing_array: np.ndarray) -> np.ndarray:
        
        """
        Predicts the target class for given input samples

        Parameters
        ----------
        testing_array: np.ndarray
            2D array of size (n_samples, n_features)

        Returns
        -------
        prediction_value: np.ndarray
            Array containing the probability that each sample belongs
            to the positive class
        
        Raises
        ------
        ValueError
            If the input array does not have the same number of input
            features as coefficients
        """

        self._verify_fit()

        test_array = _validate_arrays_logistic(testing_array)
        
        coef_array = _1D_vectorized(self.coef_)

        if test_array.shape[1] != len(coef_array):
            raise ValueError('Test array must have the same number of input features as coefficients')
        
        bias = self.bias_

        prediction_value = _sigmoid(np.matmul(test_array, coef_array) + bias)

        return prediction_value
    
    def scoring(self, testing_array: ArrayLike, actual_targets: ArrayLike, threshold: float = 0.5) -> np.ndarray:

        """
        Calculates accuracy score for logistic regression on input data

        Parameters
        ----------
        testing_array: ArrayLike
            2D array-like object of size (n_samples, n_features)
        actual_targets: ArrayLike
            1D array-like object of true classes with size (n_sample,)

        Returns
        -------
        accuracy: float
            Accuracy score for the predicted classes
        """

        predicted_target_array = self.prediction(testing_array, threshold)
        actual_target_array = _1D_vectorized(actual_targets)

        if predicted_target_array.shape != actual_target_array.shape:
            raise ValueError("Shapes of predicted and actual targets must match")
        
        accuracy = np.mean(predicted_target_array == actual_target_array)

        return accuracy
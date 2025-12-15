
"""
    Perceptron algorithms (NumPy)

    This module implements the single-layer and multi-layer Perceptron algorithms
    on numeric NumPy arrays. It supports binary and multi-class classification, and
    implements the Perceptron update rule or stochastic gradient descent using a 
    sigmoid activation function.

    Classes
    ---------
    Perceptron
        Implements a single-layer Perceptron with a sign activation function
    multilayer_Perceptron
        Implements a multi-layer Perceptron with a sigmoid activation function
"""

__all__ = [
    'Perceptron',
    'multilayer_Perceptron'
]

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters(learning_rate: Optional[float],
                         epochs: Optional[int],
                         ) -> None:

    """
    Validates hyperparameters for Perceptron

    Parameters
    ----------
    learning_rate: float, optional
        Learning rate for the model
    epochs: int, optional
        Maximum number of epochs

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

    if not isinstance(learning_rate, (int, float)):
        raise TypeError('Learning rate must be a float')
    if learning_rate <= 0:
        warnings.warn(f"For model to learn properly, learning rate should be greater than zero", UserWarning)
    if not isinstance(epochs, int):
        raise TypeError('Maximum epochs must be a float')
    if epochs <= 0:
        raise ValueError('Maximum epochs must be greater than zero')
    

def _validate_arrays_perceptron(data_array: Optional[ArrayLike] = None,
                              target_vector: Optional[ArrayLike] = None
                              ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    """
    Ensures that data array is 2D and contain only numeric data,
    target vectors are 1D, and that the shape of feature data and labels match

    Parameters
    ----------
    data_array: ArrayLike
        2D data array containing numeric values
    target_vector: ArrayLike, optional
        1D vector containing class targets

    Returns
    -------
    array: np.ndarray (if `data_array` is not None)
        2D validated feature array
    vector: np.ndarray (if `target_vector` is not None)
        1D validated label vector

    Raises
    ------
    ValueError
        If either input array contains NaN values, the shapes of the two 
        input arrays do not match, or the label vector has more than two targets

    Examples
    --------
    >>> _validate_arrays_perceptron([[1, 2], [3, 4]])
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
            raise ValueError("Perceptron only supports binary targets")
        mapping = {classes[0]: -1, classes[1]: 1}
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
    
def _activation_function(z):

    """
    Sign activation function for Perceptron
    """

    return np.where(z > 0, 1, -1)

def _sigmoid(z):
    
    """
    Sigmoid activation function for multi-layer Perceptron
    """
    
    return 1.0/(1.0 + np.exp(-z))

def derivative_sigmoid(z):

    """
    Sigmoid derivative function for multi-layer Perceptron
    """

    return _sigmoid(z) * (1.0 - _sigmoid(z))

class Perceptron():
    
    """
    Implements the Perceptron algorithm using a sign activation function
    and the Perceptron update rule

    Attributes
    ----------
    epochs: int, default = 1000
        Maximum number of epochs
    learning_rate: float, default = 0.01
        Learning rate for the model
    coef_: np.ndarray, optional
        Weight matrix corresponding to each feature
    bias_: np.ndarray
        Bias term for the model
    class_mapping_: dict
        Dictionary of class mappings to binary numeric values
    error_: list
        Stores the error for the model over time

    Methods
    -------
    fit(training_array, training_targets, random_state, shuffle):
        Fits the Perceptron based on training targets and numeric data
    prediction(testing_array):
        Predicts the class label for a testing sample
    scoring(testing_array, actual_targets)
        Scores the classifier using accuracy

    Examples
    --------
    >>> X = np.array([[0, 0],
    ...               [1, 1],
    ...               [1, 0],
    ...               [0, 1]])
    >>> y = np.array([0, 1, 1, 0])
    >>> model = Perceptron(epochs=50, learning_rate=0.1)
    >>> _ = model.fit(X, y, random_state=42)
    >>> X_test = np.array([[1, 1],
    ...                    [0, 0]])
    >>> model.prediction(X_test)
    array([1, 0])
    >>> model.scoring(X, y)
    1.0
    """

    def __init__(self,
                 epochs: int = 1000,
                 learning_rate: float = 0.01
                 ) -> None:
        
        """
        Creates associated attributes for a Perceptron with
        validated parameters

        Parameters
        ----------
        epochs: int, default = 1000
            Maximum number of epochs
        learning_rate: float, default = 0.01
            Learning rate for the model
        """

        _validate_parameters(learning_rate = learning_rate, epochs = epochs)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.class_mapping_: Optional[dict] = None
        self.error_: Optional[list] = None
    
    def fit(self, 
            training_array: np.ndarray, 
            training_targets: np.ndarray, 
            random_state: Optional[int] = None, 
            shuffle: bool = True) -> 'Perceptron':
        
        """
        Fits the Perceptron on given input data

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_targets: ArrayLike
            1D array-like object containing training labels
        random_state: int, optional
            Random state for shuffling data using a random generator; if
            None, a randomized seed is used
        shuffle: boolean, default = 'True'
            Whether to shuffle the data on each iteration

        Returns
        -------
        Perceptron
            Fitted Perceptron model

        Raises
        ------
        TypeError
            If `shuffle` is not a boolean
        ValueError
            If the input data does not match in size, or contains NaN or non-numeric values
            (for features)
        """

        if not isinstance(shuffle, bool):
            raise TypeError('Shuffle must be a boolean')
        
        rng = _random_number(random_state)

        train_array, train_targets = _validate_arrays_perceptron(training_array, training_targets)
        
        classes = np.unique(training_targets)

        self.class_mapping_ = {-1: classes[0], 1: classes[1]}
        
        train_array = np.hstack([np.ones((train_array.shape[0], 1)), train_array])
        weights = rng.normal(loc=0, scale=0.01, size=train_array.shape[1]).reshape(-1)
        
        self.error_ = []

        for iteration in range(self.epochs):
            errors = 0

            if shuffle:
                indices = rng.permutation(train_array.shape[0])
            else:
                indices = np.arange(train_array.shape[0])
            for entry in indices:
                x = train_array[entry]
                y = train_targets[entry]
                learn_rate = self.learning_rate
                y_hat = _activation_function(np.matmul(x, weights))
                error = y_hat - y

                if error != 0:
                    errors += 1

                weights -= learn_rate * error * x
            
            self.error_.append(errors)
        
        self.bias_ = weights[0]
        self.coef_ = weights[1:]

        return self

    def _verify_fit(self) -> 'Perceptron':

        """
        Verifies that the model has been fitted
        """

        if self.coef_ is None or self.bias_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    

    def prediction(self, testing_array: np.ndarray) -> np.ndarray:
        
        """
        Predicts the target class for given testing samples

        Parameters
        ----------
        testing_array: np.ndarray
            2D array of size (n_samples, n_features)

        Returns
        -------
        classification: np.ndarray
            Array of predicted class for each sample
        """

        self._verify_fit()

        test_array = _validate_arrays_perceptron(testing_array)
        
        coef_array = _1D_vectorized(self.coef_)

        if test_array.shape[1] != len(coef_array):
            raise ValueError('Test array must have the same number of input features as coefficients')
        
        bias = self.bias_
        prediction_value = _activation_function(np.matmul(test_array, coef_array) + bias)

        classification = np.array([self.class_mapping_[int(prediction)] for prediction in prediction_value])

        return classification
    
    def scoring(self, testing_array: ArrayLike, actual_targets: ArrayLike) -> np.ndarray:

        """
        Calculates accuracy score for classification on testing data

        Parameters
        ----------
        testing_array: ArrayLike
            2D array-like object of size (n_samples, n_features)
        actual_targets: ArrayLike
            1D array-like object of true classes with size (n_sample,)

        Returns
        -------
        accuracy: float
            Accuracy for the predictions compared to true labels
        """

        predicted_target_array = self.prediction(testing_array)
        actual_target_array = _1D_vectorized(actual_targets)

        if predicted_target_array.shape != actual_target_array.shape:
            raise ValueError("Shapes of predicted and actual targets must match")
        
        accuracy = np.mean(predicted_target_array == actual_target_array)

        return accuracy

class multilayer_Perceptron():
    
    """
    Implements the multi-layer Perceptron algorithm using a sigmoid activation function
    and stochastic gradient descent

    Attributes
    ----------
    layers: list
        number of neurons in each layer, including input, hidden, and output
    epochs: int, default = 1000
        Maximum number of epochs
    learning_rate: float, default = 0.01
        Learning rate for the model
    error_: list
        Stores the error for the model over time
    coef_: np.ndarray, optional
        Weight matrix corresponding to each feature
    bias_: np.ndarray
        Bias term for the model
    classes_: dict
        Array corresponding to classes

    Methods
    -------
    fit(training_array, training_targets, random_state):
        Fits the multi-layer Perceptron based on training targets and numeric data
    prediction(testing_array):
        Predicts the class probabilities and label for a testing sample

    Examples
    --------
    >>> X = np.array([[0, 0],
    ...               [0, 1],
    ...               [1, 0],
    ...               [1, 1]])
    >>> y = np.array([[0], [1], [1], [0]])
    >>> mlp = multilayer_Perceptron(layers=[2, 3, 1], epochs=200, learning_rate=0.1)
    >>> _ = mlp.fit(X, y, random_state=42)
    >>> probs, labels = mlp.prediction(X)
    >>> labels.shape
    (4, 1)
    """

    def __init__(self,
                  layers: list,
                  epochs: int = 1000,
                  learning_rate: float = 0.01
                 ) -> None:


        """
        Creates associated attributes for a multi-layer Perceptron with
        validated parameters

        Parameters
        ----------
        layers: list
            List for number of neurons in each layer
        epochs: int, default = 1000
            Maximum number of epochs
        learning_rate: float, default = 0.01
            Learning rate for the model
        """

        _validate_parameters(learning_rate = learning_rate, epochs = epochs)
        
        if not isinstance(layers, list):
            raise TypeError('Layers must be a list')
        if not all(isinstance(item, int) for item in layers):
            raise TypeError('Every element in the layers list must be an integer')
        if min(layers) <= 0:
            raise ValueError('Every element in the layers list must be greater than zero')

        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.error_: list = []
        self.coef_: Optional[list] = None
        self.bias_: Optional[list] = None
        self.classes_: Optional[np.ndarray] = None

    def _weight_initialization(self, random_state: Optional[int] = None) -> Tuple[list, list]:

        """
        Randomly initializes weight matrices with a given random state, and creates
        zero vectors for initial bias
        """

        layers = self.layers
        weights = []
        bias = []
        rng = _random_number(random_state)
        for i in range(1, len(layers)):
            layer_weight = rng.standard_normal((layers[i - 1], layers[i]))
            layer_bias = np.zeros(layers[i])
            weights.append(layer_weight)
            bias.append(layer_bias)
        
        return weights, bias
    
    def _forward_layer(self, training_array: np.ndarray, weights: list, bias: list) -> Tuple[list, list]:
        
        """
        Performs a forward pass using a 2D data array, the weight matrices,
        and the bias vectors
        """

        train_array = _validate_arrays_perceptron(training_array)

        z = []
        a = [train_array]
        for i in range(len(weights)):
            z_layer = np.matmul(a[-1], weights[i]) + bias[i]
            z.append(z_layer)
            a_layer = _sigmoid(z_layer)
            a.append(a_layer)
            
        return z, a

    def _back_propagation(self, z: list, a: list, weights: list, training_targets: np.ndarray) -> Tuple[list, list]:

        """
        Performs backpropagation to calculate the error signal for each layer
        and the amount needed to update weights and biases
        """            

        train_targets = _validate_arrays_perceptron(training_targets)
        
        L = len(self.layers) - 1
        learning_rate = self.learning_rate
        delta = dict()
        delta[L] = (a[-1] - train_targets) * derivative_sigmoid(z[-1])
        d_weights = []
        d_bias = []

        for i in range(L - 1, 0, -1):
            delta[i] = (np.matmul(delta[i + 1], weights[i].T)) * derivative_sigmoid(z[i - 1])
    
        for j in range(1, L + 1):
            d_weights_layer = learning_rate * np.matmul(a[j - 1].T, delta[j])
            d_bias_layer = learning_rate * np.mean(delta[j], axis = 0)
            d_weights.append(d_weights_layer)
            d_bias.append(d_bias_layer)

        return d_weights, d_bias

    def _weight_update(self, weights: list, bias: list, d_weights: list, d_bias: list) -> Tuple[list, list]:
        
        """
        Updates the weights and biases
        """

        for i in range(len(weights)):
            weights[i] -= d_weights[i]
            bias[i] -= d_bias[i]

        return weights, bias

    def fit(self, 
            training_array: np.ndarray, 
            training_targets: np.ndarray, 
            random_state: Optional[int] = None) -> 'multilayer_Perceptron':
        
        """
        Fits the multi-layer Perceptron on given input data using stochastic
        gradient descent with encoded vectors for class

        Parameters
        ----------
        training_array: ArrayLike
            2D array-like object containing training data
        training_targets: ArrayLike
            2D array-like object containing training labels
        random_state: int, optional
            Random state for shuffling data using a random generator; if
            None, a randomized seed is used

        Returns
        -------
        multilayer_Perceptron
            Fitted multi-layer Perceptron model

        Raises
        ------
        ValueError
            If the input data does not match in size, or contains NaN or non-numeric values
            (for features)
        """

        train_array = _validate_arrays_perceptron(training_array)
        train_targets = _validate_arrays_perceptron(training_targets)
        
        self.classes_, encoded = np.unique(train_targets, return_inverse=True)
        n_classes = len(self.classes_)

        if train_targets.ndim == 1 or (train_targets.ndim == 2 and train_targets.shape[1] == 1):
            
            self.classes_, encoded = np.unique(train_targets, return_inverse=True)
            n_classes = len(self.classes_)
            
            if self.layers[-1] == 1:
                    if n_classes > 2:
                        raise ValueError(f"Single output neuron must have binary 2 classes, got {n_classes}")
                    train_targets = encoded.reshape(-1, 1)

            else:
                y = np.zeros((training_targets.shape[0], n_classes))
                y[np.arange(training_targets.shape[0]), encoded] = 1
                training_targets = y
        
        self.error_ = []

        weights, bias = self._weight_initialization(random_state = random_state)

        for _ in range(self.epochs):
            z, a = self._forward_layer(train_array, weights, bias)
            d_weights, d_bias = self._back_propagation(z, a, weights, train_targets)
            weights, bias = self._weight_update(weights, bias, d_weights, d_bias)

            pred_probs = a[-1]

            if self.layers[-1] == 1:
                pred_labels = (pred_probs > 0.5).astype(int)
            else:
                pred_labels = (pred_probs > 0.5).astype(int)

            error = np.sum(pred_labels != train_targets)
            self.error_.append(error)

        self.coef_ = weights
        self.bias_ = bias

        return self
    
    def _verify_fit(self) -> 'multilayer_Perceptron':

        """
        Verifies that the model is fitted
        """

        if self.coef_ is None or self.bias_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def prediction(self, testing_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Predicts the target class and respective class probabilities
        for given input samples

        Parameters
        ----------
        testing_array: np.ndarray
            2D array of size (n_samples, n_features)

        Returns
        -------
        prediction: np.ndarray
            Array containing probability predictions for each class, per sample
        predicted_labels: np.ndarray
            Array of predicted class for each sample
        """

        self._verify_fit()

        test_array = _validate_arrays_perceptron(testing_array)

        if test_array.shape[1] != self.coef_[0].shape[0]:
            raise ValueError('Test array must have the same number of input features as coefficients')

        _, a = self._forward_layer(testing_array, self.coef_, self.bias_)

        prediction = a[-1]

        if self.layers[-1] == 1:
            prediction = prediction.reshape(-1, 1)
            prediction = np.hstack([1 - prediction, prediction])
            predicted_labels = (prediction[:, 1] > 0.5).astype(int).reshape(-1, 1)
        
        else:
            predicted_labels = np.where(prediction > 0.5, 1, 0)

        return prediction, predicted_labels
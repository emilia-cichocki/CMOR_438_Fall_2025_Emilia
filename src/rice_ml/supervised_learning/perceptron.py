"""
    Perceptron algorithms (NumPy)

    This module implements the single-layer and multilayer perceptron 

    Functions
    ---------
    

    Classes
    ---------
   
"""

# TODO: finish above!

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
from rice_ml.supervised_learning.distances import _ensure_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]


def _validate_parameters(learning_rate: Optional[float],
                         epochs: Optional[int],
                         ) -> None:

    # TODO: add docstrings, potentially add functionality for collecting/graphing error counts

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

    # TODO: add docstrings

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
    return np.where(z > 0, 1, -1)

def _sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def derivative_sigmoid(z):
    return _sigmoid(z) * (1.0 - _sigmoid(z))
    

class Perceptron():
    
    def __init__(self,
                 epochs: int = 1000,
                 learning_rate: float = 0.01
                 ) -> None:

        _validate_parameters(learning_rate = learning_rate, epochs = epochs)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.class_mapping_: Optional[dict] = None
    
    def fit(self, training_array: np.ndarray, training_targets: np.ndarray, random_state: Optional[int] = None, shuffle: bool = True) -> 'Perceptron':
        
        # TODO: docstrings/comments 

        if not isinstance(shuffle, bool):
            raise TypeError('Shuffle must be a boolean')
        
        rng = _random_number(random_state)

        train_array, train_targets = _validate_arrays_perceptron(training_array, training_targets)
        
        classes = np.unique(training_targets)

        self.class_mapping_ = {-1: classes[0], 1: classes[1]}
        
        train_array = np.hstack([np.ones((train_array.shape[0], 1)), train_array])
        weights = rng.normal(loc=0, scale=0.01, size=train_array.shape[1]).reshape(-1)
        
        for iteration in range(self.epochs):
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
                weights -= learn_rate * error * x
        
        self.bias_ = weights[0]
        self.coef_ = weights[1:]

        return self

    def _verify_fit(self) -> 'Perceptron':
        if self.coef_ is None or self.bias_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    

    def prediction(self, testing_array: np.ndarray) -> np.ndarray:
        
        # TODO: doctrings/comments

        self._verify_fit()

        test_array = _validate_arrays_perceptron(testing_array)
        
        coef_array = _1D_vectorized(self.coef_) # TODO: fix this!

        if test_array.shape[1] != len(coef_array):
            raise ValueError('Test array must have the same number of input features as coefficients')
        
        bias = self.bias_
        prediction_value = _activation_function(np.matmul(test_array, coef_array) + bias)

        classification = np.array([self.class_mapping_[int(prediction)] for prediction in prediction_value])

        return classification
    
    def scoring(self, testing_array: ArrayLike, actual_targets: ArrayLike) -> np.ndarray:

        # TODO: be consistent w/ arraylike vs np.ndarray

        predicted_target_array = self.prediction(testing_array)
        actual_target_array = _1D_vectorized(actual_targets)

        if predicted_target_array.shape != actual_target_array.shape:
            raise ValueError("Shapes of predicted and actual targets must match")
        
        accuracy = np.mean(predicted_target_array == actual_target_array)

        return accuracy

class multilayer_Perceptron():

    def __init__(self,
                  layers: ArrayLike,
                  epochs: int = 1000,
                  learning_rate: float = 0.01
                 ) -> None:

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
        self.coef_: Optional[list] = None
        self.bias_: Optional[list] = None

    def _weight_initialization(self, random_state: Optional[int] = None) -> Tuple[list, list]:
        layers = self.layers
        weights = []
        bias = []
        rng = _random_number(random_state)
        for i in range(1, len(layers)):
            layer_weight = rng.standard_normal((layers[i - 1], layers[i]))
            layer_bias = np.zeros(layers[i]) # rng.standard_normal
            weights.append(layer_weight)
            bias.append(layer_bias)
        
        return weights, bias
    
    def _forward_layer(self, training_array: np.ndarray, weights: list, bias: list) -> Tuple[list, list]:
        
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
        
        for i in range(len(weights)):
                weights[i] -= d_weights[i]
                bias[i] -= d_bias[i]

        return weights, bias

    def fit(self, training_array: np.ndarray, training_targets: np.ndarray, random_state: Optional[int] = None) -> 'multilayer_Perceptron':
        
        train_array = _validate_arrays_perceptron(training_array)
        train_targets = _validate_arrays_perceptron(training_targets)

        weights, bias = self._weight_initialization(random_state = random_state)

        for _ in range(self.epochs):
            z, a = self._forward_layer(train_array, weights, bias)
            d_weights, d_bias = self._back_propagation(z, a, weights, train_targets)
            weights, bias = self._weight_update(weights, bias, d_weights, d_bias)

        self.coef_ = weights
        self.bias_ = bias

        return self
    
    def _verify_fit(self) -> 'multilayer_Perceptron':
        if self.coef_ is None or self.bias_ is None:
            raise RuntimeError("Model is not fitted; call fit(training_array, training_targets)")

        return self
    
    def predict(self, testing_array: np.ndarray):
        
        # TODO: doctrings/comments

        self._verify_fit()

        test_array = _validate_arrays_perceptron(testing_array)

        if test_array.shape[1] != self.coef_[0].shape[0]:
            raise ValueError('Test array must have the same number of input features as coefficients')

        _, a = self._forward_layer(testing_array, self.coef_, self.bias_)

        prediction = a[-1]

        predicted_labels = np.where(prediction > 0.5, 1, 0)

        return prediction, predicted_labels

# TODO: maybe move this to another section (postprocessing)

    def scoring(self, testing_array: ArrayLike, actual_targets: ArrayLike) -> np.ndarray:

        # TODO: be consistent w/ arraylike vs np.ndarray

        predicted_target_array = self.prediction(testing_array)
        actual_target_array = _1D_vectorized(actual_targets)

        if predicted_target_array.shape != actual_target_array.shape:
            raise ValueError("Shapes of predicted and actual targets must match")
        
        accuracy = np.mean(predicted_target_array == actual_target_array)

        return accuracy

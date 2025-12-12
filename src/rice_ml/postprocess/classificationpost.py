
"""
    Postprocessing utilities for classification (Numpy)
    # TODO: finish this!!

"""

import numpy as np
from typing import *
import matplotlib.pyplot as plt
from rice_ml.preprocess.datatype import *
from scipy import stats # TODO: finish this!
from rice_ml.supervised_learning.distances import _ensure_numeric

__all__ = [
    'accuracy_score',
    'confusion_matrix',
    'precision_score',
    'recall_score',
    'f1_score',
    'roc_auc',
    'log_loss',
]

def _validate_vector_match(predicted_classes: np.ndarray, true_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    pred_class = _1D_vectorized(predicted_classes)
    true_class = _1D_vectorized(true_classes)
    if pred_class.shape[0] != true_class.shape[0]:
        raise ValueError('Predicted and true labels must be of the same length')
    
    return pred_class, true_class

def _class_counts(predicted_classes: np.ndarray, 
                  true_classes: np.ndarray, 
                  labels: Optional[Sequence] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    pred_class, true_class = _validate_vector_match(predicted_classes, true_classes)
    
    if labels is not None and not isinstance(labels, Sequence):
        raise TypeError('Labels must be a sequence')
    
    if labels is None:
        labels = np.unique(np.concatenate([pred_class, true_class]))
    
    labels = np.asarray(labels)
    label_length = len(labels)

    label_to_index = {label: i for i, label in enumerate(labels)}

    pred_class_index = np.array([label_to_index.get(label, -1) for label in pred_class])
    true_class_index = np.array([label_to_index.get(label, -1) for label in true_class])

    confusion_matrix = np.zeros((label_length, label_length))

    for true, pred in zip(true_class_index, pred_class_index):
        if true == -1 or pred == -1:
            continue
        confusion_matrix[true, pred] += 1
    
    true_pos = np.diag(confusion_matrix).astype(float)
    false_pos = confusion_matrix.sum(axis = 0) - true_pos
    false_neg = confusion_matrix.sum(axis = 1) - true_pos
    
    return true_pos, false_pos, false_neg, labels, confusion_matrix

def _plot_confusion(confusion_matrix: np.ndarray, labels: Optional[Sequence] = None, display_value: bool = True) -> None:
    
    if not isinstance(confusion_matrix, np.ndarray):
        raise TypeError("Confusion matrix must be an array")
    confusion_matrix = _2D_numeric(confusion_matrix)
    if labels is not None and not isinstance(labels, Sequence):
        raise TypeError("Labels must be a sequence")
    if not isinstance(display_value, bool):
        raise TypeError("display_value parameter must be a boolean")
    
    fig, ax = plt.subplots(figsize = (10, 6))
    im = ax.imshow(confusion_matrix)
    plt.colorbar(im, ax = ax)

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    if display_value:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, confusion_matrix[i, j], ha = 'center', va = 'center')

    plt.tight_layout()
    plt.show()

def _validate_probability(probabilities: np.ndarray, true_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    
    true_class = _ensure_numeric(true_classes)

    if not isinstance(probabilities, np.ndarray):
        raise TypeError("Probabilities must be an array")
    
    if probabilities.ndim == 1:
        probabilities = _ensure_numeric(probabilities)
        true_class, prob = _validate_vector_match(true_class, probabilities)
        prob_overall = np.stack([1 - prob, prob], axis = 1)
        n_classes = 2
    
    elif probabilities.ndim == 2:
        probabilities = _2D_numeric(probabilities)

        if probabilities.shape[0] != true_class.shape[0]:
            raise ValueError("Probabilities must have the same first dimension (number of samples) as true classes")
        
        prob_overall = probabilities.astype(float)
        n_classes = prob_overall.shape[1]
    
    else:
        raise ValueError("Probability array must be either one or two-dimensional")
    
    if np.any(prob_overall < 0) or np.any(prob_overall > 1) or np.any(~np.isfinite(prob_overall)):
        raise ValueError("Probabilities must be between 0 and 1")

    return true_class, prob_overall, n_classes

def accuracy_score(predicted_classes: np.ndarray, true_classes: np.ndarray) -> float:

    # TODO: type hints/docsrings

    pred_class, true_class = _validate_vector_match(predicted_classes, true_classes)
    accuracy = float(np.mean(pred_class == true_class))

    return accuracy

def confusion_matrix(predicted_classes: np.ndarray, 
                     true_classes: np.ndarray, 
                     plot: bool = True, 
                     display_values: bool = True, 
                     labels: Optional[Sequence] = None) -> np.ndarray:

    if not isinstance(plot, bool):
        raise TypeError('Plot parameter must be a boolean')
    
    _, _, _, labels, confusion_matrix = _class_counts(predicted_classes, true_classes, labels)
    
    labels = labels.tolist()

    if plot:
        _plot_confusion(confusion_matrix, labels, display_values)
    
    return confusion_matrix

def precision_score(predicted_classes: np.ndarray, 
                    true_classes: np.ndarray, 
                    metric: Optional[Literal['binary', 'micro', 'macro']] = 'binary',
                    labels: Optional[Sequence] = None) -> Union[float, np.ndarray]:
    
    pred_class, true_class = _validate_vector_match(predicted_classes, true_classes)

    if metric not in ['binary', 'micro', 'macro', None]:
        raise ValueError(f"Metric must be one of {['binary', 'micro', 'macro', None]}")
    
    if metric == 'binary':
        unique_class = np.unique(np.concatenate([pred_class, true_class]))
        if len(unique_class) != 2:
            raise ValueError("Binary metric requires two classes")
        positive_class_label = sorted(unique_class)[-1]

        true_positive = np.sum((true_class == positive_class_label) & (pred_class == positive_class_label))
        false_positive = np.sum((true_class != positive_class_label) & (pred_class == positive_class_label))
        try:
            precision = float(true_positive / (true_positive + false_positive))
        except ZeroDivisionError:
            precision = 0.0
        return precision

    true_pos, false_pos, false_neg, labels_out, _ = _class_counts(pred_class, true_class, labels)

    if metric == 'micro':
        true_positive = np.sum(true_pos)
        false_positive = np.sum(false_pos)

        try:
            precision = float(true_positive / (true_positive + false_positive))
        except ZeroDivisionError:
            precision = 0.0
        return precision
    
    with np.errstate(divide = "ignore", invalid = "ignore"):
        precision_per_class = np.where((true_pos + false_pos) > 0, true_pos / (true_pos + false_pos), 0.0)

    if metric == 'macro':
        precision = float(np.mean(precision_per_class))
        return precision
    
    if metric is None:
        return precision_per_class

def recall_score(predicted_classes: np.ndarray, 
                    true_classes: np.ndarray, 
                    metric: Optional[Literal['binary', 'micro', 'macro']] = 'binary',
                    labels: Optional[Sequence] = None) -> Union[float, np.ndarray]:
    
    pred_class, true_class = _validate_vector_match(predicted_classes, true_classes)

    if metric not in ['binary', 'micro', 'macro', None]:
        raise ValueError(f"Metric must be one of {['binary', 'micro', 'macro', None]}")
    
    if metric == 'binary':
        unique_class = np.unique(np.concatenate([pred_class, true_class]))
        
        if len(unique_class) != 2:
            raise ValueError("Binary metric requires two classes")
        positive_class_label = sorted(unique_class)[-1]

        true_positive = np.sum((true_class == positive_class_label) & (pred_class == positive_class_label))
        false_negative = np.sum((true_class == positive_class_label) & (pred_class != positive_class_label))
        try:
            recall = float(true_positive / (true_positive + false_negative))
        except ZeroDivisionError:
            recall = 0.0
        return recall

    true_pos, false_pos, false_neg, labels, _ = _class_counts(pred_class, true_class, labels)

    if metric == 'micro':
        true_positive = np.sum(true_pos)
        false_negative = np.sum(false_neg)

        try:
            recall = float(true_positive / (true_positive + false_negative))
        except ZeroDivisionError:
            recall = 0.0
        return recall
    
    with np.errstate(divide = "ignore", invalid = "ignore"):
        recall_per_class = np.where((true_pos + false_neg) > 0, true_pos / (true_pos + false_neg), 0.0)

    if metric == 'macro':
        precision = float(np.mean(recall_per_class))
        return precision
    
    if metric is None:
        return recall_per_class
        
def f1_score(predicted_classes: np.ndarray, 
                    true_classes: np.ndarray, 
                    metric: Optional[Literal['binary', 'micro', 'macro']] = 'binary',
                    labels: Optional[Sequence] = None) -> Union[float, np.ndarray]:
    
    pred_class, true_class = _validate_vector_match(predicted_classes, true_classes)

    if metric not in ['binary', 'micro', 'macro', None]:
        raise ValueError(f"Metric must be one of {['binary', 'micro', 'macro', None]}")
    
    if metric == 'binary':
        precision = precision_score(pred_class, true_class, 'binary')
        recall = recall_score(pred_class, true_class, 'binary')
        try:
            f1 = float(2 * (precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1 = 0.0
        return f1

    if metric == 'micro':
        precision = precision_score(pred_class, true_class, 'micro')
        recall = recall_score(pred_class, true_class, 'micro')
        try:
            f1 = float(2 * (precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f1 = 0.0
        return f1
    
    true_pos, false_pos, false_neg, labels, _ = _class_counts(pred_class, true_class, labels)

    with np.errstate(divide = "ignore", invalid = "ignore"):
        precision_per_class = np.where((true_pos + false_pos) > 0, true_pos / (true_pos + false_pos), 0.0)
        recall_per_class = np.where((true_pos + false_neg) > 0, true_pos / (true_pos + false_neg), 0.0)
        f1_per_class = np.where((precision_per_class + recall_per_class) > 0, (2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)), 0.0)

    if metric == 'macro':
        f1 = float(np.mean(f1_per_class))
        return f1
    
    if metric is None:
        return f1_per_class
    
def roc_auc(predicted_scores: np.ndarray, true_classes: np.ndarray) -> float:

    # TODO: type hints/docstrings = rank-based method

    pred_scores = _ensure_numeric(predicted_scores)
    pred_scores, true_class = _validate_vector_match(pred_scores, true_classes)

    unique_class = np.unique(true_class)
    
    if len(unique_class) != 2:
        raise ValueError("ROC AUC requires two classes")
    
    ranks = stats.rankdata(pred_scores, method = "average")
    positive_class_idx = (true_class == sorted(unique_class)[-1])
    n_positive = np.sum(positive_class_idx)
    n_negative = len(true_class) - n_positive
    sum_ranks_positive = np.sum(ranks[positive_class_idx])
    auc = float((sum_ranks_positive - n_positive * (n_negative + 1) / 2.0) / (n_positive * n_negative))

    return auc

def log_loss(predicted_scores: np.ndarray, true_classes: np.ndarray, epsilon: float = 1e-10) -> float:

    # TODO: type hints, docstrings

    true_class, pred_scores, n_classes = _validate_probability(predicted_scores, true_classes)

    if not isinstance(epsilon, (float, int)):
        raise TypeError("Epsilon must be a float or integer")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("Epsilon must be greater than zero and finite")
    
    unique_class = np.unique(true_class)

    if n_classes == 2 and len(unique_class) == 2:
        label_to_column = {unique_class.min(): 0, unique_class.max(): 1}
    
    else:
        if len(unique_class) != n_classes:
            unique_class = np.arange(n_classes)
        label_to_column = {label: i for i, label in enumerate(unique_class)}
    
    if pred_scores.ndim == 2:
        prob_sums = pred_scores.sum(axis = 1)
        if not np.allclose(prob_sums, 1.0):
            raise ValueError("Probability across each row must sum to (approximately) 1")
        
    probabilities = np.clip(pred_scores, epsilon, 1.0)
    
    columns = np.array([label_to_column.get(label, None) for label in true_class])
    if np.any(columns == None):
        raise ValueError("Some labels not mapped to probability columns")
    
    log_loss = -np.log(probabilities[np.arange(len(true_class)), columns.astype(int)])
    log_loss_mean = float(np.mean(log_loss))

    return log_loss_mean
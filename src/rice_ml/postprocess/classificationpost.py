
"""
    Postprocessing utilities for classification (NumPy)
    
    This module calculates and visualizes a comprehensive set of postprocessing 
    and evaluation metrics for classification algorithms, with support for 
    NumPy arrays.

    Functions
    ---------
    accuracy_score
        Computes accuracy score using predicted and true classes
    confusion_matrix
        Calculates the confusion matrix (optional plotting)
    precision_score
        Computes precision score using predicted and true classes
    recall_score
        Computes recall score using predicted and true classes
    f1_score
        Computes F1 score using predicted and true classes
    roc_auc
        Computes the AUC score for binary classification
    log_loss
        Computes the log loss for binary or multi-class classification
    print_model_metrics
        Prints accuracy and micro/macro precision, recall, and F1

"""

import numpy as np
from typing import *
import matplotlib.pyplot as plt
from rice_ml.preprocess.datatype import *
from scipy import stats
from rice_ml.supervised_learning.distances import _ensure_numeric
import seaborn as sns

__all__ = [
    'accuracy_score',
    'confusion_matrix',
    'precision_score',
    'recall_score',
    'f1_score',
    'roc_auc',
    'log_loss',
    'print_model_metrics'
]

def _validate_vector_match(predicted_classes: np.ndarray, true_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Validation of input vectors
    
    Converts both inputs to 1D arrays of an arbitrary data type and checks 
    that they have the same length

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels

    Returns
    -------
    pred_class: np.ndarray
        1D array of predicted class labels
    true_class: np.ndarray
        1D array of true class labels

    Raises
    ------
    ValueError
        If predicted and true class arrays are of different lengths
    """

    pred_class = _1D_vectorized(predicted_classes)
    true_class = _1D_vectorized(true_classes)
    if pred_class.shape[0] != true_class.shape[0]:
        raise ValueError('Predicted and true labels must be of the same length')
    
    return pred_class, true_class

def _class_counts(predicted_classes: np.ndarray, 
                  true_classes: np.ndarray, 
                  labels: Optional[Sequence] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Classification metrics by class

    Calculates the true positive, false positive, and false negative scores per class,
    as well as the confusion matrix and input labels 

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels
    labels: sequence, optional
        Sequence of class labels

    Returns
    -------
    true_pos: np.ndarray
        1D array of true positives by class
    false_pos: np.ndarray
        1D array of false positives by class
    false_neg: np.ndarray
        1D array of false negatives by class
    labels: np.ndarray
        1D array of class labels
    confusion_matrix
        2D confusion matrix where rows correspond to true labels and
        columns correspond to predicted labels
    
    Raises
    ------
    TypeError
        If labels are not a sequence
    """

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
    
    """
    Plot a confusion matrix heatmap

    Parameters
    ----------
    confusion_matrix: np.ndarray
        2D numeric array representing the confusion matrix, with rows corresponding
        to true labels and columns to predicted labels
    labels: sequence, optional
        Sequence of labels to display on confusion matrix axes; must match
        dimensions of the matrix
    display_value: bool, default = True
        Determines whether to display the corresponding numbers in each cell

    Raises
    ------
    TypeError
        If `confusion_matrix` is not a NumPy array, input `labels` are not a
        sequence, or the `display_value` parameter is not a boolean
    ValueError
        If `confusion_matrix` cannot be converted to a 2D numeric array
    """
    
    if not isinstance(confusion_matrix, np.ndarray):
        raise TypeError("Confusion matrix must be an array")
    confusion_matrix = _2D_numeric(confusion_matrix)
    if labels is not None and not isinstance(labels, Sequence):
        raise TypeError("Labels must be a sequence")
    if not isinstance(display_value, bool):
        raise TypeError("display_value parameter must be a boolean")
    
    fig, ax = plt.subplots(figsize = (10, 6))
    im = ax.imshow(confusion_matrix, cmap=sns.light_palette('gray', as_cmap=True))
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
    
    """
    Validation of the class probability predictions

    Ensures that the probability array is a numeric array matching the
    number of samples, and only contains values ranging from 0 to 1

    Parameters
    ----------
    probabilities: np.ndarray
        Array of predicted class probabilities
        May be either:
        - 1D array of shape (n_samples,) for binary classification
          with probabilities for the positive class
        - 2D array of shape (n_samples, n_classes) for multi-class
          classification with probabilities for each class
    true_classes: np.ndarray
        Array of true class labels

    Returns
    -------
    true_class: np.ndarray
        1D array of class labels
    prob_overall: np.ndarray
        2D array of validated probabilities of shape (n_samples, n_classes)
    n_classes: int
        Total number of classes

    Raises
    ------
    TypeError
        If `probabilities` is not a NumPy array.
    ValueError
        If the dimensions of the probability array are incorrect, do not match
        the number of samples in `true_classes`, or have values beyond 0 to 1
    """

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

    """
    Calculates accuracy score

    Calculates accuracy as the number of correctly predicted labels divided
    by the total number of samples

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels

    Returns
    -------
    accuracy: float
        Accuracy score between 0 and 1

    Raises
    ------
    ValueError
        If predicted and true class arrays have different lengths

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> accuracy_score(y_pred, y_true)
    0.75
    """

    pred_class, true_class = _validate_vector_match(predicted_classes, true_classes)
    accuracy = float(np.mean(pred_class == true_class))

    return accuracy

def confusion_matrix(predicted_classes: np.ndarray, 
                     true_classes: np.ndarray, 
                     plot: bool = True, 
                     display_values: bool = True, 
                     labels: Optional[Sequence] = None,
                     conf_matrix_labels: Optional[list] = None,
                     ) -> np.ndarray:

    """
    Calculates the confusion matrix, with the option to plot

    Calculates each element in the confusion matrix as the number of occurrences
    for a predicted label and true label pair, and displays an optional
    heatmap of the matrix

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels
    plot: bool, default = True
        Whether to display a heatmap plot of the confusion matrix
    display_value: bool, default = True
        Determines whether to display the corresponding numbers in each cell
    labels: sequence, optional
        Sequence of labels corresponding to classes for class counts
    conf_matrix_labels: list, optional
        List of labels to display on confusion matrix axes; must match
        dimensions of the matrix

    Returns
    -------
    confusion_matrix: np.ndarray
        2D array with shape (n_labels, n_labels) representing the confusion matrix; 
        rows correspond to true labels and columns to predicted labels

    Raises
    ------
    TypeError
        If `plot` is not a boolean or `conf_matrix_labels` is not a list

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> confusion_matrix(y_pred, y_true, plot = False)
    array([[2., 0.],
           [1., 1.]])
    """

    if not isinstance(plot, bool):
        raise TypeError('Plot parameter must be a boolean')
    if conf_matrix_labels is not None and not isinstance(conf_matrix_labels, list):
        raise TypeError('Confusion matrix labels must be a list')
    
    _, _, _, labels, confusion_matrix = _class_counts(predicted_classes, true_classes, labels)
    
    labels = labels.tolist()

    if plot:
        _plot_confusion(confusion_matrix, conf_matrix_labels, display_values)
    
    return confusion_matrix

def precision_score(predicted_classes: np.ndarray, 
                    true_classes: np.ndarray, 
                    metric: Optional[Literal['binary', 'micro', 'macro']] = 'binary',
                    labels: Optional[Sequence] = None) -> Union[float, np.ndarray]:
    
    """
    Calculates precision score

    Precision is calculated as the number of true positives divided by
    the total number of predicted positives. This function calculates binary,
    micro, macro, or per-class precision

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels
    metric: {'binary', 'micro', 'macro'}, optional, default = 'binary'
        Metric for calculating precision
        - 'binary': precision for positive class in binary classification
        - 'micro': calculated globally across all classes
        - 'macro': calculated as the unweighted average of per-class precision
        - None: returns an array of precision per class
    labels: sequence, optional
        Sequence of class labels; if None, these are inferred using the
        predicted and true class arrays
    
    Returns
    -------
    float or np.ndarray
        Precision score as a float ('binary', 'micro', 'macro') or as a 1D
        array with per-class values ('None')

    Raises
    ------
    ValueError
        If `metric` is not in {'binary', 'micro', 'macro', None}, or
        if more than two classes are used with 'binary' metric

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> precision_score(y_pred, y_true, metric = 'binary')
    1.0
    """

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
    
    """
    Calculates recall score

    Recall is calculated as the number of true positives divided by
    the total number of ground-truth positives. This function calculates binary,
    micro, macro, or per-class recall

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels
    metric: {'binary', 'micro', 'macro'}, optional, default = 'binary'
        Metric for calculating recall
        - 'binary': recall for positive class in binary classification
        - 'micro': calculated globally across all classes
        - 'macro': calculated as the unweighted average of per-class recall
        - None: returns an array of recall per class
    labels: sequence, optional
        Sequence of class labels; if None, these are inferred using the
        predicted and true class arrays
    
    Returns
    -------
    float or np.ndarray
        Recall score as a float ('binary', 'micro', 'macro') or as a 1D
        array with per-class values ('None')

    Raises
    ------
    ValueError
        If `metric` is not in {'binary', 'micro', 'macro', None}, or
        if more than two classes are used with 'binary' metric

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> recall_score(y_pred, y_true, metric = 'binary')
    0.5
    """
     
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
    
    """
    Calculates F1 score

    F1 is calculated as the harmonic mean of precision and recall. This function 
    calculates binary, micro, macro, or per-class F1 scores

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels
    metric: {'binary', 'micro', 'macro'}, optional, default = 'binary'
        Metric for calculating F1
        - 'binary': F1 for positive class in binary classification
        - 'micro': calculated globally across all classes
        - 'macro': calculated as the unweighted average of per-class F1
        - None: returns an array of F1 per class
    labels: sequence, optional
        Sequence of class labels; if None, these are inferred using the
        predicted and true class arrays
    
    Returns
    -------
    float or np.ndarray
        F1 score as a float ('binary', 'micro', 'macro') or as a 1D
        array with per-class values ('None')

    Raises
    ------
    ValueError
        If `metric` is not in {'binary', 'micro', 'macro', None}, or
        if more than two classes are used with 'binary' metric

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> recall_score(y_pred, y_true, metric = 'micro')
    0.75
    """
    
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

    """
    Calculates area under the curve (AUC) for the Receiver Operating Characteristic (ROC)

    Calculates ROC AUC for binary classification using the Wilcoxon rank-sum formula;
    measures ability of the model to differentiate between positive and negative classes

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted class probabilities
    true_classes: np.ndarray
        Array of true class labels

    Returns
    -------
    auc: float
        ROC AUC score between 0 and 1

    Raises
    ------
    ValueError
        If more than two classes are present in `true_classes`

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.1, 0.9, 0.8, 0.2])
    >>> roc_auc(y_scores, y_true)
    1.0
    """

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
    auc = float((sum_ranks_positive - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))

    return auc

def log_loss(predicted_scores: np.ndarray, true_classes: np.ndarray, epsilon: float = 1e-10) -> float:

    """
    Calculates log loss for binary and multi-class classification

    Calculates log loss (binary cross-entropy loss) from predicted probabilities
    to evaluate model performance while accounting for the confidence of predictions

    Parameters
    ----------
    predicted_scores: np.ndarray
        Array of predicted class probabilities
    true_classes: np.ndarray
        Array of true class labels
    epsilon: float, default = 1e-10
        Used to clip probabilities to avoid undefined calculations

    Returns
    -------
    log_loss_mean: float
        Average log loss score across all samples

    Raises
    ------
    TypeError
        If `epsilon` is not a float or integer
    ValueError
        If `epsilon` is not a finite positive value, probabilites do not
        total 1 across rows, or true class labels cannot be mapped to columns

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_scores = np.array([0.9, 0.8, 0.3, 0.1])
    >>> log_loss(y_scores, y_true)
    0.9587654910730045
    """

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

def print_model_metrics(predicted_classes: np.ndarray, true_classes: np.ndarray) -> None:
    
    """
    Prints the set of model evaluation metrics for classification

    Includes accuracy, precision, recall, and F1 scores calculated using both
    micro and macro metrics

    Parameters
    ----------
    predicted_classes: np.ndarray
        Array of predicted class labels
    true_classes: np.ndarray
        Array of true class labels

    Raises
    ------
    ValueError
        If the dimensions of `predicted_classes` and `true_classes` do not match
    """
    
    pred_class, true_class = _validate_vector_match(predicted_classes, true_classes)

    print(f"Model Metrics \n\
{'-' *13} \n\
Accuracy: {accuracy_score(pred_class, true_class):.2f} \n\
Precision (Micro): {precision_score(pred_class, true_class, 'micro'):.2f} \n\
Precision (Macro): {precision_score(pred_class, true_class, 'macro'):.2f} \n\
Recall (Micro): {recall_score(pred_class, true_class, 'micro'):.2f} \n\
Recall (Macro): {recall_score(pred_class, true_class, 'macro'):.2f} \n\
F1 (Micro): {f1_score(pred_class, true_class, 'micro'):.2f} \n\
F1 (Macro): {f1_score(pred_class, true_class, 'macro'):.2f}")
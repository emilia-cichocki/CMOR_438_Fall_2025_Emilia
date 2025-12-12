
import numpy as np
import pytest
from rice_ml.postprocess.classificationpost import _validate_vector_match, _class_counts, _validate_probability, _plot_confusion, accuracy_score, precision_score, recall_score, f1_score, roc_auc, log_loss, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

# TODO: formatting

def test_validate_vector_match_basic():
    test_pred_array = np.array([0, 1, 1, 0])
    test_true_array = np.array([1, 0, 0, 1])
    pred, true = _validate_vector_match(test_pred_array, test_true_array)
    assert isinstance(pred, np.ndarray)
    assert isinstance(true, np.ndarray)
    assert pred.shape == (4,)
    assert true.shape == (4,)

def test_validate_vector_match_strings():
    test_pred_array = np.array(['0', '1', '1', '1'])
    test_true_array = np.array(['1', '0', '0', '1'])
    pred, true = _validate_vector_match(test_pred_array, test_true_array)
    assert isinstance(pred, np.ndarray)
    assert isinstance(true, np.ndarray)
    assert pred.shape == (4,)
    assert true.shape == (4,)

def test_validate_vector_match_string_nonnumeric():
    test_pred_array = np.array(['A', 'A', 'B', 'B'])
    test_true_array = np.array(['A', 'A', 'B', 'B'])
    pred, true = _validate_vector_match(test_pred_array, test_true_array)
    assert isinstance(pred, np.ndarray)
    assert isinstance(true, np.ndarray)
    assert pred.shape == (4,)
    assert true.shape == (4,)

def test_validate_vector_match_dim_pred():
    test_pred_array = np.array([[0, 1, 1, 0]])
    test_true_array = np.array([1, 0, 0, 1])
    with pytest.raises(ValueError):
        _validate_vector_match(test_pred_array, test_true_array)

def test_validate_vector_match_dim_true():
    test_pred_array = np.array([0, 1, 1, 0])
    test_true_array = np.array([[1, 0, 0, 1]])
    with pytest.raises(ValueError):
        _validate_vector_match(test_pred_array, test_true_array)

def test_validate_vector_match_shape_mismatch():
    test_pred_array = np.array([0, 1, 1, 0])
    test_true_array = np.array([1, 0, 0, 1, 0])
    with pytest.raises(ValueError):
        _validate_vector_match(test_pred_array, test_true_array)

def test_class_counts_basic():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0])
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [1, 1])
    assert np.allclose(false_pos, [1, 1])
    assert np.allclose(false_neg, [1, 1])
    assert np.allclose(labels, [0, 1])
    assert np.allclose(conf_matrix, [[1, 1], [1, 1]])

def test_class_counts_labels():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A'])
    labels = ['A', 'B']
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true, labels)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [1, 1])
    assert np.allclose(false_pos, [1, 1])
    assert np.allclose(false_neg, [1, 1])
    assert np.array_equal(labels, ['A', 'B'])
    assert np.allclose(conf_matrix, [[1, 1], [1, 1]])

def test_class_counts_labels_none():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A'])
    labels = ['A', 'B', 'C']
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true, labels)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [1, 1, 0])
    assert np.allclose(false_pos, [1, 1, 0])
    assert np.allclose(false_neg, [1, 1, 0])
    assert np.array_equal(labels, ['A', 'B', 'C'])
    assert np.allclose(conf_matrix, [[1, 1, 0], [1, 1, 0], [0, 0, 0]])

def test_class_counts_labels_numeric():
    test_true = np.array([2, 3, 3, 2])
    test_pred = np.array([2, 3, 2, 3])
    labels = [2, 3]
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true, labels)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [1, 1])
    assert np.allclose(false_pos, [1, 1])
    assert np.allclose(false_neg, [1, 1])
    assert np.array_equal(labels, [2, 3])
    assert np.allclose(conf_matrix, [[1, 1], [1, 1]])

def test_class_counts_unknown_label():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 2])
    labels = [0, 1]
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true, labels)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [1, 1])
    assert np.allclose(false_pos, [0, 1])
    assert np.allclose(false_neg, [1, 0])
    assert np.allclose(conf_matrix, [[1, 1], [0, 1]])

def test_class_counts_multi_class():
    test_true = np.array([0, 0, 1, 2])
    test_pred = np.array([0, 1, 1, 2])
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [1, 1, 1])
    assert np.allclose(false_pos, [0, 1, 0])
    assert np.allclose(false_neg, [1, 0, 0])
    assert np.allclose(labels, [0, 1, 2])
    assert np.allclose(conf_matrix, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])

def test_class_counts_no_rep():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([1, 1, 1, 1])
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [0, 2])
    assert np.allclose(false_pos, [0, 2])
    assert np.allclose(false_neg, [2, 0])
    assert np.allclose(labels, [0, 1])
    assert np.allclose(conf_matrix, [[0, 2], [0, 2]])

def test_class_counts_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 0, 0])
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [4])
    assert np.allclose(false_pos, [0])
    assert np.allclose(false_neg, [0])
    assert np.allclose(labels, [0])
    assert np.allclose(conf_matrix, [[4]])

def test_class_counts_label_tuple():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = (0, 1)
    true_pos, false_pos, false_neg, labels, conf_matrix = _class_counts(test_pred, test_true, labels)
    assert isinstance(true_pos, np.ndarray)
    assert isinstance(false_pos, np.ndarray)
    assert isinstance(false_neg, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(conf_matrix, np.ndarray)
    assert np.allclose(true_pos, [1, 1])
    assert np.allclose(false_pos, [1, 1])
    assert np.allclose(false_neg, [1, 1])
    assert np.allclose(conf_matrix, [[1, 1], [1, 1]])

def test_class_counts_label_array():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = np.array([0, 1])
    with pytest.raises(TypeError):
        _class_counts(test_pred, test_true, labels)

def test_class_counts_dimension_true():
    test_true = np.array([[0, 0, 1, 1]])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        _class_counts(test_pred, test_true)

def test_class_counts_dimension_pred():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([[0, 1, 1, 0]])
    with pytest.raises(ValueError):
        _class_counts(test_pred, test_true)

def test_plot_confusion_basic():
    test_cm = np.array([[1, 1], [1, 1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _plot_confusion(test_cm)

def test_plot_confusion_dif_values():
    test_cm = np.array([[2, 1], [3, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _plot_confusion(test_cm)

def test_plot_confusion_labels():
    labels = [0, 1]
    test_cm = np.array([[2, 1], [3, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _plot_confusion(test_cm, labels)

def test_plot_confusion_labels_string():
    labels = ['A', 'B']
    test_cm = np.array([[2, 1], [3, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _plot_confusion(test_cm, labels)

def test_plot_confusion_display_value():
    labels = [0, 1]
    test_cm = np.array([[2, 1], [3, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _plot_confusion(test_cm, labels, False)

def test_plot_confusion_array_type():
    test_cm = [[2, 1], [3, 0]]
    with pytest.raises(TypeError):
        _plot_confusion(test_cm)

def test_plot_confusion_array_dim():
    test_cm = np.array([[[2, 1], [3, 0]]])
    with pytest.raises(ValueError):
        _plot_confusion(test_cm)

def test_plot_confusion_label_type():
    test_cm = np.array([[2, 1], [3, 0]])
    labels = np.array([0, 1])
    with pytest.raises(TypeError):
        _plot_confusion(test_cm, labels)

def test_plot_confusion_display_value_type():
    test_cm = np.array([[2, 1], [3, 0]])
    with pytest.raises(TypeError):
        _plot_confusion(test_cm, display_value = 'False')

def test_validate_probability_basic():
    test_true = np.array([0, 1, 0, 1])
    test_prob = np.array([0.1, 0.8, 0.3, 0.6])
    true, prob, n_class = _validate_probability(test_prob, test_true)
    assert isinstance(true, np.ndarray)
    assert isinstance(prob, np.ndarray)
    assert isinstance(n_class, int)
    assert true.shape == (4,)
    assert np.allclose(true, test_true)
    assert prob.shape == (4, 2)
    assert np.allclose(prob, np.stack([np.array([0.9, 0.2, 0.7, 0.4]), test_prob], axis = 1))
    assert n_class == 2

def test_validate_probability_2D():
    test_true = np.array([0, 1, 2])
    test_prob = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])
    true, prob, n_class = _validate_probability(test_prob, test_true)
    assert isinstance(true, np.ndarray)
    assert isinstance(prob, np.ndarray)
    assert isinstance(n_class, int)
    assert true.shape == (3,)
    assert np.allclose(true, test_true)
    assert prob.shape == (3, 3)
    assert np.allclose(prob, test_prob)
    assert n_class == 3

def test_validate_probability_type_true():
    test_true = np.array(['A', 'B', 'A', 'B'])
    test_prob = np.array([0.1, 0.8, 0.3, 0.6])
    with pytest.raises(TypeError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_type_prob_data():
    test_true = np.array([0, 1, 0, 1])
    test_prob = np.array(['A', 0.8, 0.3, 0.6])
    with pytest.raises(TypeError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_2D_type_prob_data():
    test_true = np.array([0, 1, 2])
    test_prob = np.array([
        ['A', 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])
    with pytest.raises(TypeError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_type_prob():
    test_true = np.array([0, 1, 0, 1])
    test_prob = 'np.array([A, 0.8, 0.3, 0.6])'
    with pytest.raises(TypeError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_type_shape_mismatch_1D():
    test_true = np.array([0, 1, 0, 1])
    test_prob = np.array([0.2, 0.8, 0.3, 0.6, 0.5])
    with pytest.raises(ValueError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_type_shape_mismatch_2D():
    test_true = np.array([0, 1])
    test_prob = np.array([
        [0.8, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])
    with pytest.raises(ValueError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_dimension_prob():
    test_true = np.array([0, 1, 0, 1])
    test_prob = np.array([[[0.2, 0.8, 0.3, 0.6]]])
    with pytest.raises(ValueError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_range():
    test_true = np.array([0, 1, 0, 1])
    test_prob = np.array([1.1, 0.8, 0.3, 0.6])
    with pytest.raises(ValueError):
        _validate_probability(test_prob, test_true)

def test_validate_probability_finite():
    test_true = np.array([0, 1, 0, 1])
    test_prob = np.array([np.inf, 0.8, 0.3, 0.6])
    with pytest.raises(ValueError):
        _validate_probability(test_prob, test_true)

def test_accuracy_score_basic():
    test_true = np.array([0, 1, 0, 0])
    test_pred = np.array([0, 1, 0, 1])
    test_acc = accuracy_score(test_pred, test_true)
    assert isinstance(test_acc, float)
    assert test_acc == 0.75

def test_accuracy_score_perfect():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 0, 1])
    test_acc = accuracy_score(test_pred, test_true)
    assert isinstance(test_acc, float)
    assert test_acc == 1.0

def test_accuracy_score_multi_class():
    test_true = np.array([0, 1, 0, 1, 2])
    test_pred = np.array([0, 1, 0, 1, 1])
    test_acc = accuracy_score(test_pred, test_true)
    assert isinstance(test_acc, float)
    assert test_acc == 0.8

def test_accuracy_score_single():
    test_true = np.array([0])
    test_pred = np.array([0])
    test_acc = accuracy_score(test_pred, test_true)
    assert isinstance(test_acc, float)
    assert test_acc == 1.0

def test_accuracy_score_empty():
    test_true = np.array([])
    test_pred = np.array([0])
    with pytest.raises(ValueError):
        accuracy_score(test_pred, test_true)

def test_accuracy_score_none():
    test_true = np.array([1, 0, 1, 0])
    test_pred = np.array([0, 1, 0, 1])
    test_acc = accuracy_score(test_pred, test_true)
    assert isinstance(test_acc, float)
    assert test_acc == 0.0

def test_accuracy_score_strings():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A'])
    test_acc = accuracy_score(test_pred, test_true)
    assert isinstance(test_acc, float)
    assert test_acc == 0.5

def test_accuracy_score_dimension():
    test_true = np.array([['A', 'A', 'B', 'B']])
    test_pred = np.array(['A', 'B', 'B', 'A'])
    with pytest.raises(ValueError):
        accuracy_score(test_pred, test_true)

def test_accuracy_score_shape_mismatch():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A', 'B'])
    with pytest.raises(ValueError):
        accuracy_score(test_pred, test_true)

def test_confusion_matrix_basic():
    test_true = np.array([0, 1, 0, 0])
    test_pred = np.array([0, 1, 0, 1])
    test_cm = confusion_matrix(test_pred, test_true, False)
    assert isinstance(test_cm, np.ndarray)
    assert np.allclose(test_cm, [[2, 1], [0, 1]])

def test_confusion_matrix_labels():
    test_true = np.array([0, 1, 0, 0])
    test_pred = np.array([0, 1, 0, 1])
    labels = [0, 1]
    test_cm = confusion_matrix(test_pred, test_true, False, False, labels)
    assert isinstance(test_cm, np.ndarray)
    assert np.allclose(test_cm, [[2, 1], [0, 1]])

def test_confusion_matrix_plot():
    test_true = np.array([0, 1, 0, 0])
    test_pred = np.array([0, 1, 0, 1])
    labels = [0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        test_cm = confusion_matrix(test_pred, test_true, True, True, labels)
    assert isinstance(test_cm, np.ndarray)
    assert np.allclose(test_cm, [[2, 1], [0, 1]])

def test_confusion_matrix_plot_values():
    test_true = np.array([0, 1, 0, 0])
    test_pred = np.array([0, 1, 0, 1])
    labels = [0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        test_cm = confusion_matrix(test_pred, test_true, True, False, labels)
    assert isinstance(test_cm, np.ndarray)
    assert np.allclose(test_cm, [[2, 1], [0, 1]])

def test_confusion_matrix_plot_no_labels():
    test_true = np.array([0, 1, 0, 0])
    test_pred = np.array([0, 1, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        test_cm = confusion_matrix(test_pred, test_true)
    assert isinstance(test_cm, np.ndarray)
    assert np.allclose(test_cm, [[2, 1], [0, 1]])

def test_confusion_matrix_plot_strings():
    test_true = np.array(['A', 'B', 'A', 'A'])
    test_pred = np.array(['A', 'B', 'A', 'B'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        test_cm = confusion_matrix(test_pred, test_true)
    assert isinstance(test_cm, np.ndarray)
    assert np.allclose(test_cm, [[2, 1], [0, 1]])

def test_confusion_matrix_plot_type():
    test_true = np.array(['A', 'B', 'A', 'A'])
    test_pred = np.array(['A', 'B', 'A', 'B'])
    with pytest.raises(TypeError):
        confusion_matrix(test_pred, test_true, 'True')

def test_precision_score_basic_binary():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    test_p = precision_score(test_pred, test_true, metric = 'binary')
    assert isinstance(test_p, float)
    assert test_p == 0.5

def test_precision_score_binary_strings():
    test_true = np.array(['A', 'B', 'A', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A'])
    test_p = precision_score(test_pred, test_true, metric = 'binary')
    assert isinstance(test_p, float)
    assert test_p == 0.5

def test_precision_score_binary_labels():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = [0, 1]
    test_p = precision_score(test_pred, test_true, metric = 'binary', labels = labels)
    assert isinstance(test_p, float)
    assert test_p == 0.5

def test_precision_score_binary_classes_3():
    test_true = np.array([0, 1, 0, 2])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        precision_score(test_pred, test_true, metric = 'binary')

def test_precision_score_binary_classes_1():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError):
        precision_score(test_pred, test_true, metric = 'binary')

def test_precision_score_basic_micro():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 1])
    test_p = precision_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_p, float)
    assert test_p == 0.75

def test_precision_score_micro_strings():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'B'])
    test_p = precision_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_p, float)
    assert test_p == 0.75

def test_precision_score_micro_labels():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 1])
    labels = [0, 1]
    test_p = precision_score(test_pred, test_true, metric = 'micro', labels = labels)
    assert isinstance(test_p, float)
    assert test_p == 0.75

def test_precision_score_micro_multi_class():
    test_true = np.array([0, 0, 1, 2])
    test_pred = np.array([0, 1, 1, 1])
    test_p = precision_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_p, float)
    assert test_p == 0.5

def test_precision_score_micro_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 1, 1])
    test_p = precision_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_p, float)
    assert test_p == 0.5

def test_precision_score_basic_macro():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    test_p = precision_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_p, float)
    assert test_p == 0.75

def test_precision_score_macro_strings():
    test_true = np.array(['A', 'A', 'A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A', 'B', 'B'])
    test_p = precision_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_p, float)
    assert test_p == 0.75

def test_precision_score_macro_labels():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    labels = [0, 1]
    test_p = precision_score(test_pred, test_true, metric = 'macro', labels = labels)
    assert isinstance(test_p, float)
    assert test_p == 0.75

def test_precision_score_macro_multi_class():
    test_true = np.array([0, 0, 1, 1, 2, 3])
    test_pred = np.array([0, 3, 1, 1, 0, 3])
    test_p = precision_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_p, float)
    assert test_p == 0.5

def test_precision_score_macro_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 1, 1, 1])
    test_p = precision_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_p, float)
    assert test_p == 0.5

def test_precision_score_basic_none():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    test_p = precision_score(test_pred, test_true, metric = None)
    assert isinstance(test_p, np.ndarray)
    assert np.allclose(test_p, [1, 0.5])

def test_precision_score_none_strings():
    test_true = np.array(['A', 'A', 'A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A', 'B', 'B'])
    test_p = precision_score(test_pred, test_true, metric = None)
    assert isinstance(test_p, np.ndarray)
    assert np.allclose(test_p, [1, 0.5])

def test_precision_score_none_labels():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    labels = [0, 1]
    test_p = precision_score(test_pred, test_true, metric = None, labels = labels)
    assert isinstance(test_p, np.ndarray)
    assert np.allclose(test_p, [1, 0.5])

def test_precision_score_none_multi_class():
    test_true = np.array([0, 0, 1, 1, 2, 3])
    test_pred = np.array([0, 3, 1, 1, 0, 3])
    test_p = precision_score(test_pred, test_true, metric = None)
    assert isinstance(test_p, np.ndarray)
    assert np.allclose(test_p, [0.5, 1, 0, 0.5])

def test_precision_score_none_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 1, 1])
    test_p = precision_score(test_pred, test_true, metric = None)
    assert isinstance(test_p, np.ndarray)
    assert np.allclose(test_p, [1, 0])

def test_precision_score_dimension_true():
    test_true = np.array([[0, 1, 0, 1]])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        precision_score(test_pred, test_true)

def test_precision_score_dimension_pred():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([[0, 1, 1, 0]])
    with pytest.raises(ValueError):
        precision_score(test_pred, test_true)

def test_precision_score_shape_mismatch():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0, 1])
    with pytest.raises(ValueError):
        precision_score(test_pred, test_true)
        
def test_precision_score_metric_type():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        precision_score(test_pred, test_true, metric = 'None')

def test_precision_score_label_type():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = np.array([0, 1])
    with pytest.raises(TypeError):
        precision_score(test_pred, test_true, metric = 'macro', labels = labels)

def test_recall_score_basic_binary():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    test_r = recall_score(test_pred, test_true, metric = 'binary')
    assert isinstance(test_r, float)
    assert test_r == 0.5

def test_recall_score_binary_strings():
    test_true = np.array(['A', 'B', 'A', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A'])
    test_r = recall_score(test_pred, test_true, metric = 'binary')
    assert isinstance(test_r, float)
    assert test_r == 0.5

def test_recall_score_binary_labels():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = [0, 1]
    test_r = recall_score(test_pred, test_true, metric = 'binary', labels = labels)
    assert isinstance(test_r, float)
    assert test_r == 0.5

def test_recall_score_binary_classes_3():
    test_true = np.array([0, 1, 0, 2])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        recall_score(test_pred, test_true, metric = 'binary')

def test_recall_score_binary_classes_1():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError):
        recall_score(test_pred, test_true, metric = 'binary')

def test_precision_score_basic_micro():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 1])
    test_r = recall_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_r, float)
    assert test_r == 0.75

def test_recall_score_micro_strings():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'B'])
    test_r = recall_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_r, float)
    assert test_r == 0.75

def test_recall_score_micro_labels():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 1])
    labels = [0, 1]
    test_r = recall_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_r, float)
    assert test_r == 0.75

def test_recall_score_micro_multi_class():
    test_true = np.array([0, 0, 1, 2])
    test_pred = np.array([0, 1, 1, 1])
    test_r = recall_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_r, float)
    assert test_r == 0.5

def test_recall_score_micro_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 1, 1])
    test_r = recall_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_r, float)
    assert test_r == 0.5

def test_recall_score_basic_macro():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    test_r = recall_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_r, float)
    assert test_r == 0.75

def test_recall_score_macro_strings():
    test_true = np.array(['A', 'A', 'A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A', 'B', 'B'])
    test_r = recall_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_r, float)
    assert test_r == 0.75

def test_recall_score_macro_labels():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    labels = [0, 1]
    test_r = recall_score(test_pred, test_true, metric = 'macro', labels = labels)
    assert isinstance(test_r, float)
    assert test_r == 0.75

def test_recall_score_macro_multi_class():
    test_true = np.array([0, 0, 1, 1, 2, 3])
    test_pred = np.array([0, 3, 1, 1, 0, 3])
    test_r = recall_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_r, float)
    assert test_r == 0.625

def test_recall_score_macro_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 1, 1, 1])
    test_r = recall_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_r, float)
    assert test_r == 0.125

def test_recall_score_basic_none():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    test_r = recall_score(test_pred, test_true, metric = None)
    assert isinstance(test_r, np.ndarray)
    assert np.allclose(test_r, [0.5, 1])

def test_recall_score_none_strings():
    test_true = np.array(['A', 'A', 'A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A', 'B', 'B'])
    test_r = recall_score(test_pred, test_true, metric = None)
    assert isinstance(test_r, np.ndarray)
    assert np.allclose(test_r, [0.5, 1])

def test_recall_score_none_labels():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    labels = [0, 1]
    test_r = recall_score(test_pred, test_true, metric = None, labels = labels)
    assert isinstance(test_r, np.ndarray)
    assert np.allclose(test_r, [0.5, 1])

def test_recall_score_none_multi_class():
    test_true = np.array([0, 0, 1, 1, 2, 3])
    test_pred = np.array([0, 3, 1, 1, 0, 3])
    test_r = recall_score(test_pred, test_true, metric = None)
    assert isinstance(test_r, np.ndarray)
    assert np.allclose(test_r, [0.5, 1, 0, 1])

def test_recall_score_none_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 1, 1])
    test_r = recall_score(test_pred, test_true, metric = None)
    assert isinstance(test_r, np.ndarray)
    assert np.allclose(test_r, [0.5, 0])

def test_recall_score_dimension_true():
    test_true = np.array([[0, 1, 0, 1]])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        recall_score(test_pred, test_true)

def test_recall_score_dimension_pred():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([[0, 1, 1, 0]])
    with pytest.raises(ValueError):
        recall_score(test_pred, test_true)

def test_recall_score_shape_mismatch():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0, 1])
    with pytest.raises(ValueError):
        recall_score(test_pred, test_true)
        
def test_recall_score_metric_type():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        recall_score(test_pred, test_true, metric = 'None')

def test_recall_score_label_type():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = np.array([0, 1])
    with pytest.raises(TypeError):
        recall_score(test_pred, test_true, metric = 'macro', labels = labels)

def test_f1_score_basic_binary():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    test_f1 = f1_score(test_pred, test_true, metric = 'binary')
    assert isinstance(test_f1, float)
    assert test_f1 == 0.5

def test_f1_score_binary_strings():
    test_true = np.array(['A', 'B', 'A', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A'])
    test_f1 = f1_score(test_pred, test_true, metric = 'binary')
    assert isinstance(test_f1, float)
    assert test_f1 == 0.5

def test_f1_score_binary_labels():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = [0, 1]
    test_f1 = f1_score(test_pred, test_true, metric = 'binary', labels = labels)
    assert isinstance(test_f1, float)
    assert test_f1 == 0.5

def test_f1_score_binary_classes_3():
    test_true = np.array([0, 1, 0, 2])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        f1_score(test_pred, test_true, metric = 'binary')

def test_f1_score_binary_classes_1():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError):
        f1_score(test_pred, test_true, metric = 'binary')

def test_f1_score_basic_micro():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 1])
    test_f1 = f1_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_f1, float)
    assert test_f1 == 0.75

def test_f1_score_micro_strings():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'B'])
    test_f1 = f1_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_f1, float)
    assert test_f1 == 0.75

def test_f1_score_micro_labels():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 1])
    labels = [0, 1]
    test_f1 = f1_score(test_pred, test_true, metric = 'micro', labels = labels)
    assert isinstance(test_f1, float)
    assert test_f1 == 0.75

def test_f1_score_micro_multi_class():
    test_true = np.array([0, 0, 1, 2])
    test_pred = np.array([0, 1, 1, 1])
    test_f1 = f1_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_f1, float)
    assert test_f1 == 0.5

def test_f1_score_micro_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 1, 1])
    test_f1 = f1_score(test_pred, test_true, metric = 'micro')
    assert isinstance(test_f1, float)
    assert test_f1 == 0.5

def test_f1_score_basic_macro():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    test_f1 = f1_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_f1, float)
    assert test_f1 == 2/3

def test_f1_score_macro_strings():
    test_true = np.array(['A', 'A', 'A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A', 'B', 'B'])
    test_f1 = f1_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_f1, float)
    assert test_f1 == 2/3

def test_f1_score_macro_labels():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    labels = [0, 1]
    test_f1 = f1_score(test_pred, test_true, metric = 'macro', labels = labels)
    assert isinstance(test_f1, float)
    assert test_f1 == 2/3

def test_f1_score_macro_multi_class():
    test_true = np.array([0, 0, 1, 3])
    test_pred = np.array([0, 3, 1, 3])
    test_f1 = f1_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_f1, float)
    assert np.isclose(test_f1, 7/9)

def test_f1_score_macro_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 1, 1, 1])
    test_f1 = f1_score(test_pred, test_true, metric = 'macro')
    assert isinstance(test_f1, float)
    assert test_f1 == 0.2

def test_f1_score_basic_none():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    test_f1 = f1_score(test_pred, test_true, metric = None)
    assert isinstance(test_f1, np.ndarray)
    assert np.allclose(test_f1, [2/3, 2/3])

def test_f1_score_none_strings():
    test_true = np.array(['A', 'A', 'A', 'A', 'B', 'B'])
    test_pred = np.array(['A', 'B', 'B', 'A', 'B', 'B'])
    test_f1 = f1_score(test_pred, test_true, metric = None)
    assert isinstance(test_f1, np.ndarray)
    assert np.allclose(test_f1, [2/3, 2/3])

def test_f1_score_none_labels():
    test_true = np.array([0, 0, 0, 0, 1, 1])
    test_pred = np.array([0, 1, 1, 0, 1, 1])
    labels = [0, 1]
    test_f1 = f1_score(test_pred, test_true, metric = None, labels = labels)
    assert isinstance(test_f1, np.ndarray)
    assert np.allclose(test_f1, [2/3, 2/3])

def test_f1_score_none_multi_class():
    test_true = np.array([0, 0, 1, 1, 2, 3])
    test_pred = np.array([0, 3, 1, 1, 0, 3])
    test_f1 = f1_score(test_pred, test_true, metric = None)
    assert isinstance(test_f1, np.ndarray)
    assert np.allclose(test_f1, [0.5, 1, 0, 2/3])

def test_f1_score_none_one_class():
    test_true = np.array([0, 0, 0, 0])
    test_pred = np.array([0, 0, 1, 1])
    test_f1 = f1_score(test_pred, test_true, metric = None)
    assert isinstance(test_f1, np.ndarray)
    assert np.allclose(test_f1, [2/3, 0])

def test_f1_score_dimension_true():
    test_true = np.array([[0, 1, 0, 1]])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        f1_score(test_pred, test_true)

def test_f1_score_dimension_pred():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([[0, 1, 1, 0]])
    with pytest.raises(ValueError):
        f1_score(test_pred, test_true)

def test_f1_score_shape_mismatch():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0, 1])
    with pytest.raises(ValueError):
        f1_score(test_pred, test_true)
        
def test_f1_score_metric_type():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        f1_score(test_pred, test_true, metric = 'None')

def test_f1_score_label_type():
    test_true = np.array([0, 1, 0, 1])
    test_pred = np.array([0, 1, 1, 0])
    labels = np.array([0, 1])
    with pytest.raises(TypeError):
        f1_score(test_pred, test_true, metric = 'macro', labels = labels)

def test_roc_auc_basic():
    test_true = np.array([0, 0, 1, 1])
    test_score = np.array([0.1, 0.4, 0.35, 0.8])
    auc = roc_auc(test_score, test_true)
    assert isinstance(auc, float)
    assert np.isclose(auc, 0.75)

def test_roc_auc_perfect():
    test_true = np.array([0, 0, 1, 1])
    test_score = np.array([0.1, 0.1, 0.9, 0.8])
    auc = roc_auc(test_score, test_true)
    assert isinstance(auc, float)
    assert np.isclose(auc, 1.0)

def test_roc_auc_poor():
    test_true = np.array([0, 0, 1, 1])
    test_score = np.array([0.9, 0.9, 0.1, 0.1])
    auc = roc_auc(test_score, test_true)
    assert isinstance(auc, float)
    assert np.isclose(auc, 0.0)

def test_roc_auc_ties():
    test_true = np.array([0, 0, 1, 1])
    test_score = np.array([0.5, 0.5, 0.5, 0.5])
    auc = roc_auc(test_score, test_true)
    assert isinstance(auc, float)
    assert np.isclose(auc, 0.5)

def test_roc_auc_strings():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_score = np.array([0.5, 0.5, 0.5, 0.5])
    auc = roc_auc(test_score, test_true)
    assert isinstance(auc, float)
    assert np.isclose(auc, 0.5)

def test_roc_auc_type_pred():
    test_true = np.array(['A', 'A', 'B', 'B'])
    test_score = np.array([0.5, 0.5, 0.5, 'A'])
    with pytest.raises(TypeError):
        roc_auc(test_score, test_true)

def test_roc_auc_class_count():
    test_true = np.array(['A', 'A', 'B', 'C'])
    test_score = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        roc_auc(test_score, test_true)

def test_roc_auc_shape_mismatch():
    test_true = np.array(['A', 'A', 'B', 'B' 'B'])
    test_score = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        roc_auc(test_score, test_true)

def test_log_loss_basic():
    test_true = np.array([0, 1, 1, 0])
    test_pred = np.array([0.9, 0.8, 0.2, 0.1])
    test_loss = log_loss(test_pred, test_true)
    assert isinstance(test_loss, float)
    assert test_loss > 0

def test_log_loss_extreme():
    test_true = np.array([0, 1, 1, 0])
    test_pred = np.array([0.0, 1.0, 1.0, 0.0])
    test_loss = log_loss(test_pred, test_true)
    assert isinstance(test_loss, float)
    assert np.isclose(test_loss, 1e-10, atol=1e-10)

def test_log_loss_multi_class():
    test_true = np.array([0, 1, 2])
    test_pred = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6]
    ])
    test_loss = log_loss(test_pred, test_true)
    assert isinstance(test_loss, float)
    assert test_loss > 0

def test_log_loss_strings():
    test_true = np.array(['A', 'B', 'B', 'A'])
    test_pred = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(TypeError):
        log_loss(test_pred, test_true)

def test_log_loss_probability():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([1.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError):
        log_loss(test_pred, test_true)

def test_log_loss_epsilon_type():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(TypeError):
        log_loss(test_pred, test_true, '1')

def test_log_loss_epsilon_value():
    test_true = np.array([0, 0, 1, 1])
    test_pred = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError):
        log_loss(test_pred, test_true, -1)

def test_log_loss_prob_sum():
    test_true = np.array([0, 1, 2])
    test_pred = np.array([
        [0.9, 0.1, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6]
    ])
    with pytest.raises(ValueError):
        log_loss(test_pred, test_true)

def test_log_loss_predicted_count():
    test_true = np.array([0, 1, 2])
    test_pred = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    with pytest.raises(ValueError):
        log_loss(test_pred, test_true)
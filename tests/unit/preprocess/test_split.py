
import numpy as np
import pandas as pd
import pytest
from rice_ml.preprocess.split import _bounded_count, _random_number, _stratified_indices, train_test

def test_bounded_count_basic():
    test_length, test_proportion = 3, 0.1
    result = _bounded_count(test_length, test_proportion)
    expected = min(max(1, int(round(test_proportion * test_length))), test_length - 1)
    assert isinstance(result, int)
    assert result == expected

def test_bounded_count_len_one_greatercount():
    test_length, test_proportion = 1, 10
    result = _bounded_count(test_length, test_proportion)
    assert isinstance(result, int)
    assert result == 1

def test_bounded_count_len_one_lesscount():
    test_length, test_proportion = 1, 0.1
    result = _bounded_count(test_length, test_proportion)
    assert isinstance(result, int)
    assert result == 0

def test_bounded_count_len_type_test_length():
    test_length, test_proportion = 1.1, 0.1
    with pytest.raises(TypeError):
        _bounded_count(test_length, test_proportion)

def test_bounded_count_len_type_test_proportion():
    test_length, test_proportion = 1, 'one'
    with pytest.raises(TypeError):
        _bounded_count(test_length, test_proportion)






def test_random_number_basic():
    random_state = None
    rng = _random_number(random_state)
    assert isinstance(rng, np.random.Generator)

def test_random_number_basic_int():
    random_state = 72
    rng = _random_number(random_state)
    assert isinstance(rng, np.random.Generator)

def test_random_number_type_random_state():
    random_state = 72.1
    with pytest.raises(TypeError):
        _random_number(random_state)

def test_stratified_indices_basic_array():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.25
    rng = np.random.default_rng()
    testing_indices, training_indices = _stratified_indices(test_array, test_size, rng)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert testing_indices.shape == (3,)
    assert training_indices.shape == (7,)
    assert len(np.intersect1d(testing_indices, training_indices)) == 0
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)
    classes, label_index = np.unique(test_array, return_inverse = True)
    for class_number in range(len(classes)):
        class_index = np.flatnonzero(label_index == class_number)
        testing_indices_count = np.sum(np.isin(testing_indices, class_index))
        training_indices_count = np.sum(np.isin(training_indices, class_index))
        test_number = min(max(1, int(round(test_size * len(class_index)))), len(class_index) - 1)
        assert testing_indices_count == test_number
        assert training_indices_count + testing_indices_count == len(class_index)        

def test_stratified_indices_basic_df():
    test_array = pd.Series([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.25
    rng = np.random.default_rng()
    testing_indices, training_indices = _stratified_indices(test_array, test_size, rng)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert testing_indices.shape == (3,)
    assert training_indices.shape == (7,)
    assert len(np.intersect1d(testing_indices, training_indices)) == 0
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)
    classes, label_index = np.unique(test_array, return_inverse = True)
    for class_number in range(len(classes)):
        class_index = np.flatnonzero(label_index == class_number)
        testing_indices_count = np.sum(np.isin(testing_indices, class_index))
        training_indices_count = np.sum(np.isin(training_indices, class_index))
        test_number = min(max(1, int(round(test_size * len(class_index)))), len(class_index) - 1)
        assert testing_indices_count == test_number
        assert training_indices_count + testing_indices_count == len(class_index)

def test_stratified_indices_basic_list():
    test_array = [1, 2, 3, 2, 1, 3, 1, 3, 3, 1]
    test_size = 0.25
    rng = np.random.default_rng()
    testing_indices, training_indices = _stratified_indices(test_array, test_size, rng)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert testing_indices.shape == (3,)
    assert training_indices.shape == (7,)
    assert len(np.intersect1d(testing_indices, training_indices)) == 0
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)
    classes, label_index = np.unique(test_array, return_inverse = True)
    for class_number in range(len(classes)):
        class_index = np.flatnonzero(label_index == class_number)
        testing_indices_count = np.sum(np.isin(testing_indices, class_index))
        training_indices_count = np.sum(np.isin(training_indices, class_index))
        test_number = min(max(1, int(round(test_size * len(class_index)))), len(class_index) - 1)
        assert testing_indices_count == test_number
        assert training_indices_count + testing_indices_count == len(class_index)

def test_stratified_indices_strings():
    test_array = np.array(['a', 'b', 'c', 'b', 'a', 'c', 'a', 'c', 'c', 'a'])
    test_size = 0.25
    rng = np.random.default_rng()
    testing_indices, training_indices = _stratified_indices(test_array, test_size, rng)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert testing_indices.shape == (3,)
    assert training_indices.shape == (7,)
    assert len(np.intersect1d(testing_indices, training_indices)) == 0
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)
    classes, label_index = np.unique(test_array, return_inverse = True)
    for class_number in range(len(classes)):
        class_index = np.flatnonzero(label_index == class_number)
        testing_indices_count = np.sum(np.isin(testing_indices, class_index))
        training_indices_count = np.sum(np.isin(training_indices, class_index))
        test_number = min(max(1, int(round(test_size * len(class_index)))), len(class_index) - 1)
        assert testing_indices_count == test_number
        assert training_indices_count + testing_indices_count == len(class_index)       

def test_stratified_indices_dimensions():
    test_array_2D = np.array([[1, 2, 3, 2, 1, 3, 1, 3, 3, 1]])
    test_array_3D = np.array([[[1, 2, 3, 2, 1, 3, 1, 3, 3, 1]], [[1, 2, 3, 2, 1, 3, 1, 3, 3, 1]]])
    test_size = 0.25
    rng = np.random.default_rng()
    with pytest.raises(ValueError):
         _stratified_indices(test_array_2D, test_size, rng)
    with pytest.raises(ValueError):
         _stratified_indices(test_array_3D, test_size, rng)

def test_stratified_indices_single_class():
    test_array = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    test_size = 0.25
    rng = np.random.default_rng()
    testing_indices, training_indices = _stratified_indices(test_array, test_size, rng)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert testing_indices.shape == (2,)
    assert training_indices.shape == (6,)
    assert len(np.intersect1d(testing_indices, training_indices)) == 0
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)

def test_stratified_indices_single_element():
    test_array = np.array([1])
    test_size = 0.25
    rng = np.random.default_rng()
    testing_indices, training_indices = _stratified_indices(test_array, test_size, rng)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert testing_indices.shape == (0,)
    assert training_indices.shape == (1,)
    assert len(np.intersect1d(testing_indices, training_indices)) == 0
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)

def test_stratified_indices_training_set():
    test_array = np.array([1, 1])
    test_size = 0.99
    rng = np.random.default_rng()
    testing_indices, training_indices = _stratified_indices(test_array, test_size, rng)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert testing_indices.shape == (1,)
    assert training_indices.shape == (1,)
    assert len(np.intersect1d(testing_indices, training_indices)) == 0
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)
     
def test_stratified_indices_validate():
    test_array = np.array([1, 3, 3, 1, 1, 3, 1, 3, 3, 1])
    test_size = 0.20
    val_size = 0.4
    rng = np.random.default_rng()
    testing_indices, training_indices, val_indices = _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices, val_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert isinstance(val_indices, np.ndarray)
    assert testing_indices.shape == (2,)
    assert training_indices.shape == (4,)
    assert val_indices.shape == (4,)
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)
    classes, label_index = np.unique(test_array, return_inverse = True)
    for class_number in range(len(classes)):
        class_index = np.flatnonzero(label_index == class_number)
        testing_indices_count = np.sum(np.isin(testing_indices, class_index))
        training_indices_count = np.sum(np.isin(training_indices, class_index))
        val_indices_count = np.sum(np.isin(val_indices, class_index))
        test_number = min(max(1, int(round(test_size * len(class_index)))), len(class_index) - 1)
        remaining_count = len(class_index) - test_number
        val_size_remaining = val_size / (1.0 - test_size)
        val_number = min(max(1, int(round(val_size_remaining * remaining_count))), len(class_index) - 1)
        assert testing_indices_count == test_number
        assert val_indices_count == val_number
        assert training_indices_count + testing_indices_count + val_indices_count == len(class_index) 

def test_stratified_indices_validate_single_class():
    test_array = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    test_size = 0.20
    val_size = 0.4
    rng = np.random.default_rng()
    testing_indices, training_indices, val_indices = _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices, val_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert isinstance(val_indices, np.ndarray)
    assert testing_indices.shape == (2,)
    assert training_indices.shape == (4,)
    assert val_indices.shape == (4,)
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)

def test_stratified_indices_validate_single_element():
    test_array = np.array([1])
    test_size = 0.20
    val_size = 0.4
    rng = np.random.default_rng()
    testing_indices, training_indices, val_indices = _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices, val_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert isinstance(val_indices, np.ndarray)
    assert testing_indices.shape == (0,)
    assert training_indices.shape == (1,)
    assert val_indices.shape == (0,)
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)

def test_stratified_indices_validate_single_element_training_set():
    test_array = np.array([1, 1])
    test_size = 0.20
    val_size = 0.4
    rng = np.random.default_rng()
    testing_indices, training_indices, val_indices = _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices, val_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert isinstance(val_indices, np.ndarray)
    assert testing_indices.shape == (0,)
    assert training_indices.shape == (1,)
    assert val_indices.shape == (1,)
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)

def test_stratified_indices_validate_large_val():
    test_array = np.array([1, 3, 3, 1, 1, 3, 1, 3, 3, 1])
    test_size = 0.20
    val_size = 0.75
    rng = np.random.default_rng()
    testing_indices, training_indices, val_indices = _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size)
    total_indices = np.arange(len(test_array))
    output_indices = np.concatenate([testing_indices, training_indices, val_indices])
    assert isinstance(testing_indices, np.ndarray)
    assert isinstance(training_indices, np.ndarray)
    assert isinstance(val_indices, np.ndarray)
    assert testing_indices.shape == (2,)
    assert training_indices.shape == (2,)
    assert val_indices.shape == (6,)
    assert len(total_indices) == len(output_indices)
    assert set(total_indices) == set(output_indices)

def test_stratified_indices_randomized_same():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.25
    rng_1 = np.random.default_rng(42)
    testing_indices_1, training_indices_1 = _stratified_indices(test_array, test_size, rng_1)
    rng_2 = np.random.default_rng(42)
    testing_indices_2, training_indices_2 = _stratified_indices(test_array, test_size, rng_2)
    assert isinstance(testing_indices_1, np.ndarray)
    assert isinstance(training_indices_1, np.ndarray)
    assert isinstance(testing_indices_2, np.ndarray)
    assert isinstance(training_indices_2, np.ndarray)
    assert np.array_equal(testing_indices_1, testing_indices_2)
    assert np.array_equal(training_indices_1, training_indices_2)

def test_stratified_indices_randomized_different():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.25
    rng_1 = np.random.default_rng(42)
    testing_indices_1, training_indices_1 = _stratified_indices(test_array, test_size, rng_1)
    rng_2 = np.random.default_rng(72)
    testing_indices_2, training_indices_2 = _stratified_indices(test_array, test_size, rng_2)
    assert isinstance(testing_indices_1, np.ndarray)
    assert isinstance(training_indices_1, np.ndarray)
    assert isinstance(testing_indices_2, np.ndarray)
    assert isinstance(training_indices_2, np.ndarray)
    assert np.array_equal(testing_indices_1, testing_indices_2) == False
    assert np.array_equal(training_indices_1, training_indices_2) == False

def test_stratified_indices_type_input():
    test_array = 'test_array'
    test_size = 0.25
    rng = np.random.default_rng()
    with pytest.raises(TypeError):
        _stratified_indices(test_array, test_size, rng)

def test_stratified_indices_type_test_size():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = '0.25'
    rng = np.random.default_rng()
    with pytest.raises(TypeError):
        _stratified_indices(test_array, test_size, rng)

def test_stratified_indices_test_size_limits():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size_positive = 2.0
    test_size_negative = -2.0
    rng = np.random.default_rng()
    with pytest.raises(ValueError):
        _stratified_indices(test_array, test_size_positive, rng)
    with pytest.raises(ValueError):
        _stratified_indices(test_array, test_size_negative, rng)

def test_stratified_indices_type_rng():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.2
    rng = 'np.random.default_rng'
    with pytest.raises(TypeError):
        _stratified_indices(test_array, test_size, rng)

def test_stratified_indices_type_validation():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.2
    rng = np.random.default_rng
    with pytest.raises(TypeError):
        _stratified_indices(test_array, test_size, rng, validation = 'True')

def test_stratified_indices_type_val_size():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.2
    rng = np.random.default_rng
    val_size = '0.4'
    with pytest.raises(TypeError):
        _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size)

def test_stratified_indices_val_size_limits():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.2
    rng = np.random.default_rng()
    val_size_positive = 2.0
    with pytest.raises(ValueError):
        _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size_positive)
    val_size_negative = -2.0
    with pytest.raises(ValueError):
        _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size_negative)

def test_stratified_indices_val_test_size_limits():
    test_array = np.array([1, 2, 3, 2, 1, 3, 1, 3, 3, 1])
    test_size = 0.2
    val_size = 0.9
    rng = np.random.default_rng()
    with pytest.raises(ValueError):
        _stratified_indices(test_array, test_size, rng, validation = True, val_size = val_size)
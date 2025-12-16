
import numpy as np
import pandas as pd
import pytest
from rice_ml.supervised_learning.distances import _ensure_numeric, euclidean_distance, manhattan_distance, minkowski_distance

def test_ensure_numeric_basic_array():
    test_array = np.array([1, 2, 3, 4])
    result_array = _ensure_numeric(test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert result_array.dtype == float

def test_ensure_numeric_basic_float():
    test_array = np.array([1.1, 2.2, 3.3, 4.4])
    result_array = _ensure_numeric(test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert all(isinstance(element, (int, float)) for element in result_array)

def test_ensure_numeric_basic_string():
    test_array = np.array(['1.1', 2, 3, 4])
    result_array = _ensure_numeric(test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert all(isinstance(element, (int, float)) for element in result_array)

def test_ensure_numeric_boolean():
    test_array = np.array([True, False, True, False])
    result_array = _ensure_numeric(test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert np.array_equal(result_array, np.array([1.0, 0.0, 1.0, 0.0]))

def test_ensure_numeric_basic_list():
    test_list = [1, 2, 3, 4]
    result_array = _ensure_numeric(test_list)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert all(isinstance(element, (int, float)) for element in result_array)

def test_ensure_numeric_basic_df():
    test_series = pd.Series([1, 2, 3, 4])
    result_array = _ensure_numeric(test_series)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert all(isinstance(element, (int, float)) for element in result_array)

def test_ensure_numeric_basic_tuple():
    test_tuple = (1, 2, 3, 4)
    result_array = _ensure_numeric(test_tuple)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert all(isinstance(element, (int, float)) for element in result_array)

def test_ensure_numeric_dimensions():
    test_array_2D = np.array([[1, 2, 3, 4]])
    with pytest.raises(ValueError):
        _ensure_numeric(test_array_2D)

def test_ensure_numeric_nan():
    test_array_2D = np.array([1, 2, 3, np.nan])
    with pytest.raises(ValueError):
        _ensure_numeric(test_array_2D)

def test_ensure_numeric_types():
    test_array = np.array(['one', 2, 3, 4])
    with pytest.raises(TypeError):
        _ensure_numeric(test_array)

def test_ensure_numeric_type_input():
    test_array = 'one'
    with pytest.raises(TypeError):
        _ensure_numeric(test_array)

def test_ensure_numeric_empty():
    test_array = np.array([])
    with pytest.raises(ValueError):
        _ensure_numeric(test_array)

def test_ensure_numeric_naming_input_normal():
    test_array = np.array([1, 2, 3, 4])
    result_array = _ensure_numeric(test_array, 'test_vector')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4,)
    assert all(isinstance(element, (int, float)) for element in result_array)

def test_ensure_numeric_naming_input():
    test_array = np.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        _ensure_numeric(test_array, test_array)







def test_euclidean_distance_basic_arrays():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt(np.sum((test_array_2 - test_array_1) ** 2))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_basic_lists():
    test_array_1 = [2, 3]
    test_array_2 = [5, 7]
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt((5 -2) ** 2 + (7 - 3) ** 2)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_basic_df():
    test_array_1 = pd.Series([2, 3])
    test_array_2 = pd.Series([5, 7])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt((5 -2) ** 2 + (7 - 3) ** 2)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_basic_df_row():
    test_df = pd.DataFrame({
            'A': [2, 5], 
            'B': [3, 7],
    })
    test_array_1 = test_df.iloc[0]
    test_array_2 = test_df.iloc[1]
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt((5 -2) ** 2 + (7 - 3) ** 2)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_basic_tuple():
    test_array_1 = (2, 3)
    test_array_2 = (5, 7)
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt((5 -2) ** 2 + (7 - 3) ** 2)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_basic_mixed():
    test_array_1 = np.array([2, 3])
    test_array_2 = [5, 7]
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt(np.sum((test_array_2 - test_array_1) ** 2))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_zeros():
    test_array_1 = np.array([0, 0])
    test_array_2 = np.array([0, 0])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = 0
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_negatives():
    test_array_1 = np.array([-2, 3])
    test_array_2 = np.array([-5, 7])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt(np.sum((test_array_2 - test_array_1) ** 2))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_high_dim():
    test_array_1 = np.array([1, 2, 3, 4, 5])
    test_array_2 = np.array([2, 4, 6, 8, 10])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt(np.sum((test_array_2 - test_array_1) ** 2))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_single_element():
    test_array_1 = np.array([1])
    test_array_2 = np.array([2])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt(np.sum((test_array_2 - test_array_1) ** 2))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_strings():
    test_array_1 = np.array(['2', 3])
    test_array_2 = np.array([5, 7])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt((5 -2) ** 2 + (7 - 3) ** 2)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_floats():
    test_array_1 = np.array([2.0, 3.0])
    test_array_2 = np.array([5.0, 7.0])
    result_value = euclidean_distance(test_array_1, test_array_2)
    actual_value = np.sqrt(np.sum((test_array_2 - test_array_1) ** 2))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_symmetry():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value_1 = euclidean_distance(test_array_1, test_array_2)
    result_value_2 = euclidean_distance(test_array_2, test_array_1)
    assert result_value_1 == result_value_2

def test_euclidean_distance_dimensions():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([[5, 7]])
    with pytest.raises(ValueError):
        euclidean_distance(test_array_1, test_array_2)

def test_euclidean_distance_empty():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([])
    with pytest.raises(ValueError):
        euclidean_distance(test_array_1, test_array_2)

def test_euclidean_distance_nan():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, np.nan])
    with pytest.raises(ValueError):
        euclidean_distance(test_array_1, test_array_2)

def test_euclidean_distance_datatype_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array(['a', 'b'])
    with pytest.raises(TypeError):
        euclidean_distance(test_array_1, test_array_2)

def test_euclidean_distance_type_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = 'a'
    with pytest.raises(TypeError):
        euclidean_distance(test_array_1, test_array_2)

def test_euclidean_distance_shape_mismatch():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7, 9])
    with pytest.raises(ValueError):
        euclidean_distance(test_array_1, test_array_2)

def test_euclidean_distance_naming_input_normal():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value = euclidean_distance(test_array_1, test_array_2, 'test_array_1', 'test_array_2')
    actual_value = np.sqrt(np.sum((test_array_2 - test_array_1) ** 2))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_euclidean_distance_naming_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    with pytest.raises(TypeError):
        euclidean_distance(test_array_1, test_array_2, ['test_array'], 'test_array_2')








def test_manhattan_distance_basic_arrays():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = abs(5 - 2) + abs(7 - 3)
    assert isinstance(result_value, float)
    assert actual_value == np.sum(np.abs(test_array_2 - test_array_1))
    assert result_value == actual_value

def test_manhattan_distance_basic_lists():
    test_array_1 = [2, 3]
    test_array_2 = [5, 7]
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = abs(5 - 2) + abs(7 - 3)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_basic_df():
    test_array_1 = pd.Series([2, 3])
    test_array_2 = pd.Series([5, 7])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = abs(5 - 2) + abs(7 - 3)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_basic_df_row():
    test_df = pd.DataFrame({
            'A': [2, 5], 
            'B': [3, 7],
    })
    test_array_1 = test_df.iloc[0]
    test_array_2 = test_df.iloc[1]
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = abs(5 - 2) + abs(7 - 3)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_basic_tuple():
    test_array_1 = (2, 3)
    test_array_2 = (5, 7)
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = abs(5 - 2) + abs(7 - 3)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_basic_mixed():
    test_array_1 = np.array([2, 3])
    test_array_2 = [5, 7]
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = np.sum(np.abs(test_array_2 - test_array_1))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_zeros():
    test_array_1 = np.array([0, 0])
    test_array_2 = np.array([0, 0])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = 0
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_negatives():
    test_array_1 = np.array([-2, 3])
    test_array_2 = np.array([-5, 7])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = np.sum(np.abs(test_array_2 - test_array_1))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_high_dim():
    test_array_1 = np.array([1, 2, 3, 4, 5])
    test_array_2 = np.array([2, 4, 6, 8, 10])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = np.sum(np.abs(test_array_2 - test_array_1))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_single_element():
    test_array_1 = np.array([1])
    test_array_2 = np.array([2])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = np.sum(np.abs(test_array_2 - test_array_1))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_strings():
    test_array_1 = np.array(['2', 3])
    test_array_2 = np.array([5, 7])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = abs(5 - 2) + abs(7 - 3)
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_floats():
    test_array_1 = np.array([2.0, 3.0])
    test_array_2 = np.array([5.0, 7.0])
    result_value = manhattan_distance(test_array_1, test_array_2)
    actual_value = np.sum(np.abs(test_array_2 - test_array_1))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_symmetry():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value_1 = manhattan_distance(test_array_1, test_array_2)
    result_value_2 = manhattan_distance(test_array_2, test_array_1)
    assert result_value_1 == result_value_2

def test_manhattan_distance_dimensions():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([[5, 7]])
    with pytest.raises(ValueError):
        manhattan_distance(test_array_1, test_array_2)

def test_manhattan_distance_empty():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([])
    with pytest.raises(ValueError):
        manhattan_distance(test_array_1, test_array_2)

def test_manhattan_distance_nan():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, np.nan])
    with pytest.raises(ValueError):
        manhattan_distance(test_array_1, test_array_2)

def test_manhattan_distance_datatype_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array(['a', 'b'])
    with pytest.raises(TypeError):
        manhattan_distance(test_array_1, test_array_2)

def test_manhattan_distance_type_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = 'a'
    with pytest.raises(TypeError):
        manhattan_distance(test_array_1, test_array_2)

def test_manhattan_distance_shape_mismatch():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7, 9])
    with pytest.raises(ValueError):
        manhattan_distance(test_array_1, test_array_2)

def test_manhattan_distance_naming_input_normal():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value = manhattan_distance(test_array_1, test_array_2, 'test_array_1', 'test_array_2')
    actual_value = np.sum(np.abs(test_array_2 - test_array_1))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_manhattan_distance_naming_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    with pytest.raises(TypeError):
        manhattan_distance(test_array_1, test_array_2, ['test_array'], 'test_array_2')








def test_minkowski_distance_basic_arrays():
    test_array_1 = np.array([1,2,3])
    test_array_2 = np.array([2,8,11])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    actual_value = ((2 - 1) ** 3 + (8 - 2) ** 3 + (11 - 3) ** 3) ** (1 / 3)
    assert isinstance(result_value, float)
    assert np.isclose(actual_value, np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3))))
    assert result_value == actual_value

def test_minkowski_distance_basic_lists():
    test_array_1 = [2, 3]
    test_array_2 = [5, 7]
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    test_array_1 = np.asarray(test_array_1)
    test_array_2 = np.asarray(test_array_2)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_basic_df():
    test_array_1 = pd.Series([2, 3])
    test_array_2 = pd.Series([5, 7])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    test_array_1 = np.asarray(test_array_1)
    test_array_2 = np.asarray(test_array_2)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_basic_df_row():
    test_df = pd.DataFrame({
            'A': [2, 5], 
            'B': [3, 7],
    })
    test_array_1 = test_df.iloc[0]
    test_array_2 = test_df.iloc[1]
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    test_array_1 = np.asarray(test_array_1)
    test_array_2 = np.asarray(test_array_2)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_basic_tuple():
    test_array_1 = (2, 3)
    test_array_2 = (5, 7)
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    test_array_1 = np.asarray(test_array_1)
    test_array_2 = np.asarray(test_array_2)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_basic_mixed():
    test_array_1 = np.array([2, 3])
    test_array_2 = [5, 7]
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    test_array_1 = np.asarray(test_array_1)
    test_array_2 = np.asarray(test_array_2)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_zeros():
    test_array_1 = np.array([0, 0])
    test_array_2 = np.array([0, 0])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    actual_value = 0
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_negatives():
    test_array_1 = np.array([-2, 3])
    test_array_2 = np.array([-5, 7])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_high_dim():
    test_array_1 = np.array([1, 2, 3, 4, 5])
    test_array_2 = np.array([2, 4, 6, 8, 10])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert np.isclose(result_value, actual_value)

def test_minkowski_distance_single_element():
    test_array_1 = np.array([1])
    test_array_2 = np.array([2])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_strings():
    test_array_1 = np.array(['2', 3])
    test_array_2 = np.array([5, 7])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    actual_value = np.cbrt((5 - 2) ** 3 + (7 - 3) ** 3)
    assert np.isclose(result_value, actual_value)

def test_minkowski_distance_floats():
    test_array_1 = np.array([2.0, 3.0])
    test_array_2 = np.array([5.0, 7.0])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 3)
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_symmetry():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value_1 = minkowski_distance(test_array_1, test_array_2, p = 3)
    result_value_2 = minkowski_distance(test_array_2, test_array_1, p = 3)
    assert result_value_1 == result_value_2

def test_minkowski_distance_dimensions():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([[5, 7]])
    with pytest.raises(ValueError):
        minkowski_distance(test_array_1, test_array_2, p = 3)

def test_minkowski_distance_empty():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([])
    with pytest.raises(ValueError):
        minkowski_distance(test_array_1, test_array_2, p = 3)

def test_minkowski_distance_nan():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, np.nan])
    with pytest.raises(ValueError):
        minkowski_distance(test_array_1, test_array_2, p = 3)

def test_minkowski_distance_datatype_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array(['a', 'b'])
    with pytest.raises(TypeError):
        minkowski_distance(test_array_1, test_array_2, p = 3)

def test_minkowski_distance_type_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = 'a'
    with pytest.raises(TypeError):
        minkowski_distance(test_array_1, test_array_2, p = 3)

def test_minkowski_distance_shape_mismatch():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7, 9])
    with pytest.raises(ValueError):
        minkowski_distance(test_array_1, test_array_2, p = 3)

def test_minkowski_distance_p_high():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value = minkowski_distance(test_array_1, test_array_2, p = 30)
    actual_value = float(np.exp(np.log(np.sum((np.abs(test_array_2 - test_array_1)) ** 30)) / 30))
    assert isinstance(result_value, float)
    assert np.isclose(result_value, actual_value)

def test_minkowski_distance_p_zero():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    with pytest.raises(ValueError):
        minkowski_distance(test_array_1, test_array_2, p = 0)

def test_minkowski_distance_p_negatives():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    with pytest.raises(ValueError):
        minkowski_distance(test_array_1, test_array_2, p = -1)

def test_minkowski_distance_p_float():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    with pytest.raises(TypeError):
        minkowski_distance(test_array_1, test_array_2, p = 1.5)

def test_minkowski_distance_type_p():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    with pytest.raises(TypeError):
        minkowski_distance(test_array_1, test_array_2, p = '1')

def test_minkowski_distance_naming_input_normal():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_value = minkowski_distance(test_array_1, test_array_2, 3, 'test_array_1', 'test_array_2')
    actual_value = np.cbrt(np.sum(np.abs((test_array_2 - test_array_1) ** 3)))
    assert isinstance(result_value, float)
    assert result_value == actual_value

def test_minkowski_distance_naming_input():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    with pytest.raises(TypeError):
       minkowski_distance(test_array_1, test_array_2, 3,['test_array_1'], 'test_array_2')

def test_minkowski_distance_manhattan():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_array_min = minkowski_distance(test_array_1, test_array_2, 1)
    result_array_man = manhattan_distance(test_array_1, test_array_2)
    assert result_array_min == result_array_man

def test_minkowski_distance_euclidean():
    test_array_1 = np.array([2, 3])
    test_array_2 = np.array([5, 7])
    result_array_min = minkowski_distance(test_array_1, test_array_2, 2)
    result_array_euc = euclidean_distance(test_array_1, test_array_2)
    assert result_array_min == result_array_euc

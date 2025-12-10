
import numpy as np
import pandas as pd
import pytest
from rice_ml.preprocess.datatype import *

# TODO: fix the formatting and spacing, add comments to indicate functions being tested

def test_2D_numeric_basic_array():
    test_array = np.array([[1, 2, 3], [2, 4, 6]])
    result_array = _2D_numeric(test_array, 'test_array')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 3)

def test_2D_numeric_basic_array_float():
    test_array = np.array([[1.1, 2.2, 3.3], [2.2, 4.4, 6.6]])
    result_array = _2D_numeric(test_array, 'test_array')
    assert isinstance(result_array, np.ndarray)
    assert result_array.dtype in [np.float64, np.float32]
    assert result_array.shape == (2, 3)

def test_2D_numeric_basic_df():
    test_df = pd.DataFrame({
        'A': [1, 2, 1, 4, 1],
        'B': [5, 6, 5, 8, 5],
        'C': [9, 10, 9, 12, 9]})
    result_array = _2D_numeric(test_df, 'test_df')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (5, 3)

def test_2D_numeric_basic_list():
    test_list = [[1, 2, 3], [2, 4, 6]]
    result_array = _2D_numeric(test_list, 'test_df')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (2, 3)

def test_2D_numeric_dimensions():
    test_array_1D = np.array([1, 2, 3])
    test_array_3D = np.array([[[1, 2], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        _2D_numeric(test_array_1D, 'test_array_1D')
    with pytest.raises(ValueError):
        _2D_numeric(test_array_3D, 'test_array_3D')

def test_2D_numeric_types():
    test_array = np.array([[1, 2, 3], ['a', 'b', 'c']])
    with pytest.raises(TypeError):
        _2D_numeric(test_array, 'test_array')

def test_2D_numeric_empty():
    test_array = np.array([[]])
    with pytest.raises(ValueError):
        _2D_numeric(test_array, 'test_array')

def test_2D_numeric_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        _2D_numeric(test_array, 'test_array')

def test_2D_numeric_naming_input():
    test_array = np.array([[1, 2, 3], [2, 4, 6]])
    with pytest.raises(TypeError):
        _2D_numeric(test_array, test_array)







def test_1D_numeric_basic_array():
    test_array = np.array([1, 2, 3])
    result_array = _1D_numeric(test_array, 'test_array')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3,)

def test_1D_numeric_basic_array_float():
    test_array = np.array([1.1, 2.2, 3.3])
    result_array = _1D_numeric(test_array, 'test_array')
    assert isinstance(result_array, np.ndarray)
    assert result_array.dtype in [np.float64, np.float32]
    assert result_array.shape == (3,)

def test_1D_numeric_basic_df():
    test_df = pd.Series([1, 2, 1, 4, 1])
    result_array = _1D_numeric(test_df, 'test_df')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (5,)

def test_1D_numeric_basic_list():
    test_list = [1, 2, 3]
    result_array = _1D_numeric(test_list, 'test_list')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3,)

def test_1D_numeric_dimensions():
    test_array_2D = np.array([[1, 2, 3], [2, 4, 6]])
    test_array_3D = np.array([[[1, 2], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        _1D_numeric(test_array_2D, 'test_array_2D')
    with pytest.raises(ValueError):
        _1D_numeric(test_array_3D, 'test_array_3D')

def test_1D_numeric_types():
    test_array = np.array(['a', 'b', 'c'])
    result_array = _1D_numeric(test_array, 'test_array')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3,)

def test_1D_numeric_empty():
    test_array = np.array([])
    with pytest.raises(ValueError):
        _1D_numeric(test_array)

def test_1D_numeric_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        _1D_numeric(test_array, 'test_array')

def test_1D_numeric_naming_input():
    test_array = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        _1D_numeric(test_array, test_array)




def test_shape_match_basic():
    test_array = np.array([[1, 2, 3], [2, 4, 6]])
    test_vector = np.array([1, 2])
    _shape_match(test_array, test_vector)

def test_shape_match_basic_float():
    test_array = np.array([[1.1, 2.2, 3.3], [2.2, 4.4, 6.6]])
    test_vector = np.array([1.1, 2.2])
    _shape_match(test_array, test_vector)

def test_shape_match_none():
    test_array = np.array([[1, 2, 3], [2, 4, 6]])
    test_vector = None
    _shape_match(test_array, test_vector)

def test_shape_match_dimension():
    test_array = np.array([[1, 2, 3], [2, 4, 6]])
    test_vector = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        _shape_match(test_array, test_vector)

def test_shape_match_type_input():
    test_array = 'test_array'
    test_vector = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        _shape_match(test_array, test_vector)

def test_shape_match_type_input():
    test_array = np.array([[1, 2, 3], [2, 4, 6]])
    test_vector = 'test_array'
    with pytest.raises(TypeError):
        _shape_match(test_array, test_vector)


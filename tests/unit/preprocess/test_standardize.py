
import numpy as np
import pandas as pd
import pytest
from rice_ml.preprocess.standardize import *

def test_z_score_standardize_basic_array():
    test_array = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    result_array = z_score_standardize(test_array)
    std = test_array.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    expected_array = (test_array - test_array.mean(axis=0)) / std
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_z_score_standardize_basic_df():
    test_array = pd.DataFrame({
        'A': [1, 0, 1],
        'B': [0, 1, 1],
        'C': [1, 1, 0]})
    result_array = z_score_standardize(test_array)
    test_array = np.array(test_array)
    std = test_array.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    expected_array = (test_array - test_array.mean(axis=0)) / std
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_z_score_standardize_basic_list():
    test_array = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    result_array = z_score_standardize(test_array)
    test_array = np.array(test_array)
    std = test_array.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    expected_array = (test_array - test_array.mean(axis=0)) / std
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_z_score_standardize_dimensions():
    test_array_1D = np.array([1, 2, 3])
    test_array_3D = np.array([[[1, 2], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        z_score_standardize(test_array_1D)
    with pytest.raises(ValueError):
        z_score_standardize(test_array_3D)

def test_z_score_standardize_constants():
    test_array = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]])
    result_array = z_score_standardize(test_array)
    std = test_array.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    expected_array = (test_array - test_array.mean(axis=0)) / std
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_z_score_standardize_single_row():
    test_array = np.array([[1, 2, 3]])
    result_array = z_score_standardize(test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)
    assert np.all(result_array == 0)

def test_z_score_standardize_return_params():
    test_array = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    result_array, params = z_score_standardize(test_array, return_params = True)
    std = test_array.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    mean = test_array.mean(axis=0)
    expected_array = (test_array - mean) / std
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)
    assert isinstance(params, dict)
    assert np.allclose(params['scale'], std)
    assert np.allclose(params['mean'], mean)

def test_z_score_standardize_ddof():
    test_array = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    result_array = z_score_standardize(test_array, ddof = 2)
    std = test_array.std(axis=0, ddof=2)
    std[std == 0] = 1.0
    expected_array = (test_array - test_array.mean(axis=0)) / std
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_z_score_standardize_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        z_score_standardize(test_array)

def test_z_score_standardize_type_return_params():
    test_array = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    with pytest.raises(TypeError):
        z_score_standardize(test_array, return_params = 'True')

def test_z_score_standardize_type_ddof():
    test_array = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    with pytest.raises(TypeError):
        z_score_standardize(test_array, ddof = '1')

def test_z_score_standardize_float_ddof():
    test_array = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    with pytest.raises(TypeError):
        z_score_standardize(test_array, ddof = 1.1)

def test_z_score_standardize_ddof_negative():
    test_array = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    with pytest.raises(ValueError):
        z_score_standardize(test_array, ddof = -1)




def test_min_max_standardize_basic_array():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    result_array = min_max_standardize(test_array)
    expected_array = ((test_array) - (test_array.min(axis = 0))) / (test_array.max(axis = 0) - test_array.min(axis = 0))
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_min_max_standardize_basic_df():
    test_array = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [0, 1, 1],
        'C': [3, 6, 12],
    })
    result_array = min_max_standardize(test_array)
    test_array = np.array(test_array)
    expected_array = ((test_array) - (test_array.min(axis = 0))) / (test_array.max(axis = 0) - test_array.min(axis = 0))
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_min_max_standardize_basic_list():
    test_array = [[1, 0, 3], [2, 1, 6], [3, 1, 12]]
    result_array = min_max_standardize(test_array)
    test_array = np.array(test_array)
    expected_array = ((test_array) - (test_array.min(axis = 0))) / (test_array.max(axis = 0) - test_array.min(axis = 0))
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_min_max_standardize_dimensions():
    test_array_1D = np.array([1, 2, 3])
    test_array_3D = np.array([[[1, 2], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        min_max_standardize(test_array_1D)
    with pytest.raises(ValueError):
        min_max_standardize(test_array_3D)

def test_min_max_standardize_constants():
    test_array = np.array([[1, 0, 3], [1, 1, 3], [1, 1, 3]])
    result_array = min_max_standardize(test_array)
    scale = (test_array.max(axis = 0) - test_array.min(axis = 0))
    scale[scale == 0] = 1
    expected_array = ((test_array) - (test_array.min(axis = 0))) / scale
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_min_max_standardize_single_row():
    test_array = np.array([[1, 0, 3]])
    result_array = min_max_standardize(test_array)
    scale = (test_array.max(axis = 0) - test_array.min(axis = 0))
    scale[scale == 0] = 1
    expected_array = ((test_array) - (test_array.min(axis = 0))) / scale
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)
    assert np.allclose(result_array, expected_array)

def test_min_max_standardize_return_params():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    result_array, params = min_max_standardize(test_array, return_params = True)
    minimums = test_array.min(axis = 0)
    maximums = test_array.max(axis = 0)
    scale = maximums - minimums
    scale[scale == 0] = 1
    expected_array = (test_array - minimums) / scale
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)
    assert isinstance(params, dict)
    assert np.allclose(params['minimum'], minimums)
    assert np.allclose(params['maximum'], maximums)
    assert np.allclose(params['scale'], scale)
    assert params['feature_range'] == (0.0, 1.0)

def test_min_max_standardize_feature_range():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    result_array = min_max_standardize(test_array, feature_range = (-1.0, 1.0))
    scale = (test_array.max(axis = 0) - test_array.min(axis = 0))
    scale[scale == 0] = 1
    expected_array = -1 + ((test_array) - (test_array.min(axis = 0))) / scale * (1 - (-1))
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_min_max_standardize_feature_range_int():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    result_array = min_max_standardize(test_array, feature_range = (2, 5))
    scale = (test_array.max(axis = 0) - test_array.min(axis = 0))
    scale[scale == 0] = 1
    expected_array = 2 + ((test_array) - (test_array.min(axis = 0))) / scale * (5 - 2)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_min_max_standardize_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        min_max_standardize(test_array)

def test_min_max_standardize_type_return_params():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    with pytest.raises(TypeError):
        min_max_standardize(test_array, return_params = 'True')

def test_min_max_standardize_type_feature_range_list():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    with pytest.raises(TypeError):
        min_max_standardize(test_array, feature_range = [0, 1])

def test_min_max_standardize_type_feature_range_array():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    with pytest.raises(TypeError):
        min_max_standardize(test_array, feature_range = np.array([0, 1]))

def test_min_max_standardize_type_feature_range_single_int():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    with pytest.raises(TypeError):
        min_max_standardize(test_array, feature_range = 1)

def test_min_max_standardize_type_feature_range_string():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    with pytest.raises(TypeError):
        min_max_standardize(test_array, feature_range = '(0, 1)')

def test_min_max_standardize_type_feature_range_minmax():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    with pytest.raises(ValueError):
        min_max_standardize(test_array, feature_range = (1.0, 0.0))

def test_min_max_standardize_type_feature_range_length():
    test_array = np.array([[1, 0, 3], [2, 1, 6], [3, 1, 12]])
    with pytest.raises(TypeError):
        min_max_standardize(test_array, feature_range = (0.0, 1.0, 2.0))




def test_max_abs_standardize_basic_array():
    test_array = np.array([[5, 1, 2], [8, 0, 1], [3, 1, 2]])
    result_array = max_abs_standardize(test_array)
    expected_array = test_array / test_array.max(axis = 0)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_max_abs_standardize_basic_df():
    test_array = pd.DataFrame({
        'A': [5, 8, 3],
        'B': [1, 0, 1],
        'C': [2, 1, 2]
    })
    result_array = max_abs_standardize(test_array)
    test_array = np.array(test_array)
    expected_array = test_array / test_array.max(axis = 0)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_max_abs_standardize_basic_list():
    test_array = [[5, 1, 2], [8, 0, 1], [3, 1, 2]]
    result_array = max_abs_standardize(test_array)
    test_array = np.array(test_array)
    expected_array = test_array / test_array.max(axis = 0)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_max_abs_standardize_dimensions():
    test_array_1D = np.array([1, 2, 3])
    test_array_3D = np.array([[[1, 2], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        max_abs_standardize(test_array_1D)
    with pytest.raises(ValueError):
        max_abs_standardize(test_array_3D)

def test_max_abs_standardize_constants():
    test_array = np.array([[5, 1, 2], [5, 0, 1], [5, 1, 2]])
    result_array = max_abs_standardize(test_array)
    expected_array = test_array / test_array.max(axis = 0)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)
    assert np.all(result_array[:, 0] == 1)

def test_max_abs_standardize_single_row():
    test_array = np.array([[5, 1, 2]])
    result_array = max_abs_standardize(test_array)
    expected_array = test_array / test_array.max(axis = 0)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)
    assert np.allclose(result_array, expected_array)
    assert np.all(result_array == 1)

def test_max_abs_standardize_zeros():
    test_array = np.array([[0, 1, 2], [0, 0, 1], [0, 1, 2]])
    result_array = max_abs_standardize(test_array)
    maximums = test_array.max(axis = 0)
    maximums[maximums == 0] = 1
    expected_array = test_array / maximums
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_max_abs_standardize_negatives():
    test_array = np.array([[-5, 1, 2], [4, 0, 1], [1, 1, 2]])
    result_array = max_abs_standardize(test_array)
    test_array_abs = np.abs(test_array)
    maximums = test_array_abs.max(axis = 0)
    maximums[maximums == 0] = 1
    expected_array = test_array / maximums
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.allclose(result_array, expected_array)

def test_max_abs_standardize_return_params():
    test_array = np.array([[5, 1, 2]])
    result_array, params = max_abs_standardize(test_array, return_params = True)
    maximums = test_array.max(axis = 0)
    expected_array = test_array / maximums
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)
    assert np.allclose(result_array, expected_array)
    assert np.all(result_array == 1)
    assert isinstance(params, dict)
    assert np.allclose(params['scale'], maximums)

def test_max_abs_standardize_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        max_abs_standardize(test_array)

def test_max_abs_standardize_return_params_input():
    test_array = np.array([[5, 1, 2], [5, 0, 1], [5, 1, 2]])
    with pytest.raises(TypeError):
        max_abs_standardize(test_array, return_params = 'True')


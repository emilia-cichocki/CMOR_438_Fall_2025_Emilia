
import numpy as np
import warnings
import pandas as pd
from scipy import stats
import pytest
from rice_ml.preprocess.cleaning import *

# TODO: fix the formatting and spacing, add comments to indicate functions being tested

def test_missing_data_basic_drop():
    test_array = np.array([[1, 2, 3], [2, 4, np.nan], [3, np.nan, 9]])
    result_array = missing_data(test_array, 'drop')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)

def test_missing_data_basic_mean():
    test_array = np.array([[1, 2, 3], [2, 4, np.nan], [3, np.nan, 9]])
    result_array = missing_data(test_array, 'mean')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert result_array[1, 2] == np.nanmean(test_array[:,2])
    assert result_array[2, 1] == np.nanmean(test_array[:,1])
    assert np.array_equal(result_array[0, :],test_array[0, :])

def test_missing_data_basic_median():
    test_array = np.array([[1, 2, 3], [2, 4, np.nan], [3, np.nan, 9], [1, 2, 3]])
    result_array = missing_data(test_array, 'median')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4, 3)
    assert result_array[1, 2] == np.nanmedian(test_array[:,2])
    assert result_array[2, 1] == np.nanmedian(test_array[:,1])
    assert np.array_equal(result_array[0, :],test_array[0, :])
    assert np.array_equal(result_array[3, :],test_array[3, :])

def test_missing_data_basic_mode():
    test_array = np.array([[1, 2, 3], [2, 4, np.nan], [3, np.nan, 9], [1, 2, 3]])
    result_array = missing_data(test_array, 'mode')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (4, 3)
    assert result_array[1, 2] == stats.mode(test_array[:, 2][~np.isnan(test_array[:, 2])]).mode
    assert result_array[2, 1] == stats.mode(test_array[:, 1][~np.isnan(test_array[:, 1])]).mode
    assert np.array_equal(result_array[0, :],test_array[0, :])
    assert np.array_equal(result_array[3, :],test_array[3, :])

def test_missing_data_basic_df():
    test_array = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [2, 4, np.nan],
        'C': [3, np.nan, 9]})
    result_array = missing_data(test_array, 'drop')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)
    
def test_missing_data_basic_list():
    test_array = [[1, 2, 3], [2, 4, np.nan], [3, np.nan, 9]]
    result_array = missing_data(test_array, 'drop')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)

def test_missing_data_dimensions():
    test_array_1D = np.array([1, 2, np.nan])
    test_array_3D = np.array([[[1, np.nan], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        missing_data(test_array_1D, 'drop')
    with pytest.raises(ValueError):
        missing_data(test_array_3D, 'drop')

def test_missing_data_nan_drop():
    test_array = np.array([[1, 2, np.nan], [2, 4, np.nan], [3, np.nan, np.nan]])
    result_array = missing_data(test_array, 'drop')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (0, 3)

def test_missing_data_nan_mean():
    test_array = np.array([[1, 2, np.nan], [2, 4, np.nan], [3, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
        result_array = missing_data(test_array, 'mean')
        assert isinstance(result_array, np.ndarray)
        assert result_array.shape == (3, 3)
        assert np.isnan(result_array[:, 2]).all()
        assert result_array[2, 1] == np.nanmean(test_array[:,1])

def test_missing_data_nan_median():
    test_array = np.array([[1, 2, np.nan], [2, 4, np.nan], [3, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
        result_array = missing_data(test_array, 'median')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.isnan(result_array[:, 2]).all()
    assert result_array[2, 1] == np.nanmedian(test_array[:,1])

def test_missing_data_nan_mode():
    test_array = np.array([[1, 2, np.nan], [2, 4, np.nan], [3, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
        result_array = missing_data(test_array, 'mode')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.isnan(result_array[:, 2]).all()
    assert result_array[2, 1] == stats.mode(test_array[:, 1][~np.isnan(test_array[:, 1])]).mode

def test_missing_data_all_nan_mean():
    test_array = np.array([[np.nan, np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)
        result_array = missing_data(test_array, "mean")
    assert np.isnan(test_array).all()

def test_missing_data_nonan_drop():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    result_array = missing_data(test_array, 'drop')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.array_equal(test_array, result_array)

def test_missing_data_nonan_mean():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    result_array = missing_data(test_array, 'mean')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.array_equal(test_array, result_array)

def test_missing_data_nonan_median():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    result_array = missing_data(test_array, 'median')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.array_equal(test_array, result_array)

def test_missing_data_nonan_mode():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    result_array = missing_data(test_array, 'mode')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    assert np.array_equal(test_array, result_array)

def test_missing_data_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        missing_data(test_array, 'drop')

def test_missing_data_type_strategy_string():
    test_array = np.array([[1, 2, 3], [2, 4, np.nan], [3, np.nan, 9]])
    with pytest.raises(ValueError):
        missing_data(test_array, 'drop_row')

def test_missing_data_type_strategy_other():
    test_array = np.array([[1, 2, 3], [2, 4, np.nan], [3, np.nan, 9]])
    with pytest.raises(ValueError):
        missing_data(test_array, 0)







def test_outlier_identify_basic_IQR_keep():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    result_array = outlier_identify(test_array, 'IQR')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (5, 3)

def test_outlier_identify_basic_IQR_drop():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    result_array = outlier_identify(test_array, 'IQR', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_basic_zscore_keep():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    result_array = outlier_identify(test_array, 'IQR')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (5, 3)
    
def test_outlier_identify_basic_zscore_drop():
    test_array = np.array([
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [50, 1, 1]
                        ])
    result_array = outlier_identify(test_array, 'zscore', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (10, 3)

def test_outlier_identify_basic_df():
    test_array = pd.DataFrame({
        'A': [10, 12, 11, 100, 13],
        'B': [20, 22, 21, 20, 200],
        'C': [30, 29, 31, 30, 28]})
    result_array = outlier_identify(test_array, 'IQR', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)
    
def test_outlier_identify_basic_list():
    test_array = [[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]]
    result_array = outlier_identify(test_array, 'IQR', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_dimensions():
    test_array_1D = np.array([1, 2, 3])
    test_array_3D = np.array([[[1, 2], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        outlier_identify(test_array_1D, 'IQR', drop = True)
    with pytest.raises(ValueError):
        outlier_identify(test_array_3D, 'IQR', drop = True)

def test_outlier_identify_no_outlier_IQR_keep():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31]])
    result_array = outlier_identify(test_array, 'IQR')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_no_outlier_IQR_drop():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31]])
    result_array = outlier_identify(test_array, 'IQR', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_no_outlier_zscore_keep():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31]])
    result_array = outlier_identify(test_array, 'zscore')
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_no_outlier_zscore_drop():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31]])
    result_array = outlier_identify(test_array, 'zscore', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_IQR_constants():
    test_array = np.array([[1, 1, 1], [1, 1, 1], [50, 1, 1]])
    result_array = outlier_identify(test_array, 'IQR', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_zscore_constants():
    test_array = np.array([[1, 1, 1], [1, 1, 1], [50, 1, 1]])
    result_array = outlier_identify(test_array, 'zscore', drop = True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_outlier_identify_IQR_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        outlier_identify(test_array, 'IQR')

def test_outlier_identify_zscore_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        outlier_identify(test_array, 'zscore')

def test_outlier_identify_strategy_string():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    with pytest.raises(ValueError):
        outlier_identify(test_array, 'zscoring')

def test_outlier_identify_strategy_other():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    with pytest.raises(ValueError):
        outlier_identify(test_array, 0)

def test_outlier_identify_drop_type():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    with pytest.raises(TypeError):
        outlier_identify(test_array, 'IQR', drop = 'no')

def test_outlier_identify_threshold_float():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    result_array = outlier_identify(test_array, 'zscore', threshold = 2.9)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (5, 3)

def test_outlier_identify_threshold_negative():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    with pytest.raises(ValueError):
        outlier_identify(test_array, 'zscore', threshold = -1)

def test_outlier_identify_type_threshold():
    test_array = np.array([[10, 20, 30], [12, 22, 29], [11, 21, 31], [100, 20, 30], [13, 200, 28]])
    with pytest.raises(TypeError):
        outlier_identify(test_array, 'zscore', threshold = '3')
    





def test_duplicate_identify_basic_array_keep():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [1, 2, 3], [3, 6, 9]])
    result_array = duplicate_identify(test_array)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (5, 3)
    assert np.array_equal(test_array, result_array)

def test_duplicate_identify_basic_array_drop():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [1, 2, 3], [3, 6, 9]])
    result_array = duplicate_identify(test_array, True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_duplicate_identify_basic_df():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [1, 2, 3], [3, 6, 9]])
    result_array = duplicate_identify(test_array, True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_duplicate_identify_basic_list():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [1, 2, 3], [3, 6, 9]])
    result_array = duplicate_identify(test_array, True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_duplicate_identify_dimensions():
    test_array_1D = np.array([1, 2, 3])
    test_array_3D = np.array([[[1, 2], [1, 2]],[[2, 4], [5, 10]]])
    with pytest.raises(ValueError):
        duplicate_identify(test_array_1D, True)
    with pytest.raises(ValueError):
        duplicate_identify(test_array_3D, True)

def test_duplicate_identify_no_duplicates():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    result_array = duplicate_identify(test_array, True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3, 3)

def test_duplicate_identify_only_duplicates():
    test_array = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    result_array = duplicate_identify(test_array, True)
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (1, 3)

def test_duplicate_identify_type_input():
    test_array = 'test_array'
    with pytest.raises(TypeError):
        duplicate_identify(test_array, True)

def test_duplicate_identify_drop_type():
    test_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [1, 2, 3], [3, 6, 9]])
    with pytest.raises(TypeError):
        duplicate_identify(test_array, 'drop')


import numpy as np
import pytest
from rice_ml.postprocess.regressionpost import _validate_vector_match, mae, mse, rmse, r2, adjusted_r2

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
    with pytest.raises(TypeError):
        _validate_vector_match(test_pred_array, test_true_array)

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

def test_mae_basic():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    test_mae = mae(test_pred, test_true)
    assert isinstance(test_mae, float)
    assert np.isclose(test_mae, 0.133333333)

def test_mae_perfect():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.0, 2.0, 3.0])
    test_mae = mae(test_pred, test_true)
    assert isinstance(test_mae, float)
    assert np.isclose(test_mae, 0.0)

def test_mae_negatives():
    test_true = np.array([-1, 0, 1])
    test_pred = np.array([-2, 1, 0])
    test_mae = mae(test_pred, test_true)
    assert isinstance(test_mae, float)
    assert np.isclose(test_mae, 1.0)

def test_mae_dimensions():
    test_true = np.array([[1.0, 2.0, 3.0]])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        mae(test_pred, test_true)
    
def test_mae_strings():
    test_true = np.array([1.0, 2.0, 'A'])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(TypeError):
        mae(test_pred, test_true)

def test_mae_shape_mismatch():
    test_true = np.array([1.0, 2.0, 3.0, 4.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        mae(test_pred, test_true)

def test_mse_basic():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    test_mse = mse(test_pred, test_true)
    assert isinstance(test_mse, float)
    assert np.isclose(test_mse, 0.02)

def test_mse_perfect():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.0, 2.0, 3.0])
    test_mse = mse(test_pred, test_true)
    assert isinstance(test_mse, float)
    assert np.isclose(test_mse, 0.0)

def test_mse_negatives():
    test_true = np.array([-1, 0, 1])
    test_pred = np.array([-2, 1, 0])
    test_mse = mse(test_pred, test_true)
    assert isinstance(test_mse, float)
    assert np.isclose(test_mse, 1.0)

def test_mse_dimensions():
    test_true = np.array([[1.0, 2.0, 3.0]])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        mse(test_pred, test_true)
    
def test_mse_strings():
    test_true = np.array([1.0, 2.0, 'A'])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(TypeError):
        mse(test_pred, test_true)

def test_mse_shape_mismatch():
    test_true = np.array([1.0, 2.0, 3.0, 4.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        mse(test_pred, test_true)

def test_rmse_basic():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    test_rmse = rmse(test_pred, test_true)
    assert isinstance(test_rmse, float)
    assert np.isclose(test_rmse, np.sqrt(0.02))

def test_rmse_perfect():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.0, 2.0, 3.0])
    test_rmse = rmse(test_pred, test_true)
    assert isinstance(test_rmse, float)
    assert np.isclose(test_rmse, 0.0)

def test_rmse_negatives():
    test_true = np.array([-1, 0, 1])
    test_pred = np.array([-2, 1, 0])
    test_rmse = rmse(test_pred, test_true)
    assert isinstance(test_rmse, float)
    assert np.isclose(test_rmse, 1.0)

def test_rmse_dimensions():
    test_true = np.array([[1.0, 2.0, 3.0]])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        rmse(test_pred, test_true)
    
def test_rmse_strings():
    test_true = np.array([1.0, 2.0, 'A'])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(TypeError):
        rmse(test_pred, test_true)

def test_rmse_shape_mismatch():
    test_true = np.array([1.0, 2.0, 3.0, 4.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        rmse(test_pred, test_true)

def test_r2_basic():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    test_r2 = r2(test_pred, test_true)
    expected = 1 - np.sum((test_pred - test_true) ** 2) / np.sum((test_true - np.mean(test_true)) ** 2)
    assert isinstance(test_r2, float)
    assert np.isclose(test_r2, expected)

def test_r2_perfect():
    test_true = np.array([1.0, 2.0, 3.0])
    test_pred = np.array([1.0, 2.0, 3.0])
    test_r2 = r2(test_pred, test_true)
    assert isinstance(test_r2, float)
    assert np.isclose(test_r2, 1.0)

def test_r2_negatives():
    test_true = np.array([-1, 0, 1])
    test_pred = np.array([-2, 1, 0])
    test_r2 = r2(test_pred, test_true)
    assert isinstance(test_r2, float)
    assert np.isclose(test_r2, -0.5)

def test_r2_dimensions():
    test_true = np.array([[1.0, 2.0, 3.0]])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        r2(test_pred, test_true)
    
def test_r2_strings():
    test_true = np.array([1.0, 2.0, 'A'])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(TypeError):
        r2(test_pred, test_true)

def test_r2_shape_mismatch():
    test_true = np.array([1.0, 2.0, 3.0, 4.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        r2(test_pred, test_true)

def test_r2_constant_errors():
    test_true = np.array([1.0, 1.0, 1.0])
    test_pred = np.array([1.1, 1.9, 3.2])
    with pytest.raises(ValueError):
        r2(test_pred, test_true)

def test_r2_constant_perfect():
    test_true = np.array([1.0, 1.0, 1.0])
    test_pred = np.array([1.0, 1.0, 1.0])
    test_r2 = r2(test_pred, test_true)
    assert isinstance(test_r2, float)
    assert np.isclose(test_r2, 1.0)

def test_adj_r2_basic():
    test_true = np.array([1, 2, 3])
    test_pred = np.array([1.1, 1.9, 3.2])
    test_adj_r2 = adjusted_r2(test_pred, test_true, n_features = 1)
    test_r2 = r2(test_pred, test_true)
    expected_adj_r2 = 1 - ((1 - test_r2) * (3 - 1) / (3 - 1 - 1))
    assert isinstance(test_adj_r2, float)
    assert np.isclose(test_adj_r2, expected_adj_r2)

def test_adj_r2_perfect():
    test_true = np.array([1, 2, 3])
    test_pred = np.array([1, 2, 3])
    test_adj_r2 = adjusted_r2(test_pred, test_true, n_features = 1)
    test_r2 = r2(test_pred, test_true)
    expected_adj_r2 = 1 - ((1 - test_r2) * (3 - 1) / (3 - 1 - 1))
    assert isinstance(test_adj_r2, float)
    assert np.isclose(test_adj_r2, expected_adj_r2)
    assert test_adj_r2 == 1.0

def test_adj_r2_multi_feature():
    test_true = np.array([2, 4, 6, 8])
    test_pred = np.array([2.1, 3.9, 6.2, 7.8])
    test_adj_r2 = adjusted_r2(test_pred, test_true, n_features = 2)
    test_r2 = r2(test_pred, test_true)
    expected_adj_r2 = 1 - ((1 - test_r2) * (4 - 1) / (4 - 2 - 1))
    assert isinstance(test_adj_r2, float)
    assert np.isclose(test_adj_r2, expected_adj_r2)

def test_adj_r2_negatives():
    test_true = np.array([-1, 0, 1, 1])
    test_pred = np.array([-2, 1, 0, 1])
    test_adj_r2 = adjusted_r2(test_pred, test_true, n_features = 2)
    test_r2 = r2(test_pred, test_true)
    expected_adj_r2 = 1 - ((1 - test_r2) * (4 - 1) / (4 - 2 - 1))
    assert isinstance(test_adj_r2, float)
    assert np.isclose(test_adj_r2, expected_adj_r2)

def test_adj_r2_feature_min():
    test_true = np.array([-1, 0, 1])
    test_pred = np.array([-2, 1, 0])
    with pytest.raises(ValueError):
        adjusted_r2(test_pred, test_true, n_features = 2)

def test_adj_r2_feature_type():
    test_true = np.array([-1, 0, 1])
    test_pred = np.array([-2, 1, 0])
    with pytest.raises(TypeError):
        adjusted_r2(test_pred, test_true, n_features = '1')

def test_adj_r2_feature_value():
    test_true = np.array([-1, 0, 1])
    test_pred = np.array([-2, 1, 0])
    with pytest.raises(ValueError):
        adjusted_r2(test_pred, test_true, n_features = -1)

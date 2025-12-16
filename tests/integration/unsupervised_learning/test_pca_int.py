
import numpy as np
import pandas as pd
import pytest
from rice_ml.preprocess.standardize import z_score_standardize
from rice_ml.unsupervised_learning.pca import *

def test_pca_fit_basic_array():
    test_array = np.array([
            [0, 1, 2],
            [3, 1, 3],
            [5, 4.5, 0],
            [-1, 2, 4],
            [0, 5.5, 10.5]
        ])
    pca = PCA(2)
    pca.fit(test_array)
    assert isinstance(pca.eigenvalues, np.ndarray)
    assert isinstance(pca.variance, np.ndarray)
    assert isinstance(pca.components, np.ndarray)
    assert pca.eigenvalues.shape == (2,)
    assert pca.variance.shape == (2,)
    assert pca.components.shape == (3, 2)

def test_pca_fit_basic_df():
    test_array = pd.DataFrame({
            'A': [0, 1, 2],
            'B': [3, 1, 3],
            'C': [5, 4.5, 0],
            'D': [-1, 2, 4],
            'E': [0, 5.5, 10.5]})
    pca = PCA(2)
    pca.fit(test_array)
    assert isinstance(pca.eigenvalues, np.ndarray)
    assert isinstance(pca.variance, np.ndarray)
    assert isinstance(pca.components, np.ndarray)
    assert pca.eigenvalues.shape == (2,)
    assert pca.variance.shape == (2,)
    assert pca.components.shape == (5, 2)

def test_pca_fit_dimensions():
    test_array = np.array([[
            [0, 1, 2],
            [3, 1, 3],
            [5, 4.5, 0],
            [-1, 2, 4],
            [0, 5.5, 10.5]
        ]])
    pca = PCA(2)
    with pytest.raises(ValueError):
        pca.fit(test_array)

def test_pca_fit_features():
    test_array = np.array([
            [0, 1, 2],
            [3, 1, 3],
            [5, 4.5, 0],
            [-1, 2, 4],
            [0, 5.5, 10.5]
        ])
    pca = PCA(4)
    with pytest.raises(ValueError):
        pca.fit(test_array)

def test_pca_fit_sum():
    test_array = np.array([
            [0, 1, 2],
            [3, 1, 3],
            [5, 4.5, 0],
            [-1, 2, 4],
            [0, 5.5, 10.5]
        ])
    pca = PCA(3)
    pca.fit(test_array)
    assert np.isclose(np.sum(pca.variance), 1.0)

def test_pca_fit_orthonormal():
    test_array = np.array([
        [0, 1, 2],
        [3, 1, 3],
        [5, 4.5, 0],
        [-1, 2, 4],
        [0, 5.5, 10.5]
    ])
    pca = PCA(3)
    pca.fit(test_array)
    comp = pca.components
    identity = np.dot(comp.T, comp)
    assert np.allclose(identity, np.eye(3), atol=1e-6)

def test_pca_transform_basic():
    test_array = np.array([
        [0, 1, 2],
        [3, 1, 3],
        [5, 4.5, 0],
        [-1, 2, 4],
        [0, 5.5, 10.5]
    ])
    pca = PCA(2)
    pca.fit(test_array)
    transformed = pca.transform(test_array)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (5, 2)

def test_pca_transform_values():
    test_array = np.array([
        [0, 1, 2],
        [3, 1, 3],
        [5, 4.5, 0],
        [-1, 2, 4],
        [0, 5.5, 10.5]
    ])
    pca = PCA(2)
    pca.fit(test_array)
    test_array_standardized = z_score_standardize(test_array)
    transformed = pca.transform(test_array)
    np.testing.assert_allclose(transformed, np.dot(test_array_standardized, pca.components), atol=1e-6)

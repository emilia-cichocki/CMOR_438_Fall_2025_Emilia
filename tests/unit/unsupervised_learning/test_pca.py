
import numpy as np
import pandas as pd
import pytest
from rice_ml.preprocess.standardize import z_score_standardize
from rice_ml.unsupervised_learning.pca import *

def test_pca_init_basic():
    pca = PCA(3)
    assert pca.n_components == 3
    assert pca.components is None
    assert pca.eigenvalues is None
    assert pca.variance is None

def test_pca_init_type_input():
    with pytest.raises(TypeError):
        PCA('3')
    with pytest.raises(TypeError):
        PCA(3.5)

def test_pca_init_value_input():
    with pytest.raises(ValueError):
        PCA(-3)

def test_pca_verify_fit_basic():
    test_array = np.array([
        [0, 1, 2],
        [3, 1, 3],
        [5, 4.5, 0],
        [-1, 2, 4],
        [0, 5.5, 10.5]
    ])
    pca = PCA(3)
    pca.fit(test_array)
    pca._verify_fit()

def test_pca_verify_fit_unfit():
    pca = PCA(3)
    with pytest.raises(RuntimeError):
        pca._verify_fit()
"""
    Perceptron algorithms (NumPy)

    This module implements the single-layer and multilayer perceptron 

    Functions
    ---------
    

    Classes
    ---------
   
"""

# TODO: finish above!

__all__ = [
    'perceptron',
]

import numpy as np
import pandas as pd
from typing import *
import warnings
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]
"""
    Community detection algorithms (NumPy)

    This module implements label propagation for community detection.

    # TODO: finish this!

    Functions
    ---------
    

    Classes
    ---------
   
"""

# TODO: finish above! and check below for redundant imports

__all__ = [
    'label_prop',
]

import numpy as np
import pandas as pd
from typing import *
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric, euclidean_distance

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]


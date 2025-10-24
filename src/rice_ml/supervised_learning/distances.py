
# Houses distance functions

# "Write two distance functions in Python w/ professional error handling and accompanying unit tests. Please have Euclidean distance as one of the functions. Also, please write NumPy style docstrings that can be tested automatically"
# Rewrite to accept numpy arrays as inputs
# NumPy style docstrings - most conventional style
# Manhattan and Euclidean
# _function is developer-private function (not messed w/ by user)

from typing import Sequence
import math

__all__ = [
    'euclidean_distance',
    'manhattan_distance',
]

def _validate_vectors(a: Sequence[float], b: Sequence[float]) -> ...:
    ...

def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> ...:
    ...

def manhattan_distance(a: Sequence[float], b: Sequence[float]) -> ...:
    ...
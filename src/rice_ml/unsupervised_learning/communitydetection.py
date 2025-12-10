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
import networkx as nx
from typing import *
from collections import defaultdict
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number
from rice_ml.supervised_learning.distances import _ensure_numeric, euclidean_distance

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def validate_parameters(graph: nx.Graph, max_iter: int, random_state: Optional[int] = None):

    if not isinstance(graph, nx.Graph):
        raise TypeError("Input graph must be a NetworkX graph")
    if not isinstance(max_iter, int):
        raise TypeError("Maximum number of iterations must be an integer")
    if max_iter <= 0:
        raise ValueError("Maximum number of iterations must be greater than zero")
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError("Random state parameter must be an integer")

class label_propagation():

    def __init__(self,
                 graph: nx.Graph,
                 max_iter: int,
                 random_state: Optional[int] = None) -> None:
        validate_parameters(graph, max_iter, random_state)

        self.graph = graph
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels: Dict[Hashable, Hashable] = {node: node for node in self.graph.nodes}
        self.fit_: bool = False

    def _label_updates(self, 
                       node: Hashable) -> Hashable:
        
        try:
            hash(node)
        except TypeError:
            raise TypeError(f"Input for node ({node}) is not hashable")
        
        rng = _random_number(self.random_state)
        
        frequency_counts = defaultdict(int)
        for neighbor in list(self.graph.neighbors(node)):
            frequency_counts[self.labels[neighbor]] += 1

        if not frequency_counts:
            return self.labels[node]
        
        maximum_count = max(frequency_counts.values())
        tied_labels = [label for label, count in frequency_counts.items() if count == maximum_count]
        tiebreaker = rng.choice(tied_labels)

        return tiebreaker
    
    def propagation(self) -> 'label_propagation':

        # TODO: type hints/docstrings - the dictionary is node, label

        rng = _random_number(self.random_state)

        for _ in range(self.max_iter):

            changed = False

            shuffled_nodes = rng.permutation(list(self.graph.nodes()))

            for node in shuffled_nodes:
                new_label = self._label_updates(node)
                if new_label != self.labels[node]:
                    self.labels[node] = new_label
                    changed = True

            if not changed:
                break
            
        self.fit_ = True

        return self
    
    def _verify_fit(self) -> "label_propagation":
        if not self.fit_:
            raise RuntimeError("Label propagation algorithm has not been run; call propagation()")
        
    def get_communities(self, data_type = Literal['dict', 'list']) -> list:

        self._verify_fit()

        if data_type not in ('dict', 'list'):
            raise ValueError(f"Data type must be one of {['dict', 'list']}")
        community_dict = defaultdict(set)
        for node, label in self.labels.items():
            community_dict[label].add(node)

        if data_type == 'dict':
            return dict(community_dict)
        
        elif data_type == 'list':
            return list(community_dict.values())
    

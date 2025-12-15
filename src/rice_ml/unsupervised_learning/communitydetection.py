
"""
    Community detection algorithms (NetworkX)

    This module implements label propagation for unsupervised community detection.
    It supports label propagation for NetworkX graphs, using an iterative process that
    diffuses unique labels through the network.
    
    Classes
    ---------
    label_propagation:
        Implements the unsupervised label propagation algorithm
"""

__all__ = [
    'label_propagation',
]

import numpy as np
import pandas as pd
import networkx as nx
from typing import *
from collections import defaultdict
from rice_ml.preprocess.datatype import *
from rice_ml.preprocess.split import _random_number

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]], pd.DataFrame, pd.Series]

def _validate_parameters(graph: nx.Graph, max_iter: int, random_state: Optional[int] = None) -> None:
    
    """
    Validates hyperparameters for label propagation

    Parameters
    ----------
    graph: nx.Graph
        Graph used in label propagation
    max_iter: int
        Maximum number of iterations
    random_state: int, optional
        Random state for tiebreaker selection

    Raises
    ------
    TypeError
        If parameters are not of valid types
    ValueError
        If parameters do not have appropriate values
    """

    if not isinstance(graph, nx.Graph):
        raise TypeError("Input graph must be a NetworkX graph")
    if not isinstance(max_iter, int):
        raise TypeError("Maximum number of iterations must be an integer")
    if max_iter <= 0:
        raise ValueError("Maximum number of iterations must be greater than zero")
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError("Random state parameter must be an integer")

class label_propagation():
    
    """
    Class for the unsupervised label propagation algorithm

    Attributes
    ----------
    graph: nx.Graph
        Graph used in label propagation
    max_iter: int
        Maximum number of iterations
    random_state: int, optional
        Random state for tiebreaker selection; if None, a randomized
        seed is used
    labels: dictionary
        Dictionary containing node labels
    rng: Generator
        Random number generator created using the given random state
    fit_: boolean
        Whether the model has been fit

    Methods
    -------
    propagation():
        Implements the label propagation algorithm on a given graph
    get_communities(data_type):
        Returns the labeled communities as a dictionary or list
    
    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> model = label_propagation(G, max_iter=10, random_state=42)
    >>> _ = model.propagation()
    >>> communities_dict = model.get_communities(data_type='dict')
    >>> isinstance(communities_dict, dict)
    True
    >>> communities_list = model.get_communities(data_type='list')
    >>> isinstance(communities_list, list)
    True
    """

    def __init__(self,
                 graph: nx.Graph,
                 max_iter: int,
                 random_state: Optional[int] = None) -> None:
        """
        Creates associated attributes for a label propagation model with
        validated parameters

        Parameters
        ----------
        graph: nx.Graph
            Graph used in label propagation
        max_iter: int
            Maximum number of iterations
        random_state: int, optional
            Random state for tiebreaker selection
        """
        _validate_parameters(graph, max_iter, random_state)

        self.graph = graph
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels: Dict[Hashable, Hashable] = {node: node for node in self.graph.nodes}
        self.rng: Generator = _random_number(random_state)
        self.fit_: bool = False

    def _label_updates(self, 
                       node: Hashable) -> Hashable:
        
        """
        Updates a node label based on the label frequency and type of 
        neighboring nodes
        """

        try:
            hash(node)
        except TypeError:
            raise TypeError(f"Input for node ({node}) is not hashable")
                
        frequency_counts = defaultdict(int)
        for neighbor in list(self.graph.neighbors(node)):
            frequency_counts[self.labels[neighbor]] += 1

        if not frequency_counts:
            return self.labels[node]
        
        maximum_count = max(frequency_counts.values())
        tied_labels = [label for label, count in frequency_counts.items() if count == maximum_count]
        tiebreaker = self.rng.choice(tied_labels)

        return tiebreaker
    
    def propagation(self) -> 'label_propagation':

        """
        Executes the label propagation algorithm on the graph through
        iteratively updating node labels based on connecting nodes

        Returns
        -------
        label_propagation
            Fitted label propagation model

        Raises
        ------
        TypeError
            If the input for a node is not hashable
        """

        for _ in range(self.max_iter):

            changed = False

            shuffled_nodes = self.rng.permutation(list(self.graph.nodes()))

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

        """
        Verifies that the model has been properly run
        """

        if not self.fit_:
            raise RuntimeError("Label propagation algorithm has not been run; call propagation()")
        
    def get_communities(self, data_type = Literal['dict', 'list']) -> Union[dict, list]:

        """
        Gets the labels for each node as a dictionary or list

        Parameters
        ----------
        data_type: {'dict', 'list'}
            Data type that is returned
            - dict: returns a dictionary of communities with associated nodes
            - list: returns a list of nodes in each community

        Returns
        -------
        dict
            Dictionary of nodes in a given community
        list
            List of nodes in each community

        Raises
        ------
        ValueError
            If `data_type` is not in {'dict', 'list'}
        RuntimeError
            If the model propagation step has not been run
        """

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
    

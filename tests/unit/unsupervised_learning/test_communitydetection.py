
import pytest
import networkx as nx
from collections.abc import Hashable
from rice_ml.unsupervised_learning.communitydetection import _validate_parameters, label_propagation

def test_validate_parameters_basic():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2), (2, 3)])
    _validate_parameters(G, 100, None)

def test_validate_parameters_type_inputs():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2), (2, 3)])
    with pytest.raises(TypeError):
        _validate_parameters('G', 100, None)
    with pytest.raises(TypeError):
        _validate_parameters(G, '100', None)
    with pytest.raises(TypeError):
        _validate_parameters(G, '100', '42')

def test_validate_parameters_input_values():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2), (2, 3)])
    with pytest.raises(ValueError):
        _validate_parameters(G, -1, None)

def test_label_prop_init_basic():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    lprop = label_propagation(G, 100)
    assert lprop.graph == G
    assert lprop.max_iter == 100
    assert lprop.random_state is None
    assert isinstance(lprop.labels, dict)
    assert lprop.labels == {1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
    assert not lprop.fit_

def test_label_prop_init_type_inputs():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    with pytest.raises(TypeError):
        label_propagation('G', 100)
    with pytest.raises(TypeError):
        label_propagation(G, '100')
    with pytest.raises(TypeError):
        label_propagation(G, 100, 'None')

def test_label_prop_init_input_values():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    with pytest.raises(ValueError):
        label_propagation(G, -1, 42)

def test_label_updates_basic():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    lprop = label_propagation(G, 100, 42)
    test_case = lprop._label_updates(1)
    assert isinstance(test_case, Hashable)
    assert test_case in [2, 3]

def test_label_updates_no_neighbor():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    G.add_node(7)
    lprop = label_propagation(G, 100, 42)
    test_case = lprop._label_updates(7)
    assert isinstance(test_case, Hashable)
    assert test_case == 7

def test_label_updates_type_input():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    lprop = label_propagation(G, 100, 42)
    with pytest.raises(TypeError):
        lprop._label_updates([1])

def test_label_updates_type_nodes():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'),('B', 'C'),('C', 'A'),('D', 'E'),('E', 'F'),('F', 'D'),('C', 'D')])
    lprop = label_propagation(G, 100, 42)
    test_case = lprop._label_updates('A')
    assert isinstance(test_case, Hashable)
    assert test_case in ['B', 'C']

def test_label_updates_ties():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    lprop_1 = label_propagation(G, 100, 42)
    test_case_1 = lprop_1._label_updates(1)
    lprop_2 = label_propagation(G, 100, 42)
    test_case_2 = lprop_2._label_updates(1)
    assert test_case_1 == test_case_2

def test_label_updates_single_neighbor():
    G = nx.Graph()
    G.add_edges_from([(1, 2)])
    lprop = label_propagation(G, 100, 42)
    test_case = lprop._label_updates(1)
    assert isinstance(test_case, Hashable)
    assert test_case == 2

def test_verify_fit_basic():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    lprop = label_propagation(G, 100, 42)
    lprop.propagation()
    lprop = lprop._verify_fit()
    assert lprop is lprop

def test_verify_fit_unfit():
    G = nx.Graph()
    G.add_edges_from([(1,2),(2,3),(3,1),(4,5),(5,6),(6,4),(3,4)])
    lprop = label_propagation(G, 100, 42)
    with pytest.raises(RuntimeError):
        lprop._verify_fit()
"""
Test suite for QuantumNetwork qpu_topologies parameter.

Tests the QuantumNetwork.create() factory method for creating networks
with different inter-qpu topologies.
"""

import pytest
from disqco import QuantumNetwork
import networkx as nx


def test_topology_stored_on_construction():
    """Test that the qpu_topologies parameter is stored correctly in the QuantumNetwork object."""
    graph = nx.path_graph(4)
    network = QuantumNetwork.create([4, 4], qpu_topologies={0: graph})
    assert network.qpu_topologies[0] is graph
    assert 1 not in network.qpu_topologies  # No topology specified for QPU 1

def test_default_is_empty_dict():
    """Test that the default qpu_topologies is an empty dict if not provided."""
    network = QuantumNetwork.create([4, 4])
    assert network.qpu_topologies == {}

def test_default_not_shared_between_instances():
    """Test that the default qpu_topologies is not shared between instances."""
    network1 = QuantumNetwork.create([4, 4])
    network2 = QuantumNetwork.create([4, 4])
    assert network1.qpu_topologies is not network2.qpu_topologies

def test_rejects_non_dict():
    """Test that providing a non-dict for qpu_topologies raises a TypeError."""
    with pytest.raises(TypeError):
        QuantumNetwork.create([4, 4], qpu_topologies=[nx.path_graph(4)])

def test_rejects_unknown_qpu_index():
    """Tests that providing an invalid qpu index raises a ValueError."""
    with pytest.raises(ValueError):
        QuantumNetwork.create([4, 4], qpu_topologies={2: nx.path_graph(4)})

def test_non_graph_value():
    """Tests that providing a non-graph value raises a TypeError."""
    with pytest.raises(TypeError):
        QuantumNetwork.create([4, 4], qpu_topologies={0: "linear"})

def test_rejects_wrong_node_set():
    """Tests that providing a graph with the wrong node set raises a ValueError."""
    with pytest.raises(ValueError):
        QuantumNetwork.create([4, 4], qpu_topologies={0: nx.path_graph(5)})

def test_rejects_disconnected_graph():
    """Tests that providing a disconnected graph raises a ValueError."""
    with pytest.raises(ValueError):
        graph = nx.path_graph(4)
        graph.remove_edge(1, 2)  # Disconnect the graph
        QuantumNetwork.create([4, 4], qpu_topologies={0: graph})

def test_copy_preserves_topologies():
    """Tests that copying a QuantumNetwork preserves the qpu_topologies."""
    graph = nx.path_graph(4)
    network = QuantumNetwork.create([4, 4], qpu_topologies={0: graph})
    network_copy = network.copy()
    assert network_copy.qpu_topologies[0] is graph
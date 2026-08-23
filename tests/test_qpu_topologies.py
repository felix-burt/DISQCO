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
    """A graph whose node count is neither n (data only) nor n + c
    (data + comm) raises a ValueError. With n=4, c=1 both 4 and 5 nodes
    are legal, so 6 is the smallest illegal size."""
    with pytest.raises(ValueError):
        QuantumNetwork.create([4, 4], qpu_topologies={0: nx.path_graph(6)})
    with pytest.raises(ValueError):
        QuantumNetwork.create([4, 4], qpu_topologies={0: nx.path_graph(3)})

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


# ---------------------------------------------------------------------------
# Comm qubits as topology nodes
# ---------------------------------------------------------------------------

def test_comm_inclusive_topology_accepted():
    """n data + c comm nodes is a legal topology and marks the QPU comm-constrained."""
    network = QuantumNetwork([4, 4], comm_sizes=[1, 1],
                             qpu_topologies={0: nx.path_graph(5)})
    assert network.comm_constrained(0) is True
    assert network.comm_constrained(1) is False  # no topology at all


def test_data_only_topology_not_comm_constrained():
    """A data-only topology leaves comm qubits all-to-all."""
    network = QuantumNetwork([4, 4], qpu_topologies={0: nx.path_graph(4)})
    assert network.comm_constrained(0) is False


def test_comm_node_labels():
    """Comm qubit k of QPU p is node qpu_sizes[p] + k."""
    network = QuantumNetwork([4, 6], comm_sizes=[2, 2])
    assert network.comm_node(0, 0) == 4
    assert network.comm_node(0, 1) == 5
    assert network.comm_node(1, 0) == 6


# ---------------------------------------------------------------------------
# comm_links: binding comm qubits to inter-QPU links
# ---------------------------------------------------------------------------

def test_comm_links_default_is_empty():
    network = QuantumNetwork([4, 4])
    assert network.comm_links == {}


def test_comm_links_valid_edge_accepted():
    """Binding to an existing network link is stored and queryable."""
    network = QuantumNetwork([4, 4], comm_sizes=[2, 2], comm_links={0: {0: 1}})
    assert network.comm_links == {0: {0: 1}}
    assert network.comm_qubits_for_link(0, 1) == [0]


def test_comm_qubits_for_link_unbound_returns_all():
    """With no bindings for a neighbour, every comm qubit is eligible."""
    network = QuantumNetwork([4, 4], comm_sizes=[2, 2])
    assert network.comm_qubits_for_link(0, 1) == [0, 1]


def test_comm_qubits_for_link_partial_binding():
    """Bindings to other neighbours do not restrict an unbound link."""
    network = QuantumNetwork.create([4, 4, 4], 'linear', comm_sizes=[2, 2, 2],
                                    comm_links={1: {0: 0}})
    assert network.comm_qubits_for_link(1, 0) == [0]      # bound
    assert network.comm_qubits_for_link(1, 2) == [0, 1]   # unbound -> all


def test_comm_links_rejects_non_edge():
    """Binding a comm qubit to a QPU with no direct link raises."""
    with pytest.raises(ValueError, match="no network link"):
        QuantumNetwork.create([4, 4, 4], 'linear', comm_links={0: {0: 2}})


def test_comm_links_rejects_bad_comm_index():
    with pytest.raises(ValueError, match="out of range"):
        QuantumNetwork([4, 4], comm_sizes=[1, 1], comm_links={0: {3: 1}})


def test_comm_links_rejects_unknown_qpu():
    with pytest.raises(ValueError):
        QuantumNetwork([4, 4], comm_links={7: {0: 1}})


def test_comm_links_rejects_unknown_neighbor():
    with pytest.raises(ValueError):
        QuantumNetwork([4, 4], comm_links={0: {0: 7}})


def test_comm_links_rejects_non_dict():
    with pytest.raises(TypeError):
        QuantumNetwork([4, 4], comm_links=[(0, 0, 1)])


def test_copy_preserves_comm_links():
    network = QuantumNetwork([4, 4], comm_links={0: {0: 1}})
    assert network.copy().comm_links == {0: {0: 1}}


def test_factory_passes_comm_links():
    network = QuantumNetwork.create([4, 4], 'all_to_all', comm_links={0: {0: 1}})
    assert network.comm_links == {0: {0: 1}}
"""
Test suite for hypergraph drawing functionality.

Tests both the .draw() method with 'tikz' and 'mpl' output formats,
as well as direct calls to the underlying drawing functions.
"""

import pytest
import numpy as np
import re
from qiskit import QuantumCircuit, transpile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from disqco import QuantumNetwork, QuantumCircuitHyperGraph
from disqco.circuit_extraction import HypergraphToDistributed
from disqco.circuits.cp_fraction import cp_fraction
from disqco.drawing import DistributedCircuitQPIC
from disqco import set_initial_partition_assignment


@pytest.fixture
def test_circuit():
    """Create a small test circuit for drawing"""
    circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return circuit


@pytest.fixture
def test_network():
    """Create a small network for testing"""
    return QuantumNetwork.create([3, 3, 3], 'linear')


@pytest.fixture
def test_hypergraph(test_circuit):
    """Create a hypergraph from test circuit"""
    return QuantumCircuitHyperGraph(test_circuit)


@pytest.fixture
def test_assignment(test_hypergraph, test_network):
    """Create a test assignment"""
    return set_initial_partition_assignment(test_hypergraph, test_network)


def test_draw_method_tikz_output(test_hypergraph, test_network, test_assignment):
    """Test hypergraph.draw() with tikz output format"""
    # Test that draw method with tikz returns TikZ code
    result = test_hypergraph.draw(
        network=test_network,
        assignment=test_assignment,
        output='tikz',
        tikz_raw=True  # Get raw TikZ code instead of rendering
    )
    
    assert result is not None
    assert isinstance(result, str)
    # Check for TikZ document structure
    assert r'\documentclass' in result
    assert r'\begin{tikzpicture}' in result
    assert r'\end{tikzpicture}' in result
    assert r'\begin{document}' in result
    assert r'\end{document}' in result
    
    print("\n✓ TikZ output contains proper document structure")


def test_draw_method_tikz_default(test_hypergraph, test_network, test_assignment):
    """Test hypergraph.draw() with default tikz output"""
    # Default should be tikz
    result = test_hypergraph.draw(
        network=test_network,
        assignment=test_assignment,
        tikz_raw=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert r'\begin{tikzpicture}' in result
    
    print("\n✓ Default output format is TikZ")


def test_draw_method_mpl_output(test_hypergraph, test_network, test_assignment):
    """Test hypergraph.draw() with matplotlib output format"""
    # Test that draw method with mpl returns matplotlib figure/axes
    result = test_hypergraph.draw(
        network=test_network,
        assignment=test_assignment,
        output='mpl',
        figsize=(8, 6)
    )
    
    # matplotlib drawing returns a figure
    assert result is not None
    # The result should be a matplotlib figure or axes
    assert hasattr(result, 'figure') or isinstance(result, plt.Figure)
    
    # Close the figure to free memory
    if hasattr(result, 'figure'):
        plt.close(result.figure)
    else:
        plt.close(result)
    
    print("\n✓ Matplotlib output produces figure")


def test_draw_method_invalid_output_type(test_hypergraph, test_network, test_assignment):
    """Test that invalid output type raises ValueError"""
    with pytest.raises(ValueError, match="Unknown output type"):
        test_hypergraph.draw(
            network=test_network,
            assignment=test_assignment,
            output='invalid_type'
        )
    
    print("\n✓ Invalid output type raises ValueError")


def test_draw_method_with_show_labels_true(test_hypergraph, test_network, test_assignment):
    """Test drawing with show_labels=True"""
    result = test_hypergraph.draw(
        network=test_network,
        assignment=test_assignment,
        output='tikz',
        show_labels=True,
        tikz_raw=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    # With labels, we should see node definitions with coordinates
    assert r'\node' in result
    
    print("\n✓ Drawing with show_labels=True works")


def test_draw_method_with_show_labels_false(test_hypergraph, test_network, test_assignment):
    """Test drawing with show_labels=False"""
    result = test_hypergraph.draw(
        network=test_network,
        assignment=test_assignment,
        output='tikz',
        show_labels=False,
        tikz_raw=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert r'\begin{tikzpicture}' in result
    
    print("\n✓ Drawing with show_labels=False works")


def test_draw_method_with_default_assignment(test_hypergraph, test_network):
    """Test drawing without providing assignment (should use default)"""
    result = test_hypergraph.draw(
        network=test_network,
        output='tikz',
        tikz_raw=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert r'\begin{tikzpicture}' in result
    
    print("\n✓ Drawing with default assignment works")


def test_draw_method_mpl_with_default_assignment(test_hypergraph):
    """Test matplotlib drawing without providing assignment"""
    # Use a single QPU network with default assignment (all zeros)
    single_qpu_network = QuantumNetwork({0: test_hypergraph.num_qubits})
    
    result = test_hypergraph.draw(
        network=single_qpu_network,
        output='mpl'
    )
    
    assert result is not None
    
    # Close the figure
    if hasattr(result, 'figure'):
        plt.close(result.figure)
    else:
        plt.close(result)
    
    print("\n✓ Matplotlib drawing with default assignment works")


def test_draw_method_with_color_options(test_hypergraph, test_network, test_assignment):
    """Test drawing with color inversion options"""
    # Test with invert_colors
    result_inverted = test_hypergraph.draw(
        network=test_network,
        assignment=test_assignment,
        output='tikz',
        invert_colors=True,
        tikz_raw=True
    )
    
    assert result_inverted is not None
    assert isinstance(result_inverted, str)
    
    # Test with fill_background
    result_no_bg = test_hypergraph.draw(
        network=test_network,
        assignment=test_assignment,
        output='tikz',
        fill_background=False,
        tikz_raw=True
    )
    
    assert result_no_bg is not None
    assert isinstance(result_no_bg, str)
    
    print("\n✓ Drawing with color options works")


def test_direct_call_draw_graph_tikz(test_hypergraph, test_network, test_assignment):
    """Test direct call to draw_graph_tikz function"""
    from disqco.drawing.tikz_drawing import draw_graph_tikz
    
    result = draw_graph_tikz(
        hypergraph=test_hypergraph,
        network=test_network,
        assignment=test_assignment,
        tikz_raw=True,
        show_labels=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert r'\documentclass' in result
    assert r'\begin{tikzpicture}' in result
    
    print("\n✓ Direct call to draw_graph_tikz works")


def test_direct_call_hypergraph_to_tikz(test_hypergraph, test_network, test_assignment):
    """Test direct call to hypergraph_to_tikz function"""
    from disqco.drawing.tikz_drawing import hypergraph_to_tikz
    
    qpu_sizes = test_network.qpu_sizes
    
    result = hypergraph_to_tikz(
        H=test_hypergraph,
        assignment=test_assignment,
        qpu_info=qpu_sizes,
        show_labels=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert r'\documentclass' in result
    assert r'\begin{tikzpicture}' in result
    assert r'\end{tikzpicture}' in result
    
    print("\n✓ Direct call to hypergraph_to_tikz works")


def test_direct_call_draw_hypergraph_mpl(test_hypergraph, test_network, test_assignment):
    """Test direct call to draw_hypergraph_mpl function"""
    from disqco.drawing.mpl_drawing import draw_hypergraph_mpl
    
    qpu_sizes = test_network.qpu_sizes
    
    fig = draw_hypergraph_mpl(
        H=test_hypergraph,
        assignment=test_assignment,
        qpu_info=qpu_sizes,
        show_labels=True,
        figsize=(8, 6)
    )
    
    assert fig is not None
    # Should return a figure or axes
    assert hasattr(fig, 'figure') or isinstance(fig, plt.Figure)
    
    # Close the figure
    if hasattr(fig, 'figure'):
        plt.close(fig.figure)
    else:
        plt.close(fig)
    
    print("\n✓ Direct call to draw_hypergraph_mpl works")


def test_tikz_output_with_save_option(test_hypergraph, test_network, test_assignment, tmp_path):
    """Test saving TikZ output to file"""
    from disqco.drawing.tikz_drawing import hypergraph_to_tikz
    
    output_file = tmp_path / "test_hypergraph.tex"
    
    result = hypergraph_to_tikz(
        H=test_hypergraph,
        assignment=test_assignment,
        qpu_info=test_network.qpu_sizes,
        save=True,
        path=str(output_file)
    )
    
    assert result is not None
    assert output_file.exists()
    
    # Read the file and verify content
    with open(output_file, 'r') as f:
        content = f.read()
    
    assert r'\documentclass' in content
    assert r'\begin{tikzpicture}' in content
    
    print(f"\n✓ TikZ output saved to {output_file}")


def test_mpl_output_with_save_option(test_hypergraph, test_network, test_assignment, tmp_path):
    """Test saving matplotlib output to file"""
    from disqco.drawing.mpl_drawing import draw_hypergraph_mpl
    
    output_file = tmp_path / "test_hypergraph.png"
    
    fig = draw_hypergraph_mpl(
        H=test_hypergraph,
        assignment=test_assignment,
        qpu_info=test_network.qpu_sizes,
        save_path=str(output_file),
        dpi=100
    )
    
    assert fig is not None
    assert output_file.exists()
    
    # Close the figure
    if hasattr(fig, 'figure'):
        plt.close(fig.figure)
    else:
        plt.close(fig)
    
    print(f"\n✓ Matplotlib output saved to {output_file}")


def test_distributed_qpic_declares_all_data_wires():
    """Regression check: q/c wire declarations are exact (no missing, no phantom)."""
    num_qubits = 6
    circuit = cp_fraction(num_qubits=num_qubits, depth=2 * num_qubits, fraction=0.5, seed=52345)
    circuit = transpile(circuit, basis_gates=['u', 'cx'])

    qnet = QuantumNetwork([3, 3, 3])
    graph = QuantumCircuitHyperGraph(circuit=circuit, group_gates=True, anti_diag=True)
    assignment = set_initial_partition_assignment(graph=graph, network=qnet)

    dc = HypergraphToDistributed(graph, qnet, assignment).build()
    qpic = DistributedCircuitQPIC(dc).to_qpic_string()

    decl_block = qpic.split("\n\n", maxsplit=1)[0]
    declared_wires = set(re.findall(r"\bq\d+p\d+\b", decl_block))
    referenced_wires = set(re.findall(r"\bq\d+p\d+\b", qpic))
    declared_comm = set(re.findall(r"\bc\d+_\d+\b", decl_block))
    referenced_comm = set(re.findall(r"\bc\d+_\d+\b", qpic))

    assert referenced_wires.issubset(declared_wires), (
        f"Undeclared q-wires referenced: {sorted(referenced_wires - declared_wires)}"
    )
    assert declared_wires.issubset(referenced_wires), (
        f"Phantom q-wires declared but unused: {sorted(declared_wires - referenced_wires)}"
    )
    assert referenced_comm.issubset(declared_comm), (
        f"Undeclared comm wires referenced: {sorted(referenced_comm - declared_comm)}"
    )
    assert declared_comm.issubset(referenced_comm), (
        f"Phantom comm wires declared but unused: {sorted(declared_comm - referenced_comm)}"
    )


def test_drawing_with_different_network_topologies(test_hypergraph, test_assignment):
    """Test drawing with different network topologies"""
    # Linear network
    linear_net = QuantumNetwork.create([3, 3, 3], 'linear')
    result_linear = test_hypergraph.draw(
        network=linear_net,
        assignment=test_assignment,
        output='tikz',
        tikz_raw=True
    )
    assert result_linear is not None
    
    # Grid network (need 4 QPUs for 2x2 grid)
    grid_assignment = np.zeros((test_hypergraph.depth, test_hypergraph.num_qubits), dtype=int)
    for q in range(test_hypergraph.num_qubits):
        grid_assignment[:, q] = q % 4
    
    grid_net = QuantumNetwork.create([2, 2, 2, 2], 'grid')
    result_grid = test_hypergraph.draw(
        network=grid_net,
        assignment=grid_assignment,
        output='tikz',
        tikz_raw=True
    )
    assert result_grid is not None
    
    # All-to-all network
    alltoall_net = QuantumNetwork.create([3, 3, 3], 'all_to_all')
    result_alltoall = test_hypergraph.draw(
        network=alltoall_net,
        assignment=test_assignment,
        output='tikz',
        tikz_raw=True
    )
    assert result_alltoall is not None
    
    print("\n✓ Drawing works with different network topologies")


def test_drawing_single_qpu_network(test_hypergraph):
    """Test drawing with a single QPU (all-to-one partitioning)"""
    single_qpu_net = QuantumNetwork({0: test_hypergraph.num_qubits})
    
    result = test_hypergraph.draw(
        network=single_qpu_net,
        output='tikz',
        tikz_raw=True
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert r'\begin{tikzpicture}' in result
    
    print("\n✓ Drawing with single QPU network works")


def test_mpl_drawing_returns_figure_with_axes(test_hypergraph, test_network, test_assignment):
    """Test that matplotlib drawing returns a proper figure with axes"""
    from disqco.drawing.mpl_drawing import draw_hypergraph_mpl
    
    # Create a custom axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    result = draw_hypergraph_mpl(
        H=test_hypergraph,
        assignment=test_assignment,
        qpu_info=test_network.qpu_sizes,
        ax=ax
    )
    
    assert result is not None
    
    # Close the figure
    plt.close(fig)
    
    print("\n✓ Matplotlib drawing works with custom axes")


def test_tikz_output_structure(test_hypergraph, test_network, test_assignment):
    """Test that TikZ output has expected structure"""
    from disqco.drawing.tikz_drawing import hypergraph_to_tikz
    
    tikz_code = hypergraph_to_tikz(
        H=test_hypergraph,
        assignment=test_assignment,
        qpu_info=test_network.qpu_sizes
    )
    
    # Check for essential TikZ components
    assert r'\documentclass' in tikz_code
    assert r'\usepackage{tikz}' in tikz_code
    assert r'\begin{document}' in tikz_code
    assert r'\begin{tikzpicture}' in tikz_code
    assert r'\node' in tikz_code  # Should have node definitions
    assert r'\end{tikzpicture}' in tikz_code
    assert r'\end{document}' in tikz_code
    
    # Check for style definitions
    assert r'\tikzstyle' in tikz_code or r'\tikzset' in tikz_code
    
    print("\n✓ TikZ output has proper structure")


def test_drawing_with_qpu_info_dict(test_hypergraph):
    """Test drawing with qpu_info as dict instead of list"""
    qpu_info_dict = {0: 3, 1: 3, 2: 3}
    
    # Create a proper network and assignment matching qpu_info_dict
    network = QuantumNetwork(qpu_info_dict)
    assignment = set_initial_partition_assignment(test_hypergraph, network)
    
    # TikZ version
    result_tikz = test_hypergraph.draw(
        network=network,
        assignment=assignment,
        output='tikz',
        tikz_raw=True
    )
    assert result_tikz is not None
    
    # Matplotlib version
    result_mpl = test_hypergraph.draw(
        network=network,
        assignment=assignment,
        output='mpl'
    )
    assert result_mpl is not None
    
    # Close matplotlib figure
    if hasattr(result_mpl, 'figure'):
        plt.close(result_mpl.figure)
    else:
        plt.close(result_mpl)
    
    print("\n✓ Drawing works with qpu_info as dict")

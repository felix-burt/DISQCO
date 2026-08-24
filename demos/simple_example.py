from pathlib import Path

from qiskit import transpile

from disqco import (
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco import PartitionedCircuitExtractor
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti import FiducciaMattheyses
import networkx as nx

demo_dir = Path(__file__).parent

# Dense random circuit of pairwise cp gates: many qubit pairs end up
# co-located but non-adjacent on a line topology, forcing the extractor
# to insert local routing SWAPs before those gates can be applied.
circuit = cp_fraction(num_qubits=8, depth=16, fraction=0.7, seed=42)
circuit = transpile(circuit, basis_gates=["u", "cp"])

circuit.draw(output="mpl", style="bw", filename=str(demo_dir / "circuit.png"))
print(f"Saved original circuit to {demo_dir / 'circuit.png'}")

hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)

# Two QPUs of 5 data qubits each. Each QPU's internal coupling is a line
# (0-1-2-3-4): only neighbouring slots can interact directly. Comm qubits
# are nodes 5..8, a mutually adjacent clique attached to data slot 4 -- the
# QPU's "port". comm_sizes must cover the circuit's peak number of
# simultaneously open links (hard capacity on comm-constrained QPUs).
NUM_COMM = 3

def line_with_ports(n_data, n_comm):
    topo = nx.path_graph(n_data)
    comm_nodes = range(n_data, n_data + n_comm)
    for c in comm_nodes:
        topo.add_edge(n_data - 1, c)
    for a in comm_nodes:
        for b in comm_nodes:
            if a < b:
                topo.add_edge(a, b)
    return topo

network = QuantumNetwork(
    {0: 5, 1: 5},
    qpu_topologies={
        0: line_with_ports(5, NUM_COMM),
        1: line_with_ports(5, NUM_COMM),
    },
    # every comm qubit of QPU 0 serves the link to QPU 1 and vice versa
    comm_links={0: {k: 1 for k in range(NUM_COMM)},
                1: {k: 0 for k in range(NUM_COMM)}},
    comm_sizes=[NUM_COMM, NUM_COMM],
)

initial_assignment = set_initial_partition_assignment(hypergraph, network, round_robin=True)
partitioner = FiducciaMattheyses(circuit, network, initial_assignment=initial_assignment)
results = partitioner.partition(num_passes=50)

print("Best cost (e-bits):", results["best_cost"])

hypergraph_path = demo_dir / "partition_result.png"
hypergraph.draw(
    network=network,
    assignment=results["best_assignment"],
    show_labels=False,
    output="mpl",
    dpi=150,
    save_path=str(hypergraph_path),
)
print(f"Saved partitioned hypergraph to {hypergraph_path}")

extractor = PartitionedCircuitExtractor(
    hypergraph, network, partition_assignment=results["best_assignment"]
)
partitioned_circuit = extractor.extract_partitioned_circuit()

print("Number of e-bits requested:", partitioned_circuit.count_ops().get("EPR", 0))
print("Local routing SWAPs inserted:", extractor.local_swap_count)

partitioned_circuit_path = demo_dir / "partitioned_circuit.png"
partitioned_circuit.draw(
    output="mpl", style="bw", fold=50, filename=str(partitioned_circuit_path)
)
print(f"Saved partitioned circuit to {partitioned_circuit_path}")

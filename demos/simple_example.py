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

# Two QPUs of 5 qubits each. Each QPU's internal coupling is a line
# (0-1-2-3-4): only neighbouring slots can interact directly. Omitting
# qpu_topologies (or a QPU's entry) means all-to-all, the old behaviour.
network = QuantumNetwork(
    {0: 5, 1: 5},
    qpu_topologies={
        0: nx.path_graph(5),
        1: nx.path_graph(5),
    },
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

ops = partitioned_circuit.count_ops()
print("Number of e-bits requested:", ops.get("EPR", 0))
print("Local routing SWAPs inserted:", ops.get("swap", 0))

partitioned_circuit_path = demo_dir / "partitioned_circuit.png"
partitioned_circuit.draw(
    output="mpl", style="bw", fold=50, filename=str(partitioned_circuit_path)
)
print(f"Saved partitioned circuit to {partitioned_circuit_path}")

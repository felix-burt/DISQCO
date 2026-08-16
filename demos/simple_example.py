from pathlib import Path

from qiskit import QuantumCircuit, transpile

from disqco import (
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco import PartitionedCircuitExtractor
from disqco.parti import FiducciaMattheyses

demo_dir = Path(__file__).parent

circuit = QuantumCircuit(4)
circuit.h(0)
for i in range(3):
    circuit.cx(0, i + 1)
circuit = transpile(circuit, basis_gates=["u", "cp"])

circuit.draw(output="mpl", style="bw", filename=str(demo_dir / "circuit.png"))
print(f"Saved original circuit to {demo_dir / 'circuit.png'}")

hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)

network = QuantumNetwork({0: 2, 1: 2, 2: 2})

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

partitioned_circuit_path = demo_dir / "partitioned_circuit.png"
partitioned_circuit.draw(
    output="mpl", style="bw", fold=50, filename=str(partitioned_circuit_path)
)
print(f"Saved partitioned circuit to {partitioned_circuit_path}")

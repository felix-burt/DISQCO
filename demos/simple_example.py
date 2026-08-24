from pathlib import Path

from qiskit import QuantumCircuit

from disqco import (
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco import PartitionedCircuitExtractor
import networkx as nx

demo_dir = Path(__file__).parent

# A tiny, hand-built circuit: one CNOT between qubits 0 and 4. Under the
# round-robin assignment below they both land on QPU 0, at ring slots that
# are NOT adjacent -- so extraction must insert a local routing SWAP before
# the CNOT can be applied.
circuit = QuantumCircuit(5)
for i in range(5):
    circuit.h(i)
circuit.cx(0, 4)
circuit.cx(1, 4)
circuit.cx(0, 3)
circuit.cx(3, 4)

circuit.draw(output="mpl", style="bw", filename=str(demo_dir / "circuit.png"))
print(f"Saved original circuit to {demo_dir / 'circuit.png'}")

hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)

# Two QPUs of 5 data qubits each. Each QPU's internal coupling is a RING
# (0-1-2-3-4-0): only neighbouring slots can interact directly, and slot 0
# is two hops from slot 2 either way round. Comm qubits are nodes 5..7, a
# mutually adjacent clique attached to data slot 4 -- the QPU's "port".
# comm_sizes must cover the circuit's peak number of simultaneously open
# links (hard capacity on comm-constrained QPUs).
NUM_COMM = 2

def ring_with_ports(n_data, n_comm):
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
    {0: 3, 1: 2},
    qpu_topologies={
        0: ring_with_ports(3, NUM_COMM),
        1: ring_with_ports(2, NUM_COMM),
    },
    # every comm qubit of QPU 0 serves the link to QPU 1 and vice versa
    comm_links={0: {k: 1 for k in range(NUM_COMM)},
                1: {k: 0 for k in range(NUM_COMM)}},
    comm_sizes=[NUM_COMM, NUM_COMM],
)

# Round-robin puts qubits 0, 2, 4 on QPU 0 in that order, filling ring
# slots 0, 1, 2 respectively -- so the CNOT(0, 4) above acts on slots 0
# and 2, which are not adjacent on the ring. No optimizer needed here:
# the circuit is trivial (a single gate, already co-located), so we
# extract straight from the initial assignment.
assignment = set_initial_partition_assignment(hypergraph, network, round_robin=True)

hypergraph_path = demo_dir / "partition_result.png"
hypergraph.draw(
    network=network,
    assignment=assignment,
    show_labels=False,
    output="mpl",
    dpi=150,
    save_path=str(hypergraph_path),
)

extractor = PartitionedCircuitExtractor(
    hypergraph, network, partition_assignment=assignment
)
partitioned_circuit = extractor.extract_partitioned_circuit()

print("Number of e-bits requested:", partitioned_circuit.count_ops().get("EPR", 0))
print("Local routing SWAPs inserted:", extractor.local_swap_count)

partitioned_circuit_path = demo_dir / "partitioned_circuit.png"
partitioned_circuit.draw(
    output="mpl", style="bw", fold=50, filename=str(partitioned_circuit_path)
)

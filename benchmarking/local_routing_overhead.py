"""
Estimate local-routing SWAP overhead across circuits and intra-QPU topologies.

For each (circuit, topology) configuration this script partitions with FM,
extracts the distributed circuit, and reports:

  - ebits:     entanglement cost of the partition (results['best_cost'])
  - swaps:     local routing SWAPs inserted during extraction
  - swaps/ebit ratio: the go/no-go number for further routing work
"""

import random

import networkx as nx
import numpy as np
from qiskit import transpile

from disqco import (
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    PartitionedCircuitExtractor,
    set_initial_partition_assignment,
)
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti import FiducciaMattheyses

SEED = 42
NUM_QPUS = 2
FM_PASSES = 10

# (num_qubits, depth, cp fraction)
CIRCUITS = [
    (8, 16, 0.5),
    (12, 24, 0.5),
    (16, 32, 0.5),
]

# Comm qubits per QPU for the "+ports" configs (hard capacity: must cover the
# circuit's peak simultaneous link demand or the row reports an ERROR).
COMM_SIZE = 8


def with_comm_ports(data_topo, n_data, n_comm, attach_at):
    """Attach a clique of n_comm comm nodes to data slot attach_at."""
    topo = data_topo.copy()
    comm_nodes = list(range(n_data, n_data + n_comm))
    for c in comm_nodes:
        topo.add_edge(attach_at, c)
    for i, a in enumerate(comm_nodes):
        for b in comm_nodes[i + 1:]:
            topo.add_edge(a, b)
    return topo


def line_with_ports(n):
    return with_comm_ports(nx.path_graph(n), n, COMM_SIZE, n - 1)


def ring_with_ports(n):
    return with_comm_ports(nx.cycle_graph(n), n, COMM_SIZE, 0)


# name -> (data-topology factory or None, comm qubits included in topology?)
TOPOLOGIES = {
    "all_to_all": (None, False),
    "line": (nx.path_graph, False),
    "ring": (nx.cycle_graph, False),
    "line+ports": (line_with_ports, True),
    "ring+ports": (ring_with_ports, True),
}


def make_reproducible(seed):
    """FM_pass calls random.seed() with no args, reseeding from OS entropy.
    Turn the no-arg call into a no-op, then seed once."""
    original_seed = random.seed

    def selective_seed(*args, **kwargs):
        if not args and not kwargs:
            return
        original_seed(*args, **kwargs)

    random.seed = selective_seed
    original_seed(seed)
    np.random.seed(seed)


def run_config(num_qubits, depth, fraction, topo_name, topo_spec):
    topo_factory, has_ports = topo_spec
    qpu_size = num_qubits // NUM_QPUS + 1
    qpu_sizes = [qpu_size] * NUM_QPUS

    if topo_factory is None:
        qpu_topologies = None
    else:
        qpu_topologies = {p: topo_factory(qpu_size) for p in range(NUM_QPUS)}

    circuit = cp_fraction(num_qubits=num_qubits, depth=depth,
                          fraction=fraction, seed=SEED)
    circuit = transpile(circuit, basis_gates=["u", "cp"])
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
    comm_sizes = [COMM_SIZE] * NUM_QPUS if has_ports else None
    network = QuantumNetwork(qpu_sizes, comm_sizes=comm_sizes,
                             qpu_topologies=qpu_topologies)

    initial = set_initial_partition_assignment(hypergraph, network,
                                               round_robin=True)
    partitioner = FiducciaMattheyses(circuit, network,
                                     initial_assignment=initial)
    results = partitioner.partition(num_passes=FM_PASSES)

    extractor = PartitionedCircuitExtractor(
        hypergraph, network, partition_assignment=results["best_assignment"])
    extractor.extract_partitioned_circuit()

    return results["best_cost"], extractor.local_swap_count


def main():
    make_reproducible(SEED)

    header = (f"{'circuit':>16} {'topology':>11} {'ebits':>6} "
              f"{'swaps':>6} {'swaps/ebit':>11}")
    print(header)
    print("-" * len(header))

    for num_qubits, depth, fraction in CIRCUITS:
        for topo_name, topo_factory in TOPOLOGIES.items():
            label = f"cp({num_qubits},{depth},{fraction})"
            try:
                ebits, swaps = run_config(num_qubits, depth, fraction,
                                          topo_name, topo_factory)
                ratio = f"{swaps / ebits:.2f}" if ebits else "n/a"
                print(f"{label:>16} {topo_name:>11} {ebits:>6} "
                      f"{swaps:>6} {ratio:>11}")
            except Exception as exc:
                print(f"{label:>16} {topo_name:>11} "
                      f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

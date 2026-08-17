"""
Estimate local-routing SWAP overhead across circuits and intra-QPU topologies.

For each (circuit, topology) configuration this script partitions with FM,
extracts the distributed circuit, and reports:

  - ebits:     entanglement cost of the partition (results['best_cost'])
  - swaps:     local routing SWAPs inserted during extraction
  - swaps/ebit ratio: the go/no-go number for further routing work

SWAPs are counted by wrapping DataQubitManager.swap_physical_slots (its only
caller is the local router; the final transpile decomposes swap gates, so
counting them in the output circuit would always give zero).

FM is made reproducible with the no-arg-random.seed monkeypatch: FM_pass
reseeds from OS entropy on every pass unless no-arg seed() is a no-op.

Usage:  .venv\\Scripts\\python.exe benchmarking\\local_routing_overhead.py
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
from disqco.circuit_extraction.DQC_qubit_manager import DataQubitManager
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

TOPOLOGIES = {
    "all_to_all": None,
    "line": nx.path_graph,
    "ring": nx.cycle_graph,
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


class SwapCounter:
    """Counts calls to DataQubitManager.swap_physical_slots (one per routing hop)."""

    def __init__(self):
        self.count = 0
        self._original = DataQubitManager.swap_physical_slots

    def __enter__(self):
        counter = self

        def counted(manager, p, qubit_a, qubit_b):
            counter.count += 1
            return counter._original(manager, p, qubit_a, qubit_b)

        DataQubitManager.swap_physical_slots = counted
        return self

    def __exit__(self, *exc):
        DataQubitManager.swap_physical_slots = self._original


def run_config(num_qubits, depth, fraction, topo_name, topo_factory):
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
    network = QuantumNetwork(qpu_sizes, qpu_topologies=qpu_topologies)

    initial = set_initial_partition_assignment(hypergraph, network,
                                               round_robin=True)
    partitioner = FiducciaMattheyses(circuit, network,
                                     initial_assignment=initial)
    results = partitioner.partition(num_passes=FM_PASSES)

    extractor = PartitionedCircuitExtractor(
        hypergraph, network, partition_assignment=results["best_assignment"])
    with SwapCounter() as counter:
        extractor.extract_partitioned_circuit()

    return results["best_cost"], counter.count


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

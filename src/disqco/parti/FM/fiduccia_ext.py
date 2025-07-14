# src/disqco/parti/FM/fiduccia_ext.py
from disqco.parti.partitioner import QuantumCircuitPartitioner
from disqco.parti.FM.FM_methods_ext import *
from disqco.parti.FM.FM_main_ext import run_FM          # single-pass driver
from disqco.graphs.GCP_hypergraph_extended import HyperGraph
from disqco.graphs.quantum_network import QuantumNetwork
from qiskit import QuantumCircuit
import numpy as np


class FiducciaMattheysesExt(QuantumCircuitPartitioner):
    """
    Fiduccia-Mattheyses partitioner that runs on the *extended* hypergraph backend.
    Its public API mirrors `FiducciaMattheyses` in fiduccia.py.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        network: QuantumNetwork,
        initial_qubit_assignment: np.ndarray | None = None,
        initial_gate_assignment: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(circuit, network, initial_qubit_assignment)

        self.qpu_sizes = self.network.qpu_sizes if not isinstance(
            network.qpu_sizes, dict) else list(network.qpu_sizes.values())

        # Build *extended* hypergraph
        group_gates = kwargs.get("group_gates", True)
        self.hypergraph: HyperGraph = HyperGraph()
        self.hypergraph.from_circuit(circuit)

        self.num_partitions = len(self.qpu_sizes)
        self.num_qubits, self.depth = self.hypergraph.num_qubits, self.hypergraph.depth

        # If user did not supply seed assignments build defaults
        if initial_qubit_assignment is None:
            self.ubit_assignment = set_initial_qubit_assignment(self.qpu_sizes, self.num_qubits, self.depth)
        else:
            self.qubit_assignment = initial_qubit_assignment.copy()

        if initial_gate_assignment is None:
            self.gate_assignment = set_initial_gate_assignment(self.hypergraph, self.qubit_assignment, randomise=False)
        else:
            self.gate_assignment = initial_gate_assignment.copy()

        # Cache costs once
        self.initial_cost = calculate_full_cost(
            self.hypergraph,
            self.qubit_assignment,
            self.gate_assignment,
            self.num_partitions,
        )

    # ----------  public entry points  ---------------------------------
    def partition(self, *, passes: int = 10, max_gain: int | None = None, **kwargs):
        """Run the extended FM optimiser and return a dict like the original."""
        if max_gain is None:
            max_gain = 4  # default fallback

        best_cost, best_assign, trace = run_FM(
            self.hypergraph,
            self.qubit_assignment,
            self.gate_assignment,
            self.qpu_sizes,
            passes=passes,
            limit=kwargs.get("limit", len(self.hypergraph.nodes())),
            max_gain=max_gain,
            stochastic=kwargs.get("stochastic", False),
            log=kwargs.get("log", False),
        )

        return {
            "best_cost": best_cost,
            "best_qubit_assignment": best_assign[0],
            "best_gate_assignment": best_assign[1],
            "cost_trace": trace,
        }

    # Optional multilevel wrapper re-using FM_main_ext.partition_multilevel
    def multilevel_partition(self, **kwargs):
        return partition_multilevel(
            kwargs.get("graph_list"),
            kwargs.get("mapping_list"),
            self.qubit_assignment,
            self.gate_assignment,
            self.qpu_sizes,
            self.num_partitions,
        )
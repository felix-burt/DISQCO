from qiskit import QuantumCircuit
from disqco.graphs.quantum_network import QuantumNetwork
import numpy as np
from disqco.parti.FM.FM_methods import set_initial_partitions

# Circuit partitioner base class

class QuantumCircuitPartitioner:
    """
    Base class for quantum circuit partitioners.
    """
    def __init__(self, circuit : QuantumCircuit, 
                 network: QuantumNetwork, 
                 initial_assignment: np.ndarray
                 ) -> None:
        """
        Initialize the CircuitPartitioner.

        Args:
            circuit: The quantum circuit to be partitioned.
            partitioner: The method to use for partitioning.
        """
        self.circuit = circuit
        self.network = network
        self.initial_assignment = initial_assignment

    def partition(self, **kwargs) -> list:
        """
        Partition the quantum circuit using the specified strategy.

        Returns:
            A list of partitions.
        """

        coarsener = kwargs.get('coarsener')

        
        partitioner = kwargs.get('partitioner')
        log = kwargs.get('log', False)

        results = partitioner(log=log)

        return results
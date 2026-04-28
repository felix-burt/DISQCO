"""
FGP (Fiduccia-Greedy-Push) Partitioner
This module provides a partitioner class that wraps the FGP-OEE algorithm by Roee.
The FGP partitioner is designed for all-to-all connected networks only and does not
support multilevel partitioning with coarsening.
"""

from disqco.parti.partitioner import QuantumCircuitPartitioner
from bosonic_converters import CircuitConverters
from bosonic_model import Circuit
from disqco.graphs.quantum_network import QuantumNetwork
import numpy as np
from disqco.parti.fgp.fgp_roee import main_algorithm, set_initial_partition_fgp


class FGPPartitioner(QuantumCircuitPartitioner):
    """
    FGP partitioner that wraps the main_algorithm from fgp_roee.
    
    This partitioner only works with all-to-all connected networks (homogeneous).
    It does not support multilevel partitioning with coarsening.
    """
    
    def __init__(self, 
                 circuit: Circuit, 
                 network: QuantumNetwork, 
                 initial_assignment: np.ndarray = None, 
                 **kwargs) -> None:
        """
        Initialize the FGP partitioner.

        Args:
            circuit: The quantum circuit to be partitioned.
            network: The quantum network topology (must be all-to-all connected).
            initial_assignment: Initial partition assignment as a list (optional).
            **kwargs: Additional keyword arguments.
                - remove_singles (bool): Whether to remove single qubit gates. Default True.
                - choose_initial (bool): Whether to use provided initial partition. Default False.
                - initial_search_size (int): Size for initial search. Default 10000.

        Raises:
            ValueError: If the network is not all-to-all connected (hetero=True).
        """
        super().__init__(circuit, network, initial_assignment)
        
        # Check that network is all-to-all (homogeneous)
        if self.network.hetero:
            raise ValueError(
                "FGP partitioner only supports all-to-all connected networks. "
                "The provided network has hetero=True, indicating non-uniform connectivity."
            )
        
        self.num_qubits = circuit.qubits()
        # FGP's underlying algorithm depends on Qiskit-specific circuit attributes (qregs,
        # num_qubits, depth) and Qiskit DAG layering. Keep a Qiskit copy for the inner pass.
        self._qiskit_circuit = CircuitConverters.to_qiskit(circuit)
        self.num_qpus = len(network.qpu_sizes)
        
        # Extract qpu_info as a list (required by fgp_roee main_algorithm)
        self.qpu_info = [network.qpu_sizes[i] for i in range(self.num_qpus)]
        
        # Store FGP-specific parameters
        self.remove_singles = kwargs.get('remove_singles', True)
        self.choose_initial = kwargs.get('choose_initial', False)
        self.initial_search_size = kwargs.get('initial_search_size', 10000)
        
        # Initialize default assignment if not provided
        if self.initial_assignment is None:
            # Create a simple round-robin assignment
            self.initial_assignment = self._create_default_assignment()
    
    def _create_default_assignment(self):
        """
        Create a default initial assignment using the fgp_roee function.
        
        Returns:
            list: Initial assignment list where each qubit is assigned to a QPU.
        """
        return set_initial_partition_fgp(self.qpu_info, self.num_qpus)
    
    def partition(self, **kwargs) -> dict:
        """
        Partition the quantum circuit using the FGP-OEE algorithm.
        
        This method wraps the main_algorithm from fgp_roee and returns results
        in the standard partitioner format.

        Args:
            **kwargs: Additional arguments (currently unused, included for compatibility).

        Returns:
            dict: Results dictionary with keys:
                - 'best_cost': The entanglement cost of the partition
                - 'best_assignment': The final partition assignment (2D array)
                - 'mapping': The qubit mapping at each layer
        """
        # Get the assignment to use
        assignment = kwargs.get('assignment', self.initial_assignment)
        
        # Call the main FGP algorithm
        full_partition, cost, full_mapping = main_algorithm(
            circuit=self._qiskit_circuit,
            qpu_info=self.qpu_info,
            initial_partition=assignment,
            remove_singles=self.remove_singles,
            choose_initial=self.choose_initial,
            intial_search_size=self.initial_search_size
        )
        
        # Return results in standard format
        results = {
            'best_cost': cost,
            'best_assignment': full_partition,
            'mapping': full_mapping
        }
        
        return results
    
    def multilevel_partition(self, coarsener=None, **kwargs) -> dict:
        """
        Multilevel partitioning is not supported for FGP.
        
        Raises:
            NotImplementedError: Always, as FGP does not support coarsening.
        """
        raise NotImplementedError(
            "FGP partitioner does not support multilevel partitioning with coarsening. "
            "Use the partition() method instead."
        )
    
    def net_coarsened_partition(self, **kwargs) -> dict:
        """
        Network coarsened partitioning is not supported for FGP.
        
        Raises:
            NotImplementedError: Always, as FGP requires all-to-all connectivity.
        """
        raise NotImplementedError(
            "FGP partitioner does not support network coarsened partitioning. "
            "FGP requires all-to-all connected networks."
        )

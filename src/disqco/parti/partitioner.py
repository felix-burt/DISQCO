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

        partitioner = kwargs.get('partitioner')
        results = partitioner(**kwargs)

        return results
    
    def multilevel_partition(self, coarsener, **kwargs) -> list:
        """
        Perform multilevel partitioning of the quantum circuit.

        Args:
            kwargs: Additional arguments for the partitioning process.

        Returns:
            A list of partitions.
        """
        level_limit = kwargs.get('level_limit', 1000)
        graph = kwargs.get('graph', self.hypergraph)

        graph_list, mapping_list = coarsener(hypergraph=graph)

        if self.initial_assignment is not None:
            assignment = self.initial_assignment.copy()
        else:
            assignment = None
        
        list_of_assignments = []
        list_of_costs = []
        best_cost = float('inf')
        graph_list = graph_list[::-1]
        mapping_list = mapping_list[::-1]
        graph_list = graph_list[:level_limit]
        mapping_list = mapping_list[:level_limit]

        pass_list = [10] * level_limit

        

        for i, graph in enumerate(graph_list):

            self.passes = pass_list[i]
            kwargs['graph'] = graph
            kwargs['active_nodes'] = graph.nodes
            kwargs['assignment'] = assignment
            kwargs['mapping'] = mapping_list[i]
            kwargs['limit'] = self.num_qubits
            kwargs['passes'] = pass_list[i]
            results = self.partition(**kwargs)

            best_cost_level = results['best_cost']
            best_assignment_level = results['best_assignment']

            if best_cost_level < best_cost:
            # Keep track of the result
                best_cost = best_cost_level
                assignment = best_assignment_level

            # if log:
            print(f'Best cost at level {i}: {best_cost}')

            refined_assignment = self.refine_assignment(i, len(graph_list), assignment, mapping_list)
            assignment = refined_assignment
            kwargs['seed_partitions'] = [assignment]


            list_of_assignments.append(assignment)
            list_of_costs.append(best_cost)
        
        final_cost = min(list_of_costs)
        final_assignment = list_of_assignments[np.argmin(list_of_costs)]

        results = {'best_cost' : final_cost, 'best_assignment' : final_assignment}

        return results

    def refine_assignment(self, level, num_levels, assignment, mapping_list):
        new_assignment = assignment
        if level < num_levels -1:
            mapping = mapping_list[level]
            for super_node_t in mapping:
                for t in mapping[super_node_t]:
                    new_assignment[t] = assignment[super_node_t]
        return new_assignment
        
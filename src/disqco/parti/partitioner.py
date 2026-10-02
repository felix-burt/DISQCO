from qiskit import QuantumCircuit
from disqco import QuantumNetwork
import numpy as np

# Circuit partitioner base class

class QuantumCircuitPartitioner:
    """
    Base class for quantum circuit partitioners.
    """
    def __init__(self, 
                 circuit : QuantumCircuit | None = None, 
                 network: QuantumNetwork | None = None, 
                 initial_assignment: np.ndarray | None = None
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
    
    @classmethod
    def create(cls, partitioner_type: str, circuit: QuantumCircuit, 
               network: QuantumNetwork, **kwargs):
        """
        Factory method to create a partitioner instance based on a string type.
        
        Args:
            partitioner_type: String specifying the partitioner type.
                Supported values:
                - 'fm', 'fiduccia', 'fiducciamattheyses': FiducciaMattheyses
                - 'genetic', 'ga': GeneticPartitioner
                - 'fgp': FGPPartitioner
            circuit: The quantum circuit to be partitioned.
            network: The quantum network topology.
            **kwargs: Additional arguments passed to the partitioner constructor.
        
        Returns:
            An instance of the specified partitioner class.
        
        Raises:
            ValueError: If partitioner_type is not recognized.
        
        Example:
            >>> from disqco.parti import QuantumCircuitPartitioner
            >>> partitioner = QuantumCircuitPartitioner.create('fm', circuit, network)
        """
        partitioner_type_lower = partitioner_type.lower()
        
        if partitioner_type_lower in ['fm', 'fiduccia', 'fiducciamattheyses']:
            from disqco.parti import FiducciaMattheyses
            return FiducciaMattheyses(circuit, network, **kwargs)
        
        elif partitioner_type_lower in ['genetic', 'ga']:
            from disqco.parti.genetic.genetic_algorithm_beta import GeneticPartitioner
            return GeneticPartitioner(circuit, network, **kwargs)
        
        elif partitioner_type_lower in ['fgp']:
            from disqco.parti.fgp.fgp_partitioner import FGPPartitioner
            return FGPPartitioner(circuit, network, **kwargs)
        
        else:
            raise ValueError(
                f"Unknown partitioner type: '{partitioner_type}'. "
                f"Supported types: 'fm', 'fiduccia', 'fiducciamattheyses', "
                f"'genetic', 'ga', 'fgp'"
            )
    

    def partition(self, **kwargs) -> dict:
        """
        Partition the quantum circuit using the specified strategy.

        Returns:
            A list of partitions.
        """
        partitioner = kwargs.get('partitioner')
        hypergraph_coarsener = kwargs.pop('hypergraph_coarsener', None)
        if hypergraph_coarsener is not None:
            results = self.multilevel_partition(hypergraph_coarsener, **kwargs)
        else:
            results = partitioner(**kwargs)

        return results
    
    def multilevel_partition(self, coarsener, **kwargs) -> dict:
        """
        Perform multilevel partitioning of the quantum circuit.

        Args:
            kwargs: Additional arguments for the partitioning process.

        Returns:
            A list of partitions.
        """
        # Extract coarsener-related kwargs
        coarsener_kwargs = {}
        for key in list(kwargs.keys()):
            if key in ['num_levels', 'num_blocks', 'block_size', 'recursion_factor']:
                coarsener_kwargs[key] = kwargs.pop(key)
        level_limit = kwargs.pop('level_limit', 100)

        graph = kwargs.get('graph', self.hypergraph)
        graph_list, mapping_list = coarsener(hypergraph=graph, **coarsener_kwargs)

        full_graph = graph_list[0]

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
        passes_per_level = kwargs.get('passes_per_level', 10)
        pass_list = [passes_per_level] * (level_limit)

        

        for i, graph in enumerate(graph_list):

            self.passes = int(pass_list[i])
            kwargs['graph'] = graph
            kwargs['active_nodes'] = graph.nodes
            kwargs['assignment'] = assignment
            kwargs['mapping'] = mapping_list[i]
            kwargs['limit'] = kwargs.get('limit', 0.75 * self.hypergraph.num_qubits)
            kwargs['passes'] = pass_list[i]
            results = self.partition(**kwargs)

            best_cost_level = results['best_cost']
            best_assignment_level = results['best_assignment']

            # if best_cost_level < best_cost:
            # # Keep track of the result
            best_cost = best_cost_level
            assignment = best_assignment_level
            refined_assignment = self.refine_assignment(i, 
                                                        len(graph_list), 
                                                        assignment, 
                                                        mapping_list, 
                                                        sparse=kwargs.get('sparse', False), 
                                                        full_subgraph=full_graph, 
                                                        next_graph=graph_list[i+1] if i+1 < len(graph_list) else None, 
                                                        qpu_sizes=self.qpu_sizes)
            
            assignment = refined_assignment
            kwargs['seed_partitions'] = [assignment]


            list_of_assignments.append(assignment)
            list_of_costs.append(best_cost)
        
        final_cost = list_of_costs[-1]
        final_assignment = list_of_assignments[-1]

        results = {'best_cost' : final_cost, 'best_assignment' : final_assignment, 'assignment_list': list_of_assignments, 'cost_list': list_of_costs}

        return results

    def net_coarsened_partition(self, **kwargs):
        """
        Perform network coarsened partitioning of the quantum circuit.

        Args:
            kwargs: Additional arguments for the partitioning process.

        Returns:
            A list of partitions.
        """
        from disqco.parti.FM.net_coarsened_partitioning import run_full_net_coarsened_FM

        results = run_full_net_coarsened_FM(
            hypergraph=self.hypergraph,
            network=self.network,
            num_qubits=self.num_qubits,
            **kwargs
        )
        
        return results

    def refine_assignment(self, level, num_levels, assignment, mapping_list, sparse=False, full_subgraph=None, next_graph=None, qpu_sizes=None):
        new_assignment = assignment

        if sparse:
            return self.refine_assignment_sparse(level, num_levels, assignment, mapping_list, full_subgraph, next_graph, qpu_sizes)
        if level <= num_levels - 1:
            mapping = mapping_list[level]
            for super_node_t in mapping:
                for t in mapping[super_node_t]:
                    new_assignment[t] = assignment[super_node_t]

        return new_assignment

    def refine_assignment_sparse(self, level, num_levels, assignment, mapping_list, subgraph, next_graph, qpu_sizes):
        new_assignment = assignment
        unassigned_nodes = {}
        # Print all inputs
        if level < num_levels - 1:
            for super_node_t in mapping_list[level]:
                for t in mapping_list[level][super_node_t]:
                    for q in range(len(assignment[0])):
                        if (q,t) in subgraph.nodes and (q, super_node_t) in subgraph.nodes:
                            new_assignment[t][q] = assignment[super_node_t][q]
                        elif (q, t) in subgraph.nodes and (q, super_node_t) not in subgraph.nodes:
                            target_partition = assignment[super_node_t][q]
                            unassigned_nodes[(q, t)] = target_partition



        partition_counts = [{qpu: 0 for qpu in qpu_sizes.keys()} for t in range(assignment.shape[0])]
        for node in set(subgraph.nodes) - unassigned_nodes.keys():
            if isinstance(node, tuple) and len(node) == 2:
                q, t = node
                node_partition = assignment[t][q]
                partition_counts[t][node_partition] += 1

        for node in unassigned_nodes.keys():
            q, t = node
            # Find the first partition with available space
            for partition, size in qpu_sizes.items():
                if partition_counts[t][partition] < size:
                    new_assignment[t][q] = partition
                    partition_counts[t][partition] += 1
                    break
      
        return new_assignment
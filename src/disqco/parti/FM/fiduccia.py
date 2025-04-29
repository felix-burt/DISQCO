from disqco.parti.partitioner import QuantumCircuitPartitioner
from qiskit import QuantumCircuit
from disqco.graphs.quantum_network import QuantumNetwork
import numpy as np
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from disqco.parti.FM.FM_methods import *
import networkx as nx
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener

class FiducciaMattheyses(QuantumCircuitPartitioner):
    """
    Fiduccia-Mattheyses partitioning algorithm for quantum circuits.
    This class implements the Fiduccia-Mattheyses algorithm for partitioning
    quantum circuits into smaller sub-circuits.
    """
    def __init__(self, 
                 circuit : QuantumCircuit, 
                 network : QuantumNetwork, 
                 initial_assignment : np.ndarray = None, 
                 **kwargs) -> None:
        """
        Initialize the FiducciaMattheyses class.

        Args:
            circuit: The quantum circuit to be partitioned.
            partitioner: The method to use for partitioning.
        """
        super().__init__(circuit, 
                         network = network,
                         initial_assignment = initial_assignment)
        

        self.qpu_sizes = self.network.qpu_sizes
        group_gates = kwargs.get('group_gates', True)

        self.hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=group_gates)
        self.num_partitions = len(self.qpu_sizes)

        self.num_qubits = self.hypergraph.num_qubits
        self.depth = self.hypergraph.depth

        self.costs = kwargs.pop('costs', self.network.get_costs())
        self.mapping = None

        if self.initial_assignment is None:
            self.initial_assignment = set_initial_partitions(network, self.num_qubits, self.depth)

        if isinstance(self.qpu_sizes, dict):
            # If qpu_sizes is a dictionary, we need to convert it to a list of lists
            self.qpu_sizes = list(self.qpu_sizes.values())

    def FM_pass(self, hypergraph, assignment, **kwargs):
        
        active_nodes = kwargs.get('active_nodes', hypergraph.nodes)
        limit = kwargs.get('limit', len(hypergraph.nodes) * 0.125)
        # print("Limit:", limit)
        spaces = find_spaces(self.num_qubits, self.depth, assignment, self.qpu_sizes)
        map_counts_and_configs(hypergraph, assignment, self.num_partitions, costs=self.costs, **kwargs)

        lock_dict = {node: False for node in active_nodes}

        array = find_all_gains(hypergraph,
                               active_nodes,
                               assignment,
                               num_partitions=self.num_partitions,
                               costs = self.costs,
                                network=self.network,
                               **kwargs
                               )
        
        buckets = fill_buckets(array, self.max_gain)
        
        gain_list = []
        gain_list.append(0)
        assignment_list = []
        assignment_list.append(assignment)
        cumulative_gain = 0
        action = 0
        iter = 0

        while iter < limit:
            action, gain = find_action(buckets, lock_dict, spaces, self.max_gain)
            if action is None:
                break
            cumulative_gain += gain
            gain_list.append(cumulative_gain)
            node = (action[1], action[0])
            destination = action[2]
            source = assignment[node[1]][node[0]]
            assignment_new, array, buckets = take_action_and_update(hypergraph,
                                                                    node,
                                                                    destination,
                                                                    array,
                                                                    buckets,
                                                                    self.num_partitions,
                                                                    lock_dict,
                                                                    assignment,
                                                                    self.costs,
                                                                    network=self.network,
                                                                    **kwargs
                                                                    )
            update_spaces(node, source, destination, spaces)
            lock_dict = lock_node(node, lock_dict)

            assignment = assignment_new
            assignment_list.append(assignment)
            iter += 1
        
        return assignment_list, gain_list
    
    def run_FM(self, **kwargs):

        passes = kwargs.pop('passes', 100)
        stochastic = kwargs.pop('stochastic', True)

        hypergraph = kwargs.pop('graph')
        assignment = kwargs.pop('assignment')

        mapping = kwargs.pop('mapping', {t : set([t]) for t in range(hypergraph.depth)})

        dummy_nodes = kwargs.get('dummy_nodes', set())
        node_map = kwargs.get('node_map', None)
        assignment_map = kwargs.get('assignment_map', None)

        log = kwargs.get('log', False)


        initial_cost = calculate_full_cost(hypergraph, 
                                           assignment, 
                                           self.num_partitions, 
                                           self.costs,
                                           network=self.network,
                                           dummy_nodes=dummy_nodes,
                                           node_map=node_map,
                                           assignment_map=assignment_map,
                                           hetero=self.network.hetero,)
        
        if log:
            print("Initial cost:", initial_cost)
        cost = initial_cost
        cost_list = []
        best_assignments = []

        cost_list.append(cost)
        best_assignments.append(assignment)
        # print("Starting FM passes...")
        self.max_gain = self.find_max_gain(mapping)
        for n in range(passes):
            assignment_list, gain_list = self.FM_pass(hypergraph, assignment, **kwargs)

            # Decide how to pick new assignment depending on stochastic or not
            if stochastic:
                if n % 2 == 0:
                    # Exploratory approach
                    assignment = assignment_list[-1]
                    cost += gain_list[-1]
                else:
                    # Exploitative approach
                    idx_best = np.argmin(gain_list)
                    assignment = assignment_list[idx_best]
                    cost += min(gain_list)
            else:
                # purely pick the best
                idx_best = np.argmin(gain_list)
                assignment = assignment_list[idx_best]
                cost += min(gain_list)

            # print(f"Running cost after pass {n}:", cost)
            cost_list.append(cost)
            best_assignments.append(assignment)

        # 5) Identify best assignment across all passes
        idx_global_best = np.argmin(cost_list)
        final_assignment = best_assignments[idx_global_best]
        final_cost = cost_list[idx_global_best]

        if log:
            print("All passes complete.")
            print("Final cost:", final_cost)

        results = {'best_cost' : final_cost, 'best_assignment' : final_assignment, 'cost_list' : cost_list}
        
        return results
    
    def partition(self, **kwargs):

        kwargs['graph'] = kwargs.get('graph', self.hypergraph)
        kwargs['assignment'] = kwargs.get('assignment', self.initial_assignment)
        kwargs['mapping'] = kwargs.get('mapping', None)
        kwargs['log'] = kwargs.get('log', False)
        kwargs['partitioner'] = kwargs.get('partitioner', self.run_FM)
        kwargs['hetero'] = self.network.hetero
    
        return super().partition(**kwargs)

    def multilevel_partition(self, **kwargs):
        kwargs['graph'] = self.hypergraph
        coarsener = kwargs.pop('coarsener', None)

        if coarsener is None:
            coarsener_class = HypergraphCoarsener()
            coarsener = coarsener_class.coarsen_recursive_batches_mapped

        return super().multilevel_partition(coarsener=coarsener, **kwargs)

    def find_max_gain(self, mapping=None):
        if mapping is None:
            base = 4
        else:
            largest_node = 1
            for s_node in mapping:
                length = len(mapping[s_node])
                if length > largest_node:
                    largest_node = length
            base = 2 * largest_node + 2
        diameter = nx.diameter(self.network.qpu_graph)
        return base * diameter

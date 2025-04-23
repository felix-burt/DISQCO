from disqco.parti.partitioner import QuantumCircuitPartitioner
from qiskit import QuantumCircuit
from disqco.graphs.quantum_network import QuantumNetwork
import numpy as np
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from disqco.parti.FM.FM_methods import *

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

        self.active_nodes = kwargs.get('active_nodes', self.hypergraph.nodes)
        self.costs = kwargs.get('costs', network.get_costs())
        self.limit = kwargs.get('limit', len(self.hypergraph.nodes) * 0.125)
        self.passes = kwargs.get('passes', 100)
        self.max_gain = kwargs.get('max_gain', 4)
        self.stochastic = kwargs.get('stochastic', True)
        self.dummy_nodes = kwargs.get('dummy_nodes', {})
        self.level_limit = kwargs.get('level_limit', int(np.ceil(np.log2(self.depth))))

        if self.initial_assignment is None:
            self.initial_assignment = set_initial_partitions(network, self.num_qubits, self.depth)

        if isinstance(self.qpu_sizes, dict):
            # If qpu_sizes is a dictionary, we need to convert it to a list of lists
            self.qpu_sizes = list(self.qpu_sizes.values())

    def FM_pass(self, assignment):

        spaces = find_spaces(self.num_qubits, self.depth, assignment, self.qpu_sizes)
        map_counts_and_configs(self.hypergraph, assignment, self.num_partitions, self.costs)

        lock_dict = {node: False for node in self.active_nodes}

        array = find_all_gains(self.hypergraph,
                               self.active_nodes,
                               assignment,
                               num_partitions=self.num_partitions,
                               costs=self.costs
                               )
        
        buckets = fill_buckets(array, self.max_gain)
        
        gain_list = []
        gain_list.append(0)
        assignment_list = []
        assignment_list.append(assignment)
        cumulative_gain = 0
        action = 0
        iter = 0

        while iter < self.limit:
            action, gain = find_action(buckets, lock_dict, spaces, self.max_gain)
            if action is None:
                break
            cumulative_gain += gain
            gain_list.append(cumulative_gain)
            node = (action[1], action[0])
            destination = action[2]
            source = assignment[node[1]][node[0]]
            assignment_new, array, buckets = take_action_and_update(self.hypergraph,
                                                                    node,
                                                                    destination,
                                                                    array,
                                                                    buckets,
                                                                    self.num_partitions,
                                                                    lock_dict,
                                                                    assignment,
                                                                    self.costs
                                                                    )
            update_spaces(node, source, destination, spaces)
            lock_dict = lock_node(node, lock_dict)

            assignment = assignment_new
            assignment_list.append(assignment)
            iter += 1
        
        return assignment_list, gain_list
    
    def run_FM(self, log = False):  
        
        assignment = self.initial_assignment.copy()

        initial_cost = calculate_full_cost(self.hypergraph, assignment, self.num_partitions, self.costs)
        
        if log:
            print("Initial cost:", initial_cost)
        cost = initial_cost
        cost_list = []
        best_assignments = []

        cost_list.append(cost)
        best_assignments.append(assignment)
        # print("Starting FM passes...")

        

        for n in range(self.passes):
            # print(f"Pass number: {n}")
            assignment_list, gain_list = self.FM_pass(assignment)

            # Decide how to pick new assignment depending on stochastic or not
            if self.stochastic:
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
    
    def MLFM(self, coarsened_hypergraphs, log=False):
        
        assignment = self.initial_assignment.copy()
        list_of_assignments = []
        list_of_assignments.append(assignment)

        list_of_costs = []

        initial_cost = calculate_full_cost(coarsened_hypergraphs[-1], assignment, self.num_partitions, costs=self.costs)
        list_of_costs.append(initial_cost)
        
        best_cost = initial_cost

        graph_list = coarsened_hypergraphs[::-1]
        active_nodes = graph_list[0].nodes
        mapping_list = mapping_list[::-1]

        graph_list = graph_list[:self.level_limit]
        mapping_list = mapping_list[:self.level_limit]

        pass_list = [int(self.passes/self.level_limit)] * self.level_limit

        for i, graph in enumerate(graph_list):

            active_nodes = graph.nodes

            if limit is None:
                limit = len(active_nodes)

            self.max_gain = find_max_gain(mapping_list, i)

            self.passes = pass_list[i]
            
            results = self.run_FM()
            best_cost_pass = results['best_cost']
            best_assignment_pass = results['best_assignment']

            
            if best_cost_pass < best_cost:
            # Keep track of the result
                best_cost = best_cost_pass
                assignment = best_assignment_pass
            else:
                assignment = initial_assignment
            

            if log:
                print(f'Best cost at level {i}: {best_cost}')

            refined_assignment = refine_assignment(i, len(graph_list), assignment, mapping_list)
            initial_assignment = refined_assignment

            list_of_assignments.append(initial_assignment)
            list_of_costs.append(best_cost)
        
        final_cost = min(list_of_costs)
        final_assignment = list_of_assignments[np.where(list_of_costs==final_cost)]

        results = {'best_cost' : final_cost, 'best_assignment' : final_assignment}

        return results


    def partition(self):
        kwargs = {'partitioner' : self.run_FM, 'log': False}
        return super().partition(**kwargs)

def find_max_gain(mapping_list, level):
    largest_node = 1
    for s_node in mapping_list[level]:
        length = len(mapping_list[level][s_node])
        if length > largest_node:
            largest_node = length
    return 2 * largest_node + 2

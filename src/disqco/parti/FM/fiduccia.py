from disqco.parti.partitioner import QuantumCircuitPartitioner
from qiskit import QuantumCircuit
from disqco.graphs.quantum_network import QuantumNetwork
import numpy as np
from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph
from disqco.parti.FM.FM_methods import *
import networkx as nx
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
import random

class FiducciaMattheyses(QuantumCircuitPartitioner):
    """
    Fiduccia-Mattheyses partitioning algorithm for quantum circuits.
    This class implements the Fiduccia-Mattheyses algorithm for partitioning
    quantum circuits into smaller sub-circuits.
    """

    # Sentinel: None = not yet attempted, False = unavailable
    _cpp_hg = None

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
        

        self.qpu_sizes = {qpu : self.network.qpu_sizes[qpu] for qpu in self.network.active_nodes}
        group_gates = kwargs.get('group_gates', True)
        hypergraph = kwargs.get('hypergraph', None)

        if hypergraph is None:
            self.hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=group_gates)
        else:
            self.hypergraph = hypergraph
            
        self.num_qubits = self.hypergraph.num_qubits
        self.depth = self.hypergraph.depth

        self.costs = kwargs.pop('costs', {})
        self.mapping = None
        self.dummy_nodes = kwargs.get('dummy_nodes', set())
        self.node_map = kwargs.get('node_map', {qpu : index for index, qpu in enumerate(self.qpu_sizes)})

        

        for node in self.hypergraph.nodes:
            if node[0] == 'dummy':
                qpu_index = node[2]
                self.dummy_nodes.add(node)
                if qpu_index not in self.node_map:
                    self.node_map[qpu_index] = len(self.node_map)
        
        



        self.sparse = kwargs.get('sparse', False)
        self.num_partitions = len(self.qpu_sizes)

        if self.initial_assignment is None:
            self.initial_assignment = set_initial_partition_assignment(graph=self.hypergraph, network=network)

        # Build C++ hypergraph lazily on first use
        self._cpp_hg = None

    def get_cpp_hg(self):
        """Return the cached C++ FMHyperGraph, building it on first call.

        Returns False if the C++ extension is unavailable so callers can
        branch to the pure-Python path without retrying each call.
        """
        if self._cpp_hg is None:
            from disqco.parti.FM._fm_cpp_builder import build_cpp_hgraph
            result = build_cpp_hgraph(self.hypergraph, self.num_partitions)
            self._cpp_hg = result if result is not None else False
        return self._cpp_hg

    def FM_pass(self, hypergraph, assignment, **kwargs):
        # C++ fast path: homo network, all nodes active (standard non-multilevel case)
        if (not self.network.hetero
                and 'active_hypergraph_nodes' not in kwargs):
            cpp_hg = self.get_cpp_hg()
            if cpp_hg is not False:
                try:
                    from disqco import _fm_cpp
                    limit = int(kwargs.get('limit', len(hypergraph.nodes) * 0.125))
                    qpu_sizes = np.array(
                        list(self.qpu_sizes.values()), dtype=np.int32
                    )
                    result = _fm_cpp.fm_pass(
                        cpp_hg, assignment, qpu_sizes, self.max_gain, limit
                    )
                    return result['assignment_list'], result['gain_list']
                except Exception:
                    pass  # fall through to Python path on any error

        random.seed()
        active_hypergraph_nodes = kwargs.get('active_hypergraph_nodes', hypergraph.nodes)
        limit = kwargs.get('limit', len(hypergraph.nodes) * 0.125)
        # print("Limit:", limit)
        spaces = find_spaces(assignment, self.qpu_sizes, hypergraph)

        map_counts_and_configs(hypergraph, 
                               assignment, 
                               self.num_partitions, 
                               costs=self.costs, 
                               node_map=self.node_map, 
                               dummy_nodes=self.dummy_nodes,
                               hetero=self.network.hetero)

        lock_dict = {node: False for node in active_hypergraph_nodes}
        lock_dict.update({node: True for node in self.dummy_nodes})

        array = find_all_gains(hypergraph,
                               assignment,
                               num_partitions=self.num_partitions,
                               costs = self.costs,
                               network=self.network,
                               node_map=self.node_map,
                               dummy_nodes=self.dummy_nodes,
                               active_qpu_nodes=self.network.active_nodes,
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
                                                                    node_map=self.node_map,
                                                                    dummy_nodes=self.dummy_nodes,
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
        self.max_gain = kwargs.pop('max_gain', self.find_max_gain(mapping))

        dummy_nodes = self.dummy_nodes
        log = kwargs.get('log', False)

        initial_cost = calculate_full_cost(hypergraph, 
                                           assignment, 
                                           self.num_partitions, 
                                           self.costs,
                                           network=self.network,
                                           node_map=self.node_map,
                                           dummy_nodes=dummy_nodes,
                                           hetero=self.network.hetero)
        
        if log:
            print("Initial cost:", initial_cost)
        cost = initial_cost
        cost_list = []
        best_assignments = []

        cost_list.append(cost)
        best_assignments.append(assignment)

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

        results = {'best_cost' : final_cost, 'best_assignment' : final_assignment, 'cost_list' : cost_list, 'assignment_list' : best_assignments}
        
        return results
    
    def partition(self, **kwargs):

        kwargs['graph'] = kwargs.get('graph', self.hypergraph)
        kwargs['assignment'] = kwargs.get('assignment', self.initial_assignment)
        kwargs['mapping'] = kwargs.get('mapping', None)
        kwargs['log'] = kwargs.get('log', False)
        kwargs['partitioner'] = kwargs.get('partitioner', self.run_FM)
        kwargs['hetero'] = self.network.hetero
    
        return super().partition(**kwargs)

    def multilevel_partition(self, coarsener=None, **kwargs):

        kwargs['graph'] = self.hypergraph
        sparse = kwargs.get('sparse', False)
        if coarsener is None:
            coarsener = kwargs.pop('coarsener', None)

        if coarsener is None:
            coarsener_class = HypergraphCoarsener()
            if sparse:
                coarsener = coarsener_class.coarsen_recursive_subgraph_batch
            else:
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

    # def net_coarsened_partition(self, **kwargs):
    #     """
    #     Partition the network using the coarsened hypergraph.
    #     """
    #     kwargs['graph'] = self.hypergraph
    #     kwargs['assignment'] = self.initial_assignment
    #     kwargs['mapping'] = kwargs.get('mapping', None)
    #     kwargs['log'] = kwargs.get('log', False)
    #     kwargs['partitioner'] = self.run_FM
    #     kwargs['hetero'] = self.network.hetero

    #     build_next_level = kwargs.get('build_next_level', True)
    #     network_level_list = kwargs.get('network_level_list', self.network.network_level_list)
    #     level_idx = kwargs.get('level_idx', 0)

    #     return super().net_coarsened_partition(build_next_level=build_next_level,
    #                                             network_level_list=network_level_list,
    #                                             level_idx=level_idx, **kwargs)
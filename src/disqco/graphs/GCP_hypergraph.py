from collections import defaultdict
from disqco.utils.qiskit_to_op_list import circuit_to_gate_layers, layer_list_to_dict
from disqco.graphs.greedy_gate_grouping import group_distributable_packets_sym, group_distributable_packets_asym
from qiskit import QuantumCircuit

class QuantumCircuitHyperGraph:
    """
    Class for temporal hypergraph representation of quantum circuit.
    """
    def __init__(self, 
                circuit : QuantumCircuit, 
                group_gates : bool = True, 
                anti_diag : bool = True,
                map_circuit : bool = True,
                qpu_sizes = None):
        # Keep a set of all nodes (qubit, time)
        self.nodes = set()
        self.hyperedges = {}
        self.node2hyperedges = defaultdict(set)
        self.adjacency = defaultdict(set)
        self.node_attrs = {}
        self.hyperedge_attrs = {}
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.depth = circuit.depth()
        if map_circuit:
            self.init_from_circuit(group_gates, anti_diag, qpu_sizes=qpu_sizes)

    def init_from_circuit(self, group_gates=True, anti_diag=False, qpu_sizes=None):
        self.layers = self.extract_layers(group_gates=group_gates, anti_diag=anti_diag, qpu_sizes=qpu_sizes)
        self.depth = len(self.layers)
        self.add_time_neighbor_edges(self.depth, range(self.num_qubits))
        self.map_circuit_to_hypergraph()

    def extract_layers(self, group_gates=True, anti_diag=False, qpu_sizes=None):
        layers = circuit_to_gate_layers(self.circuit, qpu_sizes=qpu_sizes)
        layers = layer_list_to_dict(layers)
        basis_gates = self.circuit.count_ops()
        if group_gates:
            if 'cx' in basis_gates or 'cu' in basis_gates:
                layers = group_distributable_packets_asym(layers, group_anti_diags=anti_diag)
            else:
                layers = group_distributable_packets_sym(layers, group_anti_diags=anti_diag)
        return layers

    def add_node(self, qubit, time):
        """
        Add a node (qubit, time). If it already exists, do nothing.
        """
        node = (qubit, time)
        self.nodes.add(node)

        if node not in self.node_attrs:
            self.node_attrs[node] = {}
        return node
    
    def remove_node(self, node):
        """
        Remove a node from the graph.
        """
        if node not in self.nodes:
            raise KeyError(f"Node {node} does not exist")
        
        # Remove all hyperedges that contain this node
        for edge_id in self.node2hyperedges[node]:
    
            self.remove_hyperedge(edge_id)
        
        # Remove the node itself
        self.nodes.remove(node)
        del self.node_attrs[node]
    
    def remove_hyperedge(self, edge_id):
        """
        Remove a hyperedge from the graph.
        """
        if edge_id not in self.hyperedges:
            raise KeyError(f"Hyperedge {edge_id} does not exist")
        
        # Remove the hyperedge from all nodes
        edge_data = self.hyperedges[edge_id]
        all_nodes = edge_data['root_set'].union(edge_data['receiver_set'])
        for node in all_nodes:
            self.node2hyperedges[node].remove(edge_id)
        
        # Remove the hyperedge itself
        del self.hyperedges[edge_id]
        del self.hyperedge_attrs[edge_id]
    
    def remove_node_from_hyperedge(self, node, edge_id):
        """
        Remove a node from a hyperedge.
        """
        if edge_id not in self.hyperedges:
            raise KeyError(f"Hyperedge {edge_id} does not exist")
        
        # Remove the node from the hyperedge
        edge_data = self.hyperedges[edge_id]
        if node in edge_data['root_set']:
            edge_data['root_set'].remove(node)
        elif node in edge_data['receiver_set']:
            edge_data['receiver_set'].remove(node)
        else:
            raise KeyError(f"Node {node} is not part of hyperedge {edge_id}")
        
        # Update the node2hyperedges mapping
        self.node2hyperedges[node].remove(edge_id)
        self.adjacency[node].discard(edge_id)

    def add_time_neighbor_edges(self, depth, qubits):
        """
        For each qubit in qubits, connect (qubit, t) to (qubit, t+1)
        for t in [0, max_time-1].
        """
        for qubit in qubits:
            for t in range(0,depth-1):
                node_a = (qubit, t)
                node_b = (qubit, t + 1)

                self.add_node(qubit, t)
                self.add_node(qubit, t + 1)

                self.add_edge((node_a,node_b), node_a, node_b)
    
    def add_hyperedge(self, root, root_set, receiver_set):
        """
        Create a new hyperedge with the given edge_id connecting the given node_list.
        node_list can be any iterable of (qubit, time) tuples.
        """
        edge_tuple = root
        # Optionally ensure all nodes exist in self.nodes
        # (Or do it automatically, but typically you want to be consistent)
        for node in receiver_set:
            if node not in self.nodes:
                raise ValueError(f"Node {node} not found in the graph. "
                                 "Add it first or allow auto-add.")
        
        # Store the hyperedge
        self.hyperedges[edge_tuple] = {'root_set': root_set, 'receiver_set': receiver_set}
        
        all_nodes = root_set.union(receiver_set)
        for node in all_nodes:
            self.node2hyperedges[node].add(edge_tuple)
            
        # (Optionally) update adjacency caches if you're storing them
        for node in all_nodes:
            for other_node in all_nodes:
                if other_node != node:
                    self.adjacency[node].add(other_node)
        
        if edge_tuple not in self.hyperedge_attrs:
            self.hyperedge_attrs[edge_tuple] = {}
    
    def add_edge(self, edge_id, node_a, node_b):
        """
        For a standard 2-node connection (a "regular" gate), treat it as a hyperedge of size 2.
        """
        root_set = set()
        root_set.add(node_a)
        receiver_set = set()
        receiver_set.add(node_b)
        self.add_hyperedge(edge_id, root_set, receiver_set)
    
    def neighbors(self, node):
        """
        Return all neighbors of `node`, i.e. all nodes that share
        at least one hyperedge with `node`.
        """
        nbrs = set()
        # Get all hyperedges for this node
        edge_ids = self.node2hyperedges.get(node, set())
        for e_id in edge_ids:
            # Add all nodes in that hyperedge
            nbrs.update(self.hyperedges[e_id])
        
        # Remove the node itself
        nbrs.discard(node)
        return nbrs
    
    def set_node_attribute(self, node, key, value):
        if node not in self.nodes:
            raise KeyError(f"Node {node} does not exist")
        # Ensure there's a dict to store attributes
        if node not in self.node_attrs:
            self.node_attrs[node] = {}
        self.node_attrs[node][key] = value

    def get_node_attribute(self, node, key, default=None):
        if node not in self.nodes:
            raise KeyError(f"Node {node} does not exist")
        return self.node_attrs[node].get(key, default)

    def set_hyperedge_attribute(self, edge_id, key, value):
        if edge_id not in self.hyperedges:
            raise KeyError(f"Hyperedge {edge_id} does not exist")
        if edge_id not in self.hyperedge_attrs:
            self.hyperedge_attrs[edge_id] = {}
        self.hyperedge_attrs[edge_id][key] = value

    def get_hyperedge_attribute(self, edge_id, key, default=None):
        if edge_id not in self.hyperedges:
            raise KeyError(f"Hyperedge {edge_id} does not exist")
        return self.hyperedge_attrs[edge_id].get(key, default)

    def assign_positions(self, num_qubits_phys):
        """
        Assign a 'pos' attribute to all nodes based on their (qubit, time).
        
        The position is (x, y) = (t, num_qubits_phys - q) 
        for each node (q, t).
        
        :param num_qubits_phys: The total number of physical qubits or 
                                however many 'vertical slots' you want.
        """
        for (q, t) in self.nodes:
            x = t
            y = num_qubits_phys - q
            # Store in node_attrs or via the set_node_attribute function:
            self.set_node_attribute((q, t), "pos", (x, y))
 
    def copy(self):
        """
        Create a new QuantumCircuitHyperGraph that is an identical 
        (shallow) copy of this one, so that modifications to the copy 
        do not affect the original.
        """
        # 1) Create a blank instance (no qubits/depth needed for now)
        new_graph = QuantumCircuitHyperGraph(circuit=self.circuit, map_circuit=False)

        # 2) Copy nodes
        new_graph.nodes = set(self.nodes)

        # 3) Copy hyperedges (including root/receiver sets)
        new_graph.hyperedges = {}
        for edge_id, edge_data in self.hyperedges.items():
            root_copy = set(edge_data['root_set'])
            rec_copy = set(edge_data['receiver_set'])
            new_graph.hyperedges[edge_id] = {
                'root_set': root_copy,
                'receiver_set': rec_copy
            }

        # 4) Copy node2hyperedges
        new_graph.node2hyperedges = defaultdict(set)
        for node, edge_ids in self.node2hyperedges.items():
            new_graph.node2hyperedges[node] = set(edge_ids)

        # 5) Copy adjacency
        new_graph.adjacency = defaultdict(set)
        for node, nbrs in self.adjacency.items():
            new_graph.adjacency[node] = set(nbrs)

        # 6) Copy node_attrs
        new_graph.node_attrs = {}
        for node, attr_dict in self.node_attrs.items():
            new_graph.node_attrs[node] = dict(attr_dict)

        # 7) Copy hyperedge_attrs
        new_graph.hyperedge_attrs = {}
        for edge_id, attr_dict in self.hyperedge_attrs.items():
            new_graph.hyperedge_attrs[edge_id] = dict(attr_dict)

        return new_graph
     
    def map_circuit_to_hypergraph(self,):
        layers_dict = self.layers
        for l in layers_dict:
            layer = layers_dict[l]
            for gate in layer:
                if gate['type'] == 'single-qubit':
                    qubit = gate['qargs'][0]
                    time = l
                    node = self.add_node(qubit,time)
                    self.set_node_attribute(node,'type',gate['type'])
                    self.set_node_attribute(node,'name',gate['name'])
                    self.set_node_attribute(node,'params',gate['params'])
                elif gate['type'] == 'two-qubit':
                    qubit1 = gate['qargs'][0]
                    qubit2 = gate['qargs'][1]
                    time = l
                    node1 = self.add_node(qubit1,time)
                    self.set_node_attribute(node1,'type',gate['type'])
                    self.set_node_attribute(node1,'name','control')
                    node2 = self.add_node(qubit2,time)
                    self.set_node_attribute(node2,'type',gate['type'])
                    if gate['name'] == 'cx' or gate['name'] == 'cu': # May need to specify more
                        self.set_node_attribute(node2,'name','target')
                    else:
                        self.set_node_attribute(node2,'name','control')
                    self.add_edge((node1,node2),node1,node2)
                    self.set_hyperedge_attribute((node1,node2),'type',gate['type'])
                    self.set_hyperedge_attribute((node1,node2),'name',gate['name'])
                    self.set_hyperedge_attribute((node1,node2),'params',gate['params'])
                elif gate['type'] == 'group':
                    root = gate['root']
                    start_time = l
                    root_node = self.add_node(root,start_time)
                    root_set = set()
                    root_set.add(root_node)
                    receiver_set = set()
                    for sub_gate in gate['sub-gates']:
                        if sub_gate['type'] == 'single-qubit':
                            qubit = sub_gate['qargs'][0]
                            time = sub_gate['time']
                            node = self.add_node(qubit,time)
                            root_set.add(node)
                            self.set_node_attribute(node,'type',sub_gate['type'])
                            self.set_node_attribute(node,'name',sub_gate['name'])
                            self.set_node_attribute(node,'params',sub_gate['params'])
                        elif sub_gate['type'] == 'two-qubit':
                            qubit1 = sub_gate['qargs'][0]
                            qubit2 = sub_gate['qargs'][1]
                            time = sub_gate['time']
                            node1 = self.add_node(qubit1,time)
                            root_set.add(node1)
                            if node1 == root_node:
                                type_ = 'group'
                            else:
                                type_ = 'root_t'
                            self.set_node_attribute(node1,'type',type_)
                            self.set_node_attribute(node1,'name','control')
                            node2 = self.add_node(qubit2,time)
                            receiver_set.add(node2)
                            self.set_node_attribute(node2,'type',gate['type'])
                            if sub_gate['name'] == 'cx' or sub_gate['name'] == 'cu': # May need to specify more
                                self.set_node_attribute(node2,'name','target')
                            else:
                                self.set_node_attribute(node2,'name','control')
                    for t in range(start_time,time+1):
                        root_set.add((root,t))
                        if t != start_time:
                            self.set_node_attribute((root,t),'type', 'root_t')
                    self.add_hyperedge(root_node,root_set,receiver_set)


class SubGraphManager:
    """
    Class for managing subgraphs of a larger graph.
    """

    def __init__(self, graph: QuantumCircuitHyperGraph):
        """
        Initialize the SubGraphManager with a graph.

        :param graph: The input graph to be managed.
        """
        self.initial_graph = graph
        self.subgraphs = [[graph]]

        
    def build_partition_subgraphs(self,
                                  graph: QuantumCircuitHyperGraph,
                                    assignment: list[list[int]],  # 2D assignment[t][q] -> partition id
                                    k: int,
                                    node_map = None,
                                    current_network = None,
                                    new_networks = None,
                                    assignment_map = None,
                                    dummy_nodes = set()
                                    ) -> list[QuantumCircuitHyperGraph]:
        """
        Returns k subgraphs, one for each partition p in [0..k-1].
        
        In each subgraph p:
        - Nodes in partition p remain as real nodes.
        - Nodes in other partitions p' != p become merged into a single dummy node
            that represents partition p'.
        - Any hyperedge references to nodes not in partition p are rerouted
            to the corresponding dummy node.
        - Self-loops (when root and receiver sets overlap) are automatically 'contracted'
            by removing overlapping nodes from the receiver set (and removing the hyperedge
            if it becomes empty).

        :param original_graph: The complete QuantumCircuitHyperGraph.
        :param assignment: A 2D list, assignment[t][q], giving the partition for qubit q at time t.
        :param k: Number of partitions.
        :return: A list of k QuantumCircuitHyperGraph objects, one per partition.
        """

        # -----------------------------
        # Step 1) Make k copies of the original graph
        # -----------------------------
        subgraphs = []
        for i in range(k):
            # Start each as a shallow copy of the original
            sg = graph.copy()
            subgraphs.append(sg)

        # -----------------------------
        # Step 2) For each subgraph p, create dummy nodes for each foreign partition p' != p
        # -----------------------------
        dummy_map_list = []
        
        for idx1 in range(k):
            new_network = new_networks[idx1][0]
            new_network_graph = new_network.qpu_graph
            active_nodes = new_networks[idx1][1]

            dummy_counter = 0
            counter = 0
            for node in active_nodes:
                if node in current_network.qpu_graph.nodes:
                    p = node
                    break
            sg = subgraphs[idx1]
            dummy_map = {}  # Maps p' -> dummy_node
            # print("Source partition", p)

            for idx2 in range(k + len(dummy_nodes)):
                if node_map is not None:
                    p_prime = node_map[idx2]
                else:
                    p_prime = idx2 
                if p_prime != p:
                    if p_prime not in new_network.qpu_graph.nodes:
                        for qpu in new_network.mapping:
                            if p_prime in new_network.mapping[qpu]:
                                if qpu in dummy_nodes: 
                                    dummy_map[p_prime] = dummy_map[qpu]
                                    break
                                else:
                                    p_prime = qpu
                                    break
                            
                        
                    # Create a unique dummy node
                    if p_prime not in dummy_map:
                        dummy_node = ('dummy', p, p_prime, dummy_counter)    
                        dummy_counter += 1
                        counter+= 1
                        # If your add_node expects (qubit, time), you might do a direct insertion:
                        sg.nodes.add(dummy_node)
                        sg.node_attrs[dummy_node] = {
                            "dummy": True,
                            "represents_partition": p_prime
                        }

                        dummy_map[p_prime] = dummy_node

            # dummy_map_list.append(dummy_map)
            

        # -----------------------------
        # Step 3) Merge foreign nodes into dummy nodes
        # -----------------------------
        # for idx1 in range(k):
        #     if node_map is not None:
        #         p = node_map[idx1]
        #     else:
        #         p = idx1
            sg = subgraphs[idx1]
            # dummy_map = dummy_map_list[idx1]

            # We'll iterate over a snapshot of the current sg.nodes
            # since we'll remove nodes as we go
            all_nodes = list(sg.nodes)
            for node in all_nodes:
                # Skip dummy nodes (already created)

                if node not in sg.nodes:
                    # Possibly removed or merged already
                    continue

                # node is typically (qubit, time)
                if isinstance(node, tuple) and len(node) == 4 and node[0] == "dummy":
                    node_partition = node[2]
                    # print("Current dummy node", node)
                    # print("Node partition", node_partition)
                    # print("Corresponds to dummy node", dummy_map[node_partition])
                    if node_partition in dummy_map:
                        if node == dummy_map[node_partition]:
                            continue
                    
                    # print("Dummy node not in new network graph")
                    # print("Find merged node")
                    # print("Mapping for new network", new_network.mapping)
                    if node_partition not in new_network_graph.nodes:
                        for qpu in new_network.mapping:
                            if node_partition in new_network.mapping[qpu]:
                                dummy_map[node_partition] = dummy_map[qpu] 


                else:
                    q, t = node
                    if assignment_map is not None:
                        q_sub, t_sub = assignment_map[(q,t)]
                        node_partition = assignment[t_sub][q_sub]
                    else:
                        node_partition = assignment[t][q]
                    
                    if node_map is not None:
                        node_partition = node_map[node_partition]
                    else:
                        node_partition = node_partition
                
                if node_partition != p:

                    # This node doesn't belong to subgraph p => merge with dummy for node_partition
                    dummy_node = dummy_map[node_partition]

                    # For each hyperedge containing this node, replace it with the dummy node
                    if node in sg.node2hyperedges:
                        edges_for_node = list(sg.node2hyperedges[node])
                        for edge_id in edges_for_node:
                            if edge_id not in sg.hyperedges:
                                continue
                            edge_data = sg.hyperedges[edge_id]
                            root_set = edge_data["root_set"]
                            rec_set  = edge_data["receiver_set"]

                            changed = False
                            if node in root_set:
                                root_set.remove(node)
                                root_set.add(dummy_node)
                                changed = True
                            if node in rec_set:
                                rec_set.remove(node)
                                rec_set.add(dummy_node)
                                changed = True

                            if changed:
                                sg.node2hyperedges[node].discard(edge_id)
                                sg.node2hyperedges[dummy_node].add(edge_id)

                    # Finally, remove the foreign node from the subgraph
                    sg.remove_node(node)
            qubits = set()
            for node in sg.nodes:
                qubits.add(node[0])  # Assuming node is (qubit, time)
            sg.num_qubits = len(qubits)
            # -----------------------------
            # Step 4) Contract any self-loops in hyperedges
            # -----------------------------
            # A "self-loop" here means a hyperedge whose root_set and receiver_set overlap.
            # We'll remove the overlapping nodes from receiver_set.
            # If that empties receiver_set, we remove the hyperedge altogether.
            # (Adjust if you have a different "contraction" logic in mind.)
            # -----------------------------
            for edge_id in list(sg.hyperedges.keys()):
                edge_data = sg.hyperedges[edge_id]
                root_set  = edge_data["root_set"]
                rec_set   = edge_data["receiver_set"]

                # Find intersection
                overlap = root_set.intersection(rec_set)
                if overlap:
                    # Remove these from the receiver set
                    rec_set.difference_update(overlap)

                    # If the entire rec_set is empty, remove hyperedge
                    if not rec_set:
                        sg.remove_hyperedge(edge_id)
        self.subgraphs.append(subgraphs)
        return subgraphs


        
    # def build_partition_subgraphs(self,
    #                               graph: QuantumCircuitHyperGraph,
    #                                 assignment: list[list[int]],  # 2D assignment[t][q] -> partition id
    #                                 k: int,
    #                                 node_map = None,
    #                                 current_network = None,
    #                                 new_networks = None,
    #                                 assignment_map = None
    #                                 ) -> list[QuantumCircuitHyperGraph]:
    #     """
    #     Returns k subgraphs, one for each partition p in [0..k-1].
        
    #     In each subgraph p:
    #     - Nodes in partition p remain as real nodes.
    #     - Nodes in other partitions p' != p become merged into a single dummy node
    #         that represents partition p'.
    #     - Any hyperedge references to nodes not in partition p are rerouted
    #         to the corresponding dummy node.
    #     - Self-loops (when root and receiver sets overlap) are automatically 'contracted'
    #         by removing overlapping nodes from the receiver set (and removing the hyperedge
    #         if it becomes empty).

    #     :param original_graph: The complete QuantumCircuitHyperGraph.
    #     :param assignment: A 2D list, assignment[t][q], giving the partition for qubit q at time t.
    #     :param k: Number of partitions.
    #     :return: A list of k QuantumCircuitHyperGraph objects, one per partition.
    #     """

    #     # -----------------------------
    #     # Step 1) Make k copies of the original graph
    #     # -----------------------------
    #     subgraphs = []
    #     for _ in range(k):
    #         sg = graph.copy()
    #         subgraphs.append(sg)

    #     # -----------------------------
    #     # Step 2) For each subgraph p, create dummy nodes for each foreign partition p' != p
    #     # -----------------------------

    #     print("Nodes in current network", current_network.qpu_graph.nodes)

    #     dummy_map_list = []




    #     for node_id in node_map:

    #         qpu = node_map[node_id]

        
    #     for idx1 in range(k):
    #         new_graph = new_networks[idx1][0].qpu_graph
    #         active_nodes = new_networks[idx1][1]

    #         new_node_map = {}
    #         h = 0
    #         for node in new_graph.nodes:
    #             if node in active_nodes:
    #                 new_node_map[h] = node
    #                 h += 1
            
    #         for node in new_graph.nodes:
    #             if node not in active_nodes:
    #                 new_node_map[h] = node
    #                 h += 1
    #         print("New node map", new_node_map)
    #         print("Nodes in new network", new_graph.nodes)
    #         print("Active nodes", active_nodes)
    #         dummy_counter = 0
    #         counter = 0
    #         if node_map is not None:
    #             p = node_map[idx1]
    #         else:
    #             p = idx1
    #         sg = subgraphs[idx1]
    #         dummy_map = {}  # Maps p' -> dummy_node
    #         for node_idx in new_node_map:
    #             if new_node_map is not None:
    #                 p_prime = new_node_map[node_idx]
    #             else:
    #                 p_prime = node_idx 
    #             if p_prime in active_nodes:
    #                 continue
    #             # Create a unique dummy node
    #             dummy_node = ("dummy", p, p_prime, dummy_counter)
    #             print("Dummy node", dummy_node)
    #             dummy_counter += 1
    #             counter+= 1
    #             # If your add_node expects (qubit, time), you might do a direct insertion:
    #             sg.nodes.add(dummy_node)
    #             sg.node_attrs[dummy_node] = {
    #                 "dummy": True,
    #                 "represents_partition": p_prime
    #             }

    #             dummy_map[p_prime] = dummy_node
    #         print("Dummy map", dummy_map)

    #     #     dummy_map_list.append(dummy_map)
        
    #     # print("Dummy map list", dummy_map_list)
            

    #     # -----------------------------
    #     # Step 3) Merge foreign nodes into dummy nodes
    #     # -----------------------------
    #     # for idx1 in range(k):
                
    #         sg = subgraphs[idx1]
    #         # dummy_map = dummy_map_list[idx1]

    #         # We'll iterate over a snapshot of the current sg.nodes
    #         # since we'll remove nodes as we go
    #         all_nodes = list(sg.nodes)
    #         for node in all_nodes:
    #             # Skip dummy nodes (already created)
    #             if isinstance(node, tuple) and len(node) == 4 and node[0] == "dummy":
    #                 continue
    #             if node not in sg.nodes:
    #                 # Possibly removed or merged already
    #                 continue

    #             # node is typically (qubit, time)
    #             q, t = node
    #             if assignment_map is not None:
    #                 q_sub, t_sub = assignment_map[(q,t)]
    #                 node_partition = assignment[t_sub][q_sub]
    #             else:
    #                 node_partition = assignment[t][q]
                
    #             if node_map is not None:
    #                 node_partition = node_map[node_partition]
    #             else:
    #                 node_partition = node_partition

    #             if node_partition not in active_nodes:
    #                 # This node doesn't belong to subgraph p => merge with dummy for node_partition
    #                 dummy_node = dummy_map[node_partition]
    #                 print("Dummy node", dummy_node)

    #                 # For each hyperedge containing this node, replace it with the dummy node
    #                 if node in sg.node2hyperedges:
    #                     edges_for_node = list(sg.node2hyperedges[node])
    #                     for edge_id in edges_for_node:
    #                         if edge_id not in sg.hyperedges:
    #                             continue
    #                         edge_data = sg.hyperedges[edge_id]
    #                         root_set = edge_data["root_set"]
    #                         rec_set  = edge_data["receiver_set"]

    #                         changed = False
    #                         if node in root_set:
    #                             root_set.remove(node)
    #                             root_set.add(dummy_node)
    #                             changed = True
    #                         if node in rec_set:
    #                             rec_set.remove(node)
    #                             rec_set.add(dummy_node)
    #                             changed = True

    #                         if changed:
    #                             sg.node2hyperedges[node].discard(edge_id)
    #                             sg.node2hyperedges[dummy_node].add(edge_id)

    #                 # Finally, remove the foreign node from the subgraph
    #                 sg.remove_node(node)
    #         qubits = set()
    #         for node in sg.nodes:
    #             qubits.add(node[0])  # Assuming node is (qubit, time)
    #         sg.num_qubits = len(qubits)
    #         # -----------------------------
    #         # Step 4) Contract any self-loops in hyperedges
    #         # -----------------------------
    #         # A "self-loop" here means a hyperedge whose root_set and receiver_set overlap.
    #         # We'll remove the overlapping nodes from receiver_set.
    #         # If that empties receiver_set, we remove the hyperedge altogether.
    #         # (Adjust if you have a different "contraction" logic in mind.)
    #         # -----------------------------
    #         for edge_id in list(sg.hyperedges.keys()):
    #             edge_data = sg.hyperedges[edge_id]
    #             root_set  = edge_data["root_set"]
    #             rec_set   = edge_data["receiver_set"]

    #             # Find intersection
    #             overlap = root_set.intersection(rec_set)
    #             if overlap:
    #                 # Remove these from the receiver set
    #                 rec_set.difference_update(overlap)

    #                 # If the entire rec_set is empty, remove hyperedge
    #                 if not rec_set:
    #                     sg.remove_hyperedge(edge_id)
    #     self.subgraphs.append(subgraphs)
    #     return subgraphs

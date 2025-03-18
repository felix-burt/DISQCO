from collections import defaultdict
from disqco.utils.qiskit_to_op_list import circuit_to_gate_layers, layer_list_to_dict
from disqco.graphs.greedy_gate_grouping import group_distributable_packets
from qiskit import QuantumCircuit

class QuantumCircuitHyperGraph:
    """
    Class for time extended hypergraph representation of quantum circuit.
    """
    def __init__(self, 
                circuit : QuantumCircuit, 
                group_gates : bool = True, 
                anti_diag : bool = False,
                map_circuit : bool = True):
        
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
            self.init_from_circuit(group_gates, anti_diag)

    def init_from_circuit(self, group_gates=True, anti_diag=False):
        self.add_time_neighbor_edges(self.depth, range(self.num_qubits))
        self.layers = self.extract_layers(self.circuit, group_gates=group_gates, anti_diag=anti_diag)
        self.map_circuit_to_hypergraph()

    def extract_layers(self, circuit, group_gates=True, anti_diag=False):
        layers = circuit_to_gate_layers(circuit)
        layers = layer_list_to_dict(layers)
        if group_gates:
            layers = group_distributable_packets(layers, group_anti_diags=anti_diag)
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
                    self.add_hyperedge(root_node,root_set,receiver_set)

    def map_circuit_to_hypergraph_2(self,base_graph):
        for qubit in qubit_list:
            for t in range(depth):
                node = self.add_node(qubit,t)

                self.set_node_attribute(node,'type',base_graph.get_node_attribute((qubit,t),'type'))
                self.set_node_attribute(node,'name',base_graph.get_node_attribute((qubit,t),'name'))
                self.set_node_attribute(node,'params', base_graph.get_node_attribute((qubit,t),'params'))
                edges = base_graph.node2hyperedges[(qubit,t)]
                for edge in edges:
                    edge_type = base_graph.get_hyperedge_attribute(edge,'type')
                    edge_name = base_graph.get_hyperedge_attribute(edge,'name')
                    
                    if edge_type == 'two-qubit':
                        root_set = base_graph.hyperedges[edge]['root_set']
                        receiver_set = base_graph.hyperedges[edge]['receiver_set']
                        gate_node = (root_set[0], receiver_set[0])

            
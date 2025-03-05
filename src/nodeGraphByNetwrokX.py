import networkx as nx
import matplotlib.pyplot as plt
from qiskit import transpile

class nodeGraph:
    def __init__(self, circuit):
        self.circuit = circuit
        self.G = nx.Graph()
        self.pos = {}
        self.depth = circuit.depth
    
    def QCircuitToNodeGraph(self, show_plot=True):
        num_qubits = self.circuit.num_qubits
        
        # Transpile to specific basis gates
        basis_gates = ['u', 'cp']
        self.circuit = transpile(self.circuit, basis_gates=['u', 'cp'])
        
        # Step 2: Initialize Graph
        self.G.clear()
        
        # Step 3: Extract Multi-Qubit Gates and Add to Graph
        qubit_time_map = {q: 1 for q in range(num_qubits)}  # Start from time step 1
        node_map = {q: [] for q in range(num_qubits)}  # Store nodes for horizontal edges
        
        # Add input nodes at time step 0
        for qubit in range(num_qubits):
            input_node = (0, qubit)
            self.G.add_node(input_node, pos=(0, -qubit), color='gray')
            node_map[qubit].append(input_node)  # Store node for horizontal connection
        
        for instruction in self.circuit.data:
            gate = instruction.operation.name
            qubits = [self.circuit.find_bit(q).index for q in instruction.qubits]  # Get correct qubit indices
        
            if gate in ["cx", "cp"]:  # Multi-qubit gates (CNOT, CP)
                control, target = qubits
                
                # Align both qubits to the same time step
                time_step = max(qubit_time_map[control], qubit_time_map[target])
                
                control_node = (time_step, control)
                target_node = (time_step, target)  # Ensure vertical alignment
                
                # Add nodes and vertical edge
                self.G.add_node(control_node, pos=(time_step, -control))  # Flip Y-axis
                self.G.add_node(target_node, pos=(time_step, -target))
                self.G.add_edge(control_node, target_node)  # Vertical edge
                
                # Store node for horizontal connection
                node_map[control].append(control_node)
                node_map[target].append(target_node)
                
                # Update time tracking for both qubits
                qubit_time_map[control] = time_step + 1  # Reserve next available step
                qubit_time_map[target] = time_step + 1
        
        # Add output nodes
        max_time_step = max(qubit_time_map.values())
        for qubit in range(num_qubits):
            output_node = (max_time_step, qubit)
            self.G.add_node(output_node, pos=(max_time_step, -qubit), color='gray')
            node_map[qubit].append(output_node)
        
        # Add horizontal edges
        for qubit in range(num_qubits):
            nodes = node_map[qubit]
            for i in range(len(nodes) - 1):
                self.G.add_edge(nodes[i], nodes[i + 1])
        
        # Step 4: Extract Positions for Visualization
        self.pos = {node: (x, y) for node, (x, y) in nx.get_node_attributes(self.G, 'pos').items()}  # Maintain flipped order
        colors = ['gray' if 'color' in self.G.nodes[node] and self.G.nodes[node]['color'] == 'gray' else 'black' for node in self.G.nodes]
        
        if show_plot:
            # Step 5: Draw the Graph
            plt.figure(figsize=(10, 6))
            
            # Draw network graph
            nx.draw(self.G, self.pos, with_labels=False, node_color=colors, edge_color='black', node_size=50, font_size=8)
            
            # Add qubit labels on the left
            for qubit in range(num_qubits):
                y_position = -qubit  # Flip qubit index
                plt.text(-1.5, y_position, f"q{qubit}", verticalalignment='center', color='black')
            
            # Formatting
            plt.title("Quantum Circuit Node Graph")
            plt.xlabel("Time Step")
            
            # Proper Y-tick Labels
            plt.yticks([])
            plt.xticks(range(0, max_time_step + 1, 2))  # Show time steps at intervals
            
            plt.show()
        
        return self.G, self.pos
    
    def get_circuit_depth(self):
        """Calculate and return the depth of the quantum circuit."""
        return self.depth



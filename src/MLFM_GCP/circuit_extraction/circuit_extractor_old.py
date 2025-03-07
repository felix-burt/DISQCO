import networkx as nx
from MLFM_GCP.circuit_extraction.DQC_qubit_manager import CommunicationQubitManager, ClassicalBitManager, DataQubitManager
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit   
from qiskit.circuit import Qubit
from MLFM_GCP.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
import copy 

class TeleportationManager:
    
    def __init__(self,
                qc : QuantumCircuit, 
                qubit_manager : DataQubitManager, 
                comm_manager : CommunicationQubitManager, 
                creg_manager : ClassicalBitManager) -> None:
        
        self.qc = qc
        self.qubit_manager = qubit_manager
        self.comm_manager = comm_manager
        self.creg_manager = creg_manager

    def build_epr_circuit(self,) -> QuantumCircuit:
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)

        gate = circ.to_gate()
        gate.name = "EPR"

        return gate

    def build_root_entanglement_circuit(self,) -> QuantumCircuit:

        epr_circ = self.build_epr_circuit()
        circ = QuantumCircuit(3,1)
        circ.append(epr_circ, [1, 2])
        circ.cx(0, 1)
        circ.measure(1, 0)
        circ.reset(1)
        circ.x(2).c_if(0, 1)

        instr = circ.to_instruction()
        instr.name = "entangle_root"

        return instr
    
    def build_end_entanglement_circuit(self,) -> QuantumCircuit:

        circ = QuantumCircuit(2,1)
        circ.h(1)
        circ.measure(1, 0)
        circ.reset(1)
        circ.z(0).c_if(0, 1)

        instr = circ.to_instruction()
        instr.name = "end_entanglement_link"

        return instr

    def build_teleporation_circuit(self,) -> QuantumCircuit:
        
        circ = QuantumCircuit(3,2)

        epr_circ = self.build_epr_circuit()
        circ.append(epr_circ, [1, 2])
        circ.cx(0, 1)
        circ.h(0)
        circ.measure(0, 0)
        circ.measure(1, 1)
        circ.reset(0)
        circ.reset(1)
        circ.x(2).c_if(1, 1)
        circ.z(2).c_if(0, 1)

        instr = circ.to_instruction(label="state_teleport")

        return instr

    def build_gate_teleportation_circuit(self,gate_params) -> QuantumCircuit:
        
        circ = QuantumCircuit(4,1)

        root_entanglement_circuit = self.build_root_entanglement_circuit()
        circ.append(root_entanglement_circuit, [0,1,2], [0])

        circ.cp(gate_params[0], 2, 3)

        entanglement_end_circuit = self.build_end_entanglement_circuit()

        circ.append(entanglement_end_circuit, [0,2], [0])

        instr = circ.to_instruction(label="gate_teleport")

        return instr

    def generate_epr(self, 
                    p1 : int, 
                    p2 : int, 
                    comm_id1 : Qubit = None, 
                    comm_id2 : Qubit = None) -> tuple[Qubit, Qubit]:

        # Find free communication qubits unless the ids are provided
        if comm_id1 is None:
            comm_qubit1 = self.comm_manager.find_comm_idx(p1)
        else:
            comm_qubit1 = comm_id1
        
        if comm_id2 is None:
            comm_qubit2 = self.comm_manager.find_comm_idx(p2)
        else:
            comm_qubit2 = comm_id2

        gate = self.build_epr_circuit() # Build the EPR pair sub-circuit
        self.qc.append(gate, [comm_qubit1, comm_qubit2]) # Add the sub-circuit to the quantum circuit

        self.comm_manager.in_use_comm[p1].append(comm_qubit1) # Mark communication qubits as in use
        self.comm_manager.in_use_comm[p2].append(comm_qubit2)

        return comm_qubit1, comm_qubit2
    
    def entangle_root(self, 
                    root_id : Qubit, 
                    root_comm : Qubit, 
                    rec_comm : Qubit, 
                    p_root : int) -> None:
    
        cbit = self.creg_manager.allocate_cbit() # Choose a classical bit for measurement

        instr = self.build_root_entanglement_circuit() # Build the cat-entanglement sub-circuit
        self.qc.append(instr, 
                    [root_id, root_comm, rec_comm], 
                    [cbit]) # Add the sub-circuit to the quantum circuit

        self.creg_manager.release_cbit(cbit) # Release the classical bit
        self.comm_manager.release_comm_qubit(p_root, 
                                            root_comm) # Release the communication qubit

    def entangle_root_local(self,
                            root_id : Qubit,
                            comm_id : Qubit) -> None:
        
        cbit = self.creg_manager.allocate_cbit()
        
        self.qc.cx(root_id)
        
        self.creg_manager.release_cbit(cbit)

    def end_entanglement_link(self,
                            linked_comm : Qubit, 
                            p_rec : int) -> None:
        
        cbit = self.creg_manager.allocate_cbit() # Choose a classical bit for measurement
        
        q_root = self.comm_manager.linked_qubits[linked_comm] # Find the root qubit (logical id)
        root_qubit_id = self.qubit_manager.log_to_phys_idx[q_root] # Find the physical qubit id

        instr = self.build_end_entanglement_circuit() # Build the end-entanglement sub-circuit
        self.qc.append(instr, [root_qubit_id, linked_comm], [cbit]) # Add the sub-circuit to the quantum circuit

        del self.comm_manager.linked_qubits[linked_comm] # Remove the linked logical qubit from the communication manager

        self.creg_manager.release_cbit(cbit)  # Release the classical bit
        self.comm_manager.release_comm_qubit(p_rec, linked_comm) # Release the communication qubit

    def extract_cycles_and_edges(self, G : nx.MultiDiGraph) -> tuple[list[tuple], dict[tuple, list[tuple]], list[tuple]]:
        """ 
        Determine cycles in partition assignment for ordering simultaenous teleportations.
        """
        cycles = []
        for cycle_nodes in nx.simple_cycles(G):
            cycles.append(tuple(cycle_nodes))

        all_cycle_edges = []
        cycle_edges = {}
        for cycle_nodes in cycles:
            cycle_edges[cycle_nodes] = []
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i+1) % len(cycle_nodes)]
                for qubit in G.adj[u][v]:
                    break

                cycle_edges[cycle_nodes].append(((u, v, qubit)))
                all_cycle_edges.append((u, v,qubit))
        
        G.remove_edges_from(all_cycle_edges)
        remaining_edges = [edge for edge in G.edges]
        
        return cycles, cycle_edges, remaining_edges

    def get_teleport_cycles(self,
                            assignment1 : list, 
                            assignment2 : list,
                            num_partitions : int,
                            num_qubits : int) -> tuple[list[list[int]], list[list[tuple]]]:
        """
        Identify cycles in the partition assignment for ordering simultaneous teleportations using directed multigraph.
        """
        graph = nx.MultiDiGraph()
        for p in range(num_partitions):
            graph.add_node(p)

        for q in range(num_qubits): # Query whether qubit has moved partition
            p1 = assignment1[q]
            p2 = assignment2[q]
            if p1 != p2:
                graph.add_edge(p1, p2, key = q, label=q) # Add edge to graph if qubit has moved partition

        _, cycle_edges, edges = self.extract_cycles_and_edges(graph) # Extract cycles and edges from the graph
        
        qubit_lists = []
        directions_lists = []
        for cycle in cycle_edges:
            qubits = []
            directions = []
            for edge in cycle_edges[cycle]:
                qubits.append(edge[2])
                directions.append((edge[0],edge[1]))

            qubit_lists.append(qubits)
            directions_lists.append(directions)

        qubits = []
        directions = []

        for edge in edges:
            qubits.append(edge[2])
            directions.append((edge[0],edge[1]))
        
        qubit_lists.append(qubits)
        directions_lists.append(directions)

        return qubit_lists, directions_lists

    def swap_qubits_to_phsyical(self, 
                                data_locations : list[tuple[Qubit, int, Qubit]]) -> None:
        """
        Swap qubits to their physical locations after teleportation.
        """
        for qubit, partition, data_loc in data_locations:

            slot = self.qubit_manager.allocate_data_qubit_slot(partition)
            data_q = self.qubit_manager.partition_qregs[partition][slot]

            self.qc.swap(data_loc, data_q)
            self.qc.reset(data_loc)

            self.qubit_manager.assign_to_physical(partition, slot, qubit)
            self.comm_manager.release_comm_qubit(partition, data_loc)
  
    def teleport_qubits(self,
                        old_assignment : list[int], 
                        new_assignment : list[int],
                        num_partitions : int,
                        num_qubits : int) -> None:
        """
        Teleport all qubits in q_list according to directions in direc_list. This handles teleportation swaps and cycles by holding logical states on
        data qubits until all teleportations are complete. States are then swapped onto data qubits.
        """
        q_list, direc_list = self.get_teleport_cycles(old_assignment, 
                                                      new_assignment, 
                                                      num_partitions, 
                                                      num_qubits)

         # Track the new locations of the qubits after teleportation
        for j, cycle in enumerate(q_list):
            data_locations = []
            directions = direc_list[j]
            for i, q in enumerate(cycle):

                p_source = directions[i][0] # Source partition
                p_dest = directions[i][1] # Destination partition

                data_q1 = self.qubit_manager.log_to_phys_idx[q]

                # Allocate comm qubits in each partition
                comm_source = self.comm_manager.find_comm_idx(p_source)
                comm_dest = self.comm_manager.find_comm_idx(p_dest)

                cbit1 = self.creg_manager.allocate_cbit()
                cbit2 = self.creg_manager.allocate_cbit()

                instr = self.build_teleporation_circuit()
                    
                self.qc.append(instr, [data_q1,comm_source, comm_dest], [cbit1, cbit2])

                self.comm_manager.release_comm_qubit(p_source, comm_source)
                self.qubit_manager.release_data_qubit(p_source, data_q1)

                self.creg_manager.release_cbit(cbit1)
                self.creg_manager.release_cbit(cbit2)

                data_locations.append((q, p_dest, comm_dest))
        
            self.swap_qubits_to_phsyical(data_locations)
        
        return

    def gate_teleport(self,
                      root_q : int, 
                      rec_q : int, 
                      gate : dict, 
                      p_root : int, 
                      p_rec : int) -> None:
        """
        Teleport single controlled unitary gate rooted on root_q in p_root onto rec_q in p_rec.
        """
        comm_root, comm_rec_set = self.comm_manager.allocate_comm_qubits(root_q, 
                                                                            p_root, 
                                                                            [p_rec])
        for _, comm_rec in comm_rec_set.items():
            break # Extract the receiver communication qubit
        
        gate_params = gate['params'] # Extract gate parameters         

        data_q_root = self.qubit_manager.log_to_phys_idx[root_q] # Find the physical qubit id for the root qubit
        data_q_rec = self.qubit_manager.log_to_phys_idx[rec_q] # Find the physical qubit id for the receiving qubit

        cbit1 = self.creg_manager.allocate_cbit()
       
        instr = self.build_gate_teleportation_circuit(gate_params)
        self.qc.append(instr, [data_q_root,  comm_root, comm_rec, data_q_rec], [cbit1])

        self.comm_manager.release_comm_qubit(p_root, comm_root)
        self.comm_manager.release_comm_qubit(p_rec, comm_rec)

        self.creg_manager.release_cbit(cbit1)

class PartitionedCircuitExtractor:

    def __init__(self, 
                graph : QuantumCircuitHyperGraph,
                partition_assignment : list[list[int]],
                qpu_info : list[int],
                comm_info : list[int]) -> None:
        
        self.layer_dict = graph.layers # Extract the dictionary containing the gates for each layer from the hypergraph
        self.partition_assignment = partition_assignment # Store the partition assignment for each qubit at each layer
        self.layer_dict = self.remove_empty_groups()
        self.layer_dict = self.ungroup_local_gates_commute()

        self.num_qubits = graph.num_qubits # Store the total number of qubits
        self.qpu_info = qpu_info # Store the number of data qubits in each partition
        self.comm_info = comm_info  # Store the number of communication qubits in each partition

        self.num_partitions = len(qpu_info) # Extract and store the number of partitions
        self.partition_qregs = self.create_data_qregs() # Create and store the data qubit registers
        self.comm_qregs = self.create_comm_qregs() # Create and store the communication qubit registers
        self.creg, self.result_reg = self.create_classical_registers() # Create and store the classical registers (creg for communication and result_reg for final measurement)
        self.qc = self.build_initial_circuit() # Build the initial circuit with all registers

        self.qubit_manager = DataQubitManager(self.partition_qregs, self.num_qubits, self.partition_assignment, self.qc) # Create the data qubit manager
        self.comm_manager = CommunicationQubitManager(self.comm_qregs, self.qc) # Create the communication qubit manager
        self.creg_manager = ClassicalBitManager(self.qc, self.creg) # Create the classical bit manager

        self.teleportation_manager = TeleportationManager(self.qc, self.qubit_manager, self.comm_manager, self.creg_manager) # Create the teleportation manager

        self.current_assignment = self.partition_assignment[0] # Track the partition assignment for each qubit at the "previous" layer

    def remove_empty_groups(self,) -> dict[int, list[dict]]:
        new_layers = copy.deepcopy(self.layer_dict)
        for i, layer in new_layers.items():
            for k, gate in enumerate(layer):
                if gate['type'] == 'group':
                    if len(gate['sub-gates']) == 1:

                        new_gate = gate['sub-gates'].pop(0)

                        t = new_gate['time'] 
                        del new_gate['time']

                        new_layers[t].append(new_gate)

                        layer.pop(k)
                    elif len(gate['sub-gates']) == 0:
                        layer.pop(k)
        return new_layers
    
    def ungroup_local_gates(self,) -> dict[int, list[dict]]:
        new_layers = copy.deepcopy(self.layer_dict)
        for t, layer in new_layers.items():
            for gate in layer:
                if gate['type'] == 'group':
                    root = gate['root']
                    sub_gates = gate['sub-gates']
                    gates_to_remove = []
                    for i, sub_gate in enumerate(sub_gates):
                        if self.partition_assignment[t][root] == self.partition_assignment[t][sub_gate['qargs'][1]]:
                            time = sub_gate['time']
                            del sub_gate['time']
                            new_layers[time].append(sub_gate)
                            gates_to_remove.append(sub_gate)
                    for i in gates_to_remove:
                        sub_gates.remove(i)

        return new_layers

    def ungroup_local_gates_commute(self,) -> dict[int, list[dict]]:

        layers = copy.deepcopy(self.layer_dict)
        for t, layer in layers.items():
            for gate in layer:
                if gate['type'] == 'group':
                    
                    root = gate['root']
                    sub_gates = gate['sub-gates']
                    if len(sub_gates) != 0:
                        gates_to_remove = []
                        last_t = sub_gates[-1]['time']
                        root_parts = {}

                        for j in range(t,last_t+1):
                            part = self.partition_assignment[j][root]
                            if part not in root_parts:
                                root_parts[part] = [j]
                            else:
                                root_parts[part].append(j)

                        for i, sub_gate in enumerate(sub_gates):
                            sub_gate_part = self.partition_assignment[t][sub_gate['qargs'][1]]
                            if sub_gate_part in root_parts:
                                time = sub_gate['time']
                                if sub_gate_part == self.partition_assignment[t][root]:
                                    local_time = time
                                else:
                                    local_time = root_parts[sub_gate_part][0]
                                
                                del sub_gate['time']
                                layers[local_time].append(sub_gate)
                                gates_to_remove.append(sub_gate)
                        for i in gates_to_remove:
                            sub_gates.remove(i)
        return layers

    def create_data_qregs(self,) -> list[QuantumRegister]:
        partition_qregs = []
        for i in range(self.num_partitions):
            size_i = self.qpu_info[i]
            qr = QuantumRegister(size_i, name=f"part{i}_data")
            partition_qregs.append(qr)
        return partition_qregs
    
    def create_comm_qregs(self,) -> dict[int, list[QuantumRegister]]:
        comm_qregs = {}
        for i in range(self.num_partitions):
            comm_qregs[i] = [QuantumRegister(self.comm_info[i], name=f"comm_{i}_{0}")] # Each partition has a list of comm registers in case more must be added
        return comm_qregs
    
    def create_classical_registers(self,) -> tuple[ClassicalRegister, ClassicalRegister]:
        creg = ClassicalRegister(2, name="c")
        result_reg = ClassicalRegister(self.num_qubits, name="result")
        return creg, result_reg
    
    def build_initial_circuit(self,) -> QuantumCircuit:
        comm_regs_all = [part[0] for part in self.comm_qregs.values()] # Use only the initial communication qubit registers
        qc = QuantumCircuit(
            *self.partition_qregs,
            *comm_regs_all,
            *[self.creg, self.result_reg],
            name="PartitionedCircuit"
        )
        return qc
    
    def apply_single_qubit_gate(self, gate : dict) -> None:

        q = gate['qargs'][0]
        params = gate['params']
        qubit_phys = self.qubit_manager.log_to_phys_idx[q]
        self.qc.u(*params, qubit_phys)
    
    def apply_local_two_qubit_gate(self, gate : dict) -> None:

        q0, q1 = gate['qargs']
        params = gate['params']
        qubit0 = self.qubit_manager.log_to_phys_idx[q0]
        qubit1 = self.qubit_manager.log_to_phys_idx[q1]
        self.qc.cp(params[0], qubit0, qubit1)
    
    def process_group_gate(self, gate, t : int) -> None:

        root_qubit = gate['root']
        p_root = self.current_assignment[root_qubit]
        sub_gates = gate['sub-gates']
        p_rec_set = set()
        final_gates = {}
        if sub_gates != []:
            for sub_gate in sub_gates:

                q0, q1 = sub_gate['qargs']
                time_step = sub_gate['time']
                p_rec = self.current_assignment[q1]

                if p_rec != p_root:
                    p_rec_set.add(p_rec)
                if p_rec not in final_gates:
                    final_gates[p_rec] = time_step
                else:
                    final_gates[p_rec] = max(final_gates[p_rec], time_step)


            root_qubit_phys = self.qubit_manager.log_to_phys_idx[root_qubit]
            comm_root, comm_rec_dict = self.comm_manager.allocate_comm_qubits(root_qubit, p_root, p_rec_set)

            for p_rec, comm_rec in comm_rec_dict.items():
                self.teleportation_manager.entangle_root(root_qubit_phys, comm_root, comm_rec, p_root)
            # Insert "linked" gates at the appropriate layer
            for sub_gate in sub_gates:
                q0, q1 = sub_gate['qargs']
                p1 = self.current_assignment[q1]
                time_step = sub_gate['time']
                if p1 == p_root:
                    # Same partition as root
                    new_gate = {'type': 'two-qubit', 'qargs': [q0, q1], 'params': sub_gate['params']}
                    if time_step == t:
                        qubit0 = self.qubit_manager.log_to_phys_idx[q0]
                        qubit1 = self.qubit_manager.log_to_phys_idx[q1]
                        self.qc.cp(sub_gate['params'][0], qubit0, qubit1)
                    self.layer_dict[time_step].append(new_gate)
                else:
                    # Different partition -> use the linked comm qubit
                    linked_root = comm_rec_dict[p1]
                    end = (time_step == final_gates[p1])
                    new_gate = {
                        'type': 'two-qubit-linked',
                        'qargs': [linked_root, q1],
                        'params': sub_gate['params'],
                        'end': end
                    }
                    if time_step == t:
                        qubit1 = self.qubit_manager.log_to_phys_idx[q1]
                        self.qc.cp(sub_gate['params'][0], linked_root, qubit1)
                        if end:
                            self.teleportation_manager.end_entanglement_link(linked_root, p1)
                    else:
                        self.layer_dict[time_step].append(new_gate)

    def extract_partitioned_circuit(self, ) -> QuantumCircuit:
        """
        Build a Qiskit circuit from a list of operations mapped with qubits mapped according to the
        partition assignment. Teleportation instructions (state or gate teleportation)
        inserted for state and gate teleportaitons.
        """
        for i, layer in self.layer_dict.items():
            new_assignment_layer = self.partition_assignment[i]
            # Teleport data qubits that moved partitions at layer i
            for q in range(self.num_qubits):
                if self.current_assignment[q] != new_assignment_layer[q]:
                    self.teleportation_manager.teleport_qubits(self.current_assignment, 
                                                               new_assignment_layer, 
                                                               self.num_partitions, 
                                                               self.num_qubits)
                    break

            self.current_assignment = new_assignment_layer

            for gate in layer:
                if gate['type'] == "single-qubit":
                    self.apply_single_qubit_gate(gate)
                    
                elif gate['type'] == "two-qubit":
                    
                    q0, q1 = gate['qargs']
                    p0 = self.current_assignment[q0]
                    p1 = self.current_assignment[q1]
                    if p0 == p1:
                        self.apply_local_two_qubit_gate(gate)
                    else:
                        self.teleportation_manager.gate_teleport(q0, q1, gate, p0, p1)

                elif gate['type'] == "group":
                    self.process_group_gate(gate, i)

                elif gate['type'] == "two-qubit-linked":
                    q0, q1 = gate['qargs']
                    linked_root = q0
                    p1 = self.current_assignment[q1]
                    params = gate['params']
                    qubit1 = self.qubit_manager.log_to_phys_idx[q1]
                    self.qc.cp(params[0], linked_root, qubit1)

                    if gate['end']:
                        self.teleportation_manager.end_entanglement_link(linked_root, p1)

        for i in range(self.num_qubits):
            self.qc.measure(self.qubit_manager.log_to_phys_idx[i], self.result_reg[i])

        return self.qc
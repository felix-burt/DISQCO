import copy
import numpy as np
import networkx as nx
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Qubit, Instruction
from disqco import QuantumCircuitHyperGraph
from disqco import QuantumNetwork
from disqco.circuit_extraction.DQC_qubit_manager import DataQubitManager, CommunicationQubitManager, ClassicalBitManager
import math as mt

# -------------------------------------------------------------------
# TeleportationManager
# -------------------------------------------------------------------

class TeleportationManager:
    """
    This class is responsible for managing the teleportation of qubits between partitions. This is done by building state and gate teleportation primitives
    using the starting and ending teleportation primitives.
    params:
        qc: QuantumCircuit - The quantum circuit to append the teleportation circuits to.
        qubit_manager: DataQubitManager - The qubit manager to use to allocate and release qubits.
        comm_manager: CommunicationQubitManager - The communication qubit manager to use to allocate and release communication qubits.
        creg_manager: ClassicalBitManager - The classical bit manager to use to allocate and release classical bits.
    """

    def __init__(
        self,
        qc: QuantumCircuit, 
        hypergraph: QuantumCircuitHyperGraph,
        network: QuantumNetwork,
        qubit_manager: DataQubitManager, 
        comm_manager: CommunicationQubitManager, 
        creg_manager: ClassicalBitManager
    ) -> None:
        
        self.qc = qc
        self.network = network
        self.qubit_manager = qubit_manager
        self.comm_manager = comm_manager
        self.creg_manager = creg_manager
        self.hypergraph = hypergraph

    def build_state_transfer_circuit(self) -> Instruction:
        """
        Builds the state transfer circuit. This swaps the state of a qubit onto an unused qubit slot.
        """
        circ = QuantumCircuit(2, 1)
        circ.cx(0,1)
        circ.h(0)
        circ.measure(0, 0)
        circ.reset(0)
        circ.z(1).c_if(0, 1)
        return circ.to_instruction()

    def build_epr_circuit(self) -> Instruction:
        """
        Builds the EPR circuit. This is used to entangle two qubits - is kept as a primitive operation
        since it is expected to be handled by the network.
        """
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        gate = circ.to_gate()
        gate.name = "EPR"
        return gate
    
    def build_starting_process_circuit(self) -> Instruction:
        """
        Builds the teleportation starting process circuit (cat-entangler/non-local fan out). This is used 
        to entangle the root qubit with a communication qubit in another QPU.
        """
        epr_circ = self.build_epr_circuit()
        circ = QuantumCircuit(3, 1)
        circ.append(epr_circ, [1, 2])
        circ.cx(0, 1)
        circ.measure(1, 0)
        circ.reset(1)
        circ.x(2).c_if(0, 1)

        instr = circ.to_instruction()
        instr.name = "Starting Process"
        return instr
    
    def build_ending_process_circuit(self) -> Instruction:
        """
        Builds the teleportation ending process circuit (cat-entangler/non-local fan in). This is used 
        to disentangle the root qubit from a communication qubit in another QPU, direction determines
        the final location of the root qubit.
        """
        circ = QuantumCircuit(2, 1)
        circ.h(1)
        circ.measure(1, 0)
        circ.reset(1)
        circ.z(0).c_if(0, 1)

        instr = circ.to_instruction()
        instr.name = "Ending Process"
        return instr
    
    def build_teleporation_circuit(self) -> Instruction:
        """
        Builds the state teleportation circuit. This is used to teleport the state of a qubit from one
        partition (QPU) to another.
        """
        circ = QuantumCircuit(3, 2)
        starting = self.build_starting_process_circuit()
        circ.append(starting, [0, 1, 2], [0])
        ending = self.build_ending_process_circuit()
        circ.append(ending, [2, 0], [1])

        instr = circ.to_instruction(label="State Teleportation")
        return instr
    
    def build_gate_teleportation_circuit(self, gate) -> Instruction:
        """
        Builds the gate teleportation circuit. This is used to teleport cover a single non-local gate,
        ending the link immediately afterwards.
        """
        gate_params = gate['params']
        name = gate['name']
        circ = QuantumCircuit(4, 1)
        root_entanglement_circuit = self.build_starting_process_circuit()
        circ.append(root_entanglement_circuit, [0, 1, 2], [0])
        if name == 'cp':
            circ.cp(gate_params[0], 2, 3)
        elif name == 'cx':
            circ.cx(2, 3)
        elif name == 'cz':
            circ.cz(2, 3)

        entanglement_end_circuit = self.build_ending_process_circuit()
        circ.append(entanglement_end_circuit, [0, 2], [0])

        instr = circ.to_instruction(label="Gate Teleportation")
        return instr
    
    def transfer_state(self, q1: Qubit, q2: Qubit) -> None:
        """
        Transfers the state of a qubit from one qubit slot to an unused slot.
        """
        cbit = self.creg_manager.allocate_cbit()
        instr = self.build_state_transfer_circuit()
        self.qc.append(instr, [q1, q2], [cbit])
        self.creg_manager.release_cbit(cbit)
    
    def entangle_root(self, root_idx: int, p_root: int, p_rec: int) -> None:
        """
        Entangles the root qubit with a communication qubit in another QPU using the starting process circuit.
        """
        root_q = self.qubit_manager.log_to_phys_idx[root_idx]
        root_comm = self.comm_manager.find_comm_idx(p_root)
        rec_comm = self.comm_manager.find_comm_idx(p_rec)
        cbit = self.creg_manager.allocate_cbit()
        instr = self.build_starting_process_circuit()
        self.qc.append(instr, [root_q, root_comm, rec_comm], [cbit])
        self.qubit_manager.groups[root_idx]['linked_qubits'][p_rec] = rec_comm
        self.creg_manager.release_cbit(cbit)
        self.comm_manager.release_comm_qubit(p_root, root_comm)

    def end_entanglement_link(self, q_root : int, p_root: int, p_rec: int, p_target : int) -> None:
        """
        Disentangles the root qubit from a communication qubit in another QPU using the ending process circuit.
        """
        if p_rec == p_target:
            return
        root_q = self.qubit_manager.log_to_phys_idx[q_root]
        rec_comm = self.qubit_manager.groups[q_root]['linked_qubits'][p_rec]
        if p_root == p_target:
            target_comm = root_q
        else:
            target_comm = self.qubit_manager.groups[q_root]['linked_qubits'][p_target]

        cbit = self.creg_manager.allocate_cbit()
        instr = self.build_ending_process_circuit()
        self.qc.append(instr, [target_comm, rec_comm], [cbit])
        if rec_comm._register.name[0] == 'C':
            self.comm_manager.release_comm_qubit(p_rec, rec_comm)
        else:
            self.qubit_manager.release_data_qubit(p_rec, rec_comm)
        self.creg_manager.release_cbit(cbit)
    
    def close_group(self, root_idx: int) -> None:
        """
        Disentangles the root qubit from all linked communication qubits in the group, and updates the root qubit's
        physical location to the chosen final location.
        """

        group_info = self.qubit_manager.groups[root_idx]
        p_root_init = group_info['init_p_root']
        final_p_root = group_info['final_p_root']
        linked_qubits = group_info['linked_qubits']


        for p, linked_comm in linked_qubits.items():
            # self.end_entanglement_link(root_idx, p_root_init, p, final_p_root)
            if p == final_p_root and p != p_root_init:

                if linked_comm._register.name[0] == 'C':
                    try:
                        data_qubit = self.qubit_manager.allocate_data_qubit(p)
                    except Exception as e:

                        print(f"Failed to allocate data qubit in partition {p} for root {root_idx}: {e}")
                        print(f'All groups: {self.qubit_manager.groups}')
                        raise e
                    self.transfer_state(linked_comm, data_qubit)
                    self.qubit_manager.assign_to_physical(p, data_qubit, root_idx)
                    self.comm_manager.release_comm_qubit(p, linked_comm)
            else:
                if p == p_root_init and p == final_p_root:
                    continue

                self.end_entanglement_link(root_idx, p_root_init, p, final_p_root)
                self.comm_manager.release_comm_qubit(p, linked_comm)

        del self.qubit_manager.groups[root_idx]
        
    
    def space_count(self,):
        """
        Counts the number of unused qubit slots in each partition. Used to determine the order of teleportation.
        """
        space_counts = [len(value) for value in self.qubit_manager.free_data.values()]
        return space_counts

    def choose_qubit(self, graph, space_counts):
        """
        Chooses a qubit to teleport based on the number of unused qubit slots in each partition.
        """
        for p, space_p in enumerate(space_counts):
            if space_p > 0:
                edges_in = graph.in_edges(p)
                if len(edges_in) == 0:
                    continue
                for edge in edges_in:
                    break
                qubits = graph.get_edge_data(*edge)
                for qubit in qubits:
                    break
                graph.remove_edge(edge[0], edge[1], key=qubit)
                space_counts[edge[0]] += 1
                space_counts[edge[1]] -= 1
                return qubit, edge[0], edge[1]
        return None, None, None

    def get_teleportation_order(self, assignment1: list, assignment2: list,
                            num_partitions: int, num_qubits: int) -> list[dict[str, int]]:
        """
        Function to determine the order of teleportations transitioning between two assignments.
        """
        graph = nx.MultiDiGraph()
        
        for p in range(num_partitions):
            graph.add_node(p)

        for q in range(num_qubits):
            if q in self.qubit_manager.groups:
                continue
            p1 = assignment1[q]
            p2 = assignment2[q]
            if p1 != p2:
                graph.add_edge(p1, p2, key=q)

        teleportation_order = []

        space_counts = self.space_count()
        while True:
            qubit, source, destination = self.choose_qubit(graph, space_counts)
            if qubit is None:
                break
            teleportation_order.append({'qubit': qubit, 'source': source, 'destination': destination})
        if len(graph.edges()) == 0:
            return teleportation_order
        else:
            cycles = nx.simple_cycles(graph)
            # There are cycles to be handled.
            for cycle in cycles:
                try:
                    edges = nx.find_cycle(graph, cycle)
                except nx.NetworkXNoCycle:
                    continue
                for (source, destination, qubit) in edges:
                    teleportation_order.append({'qubit': qubit, 'source': source, 'destination': destination})
                    graph.remove_edge(source, destination, key=qubit)
                    # space_counts[source] += 1
                    # space_counts[destination] -= 1
            return teleportation_order

    def swap_qubits_to_physical(self, qubit_idx : int, partition : int, data_loc : Qubit) -> bool:
        try:
            data_q = self.qubit_manager.allocate_data_qubit(partition)
            self.transfer_state(data_loc, data_q)
            self.qubit_manager.assign_to_physical(partition, data_q, qubit_idx)
            self.comm_manager.release_comm_qubit(partition, data_loc)
            return True

        except Exception as e:
            return False
            # self.qubit_manager.queue[qubit] = (data_loc, partition)
            # self.qubit_manager.log_to_phys_idx[qubit] = data_loc
            # if data_loc not in self.comm_manager.in_use_comm[partition]:
            #     self.comm_manager.in_use_comm[partition].add(data_loc)
            # if data_loc in self.comm_manager.free_comm[partition]:
            #     self.comm_manager.free_comm[partition].remove(data_loc)

    def teleport_qubits(self, old_assignment: list[int], new_assignment: list[int],
                        num_partitions: int, num_qubits: int) -> None:
        """
        Teleports qubits to transition between two assignments using the network tree path.
        For each teleportation, applies k-fold starting/ending processes along the path, and swaps qubits at the destination.
        """
        old_assignment = [int(x) for x in old_assignment]
        new_assignment = [int(x) for x in new_assignment]
        num_partitions = int(num_partitions)
        num_qubits = int(num_qubits)
        teleportation_order = self.get_teleportation_order(old_assignment, new_assignment,
                                                          num_partitions, num_qubits)
        remaining_swaps = []
        for teleportation in teleportation_order:
            qubit_idx = int(teleportation['qubit'])
            data_q1 = self.qubit_manager.log_to_phys_idx[qubit_idx]
            p_source = int(teleportation['source'])
            p_dest = int(teleportation['destination'])

            # Apply k-fold starting process along the path
            comm_qubits = self.entangle_root_on_tree(
                            root_q=qubit_idx,
                            target_partitions=[p_dest],
                            p_root=p_source,
                            num_partitions= num_partitions,
                            group_gate=False)
            
            # Swap qubit to physical at destination
            comm_dest = comm_qubits[p_dest]
            # Apply ending process circuit at destination (measure destination comm qubit, teleport state)
            cbit = self.creg_manager.allocate_cbit()
            ending_instr = self.build_ending_process_circuit()
            # After swap, the data qubit is at the destination
            data_q1 = self.qubit_manager.log_to_phys_idx[qubit_idx]
            self.qc.append(ending_instr, [comm_dest, data_q1], [cbit])
            self.qubit_manager.release_data_qubit(p_source, qubit=data_q1)
            self.creg_manager.release_cbit(cbit)
            success = self.swap_qubits_to_physical(qubit_idx, p_dest, comm_dest)
            if not success:
                remaining_swaps.append((qubit_idx, p_dest, comm_dest))

        while len(remaining_swaps) > 0:
            qubit_idx, p_dest, comm_dest = remaining_swaps.pop(0)
            success = self.swap_qubits_to_physical(int(qubit_idx), int(p_dest), comm_dest)
            if not success:
                remaining_swaps.append((qubit_idx, p_dest, comm_dest))

    def gate_teleport(self, root_q: int, rec_q: int, gate: dict, p_root: int, p_rec: int) -> None:
        """
        Performs a single non-local two-qubit gate using a gate teleportation procedure along the network tree path.
        Applies k-fold starting/ending processes, performs the gate locally at the target, then applies ending process.
        """

        # Apply k-fold starting process along the path
        comm_qubits = self.entangle_root_on_tree(
            root_q=root_q, 
            target_partitions=[p_rec], 
            p_root=p_root, 
            num_partitions=len(self.network.qpu_sizes),
            group_gate=False
        )

        comm_q_rec = comm_qubits[p_rec]
        # Perform the gate locally at the target
        data_q_root = self.qubit_manager.log_to_phys_idx[root_q]
        data_q_rec = self.qubit_manager.log_to_phys_idx[rec_q]

        name = gate['name']
        params = gate['params']
        if name == 'cp':
            self.qc.cp(params[0], comm_q_rec, data_q_rec)
        elif name == 'cx':
            self.qc.cx(comm_q_rec, data_q_rec)
        elif name == 'cz':
            self.qc.cz(comm_q_rec, data_q_rec)
        # Apply ending process at the target (measure target qubit)

        cbit = self.creg_manager.allocate_cbit()
        ending_instr = self.build_ending_process_circuit()
        self.qc.append(ending_instr, [data_q_root, comm_q_rec], [cbit])
        self.comm_manager.release_comm_qubit(p_rec, comm_q_rec)
        self.creg_manager.release_cbit(cbit)
    
    def entangle_root_on_tree(self, 
                            root_q : int, 
                            target_partitions : list[int], 
                            p_root : int, 
                            num_partitions : int,
                            group_gate=False) -> None:
        # Get the undirected tree
        undirected_tree = self.network.get_full_tree(root_p = p_root, 
                                                    target_partitions=target_partitions)
        # Convert to directed graph rooted at p_root
        if undirected_tree:
            directed_tree = nx.DiGraph()
            # BFS traversal from root
            from collections import deque
            visited = set()
            queue = deque([p_root])
            while queue:
                parent = queue.popleft()
                visited.add(parent)
                for child in undirected_tree.neighbors(parent):
                    if child not in visited:
                        directed_tree.add_edge(parent, child)
                        queue.append(child)
            # Now build the entanglement circuit using the directed tree
            node_in_comm = self.build_k_fold_starting_process(root_q, p_root, target_partitions, directed_tree)

            if group_gate:
                for p in node_in_comm:
                    comm_qubit = node_in_comm[p]
                    self.qubit_manager.groups[root_q]['linked_qubits'][p] = comm_qubit
            return node_in_comm

    def build_k_fold_starting_process(self, root_q: int, p_root: int, target_partitions: list[int], tree: nx.DiGraph) -> None:
        """
        Builds the starting process for the k-fold circuit extraction.
        Generates EPR pairs, applies CNOTs, measures, and applies classical corrections along the tree.
        Handles branching and sums measurement results modulo 2 along each branch.
        """
        # Generate EPR pairs for all edges
        edges_to_comms = {}
        for p0, p1 in tree.edges():
            comm0 = self.comm_manager.find_comm_idx(p0)
            comm1 = self.comm_manager.find_comm_idx(p1)
            epr = self.build_epr_circuit()
            self.qc.append(epr, [comm0, comm1])
            edges_to_comms[(p0, p1)] = (comm0, comm1)

        from collections import deque
        root_q_phys = self.qubit_manager.log_to_phys_idx[root_q]
        # For each node, store the path from root and the classical bits for measurements
        node_paths = {p_root: []}
        node_cbits = {p_root: []}
        # For each node, store the incoming EPR half (from parent)
        node_in_comm = {p_root: root_q_phys}
        queue = deque([p_root])

        correction_info = []  # List of tuples: (comm_child, node_cbits[child])
        while queue:
            current = queue.popleft()
            children = list(tree.successors(current))
            in_qubit = node_in_comm[current]
            for child in children:
                comm_current, comm_child = edges_to_comms[(current, child)]
                self.qc.cx(in_qubit, comm_current)
                cbit = self.creg_manager.allocate_cbit()
                self.qc.measure(comm_current, cbit)
                self.qc.reset(comm_current)
                node_paths[child] = node_paths[current] + [child]
                node_cbits[child] = node_cbits[current] + [cbit]
                node_in_comm[child] = comm_child
                correction_info.append((comm_child, list(node_cbits[child]), cbit))
                queue.append(child)
                # Release comm qubit from parent (current)
                self.comm_manager.release_comm_qubit(current, comm_current)

        for comm_child, cbits, last_cbit in correction_info:
            for cb in cbits:
                self.qc.x(comm_child).c_if(cb, 1)
            # Release classical bits after use
            self.creg_manager.release_cbit(last_cbit)
        all_nodes = set(tree.nodes())
        target_set = set(target_partitions)
        aux_nodes = [n for n in all_nodes if n not in target_set.union({p_root})]
        for aux in aux_nodes:
            # Find nearest target partition in tree
            min_path = None
            min_target = None
            for tgt in target_partitions + [p_root]:
                try:
                    path = nx.shortest_path(tree, source=aux, target=tgt)
                    if min_path is None or len(path) < len(min_path):
                        min_path = path
                        min_target = tgt
                except nx.NetworkXNoPath:
                    continue
            if min_path is None:
                continue  # No reachable target partition

            local_epr = node_in_comm[aux]
            if min_target == p_root:
                live_epr = root_q_phys
            else:
                live_epr = node_in_comm[min_target]

            # Apply ending process circuit from local_epr to live_epr
            cbit = self.creg_manager.allocate_cbit()
            ending_instr = self.build_ending_process_circuit()
            self.qc.append(ending_instr, [live_epr, local_epr], [cbit])
            self.comm_manager.release_comm_qubit(aux, local_epr)
            self.creg_manager.release_cbit(cbit)
            del node_in_comm[aux]  

        return node_in_comm
class PartitionedCircuitExtractor:
    """
    This class is responsible for extracting the partitioned circuit from the quantum circuit hypergraph.
    The partition assignment is used to determine the state teleportations, while non-local hyper-edges
    are used to determine the starting and ending process circuits for gate teleportation.

    params:
        graph: QuantumCircuitHyperGraph - The quantum circuit hypergraph to extract the partitioned circuit from.
        partition_assignment: list[list[int]] - The partition assignment to use to determine the state teleportations.
        qpu_info: list[int] - The number of qubits in each partition.
        comm_info: list[int] - The number of communication qubits in each partition.

    """

    def __init__(
        self,
        graph: QuantumCircuitHyperGraph,
        network: QuantumNetwork,
        partition_assignment: np.ndarray
    ) -> None:
        
        # The gate edges of the graph stored as a list of gates and gate groups.
        self.layer_dict = graph.layers
        self.layer_dict = self.remove_empty_groups()
        # The partition assignment to use to determine the state teleportations.
        self.partition_assignment = partition_assignment.tolist()
        self.num_qubits = graph.num_qubits
        self.qpu_info = network.qpu_sizes
        self.comm_info = network.comm_sizes
        self.depth = graph.depth
        self.num_partitions = len(self.qpu_info)
        self.graph = graph
        self.network = network
        self.basis_gates = graph.basis_gates


        # Create the quantum registers for the data qubits and communication qubits.
        # Each QPU has a quantum register for the data qubits and a separate register for communication qubits.
        # Additional communication qubits can be allocated dynamically as needed.
        self.partition_qregs = self.create_data_qregs()
        self.comm_qregs = self.create_comm_qregs()
        # Create the classical registers for the result and control bits.
        self.creg, self.result_reg = self.create_classical_registers()
        self.qc = self.build_initial_circuit()

        # Create the qubit and classical bit managers.
        self.qubit_manager = DataQubitManager(self.partition_qregs, self.num_qubits,
                                              self.partition_assignment, self.qc)
        
        self.comm_manager = CommunicationQubitManager(self.comm_qregs, self.qc)
        self.creg_manager = ClassicalBitManager(self.qc, self.creg)

        # Create the teleportation manager.
        self.teleportation_manager = TeleportationManager(self.qc, self.graph, self.network, self.qubit_manager,
                                                          self.comm_manager, self.creg_manager)
        
        # The current assignment is the partition assignment for the current time step.
        self.current_assignment = self.partition_assignment[0]

    def remove_empty_groups(self) -> dict[int, list[dict]]:
        """
        Removes empty gate groups from the layer dictionary.
        """
        new_layers = copy.deepcopy(self.layer_dict)
        for i, layer in new_layers.items():
            for k, gate in enumerate(layer[:]):
                if gate['type'] == 'group':
                    if len(gate['sub-gates']) == 1:
                        new_gate = gate['sub-gates'].pop(0)
                        t = new_gate['time']
                        del new_gate['time']
                        new_layers[t].append(new_gate)
                        layer.remove(gate)
                    elif len(gate['sub-gates']) == 0:
                        layer.remove(gate)

        return new_layers
    
    def create_data_qregs(self) -> list[QuantumRegister]:
        """
        Creates the quantum registers for the data qubits.
        """
        partition_qregs = []
        for i in range(self.num_partitions):
            size_i = self.qpu_info[i]
            qr = QuantumRegister(size_i, name=f"Q{i}_q")
            partition_qregs.append(qr)
        return partition_qregs

    def create_comm_qregs(self) -> dict[int, list[QuantumRegister]]:
        """
        Creates the quantum registers for the communication qubits.
        """
        comm_qregs = {}
        for i in range(self.num_partitions):
            comm_qregs[i] = [QuantumRegister(self.comm_info[i], name=f"C{i}_{0}")]
        return comm_qregs

    def create_classical_registers(self) -> tuple[ClassicalRegister, ClassicalRegister]:
        """
        Creates the classical registers for the result and control bits.
        """
        creg = ClassicalRegister(self.num_qubits, name="cl")
        result_reg = ClassicalRegister(self.num_qubits, name="result")
        return creg, result_reg

    def build_initial_circuit(self) -> QuantumCircuit:
        """
        Builds the initial circuit.
        """
        comm_regs_all = [part[0] for part in self.comm_qregs.values()]
        qc = QuantumCircuit(
            *self.partition_qregs,
            *comm_regs_all,
            *[self.creg, self.result_reg],
            name="PartitionedCircuit"
        )
        return qc

    def apply_single_qubit_gate(self, gate: dict) -> None:
        """
        Applies a local single-qubit gate to the circuit.
        """
        q = gate['qargs'][0]
        params = gate['params']
        name = gate['name']
        qubit_phys = self.qubit_manager.log_to_phys_idx[q]
        if name == 'u' or name == 'u3':
            self.qc.u(params[0], params[1], params[2], qubit_phys)
        elif name == 'h':
            self.qc.h(qubit_phys)
        elif name == 'x':
            self.qc.x(qubit_phys)
        elif name == 'y':
            self.qc.y(qubit_phys)
        elif name == 'z':
            self.qc.z(qubit_phys)
        elif name == 's':
            self.qc.s(qubit_phys)
        elif name == 'sdg':
            self.qc.sdg(qubit_phys)
        elif name == 't':
            self.qc.t(qubit_phys)
        elif name == 'tdg':
            self.qc.tdg(qubit_phys)
        elif name == 'rz':
            self.qc.rz(params[0], qubit_phys)

    def apply_local_two_qubit_gate(self, gate: dict) -> None:
        """
        Applies a local two-qubit gate to the circuit.
        """
        qubit0, qubit1 = gate['qargs']
        params = gate['params']
        name = gate['name']
        
        if isinstance(qubit0, int):
            qubit0 = self.qubit_manager.log_to_phys_idx[qubit0]
        if isinstance(qubit1, int):
            qubit1 = self.qubit_manager.log_to_phys_idx[qubit1]

        if name == 'cx':
            self.qc.cx(qubit0, qubit1)
        elif name == 'cz':
            self.qc.cz(qubit0, qubit1)
        elif name == 'cp':
            self.qc.cp(params[0], qubit0, qubit1)
        
    def check_qpus_local(self, qubit0, qubit1) -> bool:
        """
        Checks if all qubits in the current assignment are local to the same QPU.
        """
        if isinstance(qubit0, int):
            qubit0 : Qubit = self.qubit_manager.log_to_phys_idx[qubit0]
        if isinstance(qubit1, int):
            qubit1 : Qubit = self.qubit_manager.log_to_phys_idx[qubit1]

        # Find which register the qubits belong to.

        reg1 : QuantumRegister = qubit0._register
        reg2 : QuantumRegister = qubit1._register
        if reg1.name[1] == reg2.name[1]:
            return True
        else:
            return False
    
    def find_common_part(self, qubit0: int, qubit1: int) -> int:
        """
        Finds the common partition for two qubits.
        """
        if qubit1 in self.qubit_manager.groups:
            possible_partitions1 = set(self.qubit_manager.groups[qubit1]['linked_qubits'].keys())
        else:
            possible_partitions1 = set([self.current_assignment[qubit1]])

        if qubit0 in self.qubit_manager.groups:
            possible_partitions0 = set(self.qubit_manager.groups[qubit0]['linked_qubits'].keys())
        else:
            possible_partitions0 = set([self.current_assignment[qubit0]])

        choices = possible_partitions1.intersection(possible_partitions0)
        for p in choices:
            return p
        return -1

    def apply_non_local_two_qubit_gate(self, gate: dict, p_root: int, p1: int) -> None:
        root_q, q1 = gate['qargs']
        common_part = self.find_common_part(root_q, q1)

        if common_part == -1:
            common_part = p1
        if root_q in self.qubit_manager.groups and common_part in self.qubit_manager.groups[root_q]['linked_qubits']:
            root_q_mapped = self.qubit_manager.groups[root_q]['linked_qubits'][common_part]
        else:
            root_q_mapped = root_q
        
        if q1 in self.qubit_manager.groups and common_part in self.qubit_manager.groups[q1]['linked_qubits']:
            q1_mapped = self.qubit_manager.groups[q1]['linked_qubits'][common_part]
        else:
            q1_mapped = q1

        if not self.check_qpus_local(qubit0=root_q_mapped, qubit1=q1_mapped):
            print(f"Non-local two-qubit gate {gate} cannot be applied locally.")
            print("Root qubit:", root_q, "Q1 qubit:", q1)
            print("Mapped root qubit:", root_q_mapped, "Mapped Q1 qubit:", q1_mapped)
            print("Data qubit q1:", self.qubit_manager.log_to_phys_idx[q1])
            print("Current assignment:", self.current_assignment)
            print("Qubit manager groups:", self.qubit_manager.groups)
            raise ValueError(f"Non-local two-qubit gate {gate} cannot be applied locally.")
        
        gate['qargs'] = [root_q_mapped, q1_mapped]
        self.apply_local_two_qubit_gate(gate)

        if gate['time'] == self.qubit_manager.groups[root_q]['final_time']:
            try:
                self.teleportation_manager.close_group(root_q)
            except Exception as e:
                print(f"Error closing group {root_q}: {e}")
                print(f"Qubit manager groups: {self.qubit_manager.groups}")
                print(f"Gate: {gate}")
                print(f"Free data qubits: {self.qubit_manager.free_data}")
                print(f'In use data qubits: {self.qubit_manager.in_use_data}')
                print(f"Free comm qubits: {self.comm_manager.free_comm}")
                print(f'In use comm qubits: {self.comm_manager.in_use_comm}')
                current_assignment = self.current_assignment
                for qpu in self.qpu_info:
                    num_filled_slots = current_assignment.count(qpu)
                    # print(f'QPU {qpu} has {num_filled_slots} filled slots in current assignment')
                for qpu in self.qpu_info:
                    num_filled_slots = self.partition_assignment[gate['time']+1].count(qpu)
                #     print(f'QPU {qpu} has {num_filled_slots} filled slots in next assignment')
                # print(f'Current assignment: {self.current_assignment}')
                # print(f'Next assignment: {self.partition_assignment[gate['time']+1]}')
                # print(f'Assignment on root qubit: {self.partition_assignment[gate["time"]][root_q]}')
                # print(f'Next assignment on root qubit: {self.partition_assignment[gate["time"]+1][root_q]}')
                # print(f'Assignment on q1 qubit: {self.partition_assignment[gate["time"]][q1]}')
                # print(f'Next assignment on q1 qubit: {self.partition_assignment[gate["time"]+1][q1]}')
                # print(f'Current root data qubit: {self.qubit_manager.log_to_phys_idx[root_q]}')
                for i in range(len(self.current_assignment)):
                    print(f'Qubit {i} should be in partition {self.current_assignment[i]}')
                    data_i = self.qubit_manager.log_to_phys_idx[i]
                    print(f'Qubit {i} is on data qubit {data_i}')
                    data_i_reg = data_i._register
                    if int(data_i_reg.name[1]) == int(self.current_assignment[i]):
                        print(f'Qubit {i} is in the correct partition')
                    else:
                        print(f'Qubit {i} is in the wrong partition')

                raise e

            

    def check_diag_gate(self, gate):
        "Checks if a gate is diagonal or anti-diagonal"
        name = gate['name']
        if name == 'u' or name == 'u3':
            theta = gate['params'][0]
            if round(theta % mt.pi*2, 2) == round(0, 2):
                return 'diagonal'
            elif round(theta % mt.pi*2, 2) == round(mt.pi/2, 2):
                return 'anti-diagonal'
            else:
                return 'non-diagonal'
        else:
            if name == 'h':
                return 'non-diagonal'
            elif name == 'z' or name == 't' or name == 's' or name == 'rz' or name == 'u1' or name =='tdg' or name == 'sdg':
                return 'diagonal'
            elif name == 'x' or name == 'y':
                return 'anti-diagonal'
            else:
                return 'non-diagonal'

    def apply_linked_single_qubit_gate(self, gate: dict) -> None:
        q = gate['qargs'][0]
        p_root = self.current_assignment[q]
        diagonality = self.check_diag_gate(gate)
        if diagonality == 'diagonal':
            self.apply_single_qubit_gate(gate)
        elif diagonality == 'anti-diagonal':
            for p in range(self.num_partitions):
                if p != p_root:
                    if self.current_assignment[q] not in self.qubit_manager.groups[q][p]:
                        continue
                    for linked_part in self.qubit_manager.groups[q]['linked_qubits']:
                        comm_q = self.qubit_manager.groups[q][linked_part]
                        if comm_q == q:
                            continue 
                        self.qc.x(comm_q)
            self.apply_single_qubit_gate(gate)
        else:
            raise ValueError(f"Gate {gate} is not diagonal or anti-diagonal and shouldn't be in group.")
        
        return

    def process_group_gate(self, gate, t: int) -> None:
        """
        Processes a group gate. This requires generating links to the root qubit using the starting process circuit,
        gates are then added to the layer dictionary for later processing.
        """
        # Index of the root qubit in the group.
        root_idx = gate['root']
        # The time step of the group gate.
        start_time = gate['time']
        # The initial partition of the root qubit.
        p_root = self.partition_assignment[start_time][root_idx]
        # The sub-gates of the group gate.
        sub_gates = gate['sub-gates']
        # If there are no sub-gates, then we ignore the group.
        if not sub_gates:
            return
        # Initialise the set of partitions that the root qubit must be linked to.
        p_rec_set = set()
        # Store the time step for the last gate in each partition.
        final_gates = {}
        # The time step of the last gate in the group.
        for sub_gate in sub_gates[::-1]:
            if sub_gate['type'] == 'two-qubit':
                final_t = sub_gate['time'] 
                break
        
        # The final partition of the root qubit. Determines whether starting process must be
        # converted to a gate teleportation.
        final_p_root = int(self.partition_assignment[final_t][root_idx])


        p_root_set = set()
        for time_step in range(start_time, final_t + 1):
            p_root_set.add(int(self.partition_assignment[time_step][root_idx]))
        # Initialise the group dictionary for the root qubit.

        if p_root_set != set([p_root]):
            # This means the root qubit will undergo nested state teleportation, so we must transfer 
            # the state onto a communication qubit.
            root_q = self.qubit_manager.log_to_phys_idx[root_idx]
            root_comm = self.comm_manager.find_comm_idx(p_root)
            self.teleportation_manager.transfer_state(root_q, root_comm)
            # Source data qubit is reset by transfer_state; now free the slot in bookkeeping.
            if root_q in self.qubit_manager.in_use_data[p_root]:
                self.qubit_manager.in_use_data[p_root].pop(root_q, None)
            if root_q not in self.qubit_manager.free_data[p_root]:
                self.qubit_manager.free_data[p_root].append(root_q)
            self.qubit_manager.log_to_phys_idx[root_idx] = root_comm
            # p_rec_set.add(final_p_root)


        # Store information about the group in the qubit manager.
        self.qubit_manager.groups[root_idx] = {}
        # Find the time step for the last gate in each partition.
        for sub_gate in sub_gates:
            if sub_gate['type'] == 'two-qubit':
                q0, q1 = sub_gate['qargs']
                time_step = sub_gate['time']
                p_rec = int(self.partition_assignment[time_step][q1])
                p_rec_set.add(p_rec)
                if p_rec not in final_gates:
                    final_gates[p_rec] = time_step
                else:
                    final_gates[p_rec] = max(final_gates[p_rec], time_step)
        # Add group information to the qubit manager.
        self.qubit_manager.groups[root_idx]['final_gates'] = final_gates
        self.qubit_manager.groups[root_idx]['init_time'] = start_time
        self.qubit_manager.groups[root_idx]['final_time'] = final_t
        self.qubit_manager.groups[root_idx]['final_p_root'] = final_p_root
        self.qubit_manager.groups[root_idx]['init_p_root'] = p_root
        self.qubit_manager.groups[root_idx]['p_rec_set'] = p_rec_set
        self.qubit_manager.groups[root_idx]['p_root_set'] = p_root_set


        linked_qubits = {p_root : self.qubit_manager.log_to_phys_idx[root_idx]}
        self.qubit_manager.groups[root_idx]['linked_qubits'] = linked_qubits


        target_partitions = list(p_rec_set.union(p_root_set) - {p_root})

        target_qubits = self.teleportation_manager.entangle_root_on_tree(root_idx, target_partitions, p_root, self.num_partitions, group_gate=True)



        # Now handle sub-gates
        for sub_gate in sub_gates:
            if sub_gate['type'] == 'two-qubit':
                q0, q1 = sub_gate['qargs']
                time_step = sub_gate['time']
                p1 = int(self.partition_assignment[time_step][q1])
                new_gate = {
                        'type': 'two-qubit-linked',
                        'name': sub_gate['name'],
                        'qargs': [q0, q1],
                        'params': sub_gate['params'],
                        'time': time_step
                    }

                if p1 == p_root:
                    # same partition as root
                    if time_step == t:
                        # apply immediately
                        self.apply_local_two_qubit_gate(sub_gate)
                    else:
                        self.layer_dict[time_step].append(new_gate)
                else:
                    if time_step == t:
                        self.apply_non_local_two_qubit_gate(sub_gate, p_root, p1)
                    else:
                        self.layer_dict[time_step].append(new_gate)

            elif sub_gate['type'] == 'single-qubit':
                # We must handle 'linked' single qubit gates specially, since anti-diagonal gates
                # require us to apply an X on all linked communication qubits.
                q = sub_gate['qargs'][0]
                time_step = sub_gate['time']
                new_gate = {
                        'type': 'single-qubit-linked',
                        'qargs': [q],
                        'params': sub_gate['params'],
                        'time': time_step
                    }
                if time_step == t:
                    self.apply_linked_single_qubit_gate(sub_gate)
                else:
                    self.layer_dict[time_step].append(sub_gate)

    def extract_partitioned_circuit(self) -> QuantumCircuit:
        for i, layer in sorted(self.layer_dict.items()):
            new_assignment_layer = self.partition_assignment[i]
            for q in range(self.num_qubits):
                if self.current_assignment[q] != new_assignment_layer[q]:
                    self.teleportation_manager.teleport_qubits(self.current_assignment,
                                                               new_assignment_layer,
                                                               self.num_partitions,
                                                               self.num_qubits)

                    break

            self.current_assignment = new_assignment_layer
            self.partition_assignment[i] = new_assignment_layer

            for gate in layer:
                gtype = gate['type']

                if gtype == "single-qubit":
                    self.apply_single_qubit_gate(gate)

                elif gtype == "two-qubit":
                    q0, q1 = gate['qargs']
                    p0 = self.current_assignment[q0]
                    p1 = self.current_assignment[q1]
                    if p0 == p1:
                        self.apply_local_two_qubit_gate(gate)
                    else:
                        self.teleportation_manager.gate_teleport(q0, q1, gate, p0, p1)

                elif gtype == "group":
                    self.process_group_gate(gate, i)

                elif gtype == "two-qubit-linked":

                    q0, q1 = gate['qargs']
                    p_root = self.qubit_manager.groups[q0]['init_p_root']
                    p_rec = self.current_assignment[q1]

                    self.apply_non_local_two_qubit_gate(gate, p_root, p_rec)


        for i in range(self.num_qubits):
            self.qc.measure(self.qubit_manager.log_to_phys_idx[i], self.result_reg[i])
        
        self.qc = reorder_registers_by_index(self.qc)
        teleportation_gates = ['x', 'z']
        basis_gate_set = set(teleportation_gates).union(set(self.basis_gates)).union(set(['EPR']))
        self.qc = transpile(self.qc, basis_gates= list(basis_gate_set))
        return self.qc
    
def reorder_registers_by_index(circuit):
    # Separate quantum registers so comm registers are grouped with their corresponding data registers.
    q_groups = {}
    c_groups = {}
    other_qregs = []
    for reg in circuit.qregs:
        name = reg.name
        if name.startswith("Q"):
            try:
                i = int(name[1:].split("_")[0])
                q_groups[i] = reg
            except Exception:
                other_qregs.append(reg)
        elif name.startswith("C"):
            try:
                i = int(name[1:].split("_")[0])
                j = int(name[1:].split("_")[1]) if "_" in name[1:] else -1
                c_groups.setdefault(i, []).append((j, reg))
            except Exception:
                other_qregs.append(reg)
        else:
            other_qregs.append(reg)

    ordered_qregs = []
    for i in sorted(set(list(q_groups.keys()) + list(c_groups.keys()))):
        if i in q_groups:
            ordered_qregs.append(q_groups[i])
        if i in c_groups:
            ordered_qregs.extend([reg for j, reg in sorted(c_groups[i], key=lambda x: x[0])])
    ordered_qregs.extend(other_qregs)

    ordered_cregs = list(circuit.cregs)

    new_circ = QuantumCircuit(*ordered_qregs, *ordered_cregs)
    for instruction in circuit.data:
        new_circ.append(instruction.operation, instruction.qubits, instruction.clbits)
    return new_circ


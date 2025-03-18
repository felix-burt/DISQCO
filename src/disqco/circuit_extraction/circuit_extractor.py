import logging
import copy
import numpy as np
import networkx as nx
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from disqco.circuit_extraction.DQC_qubit_manager import DataQubitManager, CommunicationQubitManager, ClassicalBitManager
import math as mt

# -------------------------------------------------------------------
# Set up a logger
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to logging.INFO or higher to reduce verbosity

# If you want logs printed to console:
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.disabled = True

# -------------------------------------------------------------------
# TeleportationManager
# -------------------------------------------------------------------

class TeleportationManager:
    
    def __init__(
        self,
        qc: QuantumCircuit, 
        qubit_manager: DataQubitManager, 
        comm_manager: CommunicationQubitManager, 
        creg_manager: ClassicalBitManager
    ) -> None:
        
        self.qc = qc
        self.qubit_manager = qubit_manager
        self.comm_manager = comm_manager
        self.creg_manager = creg_manager

    def build_epr_circuit(self) -> QuantumCircuit:
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)

        gate = circ.to_gate()
        gate.name = "EPR"
        return gate

    def build_root_entanglement_circuit(self) -> QuantumCircuit:
        epr_circ = self.build_epr_circuit()
        circ = QuantumCircuit(3, 1)
        circ.append(epr_circ, [1, 2])
        circ.cx(0, 1)
        circ.measure(1, 0)
        circ.reset(1)
        circ.x(2).c_if(0, 1)

        instr = circ.to_instruction()
        instr.name = "entangle_root"
        return instr
    
    def build_end_entanglement_circuit(self) -> QuantumCircuit:
        circ = QuantumCircuit(2, 1)
        circ.h(1)
        circ.measure(1, 0)
        circ.reset(1)
        circ.z(0).c_if(0, 1)

        instr = circ.to_instruction()
        instr.name = "end_entanglement_link"
        return instr

    def build_teleporation_circuit(self) -> QuantumCircuit:
        circ = QuantumCircuit(3, 2)
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

    def build_gate_teleportation_circuit(self, gate_params) -> QuantumCircuit:
        circ = QuantumCircuit(4, 1)
        root_entanglement_circuit = self.build_root_entanglement_circuit()
        circ.append(root_entanglement_circuit, [0, 1, 2], [0])

        circ.cp(gate_params[0], 2, 3)
        entanglement_end_circuit = self.build_end_entanglement_circuit()
        circ.append(entanglement_end_circuit, [0, 2], [0])

        instr = circ.to_instruction(label="gate_teleport")
        return instr

    def generate_epr(self, p1: int, p2: int, comm_id1: Qubit = None, comm_id2: Qubit = None) -> tuple[Qubit, Qubit]:
        logger.debug(f"[generate_epr] Creating EPR between p1={p1}, p2={p2}")

        if comm_id1 is None:
            comm_qubit1 = self.comm_manager.find_comm_idx(p1)
        else:
            comm_qubit1 = comm_id1

        if comm_id2 is None:
            comm_qubit2 = self.comm_manager.find_comm_idx(p2)
        else:
            comm_qubit2 = comm_id2

        gate = self.build_epr_circuit()
        logger.debug(f"[generate_epr] Appending EPR circuit for comm qubits {comm_qubit1}, {comm_qubit2}")
        self.qc.append(gate, [comm_qubit1, comm_qubit2])

        self.comm_manager.in_use_comm[p1].add(comm_qubit1)
        self.comm_manager.in_use_comm[p2].add(comm_qubit2)

        return comm_qubit1, comm_qubit2
    
    def entangle_root(self, root_q: int, p_root: int, p_rec: int) -> None:
        logger.debug(f"[entangle_root] Entangling root qubit {root_q} in partition {p_root} with partition {p_rec}")
        root_phys = self.qubit_manager.log_to_phys_idx[root_q]
        root_comm = self.comm_manager.find_comm_idx(p_root)

        if p_root == p_rec:
            logger.debug("[entangle_root] Local entanglement recognized.")
            self.entangle_root_local(root_q, p_root)
            return

        rec_comm = self.comm_manager.find_comm_idx(p_rec)
        cbit = self.creg_manager.allocate_cbit()
        instr = self.build_root_entanglement_circuit()

        logger.debug(f"[entangle_root] Appending root_entanglement_circuit on qubits {[root_phys, root_comm, rec_comm]} -> cbit {cbit}")
        self.qc.append(instr, [root_phys, root_comm, rec_comm], [cbit])

        self.comm_manager.linked_qubits[rec_comm] = root_q
        self.qubit_manager.linked_comm_qubits[root_q][p_rec] = rec_comm

        self.creg_manager.release_cbit(cbit)
        self.comm_manager.release_comm_qubit(p_root, root_comm)

    def entangle_root_local(self, root_q: int, root_part: int) -> None:
        logger.debug(f"[entangle_root_local] Local entanglement of root qubit {root_q} and comm qubit in partition {root_part}")
        root_phys = self.qubit_manager.log_to_phys_idx[root_q]
        linked_root = self.comm_manager.find_comm_idx(root_part)
        self.qc.cx(root_phys, linked_root)
        self.qubit_manager.linked_comm_qubits[root_q][root_part] = linked_root
        self.comm_manager.linked_qubits[linked_root] = root_q

    def end_entanglement_link(self, linked_comm: Qubit, p_rec: int) -> None:
        logger.debug(f"[end_entanglement_link] Ending entanglement link of comm qubit {linked_comm} in partition {p_rec}")
        cbit = self.creg_manager.allocate_cbit()
        q_root = self.comm_manager.linked_qubits[linked_comm]
        root_qubit_id = self.qubit_manager.log_to_phys_idx[q_root]

        instr = self.build_end_entanglement_circuit()
        logger.debug(f"[end_entanglement_link] Appending end_entanglement_circuit on qubits {[root_qubit_id, linked_comm]} -> cbit {cbit}")
        self.qc.append(instr, [root_qubit_id, linked_comm], [cbit])

        logger.debug(f"[end_entanglement_link] Linked qubits before deletion: {self.comm_manager.linked_qubits}")
        del self.comm_manager.linked_qubits[linked_comm]
        logger.debug(f"[end_entanglement_link] Linked qubits after deletion: {self.comm_manager.linked_qubits}")

        logger.debug(f"[end_entanglement_link] linked_comm_qubits on root {q_root} in part {p_rec} before: {self.qubit_manager.linked_comm_qubits[q_root][p_rec]}")
        del self.qubit_manager.linked_comm_qubits[q_root][p_rec]
        logger.debug(f"[end_entanglement_link] groups on root {q_root} pre: {self.qubit_manager.groups[q_root]}")
        logger.debug(f"[end_entanglement_link] groups on root {q_root} post: {self.qubit_manager.groups[q_root]}")
        del self.qubit_manager.groups[q_root]['final_gates'][p_rec]
        del self.qubit_manager.groups[q_root]['linked_qubits'][p_rec]

        if self.qubit_manager.groups[q_root]['final_gates'] == {}:
            del self.qubit_manager.groups[q_root]



        self.creg_manager.release_cbit(cbit)
        self.comm_manager.release_comm_qubit(p_rec, linked_comm)

    def extract_cycles_and_edges(self, G: nx.MultiDiGraph) -> tuple[list[tuple], dict[tuple, list[tuple]], list[tuple]]:
        logger.debug("[extract_cycles_and_edges] Identifying cycles in the partition assignment graph.")
        cycles = []
        for cycle_nodes in nx.simple_cycles(G):
            cycles.append(tuple(cycle_nodes))

        all_cycle_edges = []
        cycle_edges = {}
        for cycle_nodes in cycles:
            cycle_edges[cycle_nodes] = []
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                for qubit in G.adj[u][v]:
                    break
                cycle_edges[cycle_nodes].append(((u, v, qubit)))
                all_cycle_edges.append((u, v, qubit))

        G.remove_edges_from(all_cycle_edges)
        remaining_edges = [edge for edge in G.edges]

        return cycles, cycle_edges, remaining_edges

    def get_teleport_cycles(self, assignment1: list, assignment2: list,
                            num_partitions: int, num_qubits: int) -> tuple[list[list[int]], list[list[tuple]]]:
        logger.debug("[get_teleport_cycles] Building directed multi-graph for qubit movement.")
        graph = nx.MultiDiGraph()
        for p in range(num_partitions):
            graph.add_node(p)

        for q in range(num_qubits):
            p1 = assignment1[q]
            p2 = assignment2[q]
            if p1 != p2:
                graph.add_edge(p1, p2, key=q, label=q)

        _, cycle_edges, edges = self.extract_cycles_and_edges(graph)

        qubit_lists = []
        directions_lists = []
        for cycle in cycle_edges:
            qubits = []
            directions = []
            for edge in cycle_edges[cycle]:
                qubits.append(edge[2])
                directions.append((edge[0], edge[1]))
            qubit_lists.append(qubits)
            directions_lists.append(directions)

        qubits = []
        directions = []
        for edge in edges:
            qubits.append(edge[2])
            directions.append((edge[0], edge[1]))
        qubit_lists.append(qubits)
        directions_lists.append(directions)

        used = set()
        new_qubit_lists = []
        new_directions_lists = []
        for i, cycle in enumerate(qubit_lists):
            cycle_list = []
            direc_list = []
            for j, q in enumerate(cycle):
                if q not in used:
                    cycle_list.append(q)
                    direc_list.append(directions_lists[i][j])
                    used.add(q)
            new_qubit_lists.append(cycle_list)
            new_directions_lists.append(direc_list)

        return new_qubit_lists, new_directions_lists

    def swap_qubits_to_phsyical(self, data_locations: list[tuple[Qubit, int, Qubit]]) -> None:
        logger.debug("[swap_qubits_to_phsyical] Swapping teleported qubits to physical data qubits.")
        for qubit, partition, data_loc in data_locations:
            try:
                data_q = self.qubit_manager.allocate_data_qubit(partition)
                logger.debug(f"  Swapping from comm {data_loc} to data {data_q} for logical qubit {qubit} in partition {partition}")
                self.qc.swap(data_loc, data_q)
                self.qc.reset(data_loc)
                self.qubit_manager.assign_to_physical(partition, data_q, qubit)
                self.comm_manager.release_comm_qubit(partition, data_loc)

            except Exception as e:
                logger.warning(f"  No data space for qubit {qubit} in partition {partition}, leaving on comm qubit {data_loc}. Error: {e}")
                self.qubit_manager.queue[qubit] = (data_loc, partition)
                self.qubit_manager.log_to_phys_idx[qubit] = data_loc
                if data_loc not in self.comm_manager.in_use_comm[partition]:
                    self.comm_manager.in_use_comm[partition].add(data_loc)
                if data_loc in self.comm_manager.free_comm[partition]:
                    self.comm_manager.free_comm[partition].remove(data_loc)

    def teleport_qubits(self, old_assignment: list[int], new_assignment: list[int],
                        num_partitions: int, num_qubits: int) -> None:
        logger.debug("[teleport_qubits] Teleporting qubits from old_assignment to new_assignment.")
        q_list, direc_list = self.get_teleport_cycles(old_assignment, new_assignment,
                                                      num_partitions, num_qubits)

        logger.debug(f"  Teleport cycles qubit_lists: {q_list}")
        logger.debug(f"  Teleport cycles directions_lists: {direc_list}")

        for j, cycle in enumerate(q_list):
            data_locations = []
            directions = direc_list[j]
            for i, q in enumerate(cycle):
                p_source = directions[i][0]
                p_dest = directions[i][1]
                logger.info(f"[teleport_qubits] Teleporting qubit {q} from partition {p_source} to {p_dest}.")
                data_q1 = self.qubit_manager.log_to_phys_idx[q]

                comm_source = self.comm_manager.find_comm_idx(p_source)
                comm_dest = self.comm_manager.find_comm_idx(p_dest)

                cbit1 = self.creg_manager.allocate_cbit()
                cbit2 = self.creg_manager.allocate_cbit()
                
                instr = self.build_teleporation_circuit()
                logger.debug(f"  Appending state_teleport on qubits {[data_q1, comm_source, comm_dest]} -> cbits {[cbit1, cbit2]}")
                self.qc.append(instr, [data_q1, comm_source, comm_dest], [cbit1, cbit2])

                self.comm_manager.release_comm_qubit(p_source, comm_source)
                self.qubit_manager.release_data_qubit(p_source, data_q1)

                self.creg_manager.release_cbit(cbit1)
                self.creg_manager.release_cbit(cbit2)

                data_locations.append((q, p_dest, comm_dest))

            self.swap_qubits_to_phsyical(data_locations)

    def gate_teleport(self, root_q: int, rec_q: int, gate: dict, p_root: int, p_rec: int) -> None:
        logger.debug(f"[gate_teleport] Teleporting gate from root_q={root_q} (p_root={p_root}) to rec_q={rec_q} (p_rec={p_rec}).")
        comm_root = self.comm_manager.find_comm_idx(p_root)
        comm_rec = self.comm_manager.find_comm_idx(p_rec)

        gate_params = gate['params']
        data_q_root = self.qubit_manager.log_to_phys_idx[root_q]
        data_q_rec = self.qubit_manager.log_to_phys_idx[rec_q]

        cbit1 = self.creg_manager.allocate_cbit()
        instr = self.build_gate_teleportation_circuit(gate_params)

        logger.debug(f"  Appending gate_teleport on qubits {[data_q_root, comm_root, comm_rec, data_q_rec]} -> cbit {cbit1}")
        self.qc.append(instr, [data_q_root, comm_root, comm_rec, data_q_rec], [cbit1])

        self.comm_manager.release_comm_qubit(p_root, comm_root)
        self.comm_manager.release_comm_qubit(p_rec, comm_rec)
        self.creg_manager.release_cbit(cbit1)


# -------------------------------------------------------------------
# PartitionedCircuitExtractor
# -------------------------------------------------------------------
class PartitionedCircuitExtractor:

    def __init__(
        self,
        graph: QuantumCircuitHyperGraph,
        partition_assignment: list[list[int]],
        qpu_info: list[int],
        comm_info: list[int]
    ) -> None:
        self.layer_dict = graph.layers
        self.partition_assignment = partition_assignment
        self.layer_dict = self.remove_empty_groups()

        self.num_qubits = graph.num_qubits
        self.qpu_info = qpu_info
        self.comm_info = comm_info
        self.depth = graph.depth
        self.num_partitions = len(qpu_info)
        self.partition_qregs = self.create_data_qregs()
        self.comm_qregs = self.create_comm_qregs()
        self.creg, self.result_reg = self.create_classical_registers()
        self.qc = self.build_initial_circuit()

        self.qubit_manager = DataQubitManager(self.partition_qregs, self.num_qubits,
                                              self.partition_assignment, self.qc)
        self.comm_manager = CommunicationQubitManager(self.comm_qregs, self.qc)
        self.creg_manager = ClassicalBitManager(self.qc, self.creg)

        self.teleportation_manager = TeleportationManager(self.qc, self.qubit_manager,
                                                          self.comm_manager, self.creg_manager)

        # Keep track of current assignment for each qubit from the last layer
        self.current_assignment = self.partition_assignment[0]

    def remove_empty_groups(self) -> dict[int, list[dict]]:
        logger.debug("[remove_empty_groups] Removing empty or single-gate groups.")
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
        partition_qregs = []
        for i in range(self.num_partitions):
            size_i = self.qpu_info[i]
            qr = QuantumRegister(size_i, name=f"part{i}_data")
            partition_qregs.append(qr)
        return partition_qregs

    def create_comm_qregs(self) -> dict[int, list[QuantumRegister]]:
        comm_qregs = {}
        for i in range(self.num_partitions):
            comm_qregs[i] = [QuantumRegister(self.comm_info[i], name=f"comm_{i}_{0}")]
        return comm_qregs

    def create_classical_registers(self) -> tuple[ClassicalRegister, ClassicalRegister]:
        creg = ClassicalRegister(2, name="c")
        result_reg = ClassicalRegister(self.num_qubits, name="result")
        return creg, result_reg

    def build_initial_circuit(self) -> QuantumCircuit:
        comm_regs_all = [part[0] for part in self.comm_qregs.values()]
        qc = QuantumCircuit(
            *self.partition_qregs,
            *comm_regs_all,
            *[self.creg, self.result_reg],
            name="PartitionedCircuit"
        )
        return qc

    def apply_single_qubit_gate(self, gate: dict) -> None:
        q = gate['qargs'][0]
        params = gate['params']
        qubit_phys = self.qubit_manager.log_to_phys_idx[q]
        logger.debug(f"[apply_single_qubit_gate] Gate U({params}) on logical {q} -> physical {qubit_phys}")
        self.qc.u(*params, qubit_phys)

    def apply_local_two_qubit_gate(self, gate: dict) -> None:
        qubit0, qubit1 = gate['qargs']
        params = gate['params']
        logger.debug(f"[apply_local_two_qubit_gate] Gate CP({params[0]}) on logical ({qubit0}, {qubit1})") 
        
        if isinstance(qubit0, int):
            qubit0 = self.qubit_manager.log_to_phys_idx[qubit0]
        if isinstance(qubit1, int):
            qubit1 = self.qubit_manager.log_to_phys_idx[qubit1]
        logger.debug(f" -> physical ({qubit0}, {qubit1})")
        self.qc.cp(params[0], qubit0, qubit1)

    def teleport_root_early(self, gate: dict, root_qubit: int, p_root: int, final_p_root: int) -> None:
        current_t = gate['time']
        k = current_t
        old_assignment = copy.deepcopy(self.current_assignment)
        full_assignment_copy = copy.deepcopy(self.partition_assignment)

        while k < len(full_assignment_copy):
            p_root_k = full_assignment_copy[k][root_qubit]
            if p_root_k == final_p_root:
                break
            full_assignment_copy[k][root_qubit] = final_p_root
            k += 1

        new_assignment = full_assignment_copy[current_t]
        logger.debug(f"[teleport_root_early] Teleporting root {root_qubit} from partition {p_root} to {final_p_root}")
        logger.debug(f"Active receivers: {self.qubit_manager.active_receivers}")
        # if root_qubit in self.qubit_manager.active_receivers:
        #     logger.debug(f"Root qubit is active receiver: {self.qubit_manager.active_receivers[root_qubit]}")
        #     end_time = current_t
        #     for key in self.qubit_manager.active_receivers[root_qubit]:
        #         time = self.qubit_manager.active_receivers[root_qubit][key]
        #         if time > end_time:
        #             end_time = time
        #     logger.debug(f"End time for root qubit {root_qubit} is {end_time}")
        #     if end_time >= current_t:
        #         if self.qubit_manager.linked_comm_qubits[root_qubit][p_root] != set():
        #             for comm_root in self.qubit_manager.linked_comm_qubits[root_qubit][p_root]:
        #                 self.qubit_manager.groups[root_qubit][p_root] = end_time
        #                 break
        #         else:
        if p_root not in self.qubit_manager.linked_comm_qubits[root_qubit]:
            if p_root in self.qubit_manager.groups[root_qubit]['final_gates']:
                if self.qubit_manager.groups[root_qubit]['final_gates'][p_root] >= current_t:
                    self.teleportation_manager.entangle_root_local(root_qubit, p_root)
                    self.qubit_manager.groups[root_qubit]['linked_qubits'][p_root] = self.qubit_manager.linked_comm_qubits[root_qubit][p_root]
        # self.comm_manager.linked_qubits[comm_root] = root_qubit
        # self.qubit_manager.groups[root_qubit]['final_gates'][p_root] = end_time
        # self.qubit_manager.relocated_receivers[root_qubit] = (p_root, end_time)
        # logger.debug(f"Root qubit {root_qubit} is relocated to partition {p_root} at time {end_time}")

        self.teleportation_manager.teleport_qubits(old_assignment, new_assignment,
                                                    self.num_partitions, self.num_qubits)
        self.current_assignment = new_assignment
        self.partition_assignment = full_assignment_copy
    
    def find_common_part(self, q0: int, q1: int) -> int:
        logger.debug(f'q0: {q0} assigned to {self.current_assignment[q0]}')
        logger.debug(f'q1: {q1} assigned to {self.current_assignment[q1]}')


        group0_links = self.qubit_manager.groups[q0]['linked_qubits']
        logger.debug(f'group0 info: {self.qubit_manager.groups[q0]}')
        logger.debug(f'Linked comms: {self.qubit_manager.linked_comm_qubits[q0]}')
        group1_links = self.qubit_manager.groups[q1]['linked_qubits']
        logger.debug(f'group1 info: {self.qubit_manager.groups[q1]}')
        logger.debug(f'Linked comms: {self.qubit_manager.linked_comm_qubits[q1]}')

        part_set0 = set()
        for part in group0_links:
            part_set0.add(int(part))
        part_set1 = set()
        for part in group1_links:
            part_set1.add(int(part))

        p0 = self.current_assignment[q0]
        p1 = self.current_assignment[q1]

        q0_phys = self.qubit_manager.log_to_phys_idx[q0]

        if p0 in part_set1:
            logger.debug(f'q0 in part_set1')
            return q0_phys, self.qubit_manager.linked_comm_qubits[q1][p0]
        if p1 in part_set0:
            logger.debug(f'q1 in part_set0')
            return self.qubit_manager.linked_comm_qubits[q0][p1], q1
        for p0 in part_set0:
            for p1 in part_set1:
                if int(p0) == int(p1):
                    logger.debug(f'Common partition found: {p0}')
                    logger.debug(f'Linked comm qubit q0: {self.qubit_manager.linked_comm_qubits[q0][p0]}')
                    logger.debug(f'Linked comm qubit q1: {self.qubit_manager.linked_comm_qubits[q1][p0]}')
                    return self.qubit_manager.linked_comm_qubits[q0][p0], self.qubit_manager.linked_comm_qubits[q1][p0]
        
        return None, None

    def apply_non_local_two_qubit_gate(self, gate: dict, p_root: int, p1: int) -> None:
        logger.debug(f"[apply_non_local_two_qubit_gate] Applying non-local two-qubit gate {gate} between partitions {p_root} and {p1}")
        if p_root == p1:
            logger.debug(f"[apply_non_local_two_qubit_gate] Gate occurs in the same partition {p_root}")
            self.apply_local_two_qubit_gate(gate)
            return
        
        root_q, q1 = gate['qargs']
        if root_q in self.qubit_manager.groups:
            logger.debug(f"[apply_non_local_two_qubit_gate] Non local gate in group {root_q}")
            if p1 == self.qubit_manager.groups[root_q]['final_p_root']:
                logger.debug(f"[apply_non_local_two_qubit_gate] Gate occurs in the final partition for the group - teleport root to this location and apply locally")
                # Gate occurs in the final partition for the group - teleport root to this location and apply locally
                self.teleport_root_early(gate, root_q, p_root, p1)
                self.apply_local_two_qubit_gate(gate)
                return
            else:
                # Gate occurs in a non-local partition for the group
                logger.debug(f"[apply_non_local_two_qubit_gate] Gate occurs in a non-local partition for the group")
                if p1 in self.qubit_manager.groups[root_q]['linked_qubits']:
                    logger.debug(f"[apply_non_local_two_qubit_gate] Communication qubit has been linked already")
                    # A communication qubit has been linked already
                    linked_root = self.qubit_manager.groups[root_q]['linked_qubits'][p1]
                    gate['qargs'] = [linked_root, gate['qargs'][1]]
                    self.apply_local_two_qubit_gate(gate)
                    logger.debug(f"Gates in group {self.qubit_manager.groups[root_q]['final_gates']}")
                    if gate['time'] == self.qubit_manager.groups[root_q]['final_gates'][p1]:
                        logger.debug(f"[apply_non_local_two_qubit_gate] Ending entanglement link for linked comm qubit {linked_root} in partition {p1}")
                        # Final gate for this partition, end the link to free communication qubit capacity
                        self.teleportation_manager.end_entanglement_link(linked_root, p1)
                else:
                    logger.debug(f"[apply_non_local_two_qubit_gate] No linked communication qubits - check for relocated receivers")
                    if q1 in self.qubit_manager.groups:
                        logger.debug(f"Find common partition for gate between {root_q} and {q1}")
                        qubits = self.find_common_part(root_q,q1)
                        if qubits[0] is not None:
                            linked_root, linked_comm = qubits
                            logger.debug(f"[apply_non_local_two_qubit_gate] Common partition found, apply detached")
                            gate['qargs'] = [linked_root, linked_comm]
                            self.apply_local_two_qubit_gate(gate)
                            # if gate['time'] == self.qubit_manager.groups[root_q]['final_gates'][common_p]:
                            #     self.teleportation_manager.end_entanglement_link(linked_root, common_p)
                            return
                    
                    logger.debug(f"[apply_non_local_two_qubit_gate] Must entangle root qubit {root_q} with partition {p1}")
                    self.teleportation_manager.entangle_root(root_q, p_root, p1)
                    linked_root = self.qubit_manager.linked_comm_qubits[root_q][p1]
                    self.qubit_manager.groups[root_q]['linked_qubits'][p1] = linked_root
                    gate['qargs'] = [linked_root, gate['qargs'][1]]
                    self.apply_local_two_qubit_gate(gate)
                    logger.debug(f"Gates in group {self.qubit_manager.groups[root_q]['final_gates']}")
                    logger.debug(f"Gate time: {gate['time']}")
                    if gate['time'] == self.qubit_manager.groups[root_q]['final_gates'][p1]:
                        logger.debug(f"[apply_non_local_two_qubit_gate] Ending entanglement link for linked comm qubit {linked_root} in partition {p1}")
                        self.teleportation_manager.end_entanglement_link(linked_root, p1)

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
            elif name == 'z' or name == 't' or name == 's' or name == 'rz' or name == 'u1':
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
                    if self.current_assignment[q] in self.qubit_manager.linked_comm_qubits[q][p]:
                        comm_q = self.qubit_manager.linked_comm_qubits[q][p]
                        self.qc.x(comm_q)
            self.apply_single_qubit_gate(gate)
        else:
            raise ValueError(f"Gate {gate} is not diagonal or anti-diagonal and shouldn't be in group.")
        
        return



    def process_group_gate(self, gate, t: int) -> None:
        logger.debug(f"[process_group_gate] Processing 'group' gate at layer {t} -> {gate}")
        root_qubit = gate['root']
        start_time = gate['time']
        p_root = self.current_assignment[root_qubit]
        sub_gates = gate['sub-gates']
        p_rec_set = set()
        final_gates = {}
        final_t = sub_gates[-1]['time']
        final_p_root = int(self.partition_assignment[final_t][root_qubit])
        p_root_set = set()
        for i in range(start_time, final_t+1):
            p_root_set.add(int(self.partition_assignment[i][root_qubit]))

        if sub_gates:
            self.qubit_manager.groups[root_qubit] = {}

            for sub_gate in sub_gates:
                q0, q1 = sub_gate['qargs']
                time_step = sub_gate['time']
                p_rec = int(self.partition_assignment[time_step][q1])
                p_rec_set.add(p_rec)
                if p_rec != final_p_root:
                    if p_rec not in final_gates:
                        final_gates[p_rec] = time_step
                    else:
                        final_gates[p_rec] = max(final_gates[p_rec], time_step)

            self.qubit_manager.groups[root_qubit]['final_gates'] = final_gates

            logger.debug(f"[process_group_gate] root_qubit={root_qubit}, p_root={p_root}, final_p_root={final_p_root}, p_rec_set={p_rec_set}")

            p_rec_set_nl = p_rec_set - p_root_set
            
            self.qubit_manager.groups[root_qubit]['p_rec_set_nl'] = p_rec_set_nl
            self.qubit_manager.groups[root_qubit]['p_root_set'] = p_root_set
            self.qubit_manager.groups[root_qubit]['final_p_root'] = final_p_root
            self.qubit_manager.groups[root_qubit]['linked_qubits'] = {}

            # Now handle sub-gates
            for sub_gate in sub_gates:
                if sub_gate['type'] == 'two-qubit':
                    q0, q1 = sub_gate['qargs']
                    time_step = sub_gate['time']
                    p1 = int(self.partition_assignment[time_step][q1])
                    new_gate = {
                            'type': 'two-qubit-linked',
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

    def manage_qubit_queue(self) -> None:
        logger.debug("[manage_qubit_queue] Managing queued qubits to see if we can free comm qubits.")
        for qubit in list(self.qubit_manager.queue.keys()):
            comm_qubit, partition = self.qubit_manager.queue[qubit]
            if not self.qubit_manager.free_data[partition]:
                logger.debug(f"  No free data qubits in partition {partition} for logical qubit {qubit}")
                continue
            data_qubit = self.qubit_manager.free_data[partition].pop(0)
            logger.debug(f"  Swapping queue comm qubit {comm_qubit} with free data qubit {data_qubit} for logical qubit {qubit}")
            self.qc.swap(comm_qubit, data_qubit)
            self.qc.reset(qubit)
            self.qubit_manager.assign_to_physical(partition, data_qubit, qubit)
            self.comm_manager.release_comm_qubit(partition, comm_qubit)
            del self.qubit_manager.queue[qubit]

    def extract_partitioned_circuit(self) -> QuantumCircuit:
        logger.info("[extract_partitioned_circuit] Beginning circuit extraction.")
        for i, layer in sorted(self.layer_dict.items()):
            logger.debug(f"  --- LAYER {i} ---")
            new_assignment_layer = self.partition_assignment[i]

            # Check if any root qubits must teleport
            for q in range(self.num_qubits):
                if self.current_assignment[q] != new_assignment_layer[q]:
                    if q in self.qubit_manager.groups:
                        if self.current_assignment[q] in self.qubit_manager.groups[q]['final_gates']:
                            if self.qubit_manager.groups[q]['final_gates'][self.current_assignment[q]] >= int(i):
                                self.teleportation_manager.entangle_root_local(q, self.current_assignment[q])
                                self.qubit_manager.groups[q]['linked_qubits'][self.current_assignment[q]] = self.qubit_manager.linked_comm_qubits[q][self.current_assignment[q]]

            # If assignment changes for any qubit, we do a group teleport
            for q in range(self.num_qubits):
                if self.current_assignment[q] != new_assignment_layer[q]:
                    logger.debug(f"  [extract_partitioned_circuit] Teleporting because assignment changed: qubit {q}")
                    self.teleportation_manager.teleport_qubits(self.current_assignment,
                                                               new_assignment_layer,
                                                               self.num_partitions,
                                                               self.num_qubits)
                    break

            self.current_assignment = new_assignment_layer
            self.partition_assignment[i] = new_assignment_layer

            # Manage any queue
            self.manage_qubit_queue()

            # Now apply gates in the layer
            for gate in layer:
                logger.debug(f"  Gate encountered: {gate}")
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
                    p_root = self.current_assignment[q0]
                    p_rec = self.current_assignment[q1]

                    self.apply_non_local_two_qubit_gate(gate, p_root, p_rec)
                    
                    if q1 in self.qubit_manager.active_receivers:
                        if q0 in self.qubit_manager.active_receivers[q1]:
                            if self.qubit_manager.active_receivers[q1][q0] == i:
                                del self.qubit_manager.active_receivers[q1][q0]
                                if self.qubit_manager.active_receivers[q1] == {}:
                                    del self.qubit_manager.active_receivers[q1]
                    
                    if q0 in self.qubit_manager.active_receivers:
                        if q1 in self.qubit_manager.active_receivers[q0]:
                            if self.qubit_manager.active_receivers[q0][q1] == i:
                                del self.qubit_manager.active_receivers[q0][q1]
                                if self.qubit_manager.active_receivers[q0] == {}:
                                    del self.qubit_manager.active_receivers[q0]

                    

        # Finally, measure all qubits
        for i in range(self.num_qubits):
            self.qc.measure(self.qubit_manager.log_to_phys_idx[i], self.result_reg[i])

        logger.info("[extract_partitioned_circuit] Circuit extraction complete.")
        return self.qc
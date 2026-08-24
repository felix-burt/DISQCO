from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit
import copy
import networkx as nx

def find_swap_path(topology, src_idx, dst_idx) -> list[tuple[int, int]]:
    """
    Finds a path of swaps between two qubits in the topology.
    """
    path = nx.shortest_path(topology, source=src_idx, target=dst_idx)
    swap_path = [(path[i], path[i + 1]) for i in range(len(path) - 2)]
    return swap_path

# -------------------------------------------------------------------
# CommunicationQubitManager
# -------------------------------------------------------------------
class CommunicationQubitManager:
    """
    Manages communication qubits on a per-partition basis. Allocates communication qubits for tasks 
    requiring entanglement and releases them when done.
    """
    def __init__(self, comm_qregs: dict, qc: QuantumCircuit, network=None):
        self.qc = qc  # Store copy of the QuantumCircuit
        self.comm_qregs = comm_qregs  # Store the QuantumRegisters for communication qubits
        self.free_comm = {}  # Store free communication qubits for each partition
        self.in_use_comm = {}  # Store in-use communication qubits for each partition
        self.network = network

        # self.linked_qubits = {}  # Store comm qubits linked to root qubits for gate teleportation

        self.initilize_communication_qubits()

    def initilize_communication_qubits(self) -> None:
        """
        Set all communication qubits to free.
        """
        for p, reg_list in self.comm_qregs.items():
            self.free_comm[p] = []
            self.in_use_comm[p] = set()
            for reg in reg_list:
                for qubit in reg:
                    self.free_comm[p].append(qubit)

    def comm_index(self, p: int, qubit: Qubit):
        """
        Returns index of a comm qubit within QPU p's comm qubits or None if it was created.
        """
        if qubit._register is self.comm_qregs[p][0]:
            return qubit._index
        return None

    def find_comm_idx(self, p: int, neighbor: int | None = None) -> Qubit:
        """
        Allocate a free communication qubit in partition p.
        """
        free_comm_p = self.free_comm[p]
        if self.network is not None and neighbor is not None and p in self.network.comm_links:
            eligible = set(self.network.comm_qubits_for_link(p, neighbor))
            candidates = [q for q in free_comm_p if self.comm_index(p, q) in eligible]
            if not candidates:
                raise RuntimeError(f"QPU {p} has no free comm qubit serving link to {neighbor}")
            comm_qubit = candidates[0]
            free_comm_p.remove(candidates[0])

        elif free_comm_p:
            comm_qubit = free_comm_p.pop(0)
        else:
            if self.network is not None and self.network.comm_constrained(p):
                raise RuntimeError(f"QPU {p} has no free communication qubits")
            # Create a new communication qubit by adding a new register
            num_regs_p = len(self.comm_qregs[p])
            new_reg = QuantumRegister(1, name=f"C{p}_{num_regs_p}")
            self.comm_qregs[p].append(new_reg)
            self.qc.add_register(new_reg)
            comm_qubit = new_reg[0]

        self.in_use_comm[p].add(comm_qubit)

        return comm_qubit

    def release_comm_qubit(self, p: int, comm_qubit: Qubit) -> None:
        """
        Resets the qubit and returns it to the free pool in partition p.
        """
        if comm_qubit in self.in_use_comm[p]:
            self.in_use_comm[p].remove(comm_qubit)
            self.free_comm[p].append(comm_qubit)

    def get_status(self, p: int) -> tuple[list, list]:
        """
        Return a tuple (in_use, free) for partition p.
        """
        return self.in_use_comm.get(p, []), self.free_comm.get(p, [])

# -------------------------------------------------------------------
# ClassicalBitManager
# -------------------------------------------------------------------
class ClassicalBitManager:
    """
    Manages classical bits, allocating from a pool and releasing after use.
    """
    def __init__(self, qc: QuantumCircuit, creg: ClassicalRegister):
        self.qc = qc          # Store copy of the QuantumCircuit
        self.creg = creg      # Store the ClassicalRegister for classical bits
        self.free_cbit = []   # Store free classical bits
        self.in_use_cbit = {} # Store in-use classical bits

        self.initilize_classical_bits()

    def initilize_classical_bits(self) -> None:
        """
        Mark all classical bits as free.
        """
        for cbit in self.creg:
            self.free_cbit.append(cbit)

    def allocate_cbit(self) -> Clbit:
        """
        Allocate a classical bit for a measurement operation.
        """
        if len(self.free_cbit) == 0:
            # Add a new classical register of size 1
            idx = len(self.creg)
            new_creg = ClassicalRegister(1, name=f"cl_{idx}")
            self.qc.add_register(new_creg)
            self.creg = new_creg
            self.free_cbit.append(new_creg[0])

        cbit = self.free_cbit.pop(0)
        self.in_use_cbit[cbit] = True
        return cbit

    def release_cbit(self, cbit: Clbit) -> None:
        """
        Release a classical bit after use.
        """
        if cbit in self.in_use_cbit:
            del self.in_use_cbit[cbit]
            self.free_cbit.insert(0, cbit)


# -------------------------------------------------------------------
# DataQubitManager
# -------------------------------------------------------------------
class DataQubitManager:
    """
    Manages data qubits for teleportation of quantum states. Allocates and releases data qubits as needed,
    tracking which slots are free and which logical qubits are mapped to which slots.
    """
    def __init__(
        self,
        partition_qregs: list[QuantumRegister],
        num_qubits_log: int,
        partition_assignment: list[list],
        qc: QuantumCircuit,
        network = None
    ):
        self.qc = qc
        self.network = network
        self.partition_qregs = partition_qregs
        self.num_qubits_log = num_qubits_log
        self.in_use_data = {}
        self.free_data = {}
        self.partition_assignment = partition_assignment
        self.log_to_phys_idx = {}
        self.num_partitions = len(partition_qregs)
        # self.linked_comm_qubits = {i : {} for i in range(self.num_qubits_log)}
        self.num_data_qubits_per_partition = []
        self.active_roots = {}
        self.queue = {}
        self.groups = {}
        self.active_receivers = {}
        self.relocated_receivers = {}
        self.local_swap_count = 0

        self.initialise_data_qubits()
        self.initial_placement(partition_assignment)
        self.inital_qubit_placement = copy.deepcopy(self.log_to_phys_idx)
        self.reg_name_to_partition = {reg.name: p for p, reg in enumerate(self.partition_qregs)}

    def locate_data_qubit(self, qubit: Qubit) -> tuple[int, int] | None:
        """
        Returns the partition and index of a data qubit if it belongs to a data register, or None otherwise.
        """
        p = self.reg_name_to_partition.get(qubit._register.name)
        if p is None:
            return None
        return p, qubit._index

    def locate_comm_qubit(self, qubit: Qubit) -> tuple[int, int] | None:
            """
            Returns the partition and index of a comm qubit if it belongs to a comm register, or None otherwise.
            """
            name = qubit._register.name
            if not name.startswith("C"):
                return None
            p = int(name[1:].split("_")[0])
            if name != f"C{p}_0":
                return None # created register
            return p, qubit._index

    def route_to_adjacency(self, p: int, qubit: Qubit, target_idx: int) -> Qubit:
        if self.network is None:
            return qubit
        topology = self.network.qpu_topologies.get(p)
        if topology is None:
            return qubit
        if topology.has_edge(qubit._index, target_idx):
            return qubit
    
        reg = self.partition_qregs[p]
        path = find_swap_path(topology, qubit._index, target_idx)
        for a, b in path:
            self.qc.swap(reg[a], reg[b])
            self.swap_physical_slots(p, reg[a], reg[b])
            self.local_swap_count += 1
        return reg[path[-1][1]]

    def route_to_port(self, p, data_qubit, comm_k):
        if self.network is None or not self.network.comm_constrained(p):
            return data_qubit
        if self.locate_data_qubit(data_qubit) is None:
            return data_qubit
        return self.route_to_adjacency(p, data_qubit, self.network.comm_node(p, comm_k))

    def initialise_data_qubits(self) -> None:
        """
        Initialize the free_data and in_use_data dictionaries.
        """
        for p in range(self.num_partitions):
            reg = self.partition_qregs[p]
            num_qubits_p = len(reg)
            self.free_data[p] = [qubit for qubit in reg]
            self.in_use_data[p] = {}
            self.num_data_qubits_per_partition.append(num_qubits_p)

    def initial_placement(self, partition_assignment: list[list]) -> None:
        """
        At t=0, place each logical qubit in the partition specified by partition_assignment[0].
        """
        for q in range(self.num_qubits_log):
            part0 = partition_assignment[0][q]
            qubit0 = self.allocate_data_qubit(part0)
            self.assign_to_physical(part0, qubit0, q)

    def allocate_data_qubit(self, p: int) -> Qubit:
        """
        Allocate a free data qubit slot in partition p.
        """
        # if not self.free_data[p]:
        #     logger.warning(f"[allocate_data_qubit] No free data qubits in partition {p}; adding new QRegister.")
        #     # Create a new data qubit in partition p
        #     idx = len(self.partition_qregs[p])
        #     new_reg = QuantumRegister(1, name=f"part{p}_data_{idx}")
        #     self.partition_qregs[p].append(new_reg)
        #     self.qc.add_register(new_reg)
        #     new_qubit = new_reg[0]
        #     self.free_data[p].append(new_qubit)

        qubit = self.free_data[p].pop(0)
        return qubit

    def assign_to_physical(self, part: int, qubit_phys: Qubit, qubit_log: int):
        """
        Assign a logical qubit to a physical qubit slot in a partition.
        """
        self.log_to_phys_idx[qubit_log] = qubit_phys
        self.in_use_data[part][qubit_phys] = qubit_log

    def release_data_qubit(self, p: int, qubit: Qubit) -> None:
        """
        Release a data qubit, clearing any state. 
        Note: Qiskit doesn't have a direct 'free' notion, so we reset or reuse.
        """

        if qubit in self.in_use_data[p]:
            log_qubit = self.in_use_data[p].pop(qubit)
            del self.log_to_phys_idx[log_qubit]
            self.qc.reset(qubit)
            self.free_data[p].append(qubit)
        # """
        # Release a data qubit after the state has been teleported to another partition.
        # """
        # if qubit in self.in_use_data[p]:
        #     del self.in_use_data[p][qubit] # Remove the logical qubit from the in_use_data dictionary
        if qubit not in self.free_data[p]:
            self.free_data[p].append(qubit) # Add the slot to the free_data list

    def swap_physical_slots(self, p: int, qubit_a: Qubit, qubit_b: Qubit):
        """
        Swap the contents of two physical slots in partition p.
        """
        a_in_use = qubit_a in self.in_use_data[p]
        b_in_use = qubit_b in self.in_use_data[p]

        # No qubits in use
        if not a_in_use and not b_in_use:
            return

        # Both qubits in use
        elif a_in_use and b_in_use:
            logical_a = self.in_use_data[p][qubit_a]
            logical_b = self.in_use_data[p][qubit_b]

            self.log_to_phys_idx[logical_a] = qubit_b
            self.log_to_phys_idx[logical_b] = qubit_a
            self.in_use_data[p][qubit_a] = logical_b
            self.in_use_data[p][qubit_b] = logical_a

            self._update_group_links(qubit_a, qubit_b)
            return

        # One qubit in use, one free
        if b_in_use:
            qubit_a, qubit_b = qubit_b, qubit_a  # Swap to ensure qubit_a is in use and qubit_b is free

        logical_q = self.in_use_data[p].pop(qubit_a)
        self.in_use_data[p][qubit_b] = logical_q
        self.log_to_phys_idx[logical_q] = qubit_b
        self._update_group_links(qubit_a, qubit_b)
        self.free_data[p].remove(qubit_b)
        self.free_data[p].append(qubit_a)

    def _update_group_links(self, qubit_a: Qubit, qubit_b: Qubit) -> None:
        """
        The contents of slots qubit_a and qubit_b have been exchanged; any
        gate-group reference pointing at either slot must follow its state.
        """
        for group_info in self.groups.values():
            linked = group_info.get('linked_qubits')
            if not linked:
                continue
            for part, linked_qubit in linked.items():
                if linked_qubit == qubit_a:
                    linked[part] = qubit_b
                elif linked_qubit == qubit_b:
                    linked[part] = qubit_a

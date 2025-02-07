from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit, Clbit

class CommunicationQubitManager:
    """
    Manages communication qubits on a per-partition basis. Allocates communication qubits for tasks requring entanglement and releases them when done.
    """
    def __init__(self, comm_qregs : dict, qc: QuantumCircuit):
        self.qc = qc # Store copy of the QuantumCircuit
        self.comm_qregs = comm_qregs # Store the QuantumRegisters for communication qubits
        self.free_comm = {} # Store free communication qubits for each partition
        self.in_use_comm = {} # Store in-use communication qubits for each partition
        self.linked_qubits = {} # Store comm qubits linked to root qubits for gate teleportation

        self.initilize_communication_qubits()
    
    def initilize_communication_qubits(self,) -> None:
        """
        Set all communication qubits to free."""
        ""
        for p, reg_list in self.comm_qregs.items():
            self.free_comm[p] = []
            self.in_use_comm[p] = []
            for reg in reg_list:
                for qubit in reg:
                    self.free_comm[p].append(qubit)

    def find_comm_idx(self, p: int) -> Qubit:
        """
        Allocate free communication qubit in partition p.
        """
        free_comm_p = self.free_comm.get(p, [])
        if free_comm_p != []: # If there are free communication qubits in partition p then allocate one
            comm_qubit = free_comm_p.pop(0)
        else:
            # Create a new communication qubit by adding a new register of 1 qubit.
            num_regs_p = len(self.comm_qregs[p])
            new_reg = QuantumRegister(1, name=f"comm_{p}_{num_regs_p}")
            self.comm_qregs[p].append(new_reg)
            self.qc.add_register(new_reg)
            comm_qubit = new_reg[0]
        # Mark as in use
        self.in_use_comm[p].append(comm_qubit)
        return comm_qubit

    def allocate_comm_qubits(self, root_q, p_root, p_set_rec):
        """
        Allocate communication qubits for multi-gate teleportation.
        Allocate one communication qubit in p_root, plus one in each partition in p_set_rec.
        Link them to root_q as needed.
        """
        comm_root = self.find_comm_idx(p_root)
        comm_rec_dict = {}

        for p_rec in p_set_rec:
            comm_rec = self.find_comm_idx(p_rec)
            comm_rec_dict[p_rec] = comm_rec
            # Mark that comm_rec is "linked" to root_q
            self.linked_qubits[comm_rec] = root_q

        return comm_root, comm_rec_dict

    def release_comm_qubit(self, p : int, comm_qubit : Qubit) -> None:
        """
        Resets the qubit and returns it to the free pool in partition p.
        """
        if comm_qubit in self.in_use_comm[p]:
            self.in_use_comm[p].remove(comm_qubit)
            self.free_comm[p].insert(0, comm_qubit)
        else:
            print(f"  WARNING: Tried to release comm_qubit {comm_qubit} not found in in_use_comm[{p}]")

    def get_status(self, p : int) -> tuple[list,list]:
        """
        Return a tuple (in_use, free) for partition p.
        """
        return self.in_use_comm.get(p, []), self.free_comm.get(p, [])

class ClassicalBitManager:
    """
    Manages classical bits, allocating from a pool and releasing after use.
    """
    def __init__(self, qc: QuantumCircuit, creg : ClassicalRegister):
        self.qc = qc # Store copy of the QuantumCircuit
        self.creg = creg # Store the ClassicalRegister for classical bits
        self.free_cbit = [] # Store free classical bits
        self.in_use_cbit = {} # Store in-use classical bits

        self.initilize_classical_bits()

    def initilize_classical_bits(self,) -> None:
        """
        Mark all classical bits as free.
        """
        for cbit in self.creg:
            self.free_cbit.append(cbit)

    def allocate_cbit(self,) -> Clbit:
        """
        Allocate a classical bit for a measurement operation.
        """
        if len(self.free_cbit) == 0: # If there are no free classical bits then create a new register of 1 classical bit
            # Add a new classical register of size 1
            idx = len(self.creg)
            new_creg = ClassicalRegister(1, name=f"c_{idx}")
            self.qc.add_register(new_creg)
            self.creg = new_creg  # or you might want to append to a list
            self.free_cbit.append(new_creg[0])

        cbit = self.free_cbit.pop(0)
        self.in_use_cbit[cbit] = True # Mark as in use

        return cbit
    
    def release_cbit(self, cbit : Clbit) -> None:
        """
        Release a classical bit after use.
        """
        if cbit in self.in_use_cbit:
            del self.in_use_cbit[cbit]
            self.free_cbit.insert(0, cbit) # Add to the front of the list
        else:
            print(f"  WARNING: Tried to release cbit={cbit} which was not in in_use_cbit")

class DataQubitManager:
    """
    Manages data qubits for teleportation of quantum states. Allocates and releases data qubits as needed,
    tracking which slots are free and which logical qubits are mapped to which slots.
    """
    def __init__(self, partition_qregs : list[QuantumRegister],
                num_qubits_log : int,
                partition_assignment : list[list],
                qc : QuantumCircuit):
        
        self.qc = qc # Store copy of the QuantumCircuit
        self.partition_qregs = partition_qregs # Store the QuantumRegisters for each partition
        self.num_qubits_log = num_qubits_log # Store the number of logical qubits
        self.in_use_data = {} # Store in-use data qubits for each partition
        self.free_data = {} # Store free data qubits for each partition
        self.partition_assignment = partition_assignment # Store the partition assignment for each logical qubit at each time-step
        self.log_to_phys_idx = {} # Store the mapping of logical qubits to physical qubits

        self.num_partitions = len(partition_qregs) # Store the number of partitions
        self.usage_counters = [0]*self.num_partitions # Track the number of data qubits in use for each partition
        self.num_data_qubits_per_partition = [] # Store the number of data qubits in each partition

        # Build free_data/in_use_data for each partition
        self.initialise_data_qubits()

        # Place each logical qubit according to partition_assignment at t=0
        self.initial_placement(partition_assignment)

    def initialise_data_qubits(self,) -> None:
        """
        Initialize the free_data and in_use_data dictionaries.
        """
        for p in range(self.num_partitions):
            reg = self.partition_qregs[p] # Get the QuantumRegister for partition p
            num_qubits_p = len(reg) # Get the number of qubits in the QuantumRegister
            self.free_data[p] = list(range(num_qubits_p)) # Fill the free data qubits with all slots in the QuantumRegister
            self.in_use_data[p] = {} # Initialize the in_use_data dictionary for partition p as empty
            self.num_data_qubits_per_partition.append(num_qubits_p) # Store the number of data qubits in partition p

    def allocate_data_qubit_slot(self, p : int) -> int:
        """
        Allocate a free data qubit slot in partition p.
        """
        slot = self.free_data[p].pop(0)  # Grab first free slot in the partition p
        self.usage_counters[p] += 1 # Increment the usage counter for partition p
        return slot
    
    def assign_to_physical(self, part, slot, qubit_log):
        """
        Assign a logical qubit to a physical qubit slot in a partition.
        """
        qubit_phys = self.partition_qregs[part][slot] # Get the physical qubit object using the slot
        self.log_to_phys_idx[qubit_log] = qubit_phys # Store the mapping of logical qubit to physical qubit
        self.in_use_data[part][slot] = qubit_log # Mark the slot as in use

    def initial_placement(self, partition_assignment : list[list]) -> None:
        """
        At t=0, place each logical qubit in the partition specified by partition_assignment[0].
        """
        for q in range(self.num_qubits_log):
            part0 = partition_assignment[0][q] # Get the partition for logical qubit q at t=0
            slot0 = self.allocate_data_qubit_slot(part0) # Allocate a free slot in partition part0
            self.assign_to_physical(part0, slot0, q) # Assign the logical qubit q to the slot in partition part0

    def release_data_qubit(self, p : int, qubit : Qubit) -> None:
        """
        Release a data qubit after the state has been teleported to another partition.
        """
        slot = qubit._index  # Return the index (slot number) within the QuantumRegister
        qubit_log = self.in_use_data[p].pop(slot) # Remove the logical qubit from the in_use_data dictionary
        self.free_data[p].append(slot) # Add the slot to the free_data list
        self.usage_counters[p] -= 1 # Decrement the usage counter for partition p
        # Also remove from the logical qubit from the log_to_phys_idx dictionary since it is
        if qubit_log in self.log_to_phys_idx:
            del self.log_to_phys_idx[qubit_log]
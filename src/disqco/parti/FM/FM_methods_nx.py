
import random
import numpy as np
from disqco.graphs.hypergraph_methods import *
from disqco.graphs.quantum_network import QuantumNetwork
import networkx as nx
from collections import deque
from disqco.parti.fgp.fgp_roee import calculate_W_matrix_cols, calculate_D_from_W

def set_initial_partition_assignment(graph : nx.Graph,
                                    qpu_sizes : dict[int, int],
                                    method = 'greedy') -> np.ndarray:

    node_count = graph.number_of_nodes()
    static_assignment = np.zeros(node_count,dtype=int)
    num_partitions = len(qpu_sizes)
    if method == 'greedy':
        counter = 0
        breaker = False
        for n in range(num_partitions):
            for k in range(qpu_sizes[n]):
                static_assignment[counter] = n
                counter += 1
                if counter >= node_count:
                    breaker = True
                    break
            if breaker:
                break
    elif method == 'round_robin':
        for n in range(node_count):
            static_assignment[n] = n % num_partitions

    return static_assignment

def find_spaces(assignment : np.ndarray, qpu_sizes: dict[int, int], graph : nx.Graph) -> dict[int, int]:
    """
    Find the number of free qubits in each partition at each time step.
    num_qubits: number of logical qubits in the circuit
    assignment: function that maps qubits to partitions
    network: quantum network object
    """

    spaces = {k: v for k, v in qpu_sizes.items()}
    

    if graph is not None:
        for node in graph.nodes():
            part = assignment[node]
            spaces[part] -= 1

    return spaces

def find_spaces_with_ancillae(assignment : np.ndarray, qpu_sizes: dict[int, int], graph : nx.Graph) -> dict[int, int]:
    """
    Find the number of free qubits in each partition at each time step.
    num_qubits: number of logical qubits in the circuit
    assignment: function that maps qubits to partitions
    network: quantum network object
    """

    spaces = {k: v for k, v in qpu_sizes.items()}
    

    for node in graph.nodes():
        part = assignment[node]
        spaces[part] -= 1

    for edge in graph.edges():
        node1 = edge[0]
        node2 = edge[1]
        part1 = assignment[node1]
        part2 = assignment[node2]

        if part1 != part2:
            spaces[part1] -= 1
            spaces[part2] -= 1
        else:
            spaces[part1] -= 1

    return spaces

def check_valid(graph : nx.Graph, node : tuple[int,int], assignment, destination: int, spaces: dict[int, int], ancilla: bool = False) -> bool:
    """
    Check if the destination partition has free data qubit slots.
    node: tuple of (qubit index, time step)
    destination: destination partition
    spaces: dictionary of free qubit slots in each partition at each time step
    """
    if not ancilla:
        if spaces[destination] > 0:
            return True
        return False
    else:
        source = assignment[node]
        new_ancilla_counter = 0
        for neighbour in graph.neighbors(node):
            neighbour_source = assignment[neighbour]
            if neighbour_source != destination:
                new_ancilla_counter += 1

        if spaces[destination] - new_ancilla_counter > 1:
            return True

    return False

def move_node(node: int, 
              destination: int, 
              assignment: np.ndarray, 
              ) -> np.ndarray:
    """ 
    Move a node to a new destination partition by updating the assignment.
    node: tuple of (qubit index, time step)
    destination: destination partition
    assignment: function that maps qubits to partitions
    """
    assignment_new = assignment.copy()  
    assignment_new[node] = destination

    return assignment_new

def calculate_W_matrix(graph):
    "Calculate the weight matrix of the graph."
    max_node_index = max(graph.nodes())+1
    w_matrix = np.zeros((max_node_index,max_node_index))
    for edge in graph.edges():
        qubit1 = edge[0]
        qubit2 = edge[1]
        w_matrix[qubit1][qubit2] = graph.edges()[edge].get('weight', 1)
        w_matrix[qubit2][qubit1] = graph.edges()[edge].get('weight', 1)
    return w_matrix

# def calculate_W_matrix_cols(graph, W, num_partitions, qpu_sizes, partition):
#     """Calculate the weight matrix columns efficiently - O(E) instead of O(n²)"""
#     max_node_index = max(graph.nodes()) + 1
#     num_partitions = max(qpu_sizes.keys()) + 1
#     W_cols = np.zeros((max_node_index, num_partitions))
    
#     # Only iterate over actual neighbors instead of all node pairs
#     # This is much faster for sparse graphs (O(E) vs O(n²))
#     for node in graph.nodes():
#         for neighbor in graph.neighbors(node):
#             weight = graph[node][neighbor].get('weight', 1)
#             neighbor_partition = partition[neighbor]
#             W_cols[node][neighbor_partition] += weight
    
#     return W_cols

def calculate_W_matrix_cols(graph, W, num_partitions, qpu_sizes, partition):
    """Vectorized numpy implementation"""
    max_node_index = max(graph.nodes()) + 1
    num_partitions = max(qpu_sizes.keys()) + 1
    W_cols = np.zeros((max_node_index, num_partitions))
    
    # Use numpy advanced indexing for speed
    for i in graph.nodes():
        for p in range(num_partitions):
            # Find all nodes in partition p
            nodes_in_partition = np.where(partition == p)[0]
            if len(nodes_in_partition) > 0:
                W_cols[i][p] = np.sum(W[i][nodes_in_partition])
    
    return W_cols

def calculate_D_from_W(graph, W_cols, partition, num_partitions, qpu_sizes, max_gain):
    "Calculate the D matrix from the W matrix."
    num_qubits = len(W_cols)
    num_partitions = max(qpu_sizes.keys()) + 1
    D = np.zeros((num_qubits,num_partitions), dtype=int)
    D -= (max_gain + 1)
    for i in graph.nodes():
        col_i = partition[i]
        for k in qpu_sizes.keys():
            D[i][k] = W_cols[i][k] - W_cols[i][col_i]
    return D

def find_all_gains(assignment: np.ndarray, 
                   num_partitions: int,
                   qpu_sizes: dict[int, int], 
                   W: np.ndarray,
                   graph: nx.Graph,
                   max_gain: int) -> np.ndarray:
    # Calculate weight matrix

    W_cols = calculate_W_matrix_cols(graph, W, num_partitions, qpu_sizes, assignment)
    # Calculate D values from weights
    D = calculate_D_from_W(graph, W_cols, assignment, num_partitions, qpu_sizes, max_gain)
    return D

def fill_buckets(D : np.ndarray, assignment: np.ndarray, num_partitions : int, max_gain : int):
    buckets = []
    # Initialize buckets for gains from -max_gain to max_gain
    for i in range(0, 2 * max_gain + 2):
        buckets.append({
            'queue': deque(),
            'set': set()
        })
    for node in range(D.shape[0]):
        source = int(assignment[node])
        external_partitions = set(range(num_partitions)) - {source}
        for part in external_partitions:
            gain = D[node][part]
            if gain < -max_gain:
                continue
            if gain == 0:
                continue
            bucket_index = int(gain + max_gain)
            action = (int(node), int(part))
            buckets[bucket_index]['queue'].append(action)
            buckets[bucket_index]['set'].add(action)

    return buckets

def find_member_random(member_set: set):
    # Choose random element from set
    if not member_set:
        return None
    member = random.choice(list(member_set))
    member_set.remove(member)
    return member

def update_D_matrix_neighbour(neighbor,node, source, destination, neighbour_source, neighbour_dest, D,W):
    """Update the D matrix after a move. Only neighbours of the moved node change in the source
    and destination columns """      
    if neighbour_source == source:
        D[neighbor][neighbour_dest] += W[node][neighbor]
    if neighbour_source == destination:
        D[neighbor][neighbour_dest] -= W[node][neighbor]
    if neighbour_dest == destination:
        D[neighbor][neighbour_dest] += W[node][neighbor]
    if neighbour_dest == source:
        D[neighbor][neighbour_dest] -= W[node][neighbor]

    return D

def G_matrix(graph : nx.Graph) -> np.ndarray:
    """G matrix is a a dynamic matrix that tracks the distance each node is from each other node in the network.
    The network scales the cost of nodes"""

def find_action(graph, buckets, assignment, spaces, max_gain, random=True, ancilla=False):
    # Iterate through gain values from highest to lowest (reversed order for best-first)
    for i in range(0, 2 * max_gain + 1)[::-1]:
        bucket_index = i  # Convert gain to bucket index
        bucket = buckets[bucket_index]
        
        while bucket['queue']:
            if not random:
                action = bucket['queue'].popleft()  # O(1) removal from front
            else:
                # For random case, use random selection from set
                # action = find_member_random(bucket['set'])
                if np.random.random() < 0.5:
                    action = bucket['queue'].popleft()
                else:
                    action = bucket['queue'].pop()


            # Check if action is still valid (not moved to another bucket)
            if action not in bucket['set']:
                # Action was moved elsewhere - it's already removed from queue above
                # Continue to next iteration (lazy cleanup)
                continue
                
            node, destination = action
            if check_valid(graph, node, assignment, destination, spaces, ancilla=ancilla):
                gain = bucket_index - max_gain  # Convert bucket index back to gain
                bucket['set'].remove(action)  # O(1) removal from set
                return node, destination, gain
            else:
                bucket['set'].remove(action)  # O(1) removal from set
    
    return None, None, None

def update_spaces(source,destination,spaces):
    spaces[destination] -= 1
    spaces[source] += 1

def update_ancilla_spaces(source,destination,neighbour_source,spaces):
    if neighbour_source == source:
        spaces[destination] -= 1
    elif neighbour_source == destination:
        spaces[destination] += 1
    else:
        spaces[source] += 1
        spaces[destination] -= 1

def move_action_between_buckets(action, old_gain, new_gain, buckets, max_gain):
    """Move an action from one bucket to another - O(1) operation"""
    old_bucket_index = int(old_gain + max_gain)
    new_bucket_index = int(new_gain + max_gain)
    
    # Remove from old bucket's set (O(1))
    buckets[old_bucket_index]['set'].discard(action)
    
    # Add to new bucket (O(1) for both)
    buckets[new_bucket_index]['queue'].append(action)
    buckets[new_bucket_index]['set'].add(action)

def take_action_and_update(graph : nx.Graph,
                           node : int,
                           source : int,
                           destination : int,
                           assignment : np.ndarray,
                           num_partitions : int,
                           W : np.ndarray,
                           D : np.ndarray,
                           buckets : list[list],
                           max_gain : int,
                           locked : set[int],
                           spaces : dict[int, int],
                           ancillae : bool = False
                           ) -> np.ndarray:

    assignment_new = move_node(node,destination,assignment)
    # print(f'Moving node {node} from partition {source} to {destination}')
    unlocked_neighbours = set(list(graph.neighbors(node))) - locked
    for neighbor in unlocked_neighbours:
        neighbour_source = assignment[neighbor]
        for neighbor_dest in range(num_partitions):
            new_action = (neighbor, neighbor_dest)
            old_gain = D[neighbor][neighbor_dest]
            D = update_D_matrix_neighbour(neighbor,node, 
                                          source, destination, 
                                          neighbour_source, neighbor_dest, D, W)
            
            new_gain = D[neighbor][neighbor_dest]
            move_action_between_buckets(new_action, old_gain, new_gain, buckets, max_gain)

        if ancillae:
            update_ancilla_spaces(source, destination, neighbour_source, spaces)

    update_spaces(source, destination, spaces)
    assignment = assignment_new
    return assignment_new

def lock_node(node : int, locked : set[int]):
    locked.add(node)

def calculate_cut_size(graph: nx.Graph, assignment: np.ndarray):
    """Calculate the cut size of a partition"""
    cut_size = 0
    for u, v in graph.edges():
        if assignment[u] != assignment[v]:
            weight = graph[u][v].get('weight', 1)
            cut_size += weight
    return cut_size


def set_sparse_assignment(subgraph: nx.Graph, full_graph: nx.Graph, global_assignment: np.ndarray, qpu_sizes: dict):

    qpus = list(qpu_sizes.keys())
    for i, node in enumerate(subgraph.nodes()):
        global_assignment[node] = qpus[i%len(qpus)]
    return global_assignment
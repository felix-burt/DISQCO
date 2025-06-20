from disqco.parti.FM.FM_methods_ext import *
from disqco.graphs.GCP_hypergraph_extended import HyperGraph
from typing import Hashable, Iterable
import numpy as np

from typing import Hashable, Iterable
# ----------------- helpers -----------------
def _build_buckets(gain_dict: dict[tuple[Hashable, int], int], max_gain: int) -> dict[int, set[tuple[Hashable, int]]]:
    """
    Gain → { (node, destination), … } map.
    """
    buckets = {g: set() for g in range(-max_gain, max_gain + 1)}
    for action, g in gain_dict.items():
        if -max_gain <= g <= max_gain:
            buckets[g].add(action)
    return buckets

def check_counts(graph, qubit_assignment, gate_assignment, num_partitions):
    for edge in graph.hyperedges():
        edge_key = edge.key
        counts = graph.get_hyperedge_attrs(edge_key, 'counts')
        counts_actual = get_edge_count(graph, edge, qubit_assignment, gate_assignment, num_partitions)
        if not np.array_equal(counts, counts_actual):
            print(f'Counts do not match for edge {edge}')
            print(f'Counts: {counts}, Actual counts: {counts_actual}')
            return False
    return True

# def check_gains(graph, qubit_assignment, gate_assignment, num_partitions, gain_dict):
#     for action, gain in gain_dict.items():
#         node = action[0]
#         destination = action[1]
#         if len(node) == 2:
#             source = qubit_assignment[node]
#         else:
#             source = gate_assignment[node]
#         gain_actual = find_gain_raw(graph, node, qubit_assignment, gate_assignment, destination)
#         if gain != gain_actual:
#             print(f'Gain does not match for action {action}')
#             print(f'Gain: {gain}, Actual gain: {gain_actual}')
#             raise ValueError(f'Gain does not match for action {action}')
#             return False
#     return True


def _unlock_all(graph: HyperGraph) -> set:
    """Handy shortcut instead of the old lock‑dict."""
    return set()


def FM_pass(graph: HyperGraph,
            max_gain: int,
            qubit_assignment: np.ndarray,
            gate_assignment: np.ndarray,
            num_partitions : int,
            qpu_sizes : list[int],
            limit: int,
            active_nodes: Iterable[Hashable] | None = None
):
    """
    One Kernighan–Lin / Fiduccia–Mattheyses pass on *graph*.
    Mirrors the original logic but uses the new gain‑/cost API.
    """
    if active_nodes is None:
        active_nodes = list(graph)                   # all vertices


    # --- initial bookkeeping on edges --------------------------------
    # assign_all_counts_and_configs(graph, assignment, num_partitions)
    locked      = _unlock_all(graph)
    
    gain_dict = find_all_gains(graph, qubit_assignment, gate_assignment, num_partitions, locked)
    # for action, gain in gain_dict.items():
    #     print(f'Checking action: {action}, gain: {gain}')
    buckets   = _build_buckets(gain_dict, max_gain)
    spaces    = find_spaces(graph, qubit_assignment, len(qubit_assignment[0]), qpu_sizes)

    # print(f'Initial assignment: {assignment}')
    # print(f'Spaces: {spaces}')

    # --- iterative improvement ---------------------------------------
    locked      = _unlock_all(graph)
    cum_gain    = 0
    gain_list   = [0]
    assign_hist = [(qubit_assignment.copy(), gate_assignment.copy() )]

    h = 0
    while h < limit:

        out = choose_action(buckets, qubit_assignment, locked,
                                     max_gain, spaces)
        if out is None:
            break
        action, gain = out
        # print(f'Action: {action}, Gain: {gain}')
        if action is None:                          # nothing movable
            break

        # for node in assignment:
        #     print(f'Node {node} is in partition {assignment[node]}')
        
        node, dest           = action
        
        if is_gate_node(node):
            src = gate_assignment[node]
        else:
            src = qubit_assignment[node]
        
        cum_gain            += gain

        # print(f'Node {node} is moving from {src} to {dest}')

        locked.add(node)              # mutate in‑place

        gain_dict, buckets   = take_action_and_update_neighbours(
                                    graph, action, qubit_assignment, gate_assignment,
                                    gain_dict, buckets, locked, num_partitions
                               )
        
        # print(check_counts(graph, assignment, num_partitions))

        # print(check_gains(graph, assignment, num_partitions, gain_dict))
        

        
        # assign_all_counts_and_configs(graph, assignment, num_partitions)
        # gain_dict = find_all_gains(graph, assignment, num_partitions, locked)
        # buckets   = _build_buckets(gain_dict, max_gain)
        
        update_spaces(node, src, dest, spaces)
        # print(f'Updated spaces: {spaces}')
        gain_list.append(cum_gain)
        assign_hist.append((qubit_assignment.copy(), gate_assignment.copy()))
        h += 1
        # print(f'Cumulative gain: {cum_gain}')

    return assign_hist, gain_list


def run_FM(graph: HyperGraph,
           initial_qubit_assignment: np.ndarray,
           initial_gate_assignment: np.ndarray,
           qpu_info,
           *,
           passes: int = 10,
           max_gain: int = 4,
           limit: int | None = None,
           stochastic: bool = True,
           log: bool = False,
           add_initial: bool = False,
):
    """
    Multi‑pass FM optimiser (rewritten for the new edge‑cost API).
    """
    if isinstance(qpu_info, dict):
        qpu_sizes = [list(qpu_info.values())]
    else:
        qpu_sizes = qpu_info                              # list or list[list]

    num_partitions = len(qpu_sizes if isinstance(qpu_sizes, (list, tuple)) else list(qpu_sizes.values()))
    # initial_assignment = np.asarray(initial_assignment, dtype=int)
    
    if limit is None:
        limit = graph.node_count           # 12.5 % of vertices

    best_assignments: list[tuple[np.ndarray, np.ndarray]] = []
    cost_list:       list[int]        = []
    initial_cost = calculate_full_cost(graph, initial_qubit_assignment, initial_gate_assignment, num_partitions)

    if add_initial:
        best_assignments.append((initial_qubit_assignment.copy(), initial_gate_assignment.copy()))
        cost_list.append(int(initial_cost))

    if log:
        print("Initial cost:", initial_cost)

    current_assign = (initial_qubit_assignment.copy(), initial_gate_assignment.copy())
    current_cost   = calculate_full_cost(graph, current_assign[0], current_assign[1], num_partitions)

    for p in range(passes):
        assigns, gains = FM_pass(graph, max_gain, current_assign[0].copy(), current_assign[1].copy(),
                                 num_partitions, qpu_sizes,     limit)
        if stochastic:
            print(f"Pass {p:2d}: cost = {current_cost}")
            if p % 2 == 0:                         # odd → exploit
                print("Exploiting")
                idx = int(np.argmin(gains))
            else:             
                # print("Exploring")               # even → explorez
                idx = -1
        else:
            idx = int(np.argmin(gains))
        # print(f"Idx: {idx}")
        current_assign = (assigns[idx][0].copy(), assigns[idx][1].copy())
        # print(f'Gains: {gains}')
        current_cost  += gains[idx]
        # print(f'Gains idx: {gains[idx]}')
        best_assignments.append((current_assign[0].copy(), current_assign[1].copy()))
        cost_list.append(int(current_cost))

        if log:
            print(f"Pass {p:2d}: cost = {current_cost}")

    # ---------------- result -----------------
    best_idx   = int(np.argmin(cost_list))
    return cost_list[best_idx], best_assignments[best_idx], cost_list



# Define a basic multilevel partitioning function to call the FM algorithm for each graph in the list. Qubit assignment must be transformed between each level.
def refine_assignment(level, num_levels, qubit_assignment, mapping_list):
    new_assignment = qubit_assignment
    print(f'Refining assignment for level {level}')
    print(f'Qubit assignment: {qubit_assignment}')
    if level < num_levels -1:
        mapping = mapping_list[level]
        print(f'Mapping: {mapping}')
        for super_node_t in mapping:
            for t in mapping[super_node_t]:
                
                print(f'Qubits {qubit_assignment[:,t]}')

                new_assignment[:,t] = qubit_assignment[:,super_node_t]
    print(f'New qubit assignment: {new_assignment}')
    return new_assignment



def partition_multilevel(graph_list, mapping_list, qubit_assignment, gate_assignment, qpu_sizes, num_partitions):
    num_levels = len(graph_list)
    min_cost = float('inf')
    for i, graph in enumerate(graph_list[::-1]):
        # Get the number of qubits and depth of the graph
        num_qubits = graph.num_qubits
        # Get the number of qubits and depth of the graph
        depth = graph.depth
        def find_max_node_degree(mapping):
            max_degree = 0  
            for super_node_t in mapping:
                if len(mapping[super_node_t]) > max_degree:
                    max_degree = len(mapping[super_node_t])
            return max_degree
        
        max_node_degree = find_max_node_degree(mapping_list[i])
        cost, best_assignments, cost_trace = run_FM(graph, qubit_assignment, gate_assignment, qpu_sizes, passes=10, max_gain=4*max_node_degree, limit=len(graph.nodes()), stochastic=False, log=True)

        qubit_assignment = best_assignments[0]

        gate_assignment = best_assignments[1]

        if cost < min_cost:
            min_cost = cost
            best_qubit_assignment = qubit_assignment
            best_gate_assignment = gate_assignment

        qubit_assignment = refine_assignment(i, len(graph_list), qubit_assignment, mapping_list)

    return min_cost, best_qubit_assignment, best_gate_assignment
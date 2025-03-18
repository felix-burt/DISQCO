from disqco.graphs.hypergraph_methods import *
from disqco.parti.FM.FM_methods import *
import copy

def find_joint_hyperedge_cost(hypergraph,edge1,edge2,assignment,costs):
    root_counts1 = hypergraph.get_hyperedge_attribute(edge1,'root_counts')
    rec_counts1 = hypergraph.get_hyperedge_attribute(edge1,'rec_counts')

    root_config1 = hypergraph.get_hyperedge_attribute(edge1,'root_config')
    rec_config1 = hypergraph.get_hyperedge_attribute(edge1,'rec_config')
    config1 = get_full_config(root_config1,rec_config1)
    cost1 = costs[config1]

    root_counts2 = hypergraph.get_hyperedge_attribute(edge2,'root_counts')
    rec_counts2 = hypergraph.get_hyperedge_attribute(edge2,'rec_counts')

    root_config2 = hypergraph.get_hyperedge_attribute(edge2,'root_config')
    rec_config2 = hypergraph.get_hyperedge_attribute(edge2,'rec_config')
    config2 = get_full_config(root_config2,rec_config2)
    cost2 = costs[config2]

    info1 = hypergraph.hyperedges[edge1]
    info2 = hypergraph.hyperedges[edge2]

    root_set1 = info1['root_set']
    rec_set1 = info1['receiver_set']

    root_set2 = info2['root_set']
    rec_set2 = info2['receiver_set']

    overlap_12 = root_set1.intersection(rec_set2)
    overlap_21 = root_set2.intersection(rec_set1)
    if overlap_12 != set():
        print("Joint edge", edge1, edge2)
        print("Root set 1", root_set1)
        print("Rec set 1", rec_set1)
        print("Root set 2", root_set2)
        print("Rec set 2", rec_set2)
        rec_counts1_12 = copy.deepcopy(list(rec_counts1))
        rec_counts2_12 = copy.deepcopy(list(rec_counts2))
        print("Rec counts 1", rec_counts1_12)
        print("Rec counts 2",rec_counts2_12)
        for member in overlap_12:
            print("Member", member)
            source = assignment[member[1]][member[0]]
            rec_counts1_12[source] += 1
            rec_counts2_12[source] -= 1
        print("Rec counts 1 post", rec_counts1_12)
        print("Rec counts 2 post",rec_counts2_12)
        rec_config1_12 = update_config(rec_config1,rec_counts1_12,source,source)
        rec_config2_12 = update_config(rec_config2,rec_counts2_12,source,source)
        config1_12 = get_full_config(root_config1, rec_config1_12)
        config2_12 = get_full_config(root_config2,rec_config2_12)    
        cost1_12 = costs[config1_12]
        cost2_12 = costs[config2_12]
    else:
        cost1_12 = cost1
        cost2_12 = cost2
    
    if overlap_21 != set():
        rec_counts2_21 = copy.deepcopy(list(rec_counts2))
        rec_counts1_21 = copy.deepcopy(list(rec_counts1))
        print("Joint edge", edge1, edge2)
        print("Root set 1", root_set1)
        print("Rec set 1", rec_set1)
        print("Root set 2", root_set2)
        print("Rec set 2", rec_set2)
        print("Rec counts 2", rec_counts2_21)
        print("Rec counts 1",rec_counts1_21)
        for member in overlap_21:
            print("Member", member)
            source = assignment[member[1]][member[0]]
            rec_counts2_21[source] += 1
            rec_counts1_21[source] -= 1
        print("Rec counts 2 post", rec_counts2_21)
        print("Rec counts 1 post",rec_counts1_21)

        rec_config2_21 = update_config(rec_config2,rec_counts2_21,source,source)
        rec_config1_21 = update_config(rec_config1,rec_counts1_21,source,source)
        config2_21 = get_full_config(root_config2,rec_config2_21)
        config1_21 = get_full_config(root_config1,rec_config1_21)
        cost2_21 = costs[config2_21]
        cost1_21 = costs[config1_21]
    else:
        cost1_21 = cost1
        cost2_21 = cost2
    
    base_cost = cost1 + cost2
    cost_12 = cost1_12 + cost2_12
    cost_21 = cost1_21 + cost2_21

    return base_cost, cost_12, cost_21
        
def optimise_joint_hyperedges(hypergraph,assignment,num_partitions,costs):
    for edge in hypergraph.hyperedges:
        hypergraph = map_counts_and_configs(hypergraph,assignment,num_partitions,costs)

    total_cost = calculate_full_cost(hypergraph,assignment,num_partitions,costs)
    cost = total_cost
    for edge1 in hypergraph.hyperedges:
        el1 = edge1[0]
        if not isinstance(el1, tuple):
            for edge2 in hypergraph.hyperedges:
                el2 = edge2[0]
                if not isinstance(el2, tuple):
                    if edge1 != edge2:
                        base_cost, cost_12, cost_21 = find_joint_hyperedge_cost(hypergraph,edge1,edge2,assignment,num_partitions,costs)
                        if cost_12 < base_cost or cost_21 < base_cost:
                            lowest_cost = min(base_cost,cost_12,cost_21)
                            cost = cost - base_cost + lowest_cost
    return cost

def find_detached_cost(hypergraph,edge1,edge2,node,assignment,costs):

    root_counts1 = hypergraph.get_hyperedge_attribute(edge1,'root_counts')
    rec_counts1 = hypergraph.get_hyperedge_attribute(edge1,'rec_counts')

    root_config1 = hypergraph.get_hyperedge_attribute(edge1,'root_config')
    rec_config1 = hypergraph.get_hyperedge_attribute(edge1,'rec_config')
    config1 = get_full_config(root_config1,rec_config1)
    cost1 = costs[config1]

    root_counts2 = hypergraph.get_hyperedge_attribute(edge2,'root_counts')
    rec_counts2 = hypergraph.get_hyperedge_attribute(edge2,'rec_counts')

    root_config2 = hypergraph.get_hyperedge_attribute(edge2,'root_config')
    rec_config2 = hypergraph.get_hyperedge_attribute(edge2,'rec_config')
    config2 = get_full_config(root_config2,rec_config2)

    for i in range(len(config1)):
        if root_config1[i] == 1 and rec_config2[i] == 1:
            loc = assignment[node[1]][node[0]]
            new_rec_counts1 = copy.deepcopy(list(rec_counts1))
            new_rec_counts1[loc] -= 1
            if new_rec_counts1[loc] < 1:
                new_rec_config1 = copy.deepcopy(list(rec_config1))
                new_rec_config1[loc] = 0
                new_config1 = get_full_config(root_config1,new_rec_config1)
                new_cost_1 = costs[new_config1]
                cost_change = new_cost_1 - cost1
                if cost_change < 0:
                    return cost_change


    cost2 = costs[config2]
    cost_change = 0
    for i in range(len(config1)):
        if root_config1[i] == 1 and root_config2[i] == 1:
            return 0


    for i in range(len(config1)):
        if config1[i] == 1 and config2[i] == 1:
            new_rec_counts2 = copy.deepcopy(list(rec_counts2))
            new_rec_counts2[i] -= 1
            if new_rec_counts2[i] < 1:
                new_rec_config2 = copy.deepcopy(list(rec_config2))
                new_rec_config2[i] = 0
                new_config2 = get_full_config(root_config2,new_rec_config2)
                new_cost_2 = costs[new_config2]
                cost_change = new_cost_2 - cost2
            break
    return cost_change

def optimise_detached_hyperedges(hypergraph,assignment,num_partitions,costs):
    hypergraph = map_counts_and_configs(hypergraph,assignment,num_partitions,costs)
    total_cost = calculate_full_cost(hypergraph,assignment,num_partitions,costs)
    print("Total cost", total_cost)
    cost = total_cost
    checked = set()
    for edge1 in hypergraph.hyperedges:
        el1 = edge1[0]
        if not isinstance(el1, tuple):
            for edge2 in hypergraph.hyperedges:
                if edge1 != edge2:
                    el2 = edge2[0]
                    if not isinstance(el2, tuple):
                        root_set1 = hypergraph.hyperedges[edge1]['root_set']
                        rec_set1 = hypergraph.hyperedges[edge1]['receiver_set']
                        root_set2 = hypergraph.hyperedges[edge2]['root_set']
                        rec_set2 = hypergraph.hyperedges[edge2]['receiver_set']
                        int12 = root_set1.intersection(rec_set2)
                        int21 = root_set2.intersection(rec_set1)
                        if int12 != set():
                            node = int12.pop()
                            if node not in checked:
                                cost_change = find_detached_cost(hypergraph,edge1,edge2,node,assignment,costs)
                                if cost_change < 0:
                                    checked.add(node)
                            else:
                                cost_change = 0
                        
                        elif int21 != set():
                            node = int21.pop()
                            if node not in checked:
                                cost_change = find_detached_cost(hypergraph,edge2,edge1,node,assignment,costs)
                                if cost_change < 0:
                                    checked.add(node)
                            else:
                                cost_change = 0

                        else:
                            cost_change = 0
                        cost = cost + cost_change
    # difference = total_cost - cost
    # final_cost = total_cost - difference/2
    print("Final cost", cost)
    return cost
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode
import numpy as np
import networkx as nx
import copy
import math as mt
import matplotlib.pyplot as plt

def get_reg_mapping(circuit):
    qubit_indeces = {}
    index = 0
    for reg in circuit.qregs:
        for n in range(reg.size):
            qubit_indeces[(reg.name,n)] = index
            index += 1
    return qubit_indeces

def circuit_to_gate_layers(circuit):
    "Uses qiskit DAG circuit to group gates into sublists by layer/timestep of the circuit"
    dag = circuit_to_dag(circuit)
    layers = list(dag.multigraph_layers())
    layer_gates = []
    qubit_mapping = get_reg_mapping(circuit)
    for layer in layers:
        layer_info = []
        for node in layer:
            if isinstance(node, DAGOpNode):
                gate_info = [node.name, [qubit_mapping[(qubit._register.name,qubit._index)] for qubit in node.qargs],[qubit._register.name for qubit in node.qargs],node.op.params]
                layer_info.append(gate_info)
        layer_gates.append(layer_info)
    return layer_gates

def circuit_to_gate_layers_dict(circuit):
    "Uses qiskit DAG circuit to group gates into sublists by layer/timestep of the circuit"
    dag = circuit_to_dag(circuit)
    layers = list(dag.multigraph_layers())
    layer_gates = []
    qubit_mapping = get_reg_mapping(circuit)
    for layer in layers:
        layer_info = []
        for node in layer:
            if isinstance(node, DAGOpNode):
                gate_info = {'name' : node.name, 'qargs' : [qubit_mapping[(qubit._register.name,qubit._index)] for qubit in node.qargs], 'qregs' : [qubit._register.name for qubit in node.qargs], 'params' : node.op.params}
                layer_info.append(gate_info)
        layer_gates.append(layer_info)
    return layer_gates

def remove_duplicated(layers):
    "We can remove the duplicate gates by creating a dictionary and running through all operations to check for doubles"
    dictionary = {}
    new_layers = copy.deepcopy(layers)
    for l, layer in enumerate(layers):
        for i in range(len(layer)):
            op = layer[i]
            if len(op) > 4:
                # Two qubit gate
                if len(op) > 5:
                    # Gate group
                    qubit1 = op[1][0]
                    qubit2 = op[1][1]
                    l_index = op[4]
                    dictionary[(qubit1,qubit2,l_index)] = True
                    last_gate = op[-1]
                    qubit1 = last_gate[0]
                    qubit2 = last_gate[1]
                    last_gate_t = last_gate[2]
                    if qubit1 == qubit2:
                        op = op[:-1]
                        sqb = ['u', [qubit1], ['q'], last_gate[3]]
                        new_layers[last_gate_t].append(sqb)
    
    for l, layer in enumerate(layers):
        index = 0
        for i in range(len(layer)):
            op = layer[i]
            if len(op) == 5:
                # Individual two qubit gate
                qubit1 = op[1][0]
                qubit2 = op[1][1]
                l_index = op[4]
                if (qubit1,qubit2,l_index) in dictionary:
                    # Remove gate from layers
                    new_layers[l].pop(index)
                    index -= 1
                dictionary[(qubit1,qubit2,l_index)] = True
            index += 1
    return new_layers

def group_distributable_packets(layers,num_qubits,anti_diag=False):
    "Uses the rules for gate packing to create groups of gates which can be distributed together"
    new_layers = copy.deepcopy(layers)
    live_controls = [[] for _ in range(num_qubits)]
    for l, layer in enumerate(layers):
        index = 0
        for i in range(len(layer)):
            op = layer[i]
            qubits = op[1]
            if len(qubits) < 2:
                qubit = qubits[0]
                # Single qubit gate kills any controls if it is not diagonal/anti-diagonal.
                # We introduce checks for diagonality based on the params.
                params = op[3]
                diag = None
                if len(params) >= 2:
                    theta = params[0]
                    if (theta % mt.pi*2) == 0:
                        diag = True
                    elif (theta % mt.pi*2) == mt.pi/2:
                        if anti_diag == True:
                            diag = True
                        else:
                            diag = False
                    else: 
                        diag = False
                else:
                    theta = None
                    if op[0] == 'h':
                        diag = False
                    elif op[0] == 'z' or 't' or 's' or 'rz' or 'u1':
                        diag = True
                    elif op[0] == 'x' or 'y':
                        if anti_diag == True:
                            diag = True
                    else:
                        diag = False
                
                if diag == False:
                    if live_controls[qubit] != []:
                        # Add the operation group back into the list
                        # print(live_controls[qubit])
                        # if len(live_controls[qubit]) == 4:
                        try:
                            start_layer = live_controls[qubit][4]
                        except IndexError as e:
                            print(live_controls[qubit])

                        new_layers[start_layer].append(live_controls[qubit])
                        live_controls[qubit] = []
                else: 
                    new_layers[l].pop(index)
                    index -= 1
                    if live_controls[qubit] != []:
                        live_controls[qubit].append([qubit,qubit,l,params,op[0]])
            else:
                # We check if there is a control available for either qubit
                qubit1 = qubits[0]
                qubit2 = qubits[1]
                params = op[3]
                # Remove the operation from the layer temporarily
                new_layers[l].pop(index)
                index -= 1
                len1 = len(live_controls[qubit1])
                if len1 != 0:
                    # There is a control available qubit 1
                    # Check the length of both chains
                    if len1 == 5: # i.e nothing added to the group yet - meaning this is the first use so we should choose this as lead and remove the partner from live controls
                        pair = live_controls[qubit1][1]
                        if pair[0] == qubit1:
                            partner = pair[1]
                        else:
                            partner = pair[0]
                        if len(live_controls[partner]) <= 5: # remove the partner from controls list
                            live_controls[partner] = []
                            live_controls[qubit1][1][0] = qubit1
                            live_controls[qubit1][1][1] = partner
                        else:
                            live_controls[qubit1] = []
                            live_controls[partner][1][0] = partner
                            live_controls[partner][1][1] = qubit1
                            len1 = 0 # Now partner becomes the lead and qubit 1 is ready for new group

                len2 = len(live_controls[qubit2])
                if len2 != 0:
                    # Control available qubit 2
                    if len2 == 5:
                        pair = live_controls[qubit2][1]
                        if pair[0] == qubit2:
                            partner = pair[1]
                        else:
                            partner = pair[0]
                        if len(live_controls[partner]) <= 5:
                            live_controls[partner] = []
                            live_controls[qubit2][1][0] = qubit2
                            live_controls[qubit2][1][1] = partner
                        else:
                            live_controls[qubit2] = []
                            live_controls[partner][1][0] = partner
                            live_controls[partner][1][1] = qubit2
                            len2 = 0
                # Now we choose the longest chain to add to
                if len1 > len2:
                    live_controls[qubit1].append([qubit1,qubit2,l,params,op[0]])
                    #print(len1,len2)
                    #print(live_controls[qubit1])
                elif len2 > len1:
                    live_controls[qubit2].append([qubit2,qubit1,l,params,op[0]])
                    #print(len1,len2)
                    #print(live_controls[qubit2])
                elif len1 == len2 and len1 != 0:
                    live_controls[qubit1].append([qubit1,qubit2,l,params,op[0]]) # No benefit to either so just choose the first
                    #print(len1,len2)
                    #print(live_controls[qubit1])

                if len1 == 0 and len2 == 0: # The final condition is when both are 0 and new source controls must be made
                    # While it is in live controls we add other operations to the group until then
                    op.append(l)
                    live_controls[qubit1] = op.copy() # This begins the group which we can add operations to
                    live_controls[qubit2] = op.copy() # We start the group in both and choose as we go which should be lead control
            index += 1
    for gate_group in live_controls:
        if gate_group != []:
            start_layer = gate_group[4]
            new_layers[start_layer].append(gate_group)
    new_layers = remove_duplicated(new_layers)
    return new_layers

def circuit_to_graph(qpu_info,circuit,layers = None, max_depth=10000,limit=True,group_gates=False,input_layers = False):
    "Main function to convert a circuit to a graph. Returns a graph and list of operations which is efficient for cost calculation"
    num_qubits_phys = np.sum(qpu_info)
    num_qubits_log = circuit.num_qubits
    if not input_layers:
        layers = circuit_to_gate_layers(circuit)
        if group_gates:
            layers = group_distributable_packets(layers,num_qubits_log)

    initial_mapping = {n : n for n in range(num_qubits_phys)}
    nodes = []
    G = nx.Graph()
    if len(layers) > max_depth:
        limit = max_depth
    else:
        limit = len(layers)
        max_depth = limit

    for i in range(limit):
        if i == 0 or i == limit-1:
            for j in range(num_qubits_log):
                node = (initial_mapping[j],i)
                nodes.append(node)
                G.add_node(node, color = 'black', pos = (i,num_qubits_phys-initial_mapping[j]), size = 100, name = "init", label = 'init',params=None, used = 'False', source = True)
        else:
            for n in range(len(layers[i])):
                gate = layers[i][n]
                name = gate[0]
                qubits = gate[1]
                params = gate[3]
                for k in range(len(qubits)):
                    if len(qubits) > 1:
                        if k == 0:
                            label = 'control'
                            color = 'blue'
                        else:
                            if name == 'cx':
                                label = 'target'
                                color = 'red'
                                
                            elif name == 'cz' or 'cp':
                                label = 'control'
                                color = 'blue'
                    else:
                        color = 'green'
                        label = 'single'
                        name = name

                    node = (initial_mapping[qubits[k]],i)
                    nodes.append(node)
                    G.add_node(node,color = color, pos = (i,num_qubits_phys-initial_mapping[qubits[k]]), size = 300, name = name,label = label,params = params, used = 'False',source = False)
                if len(qubits) > 1:
                    if len(gate) == 5:
                        if gate[4] == 'nonlocal':
                            edge_color = 'blue'
                        else:
                            edge_color = 'black'
                    else:
                        edge_color = 'black'
                    G.add_edge((initial_mapping[qubits[0]],i),(initial_mapping[qubits[1]],i),label='gate', weight = 1, name = name, params = params, color=edge_color)
                if len(gate) > 5:
                    # Gate group
                    G.nodes[(qubits[0],i)]['source'] = True
                    G.nodes[(qubits[0],i)]['label'] = 'root_control'
                    G.edges[((qubits[0],i),(qubits[1],i))]['color'] = 'red'
                    G.nodes[(qubits[1],i)]['label'] = 'receiver_' + label
                    for z in range(5,len(gate)):
                        sub_gate = gate[z]
                        target = sub_gate[1]
                        target_layer = sub_gate[2]
                        params = sub_gate[3]
                        name = sub_gate[4]
                        if name == 'cp' or name == 'cz':
                            label = 'receiver_control'
                            color = 'blue'
                        else:
                            if name == 'cx':
                                label == 'receiver_target_x'
                            else:
                                label == 'receiver_target_u'
                        node = (initial_mapping[target],target_layer)
                        if target != qubits[0]:
                            if not G.has_node(node):
                                G.add_node(node,color = color, pos = (target_layer,num_qubits_phys-initial_mapping[target]), size = 300, name = name,label = label, params = params, used = 'True',source = False)
                            else:
                                old_label = G.nodes[node]['label']
                                G.add_node(node,color = 'yellow', pos = (target_layer,num_qubits_phys-initial_mapping[target]), size = 300, name = name,label = label + old_label, params = params, used = 'True',source = False)
                            if not G.has_node((initial_mapping[qubits[0]],target_layer)):
                                
                                G.add_node((initial_mapping[qubits[0]],target_layer),color = 'gray', pos = (target_layer,num_qubits_phys-initial_mapping[qubits[0]]), size = 200, name = name, label = 'ghost_control', params = params, used = 'True', source = False)
                            else:
                                old_label = G.nodes[node]['label']
                                G.add_node((initial_mapping[qubits[0]],target_layer),color = 'yellow', pos = (target_layer,num_qubits_phys-initial_mapping[qubits[0]]), size = 200, name = name, label = old_label + 'ghost_control', params = params, used = 'True', source = False)
                            nodes.append(node)
                            nodes.append((initial_mapping[qubits[0]],target_layer))
                            G.add_edge((initial_mapping[qubits[0]],i),(initial_mapping[target],target_layer),weight=1,label='gate',name = name, params = params, color= 'red')
                        else:
                            G.add_node(node,color = 'green', pos = (target_layer,num_qubits_phys-initial_mapping[target]), size = 300, name = 'u',label = 'single_root', params = params, used = 'True',source = False)
                            nodes.append(node)
                    for q in range(i+1,target_layer):
                        if not G.has_node((initial_mapping[qubits[0]],q)):
                            G.add_node((initial_mapping[qubits[0]],q),color = 'gray', pos = (q,num_qubits_phys-initial_mapping[qubits[0]]), size = 5, name = 'id',label = 'root_t', params = None, used = 'False', source = False)
                            nodes.append((initial_mapping[qubits[0]],q))
                        else:
                            old_label = G.nodes[(initial_mapping[qubits[0]],q)]['label']
                            if old_label[0:2] == 're':
                                color = 'yellow'
                                old_name = G.nodes[(initial_mapping[qubits[0]],q)]['name']
                                old_size = G.nodes[(initial_mapping[qubits[0]],q)]['size']
                                G.add_node((initial_mapping[qubits[0]],q),color = 'yellow', pos = (q,num_qubits_phys-initial_mapping[qubits[0]]), size = old_size, name = old_name,label = old_label + 'root_t', params = None, used = 'False', source = False)
                                nodes.append((initial_mapping[qubits[0]],q))

    for i in range(num_qubits_phys):
        for j in range(max_depth):
            node = (i,j)
            if G.has_node((i,j)) == False:
                G.add_node(node,color = 'lightblue', pos = (j,num_qubits_phys-initial_mapping[i]), size = 0, name = "id",label = None, params = None,used = 'False', source = False)
                nodes.append(node)

    for n in range(len(nodes)):
        for m in range(len(nodes)):
            if nodes[n][0] == nodes[m][0] and nodes[n][1] == nodes[m][1]-1 and nodes[n][0] < num_qubits_log:
                G.add_edge(nodes[n],nodes[m],label='state',color='grey',weight=1)
            if nodes[n][0] >= num_qubits_log and nodes[n][0] == nodes[m][0]:
                G.nodes[nodes[n]]['color'] = 'lightblue'
                G.nodes[nodes[n]]['size'] = 0

    return G

def ungroup_layers(layers):
    new_layers = [[] for _ in range(len(layers))]
    for l, layer in enumerate(layers):
        for i in range(len(layer)):
            op = layer[i]
            if len(op) == 4:
                # single qubit gate
                new_layers[l].append(op)
            elif len(op) == 5:
                # two qubit gate
                new_layers[l].append(op[:-1])
            elif len(op) > 5:
                start_op = [op[0],op[1],op[2],op[3]]
                new_layers[l].append(start_op)
                for i in range(5,len(op)):
                    gate = op[i]
                    new_op = ['cp', [gate[0],gate[1]],['reg','reg'], gate[3]]
                    index = gate[2]
                    new_layers[index].append(new_op)
    return new_layers

def create_static_graph(layers,num_qubits):
    graph = nx.Graph()
    for n in range(num_qubits):
        graph.add_node(n)
    for l, layer in enumerate(layers):
        for gate in layer:
            qubits = gate[1]
            if len(qubits) == 2:
                edge = (qubits[0],qubits[1])
                if edge not in graph.edges:
                    graph.add_edge(qubits[0],qubits[1], weight = 1)
                else:
                    graph.edges[edge]['weight'] += 1
    return graph

def set_initial_partitions(qpu_info,num_layers,num_partitions,invert=False):
    static_partition = []
    for n in range(num_partitions):
        for k in range(qpu_info[n]):
            if invert == False:
                static_partition.append(n)
            else:
                static_partition.append(num_partitions-n-1)
    full_partitions = np.zeros((num_layers,len(static_partition)),dtype=int)
    for n in range(num_layers):
        full_partitions[n] = np.array(static_partition,dtype=int)
    return full_partitions

def swap_node_pos(node1,node2,graph,num_layers):
    new_graph = graph.copy()
    time = node1[1]
    for n in range(time,num_layers):
        node1 = (node1[0],n)
        node2 = (node2[0],n)
        store = new_graph.nodes[node1]['pos']
        new_graph.nodes[node1]['pos'] = new_graph.nodes[node2]['pos']
        new_graph.nodes[node2]['pos'] = store
    return new_graph
    
def get_swaps(perm1,perm2):
    "Function for finding the actions to transition from one layer of the partition to the next"
    completion = [True if perm1[n] == perm2[n] else False for n in range(len(perm1))]
    p1 = np.array(perm1)
    p2 = np.array(perm2)
    target = p1.copy()
    swaps = []
    for n in range(len(perm1)):
        if completion[n] == False:
            goal = p2[n]
            idxs = np.where(target == goal)
            for idx in idxs[0]:
                
                if p2[idx] == target[n] and completion[idx] != True:
                    swaps.append((n,idx))
                    store = target[idx]
                    target[idx] = target[n]
                    target[n] = store
                    completion[n] = True
                    completion[idx] = True
            if completion[n] == False:
                store = target[n]
                target[n] = target[idxs[0][-1]]
                target[idxs[0][-1]] = store
                swaps.append((n,idxs[0][-1]))
                completion[n] = True
    return swaps

def swap_on_graph(graph,initial_partition,partition,num_layers):
    "For visualising the effect of the partition. Use to update the position of the nodes to match the partition"
    full_sequence = np.vstack((initial_partition,partition))
    for n in range(len(full_sequence)-1):
        swaps = get_swaps(full_sequence[n],full_sequence[n+1])
        for m in range(len(swaps)):
            graph = swap_node_pos((swaps[m][0],n),(swaps[m][1],n),graph,num_layers)
    return graph

def calculate_cost_layers(layers,partition,num_qubits_log):
    cost = 0
    for l, layer in enumerate(layers):
        new_part = partition[l]
        if l > 0:
            for part1,part2 in zip(current_part[:num_qubits_log],new_part[:num_qubits_log]):
                if part1 != part2:
                    cost += 1
        for op in layer:
            qubits = op[1]
            if len(qubits) > 1:
                if new_part[qubits[0]] != new_part[qubits[1]]:
                    cost += 1
        current_part = new_part
    return cost

def calculate_cost_groups(partition,layers,num_qubits_log):
    cost = 0
    for l, layer in enumerate(layers):
        new_part = partition[l]
        new_part = partition[l]
        if l > 0:
            for part1,part2 in zip(current_part[:num_qubits_log],new_part[:num_qubits_log]):
                if part1 != part2:
                    cost += 1
        for op in layer:
            op_len = len(op)
            if op_len > 4:
                qubit1 = op[1][0]
                qubit2 = op[1][1]
                if op_len == 5:
                    if new_part[qubit1] != new_part[qubit2]:
                        cost += 1
                if op_len > 5:
                    initial_part1 = partition[l][qubit1]
                    initial_part2 = partition[l][qubit2]
                    source_parts = set()
                    source_parts.add(initial_part1)
                    rec_parts = set()
                    rec_parts.add(initial_part2)
                    for n in range(5,len(op)):
                        gate = op[n]
                        q2 = gate[1]
                        t = gate[2]
                        part = partition[t][q2]
                        rec_parts.add(part)
                    for _ in range(l,t+1):
                        part_t = partition[_][qubit1]
                        source_parts.add(part_t)
                    cost += len(rec_parts-source_parts)
        current_part = new_part
    return cost

def get_pos(graph,initial_partition,partition,qpu_info,num_layers):
    filled_pos = set()
    num_partitions = len(qpu_info)
    num_qubits_phys = np.sum(qpu_info)
    layer_dict = []
    for _ in range(num_layers):
        dictionary = {n: [] for n in range(num_partitions)}
        index = 0
        for n in range(num_partitions):
            for k in range(qpu_info[n]):
                dictionary[n].append(num_qubits_phys - index)
                index += 1
        layer_dict.append(dictionary)

    for n in range(num_qubits_phys):
        for m in range(num_layers):
            node = (n,m)
            print("node", node)
            initial_pos = (node[1],len(initial_partition[0]) - node[0])
            required_destination = partition[node[1]][node[0]]
            print("required dest", required_destination)
            print("Options", layer_dict[node[1]][required_destination])
            pos_index = layer_dict[node[1]][required_destination][0]
            layer_dict[node[1]][required_destination].remove(pos_index)
            print("pos index", pos_index)
            graph.nodes[node]['pos'] = (initial_pos[0],pos_index)
            print(layer_dict)
    print(layer_dict)
    return graph

def draw_graph_(graph, qpu_info, partition = None, divide=True, save_fig = False, path = None,spring = False):
    graph_partitioned = graph.copy()
    if partition is not None:
        initial_partition = set_initial_partitions(qpu_info,len(partition),len(qpu_info))
        base_partition = initial_partition[0]
        # graph_partitioned = get_pos(graph_partitioned,initial_partition,partition,qpu_info,len(partition))
        graph_partitioned = swap_on_graph(graph_partitioned,base_partition,partition,len(partition))

    colors = [graph_partitioned.nodes[node]['color'] for node in graph.nodes]
    edge_colors = [graph_partitioned.edges[edge]['color'] for edge in graph.edges]
    edge_widths = [graph_partitioned.edges[edge]['weight']*1.5 for edge in graph.edges]
    node_sizes = [graph_partitioned.nodes[node]['size']/(0.03*len(graph.nodes)) for node in graph.nodes]
    if not spring:
        pos = nx.get_node_attributes(graph_partitioned, 'pos')
    else:
        pos = nx.spring_layout(graph_partitioned)
        divide = False
    nx.draw(graph_partitioned, with_labels=False, font_weight='light',pos=pos,node_color=colors,node_size = node_sizes, width=edge_widths,edge_color=edge_colors)
    y_lines = []
    point = 0.5
    for n in range(len(qpu_info)-1):
        point += qpu_info[n]
        y_lines.append(point)
    ax = plt.gca()
    if divide == True:
        for y in y_lines:
            ax.axhline(y=np.sum(qpu_info)+1-y, color='gray', linestyle='--', linewidth=1)
    plt.axis('off')
    if save_fig:
        plt.savefig(path)
    plt.show()
    return ax

def ungroup_local_gates(layers,partition):
    new_layers = [[] for _ in range(len(layers))]
    for l, layer in enumerate(layers):
        for gate in layer:
            # print(gate)
            gate_length = len(gate)
            # print("gate length:",gate_length)
            if gate_length <= 4:
                # Single qubit gate
                new_layers[l].append(gate)
            elif gate_length == 5:
                # Two qubit gate
                qubits = gate[1]
                if partition[l][qubits[0]] == partition[l][qubits[1]]:
                    gate[4] = 'local'
                    new_layers[l].append(gate)
                else:
                    gate[4] = 'nonlocal'
                    new_layers[l].append(gate)
            else:
                # Gate group
                group = True
                time = l
                end_time = gate[-1][2]
                qpu_set = set()
                for i in range(time,end_time+1):
                    qpu_set.add(partition[i][gate[1][0]])

                while True:
                    qubits = gate[1]
                    # print("Gate group")
                    # print(gate)
                    if partition[time][qubits[1]] == partition[time][qubits[0]]:
                        if len(gate) <= 5:
                            # print("Gate no longer a group")
                            group = False
                            time = gate[4]
                            # print("Time:",time)
                            gate[4] = 'local'
                            new_layers[time].append(gate)
                            # print("New layers:", new_layers[time])
                            break
                        time = gate[4]
                        local_gate = gate[:4]
                        local_gate.append('local')
                        new_layers[time].append(local_gate)
                        new_info = gate[5]
                        if new_info[0] == new_info[1]:
                            # print("Single qubit gate")
                            qubits = [new_info[0]]
                            time_s = new_info[2]
                            params = new_info[3]
                            gate_type = new_info[4]
                            new_single_gate = [gate_type, qubits, ['q'], params]
                            new_layers[time_s].append(new_single_gate)
                            new_info = gate[6]
                        # print("Two qubit gate")
                        qubits = [new_info[0],new_info[1]]
                        new_start_gate = []
                        qubits = [new_info[0],new_info[1]]
                        time = new_info[2]
                        params = new_info[3]
                        gate_type = new_info[4]
                        new_start_gate.append(gate_type)
                        new_start_gate.append(qubits)
                        new_start_gate.append(['q','q'])
                        new_start_gate.append(params)
                        new_start_gate.append(time)
                        # print("New start gate:",new_start_gate)
                        gate = new_start_gate + gate[6:]
                        # print("New gate:",gate)
                    else:
                        break
                # print("Is gate group?:",group)
                if group:
                    if len(gate) <= 5:
                        # print("Gate no longer a group")
                        # print(gate)
                        group = False
                        time = gate[4]
                        # print("Time:",time)
                        gate[4] = 'nonlocal'
                        new_layers[time].append(gate)
                        # print("New layers:",new_layers[time])
                    else:
                        gate_group = gate[0:5]
                        # print("Gate group start:",gate_group)
                        for n in range(5,len(gate)):
                            new_info = gate[n]           
                            if new_info[0] == new_info[1]:
                                # print("Single qubit gate")
                                qubits = [new_info[0]]
                                time_s = new_info[2]
                                params = new_info[3]
                                gate_type = new_info[4]
                                new_single_gate = [gate_type, qubits, ['q'], params]
                                new_layers[time_s].append(new_single_gate)
                                new_info = gate[6]
                            else:    
                                qubits = [new_info[0], new_info[1]]
                                if partition[time][qubits[1]] in qpu_set:
                                    local_gate = []
                                    time = new_info[2]
                                    params = new_info[3]
                                    gate_type = new_info[4]
                                    local_gate.append(gate_type)
                                    local_gate.append(qubits)
                                    local_gate.append(['q','q'])
                                    local_gate.append(params)
                                    local_gate.append(time)
                                    new_layers[time].append(local_gate)
                                else:
                                    gate_group.append(new_info)
                        new_layers[l].append(gate_group)
    return new_layers

def graph_to_tikz(G, partition, qpu_info, scale=1.0, save=False, path=None):
    """
    Takes a networkx Graph G and a position dictionary pos,
    and returns a TikZ string with nodes and edges.
    """
    # Header
    tikz_code = []
    tikz_code.append(r"\begin{tikzpicture}[scale=%.2f]" % scale)
    # tikz_code.append("  [%s]" % f"every node/.style={{{node_style}}}")

    # Create a name for each node that’s TeX-friendly:
    # e.g. node (r_c) for node (r, c) in a grid_2d_graph
    def node_name(n):
        # if n is a tuple, you can replace with underscores
        if isinstance(n, tuple):
            return "n" + "_".join(str(x) for x in n)
        # else just string-ify it
        return str(n)
    
    if partition is not None:
        initial_partition = set_initial_partitions(qpu_info,len(partition),len(qpu_info))
        base_partition = initial_partition[0]
        # graph_partitioned = get_pos(graph_partitioned,initial_partition,partition,qpu_info,len(partition))
        G = swap_on_graph(G,base_partition,partition,len(partition))

    # 1) Print each node
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    for n in G.nodes():
        colour = G.nodes[n]['color']
        if colour == 'blue' or colour == "yellow":
            style = "new style 0"
        elif colour == 'black':
            style = "new style 1"
        elif colour == "green":
            style = "new style 2"
        else:
            style = "none"
        pos = G.nodes[n]['pos']
        x, y = pos  # get position from dictionary
        # Example: \node (n0_1) at (0,1) {};
        tikz_code.append(f"    \\node [style={style}] ({node_name(n)}) at ({2*x},{y}) {{}};")
    tikz_code.append(r"  \end{pgfonlayer}")   
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")     
    # 2) Print edges
    for u, v in G.edges():
        # example: \draw (n0_1) -- (n0_2);
        if u[0] != v[0]:
            bend = "[bend right=15] "
        else:
            bend = ""
        tikz_code.append(f"    \\draw {bend}({node_name(u)}) to ({node_name(v)});")
    tikz_code.append(r"  \end{pgfonlayer}") 
    # Footer
    tikz_code.append(r"\end{tikzpicture}")
    
    if save:
        with open(path, "w") as f:
            f.write("\n".join(tikz_code))

    return "\n".join(tikz_code)
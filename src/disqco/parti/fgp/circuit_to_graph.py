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
                # Single qubit gate kills any controls - only if it is not diagonal/anti-diagonal. (diagonal only for now)
                # We introduce checks for diagonality based on the params
                params = op[3]
                theta = params[0]
                diag = None
                if (theta % mt.pi*2) == 0:
                    diag = True
                elif (theta % mt.pi*2) == mt.pi/2:
                    if anti_diag == True:
                        diag = True
                    else:
                        diag = False
                else: 
                    diag = False
                if diag == False:
                    if live_controls[qubit] != []:
                        # Add the operation group back into the list
                        # print(live_controls[qubit])
                        # if len(live_controls[qubit]) == 4:

                        start_layer = live_controls[qubit][4]
                        new_layers[start_layer].append(live_controls[qubit])
                        live_controls[qubit] = []
                else: 
                    new_layers[l].pop(index)
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

def circuit_to_graph(qpu_info,circuit,max_depth=10000,limit=True,group_gates=False):
    "Main function to convert a circuit to a graph. Returns a graph and list of operations which is efficient for cost calculation"
    num_qubits_phys = np.sum(qpu_info)
    num_qubits_log = circuit.num_qubits
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
                    G.add_edge((initial_mapping[qubits[0]],i),(initial_mapping[qubits[1]],i),label='gate',name = name, params = params, color='black')
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
                            G.add_node(node,color = color, pos = (target_layer,num_qubits_phys-initial_mapping[target]), size = 300, name = name,label = label, params = params, used = 'True',source = False)
                            G.add_node((initial_mapping[qubits[0]],target_layer),color = 'gray', pos = (target_layer,num_qubits_phys-initial_mapping[qubits[0]]), size = 200, name = name, label = 'ghost_control', params = params, used = 'True', source = False)
                            nodes.append(node)
                            nodes.append((initial_mapping[qubits[0]],target_layer))
                            G.add_edge((initial_mapping[qubits[0]],i),(initial_mapping[target],target_layer),label='gate',name = name, params = params, color='red')
                        else:
                            G.add_node(node,color = 'green', pos = (target_layer,num_qubits_phys-initial_mapping[target]), size = 300, name = 'u',label = 'single', params = params, used = 'True',source = False)
                            nodes.append(node)

    for i in range(num_qubits_phys):
        for j in range(max_depth):
            node = (i,j)
            if G.has_node((i,j)) == False:
                G.add_node(node,color = 'lightblue', pos = (j,num_qubits_phys-initial_mapping[i]), size = 0, name = "id",label = None, params = None,used = 'False', source = False)
                nodes.append(node)

    for n in range(len(nodes)):
        for m in range(len(nodes)):
            if nodes[n][0] == nodes[m][0] and nodes[n][1] == nodes[m][1]-1 and nodes[n][0] < num_qubits_log:
                G.add_edge(nodes[n],nodes[m],label='state',color='grey')
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

def draw_graph_(G,qpu_info,divide=True, save_fig = False, path = None):

    colors = [G.nodes[node]['color'] for node in G.nodes]
    edge_colors = [G.edges[edge]['color'] for edge in G.edges]
    node_sizes = [G.nodes[node]['size']/(0.03*len(G.nodes)) for node in G.nodes]
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, with_labels=False, font_weight='light',pos=pos,node_color=colors,node_size = node_sizes, width = 1.5,edge_color=edge_colors)
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


def layer_dict_to_list(layers):
    "Converts the dictionary of layers to a list of layers"
    new_layers = []
    for layer in layers:
        new_layer = []
        for key in layer:
            new_layer.append(layer[key])
        new_layers.append(new_layer)
    return new_layers
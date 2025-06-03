import copy
import math as mt

def check_diag_gate(gate, include_anti_diags = False):
    "Checks if a gate is diagonal or anti-diagonal"
    name = gate['name']
    if name == 'u' or name == 'u3':
        theta = gate['params'][0]
        if round(theta % mt.pi*2, 2) == round(0, 2):
            return True
        elif round(theta % mt.pi*2, 2) == round(mt.pi/2, 2):
            if include_anti_diags:
                return True
            else:
                return False
        else:
            return False
    else:
        if name == 'h':
            return False
        elif name == 'z' or name == 't' or name == 's' or name == 'rz' or name == 'u1':
            return True
        elif name == 'x' or 'y':
            if include_anti_diags:
                return True
            else:
                return False
        else:
            return False

def group_size(gate):
    sub_gates = gate['sub-gates']
    counter = 0
    for sub_gate in sub_gates:
        if sub_gate['type'] == 'two-qubit':
            counter += 1
    return counter

def group_distributable_packets_sym(layers,group_anti_diags=True):
    "Uses the rules for gate packing to create groups of gates which can be distributed together"
    # new_layers = copy.deepcopy(layers)
    live_controls = {}
    new_layers = {i : [] for i in range(len(layers))}
    unchosen_groups = {}
    for l in layers:
        layer = layers[l]
        for i in range(len(layer)):
            
            op = layer[i]
            gate_type = op['type']
            if gate_type == 'single-qubit':
                qubit = op['qargs'][0]
                diag = None
                diag = check_diag_gate(op, include_anti_diags=group_anti_diags)
                
                gate = copy.deepcopy(op)
                if diag == False:
                    if qubit in live_controls:
                        group = live_controls[qubit]
                        start_layer = group['time']
                        if qubit in unchosen_groups:
                            partner = unchosen_groups[qubit]
                            del unchosen_groups[qubit]
                            del unchosen_groups[partner]
                            del live_controls[qubit] 
                        else:
                            # if len(group['sub-gates']) == 1:
                            #     new_layers[start_layer].append(group['sub-gates'][0])
                            # else:
                            #     del group['time']
                            new_layers[start_layer].append(group)
                            del live_controls[qubit]
                    new_layers[l].append(gate)
                else:
                    if qubit in live_controls:
                        group = live_controls[qubit]
                        gate['time'] = l
                        group['sub-gates'].append(gate)
                    else:
                        new_layers[l].append(gate)
            elif gate_type == 'two-qubit':
                qubits = op['qargs']
                gate_name = op['name']
                # We check if there is a control available for either qubit
                qubit1 = qubits[0]
                if qubit1 in live_controls:
                    group1 = live_controls[qubit1]
                    num_sub_gates1 = group_size(group1)
                else:
                    group1 = None
                    num_sub_gates1 = 0
                
                if gate_name == 'cp' or gate_name == 'cz':
                    qubit2 = qubits[1]
                    symmetric = True
                    if qubit2 in live_controls:
                        group2 = live_controls[qubit2]
                        num_sub_gates2 = group_size(group2)
                    else:
                        group2 = None
                        num_sub_gates2 = 0
                else:
                    symmetric = False

                if not symmetric:
                    group = group1
                    num_sub_gates = num_sub_gates1
                    root = qubit1
                else:
                    if num_sub_gates2 > num_sub_gates1:
                        root = qubit2
                        group = group2
                        num_sub_gates = num_sub_gates2

                        other_root = qubit1
                        other_group = group1
                        other_num_sub_gates = num_sub_gates1
                    else:
                        root = qubit1
                        group = group1
                        num_sub_gates = num_sub_gates1

                        other_root = qubit2
                        other_group = group2
                        other_num_sub_gates = num_sub_gates2
                
                if group is not None:
                    sub_gates = group['sub-gates']
                    if num_sub_gates == 1:
                        gate = sub_gates[0]
                        pair = gate['qargs']
                        if pair[0] == root:
                            partner = pair[1]
                        else:
                            partner = pair[0]

                        if partner in unchosen_groups and root in unchosen_groups:
                            if unchosen_groups[partner] == root and unchosen_groups[root] == partner:
                                del unchosen_groups[partner]
                                del live_controls[partner] 
                                del unchosen_groups[root]


                        live_controls[root]['sub-gates'][0]['qargs'] = [root,partner]
                    
                    if qubit1 == root:
                        qargs = [qubit1,qubit2]
                    else:
                        qargs = [qubit2,qubit1]
                    op['qargs'] = qargs
                    op['time'] = l
                    group['sub-gates'].append(op)
                    live_controls[root] = group
                else:
                    group = {}
                    group['type'] = 'group'
                    group['root'] = root
                    group['time'] = l
                    op['time'] = l
                    group['sub-gates'] = [op]

                    other_group = {}
                    other_group['type'] = 'group'
                    other_group['root'] = other_root
                    other_group['time'] = l
                    other_op = copy.deepcopy(op)
                    other_op['qargs'] = [other_op['qargs'][1],other_op['qargs'][0]]
                    other_op['time'] = l
                    other_group['sub-gates'] = [other_op]

                    live_controls[root] = group
                    live_controls[other_root] = other_group
                    unchosen_groups[root] = other_root
                    unchosen_groups[other_root] = root


    checked = set()
    for root in unchosen_groups:
        if root not in checked:
            partner = unchosen_groups[root]
            del live_controls[partner]
            checked.add(partner)
    
    for root in live_controls:
        group = live_controls[root]
        start_layer = group['time']
        # del group['time']
        sub_gates = group['sub-gates']

        new_layers[start_layer].append(group)
    
    # layers_to_remove = set()
    # for i, layer in new_layers.items():
    #     print(f"Layer {i}: {layer}")
    #     if len(layer) == 0:
    #         layers_to_remove.add(i)
    # for i in layers_to_remove:
    #     del new_layers[i]
    # new_layers = remove_duplicated_dict(new_layers)
    return new_layers

def group_distributable_packets_asym(layers : dict, group_anti_diags : bool = True):
    "Uses the rules for gate packing to create groups of gates which can be distributed together. Support for asymmetric two-qubit gates."
    live_controls = {}
    new_layers = {i : [] for i in range(len(layers))}
    
    for l in layers:
        layer = layers[l]
        for i in range(len(layer)):
            op = layer[i]
            gate_type = op['type']
            if gate_type == 'single-qubit':
                qubit = op['qargs'][0]
                diag = None
                diag = check_diag_gate(op, include_anti_diags=group_anti_diags)
                gate = copy.deepcopy(op)
                if diag == False:
                    if qubit in live_controls:
                        group = live_controls[qubit]
                        start_layer = group['time']
                        new_layers[start_layer].append(group)
                        del live_controls[qubit]

                    new_layers[l].append(gate)
                else:
                    if qubit in live_controls:
                        group = live_controls[qubit]
                        gate['time'] = l
                        group['sub-gates'].append(gate)
                    else:
                        new_layers[l].append(gate)
            elif gate_type == 'two-qubit':
                qubits = op['qargs']
                # We check if there is a root available for the control qubit
                control_qubit = qubits[0]
                target_qubit = qubits[1]
                if control_qubit in live_controls:
                    group = live_controls[control_qubit]
                    op['time'] = l
                    group['sub-gates'].append(op)
                else:
                    group = {}
                    group['type'] = 'group'
                    group['root'] = control_qubit
                    group['time'] = l
                    op['time'] = l
                    group['sub-gates'] = [op]
                    live_controls[control_qubit] = group
                
                if target_qubit in live_controls:
                    group = live_controls[target_qubit]
                    start_layer = group['time']
                    new_layers[start_layer].append(group)
                    del live_controls[target_qubit]

    for root in live_controls:
        group = live_controls[root]
        start_layer = group['time']
        new_layers[start_layer].append(group)
    
    return new_layers

def remove_duplicated_dict(layers):
    "We can remove the duplicate gates by creating a dictionary and running through all operations to check for doubles"
    dictionary = {}
    for l in layers:
        layer = layers[l]
        for i in range(len(layer)):
            op = layer[i]
            if op['type'] == 'group':
                sub_gates = op['sub-gates']
                for gate in sub_gates:
                    qubits = gate['qargs']
                    dictionary[(qubits[0],qubits[1],l)] = True
                    dictionary[(qubits[1],qubits[0],l)] = True

    
    for l in layers:
        layer = layers[l]
        index = 0
        for i in range(len(layer)):
            op = layer[i]
            if op['type'] == 'two-qubit':
                # Individual two qubit gate
                qubits = op['qargs']
                qubit1 = qubits[0]
                qubit2 = qubits[1]
                l_index = l
                if (qubit1,qubit2,l_index) in dictionary:
                    # Remove gate from layers
                    layers[l].pop(index)
                    index -= 1
                
                elif (qubit2,qubit1,l_index) in dictionary:
                    # Remove gate from layers
                    layers[l].pop(index)
                    index -= 1
                dictionary[(qubit1,qubit2,l_index)] = True
            index += 1
    
    for l in layers:
        layer = layers[l]
        for gate in layer:
            if gate == 'group':
                root = gate['root']
                first_gate = gate['sub-gates'][0]
                qubits = first_gate['qargs']


    return layers

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

def ungroup_local_gates(layers,partition):
    new_layers = {}
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

import copy
import math as mt

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

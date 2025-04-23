from typing import List, Callable, Tuple
import numpy as np
import random
from functools import partial
import multiprocessing as mp
from qiskit import transpile
from scipy.special import softmax
from disqco.utils.qiskit_to_op_list import circuit_to_gate_layers
from disqco.parti.fgp.circuit_to_graph import *
import time
Genome = List[List]
Population = List[Genome]
FitnessFunc = Callable[[Genome,List[List]],int]
PopulateFunc = Callable[[int,int],Population]
SelectFunc = Callable[[Population,FitnessFunc],Population]
CrossoverFunc = Callable[[Genome,Genome], Tuple[Genome,Genome]]
MutationFunc = Callable[[Genome,int,int],Genome]

class Genetic_Partitioning():

    def __init__(self,circuit,qpu_info, choose_layers = False, layers = None, max_depth=10000, multi_process = True, gate_packing = False,transpile_circuit=False) -> None:
        if transpile_circuit == True:
            self.circuit = transpile(circuit,basis_gates=['u','cp'])
        else:
            self.circuit = circuit
        self.qpu_info = qpu_info # List of QPU sizes
        self.max_depth = max_depth
        self.num_qubits_log = self.circuit.num_qubits
        if choose_layers == False:
            self.layers = circuit_to_gate_layers(self.circuit) # List of all operations in layers
            if gate_packing: # Pre process the gates to group distributable packets
                self.layers = group_distributable_packets(self.layers,self.num_qubits_log)
                self.layers = remove_duplicated(self.layers)
        else:
            self.layers = layers

        self.num_layers = len(self.layers)
        self.num_qubits_phys = np.sum(qpu_info) # Total number of qubits across all QPUs
        self.num_partitions = len(qpu_info)

        # self.num_two_qubit_gates = self.circuit.count_ops()['cp']
        self.initial_partition = set_initial_partitions(self.qpu_info,self.num_layers,self.num_partitions)



        self.partition_single_layer = self.initial_partition[0]
        self.multi_process = multi_process
        self.gate_packing = gate_packing

    def run(self, pop_size,num_generations, initial_partition=None, random_start = False, mutation_rate = 0.5,search_method = False,search_number=100,log = True,log_frequency = 50,multi_process=True, choose_initial = False) -> Tuple[Population, int]:  
        max_over_time = []
        pool = mp.Pool(mp.cpu_count())
        if not choose_initial:
            population = generate_population(pop_size,self.qpu_info,self.num_layers,random_start)
            if not multi_process:
                fitness_scores = [fitness_function(partition,self.layers,self.num_qubits_log,self.gate_packing) for partition in population]
                indices = list(enumerate(fitness_scores))
                sorted_indices = sorted(indices, key=lambda x: x[1],reverse=False)
                population = [(population[index],value) for index, value in sorted_indices]
            else:
                fitness_scores = pool.map(partial(fitness_function, 
                                                layers=self.layers,
                                                num_qubits_log=self.num_qubits_log,
                                                gate_packing=self.gate_packing),
                                                population)
                indices = list(enumerate(fitness_scores))
                sorted_indices = sorted(indices, key=lambda x: x[1],reverse=False)
                population = [(population[index],value) for index, value in sorted_indices]
        else:
            initial_partition1 = initial_partition.copy()
            initial_partition2 = np.array([element[::-1] for element in initial_partition])
            member1 = (initial_partition1, fitness_function(initial_partition1,self.layers,self.num_qubits_log,gate_packing=self.gate_packing))
            member2 = (initial_partition2, fitness_function(initial_partition2,self.layers,self.num_qubits_log,gate_packing=self.gate_packing))
            population = [member1,member2]
            if not multi_process:
                for _ in range(int(pop_size/2)-1):
                    offspring_a, offspring_b = create_offspring([member1,member2],self.qpu_info,self.num_layers,self.num_qubits_log,self.layers,mutation_rate,search_method,search_number,self.gate_packing)

                    population.append(offspring_a)
                    population.append(offspring_b)
            else:
                tasks = [([member1,member2],self.qpu_info,self.num_layers,self.num_qubits_log,self.layers,mutation_rate,search_method,search_number,self.gate_packing) for _ in range(int(pop_size/2)-1)]
                results = pool.starmap(create_offspring, tasks)
                for offspring_a, offspring_b in results:
                    population.append(offspring_a)
                    population.append(offspring_b)
            population = sorted(population, key = lambda x : x[1],reverse=False)

        for t in range(num_generations):
            self.best = population[0][1]
            if log == True:
                if (t % log_frequency) == 0:
                    print("Current best cut:", self.best)
            next_generation = population[0:2]
            #total_fitness = np.sum([10000-population[n][1] for n in range(len(population))])
            probabilities = softmax([-population[n][1] for n in range(len(population))])
            #probabilities = [(10000-population[n][1])/total_fitness for n in range(len(population))]
            if not multi_process:
                for _ in range(int(len(population)/2)-1):
                    parents = selection_pair(population,probabilities)
                    # start = time.time()
                    offspring_a, offspring_b = create_offspring(parents,
                                                                self.qpu_info,
                                                                self.num_layers,
                                                                self.num_qubits_log,
                                                                self.layers,
                                                                mutation_rate,
                                                                search_method,
                                                                search_number,
                                                                self.gate_packing)
                    # stop = time.time()
                    # print("Time taken to create offspring", stop - start)
                    next_generation.append(offspring_a)
                    next_generation.append(offspring_b)
            else:
                next_generation = population[0:2]
                tasks = [(selection_pair(population, probabilities), 
                          self.qpu_info, 
                          self.num_layers, 
                          self.num_qubits_log, 
                          self.layers, 
                          mutation_rate,
                          search_method,
                          search_number,
                          self.gate_packing) for _ in range(int(len(population)/2)-1)]
                results = pool.starmap(create_offspring, tasks)

                for offspring_a, offspring_b in results:
                    next_generation.append(offspring_a)
                    next_generation.append(offspring_b)
            population = sorted(next_generation, key = lambda x : x[1],reverse=False)
            max_over_time.append(population[0][1])
        pool.close()
        pool.join()
        return population,max_over_time

    def view_graphs(self,partition):
        initial_graph = circuit_to_graph(self.qpu_info,self.circuit,group_gates=self.gate_packing)
        final_graph = swap_on_graph(initial_graph,self.partition_single_layer,partition,len(partition))
        return initial_graph,final_graph

def generate_partition(qpu_info: int, num_layers: int, random_start = False) -> Genome:
    "Create candidate partition"
    candidate_layer = np.zeros(np.sum(qpu_info),dtype=int)
    counter = 0
    for k in range(len(qpu_info)):
        qpu_size = qpu_info[k]
        for n in range(qpu_size):
            candidate_layer[counter] = k
            counter += 1
    candidate = np.zeros((num_layers,np.sum(qpu_info)),dtype=int)
    layer = np.random.permutation(candidate_layer)
    for l in range(num_layers):
        candidate[l] = layer
        if random_start == True:
            if random.random() > 0.5:
                layer = np.random.permutation(candidate_layer)
    return candidate

def generate_population(size: int, qpu_info: list,num_layers: int, random_start = False) -> Population:
    #population = np.zeros((size,num_layers,num_qubits_phys),dtype=int)
    population = []
    for n in range(size):
        population.append(generate_partition(qpu_info,num_layers,random_start))
    return population

def check_validity(partition: Genome,qpu_info : List[int], num_layers : int) -> bool:
    valid = True
    for l in range(num_layers):
        layer = partition[l]
        for q in range(len(qpu_info)):
            if np.count_nonzero(layer == q) > qpu_info[q]:
                print(layer, qpu_info)
                valid = False
                break
        if valid == False:
            break
    return valid

def calculate_cost_in_layers(layers,partition,num_qubits_log):
    cost = 0
    costs = np.zeros(len(layers),dtype=int)
    for l, layer in enumerate(layers):
        new_part = partition[l]
        if l > 0:
            for part1,part2 in zip(current_part[:num_qubits_log],new_part[:num_qubits_log]):
                if part1 != part2:
                    costs[l] += 1
        for op in layer:
            qubits = op[1]
            if len(qubits) > 1:
                if new_part[qubits[0]] != new_part[qubits[1]]:
                    costs[l] += 1
        current_part = new_part
    return costs

def fitness_function(partition, layers,num_qubits_log, gate_packing = False) -> int:
    if gate_packing == False:
        cost = calculate_cost_layers(layers,partition,num_qubits_log)
    else:
        cost = calculate_cost_groups(partition,layers,num_qubits_log)
    return cost

def selection_pair(population, probs) -> Population:
    parent_indeces = np.random.choice(a=len(population), size=2, p=probs, replace=False)
    parents = [population[parent_indeces[0]],population[parent_indeces[1]]]
    return parents

def single_point_crossover_layered(a: Genome, b: Genome,prob = 0.5) -> Tuple[Genome, Genome]:
    if random.random() < prob:
        p = random.randint(1, len(a) - 1)
        g1 = np.concatenate((a[0:p], b[p:]))
        g2 = np.concatenate((b[0:p], a[p:]))
    else:
        g1 = a
        g2 = b
    return g1, g2

def create_offspring(parents, qpu_info, num_layers, num_qubits_log, layers,mutation_rate = 0.5,search_method = False, search_number=100, gate_packing=False):
    offspring_a, offspring_b = crossover(parents[0], parents[1], 0.5)
    if random.random() < mutation_rate:
        if search_method:
            offspring_a = search_mutation(offspring_a, layers,qpu_info,num_qubits_log,search_number)
            offspring_b = search_mutation(offspring_b, layers,qpu_info,num_qubits_log,search_number)
        else:
            if random.random() > 0.5: 
                offspring_a = swap_mutation_propogate(offspring_a, np.sum(qpu_info), num_layers,num=1,prob=0.5)
                offspring_b = swap_mutation_propogate(offspring_b, np.sum(qpu_info),num_layers,num=1,prob=0.5)
            else:
                offspring_a = match_mutation(offspring_a, num_layers,num=1, prob=0.5)
                offspring_b = match_mutation(offspring_b, num_layers,num=1, prob=0.5)


    cut_a = fitness_function(offspring_a,layers,num_qubits_log,gate_packing)
    cut_b = fitness_function(offspring_b,layers,num_qubits_log,gate_packing)
    return (offspring_a, cut_a), (offspring_b, cut_b)

def swap_mutation_propogate(genome: Genome, num_qubits: int, num_layers: int, num: int = 1, prob: float = 0.5) -> Genome:
    new_genome = genome.copy()
    for _ in range(num):
        start_layer = np.random.choice(num_layers)
        end_layer = num_layers
        qubit_indices = np.random.choice(num_qubits, size=2)
        if random.random() > prob:
            for n in range(start_layer, end_layer):
                store = new_genome[n][qubit_indices[0]]
                new_genome[n][qubit_indices[0]] = new_genome[n][qubit_indices[1]]
                new_genome[n][qubit_indices[1]] = store

    return genome

def swap_mutation(genome: Genome, num_qubits: int, num_layers: int, num: int = 1, prob: float = 0.5) -> Genome:
    for _ in range(num):
        layer = np.random.choice(num_layers)
        qubit_indices = np.random.choice(num_qubits, size=2)
        if random.random() > prob:
            store = genome[layer][qubit_indices[0]]
            genome[layer][qubit_indices[0]] = genome[layer][qubit_indices[1]]
            genome[layer][qubit_indices[1]] = store
    return genome

def match_mutation(genome: Genome, num_layers: int, num: int = 1, prob: float = 0.5) -> Genome:
    for _ in range(num):
        index = np.random.choice(num_layers)
        store = genome[index].copy()
        if random.random() > prob or index == 0:
            stop = np.random.choice((num_layers - index + 1))
            for n in range(index, stop):
                genome[n] = store
        else:
            start = np.random.choice(index)
            for n in range(start, index):
                genome[n] = store
    return genome

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

def swap_on_graph(graph,initial_partition,genome,num_layers):
    "For visualising the effect of the partition. Use to update the position of the nodes to match the partition"
    full_sequence = np.vstack((initial_partition,genome))
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

def cost_single_layer(part1,part2,layers,num_qubits_log,p):
    layer = layers[p]
    part1 = np.array(part1)
    part2 = np.array(part2)
    cost = np.sum(part1[:num_qubits_log] != part2[:num_qubits_log])
    for op in layer:
        qubits = op[1]
        if len(qubits) > 1:
            if part2[qubits[0]] != part2[qubits[1]]:
                cost += 1
    return cost

def crossover_with_cost(genome1, genome2, layers, num_qubits_log, prob):
    a = genome1[0]
    fa = genome1[1]
    b = genome2[0]
    fb = genome2[1]
    if random.random() < prob:
        p = random.randint(1, len(a) - 1)
        g1 = np.concatenate((a[0:p], b[p:]))
        g2 = np.concatenate((b[0:p], a[p:]))
        # Since the cost of a layer is transition TO the layer plus its own cost we only need to re-calculate layers[p]
        p1_cost = [cost_single_layer(a[p-1],b[p],layers,num_qubits_log,p)]
        p2_cost = [cost_single_layer(b[p-1],a[p],layers,num_qubits_log,p)]
        cost1 = np.concatenate((fa[0:p],p1_cost,fb[p+1:]))
        cost2 = np.concatenate((fb[0:p],p2_cost,fa[p+1:]))
    else: 
        g1 = a
        g2 = b
        cost1 = fa
        cost2 = fb
    return (g1,cost1),(g2,cost2)

def propogate_swap(candidate,num_layers,qubit1,qubit2,layer):
    new_candidate = candidate.copy()
    for n in range(layer,num_layers):
        store = new_candidate[n][qubit1]
        new_candidate[n][qubit1] = new_candidate[n][qubit2]
        new_candidate[n][qubit2] = store
    return new_candidate

def calculate_cut_diff(gen1,gen2,layers,action,start,stop,num_qubits_log):
    part1 = gen1
    part2 = gen2
    qubit1 = action[0]
    qubit2 = action[1]
    old_cut = 0
    new_cut = 0
    for l,layer in enumerate(layers[start:stop]):
        layer_index = start + l
        for op in layer:
            qubits = op[1]
            if len(qubits) > 1:
                if qubits[0] == qubit1 or qubits[0] == qubit2 or qubits[1] == qubit1 or qubits[1] == qubit2:
                    if action != (min(qubits[0],qubits[1]),max(qubits[0],qubits[1])):
                        if part1[layer_index][qubits[0]] != part1[layer_index][qubits[1]]:
                            old_cut += 1
                        if part2[layer_index][qubits[0]] != part2[layer_index][qubits[1]]:
                            new_cut += 1
    if start != 0:
        if qubit1 < num_qubits_log:
            new_cut += 1
        if qubit2 < num_qubits_log:
            new_cut += 1
    if stop != len(layers):
        if qubit1 < num_qubits_log:
            new_cut += 1
        if qubit2 < num_qubits_log:
            new_cut += 1
    
    return old_cut - new_cut
    
def search_mutation(partition, layers, qpu_info, num_qubits_log,number):
    change = 0
    best = 0
    best_partition = partition.copy()
    for n in range(number):
        new_partition = partition.copy()
        interval = np.random.choice(len(partition), size=2,replace=False)
        start, stop = (min(interval),max(interval))
        qubit1,qubit2 = np.random.choice(np.sum(qpu_info),size=2,replace=False)
        action = (qubit1,qubit2)
        new_partition = propogate_swap(new_partition,stop,action[0],action[1],start)
        change = calculate_cut_diff(partition,new_partition,layers,action,start,stop,num_qubits_log)
        if change > best:
            best = change
            best_partition = new_partition.copy()
    return best_partition

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
                    parts = set()
                    if initial_part1 != initial_part2:
                        parts.add(initial_part2)
                    for n in range(5,len(op)):
                        gate = op[n]
                        q2 = gate[1]
                        t = gate[2]
                        part = partition[t][q2]
                        part_t = partition[t][qubit1]
                        if part != part_t and part != initial_part1:
                            parts.add(part)
                        if part_t in parts:
                            parts.remove(part_t)
                    cost += len(parts)
        current_part = new_part
    return cost

def crossover(genome1,genome2,prob):
    a = genome1[0]
    b = genome2[0]
    if random.random() < prob:
        p = random.randint(1, len(a) - 1)
        g1 = np.concatenate((a[0:p], b[p:]))
        g2 = np.concatenate((b[0:p], a[p:]))
    else: 
        g1 = a
        g2 = b
    return g1,g2

def search_mutation_layers_reuse(partition, layers, qpu_info, num_qubits_log,number):
    change = 0
    best = 0
    best_partition = partition.copy()
    for n in range(number):
        new_partition = partition.copy()
        interval = np.random.choice(len(partition), size=2,replace=False)
        start, stop = (min(interval),max(interval))
        qubit1,qubit2 = np.random.choice(np.sum(qpu_info),size=2,replace=False)
        action = (qubit1,qubit2)
        new_partition = propogate_swap(new_partition,stop,action[0],action[1],start)
        change = calculate_cut_diff(partition,new_partition,layers,action,start,stop,num_qubits_log)
        if change > best:
            best = change
            best_partition = new_partition.copy()

    return best_partition

def set_initial_partitions(qpu_info,num_layers,num_partitions):
    static_partition = []
    for n in range(num_partitions):
        for k in range(qpu_info[n]):
            static_partition.append(n)
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

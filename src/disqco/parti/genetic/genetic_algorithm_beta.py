from typing import List, Callable, Tuple
import numpy as np
import random
from functools import partial
import multiprocessing as mp
from qiskit import transpile
from scipy.special import softmax
from disqco.graphs.GCP_hypergraph import *
from disqco.parti.FM.FM_methods import set_initial_partitions
from disqco.graphs.hypergraph_methods import *
from disqco.parti.FM.FM_methods import move_node
import time
from disqco.parti.partitioner import QuantumCircuitPartitioner
from disqco.graphs.quantum_network import QuantumNetwork


Genome = List[List]
Population = List[Genome]
FitnessFunc = Callable[[Genome,List[List]],int]
PopulateFunc = Callable[[int,int],Population]
SelectFunc = Callable[[Population,FitnessFunc],Population]
CrossoverFunc = Callable[[Genome,Genome], Tuple[Genome,Genome]]
MutationFunc = Callable[[Genome,int,int],Genome]

class GeneticPartitioner(QuantumCircuitPartitioner):

    def __init__(self, 
                 circuit : QuantumCircuit, 
                 network : QuantumNetwork,
                 **kwargs) -> None:
        super().__init__(circuit=circuit, network=network, initial_assignment=None)

        group_gates = kwargs.get('group_gates', True)
        self.qpu_sizes = [size for key, size in self.network.qpu_sizes.items()] # List of QPU sizes

        self.num_qubits_log = circuit.num_qubits
        self.num_layers = circuit.depth()
        self.graph = QuantumCircuitHyperGraph(circuit,group_gates=group_gates)
        self.layers = self.graph.layers
        self.costs = kwargs.get('costs', self.network.get_costs())
        self.multi_process = kwargs.get('multi_process', False)
        self.gate_grouping = kwargs.get('gate_grouping', True)
        self.num_qubits_phys = np.sum(self.qpu_sizes) # Total number of qubits across all QPUs
        self.num_partitions = len(self.qpu_sizes)

        self.search_method = kwargs.get('search_method', True)
        self.search_number = kwargs.get('search_number', 10)
        self.mutation_rate = kwargs.get('mutation_rate', 0.9)

        self.pop_size = kwargs.get('pop_size', 100)
        self.num_generations = kwargs.get('num_generations', 100)
        self.log_frequency = kwargs.get('log_frequency', 10)

    def genetic_pass(self, graph, population) -> dict:

        self.best = population[0][1]
        next_generation = population[0:2]
        
        probabilities = softmax([-population[n][1] for n in range(len(population))])

        tasks = [(selection_pair(population, probabilities), 
            self.qpu_sizes,
            self.num_qubits_log, 
            self.num_layers,
            self.num_partitions,
            graph,
            self.costs,
            self.mutation_rate,
            self.search_number) for _ in range(int(len(population)/2)-1)]
        
            
        if self.multi_process:
            results = self.pool.starmap(create_offspring, tasks)
        else:
            results = [create_offspring(*task) for task in tasks]
            

        for offspring_a, offspring_b in results:
            next_generation.append(offspring_a)
            next_generation.append(offspring_b)

        population = sorted(next_generation, key = lambda x : x[1],reverse=False)



        return population

    def run_genetic(self, **kwargs) -> Tuple[Population, int]:  
        
        graph = kwargs.get('graph', self.graph)
        log = kwargs.get('log', False)


        max_over_time = []
        population = generate_population(self.pop_size,self.qpu_sizes,self.num_layers,self.num_qubits_log)
        
        if self.multi_process:
            self.pool = mp.Pool(mp.cpu_count())

        tasks = [(graph,
                  partition, 
                  self.num_partitions, 
                  self.costs, 
                  self.qpu_sizes, 
                  self.num_layers) for partition in population]
        if self.multi_process:
            fitness_scores = self.pool.starmap(fitness_function, tasks)
        else:
            fitness_scores = [fitness_function(*task) for task in tasks]
        indices = list(enumerate(fitness_scores))
        sorted_indices = sorted(indices, key=lambda x: x[1],reverse=False)
        population = [(population[index],value) for index, value in sorted_indices]
        population = sorted(population, key = lambda x : x[1],reverse=False)

        for t in range(self.num_generations):
            self.best = population[0][1]
            if log and (t % self.log_frequency) == 0:
                print("Current best cut:", self.best) 
            population = self.genetic_pass(graph, population)
            max_over_time.append(population[0][1])

        if self.multi_process:
            self.pool.close()
            self.pool.join()
        
        best_partition = population[0][0]
        best_cost = population[0][1]
        results = {'best_partition' : best_partition, 'best_cost' : best_cost, 'max_over_time' : max_over_time}
        return results

    def partition(self):
        kwargs = {'partitioner' : self.run_genetic, 'log': False}
        return super().partition(**kwargs)

def generate_partition(qpu_info: int, num_layers: int, num_qubits: int, random_start = False, reduced = True):
    "Create candidate partition"
    if reduced:
        length = int(num_qubits)
    else:
        length = int(np.sum(qpu_info))

    candidate_layer = np.zeros(np.sum(qpu_info),dtype=int)
    counter = 0
    for k in range(len(qpu_info)):
        qpu_size = qpu_info[k]
        for n in range(qpu_size):
            candidate_layer[counter] = k
            counter += 1
    candidate = np.zeros((num_layers,length),dtype=int)
    layer = np.random.permutation(candidate_layer)
    for l in range(num_layers):
        candidate[l] = layer[:num_qubits]
        if random_start == True:
            if random.random() > 0.5:
                layer = np.random.permutation(candidate_layer)
    return candidate

def generate_population(size: int, qpu_info: list,num_layers: int, num_qubits: int, random_start = False) -> Population:
    #population = np.zeros((size,num_layers,num_qubits_phys),dtype=int)
    population = []
    for n in range(size):
        population.append(generate_partition(qpu_info,num_layers, num_qubits))
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

def fitness_function(graph, partition, num_partitions,costs,qpu_info,num_layers) -> int:

    valid = check_validity(partition, qpu_info, num_layers)
    if valid:
        cost = calculate_full_cost(graph, partition, num_partitions, costs)
    else:
        cost = 100000
    
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

def create_offspring(parents, qpu_info, num_qubits, num_layers, num_partitions, graph , costs, mutation_rate = 0.5, number=100):
    
    cut_a = parents[0][1]
    cut_b = parents[1][1]


    offspring_a, offspring_b = crossover(parents[0], parents[1], 0.0)

    if random.random() < mutation_rate:
        # if random.random() > 0.5: 
        #     offspring_a = swap_mutation_propogate(offspring_a, num_qubits, num_layers,num=1,prob=1)
        #     offspring_b = swap_mutation_propogate(offspring_b, num_qubits, num_layers,num=1,prob=1)
        # else:
        #     offspring_a = match_mutation(offspring_a, num_layers,num=1, prob=1)
        #     offspring_b = match_mutation(offspring_b, num_layers,num=1, prob=1)

        offspring_a, gain_a = search_mutation(graph, offspring_a, num_partitions, qpu_info, num_qubits, num_layers, costs, number)
        offspring_b, gain_b = search_mutation(graph, offspring_b, num_partitions, qpu_info, num_qubits, num_layers ,costs, number)
    else:
        gain_a = 0
        gain_b = 0

    cut_a = cut_a + gain_a
    cut_b = cut_b + gain_b
    # cut_a = fitness_function(graph, offspring_a, num_partitions, costs, qpu_info, num_layers)
    # cut_b = fitness_function(graph, offspring_b, num_partitions, costs, qpu_info, num_layers)

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
    return new_genome

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
    new_genome = genome.copy()
    for _ in range(num):
        index = np.random.choice(num_layers)
        store = genome[index].copy()
        if random.random() > prob or index == 0:
            stop = np.random.choice((num_layers - index + 1))
            for n in range(index, stop):
                new_genome[n] = store
        else:
            start = np.random.choice(index)
            for n in range(start, index):
                new_genome[n] = store
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

def is_space(partition, qpu_info, destination, n):
    layer = partition[n]
    counts = sum([1 if part == destination else 0 for part in layer])
    if counts < qpu_info[destination]:
        return True
    else:
        return False
      
def find_gain(hypergraph,node,destination,assignment,num_partitions,costs,log=False):
    assignment_new = move_node(node,destination,assignment)
    edges = hypergraph.node2hyperedges[node]
    gain = 0
    if log:
        print("Node", node, "Destination", destination)
    for edge in edges:
        
        # cost1 = hypergraph.get_hyperedge_attribute(edge,'cost')
        root_counts, rec_counts = hedge_k_counts(hypergraph, edge, assignment, num_partitions, set_attrs=False)
        config1 = full_config_from_counts(root_counts, rec_counts)
        cost1 = costs[tuple(config1)]

        root_counts, rec_counts = hedge_k_counts(hypergraph, edge, assignment_new, num_partitions, set_attrs=False)
        config2 = full_config_from_counts(root_counts, rec_counts)
        cost2 = costs[tuple(config2)]

        gain += cost2 - cost1
    if log:
        print(f'Total gain = {gain}')
    return gain

def calculate_cost_interval(graph,partition,qpu_info, start,destination,qubit,num_layers, num_partitions, costs):
    gain = 0
    n = start
    new_partition = partition.copy()
    while n < num_layers:
        node = (qubit,n)
        if is_space(new_partition, qpu_info, destination, n):
            cost = find_gain(graph, node, destination, new_partition, num_partitions, costs)
            gain += cost
            n += 1
            new_partition = update_partition(new_partition, start, destination, qubit, n)
        else:
            return gain, n

    return gain, n

def update_partition(partition,start,destination,qubit,stop):
    new_partition = partition.copy()
    for n in range(start,stop):
        new_partition[n][qubit] = destination
    return new_partition
    
def search_mutation(graph, partition, num_partitions, qpu_info, num_qubits_log, num_layers, costs, number):
    best = 0
    best_partition = partition.copy()
    for n in range(number):
        # print("Number", n)
        start = np.random.choice(len(partition), size=1)[0]
        qubit = np.random.choice(num_qubits_log,size=1)[0]
        # destination = np.random.choice(num_partitions, size=1)[0]
        partition_layer = partition[start]
        counts = [0 for _ in range(num_partitions)]
        for element in partition_layer:
            counts[int(element)] += 1
        for part in range(num_partitions):
            if counts[part] < qpu_info[part]:
                destination = part
                break
        # start = time.time()
        gain, stop = calculate_cost_interval(graph,partition,qpu_info,start,destination,qubit,num_layers,num_partitions,costs)
        # end = time.time()
        # print("Time taken for gain calculation:", end-start)
        if gain < best:
            # print("Gain",gain)
            best = gain
            best_partition = update_partition(partition,start,destination,qubit,stop)
    

    return best_partition, best

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

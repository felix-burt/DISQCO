from disqco.parti.FM.FM_main import run_main_algorithm


def run_hybrid(genetic_partition, population_size, num_generations, mutation_rate, hypergraph, qpu_info, num_qubits, num_partitions, limit, passes, stochastic):

    population, _ = genetic_partition.run(population_size, num_generations, initial_partition=None, search_method=True, search_number=10, log=False, multi_process=False)

    best_assignment = population[0][0]
    best_cost = population[0][1]
    initial_ass = []
    for layer in best_assignment:
        initial_ass.append(list(layer)[:num_qubits])


    cost, assignment = run_main_algorithm(hypergraph, initial_ass, qpu_info, num_qubits, num_partitions, limit, max_gain=4, passes=passes, stochastic=stochastic)
    return cost
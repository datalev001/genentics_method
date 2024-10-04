import numpy as np
import matplotlib.pyplot as plt

# Rastrigin Function: f(x) = 10n + Σ(x_i^2 - 10 * cos(2 * pi * x_i))
def fitness_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Initialize population: real number encoding
def init_population(pop_size, x_min, x_max):
    return np.random.uniform(x_min, x_max, (pop_size, 2))

# Selection: Tournament Selection
def selection(population, fitness):
    if np.min(fitness) < 0:
        fitness = fitness - np.min(fitness)

    # Avoid division by zero or a situation where all probabilities are zero
    if fitness.sum() == 0:
        fitness = np.ones_like(fitness)

    # Ensure there are no zero probabilities
    p = fitness / fitness.sum()

    selected_indices = np.random.choice(range(len(population)), size=2, replace=False, p=p)
    return population[selected_indices]

# Normal Genetic Crossover
def normal_crossover(parents):
    alpha = np.random.rand()
    offspring = alpha * parents[0] + (1 - alpha) * parents[1]
    return offspring

# Arithmetic Crossover with ReLU and Bias
def relu_crossover(parents, bias=0):
    alpha = np.random.rand()
    offspring = np.maximum(0, alpha * parents[0] + (1 - alpha) * parents[1] + bias)
    return offspring

# Sigmoid Activation with Dynamic Scaling
def sigmoid_crossover(parents, bias=0, scale=1):
    x_min, x_max = np.min(parents), np.max(parents)
    alpha = np.random.rand()
    z = alpha * parents[0] + (1 - alpha) * parents[1] + bias
    offspring = 1 / (1 + np.exp(-z * scale))  # Sigmoid activation with scaling
    return x_min + offspring * (x_max - x_min)  # Rescaling to parent's range

# Softmax with Scaling and Adaptive Mutation Rate
def softmax_crossover(parents, bias=0, generation=1):
    x_min, x_max = np.min(parents), np.max(parents)
    
    z = np.array([parents[0], parents[1]]) + bias
    z = z - np.max(z)
    
    exp_z = np.exp(z)
    softmax = exp_z / np.sum(exp_z)
    offspring = softmax[0] * parents[0] + softmax[1] * parents[1]
    offspring = np.clip(offspring, x_min, x_max)
    
    # Adaptive mutation rate
    mutation_rate = max(0.1 / generation, 0.01)
    offspring += np.random.uniform(-mutation_rate, mutation_rate, size=offspring.shape)
    return offspring

# Tanh Function with Scaling
def tanh_crossover(parents, bias=0, scale=1):
    x_min, x_max = np.min(parents), np.max(parents)
    alpha = np.random.rand()
    z = alpha * parents[0] + (1 - alpha) * parents[1] + bias
    offspring = np.tanh(z * scale)
    return x_min + (offspring + 1) / 2 * (x_max - x_min)

# Mutation: Random perturbation
def mutation(offspring, x_min, x_max, mutation_rate=0.01):
    if np.random.rand() < mutation_rate:
        offspring += np.random.uniform(-0.5, 0.5, size=offspring.shape)
    offspring = np.clip(offspring, x_min, x_max)
    return offspring

# Evaluate fitness for a population
def evaluate_fitness(population):
    return np.array([fitness_function(x) for x in population])

# Genetic Algorithm Main Function (now set for minimization)
def genetic_algorithm(x_min, x_max, pop_size=20, generations=100, crossover_method='normal', mutation_rate=0.01, bias=0):
    population = init_population(pop_size, x_min, x_max)
    best_solutions = []
    best_fitness = np.inf
    best_solution = None
    stop_counter = 0  # Counter for early stopping
    
    for gen in range(1, generations + 1):
        fitness = evaluate_fitness(population)
        new_population = []
        
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]
            stop_counter = 0  # Reset stop counter on improvement
        else:
            stop_counter += 1
        
        if stop_counter > 15:  # Early stopping criterion
            print(f"Early stopping at generation {gen} due to no improvement.")
            break

        for _ in range(pop_size):
            parents = selection(population, fitness)
            
            if crossover_method == 'normal':
                offspring = normal_crossover(parents)
            elif crossover_method == 'relu':
                offspring = relu_crossover(parents, bias)
            elif crossover_method == 'sigmoid':
                offspring = sigmoid_crossover(parents, bias, scale=2)  # Increased scale for exploration
            elif crossover_method == 'softmax':
                offspring = softmax_crossover(parents, bias, generation=gen)
            elif crossover_method == 'tanh':
                offspring = tanh_crossover(parents, bias, scale=1.5)  # Slightly adjusted scale
            
            offspring = mutation(offspring, x_min, x_max, mutation_rate)
            new_population.append(offspring)

        population = np.array(new_population)
        best_solutions.append(best_fitness)
        print(f"Generation {gen}: Best Fitness = {best_fitness}, Best Solution = {best_solution}")
    
    return best_solutions

# Comparison of different crossover methods
def compare_crossover_methods():
    methods = ['normal', 'relu', 'sigmoid', 'softmax', 'tanh']
    x_min, x_max = -5, 5
    generations = 50
    pop_size = 30
    mutation_rate = 0.02
    bias = 0.1

    results = {}

    for method in methods:
        print(f"\nRunning Genetic Algorithm with {method} crossover...")
        best_solutions = genetic_algorithm(x_min, x_max, pop_size=pop_size, generations=generations, crossover_method=method, mutation_rate=mutation_rate, bias=bias)
        results[method] = best_solutions

    # Plot results
    plt.figure(figsize=(10, 6))
    for method, best_solutions in results.items():
        plt.plot(best_solutions, label=method)

    plt.title('Comparison of Crossover Methods in Genetic Algorithm for Rastrigin Function (Minimization)')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.show()

# Run the comparison
compare_crossover_methods()

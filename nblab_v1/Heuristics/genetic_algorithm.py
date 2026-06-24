import pandas as pd
import numpy as np
from oldideas.utils import create_sample

def create_individual(nblab):
    _, alpha, beta = nblab.generate_random_sample()
    individual = np.concatenate((alpha, beta), axis=None)
    return individual

def combine(parent_a, parent_b, comb_rate):
    if (np.random.random() <= comb_rate):
        comb_point = np.random.randint(1, len(parent_a))   
        offspring_a = np.append(parent_a[0:comb_point], parent_b[comb_point:])
        offspring_b = np.append(parent_b[0:comb_point], parent_a[comb_point:])
    else:
        offspring_a = np.copy(parent_a)
        offspring_b = np.copy(parent_b)
    return offspring_a, offspring_b

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if (np.random.random() <= mutation_rate):
            individual[i] /= np.random.random()        
    return individual

def evaluate(nblab, model, individual):
    alpha = individual[:len(individual)//2]
    beta = individual[len(individual)//2:]
    df_sample = create_sample(alpha, beta)
    # error = model.predict(df_sample)
    
    pred_beam = nblab.nb_superposition(alpha, beta)
    error = nblab.obj_fun(pred_beam) # TODO Change
    return -error


def select(population, evaluation, tournamentSize):
    winner = np.random.randint(0, len(population))
    for i in range(tournamentSize - 1):
        rival = np.random.randint(0, len(population))
        if (evaluation[rival] > evaluation[winner]):
            winner = rival
    return population[winner]

def genetic_algorithm(nblab, model):
    population_size = nblab.args["ga_population_size"]
    generations = nblab.args["ga_generations"]
    comb_rate = nblab.args["ga_comb_rate"]
    mutation_rate = nblab.args["ga_mutation_rate"]
    # Creates the initial population (it also evaluates it)
    population = [None] * population_size
    evaluation = [None] * population_size  
    for i in range(population_size):
        individual = create_individual(nblab)
        population[i] = individual
        evaluation[i] = evaluate(nblab, model, individual)
    # Keeps a record of the best individual found so far
    index = 0;
    for i in range(1, population_size):
        if (evaluation[i] > evaluation[index]):
            index = i;
    best_individual = population[index]
    best_evaluation = evaluation[index]
    # Runs the evolutionary process    
    for i in range(generations):
        print(f"Generation: {i}")
        k = 0
        new_population = [None] * population_size    
        for j in range(population_size // 2):
            parent_a = select(population, evaluation, 3)
            parent_b = select(population, evaluation, 3)
            new_population[k], new_population[k + 1] = combine(parent_a, parent_b, comb_rate)       
            k = k + 2    
        population = new_population
        for i in range(population_size):
            population[i] = mutate(population[i], mutation_rate)
            evaluation[i] = evaluate(nblab, model, population[i])
          # Keeps a record of the best individual found so far
            if (evaluation[i] > best_evaluation):
                best_evaluation = evaluation[i]
                best_individual = population[i]
        
    alpha, beta = best_individual[:len(best_individual)//2], best_individual[len(best_individual)//2:]
    return alpha, beta
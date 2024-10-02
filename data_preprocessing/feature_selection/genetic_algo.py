import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from splitting_data import splitting_data
import os


def parent_centric_crossover(parents, elite_num):
    crossover_population = np.zeros(parents.shape)
    crossover_population[0:elite_num, :] = parents[0:elite_num, :]

    for ii in range(int((parents.shape[0] - elite_num) / 2)):
        n = 2 * ii + elite_num
        parents_couple = parents[n:n+2, :]
        rand_n = np.random.randint(1, parents.shape[1] - 1)
        mask = np.random.rand(parents.shape[1]) < 0.5
        child1 = np.where(mask, parents_couple[0, :], parents_couple[1, :])
        child2 = np.where(~mask, parents_couple[0, :], parents_couple[1, :])
        crossover_population[n, :] = child1
        crossover_population[n+1, :] = child2

    return crossover_population

def mutation(crossover_population, mutation_probability, min_features, max_features):
    child_row = crossover_population.shape[0]
    child_col = crossover_population.shape[1]
    num_mutations = round(child_row * child_col * mutation_probability)

    for jj in range(num_mutations):
        ind_row = np.random.randint(0, child_row)
        ind_col = np.random.randint(0, child_col)
        if crossover_population[ind_row, ind_col] == 0 and np.sum(crossover_population[ind_row, :]) < max_features:
            crossover_population[ind_row, ind_col] = 1
        elif crossover_population[ind_row, ind_col] == 1 and np.sum(crossover_population[ind_row, :]) >= min_features + 1:
            crossover_population[ind_row, ind_col] = 0

    return crossover_population

def ensure_minimum_features(crossover_population, min_features):
    for i in range(crossover_population.shape[0]):
        num_features = np.sum(crossover_population[i, :])
        if num_features < min_features:
            # if the number of 1s is smaller than min number of features
            missing = int(min_features - num_features)
            indices = np.where(crossover_population[i,:] == 0)[0]
            position2 = np.random.choice(indices, size=missing, replace=False)
            crossover_population[i, position2] = 1 # put 1s in random positions
    return crossover_population

def generate_random_individuals(population_size, num_features, min_features, max_features):
    individuals = np.zeros((population_size, num_features))
    for i in range(population_size):
        num_ones = np.random.randint(min_features, max_features + 1)
        ones_indices = np.random.choice(num_features, num_ones, replace=False)
        individuals[i, ones_indices] = 1
    return individuals

def ga_train_model(x_train, x_test, y_train, y_test, predictor_names):
    x_train = x_train.loc[:, predictor_names]
    x_test = x_test.loc[:, predictor_names]

    mdl = RandomForestClassifier(random_state=1)
    mdl.fit(x_train, y_train)
    y_hat = mdl.predict(x_test)
    prec = precision_score(y_test, y_hat, zero_division=0, average='macro')
    return prec

def choose_parents(population, accuracy, elite_percent):
    elite_num = int(round(((elite_percent * population.shape[0]) // 2) * 2))
    ind_ac = np.argsort(-accuracy)
    top_perc = ind_ac[:elite_num]
    elite_population = population[top_perc, :]

    if accuracy.sum() == 0:
        weight_norm = np.ones_like(accuracy) / len(accuracy)
    else:
        weight_norm = accuracy / accuracy.sum()
    
    weight_comu = weight_norm.cumsum()

    num_parents_wo_elite = population.shape[0] - elite_num
    parents_wo_elite = np.empty([num_parents_wo_elite, population.shape[1]])

    for count in range(num_parents_wo_elite):
        b = weight_comu[-1]
        rand_num = np.random.uniform(0, b)
        indices = np.searchsorted(weight_comu, rand_num)
        parents_wo_elite[count, :] = population[indices, :]

    parents = np.concatenate((elite_population, parents_wo_elite), axis=0)
    return parents

def ga_feature_selection(dataframe, target_col, min_features=5, population_size=50, max_iterations=100, elite_percent=0.4, mutation_probability=0.1):
    target = dataframe[target_col]
    data_predictors = dataframe.drop(columns=[target_col], axis=1)
    predictor_names = data_predictors.columns
    num_features = data_predictors.shape[1]
    max_features = num_features
    min_features = int(os.getenv('min_features'))

    X_train, y_train, X_test, y_test = splitting_data(data_predictors, target, require_val=False)

    population = generate_random_individuals(population_size, num_features, min_features, max_features)
    accuracy = np.zeros(population_size)
    predictor_names = data_predictors.columns
    for i in range(population_size):
        predictor_names_i = predictor_names[population[i,:]==1]
        accuracy[i] = ga_train_model(X_train,X_test,y_train,y_test,predictor_names_i)

    gen = 0
    best_acc_i = np.zeros(max_iterations)
    best_acc_i[gen] = max(accuracy)

    while gen < max_iterations - 1:
        gen += 1
        parents = choose_parents(population, accuracy, elite_percent)
        children = parent_centric_crossover(parents, int(round(((elite_percent * population.shape[0]) // 2) * 2)))
        population = mutation(children, mutation_probability, min_features, max_features)
        population = ensure_minimum_features(population, min_features)

        for ind in range(population_size):
            predictor_names_ind = predictor_names[population[ind, :] == 1]
            accuracy[ind] = ga_train_model(X_train, X_test, y_train, y_test, predictor_names_ind)

        best_acc_i[gen] = max(accuracy)

    ind_max_acc = np.argmax(accuracy)
    best_features = population[ind_max_acc, :]
    best_features_names = predictor_names[best_features == 1]

    return best_features_names
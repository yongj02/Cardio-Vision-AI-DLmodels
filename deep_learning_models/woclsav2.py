import numpy as np
from numpy.random import rand
from deep_learning_models.cnnlstma import cnnlstma

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x

def initialise_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            X[i,d] = int(lb[d] + (ub[d] - lb[d]) * rand())

    return X

# WOA for hyperparameter optimization
def whale_optimization_algorithm(dataframe, target_col, opts={'N': 10, 'T': 3, 'b': 1}):
    # Parameter bounds
    lb = [4, 4, 4, 1]  # Lower bounds of exponentions for neuron1, neuron2, batch_size, and lower bound for dropout
    ub = [11, 11, 9, 5]  # Upper bounds of exponentions for neuron1, neuron2, batch_size, and lower bound for dropout

    N = opts['N']
    max_iter = opts['T']
    b = opts['b']

    dim = len(lb)
    X = initialise_position(lb, ub, N, dim)

    # Fitness evaluation at first iteration
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='int')
    fitG = float('inf')
    best_result = None

    t = 0
    # print("Generation:", t + 1)
    for i in range(N):
        result = cnnlstma(dataframe, target_col, neuron1=2**X[i,0], neuron2=2**X[i,1], batch_size=2**X[i,2], dropout_rate=X[i,3]/10)
        # print(results)
        fit[i,0] = result[0]  # Minimizing loss

        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG = fit[i,0]
            best_result = result

    curve = np.zeros([1, max_iter], dtype='float')
    curve[0,t] = fitG.copy()
    # print("Best (WOA):", curve[0,t])
    # print("Best params:", Xgb)
    t += 1

    while t < max_iter:
        # print("Generation:", t + 1)
        a = 2 - t * (2 / max_iter)

        for i in range(N):
            C = 2 * a * rand() - a
            Q = 2 * rand()
            p = rand()
            k = -1 + 2 * rand()

            if p < 0.5:
                if abs(C) < 1:
                    for d in range(dim):
                        Dx = abs(Q * Xgb[0,d] - X[i,d])
                        X[i,d] = Xgb[0,d] - C * Dx
                        X[i,d] = boundary(X[i,d], lb[d], ub[d])
                elif abs(C) >= 1:
                    for d in range(dim):
                        k = np.random.randint(low=0, high=N)
                        Dx = abs(Q * X[k,d] - X[i,d])
                        X[i,d] = X[k,d] - C * Dx
                        X[i,d] = boundary(X[i,d], lb[d], ub[d])
            elif p >= 0.5:
                for d in range(dim):
                    dist = abs(Xgb[0,d] - X[i,d])
                    X[i,d] = int(dist * np.exp(b * k) * np.cos(2 * np.pi * k) + Xgb[0,d])
                    X[i,d] = boundary(X[i,d], lb[d], ub[d])

        for i in range(N):
            result = cnnlstma(dataframe, target_col, neuron1=2**X[i,0], neuron2=2**X[i,1], batch_size=2**X[i,2], dropout_rate=X[i,3]/10)
            fit[i,0] = result[0]  # Minimizing loss

            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG = fit[i,0]
                best_result = result

        curve[0,t] = fitG.copy()
        # print("Best (WOA):", curve[0,t])
        # print("Best params:", Xgb)
        t += 1

    return best_result
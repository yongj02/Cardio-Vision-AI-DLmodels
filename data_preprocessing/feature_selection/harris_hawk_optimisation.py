import numpy as np
from numpy.random import rand
from splitting_data import splitting_data
from .fs_functions import Fun, binary_conversion, boundary, init_position, levy_distribution

def hho_feature_selection(dataframe, target_col, print_fs, k=5):
    X = dataframe.drop(columns=[target_col], axis=1).values
    y = dataframe[target_col].values

    xtrain, ytrain, xtest, ytest = splitting_data(X, y, train_size=0.8, require_val=False)
    fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

    opts = {
        'N': 50,       # Number of hawks
        'T': 100,      # Maximum number of iterations
        'k': k,        # K-value in KNN
        'fold': fold,
        'beta': 1.5    # Levy component
    }

    selected_indices = hho_jfs(xtrain, ytrain, opts, print_fs)['sf']
    feature_names = dataframe.drop(columns=[target_col]).columns

    selected_features = [feature_names[i] for i in selected_indices]

    return selected_features


def hho_jfs(xtrain, ytrain, opts, print_fs):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    beta  = 1.5    # levy component

    N        = opts['N']
    max_iter = opts['T']
    if 'beta' in opts:
        beta = opts['beta']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X     = init_position(lb, ub, N, dim)

    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xrb   = np.zeros([1, dim], dtype='float')
    fitR  = float('inf')

    curve = np.zeros([1, max_iter], dtype='float')
    t     = 0

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitR:
                Xrb[0,:] = X[i,:]
                fitR     = fit[i,0]

        # Store result
        curve[0,t] = fitR.copy()
        if print_fs:
            print("Iteration:", t + 1)
            print("Best (HHO):", curve[0,t])
        t += 1

        # Mean position of hawk (2)
        X_mu      = np.zeros([1, dim], dtype='float')
        X_mu[0,:] = np.mean(X, axis=0)

        for i in range(N):
            # Random number in [-1,1]
            E0 = -1 + 2 * rand()
            # Escaping energy of rabbit (3)
            E  = 2 * E0 * (1 - (t / max_iter))
            # Exploration phase
            if abs(E) >= 1:
                # Define q in [0,1]
                q = rand()
                if q >= 0.5:
                    # Random select a hawk k
                    k  = np.random.randint(low = 0, high = N)
                    r1 = rand()
                    r2 = rand()
                    for d in range(dim):
                        # Position update (1)
                        X[i,d] = X[k,d] - r1 * abs(X[k,d] - 2 * r2 * X[i,d])
                        # Boundary
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

                elif q < 0.5:
                    r3 = rand()
                    r4 = rand()
                    for d in range(dim):
                        # Update Hawk (1)
                        X[i,d] = (Xrb[0,d] - X_mu[0,d]) - r3 * (lb[0,d] + r4 * (ub[0,d] - lb[0,d]))
                        # Boundary
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

            # Exploitation phase
            elif abs(E) < 1:
                # Jump strength
                J = 2 * (1 - rand())
                r = rand()
                # {1} Soft besiege
                if r >= 0.5 and abs(E) >= 0.5:
                    for d in range(dim):
                        # Delta X (5)
                        DX     = Xrb[0,d] - X[i,d]
                        # Position update (4)
                        X[i,d] = DX - E * abs(J * Xrb[0,d] - X[i,d])
                        # Boundary
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

                # {2} hard besiege
                elif r >= 0.5 and abs(E) < 0.5:
                    for d in range(dim):
                        # Delta X (5)
                        DX     = Xrb[0,d] - X[i,d]
                        # Position update (6)
                        X[i,d] = Xrb[0,d] - E * abs(DX)
                        # Boundary
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

                # {3} Soft besiege with progressive rapid dives
                elif r < 0.5 and abs(E) >= 0.5:
                    # Levy distribution (9)
                    LF = levy_distribution(beta, dim)
                    Y  = np.zeros([1, dim], dtype='float')
                    Z  = np.zeros([1, dim], dtype='float')

                    for d in range(dim):
                        # Compute Y (7)
                        Y[0,d] = Xrb[0,d] - E * abs(J * Xrb[0,d] - X[i,d])
                        # Boundary
                        Y[0,d] = boundary(Y[0,d], lb[0,d], ub[0,d])

                    for d in range(dim):
                        # Compute Z (8)
                        Z[0,d] = Y[0,d] + rand() * LF[d]
                        # Boundary
                        Z[0,d] = boundary(Z[0,d], lb[0,d], ub[0,d])

                    # Binary conversion
                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    # fitness
                    fitY = Fun(xtrain, ytrain, Ybin[0,:], opts)
                    fitZ = Fun(xtrain, ytrain, Zbin[0,:], opts)
                    # Greedy selection (10)
                    if fitY < fit[i,0]:
                        fit[i,0]  = fitY
                        X[i,:]    = Y[0,:]
                    if fitZ < fit[i,0]:
                        fit[i,0]  = fitZ
                        X[i,:]    = Z[0,:]

                # {4} Hard besiege with progressive rapid dives
                elif r < 0.5 and abs(E) < 0.5:
                    # Levy distribution (9)
                    LF = levy_distribution(beta, dim)
                    Y  = np.zeros([1, dim], dtype='float')
                    Z  = np.zeros([1, dim], dtype='float')

                    for d in range(dim):
                        # Compute Y (12)
                        Y[0,d] = Xrb[0,d] - E * abs(J * Xrb[0,d] - X_mu[0,d])
                        # Boundary
                        Y[0,d] = boundary(Y[0,d], lb[0,d], ub[0,d])

                    for d in range(dim):
                        # Compute Z (13)
                        Z[0,d] = Y[0,d] + rand() * LF[d]
                        # Boundary
                        Z[0,d] = boundary(Z[0,d], lb[0,d], ub[0,d])

                    # Binary conversion
                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    # fitness
                    fitY = Fun(xtrain, ytrain, Ybin[0,:], opts)
                    fitZ = Fun(xtrain, ytrain, Zbin[0,:], opts)
                    # Greedy selection (10)
                    if fitY < fit[i,0]:
                        fit[i,0]  = fitY
                        X[i,:]    = Y[0,:]
                    if fitZ < fit[i,0]:
                        fit[i,0]  = fitZ
                        X[i,:]    = Z[0,:]


    # Best feature subset
    Gbin       = binary_conversion(Xrb, thres, 1, dim)
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    hho_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return hho_data
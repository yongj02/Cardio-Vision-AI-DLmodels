import numpy as np
from numpy.random import rand
from splitting_data import splitting_data
from .fs_functions import init_position, binary_conversion, boundary, Fun


def ba_feature_selection(dataframe, target_col, print_fs, k=5):
    X = dataframe.drop(columns=[target_col], axis=1).values
    y = dataframe[target_col].values

    xtrain, ytrain, xtest, ytest = splitting_data(X, y, train_size=0.8, require_val=False)
    fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

    opts = {
        'N': 50,       # Number of bats
        'T': 100,      # Maximum number of iterations
        'fmax': 2,     # Maximum frequency
        'fmin': 0,     # Minimum frequency
        'alpha': 0.9,  # Loudness coefficient
        'gamma': 0.9,  # Pulse rate coefficient
        'A': 2,        # Maximum loudness
        'r': 1,        # Maximum pulse rate
        'k': k,        # k-value in k-nearest neighbour
        'fold': fold
    }

    selected_indices = ba_jfs(xtrain, ytrain, opts, print_fs)['sf']
    feature_names = dataframe.drop(columns=[target_col]).columns

    selected_features = [feature_names[i] for i in selected_indices]

    return selected_features

def ba_jfs(xtrain, ytrain, opts, print_fs):
    # Parameters
    ub     = 1
    lb     = 0
    thres  = 0.5
    fmax   = 2      # maximum frequency
    fmin   = 0      # minimum frequency
    alpha  = 0.9    # constant
    gamma  = 0.9    # constant
    A_max  = 2      # maximum loudness
    r0_max = 1      # maximum pulse rate

    N          = opts['N']
    max_iter   = opts['T']
    if 'fmax' in opts:
        fmax   = opts['fmax']
    if 'fmin' in opts:
        fmin   = opts['fmin']
    if 'alpha' in opts:
        alpha  = opts['alpha']
    if 'gamma' in opts:
        gamma  = opts['gamma']
    if 'A' in opts:
        A_max  = opts['A']
    if 'r' in opts:
        r0_max = opts['r']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position & velocity
    X     = init_position(lb, ub, N, dim)
    V     = np.zeros([N, dim], dtype='float')

    # Binary conversion
    Xbin  = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')

    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t     = 0

    curve[0,t] = fitG.copy()
    if print_fs:
        print("Iteration:", t + 1)
        print("Best (BA):", curve[0,t])
    t += 1

    # Initial loudness [1 ~ 2] & pulse rate [0 ~ 1]
    A  = np.random.uniform(1, A_max, N)
    r0 = np.random.uniform(0, r0_max, N)
    r  = r0.copy()

    while t < max_iter:
        Xnew  = np.zeros([N, dim], dtype='float')

        for i in range(N):
            # beta [0 ~1]
            beta = rand()
            # frequency (2)
            freq = fmin + (fmax - fmin) * beta
            for d in range(dim):
                # Velocity update (3)
                V[i,d]    = V[i,d] + (X[i,d] - Xgb[0,d]) * freq
                # Position update (4)
                Xnew[i,d] = X[i,d] + V[i,d]
                # Boundary
                Xnew[i,d] = boundary(Xnew[i,d], lb[0,d], ub[0,d])

            # Generate local solution around best solution
            if rand() > r[i]:
                for d in range (dim):
                    # Epsilon in [-1,1]
                    eps       = -1 + 2 * rand()
                    # Random walk (5)
                    Xnew[i,d] = Xgb[0,d] + eps * np.mean(A)
                    # Boundary
                    Xnew[i,d] = boundary(Xnew[i,d], lb[0,d], ub[0,d])

        # Binary conversion
        Xbin = binary_conversion(Xnew, thres, N, dim)

        # Greedy selection
        for i in range(N):
            Fnew = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if (rand() < A[i])  and  (Fnew <= fit[i,0]):
                X[i,:]   = Xnew[i,:]
                fit[i,0] = Fnew
                # Loudness update (6)
                A[i]     = alpha * A[i]
                # Pulse rate update (6)
                r[i]     = r0[i] * (1 - np.exp(-gamma * t))

            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]

        # Store result
        curve[0,t] = fitG.copy()
        if print_fs:
            print("Iteration:", t + 1)
            print("Best (BA):", curve[0,t])
        t += 1


    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim)
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    ba_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return ba_data
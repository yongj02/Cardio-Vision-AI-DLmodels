import numpy as np
import torch
import torch.distributed as dist
from numpy.random import rand
from deep_learning_models.cnnlstma import cnnlstma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def boundary(x, lb, ub):
    return max(lb, min(x, ub))

def initialise_position(lb, ub, N, dim):
    return np.array([[int(lb[d] + (ub[d] - lb[d]) * rand()) for d in range(dim)] for _ in range(N)])

def evaluate_batch(rank, world_size, positions, dataframe, target_col):
    results = []
    for position in positions:
        result = cnnlstma(rank, world_size, dataframe, target_col, 
                          2**position[0], 2**position[1], 2**position[2], position[3]/10)
        results.append(result)
    return results

def whale_optimization_algorithm(rank, world_size, dataframe, target_col, opts):
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    lb, ub = opts['lb'], opts['ub']
    N, max_iter, b = opts['N'], opts['T'], opts['b']
    dim = len(lb)
    
    X = torch.tensor(initialise_position(lb, ub, N, dim), device=device)
    fit = torch.zeros(N, 1, dtype=torch.float32, device=device)
    Xgb = torch.zeros(1, dim, dtype=torch.int32, device=device)
    fitG = float('inf')
    best_result = None

    for t in range(max_iter):
        if t > 0:
            a = 2 - t * (2 / max_iter)
            for i in range(N):
                C, Q, p = 2 * a * rand() - a, 2 * rand(), rand()
                if p < 0.5:
                    if abs(C) < 1:
                        X[i] = torch.tensor([boundary(Xgb[0,d].item() - C * abs(Q * Xgb[0,d].item() - X[i,d].item()), lb[d], ub[d]) for d in range(dim)], device=device)
                    else:
                        k = np.random.randint(N)
                        X[i] = torch.tensor([boundary(X[k,d].item() - C * abs(Q * X[k,d].item() - X[i,d].item()), lb[d], ub[d]) for d in range(dim)], device=device)
                else:
                    k = -1 + 2 * rand()
                    X[i] = torch.tensor([boundary(int(abs(Xgb[0,d].item() - X[i,d].item()) * np.exp(b * k) * np.cos(2 * np.pi * k) + Xgb[0,d].item()), lb[d], ub[d]) for d in range(dim)], device=device)

        results = evaluate_batch(rank, world_size, X.cpu().numpy(), dataframe, target_col)
        
        for i, result in enumerate(results):
            fit[i, 0] = result[0]['loss']
            if fit[i, 0] < fitG:
                Xgb[0, :] = X[i, :]
                fitG = fit[i, 0].item()
                best_result = result

        dist.barrier()
        
        all_fitG = [None for _ in range(world_size)]
        dist.all_gather_object(all_fitG, (fitG, Xgb.cpu().numpy(), best_result))
        
        global_best = min(all_fitG, key=lambda x: x[0])
        fitG, Xgb_np, best_result = global_best
        Xgb = torch.tensor(Xgb_np, device=device)
        
        torch.cuda.empty_cache()

    return best_result
import time
import numpy as np
from cnnlstma import cnnlstma
from split_transform import split_transform

class WhaleOptimization:
    """Class implementing the Whale Optimization Algorithm (WOA)"""

    def __init__(self, opt_func, constraints, nsols, b, a, a_step, maximize=False):
        self._opt_func = opt_func
        self.results = {}
        self._constraints = constraints
        self._sols = self._init_solutions(nsols)
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solutions = []

    def get_solutions(self):
        """Return solutions"""
        return self._sols

    def optimize(self):
        """Solutions randomly encircle, search or attack"""
        ranked_sol = self._rank_solutions()
        best_sol = ranked_sol[0]
        # include best solution in next generation solutions
        new_sols = [best_sol]

        for s in ranked_sol[1:]:
            if np.random.uniform(0.0, 1.0) > 0.5:
                A = self._compute_A()
                norm_A = np.linalg.norm(A)
                if norm_A < 1.0:
                    new_s = self._encircle(s, best_sol, A)
                else:
                    random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                    new_s = self._search(s, random_sol, A)
            else:
                new_s = self._attack(s, best_sol)
            new_s = self._discretize(new_s)  # Updated to discretize neurons
            new_sols.append(self._constrain_solution(new_s))

        self._sols = np.stack(new_sols)
        self._a -= self._a_step

    def _init_solutions(self, nsols):
        """Initialize solutions uniformly at random within the constraints"""
        sols = []
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))

        sols = np.stack(sols, axis=-1)
        return sols

    def _constrain_solution(self, sol):
        """Ensure solutions are valid with respect to constraints"""
        constrained_sol = []
        for c, s in zip(self._constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]
            constrained_sol.append(s)
        return constrained_sol

    def _rank_solutions(self):
        """Rank solutions based on fitness"""
        fitness = np.array([self._opt_func(self._discretize(sol)) for sol in self._sols])  # Ensure discretization before evaluation
        sol_fitness = [(f, s) for f, s in zip(fitness, self._sols)]
        ranked_sol = sorted(sol_fitness, key=lambda x: (x[0][1], x[0][2], x[0][5], x[0][0]), reverse=self._maximize)
        self._best_solutions.append(ranked_sol[0])
        return [s[1] for s in ranked_sol]

    def print_best_solutions(self):
        '''
        print("Generation best solution history")
        print("([loss value], [neuron1, neuron2, dropout_rate, batch_size])")
        for s in self._best_solutions:
            print(s)
        print("\nBest solution")
        print("([loss value], [neuron1, neuron2, dropout_rate, batch_size])")
        '''
        best_solution = sorted(self._best_solutions, key=lambda x: (x[0][1], x[0][2], x[0][5], x[0][0]), reverse=self._maximize)[0]
        # print(best_solution)
        return best_solution

    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=4)
        return (2.0 * np.multiply(self._a, r)) - self._a

    def _compute_C(self):
        return 2.0 * np.random.uniform(0.0, 1.0, size=4)

    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)

    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        return np.abs(np.multiply(C, best_sol) - sol)

    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
        return rand_sol - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
        C = self._compute_C()
        return np.abs(np.multiply(C, rand_sol) - sol)

    def _attack(self, sol, best_sol):
        D = np.abs(best_sol - sol)
        L = np.random.uniform(-1.0, 1.0, size=4)
        return np.multiply(np.multiply(D, np.exp(self._b * L)), np.cos(2.0 * np.pi * L)) + best_sol

    def _discretize(self, sol):
        return [int(round(sol[0])), int(round(sol[1])), sol[2], int(round(sol[3]))]
    

def woclsa(dataframe, target_col):
    dataframe = dataframe.copy()
    X_train, y_train, X_valid, y_valid, num_classes = split_transform(dataframe, target_col)

    def obj_function(params):
        return cnnlstma(X_train, y_train, X_valid, y_valid, num_classes, params)
    
    # Define constraints for each parameter (neurons1, neurons2, dropout_rate, batch_size)
    constraints = [
      (128, 512),  # Neurons in first Dense layer
      (128, 512),  # Neurons in second Dense layer
      (0.1, 0.5),  # Dropout rate
      (16, 512),  # Batch size
    ]

    # Initialize and run WOA
    woa = WhaleOptimization(opt_func=obj_function, constraints=constraints, nsols=10, b=1.5, a=2, a_step=0.05, maximize=False)

    # Run the optimization for a specified number of iterations
    max_iter = 10  # Example number of iterations
    for i in range(max_iter):
      print(f"Starting iteration {i + 1}/{max_iter}")
      woa.optimize()

      # Print the best solutions found
      best_solution = woa.print_best_solutions()

      print(f'Best Parameters: {best_solution[1]}')
      print(f'Best Loss: {best_solution[0][0]}')
      print(f'Best Accuracy: {best_solution[0][1]}')
      print(f'Best Precision: {best_solution[0][2]}')
      print(f'Best Recall: {best_solution[0][3]}')
      print(f'Best F1: {best_solution[0][4]}')
      print(f'Best ROC AUC: {best_solution[0][5]}')
      print("\n\n")

    return best_solution[1], best_solution[0]


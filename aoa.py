# ==================== aoa.py ====================
# Archimedes Optimization Algorithm

import numpy as np
from config import AOA_POP_SIZE, AOA_MAX_ITER, AOA_LOWER, AOA_UPPER

def aoa_optimize(fitness_fn, dim):
    # Initialize positions
    pop = np.random.uniform(AOA_LOWER, AOA_UPPER, (AOA_POP_SIZE, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()

    for t in range(AOA_MAX_ITER):
        # Simplified AOA update steps placeholder
        candidate = pop + np.random.normal(scale=0.1, size=pop.shape)
        candidate = np.clip(candidate, AOA_LOWER, AOA_UPPER)
        for i in range(AOA_POP_SIZE):
            f = fitness_fn(candidate[i])
            if f < fitness[i]:
                pop[i] = candidate[i]
                fitness[i] = f
                if f < fitness[best_idx]:
                    best_idx = i
                    best = candidate[i].copy()
    return best
import numpy as np
from scipy.optimize import minimize
from .qaoa_statevector import expected_energy


def optimize_qaoa(energies, depth=1, optimizer="COBYLA", seed=0, maxiter=80):
    rng = np.random.default_rng(seed)
    best_x = np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])

    def obj(x):
        return expected_energy(x, energies, depth)

    if optimizer == "random_search":
        best_val = obj(best_x)
        for _ in range(maxiter):
            x = np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])
            val = obj(x)
            if val < best_val:
                best_x, best_val = x, val
        return best_x, best_val

    method = "Powell" if optimizer.lower() == "powell" else "COBYLA"
    res = minimize(obj, best_x, method=method, options={"maxiter": maxiter, "rhobeg": 0.8} if method == "COBYLA" else {"maxiter": maxiter})
    x = res.x if res.success or hasattr(res, "x") else best_x
    return x, obj(x)


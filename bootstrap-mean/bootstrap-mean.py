import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    rng = np.random.default_rng(seed)  # Modern NumPy RNG generator
    x = np.array(x, dtype=float)
    means = np.empty(n_bootstrap)
    n = x.size
    
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample = x[indices]
        means[i] = np.mean(sample)
        
    means_mean = float(np.mean(means))
    alpha = (1 - ci) / 2
    
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1 - alpha))
    
    return {"bootstrap_mean": means_mean, "lower": lower, "upper": upper}
import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=axis, keepdims=True)
    sigma = np.std(X, axis=axis, keepdims=True)   # ddof=0 par défaut → population
    return (X - mu) / np.maximum(sigma, eps)
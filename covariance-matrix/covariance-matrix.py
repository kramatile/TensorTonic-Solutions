import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X=np.array(X,dtype=float)
    Mu = np.mean(X,axis=0)
    X = X - Mu
    cov = (X.T @ X)/(len(X) - 1)
    return cov
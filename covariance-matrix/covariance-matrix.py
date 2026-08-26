import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X=np.array([np.array(xi,dtype=float) for xi in X])
    Mu = np.mean(X,axis=0)
    for j in range(len(X[0])):
        X[:,j] = X[:,j] - Mu[j]
    Xt = np.transpose(X)
    cov = (Xt @ X)/(len(X) - 1)
    return cov
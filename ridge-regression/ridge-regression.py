import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    I = np.identity(len(X[0]))
    X = np.asarray(X 
                  ,dtype=float)
    y = np.asarray(y 
                  ,dtype=float)
    return np.linalg.inv((X.T @ X + (lam*I))) @ X.T @ y
    
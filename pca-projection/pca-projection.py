import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X_bar = np.mean(X,axis=0,keepdims = True)
    X_c = X - X_bar
    C = X_c.T @ X_c / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eig(C)
    ind = np.argsort(eigenvalues)[::-1]
    top_k_ind = ind[:k]
    keigenvalues = eigenvalues[top_k_ind]
    W = eigenvectors[:,top_k_ind]
    return X_c @ W
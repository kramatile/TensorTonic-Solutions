import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    matrix = np.array(matrix,dtype=float)
    eigen_values, _ = np.linalg.eig(matrix)
    eigen_values = np.sort(eigen_values)    
    return eigen_values
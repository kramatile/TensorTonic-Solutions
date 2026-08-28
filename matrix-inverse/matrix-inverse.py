import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    # Write code here
    if len(A) != len(A[0]):
        return 
    A = np.array(A,dtype=float)
    det = np.linalg.det(A)
    if det==0:
        return 
    def adjugate(A):
        n = A.shape[0]
        adj = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                submatrix = np.delete(np.delete(A, i, axis=0), j, axis=1)
                adj[i,j] = ((-1) ** (i+j))*np.linalg.det(submatrix)
        return adj.T
    return adjugate(A)/det
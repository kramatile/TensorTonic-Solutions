import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    rows = len(A)
    columns = len(A[0])
    if rows == 1 and columns == 1 :
        return np.array(A)
    transpose = np.empty((columns,rows))
    for i in range(rows):
        for j in range(columns): 
            transpose[j][i] = A[i][j]
    return transpose

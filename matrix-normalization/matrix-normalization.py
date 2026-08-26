import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.array(matrix,dtype=float)
    if norm_type == "l2":
        divisor = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
    elif norm_type == "l1":
        divisor = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == "max":
        divisor = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown norm type '{norm_type}'")    
    safe_divisor = np.where(divisor == 0, 1.0, divisor)
    norm_matrix = matrix / safe_divisor
    n_elements = matrix.shape[axis] if axis is not None else matrix.size
    #uniform_value = float(1.0 / n_elements)
    norm_matrix = np.where(divisor == 0, 0, norm_matrix)
    return norm_matrix
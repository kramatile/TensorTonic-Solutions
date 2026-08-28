import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x_array = np.array(x, dtype=float)
    return np.array(np.maximum(x,0.0))

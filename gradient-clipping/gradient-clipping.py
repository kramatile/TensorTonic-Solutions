import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g = np.asarray(g,dtype=float)
    l2 = np.sqrt(np.sum(g**2))
    if l2 <= max_norm : 
        return g
    return g*max_norm/l2

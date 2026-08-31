import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype=float)
    erfs = np.array([math.erf(xi / math.sqrt(2)) for xi in x.flat]).reshape(x.shape)
    
    return 0.5 * x * (1.0 + erfs)    
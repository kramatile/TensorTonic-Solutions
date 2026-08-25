import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    x_n = np.array(x)
    y_n = np.array(y)
    dot = x_n * y_n 
    return float(np.sum(dot,dtype=np.float64))
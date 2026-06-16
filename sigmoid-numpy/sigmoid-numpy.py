import numpy as np
def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    numpy_x = np.array(x)
    sigmoid = 1.0 / (1.0  + np.exp(-numpy_x))    
    return sigmoid
    

import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.asarray(x,dtype=float) 
    if x.ndim > 1:

        maxs = np.max(x,axis=1,keepdims=True)
        x_rescaled = x - maxs
        sum_rescaled = np.sum(np.exp(x_rescaled),axis=1,keepdims=True)
    else :    
        maxs = np.max(x)
        x_rescaled = x - maxs
        sum_rescaled = np.sum(np.exp(x_rescaled),keepdims=True)
    print(maxs)
    print(x_rescaled)
    return np.exp(x_rescaled)/sum_rescaled
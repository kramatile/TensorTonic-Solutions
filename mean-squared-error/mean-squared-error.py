import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    if len(y_pred) != len(y_true) : 
        return None
    return np.sum((np.array(y_pred)- np.array(y_true))**2) / len(y_pred)
    

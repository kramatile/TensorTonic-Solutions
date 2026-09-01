import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)
    
    y_left = y[split_mask]
    y_right = y[~split_mask]
    
    def entropy(x):
        if len(x) == 0:
            return 0.0
        _, counts = np.unique(x, return_counts=True)
        probas = counts / len(x)  
        return -np.sum(probas * np.log2(probas))

    h_parent = entropy(y)
    h_left = entropy(y_left)
    h_right = entropy(y_right)
    
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    
    return float(h_parent - (weight_left * h_left + weight_right * h_right))
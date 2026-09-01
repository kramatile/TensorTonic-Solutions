import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y,dtype=int)
    classes, counts = np.unique(y,return_counts=True)
    probabilities = counts / len(y)
    class_counts = dict(zip(classes,probabilities))
    entropy = 0.0
    for k,v in class_counts.items():
        entropy -= v*np.log2(v)
    return entropy
        
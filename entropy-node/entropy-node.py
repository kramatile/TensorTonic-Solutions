import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if not y : 
        return 0.0
        
    elif len(y)==0:
        return 0.0
        
    entropy = 0
    values, counts = np.unique(y,return_counts=True)
    print(np.unique(y,return_counts=True))
    total_counts = np.sum(counts)
    for v,c in zip(values,counts):
        p = c/total_counts
        if p == 0 :
            continue
        entropy += p * np.log2(p)
    return -entropy
import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    n_x = np.array(x)
    sum = np.sum(n_x)
    n = len(x)
    mean = sum/n
    n_x = np.sort(n_x)
    print(n_x)
    if n % 2 != 0 : 
        middle = (n+1)//2
        median = n_x[middle-1]
    
    else : 
        middle = n//2
        median = (n_x[middle-1] + n_x[middle])/2
    ctr = Counter(x)
    mode = ctr.most_common(1)[0][0]
    return (mean,median,mode)
import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    mean = np.mean(x)
    x = np.array(x)
    ecarts = (x - mean)**2
    var = float(np.sum(ecarts)/(len(x)-1))
    return {"variance":var,"standard_deviation":float(np.sqrt(var))}
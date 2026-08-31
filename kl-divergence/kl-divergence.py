import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p,dtype=float) 
    q = np.asarray(q,dtype=float) 
    p = np.clip(p,a_min=eps,a_max=None)
    q = np.clip(q,a_min=eps,a_max=None)
    return float(np.sum(p*np.log(p/q)))
    
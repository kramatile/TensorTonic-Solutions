import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w = np.asarray(w,dtype=float)
    g = np.asarray(g,dtype=float)
    s = np.asarray(s,dtype=float)
    s = beta*s + (1-beta)*(g**2)
    w = w - (lr/(np.sqrt(s+eps)))*g
    return (w,s)
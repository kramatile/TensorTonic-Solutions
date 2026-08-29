import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    w = np.asarray(w,dtype=float)
    v = np.asarray(v,dtype=float)
    grad =np.asarray(grad,dtype=float)
    v = momentum*v + lr*grad
    w = w - v 
    
    return {"new_w": w, 'new_v': v}
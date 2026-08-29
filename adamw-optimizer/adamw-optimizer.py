import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    w=np.asarray(w,dtype=float)
    m=np.asarray(m,dtype=float)
    v=np.asarray(v,dtype=float)
    grad=np.asarray(grad,dtype=float)

    m = beta1 * m + (1 - beta1)*grad
    v = beta2 * v + (1 - beta2)*(grad**2)

    w = w - lr*(m/(np.sqrt(v)+eps)) - lr*weight_decay*w
 

    return {"new_w":w,"new_m":m,"new_v":v}
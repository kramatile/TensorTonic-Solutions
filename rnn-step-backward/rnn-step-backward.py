import numpy as np

def rnn_step_backward(dh: list, cache: list) -> dict:
    """
    Returns a dictionary with dx_t, dh_prev, dW, dU, and db.
    """
    dh = np.asarray(dh,dtype=np.float64)
    

    dz = dh * (1 - np.asarray(cache[2],float)**2)
    dx_t = np.asarray(cache[3],float).T @ dz
    db = dz 
    dh_prev = np.asarray(cache[4],float).T @ dz
    dW = np.outer(dz ,np.asarray(cache[0],float).T)

    dU = np.outer(dz ,np.asarray(cache[1],float).T)

    return {"dx_t": dx_t, "dh_prev": dh_prev, "dW": dW, "dU": dU, "db":db}
    
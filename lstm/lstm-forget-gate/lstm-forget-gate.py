import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """
    Returns the float64 forget-gate values.
    """
    
    hx = np.concatenate((np.asarray(h_prev, dtype=np.float64),
                          np.asarray(x_t,   dtype=np.float64)), axis=-1)
    z_f =  hx@W_f.T + np.asarray(b_f,dtype=np.float64)
    return np.asarray(sigmoid(z_f),dtype=np.float64)
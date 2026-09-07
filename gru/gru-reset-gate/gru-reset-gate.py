import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
          W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Returns the float64 reset-gate values.
    """
    hx = np.concatenate((h_prev,x_t),axis=-1)
    r_t = sigmoid(hx@W_r.T + b_r)
    return r_t
    #rest_concat = np.concatenate((r_t*h_prev,x_t),axis=-1)
    #h_candidate = np.tanh(rest_concat@)
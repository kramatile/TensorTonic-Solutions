import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    h_prev = h_0
    h_list = []
    T = X.shape[1]
    for t in range(T):
        x_t = X[:, t, :]                                    
        h_prev = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)  
        h_list.append(h_prev)
    
    return {"hidden_states": np.stack(h_list, axis=1).astype(np.float64),"final_hidden_state":np.asarray(h_prev,dtype=np.float64)}
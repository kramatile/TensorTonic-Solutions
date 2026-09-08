import numpy as np
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns hidden_state and cell_state as float64 arrays.
    """
    hx = np.concatenate((h_prev,x_t),axis=-1)
    f_t = sigmoid(hx@W_f.T+b_f)
    i_t = sigmoid(hx@W_i.T+b_i)
    c_tilde_t = np.tanh(hx@W_c.T + b_c)
    c_t = f_t*C_prev + i_t*c_tilde_t
    o_t = sigmoid(hx@W_o.T+b_o)
    h_t = o_t * np.tanh(c_t)
    return {"hidden_state":h_t,"cell_state":c_t,"output":o_t}
def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    """
    X = np.asarray(X, dtype=np.float64)
    B, T, _ = X.shape
    H = len(W_c)

    h_prev = np.zeros((B, H), dtype=np.float64)     # états initialisés par lot
    C_prev = np.zeros((B, H), dtype=np.float64)

    outputs = []
    for t in range(T):
        state = lstm_cell(X[:, t, :], h_prev, C_prev,
                          W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o)
        h_prev = state["hidden_state"]
        C_prev = state["cell_state"]
        outputs.append(h_prev @ W_y.T + b_y)         

    return {"outputs": np.stack(outputs, axis=1),
            "final_hidden_state": h_prev,
            "final_cell_state": C_prev}
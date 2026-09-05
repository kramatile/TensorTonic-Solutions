import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    x_t = np.asarray(x_t,float)
    h_prev = np.asarray(h_prev,float)
    Wx = np.asarray(Wx,float)
    Wh = np.asarray(Wh,float)
    b = np.asarray(b,float)
    return np.tanh(x_t@Wx+ h_prev@Wh + b)
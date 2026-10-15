import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pos = np.arange(seq_length)[:,None]
    i = np.arange(d_model//2)[None,:]
    product = pos/(10000**(2*i/d_model))
    pe = np.zeros((seq_length,d_model))
    pe[:,0::2] = np.sin(product)
    pe[:,1::2] = np.cos(product)
    return pe






    
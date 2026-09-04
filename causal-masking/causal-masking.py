import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    q_len, k_len = scores.shape[-2:]
    scores = np.asarray(scores)
    n_dim = scores.ndim
    mask = np.empty((q_len,k_len),bool)
    for i in range(q_len):
        for j in range(k_len):
            mask[i][j] = j > i 

    mask = np.where(mask,mask_value,scores)
    return mask
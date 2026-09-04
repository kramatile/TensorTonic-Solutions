import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    q_len, k_len = scores.shape[-2:]
    scores = np.asarray(scores)
    n_dim = scores.ndim
    q_idx = np.arange(q_len)[:,None]
    k_idx = np.arange(k_len)[None,:]
    mask = k_idx > q_idx

    mask = np.where(mask,mask_value,scores)
    return mask
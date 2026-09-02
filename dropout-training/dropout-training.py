import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.asarray(x,dtype=float)
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(0)
    mask = (rng.random(x.shape) < 1-p).astype(float) /(1-p)
    return x*mask,mask

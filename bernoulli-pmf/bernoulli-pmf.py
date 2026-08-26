import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    def bernouli(x:int)->float:
        if x == 0 :
            return 1-p
        return p
    pmf = np.array([bernouli(xi) for xi in x])
    mu = float(p)
    var = float(p*(1-p))
    return {"pmf":pmf,"mean":mu,"variance":var}
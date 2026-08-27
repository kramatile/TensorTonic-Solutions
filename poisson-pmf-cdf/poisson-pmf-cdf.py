import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    def poisson(i:int) :
        return math.exp(-lam)*(lam**i) / math.factorial(i)

    probas = [poisson(i) for i in range(k+1)]
    return {"cdf": sum(probas),"pmf":probas[-1]}
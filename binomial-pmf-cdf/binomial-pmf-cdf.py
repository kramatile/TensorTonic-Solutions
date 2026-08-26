import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    def binomial_proba(n,k):
        def combination(n,k):
            fact_n = math.factorial(n)
            fact_k = math.factorial(k)
            fact_n_k = math.factorial(n-k)
            return fact_n /(fact_k*fact_n_k)
        return combination(n,k)*(p**k)*((1-p)**(n-k))
    probas = [binomial_proba(n,i) for i in range(k+1)]
    return {"pmf":probas[-1],"cdf":sum(probas)}
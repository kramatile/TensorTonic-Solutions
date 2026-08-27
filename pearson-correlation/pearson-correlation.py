import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    def covariance_matrix(X:np.ndarray):
        X=np.array(X,dtype=float)
        Mu = np.mean(X,axis=0)
        X = X - Mu
        cov = (X.T @ X)/(len(X) - 1)
        return cov
    cov = covariance_matrix(X)
    var = np.sqrt(np.diag(cov))
    #var = np.empty((len(cov),1),dtype=float)
    #for i in range(len(cov)):
    #    var[i][0] = np.sqrt(float(cov[i][i]))
    var_prod = np.outer(var,var.T)
    return cov/var_prod
    
    
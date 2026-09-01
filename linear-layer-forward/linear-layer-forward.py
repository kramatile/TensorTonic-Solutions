def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    y = [[0.0] * len(W[0]) for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(W[0])):
            xw = 0.0
            for k in range(len(W)):
                xw += X[i][k]*W[k][j] 
                
            y[i][j] =  xw + b[j]
    return y
            
    
    
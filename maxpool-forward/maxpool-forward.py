def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    h_out =(len(X) - pool_size)//stride + 1
    w_out = (len(X[0]) - pool_size)//stride + 1
    out = [[0.0 for _ in range(w_out)] for _ in range(h_out)]
    for i in range(h_out):
        for j in range(w_out):
            max = -float("inf")
            for a in range(pool_size):
                for b in range(pool_size):
                    if max < X[int(i*stride+a)][int(j*stride+b)]:
                        max = X[int(i*stride+a)][int(j*stride+b)]
            out[i][j] = max
    return out
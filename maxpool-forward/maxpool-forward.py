def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    h_out =(len(X) - pool_size)//stride + 1
    w_out = (len(X[0]) - pool_size)//stride + 1
    out = [[0.0 for _ in range(w_out)] for _ in range(h_out)]
    for i in range(h_out):
        for j in range(w_out):
            r_start, r_end = i * stride, i * stride + pool_size
            c_start, c_end = j * stride, j * stride + pool_size
            maximum = max(
                val for 
                row in X[r_start:r_end]
                for val in row[c_start:c_end]
            )
            out[i][j] = maximum
    return out
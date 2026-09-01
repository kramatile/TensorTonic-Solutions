import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    bound = math.sqrt(6/(fan_in))
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i][j] = (W[i][j]*2*bound)  - bound
    return W
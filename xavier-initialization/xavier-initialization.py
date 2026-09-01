import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    bound = math.sqrt(6/(fan_in+fan_out))
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i][j] = (W[i][j]*2*bound)  - bound

    return W
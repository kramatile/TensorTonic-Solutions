import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    expected = 0.0
    for xi, pi in zip(x,p):
        expected += xi*pi
        
    return expected
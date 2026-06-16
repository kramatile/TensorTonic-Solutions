import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    numpy_x = np.array(x)
    numpy_y = np.array(y)

    difference = numpy_x - numpy_y
    abs_difference = np.absolute(difference)
    return np.sum(abs_difference).item()
import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    y_left = np.asarray(y_left,dtype=int)
    y_right = np.asarray(y_right,dtype=int)
    
    y_left_cat, y_left_count = np.unique(y_left,return_counts=True)
    y_right_cat, y_right_count = np.unique(y_right,return_counts=True)

    left_probas = y_left_count/len(y_left)
    right_probas = y_right_count/len(y_right)

    gini_left = 1 - np.sum(left_probas**2)
    gini_right = 1 - np.sum(right_probas**2)
    if len(y_left)+len(y_right) == 0:
        return 0 
    gini =( len(y_left)/(len(y_left)+len(y_right)) *gini_left)
    gini+=( len(y_right)/(len(y_left)+len(y_right)) *gini_right)
    
    return gini
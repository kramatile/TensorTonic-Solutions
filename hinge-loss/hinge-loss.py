import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    scores = margin - np.array(y_true) * np.array(y_score)
    hinge_scores = np.clip(scores,0,None)
    sum_hinge_scores = np.sum(hinge_scores)
    if reduction == "sum" : 
        return sum_hinge_scores
    return sum_hinge_scores / len(hinge_scores)
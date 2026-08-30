import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_true = np.asarray(y_true,dtype=float)
    y_pred = np.asarray(y_pred,dtype=float)
    y_pred = np.clip(y_pred, a_min=1e-15, a_max=None)
    loss = 0.0
    for i in range(len(y_true)):
        loss += -np.log(y_pred[i,int(y_true[i])])
    return loss/len(y_true)

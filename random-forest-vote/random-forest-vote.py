import numpy as np

def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    Assumes shape: (n_trees, n_samples)
    """
    preds = np.asarray(predictions)
    votes = []
    
    for col_idx in range(preds.shape[1]):
        col = preds[:, col_idx]
        vals, counts = np.unique(col, return_counts=True)
        votes.append(vals[counts == counts.max()].min())
        
    return votes
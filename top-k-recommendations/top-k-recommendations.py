import numpy as np
def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    unrated_items = [
        (score, index) 
        for index, score in enumerate(scores) 
        if index not in rated_indices
    ]
    unrated_items.sort(key=lambda x: x[0], reverse=True)
    top_k_indices = [index for score, index in unrated_items[:k]]
    return top_k_indices
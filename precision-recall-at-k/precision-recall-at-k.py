def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_three = set(recommended[:k])
    relevant_set = set(relevant)
    intersection_len = len(list(top_three & relevant_set))
    return [intersection_len/k,intersection_len/len(relevant)   ]
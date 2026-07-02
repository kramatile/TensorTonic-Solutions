def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    total_hits = 0
    for rec, gt in zip(recommendations,ground_truth):
        set_recs = set(rec[:k])
        set_gt = set(gt)
        intersection = set_gt & set_recs
        total_hits += len(intersection)
    return total_hits/len(recommendations)
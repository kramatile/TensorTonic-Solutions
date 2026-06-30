def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    if not set_a and not set_b :
        return 0.0
    set_a = set(set_a)
    set_b = set(set_b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
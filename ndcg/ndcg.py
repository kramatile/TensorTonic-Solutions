import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    dgck = 0
    idgc = 0
    sorted_relevances = sorted(relevance_scores, reverse=True)
    for i in range(min(k,len(relevance_scores))):
        dgck += (2**relevance_scores[i] - 1) / math.log2(i+2)
        idgc += (2**sorted_relevances[i] - 1) / math.log2(i+2)
    return dgck / idgc if idgc != 0 else 0

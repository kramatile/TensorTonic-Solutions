import numpy as np

def bigram_probabilities(tokens: list) -> dict:
    """
    Returns a dictionary with vocab, counts, and probabilities.
    """
    unique = np.unique(tokens)
    V = len(unique)
    unique_idx = {token: idx for idx, token in enumerate(unique)}

    counts = np.zeros((V, V), dtype=int)
    for i in range(len(tokens) - 1):
        counts[unique_idx[tokens[i]], unique_idx[tokens[i + 1]]] += 1

    row_totals = counts.sum(axis=1, keepdims=True)
    probas = (counts + 1) / (row_totals + V)

    return {
        "vocab": unique.tolist(),
        "counts": counts,
        "probabilities": probas,
    }
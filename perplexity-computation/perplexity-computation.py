import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    cross_entropy = 0.0
    for prob, token in zip(prob_distributions,actual_tokens) : 
        cross_entropy -= math.log(prob[token]) if prob[token] != 0 else 0.0
    cross_entropy = cross_entropy/(len(actual_tokens))
    perplexity = math.exp(cross_entropy)
    return perplexity 
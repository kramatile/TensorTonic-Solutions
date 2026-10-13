import torch

def noise_distribution(counts: torch.Tensor,
                       alpha: float = 0.75) -> torch.Tensor:
    """
    Returns the float64 negative-sampling distribution over the vocabulary.
    """
    normalized_counts = torch.pow(counts, alpha)
    Z = torch.sum(normalized_counts)
    return normalized_counts/Z 
    

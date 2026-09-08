import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    f_w = counts/counts.sum(dim=-1, keepdim=True)
    p_keep = torch.minimum(torch.sqrt(t / f_w), torch.tensor(1.0))
    return p_keep
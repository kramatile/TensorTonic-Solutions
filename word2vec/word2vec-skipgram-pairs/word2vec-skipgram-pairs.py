import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    pairs = []
    n = len(token_ids)
    for i in range(n):
        for j in range(max(0,i-window),min(n-1,i+window)+1):
            if i != j : 
                pairs.append([token_ids[i].item(), token_ids[j].item()])
    if not pairs :
        return torch.empty((0,2),dtype=torch.int64)
    return torch.tensor(pairs,dtype=torch.int64)
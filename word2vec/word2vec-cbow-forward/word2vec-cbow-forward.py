import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int,
                 W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 CBOW cross-entropy loss.
    """
    if context_ids.dim() > 1:
        context_ids = context_ids.squeeze(0)
    m = context_ids.shape[0]
    embed_dim = W_in.shape[1]
    h = torch.zeros(embed_dim, dtype=W_in.dtype, device=W_in.device)
    for i in range(m):
        h += W_in[context_ids[i]]
        
    h = h/m
    z = torch.mv(W_out,h)
    return - F.log_softmax(z)[target_id]
    
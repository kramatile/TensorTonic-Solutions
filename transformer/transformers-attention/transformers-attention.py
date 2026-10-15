import torch
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k = K.shape[-1]
    O = Q @ K.transpose(-2,-1) / math.sqrt(d_k)
    sof = torch.nn.Softmax(dim=-1)
    return sof(O)@V
    
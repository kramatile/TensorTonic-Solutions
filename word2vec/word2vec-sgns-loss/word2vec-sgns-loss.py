import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 SGNS loss.
    """

    neg_sum = F.softplus(-torch.dot(center_vec.T,pos_vec))
    K = neg_vecs.shape[1]
    for neg_vec in neg_vecs:
        neg_sum += F.softplus(torch.dot(center_vec.T,neg_vec))
    return neg_sum
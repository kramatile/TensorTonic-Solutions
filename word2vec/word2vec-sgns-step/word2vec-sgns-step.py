import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    new_W_in = W_in.clone()
    new_W_out = W_out.clone()
    center = W_in[center_id].clone()
    output_snapshot = W_out.clone()
    pos_vec = W_out[pos_id].clone()
    grad_u0 = (torch.sigmoid(torch.dot(center,pos_vec)) - 1.0)*center
    grad_ui = {}
    grad_vc = (torch.sigmoid(torch.dot(center,pos_vec)) - 1.0)*pos_vec
    for negative_id in neg_ids.tolist():
        coef = torch.sigmoid(torch.dot(center, output_snapshot[negative_id]))
        grad_ui[negative_id] = coef * center
        grad_vc = grad_vc + coef * output_snapshot[negative_id]

    new_W_in[center_id] -= lr*grad_vc
    new_W_out[pos_id] -= lr*grad_u0
    for negative_id in neg_ids.tolist():
        new_W_out[negative_id] -= lr*grad_ui[negative_id]
    return {"W_in":new_W_in,"W_out":new_W_out}
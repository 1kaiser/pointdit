# coding=utf-8

import torch


def rel_point_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Relative point loss - normalizes error by distance from origin.
    Directly optimizes rel_point_metric.

    Args:
        pred: Predicted points [B, 3, H, W] or [B, H, W, 3]
        gt: Ground truth points [B, 3, H, W] or [B, H, W, 3]
        eps: Small constant for numerical stability

    Returns:
        Loss tensor [B, H, W]
    """
    # Handle [B, 3, H, W] -> [B, H, W, 3]
    if pred.shape[1] == 3 and pred.dim() == 4:
        pred = pred.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)

    dist_gt = torch.norm(gt, dim=-1)  # [B, H, W]
    dist_err = (pred - gt).abs().sum(dim=-1)  # L1 norm
    return dist_err / (dist_gt + eps)

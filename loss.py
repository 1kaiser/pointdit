# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

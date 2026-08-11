# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Portions Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# Modified from D-FINE: https://github.com/Peterande/D-FINE
"""Fine-grained Distribution Refinement (FDR) utilities for D-FINE."""

from __future__ import annotations

import torch
from torch import Tensor

from ultralytics.utils.ops import xyxy2xywh


def weighting_function(reg_max: int, up: Tensor, reg_scale: float | Tensor, deploy: bool = False) -> Tensor:
    """Generate the non-uniform weighting function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of discrete bins.
        up (torch.Tensor): Controls upper bounds of the sequence (±up * H / W).
        reg_scale (float | torch.Tensor): Controls curvature of W(n).
        deploy (bool): If True, use deployment-mode construction.

    Returns:
        (torch.Tensor): Weighting function values of length ``reg_max + 1``.
    """
    # Official DFINETransformer passes Tensor reg_scale; coerce floats for robustness.
    if not isinstance(reg_scale, torch.Tensor):
        reg_scale = torch.tensor([reg_scale], dtype=up.dtype, device=up.device)
    else:
        reg_scale = reg_scale.to(dtype=up.dtype, device=up.device)

    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


def translate_gt(gt: Tensor, reg_max: int, reg_scale: float | Tensor, up: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Map continuous GT edge distances into distribution-bin indices and interpolation weights.

    Args:
        gt (torch.Tensor): Ground-truth values, any shape (flattened internally).
        reg_max (int): Maximum number of discrete bins.
        reg_scale (float | torch.Tensor): Curvature control for W(n).
        up (torch.Tensor): Upper-bound control for W(n).

    Returns:
        indices (torch.Tensor): Left-bin indices closest to each GT value.
        weight_right (torch.Tensor): Interpolation weight for the right bin.
        weight_left (torch.Tensor): Interpolation weight for the left bin.
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    indices = closest_left_indices.float()
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]
    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points: Tensor, distance: Tensor, reg_scale: float | Tensor) -> Tensor:
    """Decode edge distances into cxcywh boxes.

    Args:
        points (torch.Tensor): (..., 4) reference boxes [cx, cy, w, h].
        distance (torch.Tensor): (..., 4) distances to left/top/right/bottom.
        reg_scale (float | torch.Tensor): Curvature / scale control.

    Returns:
        (torch.Tensor): Boxes in cxcywh format with the same leading shape as ``points``.
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return xyxy2xywh(bboxes)


def bbox2distance(
    points: Tensor,
    bbox: Tensor,
    reg_max: int,
    reg_scale: float | Tensor,
    up: Tensor,
    eps: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert xyxy boxes to distribution targets relative to reference points.

    Args:
        points (torch.Tensor): (n, 4) reference [cx, cy, w, h].
        bbox (torch.Tensor): (n, 4) boxes in xyxy format.
        reg_max (int): Maximum bin value.
        reg_scale (float | torch.Tensor): Curvature control for W(n).
        up (torch.Tensor): Upper-bound control for W(n).
        eps (float): Small value ensuring target < reg_max.

    Returns:
        four_lens (torch.Tensor): Flattened left-bin indices (detached).
        weight_right (torch.Tensor): Right-bin interpolation weights (detached).
        weight_left (torch.Tensor): Left-bin interpolation weights (detached).
    """
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()

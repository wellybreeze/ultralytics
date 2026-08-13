# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Portions Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# Modified from D-FINE: https://github.com/Peterande/D-FINE
# ---------------------------------------------------------------------------------
# Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
# Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""D-FINE detection loss (VFL + L1/GIoU + FGL + DDF) adapted for Ultralytics targets."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.dfine_utils import bbox2distance
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy

from .ops import HungarianMatcher


def _is_dist_avail_and_initialized() -> bool:
    """Return True if torch.distributed is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_world_size() -> int:
    """Return distributed world size or 1."""
    return torch.distributed.get_world_size() if _is_dist_avail_and_initialized() else 1


class DFINEDetectionLoss(nn.Module):
    """D-FINE criterion ported for Ultralytics RT-DETR-style batch targets.

    Computes Varifocal (VFL), L1 / GIoU box, Fine-Grained Localization (FGL), and Decoupled Distillation Focal (DDF)
    losses with GO-LSD matching union across decoder layers.

    Args:
        nc (int): Number of classes.
        loss_gain (dict[str, float], optional): Relative weights for loss terms.
        losses (list[str], optional): Loss types to compute (``vfl``, ``boxes``, ``local``).
        alpha (float): VFL balancing factor (official default 0.75).
        gamma (float): Focusing parameter for VFL / focal matching.
        reg_max (int): FDR discrete bins.
        boxes_weight_format (str | None): Optional ``iou`` / ``giou`` box weighting.
        matcher (HungarianMatcher | None): Optional matcher override.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        losses: list[str] | None = None,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reg_max: int = 32,
        boxes_weight_format: str | None = None,
        matcher: HungarianMatcher | None = None,
    ):
        """Initialize D-FINE detection loss."""
        super().__init__()
        if loss_gain is None:
            loss_gain = {
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            }
        self.nc = nc
        self.num_classes = nc
        self.weight_dict = loss_gain
        self.losses = losses or ["vfl", "boxes", "local"]
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.boxes_weight_format = boxes_weight_format
        self.matcher = matcher or HungarianMatcher(
            cost_gain={"class": 2, "bbox": 5, "giou": 2},
            use_fl=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        self.device = None

    @staticmethod
    def _batch_to_targets(batch: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        """Convert Ultralytics ``{cls, bboxes, gt_groups}`` batch to list-of-dicts targets."""
        gt_groups = batch["gt_groups"]
        cls = batch["cls"]
        bboxes = batch["bboxes"]
        targets, start = [], 0
        for n in gt_groups:
            end = start + n
            targets.append(
                {
                    "labels": cls[start:end].long(),
                    "boxes": bboxes[start:end],
                }
            )
            start = end
        return targets

    def _match(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run Hungarian matching; return per-image local (src, tgt) indices."""
        bs = outputs["pred_logits"].shape[0]
        gt_groups = [len(t["labels"]) for t in targets]
        if sum(gt_groups) == 0:
            empty = torch.tensor([], dtype=torch.long)
            return [(empty, empty) for _ in range(bs)]

        gt_cls = torch.cat([t["labels"] for t in targets])
        gt_bboxes = torch.cat([t["boxes"] for t in targets])
        # CDN splits can leave non-contiguous views; matcher uses .view().
        indices = self.matcher(
            outputs["pred_boxes"].contiguous(),
            outputs["pred_logits"].contiguous(),
            gt_bboxes,
            gt_cls,
            gt_groups,
        )
        # Ultralytics matcher returns flat GT indices; convert to per-image local indices.
        offsets = torch.as_tensor([0, *gt_groups[:-1]]).cumsum(0)
        return [(src, dst - offsets[i]) for i, (src, dst) in enumerate(indices)]

    def loss_labels_vfl(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
        values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Varifocal classification loss."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious = bbox_iou(src_boxes, target_boxes, xywh=True).squeeze(-1).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
        boxes_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """L1 and GIoU box regression losses."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1.0 - bbox_iou(src_boxes, target_boxes, xywh=True, GIoU=True).squeeze(-1)
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_local(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
        T: int = 5,
    ) -> dict[str, torch.Tensor]:
        """Fine-Grained Localization (FGL) and Decoupled Distillation Focal (DDF) losses."""
        losses = {}
        if "pred_corners" not in outputs:
            return losses

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
        ref_points = outputs["ref_points"][idx].detach()
        with torch.no_grad():
            if self.fgl_targets_dn is None and "is_dn" in outputs:
                self.fgl_targets_dn = bbox2distance(
                    ref_points,
                    xywh2xyxy(target_boxes),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )
            if self.fgl_targets is None and "is_dn" not in outputs:
                self.fgl_targets = bbox2distance(
                    ref_points,
                    xywh2xyxy(target_boxes),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )

        target_corners, weight_right, weight_left = self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets

        ious = bbox_iou(outputs["pred_boxes"][idx], target_boxes, xywh=True).squeeze(-1)
        weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

        losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
            pred_corners,
            target_corners,
            weight_right,
            weight_left,
            weight_targets,
            avg_factor=num_boxes,
        )

        if "teacher_corners" in outputs:
            pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
            target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
            if torch.equal(pred_corners, target_corners):
                losses["loss_ddf"] = pred_corners.sum() * 0
            else:
                weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                loss_match_local = (
                    weight_targets_local
                    * (T**2)
                    * (
                        nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners / T, dim=1),
                            F.softmax(target_corners.detach() / T, dim=1),
                        )
                    ).sum(-1)
                )
                if "is_dn" not in outputs:
                    batch_scale = 8 / outputs["pred_boxes"].shape[0]
                    self.num_pos, self.num_neg = (
                        (mask.sum() * batch_scale) ** 0.5,
                        ((~mask).sum() * batch_scale) ** 0.5,
                    )
                loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses["loss_ddf"] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                    self.num_pos + self.num_neg
                )

        return losses

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Permute predictions following matching indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_go_indices(
        self,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        indices_aux_list: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get a matching union set across all decoder layers (GO-LSD)."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        """Clear cached FGL / DDF targets between forward passes."""
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Dispatch to a named loss function."""
        loss_map = {
            "boxes": self.loss_boxes,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, Any], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Compute D-FINE losses from decoder outputs and Ultralytics batch targets.

        Args:
            outputs (dict[str, torch.Tensor]): DFINEDecoder train dict with ``pred_logits``, ``pred_boxes``,
                ``pred_corners``, ``ref_points``, ``up``, ``reg_scale``, and optional ``aux_outputs``,
                ``enc_aux_outputs``, ``pre_outputs``, ``dn_*``, meta.
            batch (dict[str, Any]): Ultralytics targets with ``cls``, ``bboxes``, ``gt_groups`` (and optionally
                ``batch_idx``).

        Returns:
            (dict[str, torch.Tensor]): Weighted loss dictionary (main + aux / enc / dn terms).
        """
        targets = self._batch_to_targets(batch)
        self.device = outputs["pred_logits"].device

        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self._match(outputs_without_aux, targets)
        self._clear_cache()

        # Matching union set across decoder / encoder layers (GO-LSD).
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for aux_outputs in outputs["aux_outputs"] + [outputs["pre_outputs"]]:
                indices_aux = self._match(aux_outputs, targets)
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for aux_outputs in outputs["enc_aux_outputs"]:
                indices_enc = self._match(aux_outputs, targets)
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=self.device)
            if _is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / _get_world_size(), min=1).item()
        else:
            raise AssertionError("DFINEDetectionLoss expects aux_outputs from DFINEDecoder")

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=self.device)
        if _is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / _get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            dn_meta = outputs["dn_meta"]
            if "dn_positive_idx" not in dn_meta and "dn_pos_idx" in dn_meta:
                dn_meta = {**dn_meta, "dn_positive_idx": dn_meta["dn_pos_idx"]}
            indices_dn = self.get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = num_boxes * dn_meta["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

    def get_loss_meta_info(
        self,
        loss: str,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Optional IoU-based meta for box / VFL weighting."""
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou = bbox_iou(src_boxes.detach(), target_boxes, xywh=True).squeeze(-1)
        elif self.boxes_weight_format == "giou":
            iou = bbox_iou(src_boxes.detach(), target_boxes, xywh=True, GIoU=True).squeeze(-1)
        else:
            raise AttributeError(f"Unsupported boxes_weight_format: {self.boxes_weight_format}")

        if loss in ("boxes",):
            return {"boxes_weight": iou}
        if loss in ("vfl",):
            return {"values": iou}
        return {}

    @staticmethod
    def get_cdn_matched_indices(
        dn_meta: dict[str, Any], targets: list[dict[str, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build CDN match indices from denoising metadata."""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device if targets else torch.device("cpu")

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i].to(device), gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )
        return dn_match_indices

    def unimodal_distribution_focal_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        weight_right: torch.Tensor,
        weight_left: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduction: str = "sum",
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """Unimodal distribution focal loss for FDR corner targets."""
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1) + F.cross_entropy(
            pred, dis_right, reduction="none"
        ) * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

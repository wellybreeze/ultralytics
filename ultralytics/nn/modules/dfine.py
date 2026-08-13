# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Portions Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# Modified from D-FINE: https://github.com/Peterande/D-FINE
# Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR) Copyright (c) 2023 lyuwenyu.
"""D-FINE decoder modules for Ultralytics (Fine-grained Distribution Refinement)."""

from __future__ import annotations

import copy
import functools
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from .dfine_encoder import get_activation
from .dfine_utils import distance2bbox, weighting_function
from .utils import bias_init_with_prob, inverse_sigmoid

__all__ = ("DFINEDecoder",)


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
    method: str = "default",
) -> torch.Tensor:
    """Multi-scale deformable attention core (official D-FINE v2 implementation).

    Args:
        value: List/tuple of per-level value tensors shaped [bs, n_head, c, h*w].
        value_spatial_shapes: List of (H, W) per level.
        sampling_locations: [bs, query_length, n_head, n_levels * n_points, 2].
        attention_weights: [bs, query_length, n_head, n_levels * n_points].
        num_points_list: Number of sampling points per level.
        method: ``default`` (bilinear grid_sample) or ``discrete``.

    Returns:
        (torch.Tensor): Attention output of shape [bs, query_length, C].
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"Unknown deformable attention method: {method}")

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        else:
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)
            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)


class MLP(nn.Module):
    """Official D-FINE MLP (activation between layers)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act: str = "relu"):
        """Initialize MLP layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear layers with intermediate activations."""
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention (official D-FINE implementation)."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int | list[int] = 4,
        method: str = "default",
        offset_scale: float = 0.5,
    ):
        """Initialize MS deformable attention projections and sampling offsets."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, "num_points list length must equal num_levels"
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)
        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        """Initialize sampling offset and attention weight parameters."""
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: list,
    ) -> torch.Tensor:
        """Apply multi-scale deformable attention.

        Args:
            query: [bs, query_length, C]
            reference_points: [bs, query_length, n_levels, 2/4]
            value: Preprocessed multi-scale values from TransformerDecoder.value_op
            value_spatial_shapes: [(H_i, W_i), ...]
        """
        bs, Len_q = query.shape[:2]
        sampling_offsets = self.sampling_offsets(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        return self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )


class Gate(nn.Module):
    """Gated residual fusion used after cross-attention."""

    def __init__(self, d_model: int):
        """Initialize gate linear layer and LayerNorm."""
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Fuse two streams with learned gates."""
        gates = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    """Static layer that computes integral results from a discrete distribution."""

    def __init__(self, reg_max: int = 32):
        """Initialize with number of discrete bins."""
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x: torch.Tensor, project: torch.Tensor) -> torch.Tensor:
        """Softmax over bins then project with non-uniform weights W(n)."""
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape([*list(shape[:-1]), -1])


class LQE(nn.Module):
    """Local Quality Estimator that adjusts classification scores from corner distributions."""

    def __init__(self, k: int, hidden_dim: int, num_layers: int, reg_max: int):
        """Initialize LQE MLP."""
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores: torch.Tensor, pred_corners: torch.Tensor) -> torch.Tensor:
        """Add quality scores derived from top-k corner probabilities."""
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stat.reshape(B, L, -1))


class TransformerDecoderLayer(nn.Module):
    """Single D-FINE decoder layer with self-attn, MSDeformAttn, Gate, and FFN."""

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        n_levels: int = 4,
        n_points: int | list[int] = 4,
        cross_attn_method: str = "default",
        layer_scale: float | None = None,
    ):
        """Initialize decoder layer submodules."""
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-init FFN weights."""
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional embedding when provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """Feed-forward network."""
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        target: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes,
        attn_mask: torch.Tensor | None = None,
        query_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run self-attn → cross-attn+Gate → FFN."""
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))

        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)
        target = self.gateway(target, self.dropout2(target2))

        target2 = self.forward_ffn(target)
        target = self.norm3((target + self.dropout4(target2)).clamp(min=-65504, max=65504))
        return target


class TransformerDecoder(nn.Module):
    """Transformer decoder implementing Fine-grained Distribution Refinement (FDR)."""

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: TransformerDecoderLayer,
        decoder_layer_wide: TransformerDecoderLayer,
        num_layers: int,
        num_head: int,
        reg_max: int,
        reg_scale: torch.Tensor,
        up: torch.Tensor,
        eval_idx: int = -1,
        layer_scale: float = 2,
    ):
        """Initialize stacked decoder / LQE layers with optional wider post-eval layers."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """Preprocess values for MSDeformableAttention."""
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        """Trim layers after eval_idx and freeze weighting projection for export (idempotent)."""
        if getattr(self, "_deployed", False):
            return
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])
        self._deployed = True

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
    ):
        """Run FDR iterative refinement across decoder layers."""
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes, dec_out_logits, dec_out_pred_corners, dec_out_refs = [], [], [], []
        project = self.project if hasattr(self, "project") else weighting_function(self.reg_max, up, reg_scale)
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class DFINEDecoder(nn.Module):
    """Ultralytics-compatible D-FINE transformer decoder head.

    Ports official ``DFINETransformer`` with Ultralytics constructor / forward conventions (similar to
    ``RTDETRDecoder``): YAML-injected ``ch``, training ``batch`` dict, and inference post-processing to ``(bs, nq, 6)``.
    """

    export = False
    max_det = 300
    shapes = []
    dynamic = False
    # anchors / valid_mask: optional buffers when eval_spatial_size is set (fixed-size cache);
    # otherwise assigned lazily in _get_decoder_input when spatial_shapes change (RTDETRDecoder parity).
    # Do not use class-level Tensor placeholders — they block register_buffer().

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (256, 256, 256),
        hd: int = 256,
        nq: int = 300,
        n_heads: int = 8,
        nl: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        reg_max: int = 32,
        reg_scale: float = 4.0,
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        num_points: list | tuple | int = (3, 6, 3),
        # Official kwargs with defaults
        activation: str = "relu",
        learn_query_content: bool = False,
        eval_spatial_size: tuple[int, int] | None = None,
        feat_strides: list[int] | tuple[int, ...] | None = None,
        num_levels: int | None = None,
        eps: float = 1e-2,
        aux_loss: bool = True,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        layer_scale: float = 1,
    ):
        """Initialize D-FINE decoder.

        Args:
            nc: Number of classes.
            ch: Multi-scale feature channels from the neck (injected by parse_model).
            hd: Hidden / embedding dimension.
            nq: Number of object queries.
            n_heads: Attention heads.
            nl: Number of decoder layers.
            dim_feedforward: FFN hidden size.
            dropout: Dropout probability.
            reg_max: FDR discrete bins.
            reg_scale: FDR scale parameter.
            eval_idx: Layer used at eval (negative indexes from the end).
            nd: Max denoising queries.
            label_noise_ratio: CDN label noise ratio.
            box_noise_scale: CDN box noise scale.
            num_points: Sampling points per level (list) or shared int.
            activation: FFN / MLP activation name.
            learn_query_content: Learn query embeddings instead of encoder top-k features.
            eval_spatial_size: Optional fixed (H, W) to precompute anchors; None rebuilds from feature shapes.
            feat_strides: Feature strides aligned with ``ch``.
            num_levels: Feature levels (defaults to ``len(ch)``).
            eps: Anchor validity epsilon.
            aux_loss: Whether to attach auxiliary outputs in training dict.
            cross_attn_method: ``default`` or ``discrete``.
            query_select_method: ``default``, ``one2many``, or ``agnostic``.
            layer_scale: Wider-layer scale after ``eval_idx``.
        """
        super().__init__()
        feat_channels = list(ch)
        num_levels = len(feat_channels) if num_levels is None else num_levels
        if feat_strides is None:
            feat_strides = [8, 16, 32][: len(feat_channels)]
            while len(feat_strides) < len(feat_channels):
                feat_strides.append(feat_strides[-1] * 2)
        feat_strides = list(feat_strides)
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        if isinstance(num_points, (list, tuple)):
            num_points = list(num_points)
            assert len(num_points) == num_levels, "num_points length must equal num_levels"

        self.hidden_dim = hd
        scaled_dim = round(layer_scale * hd)
        self.nhead = n_heads
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.nc = nc
        self.num_classes = nc
        self.num_queries = nq
        self.eps = eps
        self.num_layers = nl
        self.num_decoder_layers = nl
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        assert query_select_method in {"default", "one2many", "agnostic"}
        assert cross_attn_method in {"default", "discrete"}

        self._build_input_proj_layer(feat_channels)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        decoder_layer = TransformerDecoderLayer(
            hd,
            n_heads,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hd,
            n_heads,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        self.decoder = TransformerDecoder(
            hd,
            decoder_layer,
            decoder_layer_wide,
            nl,
            n_heads,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
        )

        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if nd > 0:
            # Official uses padding class for empty CDN slots (weight conversion parity).
            self.denoising_class_embed = nn.Embedding(nc + 1, hd, padding_idx=nc)
            init.normal_(self.denoising_class_embed.weight[:-1])
        else:
            self.denoising_class_embed = nn.Embedding(nc, hd)

        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, 2)

        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hd, hd)),
                    ("norm", nn.LayerNorm(hd)),
                ]
            )
        )
        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hd, 1)
        else:
            self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, 3)

        self.eval_idx = eval_idx if eval_idx >= 0 else nl + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hd, nc) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, nc) for _ in range(nl - self.eval_idx - 1)]
        )
        self.pre_bbox_head = MLP(hd, hd, 4, 3)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hd, hd, 4 * (self.reg_max + 1), 3) for _ in range(self.eval_idx + 1)]
            + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3) for _ in range(nl - self.eval_idx - 1)]
        )
        self.integral = Integral(self.reg_max)

        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        """Prepare score / bbox heads for deployment (keep only eval layer; idempotent)."""
        if getattr(self, "_deployed", False):
            return
        self.dec_score_head = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )
        self.decoder.convert_to_deploy()
        self._deployed = True

    def _reset_parameters(self, feat_channels: list[int]):
        """Initialize heads and projections like official DFINETransformer."""
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim and not isinstance(m, nn.Identity):
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels: list[int]):
        """Build 1x1 (or strided 3x3) projections into hidden_dim."""
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Project multi-scale features and flatten to encoder memory."""
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten, spatial_shapes = [], []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: list[list[int]] | None = None,
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate log-anchor boxes and validity mask for encoder query selection."""
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            # Match official DFINETransformer: int meshgrid then divide by float size tensor.
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype, device=device)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes: list[list[int]],
        denoising_logits: torch.Tensor | None = None,
        denoising_bbox_unact: torch.Tensor | None = None,
    ):
        """Select top-k encoder queries and optionally concatenate CDN queries."""
        # Rebuild when feature map sizes change (any imgsz), matching RTDETRDecoder.
        if self.dynamic or self.shapes != spatial_shapes:
            self.anchors, self.valid_mask = self._generate_anchors(
                spatial_shapes, dtype=memory.dtype, device=memory.device
            )
            self.shapes = spatial_shapes
        anchors, valid_mask = self.anchors, self.valid_mask

        if memory.shape[0] > 1 and anchors.shape[0] == 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        if self.training:
            enc_topk_bboxes_list.append(F.sigmoid(enc_topk_bbox_unact))
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ):
        """Select top-k encoder queries by the configured selection method."""
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        else:  # agnostic
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_anchors = outputs_anchors_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1])
        )
        topk_logits = (
            outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
            if self.training
            else None
        )
        topk_memory = memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        return topk_memory, topk_logits, topk_anchors

    def forward(self, x: list[torch.Tensor], batch: dict | None = None):
        """Forward pass.

        Args:
            x: Multi-scale feature maps.
            batch: Ultralytics batch dict for CDN training (``cls``, ``bboxes``, ``batch_idx``, ``gt_groups``).

        Returns:
            Training: dict with ``pred_logits``, ``pred_boxes``, ``pred_corners``, ``ref_points``, ``up``,
                ``reg_scale``, plus aux / dn fields when enabled.
            Inference: ``(bs, nq, 6)`` tensor if ``export`` else ``(y, raw_dict)``.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        memory, spatial_shapes = self._get_encoder_input(x)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_cdn_group(
                batch,
                self.nc,
                self.num_queries,
                self.denoising_class_embed.weight[:-1]
                if self.denoising_class_embed.num_embeddings == self.nc + 1
                else self.denoising_class_embed.weight,
                self.num_denoising,
                self.label_noise_ratio,
                self.box_noise_scale,
                self.training,
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        out = {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
            "pred_corners": out_corners[-1],
            "ref_points": out_refs[-1],
            "up": self.up,
            "reg_scale": self.reg_scale,
        }
        if self.training and dn_meta is None and self.num_denoising > 0:
            # Touch embedding so DDP sees it used when a batch has zero GTs.
            out["pred_logits"] = out["pred_logits"] + 0 * self.denoising_class_embed.weight.sum()

        # Always attach aux fields so trainer validation (eval-mode loss) works like RT-DETR.
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}
            if self.training and dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                # Ultralytics get_cdn_group uses dn_pos_idx; alias for official criterion name.
                if "dn_positive_idx" not in dn_meta and "dn_pos_idx" in dn_meta:
                    dn_meta = {**dn_meta, "dn_positive_idx": dn_meta["dn_pos_idx"]}
                out["dn_meta"] = dn_meta

        if self.training:
            return out

        y = self.postprocess(out["pred_boxes"], out["pred_logits"].sigmoid())
        return y if self.export else (y, out)

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Select top predictions as [cx, cy, w, h, score, cls].

        Args:
            boxes: (bs, nq, 4) normalized cxcywh.
            scores: (bs, nq, nc) sigmoid probabilities.
        """
        k = min(self.num_queries, self.max_det) if self.export else self.num_queries
        scores, index = scores.flatten(1).topk(k)
        query_idx = torch.div(index, self.nc, rounding_mode="floor")
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4).long())
        return torch.cat([boxes, scores[..., None], (index - query_idx * self.nc)[..., None].float()], dim=-1)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """Package encoder aux outputs for the criterion."""
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        """Package decoder / denoising aux outputs for the criterion."""
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        ]

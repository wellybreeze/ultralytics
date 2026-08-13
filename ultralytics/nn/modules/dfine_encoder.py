# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# ---------------------------------------------------------------------------------
# Modified from D-FINE (https://github.com/Peterande/D-FINE)
# Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
# Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
# Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""Official D-FINE hybrid-encoder blocks for numerical parity with Peterande/D-FINE.

These differ from Ultralytics YOLO ``RepNCSPELAN4`` / ``SCDown`` (which use RepCSP).
Do not replace the YOLO blocks — use ``DFINERepNCSPELAN4`` / ``DFINESCDown`` in D-FINE YAMLs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("CSPLayer", "ConvNormLayer", "ConvNormLayer_fuse", "DFINERepNCSPELAN4", "DFINESCDown", "VGGBlock")


def get_activation(act: str | nn.Module | None, inplace: bool = True) -> nn.Module:
    """Resolve activation name or module (official D-FINE ``utils.get_activation``)."""
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act in {"silu", "swish"}:
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unknown activation: {act}")
    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m


class ConvNormLayer_fuse(nn.Module):
    """Conv-BN-(Act) with optional BN fuse for deploy (official D-FINE)."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        """Initialize fused-capable Conv-BN-Act."""
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv-bn-act or fused conv."""
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        """Fuse Conv+BN into a single biased Conv2d (idempotent)."""
        if hasattr(self, "conv_bn_fused") and not hasattr(self, "conv"):
            return
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self):
        """Return fused kernel and bias."""
        return self._fuse_bn_tensor()

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    """Conv-BN-(Act) without fuse path (official D-FINE)."""

    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        """Initialize Conv-BN-Act."""
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv-bn-act."""
        return self.act(self.norm(self.conv(x)))


class DFINESCDown(nn.Module):
    """Official D-FINE SCDown (pointwise + depthwise) for HybridEncoder PAN."""

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown.

        Args:
            c1: Input channels.
            c2: Output channels.
            k: Depthwise kernel size.
            s: Depthwise stride.
        """
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample with separable conv."""
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    """3x3 + 1x1 residual-style block used inside official CSPLayer."""

    def __init__(self, ch_in, ch_out, act="relu"):
        """Initialize VGGBlock."""
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional fused conv after deploy conversion."""
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        """Fuse parallel 3x3/1x1 branches into one 3x3 conv (idempotent)."""
        if hasattr(self, "conv") and not hasattr(self, "conv1"):
            return
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        """Return fused 3x3 kernel and bias."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    """CSP layer with VGGBlock bottlenecks (official D-FINE, not Ultralytics RepCSP)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        """Initialize CSPLayer."""
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward CSP with additive branch merge."""
        x_1 = self.bottlenecks(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class DFINERepNCSPELAN4(nn.Module):
    """Official D-FINE RepNCSPELAN4 (CSPLayer+VGGBlock) for HybridEncoder FPN/PAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 3, bias: bool = False, act: str = "silu"):
        """Initialize CSP-ELAN.

        Args:
            c1: Input channels (typically ``hidden * 2`` after Concat).
            c2: Output channels (``hidden``).
            c3: Intermediate channels (``hidden * 2``).
            c4: Bottleneck channels (``round(expansion * hidden // 2)``).
            n: Number of VGGBlocks inside each CSPLayer (``round(3 * depth_mult)``).
            bias: Conv bias flag.
            act: Activation name.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using ``chunk`` (training / alternate path)."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using ``split`` (official default)."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

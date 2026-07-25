# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""WeDetect dual-tower ONNX export helpers (aligned with WeDetect/deploy/export_onnx.py).

Layouts
-------
- ``dual`` (default): separate ``*_vision.onnx`` and ``*_language.onnx``
- ``whole``: single ``*_whole.onnx`` with both towers

Tokenization stays in Python (HuggingFace tokenizer). ``num_classes`` / ``seq_len``
are dynamic axes so exported models support custom open-vocabulary prompts.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.tal import make_anchors


class WeDetectVisionExport(nn.Module):
    """Vision tower + head + DFL decode for dynamic open-vocabulary export.

    forward(image, txt_feats) -> (bboxes, scores)
        image     : float32 [B, 3, H, W]
        txt_feats : float32 [B, K, embed]
        bboxes    : float32 [B, N, 4]  decoded xyxy in letterbox space
        scores    : float32 [B, N, K]  per-class sigmoid scores
    """

    def __init__(self, model, imgsz: int | tuple[int, int] = 640):
        """Build vision export wrapper from a ``WeDetectModel``."""
        super().__init__()
        from ultralytics.nn.modules.head import WeDetectDetect

        self.backbone = deepcopy(model.model[0])
        self.neck = deepcopy(model.model[1])
        head = deepcopy(model.model[2])
        assert isinstance(head, WeDetectDetect), f"Expected WeDetectDetect head, got {type(head)}"
        self.head = head
        self.reg_max = head.reg_max
        self.nl = head.nl
        self.stride = head.stride.clone() if isinstance(head.stride, torch.Tensor) else torch.tensor(head.stride)

        h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        # make_anchors returns grid-cell points (N,2) and strides (N,1)
        feats = [torch.zeros(1, 1, h // int(s), w // int(s)) for s in self.stride]
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        self.register_buffer("anchors", anchors.contiguous(), persistent=False)  # N,2
        self.register_buffer("strides", strides.contiguous(), persistent=False)  # N,1
        self.register_buffer(
            "proj",
            torch.arange(self.reg_max, dtype=torch.float).view(1, 1, 1, self.reg_max),
            persistent=False,
        )

    def forward(self, image: torch.Tensor, txt_feats: torch.Tensor):
        """Run vision tower and return decoded boxes + class scores."""
        if txt_feats.ndim == 2:
            txt_feats = txt_feats.unsqueeze(0)
        if txt_feats.shape[0] != image.shape[0]:
            txt_feats = txt_feats.expand(image.shape[0], -1, -1)

        feats = self.neck(self.backbone(image))
        cls_list, box_list = [], []
        for i in range(self.nl):
            feat = feats[i]
            b, _, fh, fw = feat.shape
            cls_embed = self.head.cv3[i](feat)
            cls_logit = self.head.cv4[i](cls_embed, txt_feats)  # B,K,H,W
            reg = self.head.cv2[i](feat)  # B,4*reg_max,H,W
            reg = reg.view(b, 4, self.reg_max, fh * fw).permute(0, 3, 1, 2)  # B,HW,4,reg_max
            reg = (reg.softmax(-1) * self.proj).sum(-1)  # B,HW,4
            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(b, fh * fw, -1)  # B,HW,K
            cls_list.append(cls_logit)
            box_list.append(reg)

        scores = torch.cat(cls_list, dim=1).sigmoid()  # B,N,K
        dist = torch.cat(box_list, dim=1)  # B,N,4  (l,t,r,b) in grid units
        # Grid-unit DFL + anchors -> pixel xyxy (ultralytics make_anchors convention)
        x1y1 = (self.anchors - dist[..., :2]) * self.strides
        x2y2 = (self.anchors + dist[..., 2:]) * self.strides
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        return bboxes, scores


class WeDetectLanguageExport(nn.Module):
    """Language tower export wrapper.

    forward(input_ids, attention_mask) -> txt_feats [1, K, embed]
    """

    def __init__(self, text_encoder: nn.Module):
        """Wrap an ``XLMRoberta`` text encoder for ONNX export."""
        super().__init__()
        self.model = text_encoder.model
        self.head = text_encoder.head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Encode tokenized class names into L2-normalized text features."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        feat = outputs.last_hidden_state[:, 0]
        feat = self.head(feat)
        feat = F.normalize(feat, dim=-1, p=2)
        return feat.unsqueeze(0)  # 1,K,D


class WeDetectWholeExport(nn.Module):
    """Fused language + vision graph for whole-model export."""

    def __init__(self, vision: WeDetectVisionExport, language: WeDetectLanguageExport):
        super().__init__()
        self.language = language
        self.vision = vision

    def forward(self, image, input_ids, attention_mask):
        """Encode texts then run vision tower."""
        txt_feats = self.language(input_ids, attention_mask)
        return self.vision(image, txt_feats)


def _prepare_text_encoder(model):
    """Ensure WeDetectModel has a loaded text encoder with synced weights."""
    from copy import deepcopy

    m = deepcopy(model).eval()
    if hasattr(m, "sync_text_model_weights"):
        m.sync_text_model_weights()
    if not isinstance(getattr(m, "text_model", None), nn.Module):
        if hasattr(m, "ensure_text_model"):
            m.ensure_text_model()
        enc = getattr(m, "text_model", None) or getattr(m, "clip_model", None)
        if enc is None:
            raise RuntimeError("WeDetect text encoder is missing; cannot export language tower.")
        if not isinstance(getattr(m, "text_model", None), nn.Module) and hasattr(m, "register_text_model"):
            # Prefer submodule registration so state is self-contained
            try:
                m.register_text_model()
            except Exception:
                pass
        enc = getattr(m, "text_model", None) or getattr(m, "clip_model", None)
    else:
        enc = m.text_model
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return m, enc


def _dummy_tokens(text_encoder, num_classes: int = 8):
    """Build dummy tokenizer inputs for tracing the language tower."""
    sample = ["人", "车", "狗", "猫", "椅子", "桌子", "瓶子", "杯子"][: max(2, num_classes)]
    encoded = text_encoder.tokenizer(text=sample, return_tensors="pt", padding=True, truncation=True, max_length=77)
    return encoded["input_ids"], encoded["attention_mask"]


def export_wedetect_onnx(
    model,
    file: Path | str,
    imgsz: int | tuple[int, int] = 640,
    export_mode: str = "dual",
    opset: int = 17,
    simplify: bool = False,
    device: str | torch.device = "cpu",
    prefix: str = "",
) -> list[str]:
    """Export WeDetect to dual or whole ONNX graphs.

    Args:
        model: ``WeDetectModel`` instance.
        file: Source model path (stem used for output names).
        imgsz: Square size or (h, w) for the vision tower.
        export_mode: ``dual`` or ``whole``.
        opset: ONNX opset version.
        simplify: Whether to slim graphs with onnxslim when available.
        device: Export device.
        prefix: Log prefix.

    Returns:
        (list[str]): Paths of exported ONNX files (vision first for dual).
    """
    from ultralytics.utils.export.engine import torch2onnx

    prefix = prefix or colorstr("WeDetect ONNX:")
    export_mode = (export_mode or "dual").lower()
    assert export_mode in {"dual", "whole"}, f"export_mode must be 'dual' or 'whole', got '{export_mode}'"

    file = Path(file)
    device = torch.device(device)
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    embed = int(getattr(model.model[-1].cv3[0][-1], "out_channels", 768))

    prepared, text_encoder = _prepare_text_encoder(model)
    prepared = prepared.to(device)
    text_encoder = text_encoder.to(device)

    vision = WeDetectVisionExport(prepared, imgsz=(h, w)).to(device).eval()
    language = WeDetectLanguageExport(text_encoder).to(device).eval()

    dummy_img = torch.zeros(1, 3, h, w, device=device)
    dummy_txt = torch.zeros(1, 8, embed, device=device)
    input_ids, attention_mask = _dummy_tokens(text_encoder, num_classes=8)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    exported: list[str] = []
    if export_mode == "whole":
        out = str(file.with_name(f"{file.stem}_whole.onnx"))
        whole = WeDetectWholeExport(vision, language).eval()
        LOGGER.info(f"{prefix} exporting whole model -> {out}")
        torch2onnx(
            whole,
            (dummy_img, input_ids, attention_mask),
            out,
            opset=opset,
            input_names=["image", "input_ids", "attention_mask"],
            output_names=["bboxes", "scores"],
            dynamic={
                "input_ids": {0: "num_classes", 1: "seq_len"},
                "attention_mask": {0: "num_classes", 1: "seq_len"},
                "scores": {2: "num_classes"},
            },
        )
        exported.append(out)
    else:
        v_out = str(file.with_name(f"{file.stem}_vision.onnx"))
        l_out = str(file.with_name(f"{file.stem}_language.onnx"))
        LOGGER.info(f"{prefix} exporting vision tower -> {v_out}")
        torch2onnx(
            vision,
            (dummy_img, dummy_txt),
            v_out,
            opset=opset,
            input_names=["image", "txt_feats"],
            output_names=["bboxes", "scores"],
            dynamic={
                "txt_feats": {1: "num_classes"},
                "scores": {2: "num_classes"},
            },
        )
        LOGGER.info(f"{prefix} exporting language tower -> {l_out}")
        torch2onnx(
            language,
            (input_ids, attention_mask),
            l_out,
            opset=opset,
            input_names=["input_ids", "attention_mask"],
            output_names=["txt_feats"],
            dynamic={
                "input_ids": {0: "num_classes", 1: "seq_len"},
                "attention_mask": {0: "num_classes", 1: "seq_len"},
                "txt_feats": {1: "num_classes"},
            },
        )
        exported.extend([v_out, l_out])

    # Metadata + optional slim
    try:
        import onnx
    except ImportError:
        return exported

    meta = {
        "task": "detect",
        "export_mode": export_mode,
        "imgsz": f"{h},{w}",
        "embed_dim": str(embed),
        "text_model": getattr(model, "text_model_variant", "xlm-roberta:base"),
        "stride": ",".join(str(int(s)) for s in vision.stride.tolist()),
    }
    for path in exported:
        model_onnx = onnx.load(path)
        if simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming {path} with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)
            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")
        for k, v in meta.items():
            prop = model_onnx.metadata_props.add()
            prop.key, prop.value = k, str(v)
        if getattr(model_onnx, "ir_version", 0) > 10:
            model_onnx.ir_version = 10
        onnx.save(model_onnx, path)

    return exported

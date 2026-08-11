# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""WeDetect dual-tower ONNX / TensorRT export helpers (aligned with WeDetect/deploy/export_onnx.py).

Layouts
-------
- ``dual`` (default): separate ``*_vision.onnx`` and ``*_language.onnx``
- ``whole``: single ``*_whole.onnx`` with both towers

Tokenization stays in Python (HuggingFace tokenizer). ``num_classes`` / ``seq_len``
are dynamic axes so exported models support custom open-vocabulary prompts.

TensorRT (dual only) builds ``*_vision.engine`` + ``*_language.engine`` with per-input
optimization profiles (image / txt_feats / tokens) — do not use generic ``onnx2engine``
dynamic profiles, which assume every input is an image tensor.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER, colorstr, is_dgx, is_jetson
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version
from ultralytics.utils.export.engine import apply_builder_optimization_level
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


class WeDetectVisionNMSExport(nn.Module):
    """Vision tower + TorchScript-friendly class-aware NMS → ``(B, max_det, 6)``.

    Uses ``torchvision.ops.batched_nms`` (faster than Python gather of ONNX indices).
    Trace with ``txt_feats`` padded to ``max_classes`` so the class axis is stable while
    open-vocabulary prompts remain swappable via the separate language tower.
    """

    def __init__(
        self,
        vision: WeDetectVisionExport,
        *,
        max_det: int = 300,
        conf: float = 0.25,
        iou: float = 0.45,
        max_classes: int = 80,
    ):
        super().__init__()
        self.vision = vision
        self.max_det = int(max_det)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_classes = int(max_classes)

    def forward(self, image: torch.Tensor, txt_feats: torch.Tensor) -> torch.Tensor:
        """Run vision decode + NMS; return xyxy/conf/cls padded to ``max_det``."""
        from torchvision.ops import batched_nms

        bboxes, scores = self.vision(image, txt_feats)  # B,N,4 xyxy ; B,N,K
        # Pad/truncate class axis to the traced max_classes ceiling
        k = scores.shape[-1]
        if k < self.max_classes:
            scores = F.pad(scores, (0, self.max_classes - k))
        elif k > self.max_classes:
            scores = scores[..., : self.max_classes]
        b, n, nc = scores.shape
        out = image.new_zeros(b, self.max_det, 6)
        for i in range(b):
            boxes = bboxes[i]
            sc = scores[i]
            boxes_exp = boxes.unsqueeze(1).expand(n, nc, 4).reshape(-1, 4)
            scores_flat = sc.reshape(-1)
            # cumsum on sc-backed ones keeps class ids on the runtime device (arange would bake
            # device("cpu") into the TorchScript graph when exported on CPU).
            cls_ids = sc.new_ones(n, nc).cumsum(1).long().reshape(-1) - 1
            scores_nms = scores_flat.clone()
            scores_nms[scores_flat <= self.conf] = 0
            keep = batched_nms(boxes_exp, scores_nms, cls_ids, self.iou)
            keep = keep[scores_nms[keep] > 0][: self.max_det]
            num = keep.numel()
            if num > 0:
                out[i, :num] = torch.cat(
                    [boxes_exp[keep], scores_flat[keep].unsqueeze(1), cls_ids[keep].unsqueeze(1).to(out.dtype)], 1
                )
        return out


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
    *,
    nms: bool = False,
    max_det: int = 300,
    conf: float = 0.25,
    iou: float = 0.45,
    max_classes: int = 80,
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
        nms: If True, append ONNX-native ``NonMaxSuppression`` (ORT-runnable) to the
            vision/whole graph. Distinct from TensorRT ``EfficientNMS_TRT``.
        max_det / conf / iou: Native NMS parameters when ``nms=True``.
        max_classes: Stored in metadata (prompt ceiling for DualBackend padding).

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
        "stride": str(int(max(int(s) for s in vision.stride.tolist()))),
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

    if nms:
        # ORT-native NMS on vision (dual) or whole graph — not EfficientNMS_TRT
        target = exported[0]
        append_wedetect_onnx_nms(
            target,
            output_file=target,
            max_det=max_det,
            conf=conf,
            iou=iou,
            max_classes=max_classes,
            prefix=colorstr("WeDetect ONNX-NMS:"),
        )

    return exported


def append_wedetect_onnx_nms(
    onnx_file: str | Path,
    output_file: str | Path | None = None,
    *,
    max_det: int = 300,
    conf: float = 0.25,
    iou: float = 0.45,
    max_classes: int = 80,
    prefix: str = "",
) -> str:
    """Append ONNX-native ``NonMaxSuppression`` to a WeDetect vision/whole graph.

    Runnable in ONNX Runtime (unlike ``EfficientNMS_TRT``). Keeps decoded ``bboxes``
    (xyxy) and ``scores`` as outputs and adds ``nms_indices`` ``[M,3]`` =
    ``[batch, class, box]`` for DualBackend gather packing.

    Args:
        onnx_file: Path to ``*_vision.onnx`` or ``*_whole.onnx``.
        output_file: Destination path (default: overwrite ``onnx_file``).
        max_det: ``max_output_boxes_per_class`` for the NMS node (then packed to
            ``max_det`` total detections in DualBackend).
        conf / iou: Score / IoU thresholds.
        max_classes: Written to metadata for DualBackend.
        prefix: Log prefix.

    Returns:
        (str): Path to the written ONNX file.
    """
    check_requirements("onnx_graphsurgeon>=0.3.26")
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs

    prefix = prefix or colorstr("WeDetect ONNX-NMS:")
    onnx_file = Path(onnx_file)
    output_file = Path(output_file) if output_file is not None else onnx_file

    LOGGER.info(f"{prefix} appending NonMaxSuppression to {onnx_file} -> {output_file}")
    graph = gs.import_onnx(onnx.load(str(onnx_file)))
    outs = list(graph.outputs)
    if len(outs) < 2:
        raise ValueError(f"WeDetect ONNX must have bboxes+scores outputs, got {len(outs)}")
    by_name = {o.name: o for o in outs}
    boxes = by_name.get("bboxes", outs[0])  # [B,N,4] xyxy
    scores = by_name.get("scores", outs[1])  # [B,N,K]

    # scores [B,N,K] -> [B,K,N] for ONNX NonMaxSuppression
    scores_bkn = gs.Variable(name="scores_bkn", dtype=np.float32)
    graph.layer(op="Transpose", name="scores_BNK_to_BKN", inputs=[scores], outputs=[scores_bkn], attrs={"perm": [0, 2, 1]})

    # boxes xyxy -> yxyx (center_point_box=0 TensorFlow corner format)
    split_sizes = gs.Constant("nms_box_split", np.array([1, 1, 1, 1], dtype=np.int64))
    x1 = gs.Variable("nms_x1", dtype=np.float32)
    y1 = gs.Variable("nms_y1", dtype=np.float32)
    x2 = gs.Variable("nms_x2", dtype=np.float32)
    y2 = gs.Variable("nms_y2", dtype=np.float32)
    graph.layer(
        op="Split",
        name="split_xyxy",
        inputs=[boxes, split_sizes],
        outputs=[x1, y1, x2, y2],
        attrs={"axis": 2},
    )
    boxes_yxyx = gs.Variable(name="boxes_yxyx", dtype=np.float32)
    graph.layer(op="Concat", name="concat_yxyx", inputs=[y1, x1, y2, x2], outputs=[boxes_yxyx], attrs={"axis": 2})

    max_out = gs.Constant("max_output_boxes_per_class", np.array([int(max_det)], dtype=np.int64))
    iou_t = gs.Constant("iou_threshold", np.array([float(iou)], dtype=np.float32))
    score_t = gs.Constant("score_threshold", np.array([float(conf)], dtype=np.float32))
    indices = gs.Variable(name="nms_indices", dtype=np.int64, shape=["num_selected", 3])
    graph.layer(
        op="NonMaxSuppression",
        name="NonMaxSuppression",
        inputs=[boxes_yxyx, scores_bkn, max_out, iou_t, score_t],
        outputs=[indices],
        attrs={"center_point_box": 0},
    )
    # Keep original xyxy boxes + scores for gather; indices drive DualBackend packing
    graph.outputs = [boxes, scores, indices]
    graph.cleanup().toposort()

    model_onnx = gs.export_onnx(graph)
    src = onnx.load(str(onnx_file))
    meta = {p.key: p.value for p in src.metadata_props}
    meta.update(
        {
            "end2end": "True",
            "nms": "True",
            "nms_format": "onnx_indices",
            "max_det": str(int(max_det)),
            "max_classes": str(int(max_classes)),
            "conf": str(float(conf)),
            "iou": str(float(iou)),
        }
    )
    del model_onnx.metadata_props[:]
    for k, v in meta.items():
        prop = model_onnx.metadata_props.add()
        prop.key, prop.value = k, str(v)
    if getattr(model_onnx, "ir_version", 0) > 10:
        model_onnx.ir_version = 10
    onnx.save(model_onnx, str(output_file))
    LOGGER.info(f"{prefix} saved {output_file}")
    return str(output_file)


def append_wedetect_efficient_nms(
    onnx_file: str | Path,
    output_file: str | Path | None = None,
    *,
    max_det: int = 300,
    conf: float = 0.25,
    iou: float = 0.45,
    max_classes: int = 80,
    prefix: str = "",
) -> str:
    """Append TensorRT ``EfficientNMS_TRT`` to a WeDetect dual vision ONNX graph.

    Vision export already yields ``bboxes`` (xyxy) and ``scores``; this only adds the
    NMS plugin (no YOLO Concat rewrite). The resulting ONNX is for TensorRT build only
    and is not runnable in standard ONNX Runtime.

    Args:
        onnx_file: Path to ``*_vision.onnx``.
        output_file: Destination ONNX path (default: ``*_vision_effnms.onnx``).
        max_det: ``max_output_boxes`` for the plugin.
        conf / iou: Plugin score / IoU thresholds.
        max_classes: Stored in metadata as the fixed class-count ceiling.
        prefix: Log prefix.

    Returns:
        (str): Path to the written ONNX file.
    """
    from collections import OrderedDict

    check_requirements("onnx_graphsurgeon>=0.3.26")
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs

    prefix = prefix or colorstr("WeDetect EfficientNMS:")
    onnx_file = Path(onnx_file)
    if output_file is None:
        output_file = onnx_file.with_name(f"{onnx_file.stem}_effnms.onnx")
    output_file = Path(output_file)

    LOGGER.info(f"{prefix} appending EfficientNMS_TRT to {onnx_file} -> {output_file}")
    graph = gs.import_onnx(onnx.load(str(onnx_file)))
    outs = list(graph.outputs)
    if len(outs) < 2:
        raise ValueError(f"WeDetect vision ONNX must have bboxes+scores outputs, got {len(outs)}")

    # Prefer named outputs; fall back to order from export_wedetect_onnx
    by_name = {o.name: o for o in outs}
    boxes = by_name.get("bboxes", outs[0])
    scores = by_name.get("scores", outs[1])

    op_outputs = [
        gs.Variable(name="num_dets", dtype=np.int32, shape=[1, 1]),
        gs.Variable(name="det_boxes", dtype=np.float32, shape=[1, max_det, 4]),
        gs.Variable(name="det_scores", dtype=np.float32, shape=[1, max_det]),
        gs.Variable(name="det_classes", dtype=np.int32, shape=[1, max_det]),
    ]
    # box_coding=False: WeDetect vision emits corner xyxy (trtyolo YOLO path uses center=True)
    attrs = OrderedDict(
        plugin_version="1",
        background_class=-1,
        max_output_boxes=int(max_det),
        score_threshold=float(conf),
        iou_threshold=float(iou),
        score_activation=False,
        class_agnostic=False,
        box_coding=False,
    )
    graph.layer(
        op="EfficientNMS_TRT",
        name="EfficientNMS_TRT",
        inputs=[boxes, scores],
        outputs=op_outputs,
        attrs=attrs,
    )
    graph.outputs = op_outputs
    graph.cleanup().toposort()

    model_onnx = gs.export_onnx(graph)
    # Preserve / extend metadata from the source vision ONNX
    src = onnx.load(str(onnx_file))
    meta = {p.key: p.value for p in src.metadata_props}
    meta.update(
        {
            "end2end": "True",
            "nms": "True",
            "nms_format": "efficientnms",
            "max_det": str(int(max_det)),
            "max_classes": str(int(max_classes)),
            "conf": str(float(conf)),
            "iou": str(float(iou)),
        }
    )
    del model_onnx.metadata_props[:]
    for k, v in meta.items():
        prop = model_onnx.metadata_props.add()
        prop.key, prop.value = k, str(v)
    if getattr(model_onnx, "ir_version", 0) > 10:
        model_onnx.ir_version = 10
    onnx.save(model_onnx, str(output_file))
    LOGGER.info(f"{prefix} saved {output_file}")
    return str(output_file)


def _inject_native_nms(
    network,
    trt,
    *,
    max_det: int,
    conf: float,
    iou: float,
    prefix: str = "",
) -> None:
    """Replace vision graph outputs with boxes/scores + TensorRT ``INMSLayer`` indices.

    TensorRT 11 removed ``EfficientNMS_TRT``; native ``add_nms`` is the replacement.
    Outputs: ``bboxes``, ``scores``, ``nms_indices`` ``[M,3]``, ``num_detections``.
    """
    import numpy as np

    outs = {network.get_output(i).name: network.get_output(i) for i in range(network.num_outputs)}
    if "bboxes" not in outs or "scores" not in outs:
        raise RuntimeError(f"{prefix} native NMS requires named outputs 'bboxes' and 'scores', got {list(outs)}")
    boxes, scores = outs["bboxes"], outs["scores"]
    for i in range(network.num_outputs - 1, -1, -1):
        network.unmark_output(network.get_output(i))

    max_boxes = network.add_constant((), np.array(int(max_det), dtype=np.int32)).get_output(0)
    iou_t = network.add_constant((), np.array(float(iou), dtype=np.float32)).get_output(0)
    conf_t = network.add_constant((), np.array(float(conf), dtype=np.float32)).get_output(0)
    nms = network.add_nms(boxes, scores, max_boxes)
    if nms is None:
        raise RuntimeError(f"{prefix} TensorRT add_nms failed")
    nms.name = "WeDetectNMS"
    nms.bounding_box_format = trt.BoundingBoxFormat.CORNER_PAIRS
    nms.topk_box_limit = max(int(max_det), 2000)
    nms.set_input(3, iou_t)
    nms.set_input(4, conf_t)
    indices = nms.get_output(0)
    indices.name = "nms_indices"
    num_dets = nms.get_output(1)
    num_dets.name = "num_detections"
    boxes.name = "bboxes"
    scores.name = "scores"
    network.mark_output(boxes)
    network.mark_output(scores)
    network.mark_output(indices)
    network.mark_output(num_dets)
    LOGGER.info(f"{prefix} injected native INMSLayer (max_det={max_det}, conf={conf}, iou={iou})")


def wedetect_onnx2engine(
    onnx_file: str | Path,
    output_file: str | Path | None = None,
    *,
    tower: str,
    imgsz: tuple[int, int] = (640, 640),
    embed_dim: int = 768,
    max_classes: int = 80,
    opt_classes: int = 8,
    max_seq_len: int = 77,
    opt_seq_len: int = 16,
    fixed_classes: bool = False,
    native_nms: bool = False,
    max_det: int = 300,
    conf: float = 0.25,
    iou: float = 0.45,
    workspace: int | None = None,
    quantize: int | str | None = None,
    builder_optimization_level: int | None = None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    """Convert one WeDetect dual ONNX tower to TensorRT with correct dynamic profiles.

    Args:
        onnx_file: Path to ``*_vision.onnx`` or ``*_language.onnx``.
        output_file: Destination ``.engine`` path.
        tower: ``vision`` or ``language``.
        imgsz: Vision image size ``(h, w)``.
        embed_dim: Text embedding dimension.
        max_classes / opt_classes: Dynamic ``num_classes`` profile bounds.
        max_seq_len / opt_seq_len: Dynamic token length profile bounds (language; min length is 1).
        fixed_classes: If True (NMS vision), lock ``txt_feats`` to ``max_classes``.
        native_nms: If True, inject TensorRT ``INMSLayer`` after parsing (TRT 11+ path).
        max_det / conf / iou: Native NMS parameters when ``native_nms=True``.
        workspace: TensorRT workspace size in GiB.
        quantize: ``16`` for FP16, ``None``/``32`` for FP32. INT8 is not supported.
        builder_optimization_level: TensorRT builder optimization level ``[0, 5]``; ``None`` = default.
        metadata: Optional metadata dict embedded in the engine file.
        verbose: Verbose TensorRT builder logs.
        prefix: Log prefix.

    Returns:
        (str): Path to the written engine file.
    """
    if quantize == 8:
        raise ValueError("WeDetect dual TensorRT INT8 export is not supported; use quantize=16 (FP16) or omit for FP32.")

    if is_jetson(jetpack=7) or is_dgx():
        check_tensorrt("10.15")
    try:
        import tensorrt as trt
    except ImportError:
        check_tensorrt()
        import tensorrt as trt
    check_version(trt.__version__, ">=7.0.0", hard=True)
    check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

    prefix = prefix or colorstr("WeDetect TensorRT:")
    onnx_file = Path(onnx_file)
    output_file = Path(output_file or onnx_file.with_suffix(".engine"))
    tower = tower.lower()
    assert tower in {"vision", "language"}, f"tower must be 'vision' or 'language', got '{tower}'"
    if native_nms and tower != "vision":
        raise ValueError("native_nms is only supported for the vision tower")

    LOGGER.info(f"\n{prefix} building {tower} engine from {onnx_file} -> {output_file}")
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    trt_major = int(trt.__version__.split(".", 1)[0])
    is_trt10 = trt_major >= 10
    is_trt11 = trt_major >= 11
    if workspace_bytes > 0:
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config.max_workspace_size = workspace_bytes
    apply_builder_optimization_level(config, builder_optimization_level, prefix=prefix)

    flag = 0 if is_trt10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    use_fp16 = getattr(builder, "platform_has_fast_fp16", True) and quantize == 16
    if is_trt11 and use_fp16:
        # Dual towers have non-image inputs; ModelOpt Autocast assumes a single image-shaped input.
        # Build FP32 on TRT11; use TensorRT 10.x with BuilderFlag.FP16 for dual FP16 engines.
        LOGGER.warning(
            f"{prefix} TensorRT {trt.__version__} is strongly-typed; WeDetect dual FP16 via ModelOpt "
            f"is not supported. Building FP32 engine instead (use TensorRT 10.x for FP16)."
        )
        use_fp16 = False

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    if native_nms:
        _inject_native_nms(network, trt, max_det=max_det, conf=conf, iou=iou, prefix=prefix)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    h, w = imgsz
    k_min, k_opt, k_max = 1, max(1, int(opt_classes)), max(int(opt_classes), int(max_classes))
    if fixed_classes:
        k_min = k_opt = k_max = max(1, int(max_classes))
    # Token length min must be 1: short class names (e.g. "人") tokenize to <8 tokens.
    s_min, s_opt, s_max = 1, max(1, int(opt_seq_len)), max(int(opt_seq_len), int(max_seq_len))
    profile = builder.create_optimization_profile()
    for inp in inputs:
        name = inp.name
        if tower == "vision":
            if name == "image":
                shape = (1, 3, h, w)
                profile.set_shape(name, min=shape, opt=shape, max=shape)
            elif name == "txt_feats":
                profile.set_shape(
                    name,
                    min=(1, k_min, embed_dim),
                    opt=(1, k_opt, embed_dim),
                    max=(1, k_max, embed_dim),
                )
            else:
                raise ValueError(f"Unexpected vision ONNX input '{name}'")
        else:  # language
            if name in {"input_ids", "attention_mask"}:
                profile.set_shape(name, min=(k_min, s_min), opt=(k_opt, s_opt), max=(k_max, s_max))
            else:
                raise ValueError(f"Unexpected language ONNX input '{name}'")
    config.add_optimization_profile(profile)

    if use_fp16 and not is_trt11:
        config.set_flag(trt.BuilderFlag.FP16)
        LOGGER.info(f"{prefix} FP16 enabled")

    if hasattr(builder, "build_serialized_network"):
        engine = builder.build_serialized_network(network, config)
    else:
        built = builder.build_engine(network, config)
        engine = None if built is None else built.serialize()
    if engine is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_file}")

    meta = dict(metadata or {})
    meta.setdefault("export_mode", "dual")
    meta.setdefault("imgsz", f"{h},{w}")
    meta.setdefault("embed_dim", str(embed_dim))
    meta.setdefault("task", "detect")
    meta.setdefault("tower", tower)
    meta.setdefault("max_classes", str(int(max_classes)))
    with open(output_file, "wb") as t:
        meta_json = json.dumps(meta)
        t.write(len(meta_json).to_bytes(4, byteorder="little", signed=True))
        t.write(meta_json.encode())
        t.write(engine)
    LOGGER.info(f"{prefix} saved {output_file}")
    return str(output_file)


def export_wedetect_engine(
    model,
    file: Path | str,
    imgsz: int | tuple[int, int] = 640,
    *,
    opset: int = 17,
    simplify: bool = False,
    device: str | torch.device = "cuda:0",
    workspace: int | None = None,
    quantize: int | str | None = None,
    builder_optimization_level: int | None = None,
    max_classes: int = 80,
    nms: bool = False,
    max_det: int = 300,
    conf: float = 0.25,
    iou: float = 0.45,
    verbose: bool = False,
    prefix: str = "",
) -> list[str]:
    """Export WeDetect dual ONNX then convert both towers to TensorRT engines.

    Args:
        model: ``WeDetectModel`` instance.
        file: Source model path (stem used for output names).
        imgsz: Square size or ``(h, w)``.
        opset / simplify / device: Passed through to ONNX export.
        workspace: TensorRT workspace GiB.
        quantize: ``16`` for FP16; INT8 unsupported.
        builder_optimization_level: TensorRT builder optimization level ``[0, 5]``; ``None`` = default.
        max_classes: Max class-count for TRT profiles / EfficientNMS padding ceiling.
        nms: If True, append ``EfficientNMS_TRT`` to the vision tower before build.
        max_det / conf / iou: EfficientNMS plugin parameters when ``nms=True``.
        verbose: Verbose builder logs.
        prefix: Log prefix.

    Returns:
        (list[str]): ``[vision.engine, language.engine]``.
    """
    prefix = prefix or colorstr("WeDetect TensorRT:")
    export_mode = "dual"
    onnx_paths = export_wedetect_onnx(
        model,
        file,
        imgsz=imgsz,
        export_mode=export_mode,
        opset=opset,
        simplify=simplify,
        device=device,
        prefix=colorstr("WeDetect ONNX:"),
    )
    if len(onnx_paths) < 2:
        raise RuntimeError("WeDetect TensorRT export requires dual ONNX (vision + language)")

    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    embed = int(getattr(model.model[-1].cv3[0][-1], "out_channels", 768))

    # TRT 11 removed EfficientNMS_TRT; use native INMSLayer. TRT ≤10 keeps the ONNX plugin path.
    try:
        import tensorrt as trt

        trt_major = int(trt.__version__.split(".", 1)[0])
    except Exception:
        trt_major = 0
    use_plugin_nms = bool(nms) and trt_major < 11
    use_native_nms = bool(nms) and trt_major >= 11
    if nms and trt_major >= 11:
        LOGGER.info(
            f"{prefix} TensorRT {trt_major}.x has no EfficientNMS_TRT plugin; "
            f"building vision engine with native INMSLayer instead."
        )

    meta = {
        "task": "detect",
        "export_mode": "dual",
        "imgsz": f"{h},{w}",
        "embed_dim": str(embed),
        "text_model": getattr(model, "text_model_variant", "xlm-roberta:base"),
        "max_classes": str(int(max_classes)),
        "end2end": str(bool(nms)),
        "nms": str(bool(nms)),
    }
    if nms:
        meta.update(
            {
                "max_det": str(int(max_det)),
                "conf": str(float(conf)),
                "iou": str(float(iou)),
                "nms_format": "efficientnms" if use_plugin_nms else "indices",
            }
        )

    vision_onnx, language_onnx = Path(onnx_paths[0]), Path(onnx_paths[1])
    if use_plugin_nms:
        vision_onnx = Path(
            append_wedetect_efficient_nms(
                vision_onnx,
                max_det=max_det,
                conf=conf,
                iou=iou,
                max_classes=max_classes,
                prefix=colorstr("WeDetect EfficientNMS:"),
            )
        )

    engines = []
    for onnx_path, tower in ((vision_onnx, "vision"), (language_onnx, "language")):
        # Always write sibling engines next to the original dual ONNX stem (not *_effnms)
        stem_base = Path(onnx_paths[0] if tower == "vision" else onnx_paths[1])
        eng = stem_base.with_suffix(".engine")
        wedetect_onnx2engine(
            onnx_path,
            eng,
            tower=tower,
            imgsz=(h, w),
            embed_dim=embed,
            max_classes=max_classes,
            fixed_classes=bool(nms and tower == "vision"),
            native_nms=bool(use_native_nms and tower == "vision"),
            max_det=max_det,
            conf=conf,
            iou=iou,
            workspace=workspace,
            quantize=quantize,
            builder_optimization_level=builder_optimization_level,
            metadata=meta if tower == "vision" else {**meta, "end2end": "False", "nms": "False"},
            verbose=verbose,
            prefix=prefix,
        )
        engines.append(str(eng))
    LOGGER.info(f"{prefix} exported {len(engines)} engine(s): {', '.join(engines)}")
    return engines


def export_wedetect_torchscript(
    model,
    file: Path | str,
    imgsz: int | tuple[int, int] = 640,
    *,
    device: str | torch.device = "cpu",
    nms: bool = True,
    max_det: int = 300,
    conf: float = 0.25,
    iou: float = 0.45,
    max_classes: int = 80,
    prefix: str = "",
) -> list[str]:
    """Export WeDetect dual TorchScript towers with optional in-graph NMS.

    Dual layout keeps open-vocabulary prompts: language encodes ``set_classes`` text,
    vision consumes ``txt_feats``. When ``nms=True`` (default for speed), vision emits
    packed ``(B, max_det, 6)`` via ``batched_nms`` so predictors skip Python NMS.

    Args:
        model: ``WeDetectModel`` instance.
        file: Source model path (stem used for output names).
        imgsz: Square size or ``(h, w)``.
        device: Export device.
        nms: Bake class-aware NMS into the vision tower (recommended).
        max_det / conf / iou: NMS parameters when ``nms=True``.
        max_classes: Fixed class-axis ceiling for traced vision (pad prompts to this).
        prefix: Log prefix.

    Returns:
        (list[str]): ``[vision.torchscript, language.torchscript]``.
    """
    import json

    prefix = prefix or colorstr("WeDetect TorchScript:")
    file = Path(file)
    device = torch.device(device)
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    embed = int(getattr(model.model[-1].cv3[0][-1], "out_channels", 768))
    max_classes = max(1, int(max_classes))

    prepared, text_encoder = _prepare_text_encoder(model)
    prepared = prepared.to(device)
    # Language must be traced on CPU: HF XLM-R embeds bake device("cpu") into the graph under
    # jit.trace; running that module on CUDA later fails gather. Prompts are rare vs vision frames.
    text_encoder = text_encoder.to("cpu")
    vision = WeDetectVisionExport(prepared, imgsz=(h, w)).to(device).eval()
    language = WeDetectLanguageExport(text_encoder).to("cpu").eval()
    if nms:
        vision = WeDetectVisionNMSExport(
            vision, max_det=max_det, conf=conf, iou=iou, max_classes=max_classes
        ).to(device).eval()

    dummy_img = torch.zeros(1, 3, h, w, device=device)
    # Non-zero text feats so NMS control-flow / batched_nms is recorded under jit.trace
    n_txt = max_classes if nms else 8
    dummy_txt = F.normalize(torch.randn(1, n_txt, embed, device=device), dim=-1)
    input_ids, attention_mask = _dummy_tokens(text_encoder, num_classes=min(8, max_classes))

    v_out = str(file.with_name(f"{file.stem}_vision.torchscript"))
    l_out = str(file.with_name(f"{file.stem}_language.torchscript"))
    meta = {
        "task": "detect",
        "export_mode": "dual",
        "format": "torchscript",
        "imgsz": f"{h},{w}",
        "embed_dim": str(embed),
        "text_model": getattr(model, "text_model_variant", "xlm-roberta:base"),
        "stride": str(int(max(int(s) for s in (vision.vision if nms else vision).stride.tolist()))),
        "max_classes": str(max_classes),
        "end2end": str(bool(nms)),
        "nms": str(bool(nms)),
        "nms_format": "packed6" if nms else "",
        "max_det": str(int(max_det)),
        "conf": str(float(conf)),
        "iou": str(float(iou)),
        "language_device": "cpu",
    }
    extra = {"config.txt": json.dumps(meta)}

    LOGGER.info(f"{prefix} tracing vision tower -> {v_out} (nms={nms})")
    vis_ts = torch.jit.trace(vision, (dummy_img, dummy_txt), strict=False, check_trace=False)
    vis_ts.save(v_out, _extra_files=extra)

    LOGGER.info(f"{prefix} tracing language tower on CPU -> {l_out}")
    lang_ts = torch.jit.trace(language, (input_ids, attention_mask), strict=False, check_trace=False)
    lang_meta = {**meta, "end2end": "False", "nms": "False", "nms_format": ""}
    lang_ts.save(l_out, _extra_files={"config.txt": json.dumps(lang_meta)})

    LOGGER.info(f"{prefix} exported 2 file(s): {v_out}, {l_out}")
    return [v_out, l_out]

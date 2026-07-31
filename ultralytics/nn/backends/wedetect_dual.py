# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""WeDetect dual-tower ONNX / TensorRT inference backend.

Loads ``*_vision.{onnx|engine}`` + sibling ``*_language.{onnx|engine}``, keeps HF
tokenization in Python, and returns predictions for DetectionPredictor:

- raw engines/ONNX: YOLO layout ``(B, 4+K, N)`` for Python NMS
- EfficientNMS engines: end2end layout ``(B, max_det, 6)``
"""

from __future__ import annotations

import json
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import IS_JETSON, LOGGER, PYTHON_VERSION
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version

from .base import BaseBackend

Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))


def resolve_wedetect_dual_pair(weight: str | Path) -> tuple[Path, Path]:
    """Resolve vision weight path to (vision, language) sibling paths.

    Args:
        weight (str | Path): Path ending with ``_vision.onnx`` or ``_vision.engine``.

    Returns:
        (tuple[Path, Path]): Absolute vision and language paths.

    Raises:
        FileNotFoundError: If the language sibling is missing.
        ValueError: If the stem does not follow the ``*_vision`` convention.
    """
    vision = Path(weight).resolve()
    if not vision.stem.endswith("_vision"):
        raise ValueError(
            f"WeDetect dual weight must be named '*_vision.{{onnx,engine}}', got '{vision.name}'. "
            f"Export with export_mode='dual' first."
        )
    language = vision.with_name(vision.name.replace("_vision", "_language", 1))
    if not language.is_file():
        raise FileNotFoundError(
            f"WeDetect dual language tower not found next to vision weights: expected '{language}'"
        )
    return vision, language


def is_wedetect_dual_weight(weight: str | Path | torch.nn.Module) -> bool:
    """Return True if *weight* is a WeDetect dual vision tower with a language sibling."""
    if isinstance(weight, torch.nn.Module):
        return False
    try:
        vision = Path(weight)
    except TypeError:
        return False
    if vision.suffix not in {".onnx", ".engine"}:
        return False
    if not vision.stem.endswith("_vision"):
        return False
    language = vision.with_name(vision.name.replace("_vision", "_language", 1))
    return vision.is_file() and language.is_file()


def _read_onnx_metadata(path: Path) -> dict:
    """Read Ultralytics metadata props from an ONNX file."""
    try:
        import onnx

        model = onnx.load(str(path))
        return {p.key: p.value for p in model.metadata_props}
    except Exception:
        return {}


def _read_engine_metadata(path: Path) -> dict | None:
    """Read JSON metadata prefix from a TensorRT engine file, if present."""
    try:
        with open(path, "rb") as f:
            meta_len = int.from_bytes(f.read(4), byteorder="little")
            return json.loads(f.read(meta_len).decode("utf-8"))
    except Exception:
        return None


def _normalize_wedetect_metadata(meta: dict | None) -> dict:
    """Normalize WeDetect export metadata for BaseBackend.apply_metadata.

    Dual export stores ``stride`` as ``'8,16,32'`` and ``imgsz`` as ``'640,640'``.
    ``apply_metadata`` expects stride/batch as ints and imgsz as a literal-eval string.
    """
    if not meta:
        return {}
    out = dict(meta)
    stride = out.get("stride")
    if isinstance(stride, str) and "," in stride:
        out["stride"] = str(max(int(x) for x in stride.split(",") if str(x).strip()))
    imgsz = out.get("imgsz")
    if isinstance(imgsz, str) and "," in imgsz and not imgsz.strip().startswith("("):
        parts = [int(x) for x in imgsz.replace("x", ",").split(",") if str(x).strip()]
        if parts:
            out["imgsz"] = str((parts[0], parts[-1]))
    # Bool-ish export flags often arrive as strings
    for key in ("end2end", "nms"):
        if key in out and isinstance(out[key], str):
            out[key] = out[key].strip().lower() in {"1", "true", "yes"}
    return out


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert xyxy boxes ``(..., 4)`` to xywh for YOLO NMS packing."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)], dim=-1)


def _resolve_dynamic_shape(name: str, shape: tuple) -> tuple:
    """Replace ``-1`` dims with safe max sizes for TensorRT output buffers."""
    out = []
    ndim = len(shape)
    for i, d in enumerate(shape):
        if d >= 0:
            out.append(int(d))
            continue
        if name in {"nms_indices", "det_indices"}:
            out.append(24000 if i == 0 else 3)  # max_det * max_classes, index tuple
        elif name in {"scores", "det_scores"} and i == ndim - 1:
            out.append(80)  # class ceiling used by dual EfficientNMS / native NMS exports
        elif name in {"bboxes", "scores", "det_boxes"} and i == 1:
            out.append(8400)
        else:
            out.append(80)
    if not out:
        return (1,)
    return tuple(max(1, x) for x in out)


def _as_bool(v) -> bool:
    """Coerce metadata / attribute values to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return False


class _TRTTower:
    """Thin TensorRT execution helper for one WeDetect tower (vision or language)."""

    def __init__(self, weight: Path, device: torch.device):
        """Load a TensorRT engine and prepare named I/O bindings."""
        if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
            check_requirements("numpy==1.23.5")
        try:
            import tensorrt as trt
        except ImportError:
            check_tensorrt()
            import tensorrt as trt

        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

        self.device = device if device.type == "cuda" else torch.device("cuda:0")
        logger = trt.Logger(trt.Logger.INFO)
        with open(weight, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                f.read(meta_len)  # skip metadata; DualBackend reads it separately
            except Exception:
                f.seek(0)
            engine = runtime.deserialize_cuda_engine(f.read())
        self.engine = engine
        self.context = engine.create_execution_context()
        self.is_trt10 = not hasattr(engine, "num_bindings")
        self.bindings = OrderedDict()
        self.input_names = []
        self.output_names = []
        self.fp16 = False

        num = range(engine.num_io_tensors) if self.is_trt10 else range(engine.num_bindings)
        for i in num:
            if self.is_trt10:
                name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(engine.get_tensor_shape(name))
                profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = engine.get_binding_name(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                shape = tuple(engine.get_binding_shape(i))
                profile_shape = tuple(engine.get_profile_shape(0, i)[1]) if is_input else None

            if is_input:
                self.input_names.append(name)
                if -1 in shape and profile_shape is not None:
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape = (
                tuple(self.context.get_tensor_shape(name))
                if self.is_trt10
                else tuple(self.context.get_binding_shape(i))
            )
            # Dynamic dims (-1) need a concrete allocation before execute.
            shape = _resolve_dynamic_shape(name, shape)
            buf = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, buf, int(buf.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def _prepare_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Cast inputs and update dynamic shapes / output buffers when needed."""
        prepared = {}
        for name, tensor in inputs.items():
            tensor = tensor.to(self.device).contiguous()
            want = self.bindings[name].dtype
            if want == np.float16 and tensor.dtype != torch.float16:
                tensor = tensor.half()
            elif want == np.float32 and tensor.dtype != torch.float32:
                tensor = tensor.float()
            elif want in {np.int32, np.int64} and tensor.dtype not in {torch.int32, torch.int64}:
                tensor = tensor.to(torch.int64 if want == np.int64 else torch.int32)
            prepared[name] = tensor
            shape = tuple(tensor.shape)
            if shape != self.bindings[name].shape:
                if self.is_trt10:
                    self.context.set_input_shape(name, shape)
                else:
                    self.context.set_binding_shape(self.engine.get_binding_index(name), shape)
                self.bindings[name] = self.bindings[name]._replace(shape=shape)
        # Refresh output buffers after any input shape change (skip still-dynamic -1 dims)
        for oname in self.output_names:
            oshape = (
                tuple(self.context.get_tensor_shape(oname))
                if self.is_trt10
                else tuple(self.context.get_binding_shape(self.engine.get_binding_index(oname)))
            )
            if any(d < 0 for d in oshape):
                continue
            if oshape != self.bindings[oname].shape:
                self.bindings[oname].data.resize_(oshape)
                self.bindings[oname] = self.bindings[oname]._replace(
                    shape=oshape, ptr=int(self.bindings[oname].data.data_ptr())
                )
        self.binding_addrs = OrderedDict((n, int(d.data.data_ptr())) for n, d in self.bindings.items())
        for name, tensor in prepared.items():
            self.binding_addrs[name] = int(tensor.data_ptr())
        return prepared

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Execute the tower with named inputs; return named output tensors."""
        prepared = self._prepare_inputs(inputs)
        self.context.execute_v2(list(self.binding_addrs.values()))
        # Keep prepared tensors alive across execute_v2
        _ = prepared
        # Slice dynamic outputs to the runtime shape reported by the execution context
        result = {}
        for name in self.output_names:
            data = self.bindings[name].data
            if self.is_trt10:
                oshape = tuple(self.context.get_tensor_shape(name))
            else:
                oshape = tuple(self.context.get_binding_shape(self.engine.get_binding_index(name)))
            if oshape and all(d >= 0 for d in oshape) and tuple(data.shape) != oshape:
                # View the valid prefix without reallocating device memory
                numel = 1
                for d in oshape:
                    numel *= d
                result[name] = data.reshape(-1)[:numel].reshape(oshape)
            else:
                result[name] = data
        return result


class WeDetectDualBackend(BaseBackend):
    """WeDetect dual-tower backend for ``*_vision`` + ``*_language`` ONNX/TensorRT weights."""

    def load_model(self, weight: str | Path) -> None:
        """Load vision + language towers and prepare the HF tokenizer."""
        self.vision_path, self.language_path = resolve_wedetect_dual_pair(weight)
        self.format = "engine" if self.vision_path.suffix == ".engine" else "onnx"
        LOGGER.info(f"Loading WeDetect dual {self.format}: vision={self.vision_path.name}, language={self.language_path.name}")

        if self.format == "onnx":
            self._load_onnx()
        else:
            if self.device.type == "cpu":
                self.device = torch.device("cuda:0")
            self._load_engine()

        meta = self.metadata or {}
        if str(meta.get("export_mode", "dual")).lower() not in {"dual", ""}:
            LOGGER.warning(f"WeDetect dual backend loaded weights with export_mode={meta.get('export_mode')!r}")

        imgsz = getattr(self, "imgsz", (640, 640))
        if isinstance(imgsz, (list, tuple)):
            self.imgsz = (int(imgsz[0]), int(imgsz[-1]))
        else:
            self.imgsz = (int(imgsz), int(imgsz))
        self.stride = int(getattr(self, "stride", 32) or 32)
        self.embed_dim = int(meta.get("embed_dim", getattr(self, "embed_dim", 768)))
        self.max_classes = int(meta.get("max_classes", getattr(self, "max_classes", 80)) or 80)
        self.max_det = int(meta.get("max_det", getattr(self, "max_det", 300)) or 300)
        self.text_model_name = str(meta.get("text_model", "xlm-roberta-base")).split(":")[-1]
        if self.text_model_name in {"base", "large"}:
            self.text_model_name = f"xlm-roberta-{self.text_model_name}"
        self.task = "detect"
        # Image H/W is fixed at export; only class-count inputs are dynamic. Keep dynamic=False so
        # LetterBox does not use min-rect (which would break fixed 640x640 ONNX/TRT vision inputs).
        self.dynamic = False
        self.end2end = _as_bool(meta.get("end2end", False)) or _as_bool(meta.get("nms", False))
        if self.end2end and self.format == "onnx":
            raise RuntimeError(
                "WeDetect EfficientNMS ONNX is TensorRT-only and cannot run in ONNX Runtime. "
                "Export/load ``*_vision.engine`` (nms=True) instead."
            )
        self.txt_feats: torch.Tensor | None = None
        self.nc_active = 0
        self._tokenizer = None
        # Default prompt so warmup / accidental predict without set_classes still runs
        self.set_classes(["object"], _quiet=True)

    def _load_onnx(self) -> None:
        """Create ONNX Runtime sessions for both towers."""
        check_requirements("onnxruntime-gpu" if self.device.type != "cpu" else "onnxruntime")
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self.device.type == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": self.device.index or 0}), "CPUExecutionProvider"]
        self.lang_session = ort.InferenceSession(str(self.language_path), providers=providers)
        self.vis_session = ort.InferenceSession(str(self.vision_path), providers=providers)
        self.lang_tower = None
        self.vis_tower = None
        meta = _read_onnx_metadata(self.vision_path) or _read_onnx_metadata(self.language_path)
        self.apply_metadata(_normalize_wedetect_metadata(meta))
        self.model = self.vis_session

    def _load_engine(self) -> None:
        """Create TensorRT towers for language and vision."""
        self.lang_tower = _TRTTower(self.language_path, self.device)
        self.vis_tower = _TRTTower(self.vision_path, self.device)
        self.lang_session = None
        self.vis_session = None
        self.fp16 = bool(self.vis_tower.fp16 or self.lang_tower.fp16)
        meta = _read_engine_metadata(self.vision_path) or _read_engine_metadata(self.language_path) or {}
        self.apply_metadata(_normalize_wedetect_metadata(meta))
        self.model = self.vis_tower.engine

    def _get_tokenizer(self):
        """Lazily construct the HuggingFace tokenizer for class prompts."""
        if self._tokenizer is None:
            check_requirements("transformers")
            from transformers import AutoTokenizer

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, local_files_only=True)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        return self._tokenizer

    def set_classes(self, classes: list[str], _quiet: bool = False) -> None:
        """Encode class name prompts into cached ``txt_feats`` via the language tower.

        Args:
            classes (list[str]): Open-vocabulary class prompts (e.g. ``["车", "人"]``).
            _quiet (bool): Suppress info log (used for default warmup prompt).
        """
        if not classes:
            raise ValueError("WeDetect dual set_classes requires a non-empty class list")
        names = [str(c).strip() for c in classes if str(c).strip()]
        if not names:
            raise ValueError("WeDetect dual set_classes received only empty class strings")
        if len(names) > self.max_classes:
            raise ValueError(
                f"WeDetect dual set_classes got {len(names)} classes but engine max_classes={self.max_classes}. "
                f"Re-export with a larger max_classes or pass fewer prompts."
            )
        self.names = {i: n for i, n in enumerate(names)}
        self.nc_active = len(names)
        tokenizer = self._get_tokenizer()
        encoded = tokenizer(text=names, return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = encoded["input_ids"].to(dtype=torch.int64)
        attention_mask = encoded["attention_mask"].to(dtype=torch.int64)

        if self.format == "onnx":
            feats = self.lang_session.run(
                None,
                {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": attention_mask.cpu().numpy(),
                },
            )[0]
            self.txt_feats = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        else:
            out = self.lang_tower.run(
                {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }
            )
            # language export output name is txt_feats
            key = "txt_feats" if "txt_feats" in out else next(iter(out))
            self.txt_feats = out[key].float()
        if self.txt_feats.ndim == 2:
            self.txt_feats = self.txt_feats.unsqueeze(0)

        # EfficientNMS vision engines lock txt_feats to max_classes; pad unused slots with zeros.
        if self.end2end and self.txt_feats.shape[1] < self.max_classes:
            pad = torch.zeros(
                self.txt_feats.shape[0],
                self.max_classes - self.txt_feats.shape[1],
                self.txt_feats.shape[2],
                dtype=self.txt_feats.dtype,
                device=self.txt_feats.device,
            )
            self.txt_feats = torch.cat([self.txt_feats, pad], dim=1)
        if not _quiet:
            LOGGER.info(f"WeDetect dual prompts ({len(names)}): {names[:8]}{'...' if len(names) > 8 else ''}")

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        """Run vision tower and pack outputs for DetectionPredictor postprocess.

        Args:
            im (torch.Tensor): Letterboxed image batch ``BCHW`` in ``[0, 1]``.

        Returns:
            (torch.Tensor): Raw layout ``(B, 4+K, N)`` or end2end ``(B, max_det, 6)``.
        """
        if self.txt_feats is None:
            raise RuntimeError("WeDetect dual backend has no text features; call set_classes(...) first")

        b = im.shape[0]
        txt = self.txt_feats
        if txt.shape[0] != b:
            txt = txt.expand(b, -1, -1)

        if self.end2end:
            return self._forward_e2e(im, txt)

        if self.format == "onnx":
            image = im.detach().float().cpu().numpy()
            feats = txt.detach().float().cpu().numpy()
            bboxes, scores = self.vis_session.run(None, {"image": image, "txt_feats": feats})
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
        else:
            if self.fp16:
                im = im.half()
                txt = txt.half()
            else:
                im = im.float()
                txt = txt.float()
            out = self.vis_tower.run({"image": im.to(self.device), "txt_feats": txt.to(self.device)})
            bboxes = out.get("bboxes", out[sorted(out)[0]]).float()
            scores = out.get("scores", out[sorted(out)[-1]]).float()

        # Drop padded class columns if present
        if self.nc_active and scores.shape[-1] > self.nc_active:
            scores = scores[..., : self.nc_active]

        # Export already decoded xyxy; NMS expects xywh then converts to xyxy
        xywh = _xyxy_to_xywh(bboxes)  # B,N,4
        return torch.cat([xywh, scores], dim=-1).permute(0, 2, 1).contiguous()

    def _forward_e2e(self, im: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        """Run NMS vision engine and pack ``(B, max_det, 6)`` xyxy+conf+cls.

        Supports:
        - ``EfficientNMS_TRT`` outputs: ``det_boxes`` / ``det_scores`` / ``det_classes`` / ``num_dets``
        - TensorRT 11+ ``INMSLayer`` outputs: ``bboxes`` / ``scores`` / ``nms_indices`` / ``num_detections``
        """
        if self.fp16:
            im = im.half()
            txt = txt.half()
        else:
            im = im.float()
            txt = txt.float()
        out = self.vis_tower.run({"image": im.to(self.device), "txt_feats": txt.to(self.device)})

        if "det_boxes" in out:
            return self._pack_efficientnms(out)
        return self._pack_indices_nms(out)

    def _pack_efficientnms(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack EfficientNMS_TRT plugin outputs to ``(B, max_det, 6)``."""
        boxes = out["det_boxes"].float()
        scores = out["det_scores"].float()
        classes = out["det_classes"].float()
        num_dets = out["num_dets"].long()
        b, max_det = boxes.shape[0], boxes.shape[1]
        packed = torch.zeros(b, max_det, 6, device=self.device, dtype=torch.float32)
        packed[..., :4] = boxes
        packed[..., 4] = scores
        packed[..., 5] = classes
        for i in range(b):
            n = int(num_dets[i].view(-1)[0].item())
            if n < max_det:
                packed[i, n:, :] = 0
            if self.nc_active:
                packed[i, packed[i, :, 5] >= self.nc_active, :] = 0
        return packed

    def _pack_indices_nms(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Gather TRT ``INMSLayer`` indices into ``(B, max_det, 6)``."""
        boxes = out["bboxes"].float()  # B,N,4
        scores = out["scores"].float()  # B,N,K
        indices = out["nms_indices"].long()  # M,3 = batch, class, box
        num = out["num_detections"]
        n_sel = int(num.view(-1)[0].item()) if num.numel() else 0
        if indices.ndim == 1:
            indices = indices.view(-1, 3)
        if n_sel and indices.shape[0] >= n_sel:
            indices = indices[:n_sel]
        elif n_sel == 0:
            indices = indices[:0]

        b = boxes.shape[0]
        max_det = self.max_det
        packed = torch.zeros(b, max_det, 6, device=self.device, dtype=torch.float32)
        counts = [0] * b
        for row in indices:
            bi, ci, xi = int(row[0]), int(row[1]), int(row[2])
            if bi < 0 or bi >= b or counts[bi] >= max_det:
                continue
            if self.nc_active and ci >= self.nc_active:
                continue
            if xi < 0 or xi >= boxes.shape[1] or ci < 0 or ci >= scores.shape[-1]:
                continue
            j = counts[bi]
            packed[bi, j, :4] = boxes[bi, xi]
            packed[bi, j, 4] = scores[bi, xi, ci]
            packed[bi, j, 5] = float(ci)
            counts[bi] += 1
        return packed

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ultralytics.data.utils import check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import WeDetectUniModel
from ultralytics.utils import LOGGER, YAML, colorstr, nms
from ultralytics.utils.torch_utils import select_device, unwrap_model


def mixed_val_yaml_paths(data_arg) -> list[str]:
    """Return ``val.yolo_data`` paths from a mixed WeDetect yaml or dict.

    Empty when ``data_arg`` is a single-dataset yaml/dict.
    """
    raw = data_arg
    if isinstance(data_arg, (str, Path)):
        p = Path(str(data_arg))
        if p.suffix.lower() not in {".yaml", ".yml"} or not p.exists():
            return []
        raw = YAML.load(p)
    if not isinstance(raw, dict):
        return []
    val = raw.get("val")
    if not isinstance(val, dict):
        return []
    return [str(x) for x in (val.get("yolo_data") or []) if x]


def resolve_wedetect_class_names(data: dict) -> list[str]:
    """Resolve validation/inference class prompts for WeDetect.

    Prefers the primary entry of each ``class_texts`` row (open-vocabulary Chinese
    prompts) when present and length-matched to ``nc``; otherwise falls back to
    ``data["names"]``.
    """
    names_fallback = [str(name).split("/", 1)[0] for name in data["names"].values()]
    nc = int(data.get("nc") or len(names_fallback))
    path_str = data.get("class_texts")
    if not path_str:
        return names_fallback
    p = Path(path_str)
    if not p.exists():
        LOGGER.warning(f"class_texts not found at '{p}', using data.names for validation prompts")
        return names_fallback
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        return names_fallback
    prompts = []
    for item in raw:
        if isinstance(item, list) and item:
            prompts.append(str(item[0]).split("/", 1)[0].strip())
        else:
            prompts.append(str(item).split("/", 1)[0].strip())
    if len(prompts) < nc:
        LOGGER.warning(f"class_texts length ({len(prompts)}) < nc ({nc}); using data.names for validation prompts")
        return names_fallback
    # YAML omitted names/nc: placeholder nc=1 must not keep only the first class_texts row.
    if data.get("_ktw_auto_names"):
        return prompts
    # Allow extra OV vocabulary rows beyond nc (training negatives); val uses first nc
    if len(prompts) > nc:
        LOGGER.info(f"Using first {nc} of {len(prompts)} class_texts entries for validation prompts")
    return prompts[:nc]


def ktw_realign_val_names(data: dict | None, model_names) -> list[str] | None:
    """Return JSON-inferred val class names when they disagree with the model vocab.

    YAML without ``names`` uses a placeholder ``object`` class. The val dataset then fills ``data["names"]`` from unique
    ktw-anno tags. Confusion-matrix ``nc`` and text prompts must follow those inferred names; using the placeholder
    yields ``IndexError`` on GT class ids. Closed-set YAML / ``class_texts`` vocabs are left unchanged.
    """
    if not isinstance(data, dict):
        return None
    names_map = data.get("names") or {}
    if not names_map:
        return None
    inferred = [str(names_map[k]).split("/", 1)[0] for k in sorted(names_map, key=lambda x: int(x))]
    if isinstance(model_names, dict):
        current = [str(model_names[k]).split("/", 1)[0] for k in sorted(model_names, key=lambda x: int(x))]
    else:
        current = [str(x).split("/", 1)[0] for x in (model_names or [])]
    if inferred == current:
        return None
    placeholder = len(current) == 1 and current[0].strip().lower() == "object"
    if data.get("_ktw_auto_names") or placeholder or len(inferred) != len(current):
        return inferred
    return None


def _wedetect_prompt_owner(model):
    """Return the module that can encode WeDetect prompts.

    Standalone ``model.val()`` wraps the detector in ``AutoBackend``. ``unwrap_model`` does not strip that wrapper, and
    ``PyTorchBackend`` has no ``set_classes``. Prefer the dual backend (it owns ``set_classes``), then the inner
    ``WeDetectModel``.
    """
    m = unwrap_model(model)
    if callable(getattr(m, "set_classes", None)):
        return m
    backend = getattr(m, "backend", None)
    if callable(getattr(backend, "set_classes", None)):
        return backend
    inner = getattr(backend, "model", None) if backend is not None else getattr(m, "model", None)
    if callable(getattr(inner, "set_classes", None)):
        return inner
    return m


def prepare_wedetect_text_prompts(model, names: list[str], device=None) -> None:
    """Encode class prompts into ``txt_feats`` and align ``names`` / head ``nc``.

    Safe for both training validation (EMA) and standalone ``model.val()`` / ``model.set_classes()`` paths. Does not
    change user-facing class indices: ``names[i]`` corresponds to prediction class id ``i``.
    """
    owner = _wedetect_prompt_owner(model)
    if device is not None:
        text_enc = getattr(owner, "text_model", None) or getattr(owner, "clip_model", None)
        if isinstance(text_enc, nn.Module):
            text_enc.to(device)
    owner.set_classes(names, cache_clip_model=True)
    names_map = dict(enumerate(names))
    owner.names = names_map
    # DetMetrics / confusion matrix read names on AutoBackend, not only the inner module.
    if model is not owner:
        model.names = names_map
    backend = getattr(model, "backend", None)
    if backend is not None and backend is not owner:
        backend.names = names_map


class WeDetectValidator(DetectionValidator):
    """Validator for WeDetect open-vocabulary detection models.

    Before each validation pass, refreshes text embeddings from the current language tower (critical for OV fine-tuning
    where the LM is updated) and prefers ``class_texts`` prompts over English ``names`` when available.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize WeDetect validator; always keep per-class scores on the same box."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.multi_label = True

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """NMS with ``multi_label=True`` for train-val and standalone ``model.val()``.

        Co-located classes (e.g. 人 + 未戴安全帽的人) must not be reduced to argmax. Class-wise NMS still applies
        unless ``agnostic_nms`` is set.
        """
        self.args.multi_label = True
        nc = int(self.nc) if self.nc else 0
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _validate_mixed_sets(self, model, paths: list[str]):
        """Run standalone val on every mixed ``val.yolo_data`` entry (not only the first)."""
        self._mixed_val_running = True
        metrics_all: dict[str, float] = {}
        fitnesses: list[float] = []
        tags: list[str] = []
        saved_data = self.args.data
        saved_split = getattr(self.args, "split", None)
        saved_loader = self.dataloader
        saved_dir = self.save_dir
        try:
            for i, yaml_path in enumerate(paths):
                tag = Path(str(yaml_path)).stem or f"val{i}"
                tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
                self.args.data = yaml_path
                self.args.split = "minival" if "lvis" in str(yaml_path).lower() else "val"
                self.dataloader = None
                if saved_dir is not None:
                    d = Path(saved_dir) / tag
                    d.mkdir(parents=True, exist_ok=True)
                    self.save_dir = d
                LOGGER.info(f"{colorstr('WeDetect val:')} [{i + 1}/{len(paths)}] {tag}")
                metrics = self(model=model)
                if not isinstance(metrics, dict):
                    continue
                fit = float(metrics.pop("fitness", 0.0) or 0.0)
                fitnesses.append(fit)
                tags.append(tag)
                if i == 0:
                    metrics_all.update(metrics)
                for k, v in metrics.items():
                    metrics_all[f"{tag}/{k}"] = v
                metrics_all[f"{tag}/fitness"] = fit
        finally:
            self._mixed_val_running = False
            self.args.data = saved_data
            self.args.split = saved_split
            self.dataloader = saved_loader
            self.save_dir = saved_dir
        if fitnesses:
            n = len(fitnesses)
            cfg = saved_data
            if isinstance(cfg, (str, Path)):
                p = Path(str(cfg))
                if p.suffix.lower() in {".yaml", ".yml"} and p.exists():
                    cfg = YAML.load(p)
            raw = cfg.get("val_fitness_weights") if isinstance(cfg, dict) else None
            try:
                weights = [float(x) for x in list(raw)] if raw is not None else []
            except (TypeError, ValueError):
                weights = []
            if len(weights) != n or sum(weights) <= 0:
                weights = [1.0 / n] * n
            else:
                s = sum(weights)
                weights = [x / s for x in weights]
            combined = float(sum(w * f for w, f in zip(weights, fitnesses)))
            metrics_all["fitness"] = combined
            detail = ", ".join(f"{t}={f:.5f}(w={w:.3f})" for t, f, w in zip(tags, fitnesses, weights))
            LOGGER.info(f"{colorstr('WeDetect val:')} combined fitness={combined:.5f} ← {detail}")
        self.mixed_metrics = metrics_all
        return metrics_all

    def __call__(self, trainer=None, model=None):
        """Set / refresh dataset class prompts, then run validation."""
        if trainer is not None:
            model = unwrap_model(trainer.ema.ema)
            if isinstance(model, WeDetectUniModel):
                nc = model.embeddings.shape[0]
                model.model[-1].nc = nc
                return super().__call__(trainer, model)
            names = resolve_wedetect_class_names(trainer.data)
            # Restore after val: training uses text-slot nc which may differ from val nc
            state = (getattr(model, "names", None), getattr(model, "txt_feats", None), model.model[-1].nc)
            try:
                prepare_wedetect_text_prompts(model, names, device=trainer.device)
                LOGGER.info(f"WeDetect val prompts ({len(names)}): {names[:8]}{'...' if len(names) > 8 else ''}")
                return super().__call__(trainer, model)
            finally:
                model.names, model.txt_feats, model.model[-1].nc = state

        self.device = select_device(self.args.device, verbose=False)
        if not isinstance(model, torch.nn.Module):
            from ultralytics.nn.tasks import load_checkpoint

            model = load_checkpoint(model or self.args.model, device=self.device)[0]
        model = unwrap_model(model)
        model.eval().to(self.device)
        if isinstance(model, WeDetectUniModel):
            nc = model.embeddings.shape[0]
            model.model[-1].nc = nc
            return super().__call__(trainer, model)

        self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)
        data_arg = self.args.data
        mixed_paths = mixed_val_yaml_paths(data_arg)
        if mixed_paths and len(mixed_paths) > 1 and not getattr(self, "_mixed_val_running", False):
            return self._validate_mixed_sets(model, mixed_paths)
        # Mixed yaml may still be a dict here (e.g. model.val after train); unwrap primary val.
        if isinstance(data_arg, dict):
            if isinstance(data_arg.get("val"), dict):
                yolo_data = data_arg["val"].get("yolo_data") or []
                if not yolo_data:
                    raise FileNotFoundError("Mixed data dict has empty val.yolo_data for validation")
                data_arg = yolo_data[0]
                self.args.data = data_arg
                if isinstance(data_arg, str) and "lvis" in data_arg.lower():
                    self.args.split = getattr(self.args, "split", None) or "minival"
            elif "names" in data_arg and ("path" in data_arg or "val" in data_arg):
                data = data_arg
                names = resolve_wedetect_class_names(data)
                state = (getattr(model, "names", None), getattr(model, "txt_feats", None), model.model[-1].nc)
                prepare_wedetect_text_prompts(model, names, device=self.device)
                try:
                    # Standalone val owns args.data → loader; drop any prior-set dataloader.
                    self.dataloader = None
                    return super().__call__(trainer, model)
                finally:
                    model.names, model.txt_feats, model.model[-1].nc = state
            else:
                raise FileNotFoundError(
                    f"Unsupported data dict for WeDetect validation (need val.yolo_data or a resolved subset): "
                    f"{list(data_arg)[:8]}"
                )
        data = check_det_dataset(data_arg)
        names = resolve_wedetect_class_names(data)
        state = (getattr(model, "names", None), getattr(model, "txt_feats", None), model.model[-1].nc)
        # Placeholder {0: object} is not the val vocab; init_metrics realigns after JSON inference.
        if not (data.get("_ktw_auto_names") or (len(names) == 1 and names[0].strip().lower() == "object")):
            prepare_wedetect_text_prompts(model, names, device=self.device)
        try:
            # Standalone val owns args.data → loader; drop any prior-set dataloader.
            self.dataloader = None
            return super().__call__(trainer, model)
        finally:
            model.names, model.txt_feats, model.model[-1].nc = state

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Align ktw-anno val prompts; keep ``save_json=False`` (no auto LVIS/COCO JSON)."""
        ds = getattr(getattr(self, "dataloader", None), "dataset", None)
        data = getattr(ds, "data", None) or self.data
        names = ktw_realign_val_names(data, getattr(model, "names", None))
        if names:
            prepare_wedetect_text_prompts(model, names, device=self.device)
            LOGGER.info(f"WeDetect val prompts ({len(names)}): {names[:8]}{'...' if len(names) > 8 else ''}")
        want_json = bool(self.args.save_json)
        super().init_metrics(model)
        if not want_json:
            self.args.save_json = False


class WeDetectUniValidator(DetectionValidator):
    """Validator for WeDetect-Uni models with learnable prompt embeddings.

    WeDetect-Uni uses learnable prompt embeddings that correspond to each class in the dataset. During validation, the
    model's ``embeddings`` are used directly as text features, producing per-class scores via ``BNContrastiveHead``.

    This follows the original WeDetect ``SimpleYOLOWorldDetector`` pattern where ``num_train_classes`` equals the number
    of dataset categories.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize WeDetectUniValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.multi_label = True

    def __call__(self, trainer=None, model=None):
        """Run validation for WeDetect-Uni with per-class evaluation."""
        if trainer is not None:
            model = unwrap_model(trainer.ema.ema)
        else:
            self.device = select_device(self.args.device, verbose=False)
            if not isinstance(model, torch.nn.Module):
                from ultralytics.nn.tasks import load_checkpoint

                model = load_checkpoint(model or self.args.model, device=self.device)[0]
            model = unwrap_model(model)
            model.eval().to(self.device)

        if isinstance(model, WeDetectUniModel):
            nc = model.embeddings.shape[0]
            model.model[-1].nc = nc
        return super().__call__(trainer, model)

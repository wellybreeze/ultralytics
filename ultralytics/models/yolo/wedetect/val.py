# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.data.utils import check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import WeDetectUniModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device, unwrap_model


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
        LOGGER.warning(
            f"class_texts length ({len(prompts)}) < nc ({nc}); using data.names for validation prompts"
        )
        return names_fallback
    # Allow extra OV vocabulary rows beyond nc (training negatives); val uses first nc
    if len(prompts) > nc:
        LOGGER.info(f"Using first {nc} of {len(prompts)} class_texts entries for validation prompts")
    return prompts[:nc]


def prepare_wedetect_text_prompts(model, names: list[str], device=None) -> None:
    """Encode class prompts into ``txt_feats`` and align ``names`` / head ``nc``.

    Safe for both training validation (EMA) and standalone ``model.val()`` /
    ``model.set_classes()`` paths. Does not change user-facing class indices:
    ``names[i]`` corresponds to prediction class id ``i``.
    """
    if device is not None:
        text_enc = getattr(model, "text_model", None) or getattr(model, "clip_model", None)
        if isinstance(text_enc, nn.Module):
            text_enc.to(device)
    model.set_classes(names, cache_clip_model=True)
    model.names = dict(enumerate(names))


class WeDetectValidator(DetectionValidator):
    """Validator for WeDetect open-vocabulary detection models.

    Before each validation pass, refreshes text embeddings from the current
    language tower (critical for OV fine-tuning where the LM is updated) and
    prefers ``class_texts`` prompts over English ``names`` when available.
    """

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
        prepare_wedetect_text_prompts(model, names, device=self.device)
        try:
            # Standalone val owns args.data → loader; drop any prior-set dataloader.
            self.dataloader = None
            return super().__call__(trainer, model)
        finally:
            model.names, model.txt_feats, model.model[-1].nc = state


class WeDetectUniValidator(DetectionValidator):
    """Validator for WeDetect-Uni models with learnable prompt embeddings.

    WeDetect-Uni uses learnable prompt embeddings that correspond to each
    class in the dataset.  During validation, the model's ``embeddings``
    are used directly as text features, producing per-class scores via
    ``BNContrastiveHead``.

    This follows the original WeDetect ``SimpleYOLOWorldDetector`` pattern
    where ``num_train_classes`` equals the number of dataset categories.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize WeDetectUniValidator."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

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

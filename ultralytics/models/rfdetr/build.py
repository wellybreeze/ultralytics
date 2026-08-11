# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Thin RF-DETR builder adapting the optional ``rfdetr`` package into Ultralytics."""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import LOGGER


def _require_rfdetr():
    """Import the optional ``rfdetr`` package or raise an install hint.

    Also quiets the package's ``rf-detr`` logger so only Ultralytics ``LOGGER`` output
    (``WARNING ⚠️ …`` / plain INFO) reaches the console — matching YOLO/NAS session style.
    """
    try:
        import rfdetr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "RF-DETR requires the optional `rfdetr` package. Install with `pip install ultralytics[rfdetr]`."
        ) from exc
    # Same pattern as silencing coremltools/sentry: third-party format is not Ultralytics-style.
    # Keep ERROR+ so real failures still surface.
    import logging

    logging.getLogger("rf-detr").setLevel(logging.ERROR)


def rfdetr_class_names(num_classes: int, class_names: list[str] | None = None) -> dict[int, str]:
    """Build an Ultralytics ``names`` dict for RF-DETR class indices.

    Published COCO checkpoints emit sparse category IDs (1–90 with gaps), so a contiguous
    ``{i: f'class{i}'}`` map shows placeholders like ``class6`` instead of ``bus``. Prefer
    checkpoint-embedded names when present; otherwise use Roboflow's COCO ID→name table when
    ``num_classes == 90``, padded to a dense ``0..90`` dict so AutoBackend validation accepts it.

    Args:
        num_classes (int): Configured foreground class count (RF-DETR COCO default is 90).
        class_names (list[str], optional): Contiguous names from a fine-tuned checkpoint or dataset.

    Returns:
        (dict): Mapping from label id to display name.
    """
    if class_names:
        return {i: name for i, name in enumerate(class_names)}
    if num_classes == 90:
        _require_rfdetr()
        from rfdetr.assets.coco_classes import COCO_CLASSES

        # Pad unused COCO slots (0 and the ID gaps) so max(keys) < len(names).
        return {i: COCO_CLASSES.get(i, "") for i in range(91)}
    return {i: f"class{i}" for i in range(num_classes)}


def resolve_rfdetr_variant(cfg):
    """Resolve an RF-DETR config keyword from a YAML dict or path (scale + task), like YOLO ``scale``.

    Examples:
        {'yaml_file': 'rfdetr-nano.yaml', 'scale': 'nano'} -> 'nano'
        {'yaml_file': 'rfdetr-seg-large.yaml', 'scale': 'large'} -> 'seg-large'
        {'yaml_file': 'rfdetr-pose-preview.yaml', 'scale': 'preview'} -> 'keypoint-preview'
    """
    from ultralytics.nn.tasks import guess_rfdetr_scale

    if isinstance(cfg, dict):
        yaml_file = str(cfg.get("yaml_file", "rfdetr-nano"))
        scale = cfg.get("scale") or guess_rfdetr_scale(yaml_file)
        scales = cfg.get("scales") or {}
        if not scale and scales:
            scale = next(iter(scales))
            LOGGER.warning(f"no RF-DETR model scale passed. Assuming scale='{scale}'.")
        stem = Path(yaml_file).stem.lower().replace("_", "-")
    else:
        yaml_file = str(cfg)
        scale = guess_rfdetr_scale(yaml_file)
        stem = Path(yaml_file).stem.lower().replace("_", "-")

    if "seg" in stem:
        if scale in {"preview"}:
            return "seg-preview"
        return f"seg-{scale}" if scale else "seg-nano"
    if "pose" in stem or "keypoint" in stem:
        return "keypoint-preview"
    return scale or "nano"


def get_config_class(name: str):
    """Return the RF-DETR config class selected by a model path, stem, or variant keyword."""
    _require_rfdetr()
    from rfdetr.config import (
        RFDETRBaseConfig,
        RFDETRKeypointPreviewConfig,
        RFDETRLargeConfig,
        RFDETRMediumConfig,
        RFDETRNanoConfig,
        RFDETRSeg2XLargeConfig,
        RFDETRSegLargeConfig,
        RFDETRSegMediumConfig,
        RFDETRSegNanoConfig,
        RFDETRSegPreviewConfig,
        RFDETRSegSmallConfig,
        RFDETRSegXLargeConfig,
        RFDETRSmallConfig,
    )

    config_classes = {
        "nano": RFDETRNanoConfig,
        "small": RFDETRSmallConfig,
        "medium": RFDETRMediumConfig,
        "large": RFDETRLargeConfig,
        "base": RFDETRBaseConfig,
        "keypoint-preview": RFDETRKeypointPreviewConfig,
        "pose-preview": RFDETRKeypointPreviewConfig,
        "preview": RFDETRKeypointPreviewConfig,
        "seg-preview": RFDETRSegPreviewConfig,
        "seg-nano": RFDETRSegNanoConfig,
        "seg-small": RFDETRSegSmallConfig,
        "seg-medium": RFDETRSegMediumConfig,
        "seg-large": RFDETRSegLargeConfig,
        "seg-xlarge": RFDETRSegXLargeConfig,
        "seg-2xlarge": RFDETRSeg2XLargeConfig,
        "seg-xxlarge": RFDETRSeg2XLargeConfig,
    }
    key = resolve_rfdetr_variant(name)
    if key not in config_classes:
        raise ValueError(f"Unable to infer an RF-DETR variant from '{name}'.")
    return config_classes[key]


def sync_rfdetr_model_config(model_config, args):
    """Apply Ultralytics ``default.yaml`` fields onto an RF-DETR ``ModelConfig``.

    Maps values that RF-DETR exposes on the architecture / postprocess config:
    ``max_det`` → ``num_select``, ``mask_ratio`` → ``mask_downsample_ratio``, ``amp`` → ``amp``.

    Training loop knobs (epochs, batch, lr0, workers, augmentations, …) stay on Ultralytics
    ``args`` / ``BaseTrainer`` — they are not copied into Roboflow's unused PTL ``TrainConfig``.

    Args:
        model_config: RF-DETR pydantic model config.
        args (SimpleNamespace | dict | None): Ultralytics train/val/predict overrides.

    Returns:
        Updated model config (new copy when pydantic ``model_copy`` is available).
    """
    if args is None:
        return model_config
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    updates = {}
    max_det = get("max_det")
    if max_det is not None:
        updates["num_select"] = int(max_det)
    if getattr(model_config, "segmentation_head", False):
        mask_ratio = get("mask_ratio")
        if mask_ratio is not None:
            updates["mask_downsample_ratio"] = int(mask_ratio)
    amp = get("amp")
    if amp is not None:
        updates["amp"] = bool(amp)
    if not updates:
        return model_config
    if hasattr(model_config, "model_copy"):
        return model_config.model_copy(update=updates)
    for key, value in updates.items():
        setattr(model_config, key, value)
    return model_config


def rfdetr_model_defaults_from_args(args):
    """Build RF-DETR ``ModelDefaults`` with box gains from Ultralytics ``box``.

    YOLO uses a single ``box`` gain for localization; RF-DETR splits L1 (``bbox``) and GIoU.
    Keep the official ``bbox:giou = 5:2`` ratio while setting ``bbox_loss_coef = box``.
    """
    _require_rfdetr()
    from dataclasses import replace

    from rfdetr.models._defaults import MODEL_DEFAULTS

    if args is None:
        return MODEL_DEFAULTS
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    box = float(get("box", MODEL_DEFAULTS.bbox_loss_coef))
    giou = box * (MODEL_DEFAULTS.giou_loss_coef / MODEL_DEFAULTS.bbox_loss_coef)
    return replace(MODEL_DEFAULTS, bbox_loss_coef=box, giou_loss_coef=giou)


def rfdetr_train_config_from_args(args, model_config):
    """Build an RF-DETR train config whose loss coefficients follow Ultralytics hyps.

    Mapping (``default.yaml`` → RF-DETR):
        ``cls`` → ``cls_loss_coef``
        ``box`` → ``mask_ce_loss_coef`` / ``mask_dice_loss_coef`` (segment; same as YOLO seg gain)
        ``pose`` → ``keypoint_l1_loss_coef``
        ``kobj`` → ``keypoint_findable_loss_coef`` and ``keypoint_visible_loss_coef``
        ``rle`` → ``keypoint_nll_loss_coef``

    ``dfl`` / ``angle`` have no DETR counterparts and are ignored.
    """
    _require_rfdetr()
    from rfdetr.config import KeypointTrainConfig, SegmentationTrainConfig, TrainConfig

    get = (
        (lambda k, d=None: d)
        if args is None
        else (args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d))
    )
    kwargs = {
        "dataset_dir": ".",
        "output_dir": ".",
        "cls_loss_coef": float(get("cls", 1.0)),
    }
    if getattr(model_config, "segmentation_head", False):
        box = float(get("box", 5.0))
        return SegmentationTrainConfig(
            **kwargs,
            mask_ce_loss_coef=box,
            mask_dice_loss_coef=box,
            segmentation_head=True,
        )
    if getattr(model_config, "use_grouppose_keypoints", False):
        kobj = float(get("kobj", 1.0))
        return KeypointTrainConfig(
            **kwargs,
            keypoint_l1_loss_coef=float(get("pose", 1.0)),
            keypoint_findable_loss_coef=kobj,
            keypoint_visible_loss_coef=kobj,
            keypoint_nll_loss_coef=float(get("rle", 1.0)),
        )
    return TrainConfig(**kwargs)


def build_rfdetr_criterion(model_config, args=None):
    """Build official ``SetCriterion`` with Ultralytics ``default.yaml`` loss/postprocess hyps.

    Args:
        model_config: RF-DETR architecture config (typically after ``sync_rfdetr_model_config``).
        args (SimpleNamespace | dict | None): Ultralytics overrides; ``None`` keeps Roboflow defaults.

    Returns:
        (nn.Module): RF-DETR ``SetCriterion`` with populated ``weight_dict``.
    """
    _require_rfdetr()
    from rfdetr.models.lwdetr import build_criterion_from_config

    train_config = rfdetr_train_config_from_args(args, model_config)
    defaults = rfdetr_model_defaults_from_args(args)
    criterion, _ = build_criterion_from_config(model_config, train_config, defaults=defaults)
    return criterion


def build_rfdetr_model(cfg=None, weights=None, nc=None, verbose=True):
    """Build an LWDETR model via ``rfdetr`` and optionally load native checkpoint weights.

    Args:
        cfg (dict | str | None): Ultralytics YAML dictionary, model path, or variant name.
        weights (str | Path | None): Native RF-DETR checkpoint to load after construction.
        nc (int, optional): Number of output classes.
        verbose (bool): Kept for a consistent Ultralytics model-builder interface.

    Returns:
        (tuple): LWDETR model, RF-DETR config, and optional contiguous class names from the checkpoint.
    """
    _require_rfdetr()
    from rfdetr.assets.model_weights import get_model_cache_dir
    from rfdetr.models.lwdetr import build_model_from_config
    from rfdetr.models.weights import load_pretrain_weights

    cfg = cfg or {}
    if isinstance(cfg, dict):
        variant = resolve_rfdetr_variant(cfg)
        config_class = get_config_class(variant)
        reserved = {"nc", "imgsz", "resolution", "variant", "task", "yaml_file", "scale", "scales", "ch"}
        values = {k: v for k, v in cfg.items() if k not in reserved and k in config_class.model_fields}
        if nc is None and "nc" in cfg:
            values["num_classes"] = cfg["nc"]
    else:
        variant = resolve_rfdetr_variant(str(cfg))
        values = {}
        config_class = get_config_class(variant)

    if ("xlarge" in variant or "2xlarge" in variant) and not variant.startswith("seg-"):
        raise ImportError(
            "RF-DETR xlarge and 2xlarge require rfdetr_plus and acceptance of the Platform Model License. "
            "Pass accept_platform_model_license=True to RFDETR after installing rfdetr-plus."
        )

    # Pass the checkpoint path into the config *before* build so that:
    # 1) pydantic does not warn that the model is created from scratch, and
    # 2) LWDETR skips the unused DINOv2 backbone download (load_dinov2_weights
    #    is gated on pretrain_weights is None).
    weight_path = None
    if weights:
        weight_path = Path(weights)
        if not weight_path.is_file():
            weight_path = Path(get_model_cache_dir()) / Path(weights).name
        values["pretrain_weights"] = str(weight_path)
    else:
        values["pretrain_weights"] = None
    if nc is not None:
        values["num_classes"] = nc
    model_config = config_class(**values)
    model = build_model_from_config(model_config)
    class_names = None
    if weight_path is not None:
        model_config.pretrain_weights = str(weight_path)
        loaded_names = load_pretrain_weights(model, model_config)
        class_names = loaded_names or None
    return model, model_config, class_names

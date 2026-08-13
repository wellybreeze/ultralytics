# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training support for RF-DETR models."""

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RFDETRDetectionModel, RFDETRPoseModel, RFDETRSegmentationModel
from ultralytics.utils import LOGGER, RANK, ROOT, colorstr
from ultralytics.utils.checks import check_imgsz

from .build import build_rfdetr_model, sync_rfdetr_model_config
from .val import RFDETRDataset, RFDETRValidator

# Official Roboflow hyps mapped onto Ultralytics field names (see cfg/default-rfdetr.yaml).
DEFAULT_CFG_RFDETR = ROOT / "cfg/default-rfdetr.yaml"


class RFDETRTrainer(DetectionTrainer):
    """Train native RF-DETR architectures through the Ultralytics training interface."""

    def __init__(self, cfg=DEFAULT_CFG_RFDETR, overrides=None, _callbacks=None):
        """Initialize with ``default-rfdetr.yaml`` so TrainConfig/MODEL_DEFAULTS map into Ultralytics args."""
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the wrapper matching the YAML task and optionally load its native weights."""
        task = getattr(self.args, "task", None) or "detect"
        if isinstance(cfg, dict):
            task = cfg.get("task", task)
        model_class = {
            "segment": RFDETRSegmentationModel,
            "pose": RFDETRPoseModel,
        }.get(task, RFDETRDetectionModel)
        if isinstance(weights, torch.nn.Module) and hasattr(weights, "model_config"):
            # Reuse the already-built RFDETR*Model from RFDETR._load / Model.train().
            nc = int(self.data["nc"])
            weights.nc = nc
            if self.data.get("names"):
                weights.names = self.data["names"]
            # Official RF-DETR resizes the class head when dataset nc differs from the checkpoint.
            if int(getattr(weights.model_config, "num_classes", nc)) != nc:
                weights.model.reinitialize_detection_head(nc + 1)
                if hasattr(weights.model_config, "model_copy"):
                    weights.model_config = weights.model_config.model_copy(update={"num_classes": nc})
                else:
                    weights.model_config.num_classes = nc
                weights.criterion = None  # rebuilt on next loss() with the new class count
            return weights
        model = model_class(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            from .build import rfdetr_class_names

            model.model, model.model_config, class_names = build_rfdetr_model(
                cfg=cfg or model.yaml, weights=weights, nc=self.data["nc"], verbose=verbose
            )
            # Dataset names win for fine-tuning; fall back to checkpoint / COCO mapping.
            model.names = self.data.get("names") or rfdetr_class_names(model.model_config.num_classes, class_names)
            model.nc = self.data["nc"]
            # Keep Ultralytics stride aligned with rfdetr block_size after optional weight reload.
            block = int(model.model_config.patch_size) * int(model.model_config.num_windows)
            model.stride = torch.Tensor([block])
        return model

    def set_model_attributes(self):
        """Attach dataset metadata and sync Ultralytics ``default.yaml`` fields onto RF-DETR config."""
        super().set_model_attributes()
        if hasattr(self.model, "model_config"):
            self.model.model_config = sync_rfdetr_model_config(self.model.model_config, self.args)
            # Rebuild criterion so box/cls/mask/pose gains pick up the attached ``model.args``.
            self.model.criterion = None
        self.model.overlap_mask = bool(getattr(self.args, "overlap_mask", True))

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build an RF-DETR fixed-shape dataset for training or validation."""
        # Re-validate against patch_size * num_windows (RF-DETR block_size) before LetterBox.
        stride = int(self.model.stride.max()) if hasattr(self.model, "stride") else 32
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=stride, floor=stride, max_dim=1)
        config = getattr(self.model, "model_config", None)
        if config is not None and RANK in {-1, 0} and mode == "train":
            LOGGER.info(
                f"{colorstr('RF-DETR:')} imgsz must be divisible by "
                f"patch_size×num_windows={config.patch_size}×{config.num_windows}={stride}; "
                f"using imgsz={self.args.imgsz}"
            )
        return RFDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Return the RF-DETR validator and report its principal detection losses."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RFDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

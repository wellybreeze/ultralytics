# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-FINE trainer — RT-DETR training loop with ``DFINEDetectionModel``."""

from __future__ import annotations

from copy import copy

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import DFINEDetectionModel
from ultralytics.utils import RANK

from .val import DFINEValidator


class DFINETrainer(RTDETRTrainer):
    """Trainer for D-FINE detection models (dataset / loop shared with RT-DETR)."""

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """Build a ``DFINEDetectionModel``."""
        model = DFINEDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a ``DFINEValidator`` and set loss display names."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return DFINEValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

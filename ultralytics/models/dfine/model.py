# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for D-FINE, a DETR-based detector with Fine-grained Distribution Refinement.

References:
    https://arxiv.org/abs/2410.13842
    https://github.com/Peterande/D-FINE
"""

from __future__ import annotations

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DFINEDetectionModel
from ultralytics.utils import ROOT, YAML
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import DFINEPredictor
from .train import DFINETrainer
from .val import DFINEValidator


def dfine_class_names(nc: int) -> dict[int, str]:
    """Return class-name map for a D-FINE head with ``nc`` classes.

    Official Objects365 checkpoints use ``num_classes=366`` and reserve index 0, mapping dataset class ``i`` to head
    index ``i + 1`` (see Peterande/D-FINE ``map_class_weights``). COCO checkpoints use contiguous ``0..79``.
    """
    if nc == 80:
        return YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]
    if nc == 366:
        base = YAML.load(ROOT / "cfg/datasets/Objects365.yaml")["names"]
        return {0: "background", **{int(i) + 1: name for i, name in base.items()}}
    return {i: str(i) for i in range(nc)}


def ensure_dfine_class_names(model) -> None:
    """Replace placeholder ``{i: str(i)}`` names on a loaded/converted D-FINE model when possible."""
    names = getattr(model, "names", None)
    if not isinstance(names, dict) or not names:
        return
    if not all(str(k) == str(v) for k, v in names.items()):
        return
    head = model.model[-1] if hasattr(model, "model") else None
    nc = int(getattr(head, "nc", None) or getattr(model, "nc", None) or len(names))
    model.names = dfine_class_names(nc)
    model.nc = nc


class DFINE(Model):
    """Ultralytics interface for the D-FINE object detection model.

    Attributes:
        model (str): Path to the model weights or YAML config.

    Examples:
        >>> from ultralytics import DFINE
        >>> model = DFINE("dfine-l.yaml")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "dfine-l.pt") -> None:
        """Initialize D-FINE with a weights file or YAML config.

        Args:
            model (str): Path to ``.pt`` / ``.yaml`` model.
        """
        assert TORCH_1_11, "DFINE requires torch>=1.11"
        super().__init__(model=model, task="detect")
        if self.model is not None:
            ensure_dfine_class_names(self.model)

    @property
    def task_map(self) -> dict:
        """Return the detect task map for D-FINE components."""
        return {
            "detect": {
                "predictor": DFINEPredictor,
                "validator": DFINEValidator,
                "trainer": DFINETrainer,
                "model": DFINEDetectionModel,
            }
        }

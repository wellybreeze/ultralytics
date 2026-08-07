# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for D-FINE, a DETR-based detector with Fine-grained Distribution Refinement.

References:
    https://arxiv.org/abs/2410.13842
    https://github.com/Peterande/D-FINE
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DFINEDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import DFINEPredictor
from .train import DFINETrainer
from .val import DFINEValidator


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

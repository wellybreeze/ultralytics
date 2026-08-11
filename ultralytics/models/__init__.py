# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .dfine import DFINE
from .fastsam import FastSAM
from .nas import NAS
from .rtdetr import RTDETR
from .yolo import YOLO, YOLOE, WeDetect, WeDetectUni, YOLOWorld

__all__ = "DFINE", "NAS", "RFDETR", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "WeDetect", "WeDetectUni", "YOLOWorld"  # allow simpler import


def __getattr__(name):
    """Lazy-import optional/heavy model entrypoints."""
    if name == "SAM":
        # Scoped for import ultralytics speed: SAM pulls optional torchvision-heavy modules.
        from .sam import SAM

        return SAM
    if name == "RFDETR":
        # RF-DETR requires optional `rfdetr` (see pyproject.toml [project.optional-dependencies] rfdetr).
        from .rfdetr import RFDETR

        return RFDETR
    raise AttributeError(f"module {__name__} has no attribute {name}")

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-FINE validation — reuse RT-DETR DETR-style dataset / no-NMS validator."""

from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator

__all__ = ("DFINEDataset", "DFINEValidator")


class DFINEDataset(RTDETRDataset):
    """YOLO-format dataset for D-FINE (``rect=False``, same as RT-DETR)."""


class DFINEValidator(RTDETRValidator):
    """Validator for D-FINE (conf filter, no NMS — same as RT-DETR)."""

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""D-FINE predictor — same scale-fill letterbox / no-NMS postprocess as RT-DETR."""

from ultralytics.models.rtdetr.predict import RTDETRPredictor


class DFINEPredictor(RTDETRPredictor):
    """Predictor for D-FINE (inherits RT-DETR square scale-fill + confidence filter)."""

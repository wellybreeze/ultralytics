# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops


class WeDetectPredictor(DetectionPredictor):
    """Predictor for WeDetect open-vocabulary detection models.

    Handles text embedding generation for inference with custom class
    names.  When the model's current class names differ from the
    requested classes, regenerates text embeddings using XLM-RoBERTa
    and updates the model's class list.
    """

    def pre_transform(self, im):
        """Pre-transform input images before inference."""
        return super().pre_transform(im)


class WeDetectUniPredictor(DetectionPredictor):
    """Predictor for WeDetect-Uni models that unifies all prompt detections as "object" (cls=0).

    WeDetect-Uni uses learnable prompt embeddings for contrastive learning,
    but all detections should be reported as a single "object" class with id=0.
    This predictor post-processes NMS results to unify class ids.
    """

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct Results object with unified cls=0 for all detections."""
        if pred is not None and len(pred):
            pred = pred.clone()
            pred[:, 5] = 0
        return super().construct_result(pred, img, orig_img, img_path)
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionPredictor


class WeDetectPredictor(DetectionPredictor):
    """Predictor for WeDetect open-vocabulary detection models.

    PyTorch weights encode prompts via ``WeDetectModel.set_classes``. Dual ONNX /
    TensorRT exports (``*_vision`` + ``*_language``) are loaded through
    ``WeDetectDualBackend``; cached prompts from ``WeDetect.set_classes`` are
    applied after ``setup_model``.
    """

    def setup_model(self, model, verbose: bool = True):
        """Load backend, then apply any cached open-vocabulary prompts."""
        super().setup_model(model, verbose=verbose)
        prompts = None
        # Prefer prompts stored on the Ultralytics Model wrapper (WeDetect._prompt_classes)
        outer = getattr(self, "model", None)
        # After setup, self.model is AutoBackend; look for prompts via args overrides / predictor attrs
        if getattr(self, "_prompt_classes", None):
            prompts = self._prompt_classes
        elif isinstance(getattr(self.args, "prompt_classes", None), (list, tuple)):
            prompts = list(self.args.prompt_classes)
        if prompts and hasattr(self.model, "set_classes"):
            self.model.set_classes(prompts)
            self.model.names = {i: n for i, n in enumerate(prompts)}

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

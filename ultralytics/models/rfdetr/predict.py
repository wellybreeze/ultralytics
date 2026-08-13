# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prediction support for native RF-DETR models."""

import torch
from rfdetr.models.postprocess import PostProcess

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RFDETRPredictor(BasePredictor):
    """Predict with RF-DETR and convert native outputs to Ultralytics Results."""

    def pre_transform(self, im):
        """Resize images to a square RF-DETR input without preserving aspect ratio."""
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=image) for image in im]

    def postprocess(self, preds, img, orig_imgs):
        """Postprocess RF-DETR logits, boxes, masks, or keypoints into Results objects."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        target_sizes = torch.as_tensor([image.shape[:2] for image in orig_imgs], device=img.device, dtype=img.dtype)
        config = getattr(self.model, "model_config", None)
        postprocessor = PostProcess(
            num_select=getattr(config, "num_select", self.args.max_det),
            num_keypoints_per_class=getattr(config, "num_keypoints_per_class", []),
            trace_alpha=getattr(config, "postprocess_trace_alpha", 0.2),
        )
        outputs = postprocessor(preds, target_sizes)
        results = []
        for output, orig_img, img_path in zip(outputs, orig_imgs, self.batch[0]):
            keep = output["scores"] > self.args.conf
            if self.args.classes is not None:
                keep &= (output["labels"][..., None] == torch.tensor(self.args.classes, device=img.device)).any(1)
            keep = keep.nonzero().squeeze(1)[: self.args.max_det]
            boxes = torch.cat(
                (output["boxes"][keep], output["scores"][keep, None], output["labels"][keep, None]), dim=1
            )
            kwargs = {"boxes": boxes}
            if "masks" in output:
                kwargs["masks"] = output["masks"][keep].squeeze(1)
            if "keypoints" in output:
                kwargs["keypoints"] = output["keypoints"][keep]
            results.append(Results(orig_img, path=img_path, names=self.model.names, **kwargs))
        return results

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validation support for RF-DETR models."""

from pathlib import Path

import torch

from ultralytics.data import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator
from rfdetr.models.postprocess import PostProcess
from ultralytics.utils import colorstr, ops


class RFDETRDataset(YOLODataset):
    """YOLO-format dataset configured for RF-DETR's fixed-shape inputs."""

    def load_image(self, i, rect_mode=False):
        """Load an image without forcing rectangular letterboxing."""
        return super().load_image(i=i, rect_mode=rect_mode)


class RFDETRValidator(DetectionValidator):
    """Validate RF-DETR detections using the standard Ultralytics detection metrics."""

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build an RF-DETR dataset with square, non-rectangular batches."""
        return RFDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
        )

    def scale_preds(self, predn, pbatch):
        """Return native-space predictions unchanged before metric preparation."""
        return predn

    def postprocess(self, preds):
        """Convert RF-DETR output dictionaries to DetectionValidator prediction dictionaries."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if not isinstance(preds, dict):
            raise TypeError(f"Expected RF-DETR output dictionary, received {type(preds).__name__}.")
        size = torch.full(
            (preds["pred_logits"].shape[0], 2), self.args.imgsz, device=preds["pred_logits"].device
        )
        trainer_model = getattr(getattr(self, "trainer", None), "model", None)
        config = getattr(trainer_model, "model_config", None) or getattr(self, "model_config", None)
        outputs = PostProcess(
            num_select=getattr(config, "num_select", self.args.max_det),
            num_keypoints_per_class=getattr(config, "num_keypoints_per_class", None) or [],
        )(preds, size)
        result = []
        for output in outputs:
            keep = (output["scores"] > self.args.conf).nonzero().squeeze(1)[: self.args.max_det]
            item = {
                "bboxes": output["boxes"][keep],
                "conf": output["scores"][keep],
                "cls": output["labels"][keep],
            }
            if "masks" in output:
                item["masks"] = output["masks"][keep].squeeze(1)
            if "keypoints" in output:
                item["keypoints"] = output["keypoints"][keep]
            result.append(item)
        return result

    def pred_to_json(self, predn, pbatch):
        """Serialize RF-DETR predictions in COCO's native image coordinates."""
        path = Path(pbatch["im_file"])
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz
        box = ops.xyxy2xywh(box)
        box[:, :2] -= box[:, 2:] / 2
        for bbox, score, cls in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(cls)],
                    "bbox": [round(x, 3) for x in bbox],
                    "score": round(score, 5),
                }
            )

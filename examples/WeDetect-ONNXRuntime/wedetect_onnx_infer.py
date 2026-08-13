# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
r"""WeDetect dual-tower ONNX inference with custom open-vocabulary prompts.

Export first (from a WeDetect .pt checkpoint)::

    from ultralytics import WeDetect
    model = WeDetect("wedetect_base.pt")
    model.export(format="onnx", export_mode="dual", imgsz=640)

Then run this script::

    python examples/WeDetect-ONNXRuntime/wedetect_onnx_infer.py \\
        --vision wedetect_base_vision.onnx \\
        --language wedetect_base_language.onnx \\
        --tokenizer xlm-roberta-base \\
        --source ultralytics/assets/bus.jpg \\
        --classes 人,公交车,领带

The language tower runs once per class list; vision reuses ``txt_feats``.
Tokenization stays in Python (HuggingFace), matching WeDetect/deploy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def letterbox_rgb(im_bgr: np.ndarray, imgsz: int = 640, pad_val: int = 114):
    """Letterbox BGR image to square RGB float32 CHW /255."""
    h0, w0 = im_bgr.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nh, nw = round(h0 * r), round(w0 * r)
    im = cv2.resize(im_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas = np.full((imgsz, imgsz, 3), pad_val, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = im
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return rgb[None], r, (left, top), (h0, w0)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.7, max_det: int = 300):
    """Class-agnostic torchvision NMS on xyxy boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    t_boxes = torch.from_numpy(boxes)
    t_scores = torch.from_numpy(scores)
    keep = torch.ops.torchvision.nms(t_boxes, t_scores, iou)
    return keep[:max_det].numpy()


def encode_classes(tokenizer, class_names: list[str]):
    """Tokenize class names for the language ONNX tower."""
    encoded = tokenizer(text=class_names, return_tensors="np", padding=True, truncation=True, max_length=77)
    return encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.int64)


def run_dual(
    vision_onnx: str,
    language_onnx: str,
    tokenizer_name: str,
    source: str,
    class_names: list[str],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
):
    """Run dual-tower WeDetect ONNX inference with custom prompts."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lang = ort.InferenceSession(language_onnx, providers=["CPUExecutionProvider"])
    vis = ort.InferenceSession(vision_onnx, providers=["CPUExecutionProvider"])

    input_ids, attention_mask = encode_classes(tokenizer, class_names)
    txt_feats = lang.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]  # 1,K,D

    im0 = cv2.imread(source)
    assert im0 is not None, f"Failed to read image: {source}"
    image, ratio, (pad_x, pad_y), (h0, w0) = letterbox_rgb(im0, imgsz)

    bboxes, scores = vis.run(None, {"image": image, "txt_feats": txt_feats.astype(np.float32)})
    # bboxes: 1,N,4 xyxy in letterbox space; scores: 1,N,K
    boxes = bboxes[0]
    probs = scores[0]
    cls_ids = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    keep = confs >= conf
    boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]
    if boxes.shape[0]:
        keep_idx = nms_xyxy(boxes, confs, iou=iou, max_det=max_det)
        boxes, confs, cls_ids = boxes[keep_idx], confs[keep_idx], cls_ids[keep_idx]
        # Undo letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    # Draw
    out = im0.copy()
    for box, score, cid in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{class_names[int(cid)]} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    save_path = str(Path(source).with_name(Path(source).stem + "_wedetect_onnx.jpg"))
    cv2.imwrite(save_path, out)
    print(f"Detected {len(boxes)} objects with classes={class_names}")
    print(f"Saved -> {save_path}")
    return boxes, confs, cls_ids


def parse_args():
    p = argparse.ArgumentParser(description="WeDetect dual ONNX open-vocabulary inference")
    p.add_argument("--vision", required=True, help="Path to *_vision.onnx")
    p.add_argument("--language", required=True, help="Path to *_language.onnx")
    p.add_argument("--tokenizer", default="xlm-roberta-base", help="HF tokenizer name or local path")
    p.add_argument("--source", required=True, help="Input image path")
    p.add_argument("--classes", default="人,公交车,领带", help="Comma-separated Chinese class names")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    run_dual(args.vision, args.language, args.tokenizer, args.source, names, args.imgsz, args.conf, args.iou)

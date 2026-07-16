---
title: RF-DETR: Real-Time Detection Transformer by Roboflow
comments: true
description: Explore Roboflow RF-DETR with Ultralytics — a DINOv2-based real-time Detection Transformer for object detection, instance segmentation, and pose estimation.
keywords: RF-DETR, Roboflow, Ultralytics, DINOv2, object detection, instance segmentation, pose estimation, transformer, real-time, LWDETR
---

# RF-DETR: Real-Time Detection [Transformer](https://www.ultralytics.com/glossary/transformer) by Roboflow

## Overview

[RF-DETR](https://rfdetr.roboflow.com/latest/) is a real-time Detection Transformer from Roboflow built on a DINOv2 vision-transformer backbone. Ultralytics integrates RF-DETR through the same `Model` / `task_map` engine used by [RT-DETR](rtdetr.md): [train](../modes/train.md), [validate](../modes/val.md), [predict](../modes/predict.md), and [export](../modes/export.md) without PyTorch Lightning. The neural network itself comes from the optional [`rfdetr`](https://github.com/roboflow/rf-detr) package (NAS-style optional backend).

RF-DETR was accepted at ICLR 2026. Core Nano–Large detection models, all segmentation variants, the keypoint preview model, and the training/inference code are Apache 2.0. Detection XLarge / 2XLarge weights require the optional `rfdetr_plus` package and acceptance of the Platform Model License (PML).

### Key Features

- **DINOv2 backbone:** Windowed attention provides strong [accuracy](https://www.ultralytics.com/glossary/accuracy)–latency trade-offs across edge and server deployments.
- **Unified tasks:** One Ultralytics API covers [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), and [pose](../tasks/pose.md) (keypoint preview).
- **Native Ultralytics training:** Hungarian matching (`SetCriterion`) runs on `BaseTrainer` with YOLO-format datasets — not Lightning.
- **Optional `rfdetr` backend:** Ultralytics keeps a thin facade; LWDETR and official criterion live in the Roboflow package.
- **Stride-aware `imgsz`:** Inputs must be divisible by `patch_size × num_windows` (exposed as `model.stride`); Ultralytics rounds automatically via `check_imgsz`.

## Requirements

RF-DETR inside Ultralytics requires **Python ≥ 3.10** and **torch ≥ 2.2**, plus the optional backend:

```bash
pip install ultralytics[rfdetr]
```

Detection XL / 2XL additionally need:

```bash
pip install rfdetr-plus
```

Native Roboflow checkpoints download into `~/.roboflow/models` (override with `RF_HOME`).

## Supported Models

Recommended `imgsz` matches each variant’s official `resolution`. Prefer these sizes when possible; Ultralytics rounds other values up to a multiple of `model.stride` (`patch_size × num_windows`).

| Model                 | Task    | Recommended `imgsz` | Stride (`patch×windows`) | Notes                                                         |
| --------------------- | ------- | ------------------- | ------------------------ | ------------------------------------------------------------- |
| `rfdetr-nano`         | detect  | 384                 | 16×2 = **32**            | Edge / real-time                                              |
| `rfdetr-small`        | detect  | 512                 | 32                       |                                                               |
| `rfdetr-medium`       | detect  | 576                 | 32                       |                                                               |
| `rfdetr-large`        | detect  | 704                 | 32                       | Default large (`rf-detr-large-2026.pth`)                      |
| `rfdetr-seg-nano`     | segment | 312                 | 12×1 = **12**            |                                                               |
| `rfdetr-seg-small`    | segment | 384                 | 12×2 = **24**            |                                                               |
| `rfdetr-seg-medium`   | segment | 432                 | 24                       |                                                               |
| `rfdetr-seg-large`    | segment | 504                 | 24                       |                                                               |
| `rfdetr-seg-xlarge`   | segment | 624                 | 24                       | Apache 2.0                                                    |
| `rfdetr-seg-2xlarge`  | segment | 768                 | 24                       | Apache 2.0 (`rf-detr-seg-xxlarge.pt`)                         |
| `rfdetr-pose-preview` | pose    | 576                 | 24                       | COCO person keypoints preview                                 |
| `rfdetr-xlarge`       | detect  | 700                 | Plus-specific            | Requires `rfdetr_plus` + `accept_platform_model_license=True` |
| `rfdetr-2xlarge`      | detect  | 880                 | Plus-specific            | Requires `rfdetr_plus` + `accept_platform_model_license=True` |

`imgsz` is **not** stored in model YAMLs. It comes from [`default.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml) / [`default-rfdetr.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default-rfdetr.yaml), or from CLI / Python (`imgsz=384`). Default YOLO `imgsz=640` is valid for detect (÷32) but is auto-rounded for seg/pose (÷12 or ÷24).

## Usage Examples

This example provides simple RF-DETR training and inference. For full documentation on these and other [modes](../modes/index.md) see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md) docs pages.

!!! example

    === "Python"

        ```python
        from ultralytics import RFDETR

        # Load a COCO-pretrained RF-DETR-nano model
        model = RFDETR("rfdetr-nano.pt")

        # Display model information (optional)
        model.info()

        # Train on a YOLO-format dataset (uses cfg/default-rfdetr.yaml hyps by default)
        results = model.train(data="coco8.yaml", epochs=50, imgsz=384, batch=4)

        # Run inference
        results = model.predict("path/to/bus.jpg", imgsz=384)

        # Validate
        metrics = model.val(data="coco8.yaml", imgsz=384)

        # Export to ONNX
        path = model.export(format="onnx", imgsz=384)

        # Segmentation
        seg = RFDETR("rfdetr-seg-nano.pt")
        seg.predict("path/to/bus.jpg", imgsz=312)

        # Pose (keypoint preview)
        pose = RFDETR("rfdetr-pose-preview.pt")
        pose.predict("path/to/bus.jpg", imgsz=576)

        # Platform models (PML) — after installing rfdetr-plus
        # xl = RFDETR("rfdetr-xlarge.pt", accept_platform_model_license=True)
        ```

    === "CLI"

        ```bash
        # Predict
        yolo predict model=rfdetr-nano.pt source=path/to/bus.jpg imgsz=384

        # Train from YAML (scale inferred from filename, e.g. rfdetr-nano.yaml → scale=nano)
        yolo train model=rfdetr-nano.yaml data=coco8.yaml epochs=50 imgsz=384

        # Validate / export
        yolo val model=rfdetr-nano.pt data=coco8.yaml imgsz=384
        yolo export model=rfdetr-nano.pt format=onnx imgsz=384

        # Segmentation and pose
        yolo predict model=rfdetr-seg-nano.pt source=path/to/bus.jpg imgsz=312
        yolo predict model=rfdetr-pose-preview.pt source=path/to/bus.jpg imgsz=576
        ```

!!! tip "Training hyperparameters"

    `RFDETRTrainer` defaults to [`ultralytics/cfg/default-rfdetr.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default-rfdetr.yaml), which maps official Roboflow `TrainConfig` / `MODEL_DEFAULTS` onto Ultralytics field names (`box=5.0`, `cls=1.0`, `optimizer=AdamW`, `lr0=1e-4`, `mosaic=0.0`, …). Pass `cfg="default.yaml"` (or a custom YAML) to use YOLO-style hyps instead. Loss gains such as `box` / `cls` / `pose` / `kobj` / `rle` and `max_det` / `mask_ratio` flow into the official `SetCriterion` and model config.

## Supported Tasks and Modes

This table presents the model types, the tasks supported by each model, and the various modes ([Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)) that are supported, indicated by ✅ emojis.

| Model Type           | Pretrained Weights                    | Tasks Supported                              | Inference | Validation | Training | Export |
| -------------------- | ------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| RF-DETR Nano–Large   | `rfdetr-{nano,small,medium,large}.pt` | [Object Detection](../tasks/detect.md)       | ✅        | ✅         | ✅       | ✅     |
| RF-DETR Seg Nano–2XL | `rfdetr-seg-*.pt`                     | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| RF-DETR Pose Preview | `rfdetr-pose-preview.pt`              | [Pose Estimation](../tasks/pose.md)          | ✅        | ✅         | ✅       | ✅     |
| RF-DETR XLarge / 2XL | `rfdetr-{xlarge,2xlarge}.pt`          | [Object Detection](../tasks/detect.md)       | ✅¹       | ✅¹        | ✅¹      | ✅¹    |

¹ Requires `pip install rfdetr-plus` and `accept_platform_model_license=True`.

!!! note "Architecture YAML layout"

    Per-task YAMLs follow the YOLO26 pattern: [`rfdetr.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/rfdetr/rfdetr.yaml) (detect), [`rfdetr-seg.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/rfdetr/rfdetr-seg.yaml), and [`rfdetr-pose.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/rfdetr/rfdetr-pose.yaml). Requesting `rfdetr-nano.yaml` loads `rfdetr.yaml` with `scale=nano` (same idea as `yolo26n.yaml` → `yolo26.yaml` + `scale=n`). Task inference uses `guess_model_task` (filename cues such as `-seg` / `-pose`).

## Architecture Notes

- The network is built by the optional `rfdetr` package via `build_model_from_config` / LWDETR.
- Inference uses Ultralytics `LetterBox(scale_fill=True)` and `/255` (no ImageNet mean/std in the Ultralytics path).
- Training uses Ultralytics `YOLODataset` augmentations driven by `args` / `default-rfdetr.yaml`, not the upstream Albumentations / Kornia pipelines.
- Post-processing reuses Roboflow’s `PostProcess` (scores, boxes, optional masks/keypoints).

## Citations and Acknowledgments

If you use RF-DETR in your research or development work, please cite the [original paper](https://arxiv.org/abs/2511.09554):

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{robinson2026rfdetr,
          title     = {RF-DETR: Real-Time Detection Transformer},
          author    = {Robinson, Isaac and Robicheaux, Peter and Popov, Fedor and Ramanan, Deva and Peri, Neehar},
          booktitle = {International Conference on Learning Representations (ICLR)},
          year      = {2026},
          url       = {https://arxiv.org/abs/2511.09554}
        }
        ```

We would like to acknowledge [Roboflow](https://roboflow.com/) for creating and maintaining RF-DETR. See also the [RF-DETR documentation](https://rfdetr.roboflow.com/latest/) and [GitHub repository](https://github.com/roboflow/rf-detr).

## FAQ

### What is RF-DETR and how does it differ from RT-DETR?

RF-DETR is Roboflow’s real-time Detection Transformer with a DINOv2 backbone. [RT-DETR](rtdetr.md) is Baidu’s PaddlePaddle-based DETR variant. In Ultralytics both share the same `Model` / trainer / predictor pattern, but RF-DETR loads LWDETR from the optional `rfdetr` package, supports detection **and** segmentation **and** pose, and enforces `imgsz` multiples of `patch_size × num_windows`.

### How do I install RF-DETR support in Ultralytics?

Install the optional extra (Python ≥ 3.10, torch ≥ 2.2):

```bash
pip install ultralytics[rfdetr]
```

For detection XLarge / 2XLarge, also install `rfdetr-plus` and pass `accept_platform_model_license=True` when constructing `RFDETR`.

### Why must `imgsz` be divisible by 32 (or 12 / 24)?

RF-DETR’s ViT backbone uses windowed attention. The product `patch_size × num_windows` is the model stride (32 for detect nano–large, 12 for seg-nano, 24 for other seg/pose variants). Ultralytics exposes this as `model.stride` and rounds `imgsz` with `check_imgsz` so training and inference stay valid.

### Which config file controls RF-DETR training hyperparameters?

`RFDETRTrainer` uses [`default-rfdetr.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default-rfdetr.yaml) by default (official Roboflow-mapped hyps). Override with `model.train(cfg="default.yaml", ...)` or any custom YAML that uses Ultralytics field names. Fields such as `box`, `cls`, `max_det`, and `mask_ratio` are mapped into the official criterion and model config.

### Can I train RF-DETR on a YOLO-format dataset?

Yes. Pass a standard Ultralytics data YAML (`coco8.yaml`, custom datasets, etc.) to `model.train(data=...)`. Labels use the same YOLO layout as other Ultralytics detectors; the trainer builds `RFDETRDataset` (a `YOLODataset` subclass) and applies Ultralytics augmentations from `args`.

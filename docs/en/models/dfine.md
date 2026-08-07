---
title: D-FINE: Fine-grained Distribution Refinement Detection Transformer
comments: true
description: Explore D-FINE with Ultralytics — a real-time DETR detector with Fine-grained Distribution Refinement (FDR) for high-accuracy bounding box regression.
keywords: D-FINE, DFINE, DETR, FDR, GO-LSD, Ultralytics, object detection, transformer, HGNetv2, real-time, Peterande
---

# D-FINE: Fine-grained Distribution Refinement Detection [Transformer](https://www.ultralytics.com/glossary/transformer)

!!! tip "中文文档 / Chinese docs"

    简体中文完整说明见仓库内 [`docs/zh/models/dfine.md`](../../zh/models/dfine.md)。

## Overview

[D-FINE](https://github.com/Peterande/D-FINE) (paper: [*D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement*](https://arxiv.org/abs/2410.13842)) is a real-time DETR-style [object detector](https://www.ultralytics.com/glossary/object-detection) that reframes bounding-box regression as **Fine-grained Distribution Refinement (FDR)** and adds **Global Optimal Localization Self-Distillation (GO-LSD)**. Ultralytics integrates D-FINE natively—similar to [RT-DETR](rtdetr.md)—with YAML topologies, `DFINEDecoder`, YOLO-format datasets, and the standard [train](../modes/train.md) / [val](../modes/val.md) / [predict](../modes/predict.md) / [export](../modes/export.md) workflow.

Official pretrained checkpoints use an HGNetv2 backbone plus HybridEncoder (`RepNCSPELAN4` / `SCDown`) and the D-FINE transformer decoder. Converted Ultralytics `.pt` weights match official fp32 outputs under the same preprocessing (640×640 square resize + `/255`, no ImageNet normalize).

### Key Features

- **Fine-grained Distribution Refinement (FDR):** Box edges are refined as discrete distributions (`reg_max`) instead of a single L1/GIoU offset, improving localization on blurred or crowded scenes.
- **GO-LSD:** Self-distillation between decoder layers improves localization without extra inference cost.
- **NMS-free DETR head:** End-to-end set prediction; Ultralytics postprocess uses confidence filtering (same pattern as RT-DETR).
- **Native Ultralytics stack:** `DFINE` facade, `DFINEDetectionModel`, YOLO `data=*.yaml` training, and ONNX/TensorRT-friendly export.
- **Scale coverage:** Nano → XLarge (`dfine-n/s/m/l/x`) with optional Objects365 / Objects365→COCO checkpoints via weight conversion.

## Official COCO Performance

Metrics below are from the [official D-FINE Model Zoo](https://github.com/Peterande/D-FINE) (COCO val2017, TensorRT latency on T4). Ultralytics preserves architecture and weight parity after conversion; runtime FPS depends on your export backend.

| Model     | AP<sup>val</sup> | Params | Latency (T4) | GFLOPs |
| --------- | ---------------- | ------ | ------------ | ------ |
| D-FINE-N  | 42.8             | 4M     | 2.12 ms      | 7      |
| D-FINE-S  | 48.5             | 10M    | 3.49 ms      | 25     |
| D-FINE-M  | 52.3             | 19M    | 5.62 ms      | 57     |
| D-FINE-L  | 54.0             | 31M    | 8.07 ms      | 91     |
| D-FINE-X  | 55.8             | 62M    | 12.89 ms     | 202    |

Objects365-pretrained and Objects365→COCO finetunes (higher COCO AP) are also supported after conversion; see [Weight conversion](#weight-conversion).

## Usage Examples

!!! example

    === "Python"

        ```python
        from ultralytics import DFINE

        # Load a converted Ultralytics checkpoint (or a YAML for random init)
        model = DFINE("dfine-l.pt")  # or "dfine-l.yaml"

        # Display model information (optional)
        model.info()

        # Train on a YOLO-format dataset
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Validate
        metrics = model.val(data="coco8.yaml")

        # Predict (square scale-fill letterbox, no NMS)
        results = model("path/to/bus.jpg")

        # Export
        path = model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        # Train
        yolo detect train model=dfine-l.pt data=coco8.yaml epochs=100 imgsz=640

        # Predict (stem "dfine" selects the DFINE class)
        yolo predict model=dfine-l.pt source=path/to/bus.jpg

        # Validate / export
        yolo val model=dfine-l.pt data=coco8.yaml
        yolo export model=dfine-l.pt format=onnx
        ```

!!! tip "Preprocessing"

    Match official D-FINE inference: square resize to `imgsz` (default 640) with `LetterBox(scale_fill=True)` and divide by 255. Do **not** apply ImageNet mean/std on the Ultralytics path.

## Weight Conversion

Official releases are `.pth` (keys under `model` / `ema`). Convert them to Ultralytics `.pt` with the bundled script (fp32 by default for numerical parity):

```bash
# From the ultralytics repository root
python tools/convert_dfine_weights.py \
  --weights path/to/dfine_l_coco.pth \
  --cfg dfine-l.yaml \
  --out dfine-l.pt

# Objects365 checkpoints (nc=366 inferred from the official head)
python tools/convert_dfine_weights.py \
  --weights path/to/dfine_l_obj365.pth \
  --cfg dfine-l.yaml \
  --out dfine-l-obj365.pt
```

| Flag        | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| `--weights` | Official `.pth` path                                                        |
| `--cfg`     | Ultralytics YAML (`dfine-n/s/m/l/x.yaml`)                                   |
| `--out`     | Output `.pt` path                                                           |
| `--nc`      | Optional class count override (auto-inferred from `enc_score_head` if omit) |
| `--fp16`    | Save half precision (not recommended when checking parity)                  |
| `--dry-run` | Print key-map only                                                          |

YAML notes:

- **N/S/M** enable Learnable Affine Blocks (`use_lab=True`) to match official HGNetv2.
- **L/X** set `freeze_norm: true` so backbone BatchNorm matches official `FrozenBatchNorm2d`.

## Supported Tasks and Modes

| Model Type | Config / Weights                         | Tasks Supported                        | Inference | Validation | Training | Export |
| ---------- | ---------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| D-FINE-N   | `dfine-n.yaml` / `dfine-n.pt`            | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| D-FINE-S   | `dfine-s.yaml` / `dfine-s.pt`            | Object Detection                         | ✅        | ✅         | ✅       | ✅     |
| D-FINE-M   | `dfine-m.yaml` / `dfine-m.pt`            | Object Detection                         | ✅        | ✅         | ✅       | ✅     |
| D-FINE-L   | `dfine-l.yaml` / `dfine-l.pt`            | Object Detection                         | ✅        | ✅         | ✅       | ✅     |
| D-FINE-X   | `dfine-x.yaml` / `dfine-x.pt`            | Object Detection                         | ✅        | ✅         | ✅       | ✅     |

Architecture YAMLs live under `ultralytics/cfg/models/dfine/`.

## Architecture Notes

- **Backbone:** HGNetv2 stages exposed as Ultralytics `HGStem` / `HGBlock` / `DWConv` layers.
- **Encoder:** Official HybridEncoder blocks are `DFINERepNCSPELAN4` / `DFINESCDown` (not YOLO `RepNCSPELAN4` / `SCDown`).
- **Decoder:** `DFINEDecoder` implements FDR (`Integral`, LQE, Gate) and CDN training queries.
- **Loss:** `DFINEDetectionLoss` (VFL + L1/GIoU + FGL/DDF + GO-LSD).
- **Train/val/predict:** Share DETR-style batching with RT-DETR (`rect=False`, scale-fill letterbox, no NMS).

## Citations and Acknowledgments

If you use D-FINE in research or products, please cite the [original paper](https://arxiv.org/abs/2410.13842):

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{peng2024dfine,
              title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
              author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
              year={2024},
              eprint={2410.13842},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Thanks to the [D-FINE authors](https://github.com/Peterande/D-FINE) for the open-source implementation and pretrained weights.

## FAQ

### How is D-FINE different from RT-DETR in Ultralytics?

Both are NMS-free DETR detectors with a similar Ultralytics facade. D-FINE replaces plain box regression with FDR distributions and uses GO-LSD / FGL-DDF losses. Encoder blocks must be the official `DFINERepNCSPELAN4` / `DFINESCDown` variants for weight parity.

### Where do I get `.pt` weights?

Convert official `.pth` files with `tools/convert_dfine_weights.py` (see [Weight conversion](#weight-conversion)). Keep fp32 when validating numerical parity against the official model.

### Can I train on a custom YOLO dataset?

Yes. Pass any Ultralytics detection data YAML to `model.train(data=...)`. Labels use the standard YOLO format; the trainer uses the same DETR-style dataset path as RT-DETR (`rect=False`).

### Why do L/X configs set `freeze_norm: true`?

Official L/X HGNetv2 builds use `FrozenBatchNorm2d` (`freeze_norm=True`). Ultralytics mirrors that fused `rsqrt` path so converted weights stay bit-aligned with the official forward pass.

### Does `max_det` increase the number of D-FINE queries?

No. Like RT-DETR, `max_det` only caps returned predictions. The decoder still emits a fixed number of queries (default 300). Raise `nq` in the YAML and retrain if you need more queries per image.

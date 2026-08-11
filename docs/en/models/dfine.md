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

Official pretrained checkpoints use an HGNetv2 backbone plus HybridEncoder (`RepNCSPELAN4` / `SCDown`) and the D-FINE transformer decoder. Ultralytics `.pt` weights in this repository match official fp32 outputs under the same preprocessing (640×640 square resize + `/255`, no ImageNet normalize).

### Key Features

- **Fine-grained Distribution Refinement (FDR):** Box edges are refined as discrete distributions (`reg_max`) instead of a single L1/GIoU offset, improving localization on blurred or crowded scenes.
- **GO-LSD:** Self-distillation between decoder layers improves localization without extra inference cost.
- **NMS-free DETR head:** End-to-end set prediction; Ultralytics postprocess uses confidence filtering (same pattern as RT-DETR).
- **Native Ultralytics stack:** `DFINE` facade, `DFINEDetectionModel`, YOLO `data=*.yaml` training, and ONNX/TensorRT-friendly export.
- **Scale coverage:** Nano → XLarge (`dfine-n/s/m/l/x`) with COCO, Objects365, and Objects365→COCO pretrained checkpoints.

## Model Zoo

Metrics and latency are from the [official D-FINE Model Zoo](https://github.com/Peterande/D-FINE) (COCO val2017; T4, `batch_size=1`, fp16, TensorRT). Ultralytics `.pt` weights are published on [Release v2.0.0](https://github.com/wellybreeze/ultralytics/releases/tag/v2.0.0).

### COCO

| Model | Dataset | AP<sup>val</sup> | Params | Latency (ms) | GFLOPs | Config | Weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D-FINE-N | COCO | **42.8** | 4M | 2.12ms | 7 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-n.yaml) | [42.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-n.pt) |
| D-FINE-S | COCO | **48.5** | 10M | 3.49ms | 25 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [48.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s.pt) |
| D-FINE-M | COCO | **52.3** | 19M | 5.62ms | 57 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [52.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m.pt) |
| D-FINE-L | COCO | **54.0** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [54.0](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt) |
| D-FINE-X | COCO | **55.8** | 62M | 12.89ms | 202 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [55.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x.pt) |

### Objects365+COCO

| Model | Dataset | AP<sup>val</sup> | Params | Latency (ms) | GFLOPs | Config | Weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D-FINE-S | Objects365+COCO | **50.7** | 10M | 3.49ms | 25 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [50.7](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj2coco.pt) |
| D-FINE-M | Objects365+COCO | **55.1** | 19M | 5.62ms | 57 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [55.1](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj2coco.pt) |
| D-FINE-L | Objects365+COCO | **57.3** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [57.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj2coco-e25.pt) |
| D-FINE-X | Objects365+COCO | **59.3** | 62M | 12.89ms | 202 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [59.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj2coco.pt) |

**We strongly recommend fine-tuning from Objects365-pretrained weights:**

⚠️ Note: These checkpoints help on complex scenes. For very simple class sets they may overfit and underperform.

<details>
<summary><strong>🔥 Objects365 pretrained models (best generalization)</strong></summary>

| Model | Dataset | AP<sup>val</sup> | AP<sup>5000</sup> | Params | Latency (ms) | GFLOPs | Config | Weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D-FINE-S | Objects365 | **31.0** | **30.5** | 10M | 3.49ms | 25 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [30.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj365.pt) |
| D-FINE-M | Objects365 | **38.6** | **37.4** | 19M | 5.62ms | 57 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [37.4](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj365.pt) |
| D-FINE-L | Objects365 | - | **40.6** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [40.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365.pt) |
| D-FINE-L (E25) | Objects365 | **44.7** | **42.6** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [42.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365-e25.pt) |
| D-FINE-X | Objects365 | **49.5** | **46.5** | 62M | 12.89ms | 202 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [46.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj365.pt) |

- **E25**: Official retrain extended to 25 epochs.
- **AP<sup>val</sup>** is evaluated on the full *Objects365* validation set.
- **AP<sup>5000</sup>** is evaluated on the first 5000 samples of the *Objects365* validation set.

</details>

**Notes:**

- **AP<sup>val</sup>** (COCO / Objects365+COCO tables) is evaluated on *MSCOCO val2017*.
- **Latency** is measured on a single T4 GPU with $batch\_size = 1$, fp16, and TensorRT (see official D-FINE notes).
- **Objects365+COCO** means Objects365-pretrained weights fine-tuned on *COCO*.
- YAML notes: **N/S/M** enable Learnable Affine Blocks (`use_lab=True`); **L/X** set `freeze_norm: true` to match official `FrozenBatchNorm2d`.

## Usage Examples

!!! example

    === "Python"

        ```python
        from ultralytics import DFINE

        # Load Ultralytics pretrained weights from Release v2.0.0 (local path or YAML also OK)
        model = DFINE("https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt")
        # model = DFINE("dfine-l.yaml")  # random init

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
        # Download weights, then train / predict (stem "dfine" selects the DFINE class)
        wget https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt
        yolo detect train model=dfine-l.pt data=coco8.yaml epochs=100 imgsz=640
        yolo predict model=dfine-l.pt source=path/to/bus.jpg
        yolo val model=dfine-l.pt data=coco8.yaml
        yolo export model=dfine-l.pt format=onnx
        ```

!!! tip "Preprocessing"

    Match official D-FINE inference: square resize to `imgsz` (default 640) with `LetterBox(scale_fill=True)` and divide by 255. Do **not** apply ImageNet mean/std on the Ultralytics path.

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

Download from [Release v2.0.0](https://github.com/wellybreeze/ultralytics/releases/tag/v2.0.0), or use the links in the [Model Zoo](#model-zoo) tables above. Metrics match the official zoo; weights are Ultralytics-format checkpoints.

### Can I train on a custom YOLO dataset?

Yes. Pass any Ultralytics detection data YAML to `model.train(data=...)`. Labels use the standard YOLO format; the trainer uses the same DETR-style dataset path as RT-DETR (`rect=False`).

### Why do L/X configs set `freeze_norm: true`?

Official L/X HGNetv2 builds use `FrozenBatchNorm2d` (`freeze_norm=True`). Ultralytics mirrors that fused `rsqrt` path so converted weights stay bit-aligned with the official forward pass.

### Does `max_det` increase the number of D-FINE queries?

No. Like RT-DETR, `max_det` only caps returned predictions. The decoder still emits a fixed number of queries (default 300). Raise `nq` in the YAML and retrain if you need more queries per image.

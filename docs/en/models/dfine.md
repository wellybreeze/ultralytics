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

[D-FINE](https://github.com/Peterande/D-FINE) (paper: [_D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement_](https://arxiv.org/abs/2410.13842)) is a real-time DETR-style [object detector](https://www.ultralytics.com/glossary/object-detection) that reframes bounding-box regression as **Fine-grained Distribution Refinement (FDR)** and adds **Global Optimal Localization Self-Distillation (GO-LSD)**. Ultralytics integrates D-FINE natively—similar to [RT-DETR](rtdetr.md)—with YAML topologies, `DFINEDecoder`, YOLO-format datasets, and the standard [train](../modes/train.md) / [val](../modes/val.md) / [predict](../modes/predict.md) / [export](../modes/export.md) workflow.

Official pretrained checkpoints use an HGNetv2 backbone plus HybridEncoder (`RepNCSPELAN4` / `SCDown`) and the D-FINE transformer decoder. Ultralytics `.pt` weights in this repository match official fp32 outputs under the same preprocessing (640×640 square resize + `/255`, no ImageNet normalize).

### Key Features

- **Fine-grained Distribution Refinement (FDR):** Box edges are refined as discrete distributions (`reg_max`) instead of a single L1/GIoU offset, improving localization on blurred or crowded scenes.
- **GO-LSD:** Self-distillation between decoder layers improves localization without extra inference cost.
- **NMS-free DETR head:** End-to-end set prediction; Ultralytics postprocess uses confidence filtering (same pattern as RT-DETR).
- **Native Ultralytics stack:** `DFINE` facade, `DFINEDetectionModel`, YOLO `data=*.yaml` training, and ONNX/TensorRT-friendly export. Objects365 weights can also serve as a [WeDetect](wedetect.md) pseudo-label teacher.
- **Scale coverage:** Nano → XLarge (`dfine-n/s/m/l/x`) with COCO, Objects365, and Objects365→COCO pretrained checkpoints.

## Model Zoo

Metrics and latency are from the [official D-FINE Model Zoo](https://github.com/Peterande/D-FINE) (COCO val2017; T4, `batch_size=1`, fp16, TensorRT). Ultralytics `.pt` weights are published on [Release v2.0.0](https://github.com/wellybreeze/ultralytics/releases/tag/v2.0.0).

### COCO

|  Model   | Dataset | AP<sup>val</sup> | Params | Latency (ms) | GFLOPs |                           Config                           |                                        Weights                                         |
| :------: | :-----: | :--------------: | :----: | :----------: | :----: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| D-FINE-N |  COCO   |     **42.8**     |   4M   |    2.12ms    |   7    | [yaml](../../../ultralytics/cfg/models/dfine/dfine-n.yaml) | [42.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-n.pt) |
| D-FINE-S |  COCO   |     **48.5**     |  10M   |    3.49ms    |   25   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [48.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s.pt) |
| D-FINE-M |  COCO   |     **52.3**     |  19M   |    5.62ms    |   57   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [52.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m.pt) |
| D-FINE-L |  COCO   |     **54.0**     |  31M   |    8.07ms    |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [54.0](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt) |
| D-FINE-X |  COCO   |     **55.8**     |  62M   |   12.89ms    |  202   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [55.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x.pt) |

### Objects365+COCO

|  Model   |     Dataset     | AP<sup>val</sup> | Params | Latency (ms) | GFLOPs |                           Config                           |                                               Weights                                               |
| :------: | :-------------: | :--------------: | :----: | :----------: | :----: | :--------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| D-FINE-S | Objects365+COCO |     **50.7**     |  10M   |    3.49ms    |   25   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) |   [50.7](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj2coco.pt)   |
| D-FINE-M | Objects365+COCO |     **55.1**     |  19M   |    5.62ms    |   57   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) |   [55.1](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj2coco.pt)   |
| D-FINE-L | Objects365+COCO |     **57.3**     |  31M   |    8.07ms    |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [57.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj2coco-e25.pt) |
| D-FINE-X | Objects365+COCO |     **59.3**     |  62M   |   12.89ms    |  202   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) |   [59.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj2coco.pt)   |

**We strongly recommend fine-tuning from Objects365-pretrained weights:**

⚠️ Note: These checkpoints help on complex scenes. For very simple class sets they may overfit and underperform.

<details>
<summary><strong>🔥 Objects365 pretrained models (best generalization)</strong></summary>

|     Model      |  Dataset   | AP<sup>val</sup> | AP<sup>5000</sup> | Params | Latency (ms) | GFLOPs |                           Config                           |                                              Weights                                              |
| :------------: | :--------: | :--------------: | :---------------: | :----: | :----------: | :----: | :--------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
|    D-FINE-S    | Objects365 |     **31.0**     |     **30.5**      |  10M   |    3.49ms    |   25   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) |   [30.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj365.pt)   |
|    D-FINE-M    | Objects365 |     **38.6**     |     **37.4**      |  19M   |    5.62ms    |   57   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) |   [37.4](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj365.pt)   |
|    D-FINE-L    | Objects365 |        -         |     **40.6**      |  31M   |    8.07ms    |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) |   [40.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365.pt)   |
| D-FINE-L (E25) | Objects365 |     **44.7**     |     **42.6**      |  31M   |    8.07ms    |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [42.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365-e25.pt) |
|    D-FINE-X    | Objects365 |     **49.5**     |     **46.5**      |  62M   |   12.89ms    |  202   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) |   [46.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj365.pt)   |

- **E25**: Official retrain extended to 25 epochs.
- **AP<sup>val</sup>** is evaluated on the full _Objects365_ validation set.
- **AP<sup>5000</sup>** is evaluated on the first 5000 samples of the _Objects365_ validation set.

</details>

**Notes:**

- **AP<sup>val</sup>** (COCO / Objects365+COCO tables) is evaluated on _MSCOCO val2017_.
- **Latency** is measured on a single T4 GPU with $batch\_size = 1$, fp16, and TensorRT (see official D-FINE notes).
- **Objects365+COCO** means Objects365-pretrained weights fine-tuned on _COCO_.
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

    **Predict / teacher inference** matches official D-FINE: square stretch to `imgsz` with `LetterBox(scale_fill=True)` and `/255`. Do **not** apply ImageNet mean/std.

    **Val** uses padded `LetterBox(scale_fill=False, scaleup=False)` (RT-DETR dataset path). **Train** uses YOLO mosaic / `RandomPerspective` onto a square `imgsz`, not official scale-fill. Compare val mAP to the official zoo with that difference in mind.

## Supported Tasks and Modes

| Model Type | Config / Weights              | Tasks Supported                        | Inference | Validation | Training | Export |
| ---------- | ----------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| D-FINE-N   | `dfine-n.yaml` / `dfine-n.pt` | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| D-FINE-S   | `dfine-s.yaml` / `dfine-s.pt` | Object Detection                       | ✅        | ✅         | ✅       | ✅     |
| D-FINE-M   | `dfine-m.yaml` / `dfine-m.pt` | Object Detection                       | ✅        | ✅         | ✅       | ✅     |
| D-FINE-L   | `dfine-l.yaml` / `dfine-l.pt` | Object Detection                       | ✅        | ✅         | ✅       | ✅     |
| D-FINE-X   | `dfine-x.yaml` / `dfine-x.pt` | Object Detection                       | ✅        | ✅         | ✅       | ✅     |

Architecture YAMLs live under `ultralytics/cfg/models/dfine/`.

## Full pipeline

```text
YOLO data YAML  →  DFINE.train()  →  best.pt
        ↓
DFINE.val() / DFINE.predict() / DFINE.export()
        ↓
optional: WeDetect pseudo_label_model (Objects365 teacher)
```

`YOLO("dfine-*.pt")` morphs into `DFINE` when the head is `DFINEDecoder`. Prefer `from ultralytics import DFINE`. Requires `torch>=1.11`.

### Train

`DFINETrainer` reuses the RT-DETR loop (`rect=False`, DETR-style batch dict for CDN). Loss display names are `giou_loss`, `cls_loss`, `l1_loss` (FGL/DDF are in the total loss but not shown). The criterion is `DFINEDetectionLoss`:

| Term   | Default weight  | Role                                  |
| ------ | --------------- | ------------------------------------- |
| VFL    | `loss_vfl=1.0`  | Classification                        |
| L1 box | `loss_bbox=5.0` | Box regression                        |
| GIoU   | `loss_giou=2.0` | Box overlap                           |
| FGL    | `loss_fgl=0.15` | Fine-grained localization (FDR)       |
| DDF    | `loss_ddf=1.5`  | Decoupled distillation focal (GO-LSD) |

```python
from ultralytics import DFINE

# Fine-tune from Objects365 (recommended for custom / crowded scenes)
model = DFINE("pretrained_weights/dfine-x-obj365.pt")
model.train(data="coco8.yaml", epochs=50, imgsz=640, batch=4, device=0)
```

YAML knobs:

- **N/S/M:** Learnable Affine Blocks (`use_lab=True` on `HGStem` / `HGBlock`).
- **L/X:** `freeze_norm: true` → FrozenBatchNorm2d in the backbone (official parity).
- Decoder: `nq=300` queries, `reg_max=32` FDR bins. Changing `nc` reinitializes the class head; Objects365 checkpoints use `nc=366`.

`fuse()` follows official `DFINE.deploy()` (decoder / selected lateral Convs), not YOLO Conv+BN fuse of the whole backbone. AMP can produce NaN / matcher failures (same RT-DETR note); `F.grid_sample` does not support `deterministic=True`.

### Val and predict

Preprocess **differs by mode**:

| Mode                       | Resize                                                                     |
| -------------------------- | -------------------------------------------------------------------------- |
| Predict / WeDetect teacher | `LetterBox(..., scale_fill=True)` — stretch to square (official inference) |
| Val                        | `LetterBox(..., scale_fill=False, scaleup=False)` — keep aspect ratio, pad |
| Train                      | Mosaic + `RandomPerspective` to square `imgsz`                             |

All paths `/255` with no ImageNet mean/std. Postprocess is confidence filtering only (no NMS), then scale boxes. `max_det` caps returned rows; the decoder still emits `nq` queries (default 300). Export clamps `max_det` to `num_queries` and **forces `nms=False`**.

**Non-640 `imgsz`:** `DFINEDecoder` rebuilds log-anchors when feature-map `spatial_shapes` change (`self.shapes` cache; N YAML may bake `eval_spatial_size=[640,640]` which the first forward overwrites). Do not assume a frozen 640-only anchor buffer.

### Class names

| Checkpoint                     | Head `nc` | `names`                                                  |
| ------------------------------ | --------- | -------------------------------------------------------- |
| COCO `dfine-*.pt`              | 80        | COCO 80, ids `0..79`                                     |
| Objects365 `dfine-*-obj365.pt` | 366       | index `0` = `background`; dataset class `i` → head `i+1` |

`ensure_dfine_class_names` replaces placeholder `{i: str(i)}` maps after load. When D-FINE is a **WeDetect teacher**, English `pseudo_label_classes` (for example `Objects365.yaml`) are matched to these head names; YOLO local ids are not used as head indices.

### Export

Same exporter path as RT-DETR (`DFINEDecoder` in `_DETR_DECODERS`). ONNX requires **opset ≥ 16**. TensorFlow export needs opset 16–19. DETR heads force **`nms=False`**. CoreML does not support `dynamic=True` for this family. After `fuse()`, run `export(format="onnx")` or `engine` as usual.

```python
from ultralytics import DFINE

DFINE("dfine-l.pt").export(format="onnx", imgsz=640, opset=17)
```

### As a WeDetect pseudo-label teacher

Set on a WeDetect **train subset** YAML (see [WeDetect](wedetect.md)):

```yaml
pseudo_label: true
pseudo_label_model: ./pretrained_weights/dfine-x-obj365.pt
pseudo_label_classes: Objects365.yaml
pseudo_label_class_texts: texts/objects365_zh_class_texts.json
pseudo_label_conf: 0.2
pseudo_label_imgsz: 640 # independent of WeDetect train imgsz
```

Teacher inference uses `DFINE.predict` (scale-fill, no NMS). `pseudo_label_imgsz` defaults to 640 even if WeDetect trains at 1280. If the teacher OOMs **after** some images are already committed to cache, auto batch-halving is not applied — lower `pseudo_label_batch` and re-run.

## Citations and Acknowledgments

- **Backbone:** HGNetv2 stages exposed as Ultralytics `HGStem` / `HGBlock` / `DWConv` layers.
- **Encoder:** Official HybridEncoder blocks are `DFINERepNCSPELAN4` / `DFINESCDown` (not YOLO `RepNCSPELAN4` / `SCDown`).
- **Decoder:** `DFINEDecoder` implements FDR (`Integral`, LQE, Gate) and CDN training queries.
- **Loss:** `DFINEDetectionLoss` (VFL + L1/GIoU + FGL/DDF + GO-LSD).
- **Train/val/predict:** Share DETR-style batching with RT-DETR (`rect=False`, no NMS). **Predict** uses scale-fill letterbox; **val** uses padded letterbox; **train** uses mosaic / affine. Anchors rebuild when `imgsz` changes.
- **BN:** Ultralytics resets `BatchNorm2d` to official `eps=1e-5`, `momentum=0.1` (not YOLO's `1e-3` / `0.03`).

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

### Can I run inference at 1280 after training at 640?

Yes. Decoder anchors are rebuilt from current feature-map sizes. The same path is used when WeDetect calls D-FINE as a teacher with `pseudo_label_imgsz=1280`.

### How do Objects365 class indices work?

Official Obj365 weights use 366 logits: background at 0, then the 365 dataset names shifted by +1. `DFINE("dfine-x-obj365.pt").names[0]` is `"background"`. Filter with `classes=` using those head indices, not 0-based Objects365 yaml ids.

### Can I use D-FINE as a WeDetect pseudo-label teacher?

Yes. Point `pseudo_label_model` at a D-FINE `.pt` and set `pseudo_label_classes` to `Objects365.yaml` (or COCO names for COCO checkpoints). See [WeDetect](wedetect.md) and the [OV fine-tune guide](../guides/wedetect-ov-finetune.md).

### Does `max_det` increase the number of D-FINE queries?

No. Like RT-DETR, `max_det` only caps returned predictions. The decoder still emits a fixed number of queries (default 300). Raise `nq` in the YAML and retrain if you need more queries per image.

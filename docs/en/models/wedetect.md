---
title: WeDetect Open-Vocabulary Detection
comments: true
description: Train, validate, predict, and export WeDetect — ConvNeXt + XLM-RoBERTa open-vocabulary detection with mixed datasets, pseudo labels, and dual ONNX/TensorRT export.
keywords: WeDetect, open-vocabulary, XLM-RoBERTa, ConvNeXt, Ultralytics, object detection, class_texts, pseudo labels, D-FINE teacher, dual export
---

# WeDetect: Open-Vocabulary Detection

!!! tip "中文文档 / Chinese docs"

    简体中文模型页见 [`docs/zh/models/wedetect.md`](../../zh/models/wedetect.md)。开放词汇微调全流程见 [`docs/zh/guides/wedetect-ov-finetune.md`](../../zh/guides/wedetect-ov-finetune.md)。

[WeDetect](https://github.com/WeDetect/WeDetectPT) is an open-vocabulary detector: a **ConvNeXt** vision backbone plus an **XLM-RoBERTa** text tower. Category identity is the prompt string, not a fixed class index, so mixed YOLO subsets can keep local `class_id` schemes. Ultralytics wires it as `WeDetect` with the usual [train](../modes/train.md) / [val](../modes/val.md) / [predict](../modes/predict.md) / [export](../modes/export.md) modes.

Load a checkpoint whose stem contains `wedetect` with either `WeDetect(...)` or `YOLO(...)` — `YOLO()` morphs into `WeDetect` automatically. **WeDetect-Uni** weights must be loaded with `WeDetectUni(...)`; `YOLO("wedetect-uni-*.pt")` still becomes `WeDetect`.

## Architecture

| Piece                   | Implementation                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| Vision backbone         | ConvNeXt (`tiny` / `base` / `large` / `xlarge`)                                             |
| Neck                    | CSPRepBiFPAN                                                                                |
| Head                    | `WeDetectDetect` (contrastive region–text scoring, embed dim 768)                           |
| Text tower              | XLM-RoBERTa base (`xlm-roberta:base`)                                                       |
| Model class             | `WeDetectModel` in `ultralytics/nn/tasks.py` (built in code, not `parse_model` layer lists) |
| Trainer / val / predict | `WeDetectTrainer`, `WeDetectValidator`, `WeDetectPredictor`                                 |

YAML configs live under `ultralytics/cfg/models/wedetect/` (`wedetect-tiny.yaml`, `wedetect-base.yaml`, `wedetect-large.yaml`, `wedetect-xlarge.yaml`). Prompt-free **WeDetect-Uni** variants (`wedetect-uni-*.yaml`) replace the live LM with learnable embeddings (`WeDetectUni`).

Two fine-tune modes:

- **Open-vocabulary (OV):** `freeze_text_encoder=False` (use `ultralytics/cfg/wedetect_finetune.yaml`). The LM stays online and updates at `lr0 * text_lr_mult`.
- **Close-set:** `freeze_text_encoder=True` (package `default.yaml`) or `close_set=True`. Embeddings are cached; the LM is not updated.

## Pipeline

```text
YOLO labels + class_texts JSON
        ↓
optional teacher pseudo labels (SAM3 / YOLO / WeDetect / D-FINE)
        ↓
WeDetect.train(cfg=wedetect_finetune.yaml)
        ↓
best.pt  (vision + text_model_weights)
        ↓
export_mode=dual  →  *_vision.* + *_language.*
        ↓
set_classes([...]).predict(...)
```

### Train

```python
from ultralytics import WeDetect

model = WeDetect("pretrained_weights/wedetect_base.pt")
model.train(
    data="ultralytics/cfg/datasets/wedetect_coco.yaml",
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    freeze_text_encoder=False,
    epochs=12,
    imgsz=640,
    device=0,
)
```

CLI:

```bash
yolo cfg=ultralytics/cfg/wedetect_finetune.yaml \
  model=pretrained_weights/wedetect_base.pt \
  data=ultralytics/cfg/datasets/wedetect_coco.yaml \
  device=0
```

`WeDetectTrainer.get_dataset()` loads a single YAML or mixed `train.yolo_data` / `grounding_data`, then runs `maybe_build_pseudo_labels` **before** the train dataloader. Checkpoints store `text_model_weights` so export and later `set_classes` reuse the updated LM.

First-epoch CUDA OOM on a single GPU halves `batch` (max 3 retries) and rebuilds the pipeline. Set `batch` to a size that fits (for example `2` at `imgsz=1280`) to skip those rebuilds.

### Val

- **Single YAML:** encode the first `nc` rows of `class_texts` (or `names`) and score mAP as usual.
- **Mixed `val.yolo_data`:** every epoch (and `final_eval`) switches `nc` / `names` / `class_texts` and rebuilds the dataloader. LVIS prefers `minival` when present.

`results.csv` columns:

| Column                            | Meaning                                                                   |
| --------------------------------- | ------------------------------------------------------------------------- |
| `metrics/mAP50-95(B)` (no prefix) | Copy of the **first** val set (YAML order), for default Ultralytics plots |
| `<dataset>/metrics/...`           | That subset's own metrics                                                 |
| `fitness` (no prefix)             | **Weighted average** of per-set mAP50-95 — this selects `best.pt`         |
| `<dataset>/fitness`               | That subset's mAP50-95                                                    |

Optional mixed-val knobs (dataset YAML or `default.yaml`):

| Key                                    | Default        | Role                                                            |
| -------------------------------------- | -------------- | --------------------------------------------------------------- |
| `val_fitness_weights`                  | equal share    | Epoch-1 weights (same order as `val.yolo_data`)                 |
| `val_fitness_dynamic`                  | `False`        | Epoch 2+ reweight from previous mAP50-95 gaps                   |
| `val_fitness_lvis_target_mult`         | `2.0`          | When a val set is LVIS, its target = `mult ×` mean customer mAP |
| `val_fitness_dynamic_ema`              | `0.5`          | EMA on dynamic weights                                          |
| `val_fitness_weight_clip_min` / `_max` | `0.5` / `20.0` | Clip raw `target/mAP` before normalize                          |

Do not treat unprefixed `metrics/mAP50-95(B)` as mixed-training fitness.

### Predict

```python
from ultralytics import WeDetect

model = WeDetect("runs/.../weights/best.pt")
model.set_classes(["人", "公交车", "车"])
results = model.predict(source="ultralytics/assets/bus.jpg", conf=0.25, save=True)
```

`set_classes` encodes prompts with XLM-R + `text_model_weights` and caches `txt_feats` on the head.

### Export

Use **`export_mode=dual`** (default) so prompts stay swappable after export. Tokenizer stays in Python.

| Format           | Typical call                                                 | Output                                                                    |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Dual ONNX        | `export(format="onnx", export_mode="dual")`                  | `*_vision.onnx` + `*_language.onnx`                                       |
| Dual TensorRT    | `export(format="engine", export_mode="dual", device=0)`      | sibling `*.engine`                                                        |
| Dual TorchScript | `export(format="torchscript", export_mode="dual", nms=True)` | sibling `*.torchscript`                                                   |
| Whole ONNX       | `export(format="onnx", export_mode="whole")`                 | `*_whole.onnx` (example script only; official `predict` does not load it) |

```python
from ultralytics import WeDetect

WeDetect("best.pt").export(format="onnx", export_mode="dual", imgsz=640)
m = WeDetect("best_vision.onnx")  # sibling *_language.onnx required
m.set_classes(["车", "人"])
m.predict("ultralytics/assets/bus.jpg")
```

Do not fuse `WeDetectDetect` with a fixed vocabulary if you still need to change prompts.

## Data: `class_texts` and mixed YAML

Labels are YOLO detect rows (`cls x y w h`, normalized). Optional segment polygons enable `mask_refine=True`.

`class_texts` is `list[list[str]]` aligned with `names`:

- Length ≥ `nc`: rows `0 .. nc-1` are annotated classes (one synonym sampled per batch).
- Extra rows are **train-only negative prompts**; val metrics use only the first `nc` rows.
- Put near-synonyms on the same row. Do **not** mix attributes (color, hat, …) into the base-class row.

Mixed config (`ultralytics/cfg/datasets/wedetect_mixed.yaml`):

```yaml
train:
    yolo_data:
        - ultralytics/cfg/datasets/customer/vehicle.yaml
        - ultralytics/cfg/datasets/wedetect_coco.yaml
val:
    yolo_data:
        - ultralytics/cfg/datasets/customer/lvis.yaml
        - ultralytics/cfg/datasets/customer/vehicle.yaml
val_fitness_weights: [2, 1]
val_fitness_dynamic: true
```

With `mix_global_texts=True` (default), synonym overlap merges a global vocab and remaps local `cls`. Optional `train.grounding_data` uses `GroundingDataset` (full COCO-style JSON with `caption` + `tokens_positive`, not JSONL).

## Train-time pseudo labels

Enabled per **train** subset (dataset YAML keys override `train()` / CLI). Original `labels/` and source `class_texts` JSON are never overwritten.

Resolution order: subset YAML `pseudo_label*` → mixed top-level YAML → finetune/CLI args → defaults.

| Key                         | Default                    | Meaning                                                  |
| --------------------------- | -------------------------- | -------------------------------------------------------- |
| `pseudo_label`              | `False`                    | Enable for this train subset                             |
| `pseudo_label_model`        | `sam3.pt`                  | Teacher: SAM3 / YOLO / WeDetect / D-FINE `.pt`           |
| `pseudo_label_classes`      | COCO `names`               | Teacher English vocab (same order as texts)              |
| `pseudo_label_class_texts`  | `coco_zh_class_texts.json` | Chinese rows written into `*_train.json`                 |
| `pseudo_label_conf`         | `0.25`                     | Teacher confidence                                       |
| `pseudo_label_imgsz`        | `640`                      | Teacher inference size; **independent of train `imgsz`** |
| `pseudo_label_batch`        | `0` (auto)                 | Image batch (YOLO/WeDetect/D-FINE) or SAM3 prompt chunk  |
| `pseudo_label_mem_fraction` | `0.85`                     | Target fraction of free VRAM when auto-batching          |
| `pseudo_label_flush_every`  | `200`                      | Incremental teacher-cache flush (images)                 |
| `pseudo_label_prefetch`     | `2`                        | Loader batches to prefetch; `0` disables                 |

Artifacts (dataset `path` root / first image directory):

| File                               | Role                                                              |
| ---------------------------------- | ----------------------------------------------------------------- |
| `pseudo_labels-{model_stem}.cache` | Teacher boxes only (`DATASET_CACHE_VERSION`, currently `1.0.4`)   |
| `labels_pseudo_merged.cache`       | GT + pseudo; train `labels_dir=labels_pseudo_merged`              |
| `pseudo_label_meta.json`           | Idempotency hash, vocab, cache paths                              |
| `<stem>_train.json`                | Merged texts (GT prefix + kept teacher rows + leftover negatives) |

Cache hit requires matching meta hash, on-disk `*_train.json`, merged cache, **and** a complete teacher cache whose **version + hash** match. A version bump (for example `1.0.3` → `1.0.4`) misses at `apply_pseudo_labels_to_subset` and re-runs the teacher. `YOLODataset.get_labels()` is more tolerant: hash-matched merged caches can load across versions; if merged cache is gone, it rebuilds from teacher cache + GT **without** re-inference.

Class-level synonym overlap drops colliding teacher classes. Remaining classes append after GT ids. Boxes are concatenated with **no IoU NMS**.

### D-FINE as teacher

Objects365 checkpoints (`dfine-*-obj365.pt`) use head `nc=366` with index `0` reserved as background (`dfine_class_names`). Teacher class ids are matched by English prompt, not by YOLO `class_id`. D-FINE preprocess is square `LetterBox(scale_fill=True)` and `/255` (no ImageNet mean/std). Decoder anchors rebuild when feature-map size changes, so `pseudo_label_imgsz=1280` is valid.

Example subset YAML:

```yaml
pseudo_label: true
pseudo_label_model: ./pretrained_weights/dfine-x-obj365.pt
pseudo_label_classes: Objects365.yaml
pseudo_label_class_texts: texts/objects365_zh_class_texts.json
pseudo_label_conf: 0.2
pseudo_label_imgsz: 1280
```

See [D-FINE](dfine.md) for train/val/predict/export of the teacher itself.

## Supported tasks and modes

| Model        | Config / weights                    | Task                       | Train | Val | Predict | Export        |
| ------------ | ----------------------------------- | -------------------------- | ----- | --- | ------- | ------------- |
| WeDetect     | `wedetect-*.yaml` / `wedetect_*.pt` | Detect (open-vocab)        | ✅    | ✅  | ✅      | ✅ dual/whole |
| WeDetect-Uni | `wedetect-uni-*.yaml`               | Detect (learnable prompts) | ✅    | ✅  | ✅      | ✅            |

## Config files

| File                                                 | Use                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `ultralytics/cfg/wedetect_finetune.yaml`             | OV fine-tune (`freeze_text_encoder=False`, `lr0=5e-6`, `text_lr_mult=0.01`)                 |
| `ultralytics/cfg/wedetect_finetune_mask_refine.yaml` | OV + `mask_refine` (closer to original `lr0=2e-5` / `close_mosaic=4`)                       |
| `ultralytics/cfg/wedetect_scratch.yaml`              | From-scratch mixed training                                                                 |
| `ultralytics/cfg/default.yaml`                       | Global defaults (`freeze_text_encoder=True`, `export_mode=dual`, pseudo / val-fitness keys) |

Pass `cfg=wedetect_finetune.yaml` for OV work. Training with only `default.yaml` freezes the text tower.

Tokenizer search order (`ultralytics/nn/text_model.py`): repo `xlm-roberta-base/` or `checkpoints/xlm-roberta-base/` (must contain `config.json`) → Hugging Face cache → download, then `save_pretrained` to repo root. Do not leave an **empty** `xlm-roberta-base/` directory.

## FAQ

### Why are `metrics/mAP50-95(B)` and `lvis/metrics/mAP50-95(B)` identical?

The first `val.yolo_data` entry is copied to unprefixed `metrics/*` for default plots. If that entry is LVIS, the two columns match. Combined `fitness` is the weighted average and is what writes `best.pt`.

### Why did training re-run D-FINE on tens of thousands of images?

Teacher reuse is a cache hit, not a requirement. `try_load_cache` rejects a version mismatch. After a cache-format bump, one full teacher pass rewrites `1.0.4` caches; later starts should log `cache hit`.

### Can train `imgsz=1280` keep teacher inference at 640?

Yes. Set `pseudo_label_imgsz` (default `640`). It is hashed into the teacher cache, so changing it invalidates that cache.

### WeDetect-Uni vs WeDetect?

Uni drops the live LM at inference and uses learnable prompt embeddings (`WeDetectUniTrainer`). It is faster and not prompt-swappable via `set_classes` the same way. Load with `from ultralytics import WeDetectUni` — `YOLO("wedetect-uni-*.pt")` morphs to **`WeDetect`**, not Uni, because the stem still contains `wedetect`.

---
title: WeDetect Open-Vocabulary Fine-Tune
comments: true
description: End-to-end WeDetect OV fine-tune on Ultralytics — data, mixed YAML, pseudo labels, val fitness, dual ONNX/TensorRT export.
keywords: WeDetect, open-vocabulary fine-tune, class_texts, pseudo labels, D-FINE teacher, mixed dataset, dual export, Ultralytics
---

# WeDetect open-vocabulary fine-tune

This page is the English pipeline counterpart of the in-repo Chinese tutorial. Architecture and API summary: [WeDetect model](../models/wedetect.md). Step-by-step Chinese guide (data layout, export flags, FAQ): [`docs/zh/guides/wedetect-ov-finetune.md`](../../zh/guides/wedetect-ov-finetune.md).

Work from the repository root (the layer that contains `ultralytics/` and `pretrained_weights/`).

## Flow

```text
YOLO labels + Chinese class_texts (optional grounding JSON)
        ↓
single YAML or wedetect_mixed*.yaml ; cfg=wedetect_finetune.yaml
        ↓
optional teacher pseudo labels (SAM3 / YOLO / WeDetect / D-FINE)
        ↓
WeDetect(...).train(...)   # freeze_text_encoder=False
        ↓
best.pt (vision + text_model_weights)
        ↓
export_mode=dual → *_vision.* + *_language.*
        ↓
set_classes + predict
```

Always pass `cfg=ultralytics/cfg/wedetect_finetune.yaml` (or the mask-refine variant). `default.yaml` sets `freeze_text_encoder=True`.

## Environment and weights

```bash
pip install -e .
pip install transformers sentencepiece onnx onnxruntime
```

Place official checkpoints under `pretrained_weights/` (`wedetect_tiny.pt` / `wedetect_base.pt` / `wedetect_large.pt`). XLM-R loads from `xlm-roberta-base/` or `checkpoints/xlm-roberta-base/` (must include `config.json`), then the Hugging Face cache, then the network. Do not leave an empty `xlm-roberta-base/` directory.

## Data

YOLO detect txt (`cls x y w h`, normalized). `class_texts` is `list[list[str]]`:

- Rows `0 .. nc-1` match annotated classes; extra rows are train-only negatives.
- Synonyms may share a row; attributes (color, hat) must not share a row with the base class.

Single-dataset example: `ultralytics/cfg/datasets/wedetect_coco.yaml`. Mixed: `wedetect_mixed.yaml` / `wedetect_mixed_customer.yaml` (`train.yolo_data` + `val.yolo_data`). Grounding uses a full COCO-style JSON (`caption` + `tokens_positive`), not JSONL.

## Train

```python
from ultralytics import WeDetect

model = WeDetect("pretrained_weights/wedetect_base.pt")
model.train(
    data="ultralytics/cfg/datasets/wedetect_coco.yaml",
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    freeze_text_encoder=False,
    text_lr_mult=0.01,
    epochs=12,
    batch=4,
    imgsz=640,
    device=0,
)
```

What happens:

1. `get_dataset` → optional `maybe_build_pseudo_labels` (per train subset).
2. Mixed subsets merge texts (`mix_global_texts=True`) and remap local ids.
3. `freeze_text_encoder=False` registers the LM; text params use `lr0 * text_lr_mult`.
4. `save_model` writes `text_model_weights` into `best.pt` / `last.pt`.

Single-GPU first-epoch OOM halves `batch` up to 3 times and rebuilds dataloaders. Prefer an explicit `batch` that fits (for example `2` at 1280).

## Pseudo labels

Keys live on the **subset** YAML (override train args). Teacher size is `pseudo_label_imgsz` (default **640**), not train `imgsz`. Flush every `pseudo_label_flush_every` images (default **200**). Prefetch `pseudo_label_prefetch` (default **2**).

D-FINE Objects365 teacher: `pseudo_label_model=.../dfine-x-obj365.pt`, `pseudo_label_classes=Objects365.yaml`. Head `nc=366` (index 0 = background); English names map to head ids. Square scale-fill letterbox; anchors rebuild for non-640 `imgsz`.

Idempotency: `pseudo_label_meta.json` hash + `*_train.json` + both caches. Teacher cache **version must equal** `DATASET_CACHE_VERSION` (`1.0.4`) or the teacher runs again. Merged-cache load in `get_labels` can tolerate a version mismatch when the hash still matches, and can rebuild from teacher cache + GT without inference.

Original `labels/` and source `class_texts` JSON are never modified. Merged labels exist only as `labels_pseudo_merged.cache` (no per-image txt).

## Val and `fitness`

Each mixed val set is scored with its own prompts. Unprefixed `metrics/*` copy the **first** YAML val set. Unprefixed `fitness` is the weighted average that selects `best.pt`. Enable `val_fitness_dynamic` so epoch 2+ reweights from previous mAP50-95; LVIS target = `val_fitness_lvis_target_mult ×` mean customer mAP.

## Predict and export

```python
from ultralytics import WeDetect

m = WeDetect("runs/.../weights/best.pt")
m.set_classes(["人", "车"])
m.predict("ultralytics/assets/bus.jpg", save=True)

m.export(format="onnx", export_mode="dual", imgsz=640)
WeDetect(".../best_vision.onnx").set_classes(["人", "车"]).predict("bus.jpg")
```

Use `export_mode=dual` for swappable prompts (`onnx` / `engine` / `torchscript`). `whole` ONNX is example-script only.

## Checklist

- [ ] `class_id` aligns with the first `nc` `class_texts` rows
- [ ] `cfg=wedetect_finetune.yaml` and `freeze_text_encoder=False`
- [ ] `best.pt` contains non-empty `text_model_weights`
- [ ] Mixed `fitness` (not unprefixed LVIS mAP) is the number you track for `best.pt`
- [ ] Dual export: sibling `*_language.*` next to `*_vision.*`

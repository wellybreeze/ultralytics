# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""WeDetect train-time pseudo-label generation and merge (before labels *.cache)."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultralytics.data.utils import (
    IMG_FORMATS,
    dataset_root as _dataset_root,
    exif_size,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    portable_paths_hash,
    rel_path_key as _rel_key,
    remap_label_im_files,
    save_dataset_cache_file,
)
from ultralytics.utils import LOGGER, ROOT, TQDM, YAML, colorstr
from ultralytics.utils.checks import check_file

PSEUDO_LABELS_DIR = "labels_pseudo_merged"
PSEUDO_META_NAME = "pseudo_label_meta.json"
PSEUDO_ONLY_CACHE_NAME = "pseudo_labels.cache"
# Keep in sync with ultralytics.data.dataset.DATASET_CACHE_VERSION
DATASET_CACHE_VERSION = "1.0.3"
DEFAULT_CONF = 0.25
DEFAULT_CLASSES = ROOT / "cfg/datasets/coco.yaml"
DEFAULT_CLASS_TEXTS = ROOT / "cfg/datasets/texts/coco_zh_class_texts.json"
PSEUDO_CACHE_PIPELINE = "pseudo_cache_portable_v1"  # relative-path hashes; cross-machine reusable
PORTABLE_HASH_MODE = "rel_v1"
# Avoid O(N^2) full-cache rewrite
PSEUDO_FLUSH_EVERY = 50  # images between incremental cache flushes

# Dataset-yaml keys (per-subset). Resolution: data[key] > train args > defaults.
PSEUDO_CFG_KEYS = (
    "pseudo_label",
    "pseudo_label_model",
    "pseudo_label_classes",
    "pseudo_label_class_texts",
    "pseudo_label_conf",
    "pseudo_label_batch",
    "pseudo_label_mem_fraction",
)


def resolve_pseudo_cfg(data: dict | None, args: Any = None) -> dict[str, Any]:
    """Resolve pseudo-label settings for one dataset subset.

    Priority (highest first):
      1) keys on the dataset dict (from that dataset's YAML)
      2) trainer/global ``args`` (finetune yaml / CLI)
      3) built-in defaults

    This lets mixed training enable pseudo labels only on selected subsets
    (e.g. vehicle yes, LVIS no) without a global ``pseudo_label=True``.
    """
    data = data or {}

    def _from_args(key: str, default: Any = None) -> Any:
        if args is None:
            return default
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    def _pick(key: str, default: Any = None) -> Any:
        if key in data and data[key] is not None and data[key] != "":
            return data[key]
        val = _from_args(key, None)
        if val is not None and val != "":
            return val
        return default

    enabled = bool(_pick("pseudo_label", False))
    conf = float(_pick("pseudo_label_conf", DEFAULT_CONF) or DEFAULT_CONF)
    conf = float(np.clip(conf, 0.0, 1.0))
    batch_arg = _pick("pseudo_label_batch", 0)
    try:
        batch_i = int(batch_arg) if batch_arg is not None else 0
    except (TypeError, ValueError):
        batch_i = 0
    mem_fraction = float(_pick("pseudo_label_mem_fraction", 0.85) or 0.85)
    mem_fraction = float(np.clip(mem_fraction, 0.1, 0.95))
    imgsz = int(_from_args("imgsz", 640) or 640)
    # imgsz is train-global; allow per-dataset override if present
    if "imgsz" in data and data["imgsz"] is not None:
        try:
            imgsz = int(data["imgsz"])
        except (TypeError, ValueError):
            pass

    return {
        "pseudo_label": enabled,
        "pseudo_label_model": str(_pick("pseudo_label_model", "sam3.pt") or "sam3.pt"),
        "pseudo_label_classes": str(_pick("pseudo_label_classes", "") or ""),
        "pseudo_label_class_texts": str(_pick("pseudo_label_class_texts", "") or ""),
        "pseudo_label_conf": conf,
        "pseudo_label_batch": batch_i,  # <=0 means auto
        "pseudo_label_mem_fraction": mem_fraction,
        "imgsz": imgsz,
    }


def subset_wants_pseudo(data: dict | None, args: Any = None) -> bool:
    """Return True if this subset should run the pseudo-label pipeline."""
    return bool(resolve_pseudo_cfg(data, args).get("pseudo_label"))


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _as_text_groups(raw: Any) -> list[list[str]]:
    """Normalize class text definitions to list[list[str]]."""
    if isinstance(raw, dict):
        # yaml names: {0: "person", ...} or {0: ["车","车辆"]}
        items = []
        for k in sorted(raw.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            v = raw[k]
            if isinstance(v, (list, tuple)):
                items.append([str(x) for x in v if str(x).strip()])
            else:
                # English COCO-style "a/b/c" synonyms
                parts = [p.strip() for p in str(v).split("/") if p.strip()]
                items.append(parts or [str(v)])
        return items
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                out.append([str(x) for x in item if str(x).strip()])
            else:
                parts = [p.strip() for p in str(item).split("/") if p.strip()]
                out.append(parts or [str(item)])
        return out
    raise TypeError(f"Unsupported class definition type: {type(raw)}")


def load_text_groups(path: str | Path | None, default: Path) -> list[list[str]]:
    """Load class synonym groups from yaml/json; empty path uses default."""
    p = Path(path) if path else default
    if not str(path or "").strip():
        p = default
    p = Path(check_file(str(p)))
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = YAML.load(p)
        if isinstance(data, dict) and "names" in data:
            return _as_text_groups(data["names"])
        return _as_text_groups(data)
    with open(p, encoding="utf-8") as f:
        return _as_text_groups(json.load(f))


def load_gt_text_groups(data: dict) -> list[list[str]]:
    """Build GT synonym groups from data names + optional **source** class_texts JSON.

    Always prefers the original (non-``*_train``) file so regenerating pseudo labels
    does not treat a previous merged vocab as GT.
    """
    names = data.get("names") or {}
    groups = _as_text_groups(names)
    ct = data.get("class_texts")
    if not ct:
        return groups
    p = Path(ct)
    if not p.is_absolute() and data.get("path"):
        cand = Path(data["path"]) / ct
        if cand.exists() or source_class_texts_path(cand).exists():
            p = cand
    src = source_class_texts_path(p)
    if src.exists():
        p = src
    elif not p.exists():
        LOGGER.warning(f"{colorstr('WeDetect pseudo:')} GT class_texts not found at '{p}', using names only")
        return groups
    with open(p, encoding="utf-8") as f:
        loaded = _as_text_groups(json.load(f))
    # Prefer class_texts rows for overlapping indices; keep extras from class_texts as OV negatives
    nc = int(data.get("nc") or len(groups))
    merged = []
    for i in range(max(nc, len(loaded))):
        if i < len(loaded) and loaded[i]:
            merged.append(loaded[i])
        elif i < len(groups):
            merged.append(groups[i])
    return merged


def gt_synonym_set(gt_groups: list[list[str]]) -> set[str]:
    return {_norm(x) for g in gt_groups for x in g}


def filter_pseudo_classes(
    en_groups: list[list[str]],
    zh_groups: list[list[str]],
    gt_set: set[str],
) -> tuple[list[int], list[list[str]], list[list[str]]]:
    """Drop pseudo classes that share any synonym with GT.

    Returns:
        kept_src_ids: original indices in the pseudo vocabulary
        kept_en, kept_zh: filtered synonym groups (same length)
    """
    n = min(len(en_groups), len(zh_groups))
    if len(en_groups) != len(zh_groups):
        LOGGER.warning(
            f"{colorstr('WeDetect pseudo:')} class file length mismatch "
            f"(en={len(en_groups)}, zh={len(zh_groups)}); using first {n}"
        )
    kept_ids, kept_en, kept_zh = [], [], []
    for i in range(n):
        syns = {_norm(x) for x in en_groups[i] + zh_groups[i]}
        if syns & gt_set:
            continue
        kept_ids.append(i)
        kept_en.append(en_groups[i])
        kept_zh.append(zh_groups[i])
    return kept_ids, kept_en, kept_zh


_TRAIN_STEM_SUFFIX = "_train"


def is_train_class_texts_path(path: Path | str) -> bool:
    """True if path stem already ends with ``_train`` (e.g. ``foo_train.json``)."""
    return Path(path).stem.endswith(_TRAIN_STEM_SUFFIX)


def source_class_texts_path(path: Path | str) -> Path:
    """Map ``foo_train.json`` → ``foo.json``; leave other paths unchanged."""
    p = Path(path)
    if p.stem.endswith(_TRAIN_STEM_SUFFIX):
        return p.with_name(p.stem[: -len(_TRAIN_STEM_SUFFIX)] + p.suffix)
    return p


def train_class_texts_path(path: Path | str) -> Path:
    """Map ``foo.json`` → ``foo_train.json``; leave ``foo_train.json`` unchanged."""
    p = Path(path)
    if p.stem.endswith(_TRAIN_STEM_SUFFIX):
        return p
    return p.with_name(f"{p.stem}{_TRAIN_STEM_SUFFIX}{p.suffix}")


def resolve_class_texts_path(data: dict) -> Path:
    """Resolve yaml ``class_texts`` path (may be original or already ``*_train``)."""
    ct = data.get("class_texts")
    root = Path(data["path"]) if data.get("path") else Path(".")
    if ct:
        p = Path(ct)
        if not p.is_absolute():
            p = root / ct
        return p
    return root / "class_texts.json"


def resolve_train_class_texts_path(data: dict) -> Path:
    """Merged vocabulary path for train/finetune: ``<stem>_train.json`` next to source."""
    return train_class_texts_path(resolve_class_texts_path(data))


def resolve_source_class_texts_path(data: dict) -> Path:
    """Original GT ``class_texts`` path (never the ``*_train`` write target)."""
    return source_class_texts_path(resolve_class_texts_path(data))


def build_merged_vocabulary(
    data: dict,
    en_groups: list[list[str]],
    zh_groups: list[list[str]],
) -> dict[str, Any]:
    """Resolve class conflicts and build ordered vocabulary.

    Order (fixed):
      1) annotated GT classes ``0 .. nc_gt-1`` (from names / class_texts prefix)
      2) kept pseudo Chinese groups appended in original relative order (ids ``nc_gt+k``)
      3) leftover original negatives after ``nc_gt`` that do not overlap kept pseudo

    Returns:
        dict with keys: nc_gt, new_nc, new_names, new_texts, kept_ids, kept_en, kept_zh,
        dropped_src_ids.
    """
    gt_groups = load_gt_text_groups(data)
    nc_gt = int(data.get("nc") or len(data.get("names") or {}))
    # Conflict check uses full GT synonym set (annotated + existing negatives)
    gt_set = gt_synonym_set(gt_groups)
    kept_ids, kept_en, kept_zh = filter_pseudo_classes(en_groups, zh_groups, gt_set)

    new_names: dict[int, str] = {}
    for i in range(nc_gt):
        if i < len(gt_groups) and gt_groups[i]:
            new_names[i] = gt_groups[i][0]
        else:
            new_names[i] = str((data.get("names") or {}).get(i, i))
    for k, zh in enumerate(kept_zh):
        new_names[nc_gt + k] = zh[0]

    kept_syn = {_norm(x) for g in kept_zh for x in g}
    leftover_neg = []
    for g in gt_groups[nc_gt:]:
        if g and not ({_norm(x) for x in g} & kept_syn):
            leftover_neg.append(g)

    new_texts = list(gt_groups[:nc_gt]) + list(kept_zh) + leftover_neg
    new_nc = nc_gt + len(kept_zh)
    n_src = min(len(en_groups), len(zh_groups))
    dropped = [i for i in range(n_src) if i not in set(kept_ids)]
    return {
        "nc_gt": nc_gt,
        "new_nc": new_nc,
        "new_names": new_names,
        "new_texts": new_texts,
        "kept_ids": kept_ids,
        "kept_en": kept_en,
        "kept_zh": kept_zh,
        "dropped_src_ids": dropped,
        "gt_groups": gt_groups,
    }


def write_class_texts(path: Path, new_texts: list[list[str]]) -> Path:
    """Write merged class_texts JSON to ``*_train.json`` (does not overwrite source JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new_texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(f"{colorstr('WeDetect pseudo:')} wrote train vocabulary ({len(new_texts)} rows) -> {path}")
    return path


def _texts_hash(texts: list[list[str]]) -> str:
    return get_hash([json.dumps(texts, ensure_ascii=False, sort_keys=False)])


def read_yolo_labels(label_path: Path) -> np.ndarray:
    """Read detect labels as (N,5) cls+xywhn; empty if missing."""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        # detect: cls xywh [conf]; skip segment polygons (many coords)
        if len(parts) < 5 or len(parts) > 6:
            continue
        cls = float(parts[0])
        xywh = list(map(float, parts[1:5]))
        rows.append([cls, *xywh])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def write_yolo_labels(label_path: Path, labels: np.ndarray) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in labels:
        cls = int(row[0])
        xywh = row[1:5]
        lines.append(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def collect_image_files(img_path: str | Path) -> list[str]:
    """Collect image paths from a directory or list file (YOLO train split)."""
    import glob

    f: list[str] = []
    p = Path(img_path)
    if p.is_dir():
        f += glob.glob(str(Path(glob.escape(p)) / "**" / "*.*"), recursive=True)
    elif p.is_file():
        with open(p, encoding="utf-8") as t:
            lines = t.read().strip().splitlines()
            parent = str(p.parent) + os.sep
            f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in lines]
    else:
        raise FileNotFoundError(f"Train image path does not exist: {img_path}")
    im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
    if not im_files:
        raise FileNotFoundError(f"No images found under {img_path}")
    return im_files


def _is_sam3(model: str) -> bool:
    return "sam3" in Path(model).stem.lower()


def _is_wedetect_ckpt(model: str) -> bool:
    stem = Path(model).stem.lower()
    if "wedetect" in stem:
        return True
    try:
        import torch

        ckpt = torch.load(model, map_location="cpu", weights_only=False)
        m = ckpt.get("ema") or ckpt.get("model")
        return m is not None and type(m).__name__ in {"WeDetectModel", "WeDetectUniModel"}
    except Exception:
        return False


def _results_to_xywhn(result) -> np.ndarray:
    """Convert ultralytics Results to detect rows (N,5) cls+xywhn.

    For SAM3, masks may exist but only bounding boxes are written (not polygons).
    """
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xywhn = result.boxes.xywhn.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([cls, xywhn], axis=1).astype(np.float32)


def _resolve_cuda_index(device: str | int | None) -> int | None:
    """Return CUDA device index, or None for CPU/MPS/unavailable."""
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    if device is None or device == "":
        return torch.cuda.current_device()
    s = str(device).strip().lower()
    if s in {"cpu", "mps"}:
        return None
    if s.startswith("cuda:"):
        return int(s.split(":")[-1])
    try:
        return int(s)
    except ValueError:
        return torch.cuda.current_device()


def _cuda_mem_gib(device_index: int) -> tuple[float, float]:
    """Return (free_GiB, total_GiB) for a CUDA device."""
    import torch

    free_b, total_b = torch.cuda.mem_get_info(device_index)
    gib = float(1 << 30)
    return free_b / gib, total_b / gib


def _is_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def auto_sam3_prompt_chunk(
    device: str | int | None,
    n_prompts: int,
    fraction: float = 0.85,
    max_chunk: int = 128,
) -> int:
    """Pick SAM3 text-prompt chunk size from free VRAM (image batch stays 1)."""
    idx = _resolve_cuda_index(device)
    if idx is None or n_prompts <= 1:
        return max(1, min(32, n_prompts))
    free_gib, total_gib = _cuda_mem_gib(idx)
    usable = free_gib * float(fraction)
    # Heuristic: larger free memory -> larger prompt chunks (SAM is image-serial)
    if usable >= 24:
        chunk = 128
    elif usable >= 12:
        chunk = 64
    elif usable >= 6:
        chunk = 48
    elif usable >= 3:
        chunk = 32
    else:
        chunk = 16
    chunk = int(max(8, min(max_chunk, chunk, n_prompts)))
    LOGGER.info(
        f"{colorstr('WeDetect pseudo:')} SAM3 prompt_chunk={chunk} "
        f"(CUDA:{idx} free={free_gib:.1f}/{total_gib:.1f} GiB, target={fraction:.0%})"
    )
    return chunk


def auto_predict_batch(
    model: Any,
    device: str | int | None,
    sample_im: str,
    conf: float = DEFAULT_CONF,
    imgsz: int = 640,
    fraction: float = 0.85,
    max_batch: int = 128,
    classes: list[int] | None = None,
) -> int:
    """Probe YOLO/WeDetect predict batch from free VRAM + OOM-safe geometric search.

    1) Warm up with batch=1 and measure activation memory
    2) Estimate candidate from free*fraction / per-image
    3) Grow 1,2,4,... up to estimate and back off on CUDA OOM
    """
    import torch

    idx = _resolve_cuda_index(device)
    if idx is None:
        batch = min(8, max_batch)
        LOGGER.info(f"{colorstr('WeDetect pseudo:')} non-CUDA device, predict batch={batch}")
        return batch

    predict_kw = dict(
        conf=conf,
        imgsz=imgsz,
        device=device if device is not None else idx,
        verbose=False,
        save=False,
        stream=False,
    )
    if classes is not None:
        predict_kw["classes"] = classes

    def _run(b: int) -> None:
        src = [sample_im] * b
        model.predict(source=src, batch=b, **predict_kw)

    # Warmup + per-image activation estimate
    torch.cuda.set_device(idx)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(idx)
    base = torch.cuda.memory_allocated(idx)
    try:
        _run(1)
    except Exception as e:
        LOGGER.warning(f"{colorstr('WeDetect pseudo:')} batch probe warmup failed ({e}); using batch=1")
        return 1
    torch.cuda.synchronize(idx)
    peak = torch.cuda.max_memory_allocated(idx)
    per_img = max(int(peak - base), 1)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    usable = int(free_b * float(fraction))
    est = max(1, min(max_batch, usable // per_img if per_img else 1))

    # Geometric search capped by estimate (+1 step) for safety margin discovery
    candidates = [b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b <= max_batch and b <= max(est * 2, 1)]
    if est not in candidates:
        candidates.append(est)
    candidates = sorted(set(candidates))

    best = 1
    for b in candidates:
        if b == 1:
            best = 1
            continue
        try:
            torch.cuda.empty_cache()
            _run(b)
            best = b
        except RuntimeError as e:
            if _is_oom_error(e):
                torch.cuda.empty_cache()
                LOGGER.info(f"{colorstr('WeDetect pseudo:')} batch={b} OOM, fallback to {best}")
                break
            raise

    free_gib, total_gib = free_b / float(1 << 30), total_b / float(1 << 30)
    LOGGER.info(
        f"{colorstr('WeDetect pseudo:')} auto predict batch={best} "
        f"(est={est}, per_img≈{per_img / (1 << 20):.0f}MiB, "
        f"CUDA:{idx} free={free_gib:.1f}/{total_gib:.1f} GiB, target={fraction:.0%})"
    )
    torch.cuda.empty_cache()
    return int(best)


def _predict_path_list(
    model: Any,
    im_files: list[str],
    *,
    bsz: int,
    conf: float,
    imgsz: int,
    device: str | int | None,
    classes: list[int] | None = None,
    work_dir: Path | None = None,
):
    """Stream predictions over image paths with true image-batch size ``bsz``.

    Critical: do **not** pass a Python ``list`` of paths to ``model.predict``.
    Ultralytics ``check_source`` treats lists via ``autocast_list`` →
    ``LoadPilAndNumpy`` with ``bs=len(list)``, which loads the entire list into
    **one** GPU batch (e.g. 512 images → OOM). A ``.txt`` source uses
    ``LoadImagesAndVideos`` and respects ``batch=bsz``.
    """
    work_dir = Path(work_dir) if work_dir is not None else Path(im_files[0]).parent
    work_dir.mkdir(parents=True, exist_ok=True)
    txt = work_dir / f".wedetect_pseudo_sources_{os.getpid()}.txt"
    txt.write_text("\n".join(str(p) for p in im_files) + "\n", encoding="utf-8")
    predict_kw = dict(
        source=str(txt),
        stream=True,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save=False,
        batch=int(bsz),
        rect=False,  # fixed square letterbox → stable VRAM vs default rect=True
    )
    if classes is not None:
        predict_kw["classes"] = classes
    try:
        yield from model.predict(**predict_kw)
    finally:
        try:
            txt.unlink(missing_ok=True)
        except OSError:
            pass


def run_teacher_inference(
    model_path: str,
    im_files: list[str],
    kept_src_ids: list[int],
    kept_zh: list[list[str]],
    kept_en: list[list[str]],
    device: str | int | None,
    conf: float = DEFAULT_CONF,
    imgsz: int = 640,
    batch: int | None = None,
    mem_fraction: float = 0.85,
    on_image: Any | None = None,
) -> dict[str, np.ndarray]:
    """Run teacher model; return {im_file: (N,5) with cls in 0..len(kept)-1}.

    YOLO/WeDetect use image ``batch`` (auto from free VRAM when ``batch`` is None/<=0).
    SAM3 does not support image batching; prompt chunk size is auto-tuned instead.

    ``on_image(im_file, arr_local)`` is invoked after each image finishes all class
    prompts (used for per-image cache flush / crash resume).
    """
    prompts_zh = [g[0] for g in kept_zh]
    out: dict[str, np.ndarray] = {}
    if not im_files:
        return out

    committed_paths: set[str] = set()
    committed_names: set[str] = set()

    def _emit(im: str, arr: np.ndarray) -> None:
        committed_paths.add(im)
        committed_names.add(Path(im).name)
        if on_image is not None:
            # Caller owns persistence; avoid duplicating all preds in ``out`` (RAM)
            on_image(im, arr)
        else:
            out[im] = arr

    def _fill_missing() -> None:
        for im in im_files:
            if im in committed_paths or Path(im).name in committed_names:
                if on_image is None and im not in out:
                    hit = next((v for k, v in out.items() if Path(k).name == Path(im).name), None)
                    if hit is not None:
                        out[im] = hit
                    else:
                        _emit(im, np.zeros((0, 5), dtype=np.float32))
                continue
            _emit(im, np.zeros((0, 5), dtype=np.float32))

    if _is_sam3(model_path):
        from ultralytics.models.sam import SAM3SemanticPredictor

        # SAM3 stride=14; pre-align so check_imgsz does not warn (e.g. 640 → 644)
        sam_stride = 14
        sam_imgsz = int(max(math.ceil(int(imgsz) / sam_stride) * sam_stride, sam_stride))
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            save=False,
            verbose=False,
            device=device if device is not None else "",
            batch=1,  # SAM API is image-serial
            imgsz=sam_imgsz,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        step = (
            int(batch)
            if batch is not None and int(batch) > 0
            else auto_sam3_prompt_chunk(device, len(prompts_zh), fraction=mem_fraction)
        )
        LOGGER.info(
            f"{colorstr('WeDetect pseudo:')} SAM3 semantic teacher on {len(im_files)} images, "
            f"{len(prompts_zh)} classes, prompt_chunk={step}"
        )
        for im in im_files:
            predictor.set_image(im)
            chunks = []
            s = 0
            while s < len(prompts_zh):
                part = prompts_zh[s : s + step]
                try:
                    results = predictor(text=part)
                except RuntimeError as e:
                    if not _is_oom_error(e) or step <= 8:
                        raise
                    import torch

                    torch.cuda.empty_cache()
                    step = max(8, step // 2)
                    LOGGER.warning(f"{colorstr('WeDetect pseudo:')} SAM3 OOM, reduce prompt_chunk -> {step}")
                    continue
                arr = _results_to_xywhn(results[0])
                if len(arr):
                    arr = arr.copy()
                    arr[:, 0] = arr[:, 0] + s
                    chunks.append(arr)
                s += len(part)
            _emit(im, np.concatenate(chunks, 0) if chunks else np.zeros((0, 5), dtype=np.float32))
        _fill_missing()
        return out

    if _is_wedetect_ckpt(model_path):
        from ultralytics import WeDetect

        model = WeDetect(model_path)
        model.set_classes(prompts_zh)
        bsz = (
            int(batch)
            if batch is not None and int(batch) > 0
            else auto_predict_batch(
                model, device, im_files[0], conf=conf, imgsz=imgsz, fraction=mem_fraction
            )
        )
        LOGGER.info(
            f"{colorstr('WeDetect pseudo:')} WeDetect teacher, {len(prompts_zh)} classes, "
            f"batch={bsz}, images={len(im_files)} (txt source, not path-list)"
        )
        results = _predict_path_list(
            model,
            im_files,
            bsz=bsz,
            conf=conf,
            imgsz=imgsz,
            device=device,
            work_dir=Path(im_files[0]).parent,
        )
        for r in results:
            path = str(r.path)
            _emit(path, _results_to_xywhn(r))
            r.orig_img = None
            r.boxes = None
        _fill_missing()
        return out

    # YOLO / other detect models with fixed id space (COCO-style)
    from ultralytics import YOLO

    model = YOLO(model_path)
    src_to_kept = {sid: k for k, sid in enumerate(kept_src_ids)}
    classes_filter = kept_src_ids
    bsz = (
        int(batch)
        if batch is not None and int(batch) > 0
        else auto_predict_batch(
            model,
            device,
            im_files[0],
            conf=conf,
            imgsz=imgsz,
            fraction=mem_fraction,
            classes=classes_filter if classes_filter else None,
        )
    )
    while bsz >= 1:
        try:
            LOGGER.info(
                f"{colorstr('WeDetect pseudo:')} YOLO teacher '{model_path}', batch={bsz}, "
                f"images={len(im_files)}, "
                f"classes={classes_filter[:8]}{'...' if len(classes_filter) > 8 else ''}"
            )
            results = _predict_path_list(
                model,
                im_files,
                bsz=bsz,
                conf=conf,
                imgsz=imgsz,
                device=device,
                classes=classes_filter if classes_filter else None,
                work_dir=Path(im_files[0]).parent,
            )
            for r in results:
                arr = _results_to_xywhn(r)
                if len(arr):
                    mapped = []
                    for row in arr:
                        sid = int(row[0])
                        if sid not in src_to_kept:
                            continue
                        mapped.append([src_to_kept[sid], *row[1:5]])
                    arr = np.array(mapped, dtype=np.float32) if mapped else np.zeros((0, 5), dtype=np.float32)
                _emit(str(r.path), arr)
                r.orig_img = None
                r.boxes = None
            break
        except RuntimeError as e:
            if not _is_oom_error(e) or bsz <= 1:
                raise
            import torch

            torch.cuda.empty_cache()
            if committed_paths:
                # Partial writes already flushed; ask user to lower batch and resume
                raise RuntimeError(
                    f"YOLO teacher OOM at batch={bsz} after {len(committed_paths)} images; "
                    f"set pseudo_label_batch={max(1, bsz // 2)} and re-run to resume. ({e})"
                ) from e
            bsz = max(1, bsz // 2)
            LOGGER.warning(f"{colorstr('WeDetect pseudo:')} YOLO OOM before first image, retry batch={bsz}")
    _fill_missing()
    return out


def merge_labels(gt: np.ndarray, pseudo: np.ndarray, nc_gt: int) -> np.ndarray:
    """Keep GT ids; remap pseudo cls to ``nc_gt+k`` and append (no IoU suppress).

    Class-level dedup is done earlier by filtering pseudo vocabulary against GT
    synonyms; spatially overlapping boxes of different classes are kept.
    """
    if gt.size == 0 and pseudo.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    kept_pseudo = np.zeros((0, 5), dtype=np.float32)
    if len(pseudo):
        kept_pseudo = pseudo.copy()
        kept_pseudo[:, 0] = kept_pseudo[:, 0] + nc_gt
    parts = [x for x in (gt, kept_pseudo) if len(x)]
    return np.concatenate(parts, 0) if parts else np.zeros((0, 5), dtype=np.float32)


def _resource_key(path: str | Path | None) -> str:
    """Stable identity for model/config paths across machines (basename)."""
    if path is None or not str(path).strip():
        return ""
    return Path(str(path)).name


def _meta_hash(
    im_files: list[str],
    gt_label_files: list[str],
    model: str,
    classes_path: str,
    texts_teacher_path: str,
    class_texts_path: str,
    kept_ids: list[int],
    conf: float,
    new_texts: list[list[str]],
    root: Path | None = None,
) -> str:
    payload = {
        "im": portable_paths_hash(im_files, root),
        "gt": portable_paths_hash([str(p) for p in gt_label_files], root) if gt_label_files else "",
        "model": _resource_key(model),
        "classes": _resource_key(classes_path),
        "texts_teacher": _resource_key(texts_teacher_path),
        "class_texts_path": _rel_key(class_texts_path, root) if class_texts_path else "",
        "kept": kept_ids,
        "conf": float(conf),
        "new_texts": _texts_hash(new_texts),
        "dedup": "class_only",
        "pipeline": PSEUDO_CACHE_PIPELINE,
        "path_mode": PORTABLE_HASH_MODE,
    }
    return get_hash([json.dumps(payload, sort_keys=True, ensure_ascii=False)])


def _pseudo_only_hash(
    im_files: list[str],
    model: str,
    classes_path: str,
    texts_teacher_path: str,
    kept_ids: list[int],
    conf: float,
    nc_gt: int,
    root: Path | None = None,
) -> str:
    payload = {
        "im": portable_paths_hash(im_files, root),
        "model": _resource_key(model),
        "classes": _resource_key(classes_path),
        "texts_teacher": _resource_key(texts_teacher_path),
        "kept": kept_ids,
        "conf": float(conf),
        "nc_gt": int(nc_gt),
        "pipeline": PSEUDO_CACHE_PIPELINE,
        "path_mode": PORTABLE_HASH_MODE,
    }
    return get_hash([json.dumps(payload, sort_keys=True, ensure_ascii=False)])


def merged_cache_path(im_files: list[str]) -> Path:
    """Path of ``labels_pseudo_merged.cache`` (same convention as YOLODataset.get_labels)."""
    label_files = img2label_paths(im_files, label_dir=PSEUDO_LABELS_DIR)
    return Path(label_files[0]).parent.with_suffix(".cache")


def merged_cache_hash(im_files: list[str], root: Path | None = None) -> str:
    label_files = img2label_paths(im_files, label_dir=PSEUDO_LABELS_DIR)
    return portable_paths_hash([str(p) for p in label_files] + list(im_files), root)


def pseudo_only_cache_path(data: dict, im_files: list[str] | None = None) -> Path:
    """``pseudo_labels.cache`` under dataset root (fallback: parent of first image)."""
    if data.get("path"):
        return Path(data["path"]) / PSEUDO_ONLY_CACHE_NAME
    if im_files:
        return Path(im_files[0]).parents[1] / PSEUDO_ONLY_CACHE_NAME
    return Path(PSEUDO_ONLY_CACHE_NAME)


def _image_shape_hw(im_file: str) -> tuple[int, int]:
    im = Image.open(im_file)
    w, h = exif_size(im)
    return int(h), int(w)


def _empty_xywh() -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float32)


def _empty_cls() -> np.ndarray:
    return np.zeros((0, 1), dtype=np.float32)


def _label_entry(
    im_file: str,
    shape: tuple[int, int],
    cls: np.ndarray,
    bboxes: np.ndarray,
    segments: list | None = None,
    keypoints=None,
) -> dict:
    return {
        "im_file": im_file,
        "shape": shape,
        "cls": cls.reshape(-1, 1).astype(np.float32) if len(cls) else _empty_cls(),
        "bboxes": bboxes.reshape(-1, 4).astype(np.float32) if len(bboxes) else _empty_xywh(),
        "segments": segments or [],
        "keypoints": keypoints,
        "normalized": True,
        "bbox_format": "xywh",
    }


def _index_by_image(labels: list[dict]) -> dict[str, dict]:
    by_path: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for lb in labels:
        im = str(lb["im_file"])
        by_path[im] = lb
        by_name[Path(im).name] = lb
    return {"path": by_path, "name": by_name}


def _lookup_label(index: dict[str, dict], im_file: str) -> dict | None:
    return index["path"].get(im_file) or index["name"].get(Path(im_file).name)


def try_load_cache(path: Path, expected_hash: str) -> dict | None:
    """Load Ultralytics cache if version/hash match; else None."""
    try:
        if not path.exists():
            return None
        cache = load_dataset_cache_file(path)
        if cache.get("version") != DATASET_CACHE_VERSION:
            return None
        if cache.get("hash") != expected_hash:
            return None
        if not cache.get("labels"):
            return None
        return cache
    except Exception:
        return None


def load_gt_label_entries(im_files: list[str], nc_gt: int) -> list[dict]:
    """Load GT label dicts from ``labels.cache`` or scan ``labels/*.txt`` + image shapes."""
    gt_label_files = img2label_paths(im_files, label_dir="labels")
    gt_cache_path = Path(gt_label_files[0]).parent.with_suffix(".cache")
    expected = get_hash(gt_label_files + im_files)
    cached = try_load_cache(gt_cache_path, expected)
    if cached is not None:
        LOGGER.info(f"{colorstr('WeDetect pseudo:')} loaded GT from {gt_cache_path}")
        index = _index_by_image(cached["labels"])
        out = []
        for im in im_files:
            lb = _lookup_label(index, im)
            if lb is None:
                try:
                    shape = _image_shape_hw(im)
                except Exception:
                    shape = (0, 0)
                out.append(_label_entry(im, shape, _empty_cls(), _empty_xywh()))
                continue
            cls = np.asarray(lb["cls"], dtype=np.float32).reshape(-1)
            bboxes = np.asarray(lb["bboxes"], dtype=np.float32).reshape(-1, 4)
            if len(cls):
                keep = (cls >= 0) & (cls < nc_gt)
                cls, bboxes = cls[keep], bboxes[keep]
            entry = dict(lb)
            entry["im_file"] = im
            entry["cls"] = cls.reshape(-1, 1) if len(cls) else _empty_cls()
            entry["bboxes"] = bboxes if len(bboxes) else _empty_xywh()
            out.append(entry)
        return out

    LOGGER.info(f"{colorstr('WeDetect pseudo:')} scanning GT labels from txt ({len(im_files)} images)")
    labels = []
    for im, lb_path in zip(im_files, gt_label_files):
        try:
            shape = _image_shape_hw(im)
        except Exception as e:
            LOGGER.warning(f"{colorstr('WeDetect pseudo:')} corrupt image {im}: {e}; using shape (0, 0)")
            shape = (0, 0)
        gt = read_yolo_labels(Path(lb_path))
        if len(gt):
            gt = gt[(gt[:, 0] >= 0) & (gt[:, 0] < nc_gt)]
        cls = gt[:, 0:1] if len(gt) else _empty_cls()
        boxes = gt[:, 1:5] if len(gt) else _empty_xywh()
        labels.append(_label_entry(im, shape, cls, boxes))
    return labels


def preds_to_pseudo_entries(
    im_files: list[str],
    preds: dict[str, np.ndarray],
    gt_entries: list[dict],
    nc_gt: int,
) -> list[dict]:
    """Build pseudo-only label entries; teacher local cls remapped to ``nc_gt+k``."""
    gt_index = _index_by_image(gt_entries)
    pred_by_name = {Path(k).name: v for k, v in preds.items()}
    entries = []
    for im in im_files:
        gt_lb = _lookup_label(gt_index, im)
        if gt_lb is None:
            try:
                shape = _image_shape_hw(im)
            except Exception:
                continue
        else:
            shape = tuple(gt_lb["shape"])
        pseudo = preds.get(im)
        if pseudo is None:
            pseudo = pred_by_name.get(Path(im).name, np.zeros((0, 5), dtype=np.float32))
        if len(pseudo):
            remapped = pseudo.copy()
            remapped[:, 0] = remapped[:, 0] + nc_gt
            cls, boxes = remapped[:, 0:1], remapped[:, 1:5]
        else:
            cls, boxes = _empty_cls(), _empty_xywh()
        entries.append(_label_entry(im, shape, cls, boxes))
    return entries


def save_label_cache(
    path: Path,
    labels: list[dict],
    cache_hash: str,
    prefix: str = "WeDetect pseudo: ",
    *,
    complete: bool | None = None,
    quiet: bool = False,
    extra: dict | None = None,
) -> dict:
    """Save YOLO-format dataset cache atomically (write ``*.tmp`` then replace)."""
    nf = sum(1 for lb in labels if len(lb["cls"]))
    ne = sum(1 for lb in labels if len(lb["cls"]) == 0)
    x = {
        "labels": labels,
        "hash": cache_hash,
        "results": (nf, 0, ne, 0, len(labels)),  # found, missing, empty, corrupt, total
        "msgs": [],
    }
    if complete is not None:
        x["complete"] = bool(complete)
    if extra:
        x.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Avoid noisy "New cache created" on every image: write ourselves when quiet
    x["version"] = DATASET_CACHE_VERSION
    try:
        if tmp.exists():
            tmp.unlink()
        with open(str(tmp), "wb") as f:
            np.save(f, x)
        tmp.replace(path)
        if not quiet:
            LOGGER.info(f"{prefix}New cache created: {path}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        LOGGER.warning(f"{prefix}WARNING ⚠️ Failed to save cache to {path}: {e}")
        # Fallback to stock saver
        save_dataset_cache_file(prefix, path, x, DATASET_CACHE_VERSION)
    return x


def pred_local_to_pseudo_entry(
    im_file: str,
    pred_local: np.ndarray,
    shape: tuple[int, int],
    nc_gt: int,
) -> dict:
    """Remap teacher local cls ``0..K-1`` to ``nc_gt+k`` and build one cache label entry."""
    if pred_local is None or len(pred_local) == 0:
        return _label_entry(im_file, shape, _empty_cls(), _empty_xywh())
    remapped = np.asarray(pred_local, dtype=np.float32).copy()
    remapped[:, 0] = remapped[:, 0] + nc_gt
    return _label_entry(im_file, shape, remapped[:, 0:1], remapped[:, 1:5])


def load_pseudo_progress(
    path: Path,
    expected_hash: str,
    im_files: list[str] | None = None,
    root: Path | None = None,
) -> list[dict]:
    """Load partial/full ``pseudo_labels.cache`` when hash matches; remap ``im_file`` to current paths."""
    labels: list[dict] = []
    cache = try_load_cache(path, expected_hash)
    if cache is not None:
        labels = list(cache.get("labels") or [])
    else:
        try:
            if not path.exists():
                return []
            raw = load_dataset_cache_file(path)
            if raw.get("version") != DATASET_CACHE_VERSION or raw.get("hash") != expected_hash:
                return []
            labels = list(raw.get("labels") or [])
        except Exception:
            return []
    if im_files is not None:
        labels = remap_label_im_files(labels, im_files, root)
    return labels


def _ordered_pseudo_labels(im_files: list[str], entries_by_key: dict[str, dict]) -> list[dict]:
    """Return label entries in ``im_files`` order for keys present in ``entries_by_key``."""
    out = []
    for im in im_files:
        lb = entries_by_key.get(im) or entries_by_key.get(Path(im).name)
        if lb is not None:
            e = dict(lb)
            e["im_file"] = im
            out.append(e)
    return out


def generate_pseudo_cache_incremental(
    *,
    model_path: str,
    im_files: list[str],
    gt_entries: list[dict],
    kept_src_ids: list[int],
    kept_zh: list[list[str]],
    kept_en: list[list[str]],
    nc_gt: int,
    device: str | int | None,
    conf: float,
    imgsz: int,
    batch: int | None,
    mem_fraction: float,
    p_cache_path: Path,
    p_hash: str,
    root: Path | None = None,
) -> list[dict]:
    """Infer per image, remap ids, and flush ``pseudo_labels.cache`` after each image (resumable)."""
    root = root or _dataset_root(im_files=im_files)
    gt_index = _index_by_image(gt_entries)
    existing = load_pseudo_progress(p_cache_path, p_hash, im_files=im_files, root=root)
    entries_by_key: dict[str, dict] = {}
    for lb in existing:
        im = str(lb["im_file"])
        entries_by_key[im] = lb
        entries_by_key[Path(im).name] = lb

    done_n = sum(1 for im in im_files if im in entries_by_key or Path(im).name in entries_by_key)
    pending = [im for im in im_files if im not in entries_by_key and Path(im).name not in entries_by_key]
    if not pending:
        LOGGER.info(f"{colorstr('WeDetect pseudo:')} reuse complete teacher cache {p_cache_path}")
        return _ordered_pseudo_labels(im_files, entries_by_key)

    if done_n:
        LOGGER.info(
            f"{colorstr('WeDetect pseudo:')} resume {p_cache_path.name}: "
            f"{done_n}/{len(im_files)} done, {len(pending)} remaining"
        )

    pbar = TQDM(
        total=len(im_files),
        initial=done_n,
        desc=f"{colorstr('WeDetect pseudo:')} {p_cache_path.name}",
        unit="img",
    )
    since_flush = 0

    def _flush(*, force: bool = False) -> None:
        nonlocal since_flush
        if not force and since_flush < PSEUDO_FLUSH_EVERY:
            return
        labels = _ordered_pseudo_labels(im_files, entries_by_key)
        n = len(labels)
        save_label_cache(
            p_cache_path,
            labels,
            p_hash,
            complete=n >= len(im_files),
            quiet=True,
            extra={
                "n_total": len(im_files),
                "n_done": n,
                "path_mode": PORTABLE_HASH_MODE,
            },
        )
        since_flush = 0

    def _commit(im: str, pred_local: np.ndarray) -> None:
        nonlocal since_flush
        gt_lb = _lookup_label(gt_index, im)
        if gt_lb is not None:
            shape = tuple(gt_lb["shape"])
        else:
            try:
                shape = _image_shape_hw(im)
            except Exception:
                shape = (0, 0)
        entry = pred_local_to_pseudo_entry(im, pred_local, shape, nc_gt)
        was_new = im not in entries_by_key and Path(im).name not in entries_by_key
        entries_by_key[im] = entry
        entries_by_key[Path(im).name] = entry
        if was_new:
            since_flush += 1
            pbar.update(1)
            _flush(force=False)

    try:
        run_teacher_inference(
            model_path=model_path,
            im_files=pending,
            kept_src_ids=kept_src_ids,
            kept_zh=kept_zh,
            kept_en=kept_en,
            device=device,
            conf=conf,
            imgsz=imgsz,
            batch=batch,
            mem_fraction=mem_fraction,
            on_image=_commit,
        )
        _flush(force=True)
    finally:
        pbar.close()

    LOGGER.info(
        f"{colorstr('WeDetect pseudo:')} wrote {p_cache_path} "
        f"({len(im_files)} images, resume-safe, flush_every={PSEUDO_FLUSH_EVERY}, portable hash)"
    )
    return _ordered_pseudo_labels(im_files, entries_by_key)


def merge_gt_and_pseudo_entries(gt_entries: list[dict], pseudo_entries: list[dict]) -> tuple[list[dict], int]:
    """Concatenate GT + already-remapped pseudo boxes per image. Returns (merged, n_pseudo_boxes)."""
    pseudo_index = _index_by_image(pseudo_entries)
    merged = []
    n_pseudo = 0
    for gt in gt_entries:
        im = str(gt["im_file"])
        ps = _lookup_label(pseudo_index, im)
        gt_cls = np.asarray(gt["cls"], dtype=np.float32).reshape(-1, 1)
        gt_boxes = np.asarray(gt["bboxes"], dtype=np.float32).reshape(-1, 4)
        if ps is not None and len(ps["cls"]):
            p_cls = np.asarray(ps["cls"], dtype=np.float32).reshape(-1, 1)
            p_boxes = np.asarray(ps["bboxes"], dtype=np.float32).reshape(-1, 4)
            n_pseudo += len(p_cls)
            cls = np.concatenate([gt_cls, p_cls], 0) if len(gt_cls) else p_cls
            boxes = np.concatenate([gt_boxes, p_boxes], 0) if len(gt_boxes) else p_boxes
        else:
            cls, boxes = gt_cls, gt_boxes
        merged.append(
            _label_entry(
                im,
                tuple(gt["shape"]),
                cls,
                boxes,
                segments=list(gt.get("segments") or []),
                keypoints=gt.get("keypoints"),
            )
        )
    return merged, n_pseudo


def build_merged_pseudo_cache(
    data: dict,
    im_files: list[str],
    gt_entries: list[dict],
    pseudo_entries: list[dict],
    root: Path | None = None,
) -> dict:
    """Merge GT+pseudo and write ``labels_pseudo_merged.cache``; return cache dict."""
    root = root or _dataset_root(data, im_files)
    merged_labels, _ = merge_gt_and_pseudo_entries(gt_entries, pseudo_entries)
    path = merged_cache_path(im_files)
    h = merged_cache_hash(im_files, root=root)
    return save_label_cache(path, merged_labels, h, extra={"path_mode": PORTABLE_HASH_MODE})


def rebuild_merged_pseudo_cache(data: dict, im_files: list[str]) -> dict:
    """Rebuild merged cache from existing ``pseudo_labels.cache`` + GT (no teacher).

    Used by ``YOLODataset.get_labels`` when ``labels_pseudo_merged.cache`` is missing/stale.
    Remaps cached ``im_file`` paths so caches copied across machines still work.
    """
    if not im_files:
        raise RuntimeError("rebuild_merged_pseudo_cache: empty im_files")
    nc_gt = int(data.get("nc") or len(data.get("names") or {}) or 0)
    root = _dataset_root(data, im_files)
    meta_path = (root / PSEUDO_META_NAME) if root is not None else Path(PSEUDO_META_NAME)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("nc_gt") is not None:
                nc_gt = int(meta["nc_gt"])
        except Exception:
            pass

    p_cache = pseudo_only_cache_path(data, im_files)
    if not p_cache.exists():
        raise FileNotFoundError(
            f"Missing {p_cache}. Re-run training with pseudo_label=True to regenerate teacher pseudo cache."
        )
    try:
        pseudo_cache = load_dataset_cache_file(p_cache)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load pseudo cache {p_cache}: {e}") from e
    if pseudo_cache.get("version") != DATASET_CACHE_VERSION or not pseudo_cache.get("labels"):
        raise RuntimeError(f"Invalid pseudo cache at {p_cache}; regenerate with pseudo_label=True")

    gt_entries = load_gt_label_entries(im_files, nc_gt=nc_gt)
    if not gt_entries:
        raise RuntimeError("No GT labels available to merge with pseudo cache")

    remapped = remap_label_im_files(list(pseudo_cache["labels"]), im_files, root)
    pseudo_index = _index_by_image(remapped)
    pseudo_entries = []
    for im in im_files:
        ps = _lookup_label(pseudo_index, im)
        if ps is None:
            try:
                shape = _image_shape_hw(im)
            except Exception:
                continue
            pseudo_entries.append(_label_entry(im, shape, _empty_cls(), _empty_xywh()))
        else:
            e = dict(ps)
            e["im_file"] = im
            pseudo_entries.append(e)

    cache = build_merged_pseudo_cache(data, im_files, gt_entries, pseudo_entries, root=root)
    LOGGER.info(f"{colorstr('WeDetect pseudo:')} rebuilt merged cache -> {merged_cache_path(im_files)}")
    return cache


def apply_pseudo_labels_to_subset(
    data: dict,
    args: Any,
    device: str | int | None = None,
) -> dict:
    """Generate/merge pseudo labels for one YOLO subset data dict; mutate and return it.

    Config resolution (see ``resolve_pseudo_cfg``): dataset YAML keys override train args.

    Pipeline (fixed order):
      1) resolve conflicts + build vocabulary (GT prefix, pseudo appended)
      2) write merged texts to ``<stem>_train.json`` (source JSON untouched)
      3) teacher inference -> ``pseudo_labels.cache``
      4) merge GT + pseudo -> ``labels_pseudo_merged.cache`` (no per-image txt)
    """
    cfg = resolve_pseudo_cfg(data, args)
    if not cfg["pseudo_label"]:
        return data

    model_path = cfg["pseudo_label_model"]
    classes_arg = cfg["pseudo_label_classes"]
    texts_arg = cfg["pseudo_label_class_texts"]
    conf = cfg["pseudo_label_conf"]
    imgsz = cfg["imgsz"]
    batch = None if int(cfg["pseudo_label_batch"]) <= 0 else int(cfg["pseudo_label_batch"])
    mem_fraction = cfg["pseudo_label_mem_fraction"]

    train_path = data.get("train")
    if not train_path:
        LOGGER.warning(f"{colorstr('WeDetect pseudo:')} subset has no train path; skip")
        return data

    LOGGER.info(
        f"{colorstr('WeDetect pseudo:')} config from dataset/args → "
        f"model={model_path}, conf={conf}, batch={batch or 'auto'}, "
        f"mem_fraction={mem_fraction}"
    )

    # --- 1) vocabulary ---
    en_groups = load_text_groups(classes_arg, DEFAULT_CLASSES)
    zh_groups = load_text_groups(texts_arg, DEFAULT_CLASS_TEXTS)
    vocab = build_merged_vocabulary(data, en_groups, zh_groups)
    nc_gt = vocab["nc_gt"]
    new_nc = vocab["new_nc"]
    new_names = vocab["new_names"]
    new_texts = vocab["new_texts"]
    kept_ids = vocab["kept_ids"]
    kept_en = vocab["kept_en"]
    kept_zh = vocab["kept_zh"]
    if not kept_ids:
        LOGGER.warning(f"{colorstr('WeDetect pseudo:')} no pseudo classes left after GT filtering; skip")
        return data

    source_texts_path = resolve_source_class_texts_path(data)
    class_texts_path = resolve_train_class_texts_path(data)
    im_files = collect_image_files(train_path)
    gt_label_files = img2label_paths(im_files, label_dir="labels")
    root = _dataset_root(data, im_files)
    out_root = root or Path(train_path).parent
    meta_path = out_root / PSEUDO_META_NAME
    p_cache_path = pseudo_only_cache_path(data, im_files)
    m_cache_path = merged_cache_path(im_files)
    m_hash = merged_cache_hash(im_files, root=root)
    p_hash = _pseudo_only_hash(
        im_files,
        model_path,
        classes_arg or str(DEFAULT_CLASSES),
        texts_arg or str(DEFAULT_CLASS_TEXTS),
        kept_ids,
        conf,
        nc_gt,
        root=root,
    )
    h = _meta_hash(
        im_files,
        [str(p) for p in gt_label_files],
        model_path,
        classes_arg or str(DEFAULT_CLASSES),
        texts_arg or str(DEFAULT_CLASS_TEXTS),
        str(class_texts_path),
        kept_ids,
        conf,
        new_texts,
        root=root,
    )

    # Full idempotent reuse (portable hash: same relative layout → hit across machines)
    if meta_path.exists() and class_texts_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            on_disk = _as_text_groups(json.loads(class_texts_path.read_text(encoding="utf-8")))
            merged_ok = try_load_cache(m_cache_path, m_hash) is not None
            pc = try_load_cache(p_cache_path, p_hash)
            if pc is not None:
                remapped_n = len(remap_label_im_files(list(pc.get("labels") or []), im_files, root))
            else:
                remapped_n = 0
            pseudo_ok = (
                pc is not None
                and remapped_n >= len(im_files)
                and pc.get("complete", True)
            )
            if old.get("hash") == h and on_disk == new_texts and merged_ok and pseudo_ok:
                LOGGER.info(
                    f"{colorstr('WeDetect pseudo:')} cache hit, reuse {m_cache_path.name} "
                    f"+ {p_cache_path.name}; class_texts={class_texts_path}"
                )
                data = deepcopy(data)
                data["names"] = {int(k): v for k, v in old.get("names", new_names).items()}
                data["nc"] = int(old.get("nc", new_nc))
                data["class_texts"] = str(class_texts_path)
                data["labels_dir"] = PSEUDO_LABELS_DIR
                return data
        except Exception as e:
            LOGGER.warning(f"{colorstr('WeDetect pseudo:')} meta load failed ({e}), regenerating")

    # --- 2) write train class_texts (source JSON untouched) ---
    write_class_texts(class_texts_path, new_texts)
    if source_texts_path != class_texts_path and source_texts_path.exists():
        LOGGER.info(
            f"{colorstr('WeDetect pseudo:')} source class_texts kept as-is -> {source_texts_path}"
        )

    # --- 3) GT entries (shapes + boxes); keep 1:1 with im_files for stable cache hashes ---
    gt_entries = load_gt_label_entries(im_files, nc_gt=nc_gt)
    if not gt_entries:
        LOGGER.warning(f"{colorstr('WeDetect pseudo:')} no GT entries; skip")
        return data

    # --- 4) pseudo-only cache: per-image flush + crash resume ---
    LOGGER.info(
        f"{colorstr('WeDetect pseudo:')} teacher conf={conf}; "
        f"vocab nc_gt={nc_gt} + pseudo={len(kept_zh)} (per-image cache flush, class-level dedup)"
    )
    pseudo_entries = generate_pseudo_cache_incremental(
        model_path=model_path,
        im_files=im_files,
        gt_entries=gt_entries,
        kept_src_ids=kept_ids,
        kept_zh=kept_zh,
        kept_en=kept_en,
        nc_gt=nc_gt,
        device=device,
        conf=conf,
        imgsz=imgsz,
        batch=batch,
        mem_fraction=mem_fraction,
        p_cache_path=p_cache_path,
        p_hash=p_hash,
        root=root,
    )

    # --- 5) merge -> labels_pseudo_merged.cache ---
    merged_labels, n_pseudo_boxes = merge_gt_and_pseudo_entries(gt_entries, pseudo_entries)
    save_label_cache(
        m_cache_path,
        merged_labels,
        m_hash,
        complete=True,
        extra={"path_mode": PORTABLE_HASH_MODE},
    )

    meta = {
        "hash": h,
        "model": model_path,
        "conf": conf,
        "dedup": "class_only",
        "format": "detect_xywhn",
        "pipeline": PSEUDO_CACHE_PIPELINE,
        "path_mode": PORTABLE_HASH_MODE,
        "nc_gt": nc_gt,
        "nc": new_nc,
        "kept_src_ids": kept_ids,
        "dropped_src_ids": vocab["dropped_src_ids"],
        "names": {str(k): v for k, v in new_names.items()},
        "class_texts_path": str(class_texts_path),
        "source_class_texts_path": str(source_texts_path),
        "new_texts_hash": _texts_hash(new_texts),
        "labels_dir": PSEUDO_LABELS_DIR,
        "pseudo_cache": str(p_cache_path),
        "pseudo_cache_hash": p_hash,
        "merged_cache": str(m_cache_path),
        "merged_cache_hash": m_hash,
        "n_images": len(im_files),
        "n_pseudo_boxes": n_pseudo_boxes,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(
        f"{colorstr('WeDetect pseudo:')} wrote {m_cache_path.name} "
        f"(+{len(kept_zh)} classes, ~{n_pseudo_boxes} pseudo boxes, nc={new_nc}); "
        f"class_texts(train) -> {class_texts_path}"
    )

    data = deepcopy(data)
    data["names"] = new_names
    data["nc"] = new_nc
    data["class_texts"] = str(class_texts_path)
    data["labels_dir"] = PSEUDO_LABELS_DIR
    return data


def maybe_build_pseudo_labels(trainer) -> None:
    """Entry point from WeDetectTrainer: apply pseudo labels to train subsets.

    Each subset is gated by ``resolve_pseudo_cfg`` (dataset YAML preferred over
    global train args). Safe under DDP when the caller wraps with
    ``torch_distributed_zero_first`` so rank-0 materializes files first.
    """
    args = trainer.args
    device = getattr(trainer, "device", None)
    device = str(device) if device is not None else getattr(args, "device", "")

    if getattr(trainer, "training_data", None):
        updated = {}
        any_enabled = False
        for key, d in trainer.training_data.items():
            if not isinstance(d, dict) or not d.get("train"):
                updated[key] = d
                continue
            if not subset_wants_pseudo(d, args):
                updated[key] = d
                continue
            any_enabled = True
            LOGGER.info(f"{colorstr('WeDetect pseudo:')} processing train subset '{key}'")
            updated[key] = apply_pseudo_labels_to_subset(d, args, device=device)
        trainer.training_data = updated
        if not any_enabled:
            LOGGER.info(
                f"{colorstr('WeDetect pseudo:')} skipped (no train subset has pseudo_label=True "
                f"in dataset YAML or train args)"
            )
        return

    # Single-dataset: keep val on original GT; train uses override in build_dataset
    if isinstance(getattr(trainer, "data", None), dict) and trainer.data.get("train"):
        if not subset_wants_pseudo(trainer.data, args):
            return
        LOGGER.info(f"{colorstr('WeDetect pseudo:')} processing single train set")
        train_view = apply_pseudo_labels_to_subset(deepcopy(trainer.data), args, device=device)
        trainer._train_data_override = train_view

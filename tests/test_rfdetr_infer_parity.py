# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare Ultralytics RF-DETR vs official ``rfdetr`` under shared Ultralytics preprocessing.

Feeds the same LetterBox(scale_fill) + BGR→RGB + /255 tensor into both LWDETR modules, then
decodes with the same ``PostProcess`` and checks that xyxy / conf / cls are exactly equal.

Usage:
    python tests/test_rfdetr_infer_parity.py
    python tests/test_rfdetr_infer_parity.py --tasks detect --scales nano,small
    pytest tests/test_rfdetr_infer_parity.py -q

Weights default to ``pretrained_weights/{detect,segment,pose}/`` under the repo root, with
fallback to the Roboflow model cache.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.augment import LetterBox
from ultralytics.utils import ASSETS
from ultralytics.utils.torch_utils import init_seeds, select_device

# Official Roboflow weights (Apache open set; detect Plus XL/2XL omitted).
WEIGHT_CASES = (
    # task, scale, relative weight under pretrained_weights/, official variant class name
    ("detect", "nano", "detect/rf-detr-nano.pth", "RFDETRNano"),
    ("detect", "small", "detect/rf-detr-small.pth", "RFDETRSmall"),
    ("detect", "medium", "detect/rf-detr-medium.pth", "RFDETRMedium"),
    ("detect", "large", "detect/rf-detr-large-2026.pth", "RFDETRLarge"),
    ("segment", "nano", "segment/rf-detr-seg-nano.pt", "RFDETRSegNano"),
    ("segment", "small", "segment/rf-detr-seg-small.pt", "RFDETRSegSmall"),
    ("segment", "medium", "segment/rf-detr-seg-medium.pt", "RFDETRSegMedium"),
    ("segment", "large", "segment/rf-detr-seg-large.pt", "RFDETRSegLarge"),
    ("segment", "xlarge", "segment/rf-detr-seg-xlarge.pt", "RFDETRSegXLarge"),
    ("segment", "2xlarge", "segment/rf-detr-seg-xxlarge.pt", "RFDETRSeg2XLarge"),
    ("pose", "preview", "pose/rf-detr-keypoint-preview-xlarge.pth", "RFDETRKeypointPreview"),
)


@dataclass
class DetOut:
    """Decoded detections used for parity checks."""

    xyxy: torch.Tensor  # (N, 4)
    conf: torch.Tensor  # (N,)
    cls: torch.Tensor  # (N,)


def resolve_weight(rel: str) -> Path | None:
    """Resolve a weight path from local ``pretrained_weights`` or the Roboflow cache."""
    local = ROOT / "pretrained_weights" / rel
    if local.is_file():
        return local
    try:
        from rfdetr.assets.model_weights import get_model_cache_dir

        cached = Path(get_model_cache_dir()) / Path(rel).name
        if cached.is_file():
            return cached
    except Exception:
        pass
    return None


def ultralytics_preprocess(bgr: np.ndarray, imgsz: int, device: torch.device) -> torch.Tensor:
    """Apply RFDETRPredictor-equivalent preprocessing: LetterBox stretch, RGB, /255, BCHW."""
    letterbox = LetterBox(imgsz, auto=False, scale_fill=True)
    im = letterbox(image=bgr)
    im = im[..., ::-1]  # BGR → RGB
    im = im.transpose(2, 0, 1)  # HWC → CHW
    im = np.ascontiguousarray(im)
    return torch.from_numpy(im).float().div_(255.0).unsqueeze(0).to(device)


def _as_pred_dict(preds) -> dict:
    """Normalize LWDETR forward outputs to a prediction dict."""
    if isinstance(preds, dict):
        return preds
    if isinstance(preds, tuple):
        out = {"pred_boxes": preds[0], "pred_logits": preds[1]}
        if len(preds) == 3:
            # Mask or keypoint head — PostProcess inspects keys / config.
            third = preds[2]
            if isinstance(third, dict):
                out.update(third)
            else:
                out["pred_masks"] = third
        return out
    raise TypeError(f"Unexpected RF-DETR output type: {type(preds)!r}")


@torch.inference_mode()
def decode_dets(module, batch: torch.Tensor, orig_hw: tuple[int, int], conf: float, max_det: int, model_config) -> DetOut:
    """Forward LWDETR and decode with official PostProcess (shared by both sides)."""
    from rfdetr.models.postprocess import PostProcess

    module.eval()
    preds = _as_pred_dict(module(batch))
    target_sizes = torch.tensor([orig_hw], device=batch.device, dtype=batch.dtype)
    outputs = PostProcess(
        num_select=int(getattr(model_config, "num_select", max_det)),
        num_keypoints_per_class=list(getattr(model_config, "num_keypoints_per_class", None) or []),
        trace_alpha=float(getattr(model_config, "postprocess_trace_alpha", 0.2)),
    )(preds, target_sizes)[0]
    keep = (outputs["scores"] > conf).nonzero().squeeze(1)[:max_det]
    return DetOut(
        xyxy=outputs["boxes"][keep].detach().float().cpu(),
        conf=outputs["scores"][keep].detach().float().cpu(),
        cls=outputs["labels"][keep].detach().long().cpu(),
    )


def load_ultralytics(weight: Path, device: torch.device):
    """Load Ultralytics RFDETR wrapper and return (lwdetr, model_config, imgsz)."""
    from ultralytics import RFDETR

    model = RFDETR(str(weight))
    wrapper = model.model
    lwdetr = wrapper.model.to(device).eval()
    cfg = wrapper.model_config
    imgsz = int(cfg.resolution)
    return lwdetr, cfg, imgsz


def load_official(variant: str, weight: Path, device: torch.device):
    """Load official rfdetr variant and return (lwdetr, model_config)."""
    import rfdetr

    cls = getattr(rfdetr, variant)
    model = cls(pretrain_weights=str(weight))
    # Official context keeps weights on CPU until first device move.
    ctx = model.model
    lwdetr = ctx.model.to(device).eval()
    cfg = model.model_config
    return lwdetr, cfg


def compare_dets(a: DetOut, b: DetOut, atol: float = 0.0, rtol: float = 0.0) -> list[str]:
    """Return a list of mismatch descriptions (empty means exact/allclose match)."""
    errs: list[str] = []
    if a.cls.shape != b.cls.shape:
        errs.append(f"count {a.cls.shape[0]} vs {b.cls.shape[0]}")
        return errs
    if a.cls.numel() == 0:
        return errs
    if not torch.equal(a.cls, b.cls):
        errs.append(f"cls max|Δ|={(a.cls - b.cls).abs().max().item()}")
    if not torch.allclose(a.conf, b.conf, atol=atol, rtol=rtol, equal_nan=False):
        errs.append(f"conf max|Δ|={(a.conf - b.conf).abs().max().item():.6g}")
    if not torch.allclose(a.xyxy, b.xyxy, atol=atol, rtol=rtol, equal_nan=False):
        errs.append(f"bbox max|Δ|={(a.xyxy - b.xyxy).abs().max().item():.6g}")
    return errs


def iter_images(paths: list[Path] | None = None) -> list[Path]:
    """Collect asset images (bus.jpg / zidane.jpg by default)."""
    if paths:
        return [Path(p) for p in paths]
    return sorted(p for p in Path(ASSETS).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def run_case(
    task: str,
    scale: str,
    weight: Path,
    variant: str,
    image: Path,
    device: torch.device,
    conf: float,
    max_det: int,
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    """Run one (model, image) parity check. Returns (ok, message)."""
    tag = f"{task}/{scale} @ {image.name}"
    try:
        ultra_mod, ultra_cfg, imgsz = load_ultralytics(weight, device)
        off_mod, off_cfg = load_official(variant, weight, device)
    except Exception as exc:  # noqa: BLE001 — report and continue matrix
        return False, f"FAIL  {tag}: load error: {exc}"

    bgr = cv2.imread(str(image))
    if bgr is None:
        return False, f"FAIL  {tag}: cannot read {image}"
    orig_hw = (bgr.shape[0], bgr.shape[1])
    batch = ultralytics_preprocess(bgr, imgsz, device)

    ultra = decode_dets(ultra_mod, batch, orig_hw, conf, max_det, ultra_cfg)
    official = decode_dets(off_mod, batch, orig_hw, conf, max_det, off_cfg)
    errs = compare_dets(ultra, official, atol=atol, rtol=rtol)
    if errs:
        return False, (
            f"FAIL  {tag} imgsz={imgsz} N={ultra.cls.numel()}/{official.cls.numel()}: " + "; ".join(errs)
        )
    return True, f"PASS  {tag} imgsz={imgsz} N={ultra.cls.numel()} (bbox/conf/cls identical)"


def filter_cases(tasks: set[str] | None, scales: set[str] | None):
    """Yield configured cases that exist on disk and match filters."""
    for task, scale, rel, variant in WEIGHT_CASES:
        if tasks and task not in tasks:
            continue
        if scales and scale not in scales:
            continue
        weight = resolve_weight(rel)
        if weight is None:
            print(f"SKIP  {task}/{scale}: weight not found ({rel})")
            continue
        yield task, scale, weight, variant


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run the parity matrix and return process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=str, default="", help="Comma list: detect,segment,pose (default: all)")
    parser.add_argument("--scales", type=str, default="", help="Comma list of scales (default: all available)")
    parser.add_argument("--conf", type=float, default=0.25, help="Score threshold after PostProcess")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections kept after threshold")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance (0 = exact float match)")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance")
    parser.add_argument("--device", type=str, default="", help="cuda / cpu / empty=auto")
    parser.add_argument("--images", type=str, default="", help="Comma-separated image paths (default: ASSETS)")
    args = parser.parse_args(argv)

    init_seeds(0, deterministic=True)
    device = select_device(args.device or None)
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()} or None
    scales = {s.strip() for s in args.scales.split(",") if s.strip()} or None
    images = iter_images([p.strip() for p in args.images.split(",") if p.strip()] or None)
    if not images:
        print(f"No images under {ASSETS}")
        return 2

    print(f"Device={device}  conf={args.conf}  atol={args.atol}  rtol={args.rtol}")
    print(f"Images={[p.name for p in images]}")
    print("Preprocess=Ultralytics LetterBox(scale_fill=True) + RGB + /255 (no ImageNet Normalize)")

    passed = failed = skipped_load = 0
    for task, scale, weight, variant in filter_cases(tasks, scales):
        for image in images:
            ok, msg = run_case(
                task,
                scale,
                weight,
                variant,
                image,
                device,
                conf=args.conf,
                max_det=args.max_det,
                atol=args.atol,
                rtol=args.rtol,
            )
            print(msg)
            if ok:
                passed += 1
            else:
                failed += 1
                if msg.startswith("FAIL") and "load error" in msg:
                    skipped_load += 1

    print(f"\nSummary: pass={passed} fail={failed} (load-related fails≈{skipped_load})")
    return 0 if failed == 0 else 1


def test_rfdetr_infer_parity_detect_nano():
    """Pytest smoke: detect nano on asset images must match official under Ultralytics preprocess."""
    code = main(["--tasks", "detect", "--scales", "nano"])
    assert code == 0


if __name__ == "__main__":
    raise SystemExit(main())

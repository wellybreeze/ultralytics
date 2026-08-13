# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Roboflow RF-DETR, a DINOv2-based real-time detection transformer.

RF-DETR provides detection, instance segmentation, and keypoint preview models through the same Ultralytics Model /
task_map engine used by RT-DETR. The neural network is built by the optional ``rfdetr`` package.

References:
    https://arxiv.org/abs/2511.09554
    https://github.com/roboflow/rf-detr
"""

import inspect
import sys
from pathlib import Path

import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import (
    RFDETRDetectionModel,
    RFDETRPoseModel,
    RFDETRSegmentationModel,
    guess_model_task,
    yaml_model_load,
)
from ultralytics.utils import LOGGER, RANK, ROOT, YAML, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.files import file_size

from .build import _require_rfdetr, build_rfdetr_model, get_config_class, rfdetr_class_names
from .predict import RFDETRPredictor
from .train import DEFAULT_CFG_RFDETR, RFDETRTrainer
from .val import RFDETRValidator

# Lazy-safe copy of official RF-DETR hyps for ``model.args`` (same file RFDETRTrainer uses as cfg).
DEFAULT_CFG_RFDETR_DICT = YAML.load(
    DEFAULT_CFG_RFDETR if Path(DEFAULT_CFG_RFDETR).is_file() else ROOT / "cfg/default.yaml"
)


class RFDETR(Model):
    """Interface for Roboflow RF-DETR detection, segmentation, and keypoint models.

    Args:
        model (str): Native RF-DETR weights or an RF-DETR YAML model configuration.
        task (str, optional): Explicit task. If None, inferred like YOLO via ``guess_model_task`` (architecture head,
            else filename cues such as ``-seg`` / ``-pose``).
        accept_platform_model_license (bool): Acknowledge the Platform Model License required by RF-DETR Plus models.

    Examples:
        >>> from ultralytics import RFDETR
        >>> model = RFDETR("rfdetr-nano.pt")
        >>> results = model("image.jpg", imgsz=384)
    """

    def __init__(self, model="rfdetr-nano.pt", task=None, accept_platform_model_license=False):
        """Initialize RF-DETR after checking its Python, PyTorch, package, and model-license requirements."""
        if sys.version_info < (3, 10):
            raise ImportError("RF-DETR requires Python>=3.10.")
        if not check_version(torch.__version__, "2.2.0"):
            raise ImportError("RF-DETR requires torch>=2.2.")
        _require_rfdetr()
        stem = Path(model).stem.lower().replace("_", "-")
        # Detect Plus only (PML). Seg-XL/2XL and keypoint-preview-xlarge are Apache open weights.
        is_plus = (
            ("xlarge" in stem or "2xlarge" in stem or "xxlarge" in stem)
            and "seg" not in stem
            and "keypoint" not in stem
            and "pose" not in stem
        )
        if is_plus:
            if not accept_platform_model_license:
                raise ImportError(
                    "RF-DETR xlarge and 2xlarge require the Platform Model License. "
                    "Pass accept_platform_model_license=True after reviewing and accepting it."
                )
            try:
                import rfdetr_plus  # noqa: F401
            except ImportError as exc:
                raise ImportError("RF-DETR xlarge and 2xlarge require `pip install rfdetr-plus`.") from exc
        self.accept_platform_model_license = accept_platform_model_license
        # Task inference matches YOLO: Model._new/_load → guess_model_task (head or filename -seg/-pose).
        super().__init__(model=model, task=task)

    def _new(self, cfg, task=None, model=None, verbose=False):
        """Build a native RF-DETR wrapper from a unified per-task RF-DETR YAML."""
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        model_class = self.task_map[self.task]["model"]
        self.model = (model or model_class)(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task
        # imgsz / hyps from default-rfdetr.yaml / caller kwargs (not model YAML), same pattern as YOLO.
        self.model.args = {**DEFAULT_CFG_RFDETR_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights, task=None):
        """Load an Ultralytics training checkpoint or a native RF-DETR `.pth`/`.pt` weight file.

        Ultralytics ``best.pt`` / ``last.pt`` pickle an ``nn.Module`` under ``ema``/``model`` (same shape as YOLO).
        Native Roboflow assets store a state-dict under ``model`` and are loaded via ``rfdetr``.
        """
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils.patches import torch_load

        requested = Path(str(weights).strip())
        if requested.suffix == ".pt" and requested.is_file():
            try:
                peek = torch_load(str(requested), map_location="cpu")
            except Exception:
                peek = None
            # YOLO-style train/export checkpoints pickle Modules; native RF-DETR .pt files use state-dicts.
            candidate = peek.get("ema") or peek.get("model") if isinstance(peek, dict) else None
            if isinstance(candidate, torch.nn.Module):
                self.model, self.ckpt = load_checkpoint(weights)
                self.task = task or getattr(self.model, "task", guess_model_task(weights))
                self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
                self.ckpt_path = getattr(self.model, "pt_path", str(requested))
                self.overrides["model"] = weights
                self.overrides["task"] = self.task
                self.model_name = str(weights)
                return

        from rfdetr.assets.model_weights import download_pretrain_weights, get_model_cache_dir

        self.task = task or guess_model_task(weights)
        config_class = get_config_class(str(weights))
        default_name = Path(str(config_class.model_fields["pretrain_weights"].default)).name
        native_weights = (
            str(requested.resolve()) if requested.is_file() else str(Path(get_model_cache_dir()) / default_name)
        )
        if not Path(native_weights).is_file():
            download_pretrain_weights(default_name)
            native_weights = str(Path(get_model_cache_dir()) / default_name)
        native_model, model_config, class_names = build_rfdetr_model(cfg=str(weights), weights=native_weights)
        model_class = self.task_map[self.task]["model"]
        wrapper = model_class.__new__(model_class)
        torch.nn.Module.__init__(wrapper)
        wrapper.yaml = {
            "nc": model_config.num_classes,
            "channels": 3,
            "scale": Path(weights).stem,
            "yaml_file": str(weights),
        }
        wrapper.model, wrapper.model_config = native_model, model_config
        # COCO pretrained checkpoints use sparse category IDs (1=person, 6=bus, …), not 0..nc-1.
        wrapper.names = rfdetr_class_names(model_config.num_classes, class_names)
        wrapper.nc = model_config.num_classes
        block = int(model_config.patch_size) * int(model_config.num_windows)
        wrapper.stride = torch.Tensor([block])
        wrapper.task, wrapper.inplace, wrapper.end2end = self.task, True, False
        # Truthy stub so Model.train() reuses this wrapper instead of torch-loading a native .pt name.
        self.model, self.ckpt = wrapper, {"epoch": -1}
        self.overrides = self.model.args = {**DEFAULT_CFG_RFDETR_DICT, "model": str(weights), "task": self.task}
        self.ckpt_path = native_weights
        self.model_name = str(weights)

    def export(self, format="onnx", **kwargs):
        """Export RF-DETR to ONNX by tracing the wrapped LWDETR module.

        Args:
            format (str): Export format. Currently only ``onnx`` is supported for RF-DETR.
            **kwargs (Any): Extra Exporter overrides such as ``imgsz``, ``opset``, ``simplify``.

        Returns:
            (str): Path to the exported model file.
        """
        if format.lower() not in {"onnx", ""}:
            raise NotImplementedError(f"RF-DETR currently supports format='onnx', received '{format}'.")
        self._check_is_pytorch_model()

        from ultralytics.utils.checks import check_imgsz

        imgsz = kwargs.get(
            "imgsz",
            self.overrides.get("imgsz", getattr(self.model.model_config, "resolution", 640)),
        )
        stride = int(self.model.stride.max())
        imgsz = check_imgsz(imgsz, stride=stride, floor=stride, min_dim=2)
        h, w = (int(imgsz[0]), int(imgsz[-1])) if isinstance(imgsz, (list, tuple)) else (int(imgsz), int(imgsz))
        opset = int(kwargs.get("opset", 17))
        path = Path(kwargs.get("filename") or f"{Path(self.model_name).stem}.onnx")
        lwdetr = self.model.model.eval().cpu()

        class _ExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                if isinstance(out, dict):
                    return out["pred_logits"], out["pred_boxes"]
                return out[0], out[1]

        wrapper = _ExportWrapper(lwdetr)
        if hasattr(lwdetr, "export") and callable(lwdetr.export):
            lwdetr.export()
        dummy = torch.zeros(1, 3, h, w)
        LOGGER.info(f"{colorstr('ONNX:')} starting export with opset={opset}, imgsz={h}x{w}")
        export_kwargs = {}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            opset_version=opset,
            input_names=["images"],
            output_names=["pred_logits", "pred_boxes"],
            dynamic_axes={"images": {0: "batch"}, "pred_logits": {0: "batch"}, "pred_boxes": {0: "batch"}},
            do_constant_folding=True,
            **export_kwargs,
        )
        LOGGER.info(f"{colorstr('ONNX:')} export success ✅ {file_size(path):.1f} MB, saved as '{path}'")
        return str(path)

    @property
    def task_map(self):
        """Return a task map for RF-DETR, associating tasks with Ultralytics classes."""
        return {
            "detect": {
                "predictor": RFDETRPredictor,
                "validator": RFDETRValidator,
                "trainer": RFDETRTrainer,
                "model": RFDETRDetectionModel,
            },
            "segment": {
                "predictor": RFDETRPredictor,
                "validator": RFDETRValidator,
                "trainer": RFDETRTrainer,
                "model": RFDETRSegmentationModel,
            },
            "pose": {
                "predictor": RFDETRPredictor,
                "validator": RFDETRValidator,
                "trainer": RFDETRTrainer,
                "model": RFDETRPoseModel,
            },
        }

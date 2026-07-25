# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.dataset import attach_shared_neg_queue, build_global_class_texts
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import WeDetectModel, WeDetectUniModel
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, RANK, ROOT, YAML, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


def on_pretrain_routine_end(trainer) -> None:
    """Set up model classes and text encoder at the end of the pretrain routine."""
    from ultralytics.models.yolo.wedetect.val import prepare_wedetect_text_prompts, resolve_wedetect_class_names

    names = resolve_wedetect_class_names(trainer.data)
    # Prefer the registered / cached text encoder (needed for OV fine-tuning)
    ema_model = unwrap_model(trainer.ema.ema)
    prepare_wedetect_text_prompts(ema_model, names, device=trainer.device)
    # Keep the live model prompts in sync for any non-EMA eval hooks
    live = unwrap_model(trainer.model)
    if live is not ema_model:
        prepare_wedetect_text_prompts(live, names, device=trainer.device)


class WeDetectTrainer(DetectionTrainer):
    """Trainer for WeDetect open-vocabulary detection models.

    Extends DetectionTrainer to support training WeDetect models which use
    ConvNeXt backbone and XLM-RoBERTa text encoder.  Handles text embedding
    generation and caching to accelerate training with multi-modal data.

    Two fine-tuning modes are supported via configuration:
    - **Open-vocabulary (OV)**: Text encoder is kept online and updated with a
      reduced learning rate (``text_lr_mult``).  Set ``freeze_text_encoder=False``.
    - **Close-set (CS)**: Text encoder is discarded; pre-computed class embeddings
      are cached and injected into every batch.  Set ``freeze_text_encoder=True``
      or ``close_set=True``.

    WeDetect-specific config keys (set via YAML or CLI overrides):
        freeze_text_encoder (bool): Freeze the text encoder so it does not
            participate in training.  Default ``True`` (pre-compute & cache).
        text_lr_mult (float): Learning-rate multiplier for text-encoder
            parameters when ``freeze_text_encoder=False``.  Default ``0.01``
            (mirrors the original WeDetect ``backbone.text_model lr_mult=0.01``).
        close_set (bool): Close-set fine-tuning — discard the text encoder
            entirely and use pre-computed class embeddings.  Default ``False``.
        mask_refine (bool): Refine boxes from segmentation masks (original
            WeDetect ``mask2bbox``). Requires segment-format labels. Default ``False``.
        mix_global_texts (bool): When training on multiple YOLO multimodal
            subsets, merge class texts by synonym overlap and remap local ids.
            Default ``True``.
        use_neg_queue (bool): Share a WeDetect-style ``NegQueue`` across subsets
            so recently seen class names become dynamic negatives. Default ``False``.

    Attributes:
        text_embeddings (dict[str, torch.Tensor] | None): Cached text embeddings.
        training_data (dict | None): Per-subset data dicts when using mixed yaml.
        validation_data (dict | None): Per-val-subset data dicts when using mixed multi-val.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        # close_set implies cached-embedding / frozen text encoder
        if overrides.get("close_set", False):
            overrides["freeze_text_encoder"] = True
        self.training_data = None  # filled by get_dataset() for mixed yaml
        self.validation_data = None  # path -> data dict for mixed multi-val
        self._val_fitness_weights_cfg = None  # optional list from mixed yaml
        self._train_data_override = None  # single-dataset pseudo-label train view
        super().__init__(cfg, overrides, _callbacks)
        if getattr(self.args, "close_set", False):
            self.args.freeze_text_encoder = True
        self.text_embeddings = None
        # Original WeDetect LinearLR: 1000 iters from 0.001×lr → lr
        if int(getattr(self.args, "warmup_iters", 0) or 0) > 0:
            LOGGER.info(
                f"{colorstr('WeDetect:')} warmup_iters={self.args.warmup_iters}, "
                f"warmup_start_factor={getattr(self.args, 'warmup_start_factor', 0.0)} "
                f"(overrides warmup_epochs={self.args.warmup_epochs})"
            )

    @property
    def freeze_text_encoder(self) -> bool:
        """Whether the text encoder is frozen (cached embeddings) for this run."""
        return bool(getattr(self.args, "freeze_text_encoder", True) or getattr(self.args, "close_set", False))

    def _resolve_train_nc(self) -> int:
        """Return contrastive training class slots (TAL / head), capped at 80.

        Open-vocabulary fine-tuning may keep ``data.nc`` equal to annotated classes
        (e.g. 1 for vehicle-only) while ``class_texts`` is much longer for negative
        prompts. ``RandomLoadText`` remaps labels into ``[0, len(texts))``, so the
        assigner must use the text-slot count, not annotated ``nc`` alone.
        """
        nc = int(self.data.get("nc") or 80)
        override = getattr(self, "_train_data_override", None)
        if isinstance(override, dict) and override.get("nc"):
            nc = max(nc, int(override["nc"]))
        paths: list[str] = []
        if self.data.get("class_texts"):
            paths.append(str(self.data["class_texts"]))
        if isinstance(override, dict) and override.get("class_texts"):
            paths.append(str(override["class_texts"]))
        if self.training_data:
            for d in self.training_data.values():
                if isinstance(d, dict) and d.get("class_texts"):
                    paths.append(str(d["class_texts"]))
                if isinstance(d, dict) and d.get("nc"):
                    nc = max(nc, int(d["nc"]))
        for path_str in paths:
            p = Path(path_str)
            if not p.is_file():
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    n_texts = len(json.load(f))
                nc = max(nc, n_texts)
            except Exception:
                continue
        nc = min(max(nc, 1), 80)
        if nc != int(self.data.get("nc") or 0):
            LOGGER.info(
                f"{colorstr('WeDetect:')} train text slots nc={nc} "
                f"(annotated data.nc={self.data.get('nc')}; matches RandomLoadText / OV negatives)"
            )
        return nc

    def get_model(self, cfg=None, weights: str | None = None, verbose: bool = True) -> WeDetectModel:
        """Return WeDetectModel initialized with specified config and weights."""
        # .pt checkpoints store architecture as a plain yaml dict (no yaml_file key);
        # WeDetectModel accepts either a path string or that dict.
        if isinstance(cfg, dict):
            model_cfg = cfg.get("yaml_file", cfg)
        else:
            model_cfg = cfg
        model = WeDetectModel(
            model_cfg,
            ch=self.data["channels"],
            nc=self._resolve_train_nc(),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        return model

    def setup_model(self):
        """Load model and register the text encoder for open-vocabulary fine-tuning.

        Overrides the base ``setup_model`` (not ``_setup_model``, which does not
        exist on DetectionTrainer) so the language model is attached before DDP
        wrapping and optimizer construction.
        """
        ckpt = super().setup_model()
        if not self.freeze_text_encoder:
            unwrap_model(self.model).register_text_model()
            LOGGER.info(
                f"Text encoder registered as submodule (freeze_text_encoder=False, "
                f"text_lr_mult={getattr(self.args, 'text_lr_mult', 0.01)})"
            )
        return ckpt

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with optional text-encoder learning-rate multiplier.

        When ``freeze_text_encoder=False``, text-encoder parameters are placed
        in a separate parameter group with ``lr = lr * text_lr_mult``
        (matching original WeDetect ``backbone.text_model lr_mult=0.01``).

        This overrides the base ``build_optimizer`` to add a text-encoder
        parameter group when ``text_lr_mult != 1.0``.
        """
        from functools import partial

        from ultralytics.optim import MuSGD

        freeze_text_encoder = self.freeze_text_encoder
        text_lr_mult = getattr(self.args, "text_lr_mult", 0.01)
        if freeze_text_encoder or text_lr_mult == 1.0:
            return super().build_optimizer(model, name, lr, momentum, decay, iterations)

        g = [{}, {}, {}, {}, {}]  # weight, bn, bias, muon, text_encoder
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if not param.requires_grad:
                    continue
                is_text = "text_model" in fullname
                if is_text:
                    g[4][fullname] = param
                elif param.ndim >= 2 and use_muon:
                    g[3][fullname] = param
                elif "bias" in fullname:
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1][fullname] = param
                else:
                    g[0][fullname] = param

        import torch.optim as optim

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' not found in {optimizers}.")

        num_params = [len(g[0]), len(g[1]), len(g[2])]
        g[2] = {"params": list(g[2].values()), **optim_args, "param_group": "bias"}
        g[0] = {"params": list(g[0].values()), **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": list(g[1].values()), **optim_args, "weight_decay": 0.0, "param_group": "bn"}

        param_groups = [g[0], g[1], g[2]]
        if use_muon:
            num_params[0] = len(g[3])
            param_groups.append({"params": list(g[3].values()), **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"})

        n_text = len(g[4])
        if n_text:
            # Match original WeDetect: lr_mult only; keep global weight_decay on text params
            param_groups.append({
                "params": list(g[4].values()),
                "lr": lr * text_lr_mult,
                "weight_decay": decay,
                "param_group": "text_encoder",
                **{k: v for k, v in optim_args.items() if k != "lr" and k != "weight_decay"},
            })

        optimizer = getattr(optim, name, partial(MuSGD, muon=0.2, sgd=1.0))(params=param_groups)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
            + (f", {n_text} text_encoder(lr={lr * text_lr_mult:.2e}, decay={decay})" if n_text else "")
        )
        return optimizer

    @staticmethod
    def _is_mixed_data_cfg(data: dict) -> bool:
        """Return True if data yaml uses ``train.yolo_data`` / grounding mixed format."""
        train = data.get("train")
        return isinstance(train, dict) and bool(train.get("yolo_data") or train.get("grounding_data"))

    def get_dataset(self):
        """Load single-dataset or mixed (yolo_data + grounding_data) configurations."""
        from ultralytics.models.yolo.wedetect.pseudo_label import maybe_build_pseudo_labels

        raw = self.args.data
        if not isinstance(raw, dict):
            try:
                raw = YAML.load(check_file(raw))
            except Exception:
                data = super().get_dataset()
                self.training_data = None
                self.validation_data = None
                self.data = data  # required before pseudo-label hook
                with torch_distributed_zero_first(LOCAL_RANK):
                    maybe_build_pseudo_labels(self)
                return data
        if self._is_mixed_data_cfg(raw):
            data = self._get_mixed_dataset(raw)  # sets self.data
            with torch_distributed_zero_first(LOCAL_RANK):
                maybe_build_pseudo_labels(self)
            return data
        data = super().get_dataset()
        self.training_data = None
        self.validation_data = None
        self.data = data  # required before pseudo-label hook
        with torch_distributed_zero_first(LOCAL_RANK):
            maybe_build_pseudo_labels(self)
        return data

    @staticmethod
    def _resolve_val_img_path(d: dict) -> str:
        """Pick val image list path; LVIS-style configs prefer ``minival`` when present."""
        if d.get("minival") is not None:
            mv = d["minival"]
            if not Path(str(mv)).is_absolute():
                d["minival"] = str(Path(d["path"]) / mv)
            if "lvis" in str(d.get("val", "")).lower() or "lvis" in str(d.get("path", "")).lower():
                return str(d["minival"])
        return str(d["val"])

    @staticmethod
    def _val_metric_tag(d: dict, index: int) -> str:
        """Short tag used to prefix secondary val metrics in results.csv."""
        path = Path(str(d.get("path", f"val{index}")))
        tag = path.name.strip() or f"val{index}"
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)

    def _get_mixed_dataset(self, data_yaml: dict) -> dict:
        """Process mixed data config with ``yolo_data`` + optional ``grounding_data``.

        Each YOLO subset keeps its own ``names`` / ``class_texts``. At train time,
        ``build_global_class_texts`` merges vocabularies by synonym overlap so
        category-id schemes need not match across subsets.

        Multiple ``val.yolo_data`` entries are supported: each is evaluated every
        epoch with its own ``nc`` / ``names`` / ``class_texts``. ``best.pt`` uses
        the weighted-average fitness across all val sets (default equal weights;
        optional ``val_fitness_weights``). The **first** val set still supplies
        ``self.data`` metadata / default ``test_loader``.
        """
        from ultralytics.utils import DATASETS_DIR

        final_data = {}
        self.args.data = data_yaml
        assert data_yaml.get("train", False), "train dataset not found"
        assert data_yaml.get("val", False), "validation dataset not found"
        data = {
            k: [check_det_dataset(d) for d in v.get("yolo_data", [])]
            for k, v in data_yaml.items()
            if k in {"train", "val"}
        }
        assert data["val"], "validation dataset not found"
        assert data["train"], "train dataset not found"

        for s in {"train", "val"}:
            if s == "train":
                final_data[s] = [d["train"] for d in data[s]]
            else:
                final_data[s] = [self._resolve_val_img_path(d) for d in data[s]]
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
                for k in {"img_path", "json_file"}:
                    path = Path(g[k])
                    if not path.exists() and not path.is_absolute():
                        g[k] = str((DATASETS_DIR / g[k]).resolve())
            final_data[s] += grounding_data

        # Per-val metadata (multi-val); fitness combines all sets (equal weights by default)
        self.validation_data = {}
        for d, vpath in zip(data["val"], final_data["val"]):
            if self.args.single_cls:
                d = dict(d)
                d["names"] = {0: "object"}
                d["nc"] = 1
            self.validation_data[str(vpath)] = d
        # Optional per-set weights from mixed yaml: val_fitness_weights: [0.5, 0.3, 0.2]
        self._val_fitness_weights_cfg = data_yaml.get("val_fitness_weights")

        primary = data["val"][0]
        final_data["val"] = final_data["val"][0]  # default loader / BaseTrainer expects a single path
        final_data["nc"] = primary["nc"]
        final_data["names"] = primary["names"]
        final_data["path"] = primary["path"]
        final_data["channels"] = primary["channels"]
        if "class_texts" in primary:
            final_data["class_texts"] = primary["class_texts"]
        self.data = final_data
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        top_level_class_texts = data_yaml.get("class_texts")
        use_neg = bool(getattr(self.args, "use_neg_queue", False) or data_yaml.get("use_neg_queue", False))
        # Mixed-yaml top-level pseudo_* keys fill gaps on train subsets that omit them
        # (per-subset dataset YAML still wins when the key is already present).
        from ultralytics.models.yolo.wedetect.pseudo_label import PSEUDO_CFG_KEYS

        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            if top_level_class_texts and not d.get("class_texts"):
                ct = Path(str(top_level_class_texts))
                if ct.is_absolute():
                    d["class_texts"] = str(ct)
                else:
                    # Prefer mixed-yaml dir / cfg texts, not the primary dataset image root
                    mixed_yaml = getattr(self.args, "data", None)
                    mixed_dir = Path(mixed_yaml).parent if isinstance(mixed_yaml, str) else None
                    cands = []
                    if mixed_dir is not None:
                        cands.append(mixed_dir / ct)
                    cands.append(ROOT / "cfg" / "datasets" / "texts" / ct.name)
                    cands.append(ROOT / "cfg" / "datasets" / "customer" / ct.name)
                    cands.append(Path(d.get("path", ".")) / ct)
                    hit = next((c for c in cands if c.exists()), None)
                    d["class_texts"] = str((hit or cands[0]).resolve())
            for pk in PSEUDO_CFG_KEYS:
                if pk in data_yaml and (pk not in d or d.get(pk) is None or d.get(pk) == ""):
                    d[pk] = data_yaml[pk]
            if use_neg:
                d["use_neg_queue"] = True
            self.training_data[d["train"]] = d
        n_val = len(self.validation_data)
        w = self._val_fitness_weights(n_val)
        LOGGER.info(
            f"{colorstr('WeDetect:')} mixed data config with {len(data['train'])} YOLO "
            f"+ {len(final_data['train']) - len(data['train'])} grounding train subsets, "
            f"{n_val} val set(s) (fitness weights={['%.3f' % x for x in w]})"
        )
        # Pseudo-label hook runs in get_dataset() after this returns
        return final_data

    def _val_fitness_weights(self, n: int) -> list[float]:
        """Per-val-set fitness weights; default equal share. Normalized to sum=1.

        Source (first hit): mixed yaml ``val_fitness_weights`` stored at dataset load,
        or ``args.val_fitness_weights``. Length must match ``n`` or equal weights are used.
        """
        if n <= 0:
            return []
        raw = getattr(self, "_val_fitness_weights_cfg", None)
        if raw is None:
            raw = getattr(self.args, "val_fitness_weights", None)
        if isinstance(self.args.data, dict) and raw is None:
            raw = self.args.data.get("val_fitness_weights")
        try:
            w = [float(x) for x in list(raw)] if raw is not None else []
        except (TypeError, ValueError):
            w = []
        if len(w) != n or sum(w) <= 0:
            return [1.0 / n] * n
        s = sum(w)
        return [x / s for x in w]

    def validate(self):
        """Run validation; support multiple mixed ``val.yolo_data`` entries.

        Each val set is evaluated with its own ``nc`` / ``names`` / ``class_texts``.
        ``best.pt`` fitness is the **weighted average** of per-set fitness
        (default: equal weights). Per-set metrics are logged with a
        ``<dataset>/`` prefix; the first set's metrics also keep unprefixed keys
        for CSV/plot compatibility.
        """
        import torch.distributed as dist

        from ultralytics.utils import LOCAL_RANK

        val_items = list(self.validation_data.items()) if self.validation_data else []
        if len(val_items) <= 1:
            return super().validate()

        if self.ema and self.world_size > 1:
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)

        batch_size = self.batch_size // max(self.world_size, 1)
        if self.args.task not in {"obb", "semantic"}:
            batch_size *= 2

        primary_keys = ("nc", "names", "path", "class_texts", "channels", "val")
        saved = {k: self.data.get(k) for k in primary_keys}
        saved_loader = self.validator.dataloader
        saved_test_loader = self.test_loader

        metrics_all: dict[str, float] = {}
        fitnesses: list[float] = []
        tags: list[str] = []
        try:
            for i, (val_path, vdata) in enumerate(val_items):
                self.data["nc"] = vdata["nc"]
                self.data["names"] = vdata["names"]
                self.data["path"] = vdata["path"]
                self.data["channels"] = vdata.get("channels", saved.get("channels", 3))
                if vdata.get("class_texts"):
                    self.data["class_texts"] = vdata["class_texts"]
                elif "class_texts" in self.data:
                    self.data.pop("class_texts", None)
                self.data["val"] = val_path

                loader = self.get_dataloader(val_path, batch_size=batch_size, rank=LOCAL_RANK, mode="val")
                self.validator.dataloader = loader
                self.test_loader = loader

                tag = self._val_metric_tag(vdata, i)
                LOGGER.info(f"{colorstr('WeDetect val:')} [{i + 1}/{len(val_items)}] {tag} (nc={vdata['nc']})")
                metrics = self.validator(self)
                if metrics is None:  # non-zero DDP ranks
                    continue

                fitness = metrics.pop("fitness", None)
                if fitness is None:
                    fitness = (
                        float(-self.loss.detach().cpu().numpy()) if getattr(self, "loss", None) is not None else 0.0
                    )
                fit = float(fitness)
                fitnesses.append(fit)
                tags.append(tag)

                if i == 0:
                    # Keep unprefixed keys from the first set for plot/CSV compatibility
                    metrics_all.update(metrics)
                for k, v in metrics.items():
                    metrics_all[f"{tag}/{k}"] = v
                metrics_all[f"{tag}/fitness"] = fit
        finally:
            for k, v in saved.items():
                if v is None:
                    self.data.pop(k, None)
                else:
                    self.data[k] = v
            self.validator.dataloader = saved_loader
            self.test_loader = saved_test_loader

        if not fitnesses:
            return None, None

        weights = self._val_fitness_weights(len(fitnesses))
        combined = float(sum(w * f for w, f in zip(weights, fitnesses)))
        metrics_all["fitness"] = combined
        detail = ", ".join(f"{t}={f:.5f}(w={w:.3f})" for t, f, w in zip(tags, fitnesses, weights))
        LOGGER.info(f"{colorstr('WeDetect val:')} combined fitness={combined:.5f} ← {detail}")

        if self.best_fitness is None or self.best_fitness < combined:
            self.best_fitness = combined
        return metrics_all, combined

    def final_eval(self):
        """Final val on ``best.pt`` across all mixed val sets (equal-weight fitness).

        During training ``get_dataset`` replaces ``args.data`` with the mixed yaml
        dict. Standalone ``validator(model=best.pt)`` needs concrete yaml paths.
        """
        from ultralytics.utils.torch_utils import strip_optimizer

        val_items = list(self.validation_data.items()) if self.validation_data else []
        if len(val_items) <= 1:
            data = self.args.data
            if isinstance(data, dict):
                val = None
                if isinstance(data.get("val"), dict):
                    yolo_data = data["val"].get("yolo_data") or []
                    val = yolo_data[0] if yolo_data else None
                elif data.get("yaml_file"):
                    val = data["yaml_file"]
                if val:
                    self.validator.args.data = val
                    self.validator.args.split = (
                        "minival" if isinstance(val, str) and "lvis" in str(val).lower() else "val"
                    )
            return super().final_eval()

        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if not model:
            return

        LOGGER.info(f"\nValidating {model} on {len(val_items)} val sets...")
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False

        metrics_all: dict[str, float] = {}
        fitnesses: list[float] = []
        tags: list[str] = []
        for i, (_val_path, vdata) in enumerate(val_items):
            yaml_path = vdata.get("yaml_file")
            if not yaml_path and isinstance(self.args.data, dict):
                yolo_data = (self.args.data.get("val") or {}).get("yolo_data") or []
                yaml_path = yolo_data[i] if i < len(yolo_data) else None
            if not yaml_path:
                LOGGER.warning(f"{colorstr('WeDetect:')} skip final val set {i}: no yaml_file")
                continue
            self.validator.args.data = str(yaml_path)
            self.validator.args.split = "minival" if "lvis" in str(yaml_path).lower() else "val"
            tag = self._val_metric_tag(vdata, i)
            LOGGER.info(f"{colorstr('WeDetect val:')} final [{i + 1}/{len(val_items)}] {tag}")
            metrics = self.validator(model=model)
            if metrics is None:
                continue
            fit = float(metrics.pop("fitness", 0.0) or 0.0)
            fitnesses.append(fit)
            tags.append(tag)
            if i == 0:
                metrics_all.update(metrics)
            for k, v in metrics.items():
                metrics_all[f"{tag}/{k}"] = v
            metrics_all[f"{tag}/fitness"] = fit

        if fitnesses:
            weights = self._val_fitness_weights(len(fitnesses))
            combined = float(sum(w * f for w, f in zip(weights, fitnesses)))
            detail = ", ".join(f"{t}={f:.5f}(w={w:.3f})" for t, f, w in zip(tags, fitnesses, weights))
            LOGGER.info(f"{colorstr('WeDetect val:')} final combined fitness={combined:.5f} ← {detail}")
            # Match BaseTrainer.final_eval: drop fitness before CSV callback logging
            self.metrics = metrics_all
            self.epoch += 1
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1

    def build_dataset(self, img_path, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation (single or mixed)."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        # Mixed train: img_path is a list of YOLO paths and/or grounding dicts
        if mode == "train" and isinstance(img_path, list) and self.training_data is not None:
            datasets = [
                build_yolo_dataset(
                    self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True
                )
                if isinstance(im_path, str)
                else build_grounding(
                    self.args,
                    im_path["img_path"],
                    im_path["json_file"],
                    batch,
                    stride=gs,
                    max_samples=min(int(self.data.get("nc") or 80), 80),
                )
                for im_path in img_path
            ]
            if bool(getattr(self.args, "mix_global_texts", True)):
                build_global_class_texts(datasets)
            if bool(getattr(self.args, "use_neg_queue", False)):
                attach_shared_neg_queue(datasets, size=80)
            if self.freeze_text_encoder:
                self.set_text_embeddings(datasets, batch)
            return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

        # Single-dataset: train may use pseudo-label override (expanded vocab + labels_pseudo_merged)
        data = self.data
        if mode == "train" and getattr(self, "_train_data_override", None) is not None:
            data = self._train_data_override
        elif mode == "val" and isinstance(data, dict):
            # Ensure val reads original GT labels even if train override set labels_dir
            data = {**data, "labels_dir": "labels"}

        dataset = build_yolo_dataset(
            self.args, img_path, batch, data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )
        if mode == "train" and bool(getattr(self.args, "use_neg_queue", False)):
            attach_shared_neg_queue([dataset], size=80)
        if mode == "train" and self.freeze_text_encoder:
            self.set_text_embeddings([dataset], batch)
        return dataset

    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        """Set text embeddings for datasets to accelerate training."""
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names), batch, cache_dir=Path(dataset.img_path).parent
                )
            )
        self.text_embeddings = text_embeddings

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path) -> dict[str, torch.Tensor]:
        """Generate text embeddings for a list of text samples using XLM-RoBERTa."""
        model_variant = unwrap_model(self.model).text_model_variant
        cache_path = cache_dir / f"text_embeddings_{model_variant.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        assert self.model is not None
        pe = unwrap_model(self.model).get_text_pe(texts, batch, cache_text_model=True).squeeze(0)
        txt_map = dict(zip(texts, pe[: len(texts)]))
        torch.save(txt_map, cache_path)
        return txt_map

    def get_validator(self):
        """Return WeDetectValidator so val refreshes open-vocabulary text prompts each epoch."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        from copy import copy

        from ultralytics.models.yolo.wedetect.val import WeDetectValidator

        return WeDetectValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_labels(self):
        """Plot label distribution; skip for mixed ``YOLOConcatDataset`` (no flat ``.labels``)."""
        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        if isinstance(dataset, YOLOConcatDataset):
            LOGGER.info(f"{colorstr('WeDetect:')} skip plot_training_labels for YOLOConcatDataset")
            return
        return DetectionTrainer.plot_training_labels(self)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a batch of images and text for WeDetect training.

        - **OV fine-tuning** (``freeze_text_encoder=False``): leave ``texts`` in the
          batch; ``WeDetectModel.loss`` encodes them online so the language model
          receives gradients (DDP-safe).
        - **Cached / close-set**: inject precomputed ``txt_feats`` from the cache.
        """
        batch = DetectionTrainer.preprocess_batch(self, batch)
        if self.freeze_text_encoder:
            assert self.text_embeddings is not None, "text_embeddings cache is required when freeze_text_encoder=True"
            texts = list(itertools.chain(*batch["texts"]))
            txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch

    def save_model(self):
        """Save checkpoints and persist WeDetect ``text_model_weights`` side-channel.

        Syncs the live / EMA text encoder into ``_text_sd`` and writes the same
        dict as top-level ``text_model_weights`` so fine-tuned language weights
        match the pretrained checkpoint layout used by export and reload.
        """
        import io
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__
        from ultralytics.utils import GIT
        from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16

        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])

        # Sync LM on the live model first (source of OV updates), then copy onto EMA
        live = unwrap_model(self.model)
        if hasattr(live, "sync_text_model_weights"):
            live.sync_text_model_weights()
            if getattr(live, "_text_sd", None) is not None:
                ema._text_sd = deepcopy(live._text_sd)
        if hasattr(ema, "sync_text_model_weights"):
            ema.sync_text_model_weights()

        ema = deepcopy(ema).half()
        if hasattr(ema, "criterion"):
            ema.criterion = None
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)

        text_weights = None
        if hasattr(ema, "sync_text_model_weights"):
            text_weights = ema.sync_text_model_weights()
        elif getattr(ema, "_text_sd", None) is not None:
            text_weights = deepcopy(ema._text_sd)

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "text_model_weights": text_weights,
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True


class WeDetectTrainerFromScratch(WeDetectTrainer):
    """Trainer for WeDetect models from scratch / large-scale mixed open-set data.

    Mixed-dataset support (``yolo_data`` + ``grounding_data``, global text merge,
    optional NegQueue, mixed ``final_eval``) lives on :class:`WeDetectTrainer`.
    This subclass keeps the historical name and only skips label plotting.
    """

    def plot_training_labels(self):
        """Skip label plotting for WeDetect open-vocabulary training."""
        pass


class WeDetectUniTrainer(DetectionTrainer):
    """Trainer for WeDetect-Uni models with learnable prompt embeddings.

    Follows the original WeDetect ``SimpleYOLOWorldDetector`` pattern:
    learnable ``embeddings`` are used as text features directly, passed to
    the detection head's ``BNContrastiveHead`` for contrastive scoring.
    No text encoder is needed during training or inference.

    When pretrained weights are provided, the embeddings are initialised
    from the text encoder (XLM-RoBERTa) and the entire detector is
    frozen except for the embeddings (and optional MLP adapter).

    This is suitable for:
    - Fine-tuning WeDetect-Uni on a target domain
    - Training category-specific prompt embeddings
    - Prompt-free inference after training

    Attributes:
        loss_names (tuple): Names of loss components.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return WeDetectUniModel initialised with learnable prompt embeddings.

        Follows the original WeDetect ``SimpleYOLOWorldDetector``: embeddings
        are initialised from XLM-RoBERTa text encoder output and the rest of
        the detector is frozen.  The head uses ``BNContrastiveHead`` for
        contrastive scoring (no ``fuse`` operation).

        Args:
            cfg: Model configuration.
            weights: Path to pretrained weights.
            verbose: Whether to display model info.

        Returns:
            (WeDetectUniModel): Initialised model.
        """
        nc = self.data["nc"]
        if isinstance(cfg, dict):
            model_cfg = cfg.get("yaml_file", cfg)
        else:
            model_cfg = cfg
        model = WeDetectUniModel(
            model_cfg,
            ch=self.data["channels"],
            nc=nc,
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        names = list(self.data["names"].values())
        class_texts_path = self.data.get("class_texts")
        if class_texts_path:
            p = Path(class_texts_path)
            if p.exists():
                with open(p) as f:
                    raw = json.load(f)
                if isinstance(raw, list) and len(raw) >= nc:
                    names = [x[0] if isinstance(x, list) else x for x in raw[:nc]]
                    LOGGER.info(f"Using class_texts names for WeDetectUni initialization ({len(names)} classes)")
        tpe = model.get_text_pe(names)
        model.embeddings = nn.Parameter(tpe.squeeze(0).to(model.embeddings.device))
        model.embeddings.requires_grad_(True)

        for p in model.parameters():
            p.requires_grad_(False)
        model.embeddings.requires_grad_(True)

        model.train()
        return model

    def get_validator(self):
        """Return WeDetectUniValidator for WeDetect-Uni validation."""
        self.loss_names = "box", "cls", "dfl"
        from copy import copy

        from ultralytics.models.yolo.wedetect.val import WeDetectUniValidator

        return WeDetectUniValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """Preprocess a batch of images for WeDetect-Uni training.

        Unlike WeDetectTrainer, WeDetect-Uni does not inject text features
        into the batch.  The model uses its learnable embeddings directly.
        """
        return DetectionTrainer.preprocess_batch(self, batch)

    def set_text_embeddings(self, datasets, batch: int):
        """No-op override for prompt-free training that does not require text embeddings."""
        pass
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import WeDetectModel, WeDetectUniModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import unwrap_model


def on_pretrain_routine_end(trainer) -> None:
    """Set up model classes and text encoder at the end of the pretrain routine."""
    names = [name.split("/", 1)[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
    unwrap_model(trainer.ema.ema).set_classes(names, cache_clip_model=False)


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
      (the default for backward compatibility).

    WeDetect-specific config keys (set via YAML or CLI overrides):
        freeze_text_encoder (bool): Freeze the text encoder so it does not
            participate in training.  Default ``True`` (pre-compute & cache).
        text_lr_mult (float): Learning-rate multiplier for text-encoder
            parameters when ``freeze_text_encoder=False``.  Default ``0.01``
            (mirrors the original WeDetect ``backbone.text_model lr_mult=0.01``).
        close_set (bool): Close-set fine-tuning — discard the text encoder
            entirely and use pre-computed class embeddings.  Default ``False``.

    Attributes:
        text_embeddings (dict[str, torch.Tensor] | None): Cached text embeddings.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)
        self.text_embeddings = None

    def get_model(self, cfg=None, weights: str | None = None, verbose: bool = True) -> WeDetectModel:
        """Return WeDetectModel initialized with specified config and weights."""
        model = WeDetectModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        return model

    def _setup_model(self):
        """Set up model for training, including text encoder freezing and registration.

        Extends the base freeze logic to handle WeDetect-specific text encoder
        control via ``freeze_text_encoder`` and ``text_lr_mult`` config keys.

        When ``freeze_text_encoder=False`` (open-vocabulary mode):
          - The text encoder is registered as a submodule so its parameters
            participate in gradient computation and optimizer updates.
          - ``text_lr_mult`` controls the relative learning rate for text
            encoder parameters (default 0.01, matching original WeDetect).

        When ``freeze_text_encoder=True`` (close-set / cached-embedding mode):
          - The text encoder remains a plain Python attribute (not a submodule).
          - Text embeddings are pre-computed and cached to disk.
        """
        super()._setup_model()
        freeze_text_encoder = getattr(self.args, "freeze_text_encoder", True)
        if not freeze_text_encoder:
            unwrap_model(self.model).register_text_model()
            LOGGER.info(
                f"Text encoder registered as submodule (freeze_text_encoder=False, "
                f"text_lr_mult={getattr(self.args, 'text_lr_mult', 0.01)})"
            )

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with optional text-encoder learning-rate multiplier.

        When ``freeze_text_encoder=False``, text-encoder parameters are placed
        in a separate parameter group with ``lr = lr * text_lr_mult`` and
        ``weight_decay=0.0`` (matching original WeDetect's ``paramwise_cfg``).

        This overrides the base ``build_optimizer`` to add a 4th parameter
        group for text-encoder parameters when ``text_lr_mult != 1.0``.
        """
        from functools import partial

        from ultralytics.utils.torch_utils import MuSGD

        freeze_text_encoder = getattr(self.args, "freeze_text_encoder", True)
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
            param_groups.append({
                "params": list(g[4].values()),
                "lr": lr * text_lr_mult,
                "weight_decay": 0.0,
                "param_group": "text_encoder",
                **{k: v for k, v in optim_args.items() if k != "lr" and k != "weight_decay"},
            })

        optimizer = getattr(optim, name, partial(MuSGD, muon=0.2, sgd=1.0))(params=param_groups)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
            + (f", {n_text} text_encoder(lr={lr * text_lr_mult:.2e}, decay=0.0)" if n_text else "")
        )
        return optimizer

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )
        if mode == "train":
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
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, cache_text_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a batch of images and text for WeDetect training."""
        batch = DetectionTrainer.preprocess_batch(self, batch)
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch


def _build_global_texts(datasets) -> None:
    """Build a global text vocabulary across all YOLOMultiModalDataset instances.

    Collects all unique class texts from every dataset, builds a unified
    global text list, and sets up local-to-global ID mappings so that
    ``RandomLoadText`` samples from the shared vocabulary and the model
    learns a consistent semantic space across all datasets.

    This mirrors the original WeDetect ``WeConcatDataset.init_texts()`` logic
    where all category names are merged into a single global list.

    Args:
        datasets: List of dataset instances (YOLOMultiModalDataset or GroundingDataset).
    """
    from ultralytics.data.dataset import YOLOMultiModalDataset

    multimodal = [ds for ds in datasets if isinstance(ds, YOLOMultiModalDataset)]
    if not multimodal or len(multimodal) < 2:
        for ds in multimodal:
            LOGGER.info(f"Single dataset '{ds.img_path}', using local class_texts ({len(ds.class_texts)} classes)")
        return

    global_texts: list[list[str]] = []
    text_set: set[str] = set()
    for ds in multimodal:
        for ct in ds.class_texts:
            flat = [s.strip().lower() for s in ct]
            if any(s in text_set for s in flat):
                continue
            global_texts.append(ct)
            text_set.update(flat)

    for ds in multimodal:
        local_to_global: dict[int, int] = {}
        local_ct = ds.class_texts
        for local_id, texts in enumerate(local_ct):
            matched_name = texts[0].strip().lower()
            for global_id, gtexts in enumerate(global_texts):
                if any(t.strip().lower() == matched_name for t in gtexts):
                    local_to_global[local_id] = global_id
                    break
        unmatched = set(range(len(local_ct))) - set(local_to_global.keys())
        if unmatched:
            next_id = len(global_texts)
            for uid in sorted(unmatched):
                local_to_global[uid] = next_id
                global_texts.append(local_ct[uid])
                next_id += 1
        ds.set_global_texts(global_texts, local_to_global)

    total = sum(len(ds.class_texts) for ds in multimodal)
    LOGGER.info(
        f"Global text vocab: {len(global_texts)} unique classes "
        f"from {len(multimodal)} datasets ({total} total local classes)"
    )


class WeDetectTrainerFromScratch(WeDetectTrainer):
    """Trainer for WeDetect models from scratch on open-set datasets.

    Follows the ``WorldTrainerFromScratch`` pattern: supports mixed datasets
    including both object detection and grounding datasets.  Text embeddings
    are pre-computed and cached for all class names across datasets.

    This is suitable for:
    - Training a WeDetect model from scratch on large-scale datasets
    - Mixed training with YOLO detection + grounding data
    - Domain-specific fine-tuning with a fixed vocabulary

    Methods:
        build_dataset: Build datasets for training with grounding support.
        get_dataset: Process mixed data config with yolo_data + grounding_data.
        plot_training_labels: Skip label plotting for open-vocabulary training.
        final_eval: Configure validator for the correct dataset split.
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset for training or validation with mixed dataset support.

        For training mode, supports both standard YOLO datasets and grounding
        datasets.  For validation, builds a standard dataset.

        Args:
            img_path (list[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode.
            batch (int, optional): Size of batches, used for rectangular training.

        Returns:
            (YOLOConcatDataset | Dataset): The constructed dataset.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        datasets = [
            build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(
                self.args,
                im_path["img_path"],
                im_path["json_file"],
                batch,
                stride=gs,
                max_samples=self.data["nc"],
            )
            for im_path in img_path
        ]
        _build_global_texts(datasets)
        self.set_text_embeddings(datasets, batch)
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    @staticmethod
    def check_data_config(data: dict | str | Path) -> dict:
        """Check and load the data configuration from a YAML file or dictionary."""
        if not isinstance(data, dict):
            from ultralytics.utils import YAML

            return YAML.load(check_file(data))
        return data

    def get_dataset(self):
        """Get train and validation paths from data dictionary.

        Processes the data configuration to extract paths for training and
        validation datasets, handling both YOLO detection datasets and
        grounding datasets.

        Returns:
            (dict): Final processed data configuration.
        """
        from ultralytics.utils import DATASETS_DIR

        final_data = {}
        self.args.data = data_yaml = self.check_data_config(self.args.data)
        assert data_yaml.get("train", False), "train dataset not found"
        assert data_yaml.get("val", False), "validation dataset not found"
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in {"train", "val"}:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
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
        data["val"] = data["val"][0]
        final_data["val"] = final_data["val"][0]
        final_data["nc"] = data["val"]["nc"]
        final_data["names"] = data["val"]["names"]
        final_data["path"] = data["val"]["path"]
        final_data["channels"] = data["val"]["channels"]
        if "class_texts" in data["val"]:
            final_data["class_texts"] = data["val"]["class_texts"]
        self.data = final_data
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        top_level_class_texts = data_yaml.get("class_texts")
        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            if top_level_class_texts and not d.get("class_texts"):
                d["class_texts"] = str(
                    (Path(data["val"]["path"]) / top_level_class_texts).resolve()
                    if not Path(top_level_class_texts).is_absolute()
                    else Path(top_level_class_texts)
                )
            self.training_data[d["train"]] = d
        return final_data

    def plot_training_labels(self):
        """Skip label plotting for WeDetect open-vocabulary training."""
        pass

    def final_eval(self):
        """Perform final evaluation and validation for the WeDetect model."""
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()


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
        model = WeDetectUniModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
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
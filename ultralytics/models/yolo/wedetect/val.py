# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.data.utils import check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import select_device


from ultralytics.nn.tasks import WeDetectUniModel


class WeDetectValidator(DetectionValidator):
    """Validator for WeDetect open-vocabulary detection models.

    Handles text embedding setup for standalone validation so that
    ``model.val()`` works correctly without a trainer.  During training,
    the trainer's EMA model already has text embeddings set via the
    ``on_pretrain_routine_end`` callback.

    For WeDetect-Uni models, updates the head's ``nc`` to match the
    number of learnable prompt embeddings.
    """

    def __call__(self, trainer=None, model=None):
        """Set dataset classes for standalone validation, then run validation."""
        if trainer is not None:
            model = trainer.ema.ema
            if isinstance(model, WeDetectUniModel):
                nc = model.embeddings.shape[0]
                model.model[-1].nc = nc
                return super().__call__(trainer, model)
        else:
            self.device = select_device(self.args.device, verbose=False)
            if not isinstance(model, torch.nn.Module):
                from ultralytics.nn.tasks import load_checkpoint

                model = load_checkpoint(model or self.args.model, device=self.device)[0]
            model.eval().to(self.device)
            if isinstance(model, WeDetectUniModel):
                nc = model.embeddings.shape[0]
                model.model[-1].nc = nc
                return super().__call__(trainer, model)
            self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)
            names = [name.split("/", 1)[0] for name in check_det_dataset(self.args.data)["names"].values()]
            current = model.names.values() if isinstance(model.names, dict) else model.names
            if list(current) != names:
                state = (model.names, model.txt_feats, model.model[-1].nc)
                model.set_classes(names, cache_clip_model=False)
                model.names = dict(enumerate(names))
                try:
                    return super().__call__(trainer, model)
                finally:
                    model.names, model.txt_feats, model.model[-1].nc = state
        return super().__call__(trainer, model)


class WeDetectUniValidator(DetectionValidator):
    """Validator for WeDetect-Uni models with learnable prompt embeddings.

    WeDetect-Uni uses learnable prompt embeddings that correspond to each
    class in the dataset.  During validation, the model's ``embeddings``
    are used directly as text features, producing per-class scores via
    ``BNContrastiveHead``.

    This follows the original WeDetect ``SimpleYOLOWorldDetector`` pattern
    where ``num_train_classes`` equals the number of dataset categories.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize WeDetectUniValidator."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def __call__(self, trainer=None, model=None):
        """Run validation for WeDetect-Uni with per-class evaluation."""
        if trainer is not None:
            model = trainer.ema.ema
        else:
            self.device = select_device(self.args.device, verbose=False)
            if not isinstance(model, torch.nn.Module):
                from ultralytics.nn.tasks import load_checkpoint

                model = load_checkpoint(model or self.args.model, device=self.device)[0]
            model.eval().to(self.device)

        if isinstance(model, WeDetectUniModel):
            nc = model.embeddings.shape[0]
            model.model[-1].nc = nc
        return super().__call__(trainer, model)
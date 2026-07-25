# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import os

import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import WEIGHTS_DIR, checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    clip = None


class TextModel(nn.Module):
    """Abstract base class for text encoding models.

    This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement
    the tokenize and encode_text methods to provide text tokenization and encoding functionality.

    Methods:
        tokenize: Convert input texts to tokens for model processing.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    def __init__(self):
        """Initialize the TextModel base class."""
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        """Convert input texts to tokens for model processing."""
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        """Encode tokenized texts into normalized feature vectors."""
        pass


class CLIP(TextModel):
    """Implements OpenAI's CLIP (Contrastive Language-Image Pre-training) text encoder.

    This class provides a text encoder based on OpenAI's CLIP model, which can convert text into feature vectors that
    are aligned with corresponding image features in a shared embedding space.

    Attributes:
        model (clip.model.CLIP): The loaded CLIP model.
        image_preprocess (callable): Preprocessing transform for images.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to CLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> clip_model = CLIP(size="ViT-B/32", device=device)
        >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> text_features = clip_model.encode_text(tokens)
        >>> print(text_features.shape)
    """

    def __init__(self, size: str, device: torch.device) -> None:
        """Initialize the CLIP text encoder.

        This class implements the TextModel interface using OpenAI's CLIP model for text encoding. It loads a
        pre-trained CLIP model of the specified size and prepares it for text encoding tasks.

        Args:
            size (str): Model size identifier (e.g., 'ViT-B/32').
            device (torch.device): Device to load the model on.
        """
        super().__init__()
        global clip
        if clip is None:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip as _clip

            clip = _clip
        self.model, self.image_preprocess = clip.load(size, device=device, download_root=str(WEIGHTS_DIR / "clip"))
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
        """Convert input texts to CLIP tokens.

        Args:
            texts (str | list[str]): Input text or list of texts to tokenize.
            truncate (bool, optional): Whether to trim texts that exceed CLIP's context length. Defaults to True to
                avoid RuntimeError from overly long inputs while still allowing explicit opt-out.

        Returns:
            (torch.Tensor): Tokenized text tensor with shape (batch_size, context_length) ready for model processing.

        Examples:
            >>> model = CLIP("ViT-B/32", device="cpu")
            >>> tokens = model.tokenize("a photo of a cat")
            >>> print(tokens.shape)  # torch.Size([1, 77])
            >>> strict_tokens = model.tokenize("a photo of a cat", truncate=False)  # Enforce strict length checks
            >>> print(strict_tokens.shape)  # Same shape/content as tokens since prompt less than 77 tokens
        """
        return clip.tokenize(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.

        This method processes tokenized text inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            texts (torch.Tensor): Tokenized text inputs, typically created using the tokenize() method.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with unit length (L2 norm = 1).

        Examples:
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = clip_model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])
        """
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode images into normalized feature vectors.

        This method processes image inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            image (PIL.Image | torch.Tensor): Image input as a PIL Image or preprocessed tensor. If a PIL Image is
                provided, it will be converted to a tensor using the model's image preprocessing function.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized image feature vectors with unit length (L2 norm = 1).

        Examples:
            >>> from ultralytics.nn.text_model import CLIP
            >>> from PIL import Image
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> image = Image.open("path/to/image.jpg")
            >>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
            >>> features = clip_model.encode_image(image_tensor)
            >>> features.shape
            torch.Size([1, 512])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats


class MobileCLIP(TextModel):
    """Implement Apple's MobileCLIP text encoder for efficient text encoding.

    This class implements the TextModel interface using Apple's MobileCLIP model, providing efficient text encoding
    capabilities for vision-language tasks with reduced computational requirements compared to standard CLIP models.

    Attributes:
        model (mobileclip.model.MobileCLIP): The loaded MobileCLIP model.
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.
        config_size_map (dict): Mapping from size identifiers to model configuration names.

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIP(size="s0", device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        """Initialize the MobileCLIP text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model for efficient text encoding.

        Args:
            size (str): Model size identifier (e.g., 's0', 's1', 's2', 'b', 'blt').
            device (torch.device): Device to load the model on.
        """
        try:
            import mobileclip
        except ImportError:
            # Ultralytics fork preferred since Apple MobileCLIP repo has incorrect version of torchvision
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """Convert input texts to MobileCLIP tokens.

        Args:
            texts (list[str]): List of text strings to tokenize.

        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

        Examples:
            >>> model = MobileCLIP("s0", "cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        """
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.

        Examples:
            >>> model = MobileCLIP("s0", device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # Actual dimension depends on model size
        """
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    """Load a TorchScript traced version of MobileCLIP.

    This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format, providing
    efficient text encoding capabilities for vision-language tasks with optimized inference performance.

    Attributes:
        encoder (torch.jit.ScriptModule): The loaded TorchScript MobileCLIP text encoder.
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIPTS(device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    def __init__(self, device: torch.device, weight: str = "mobileclip_blt.ts"):
        """Initialize the MobileCLIP TorchScript text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for efficient
        text encoding with optimized inference performance.

        Args:
            device (torch.device): Device to load the model on.
            weight (str): Path to the TorchScript model weights.
        """
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset(weight), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        """Convert input texts to MobileCLIP tokens.

        Args:
            texts (list[str]): List of text strings to tokenize.
            truncate (bool, optional): Whether to trim texts that exceed the tokenizer context length. Defaults to True,
                matching CLIP's behavior to prevent runtime failures on long captions.

        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

        Examples:
            >>> model = MobileCLIPTS(device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> strict_tokens = model.tokenize(
            ...     ["a very long caption"], truncate=False
            ... )  # RuntimeError if exceeds 77-token
        """
        return self.tokenizer(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.

        Examples:
            >>> model = MobileCLIPTS(device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # Actual dimension depends on model size
        """
        # NOTE: no need to do normalization here as it's embedded in the torchscript model
        return self.encoder(texts).to(dtype)


class XLMRoberta(TextModel):
    """XLM-RoBERTa text encoder for multilingual open-vocabulary detection.

    Encodes text descriptions into L2-normalized feature vectors using
    Facebook's XLM-RoBERTa model, followed by a linear projection head.
    Supports base (768→768) and large (1024→768) model sizes.

    Attributes:
        model (XLMRobertaModel): The XLM-RoBERTa encoder.
        tokenizer (AutoTokenizer): The associated tokenizer.
        head (nn.Linear): Linear projection from hidden dim to embed dim.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to token tensors.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> model = XLMRoberta(size="base", device=torch.device("cpu"))
        >>> tokens = model.tokenize([["person", "car", "dog"]])
        >>> features = model.encode_text(tokens)
    """

    _size_cfg = {
        "base": {"model_name": "xlm-roberta-base", "hidden": 768, "embed": 768},
        "large": {"model_name": "xlm-roberta-large", "hidden": 1024, "embed": 768},
    }

    def __init__(self, size: str, device: torch.device, state_dict: dict | None = None) -> None:
        """Initialize the XLM-RoBERTa text encoder.

        Args:
            size (str): Model size identifier, one of 'base', 'large'.
            device (torch.device): Device to load the model on.
            state_dict (dict | None): Optional state dict to load into the model (including
                XLM-RoBERTa backbone and head projection weights).
        """
        try:
            from transformers import AutoTokenizer, XLMRobertaModel
        except ImportError:
            checks.check_requirements("transformers")
            from transformers import AutoTokenizer, XLMRobertaModel

        super().__init__()
        cfg = self._size_cfg.get(size)
        if cfg is None:
            raise ValueError(f"Unknown XLM-RoBERTa size '{size}'. Choose from {list(self._size_cfg)}")
        model_name = cfg["model_name"]
        _root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        local_paths = [
            os.path.join(_root, model_name),
            os.path.join(_root, "checkpoints", model_name),
        ]
        local_path = None
        for p in local_paths:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
                local_path = p
                break
        has_local_weights = (
            local_path
            and (os.path.exists(os.path.join(local_path, "pytorch_model.bin"))
                 or os.path.exists(os.path.join(local_path, "model.safetensors")))
        )
        try:
            if local_path:
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                if has_local_weights:
                    self.model = XLMRobertaModel.from_pretrained(local_path, local_files_only=True)
                else:
                    from transformers import XLMRobertaConfig
                    xcfg = XLMRobertaConfig.from_pretrained(local_path)
                    self.model = XLMRobertaModel(xcfg)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = XLMRobertaModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            # Download first, then save — creating save_dir early makes HF treat the empty
            # folder as a local model path and breaks tokenizer load on retry.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = XLMRobertaModel.from_pretrained(model_name)
            save_dir = os.path.join(_root, model_name)
            self.tokenizer.save_pretrained(save_dir)
            self.model.save_pretrained(save_dir)
        self.head = nn.Linear(cfg["hidden"], cfg["embed"], bias=True)
        if state_dict is not None:
            self.model.load_state_dict(
                {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")},
                strict=False,
            )
            head_sd = {k[len("head."):]: v for k, v in state_dict.items() if k.startswith("head.")}
            if head_sd:
                self.head.load_state_dict(head_sd)
        self.to(device)
        self.device = torch.device(device) if device is not None else next(self.parameters()).device
        self.eval()  # default inference; call .train() for open-vocabulary fine-tuning

    def _input_device(self) -> torch.device:
        """Return the live parameter device (keeps tokens/buffers aligned after .to()/EMA)."""
        device = next(self.model.parameters()).device
        if self.device != device:
            self.device = device
        return device

    def tokenize(self, texts, truncate: bool = True) -> dict:
        """Convert input texts to XLM-RoBERTa token dicts.

        Args:
            texts (list[list[str]] | list[str]): Nested list of class texts
                (e.g. ``[["person", "car"]]``) or flat list of strings.
            truncate (bool): Whether to truncate long sequences.

        Returns:
            (dict): Tokenizer output with ``input_ids`` and ``attention_mask`` tensors.
        """
        import itertools

        flat = list(itertools.chain(*texts)) if texts and isinstance(texts[0], (list, tuple)) else texts
        encoded = self.tokenizer(text=flat, return_tensors="pt", padding=True, truncation=truncate, max_length=77)
        device = self._input_device()
        return {k: v.to(device) for k, v in encoded.items()}

    def encode_text(self, texts, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.

        Accepts either a tokenizer output dict (from :meth:`tokenize`) or a
        nested list of strings that will be tokenized on the fly.

        When the module is in ``train()`` mode, gradients flow through the encoder
        (required for open-vocabulary fine-tuning). In ``eval()`` mode, encoding
        runs under inference mode for speed.

        Args:
            texts (dict | list): Tokenizer output dict or nested text list.
            dtype (torch.dtype): Output data type.

        Returns:
            (torch.Tensor): Normalized text features, shape ``(1, num_classes, embed_dim)``.
        """
        if self.training:
            return self._encode_text(texts, dtype)
        with torch.inference_mode():
            return self._encode_text(texts, dtype)

    def _encode_text(self, texts, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Internal text encoding without inference-mode wrapping."""
        import itertools

        device = self._input_device()
        # HF RoBERTa keeps int buffers (e.g. token_type_ids); ensure they track params after EMA/.to()
        for buf in self.model.buffers():
            if buf.device != device:
                self.model.to(device)
                self.head.to(device)
                break

        if isinstance(texts, dict):
            encoded = {k: v.to(device) for k, v in texts.items()}
            num_classes = encoded["input_ids"].shape[0]
        else:
            is_nested = texts and isinstance(texts[0], (list, tuple))
            flat = list(itertools.chain(*texts)) if is_nested else list(texts)
            num_classes = len(flat)
            encoded = self.tokenizer(text=flat, return_tensors="pt", padding=True, truncation=True, max_length=77)
            encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = self.model(**encoded)
        txt_feats = outputs.last_hidden_state[:, 0]
        txt_feats = self.head(txt_feats)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats.reshape(1, num_classes, -1).to(dtype)


def build_text_model(variant: str, device: torch.device = None, state_dict: dict | None = None) -> TextModel:
    """Build a text encoding model based on the specified variant.

    Args:
        variant (str): Model variant in format "base:size" (e.g., "clip:ViT-B/32" or "mobileclip:s0").
        device (torch.device, optional): Device to load the model on.
        state_dict (dict | None): Optional state dict for XLM-RoBERTa models.

    Returns:
        (TextModel): Instantiated text encoding model.

    Examples:
        >>> model = build_text_model("clip:ViT-B/32", device=torch.device("cuda"))
        >>> model = build_text_model("mobileclip:s0", device=torch.device("cpu"))
    """
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    elif base == "mobileclip2":
        return MobileCLIPTS(device, weight="mobileclip2_b.ts")
    elif base == "xlm-roberta":
        return XLMRoberta(size, device, state_dict=state_dict)
    else:
        raise ValueError(
            f"Unrecognized base model '{base}'. Supported models are 'clip', 'mobileclip', 'mobileclip2', 'xlm-roberta'."
        )

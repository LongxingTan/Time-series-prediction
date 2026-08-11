"""Model Registry — centralized catalog of all available models.

Provides metadata and discovery for every model in TFTS, similar to how
transformers maintains its model hub.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model metadata registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = OrderedDict(
    [
        (
            "seq2seq",
            {
                "class_name": "Seq2seq",
                "config_class": "Seq2seqConfig",
                "description": "Basic encoder-decoder sequence-to-sequence model with attention.",
                "paper": "",
                "tags": ["baseline", "encoder-decoder"],
            },
        ),
        (
            "rnn",
            {
                "class_name": "RNN",
                "config_class": "RNNConfig",
                "description": "Stacked LSTM/GRU with optional attention for time series.",
                "paper": "",
                "tags": ["baseline", "recurrent"],
            },
        ),
        (
            "wavenet",
            {
                "class_name": "WaveNet",
                "config_class": "WaveNetConfig",
                "description": "Dilated causal convolutions for long-range temporal dependencies.",
                "paper": "https://arxiv.org/abs/1609.03499",
                "tags": ["convolutional", "long-range"],
            },
        ),
        (
            "tcn",
            {
                "class_name": "TCN",
                "config_class": "TCNConfig",
                "description": "Temporal Convolutional Network with dilated causal convolutions.",
                "paper": "https://arxiv.org/abs/1803.01271",
                "tags": ["convolutional", "efficient"],
            },
        ),
        (
            "transformer",
            {
                "class_name": "Transformer",
                "config_class": "TransformerConfig",
                "description": "Classic encoder-decoder Transformer for time series forecasting.",
                "paper": "https://arxiv.org/abs/1706.03762",
                "tags": ["attention", "encoder-decoder"],
            },
        ),
        (
            "bert",
            {
                "class_name": "Bert",
                "config_class": "BertConfig",
                "description": "BERT-style masked pre-training adapted for time series.",
                "paper": "https://arxiv.org/abs/1810.04805",
                "tags": ["pretraining", "attention", "encoder-only"],
            },
        ),
        (
            "informer",
            {
                "class_name": "Informer",
                "config_class": "InformerConfig",
                "description": "Efficient Transformer with ProbSparse self-attention for long sequences.",
                "paper": "https://arxiv.org/abs/2012.07436",
                "tags": ["attention", "long-sequence", "efficient", "SOTA"],
            },
        ),
        (
            "autoformer",
            {
                "class_name": "AutoFormer",
                "config_class": "AutoFormerConfig",
                "description": "Auto-correlation mechanism with series decomposition for seasonal-trend modeling.",
                "paper": "https://arxiv.org/abs/2106.13008",
                "tags": ["decomposition", "seasonal", "SOTA"],
            },
        ),
        (
            "tft",
            {
                "class_name": "TFTransformer",
                "config_class": "TFTransformerConfig",
                "description": "Temporal Fusion Transformer — interpretable multi-horizon forecasting.",
                "paper": "https://arxiv.org/abs/1912.09363",
                "tags": ["interpretable", "multi-horizon", "attention", "SOTA"],
            },
        ),
        (
            "unet",
            {
                "class_name": "Unet",
                "config_class": "UnetConfig",
                "description": "U-Net style architecture with skip connections for time series.",
                "paper": "https://arxiv.org/abs/1505.04597",
                "tags": ["convolutional", "skip-connection"],
            },
        ),
        (
            "nbeats",
            {
                "class_name": "NBeats",
                "config_class": "NBeatsConfig",
                "description": "Neural basis expansion — pure MLP stack with interpretable basis functions.",
                "paper": "https://arxiv.org/abs/1905.10437",
                "tags": ["mlp", "interpretable", "SOTA", "basis-expansion"],
            },
        ),
        (
            "dlinear",
            {
                "class_name": "DLinear",
                "config_class": "DLinearConfig",
                "description": "Surprisingly strong linear baseline — questions Transformer necessity.",
                "paper": "https://arxiv.org/abs/2205.13504",
                "tags": ["linear", "simple", "baseline", "SOTA"],
            },
        ),
        (
            "rwkv",
            {
                "class_name": "RWKV",
                "config_class": "RWKVConfig",
                "description": "RNN-style efficient attention — linear complexity with Transformer quality.",
                "paper": "https://arxiv.org/abs/2305.13048",
                "tags": ["efficient", "attention", "recurrent"],
            },
        ),
        (
            "patch_tst",
            {
                "class_name": "PatchTST",
                "config_class": "PatchTSTConfig",
                "description": "Patch-based time series Transformer — segments time series into patches.",
                "paper": "https://arxiv.org/abs/2211.14730",
                "tags": ["patching", "attention", "SOTA"],
            },
        ),
        (
            "deep_ar",
            {
                "class_name": "DeepAR",
                "config_class": "DeepARConfig",
                "description": "Probabilistic autoregressive RNN for uncertainty-aware forecasting.",
                "paper": "https://arxiv.org/abs/1704.04110",
                "tags": ["probabilistic", "recurrent", "uncertainty"],
            },
        ),
        (
            "itransformer",
            {
                "class_name": "iTransformer",
                "config_class": "iTransformerConfig",
                "description": "Inverted Transformer — applies attention across variates instead of time.",
                "paper": "https://arxiv.org/abs/2310.06625",
                "tags": ["attention", "multivariate", "SOTA"],
            },
        ),
        (
            "timesfm",
            {
                "class_name": "TimesFM",
                "config_class": "TimesFMConfig",
                "description": "Google's foundation model for time series — decoder-only with patching.",
                "paper": "https://arxiv.org/abs/2310.10688",
                "tags": ["foundation-model", "patching", "decoder-only", "SOTA"],
            },
        ),
        (
            "gpt",
            {
                "class_name": "Gpt",
                "config_class": "GptConfig",
                "description": "GPT-style decoder-only Transformer adapted for time series.",
                "paper": "",
                "tags": ["decoder-only", "attention", "generative"],
            },
        ),
        (
            "diffusion",
            {
                "class_name": "Diffusion",
                "config_class": "DiffusionConfig",
                "description": "Denoising diffusion probabilistic model for time series generation.",
                "paper": "https://arxiv.org/abs/2006.11239",
                "tags": ["generative", "diffusion", "probabilistic"],
            },
        ),
        (
            "tide",
            {
                "class_name": "TiDE",
                "config_class": "TiDEConfig",
                "description": "Time-series Dense Encoder — simple MLP with covariate projection.",
                "paper": "https://arxiv.org/abs/2304.08424",
                "tags": ["mlp", "efficient", "covariates"],
            },
        ),
    ]
)


def list_models(tag: Optional[str] = None) -> List[str]:
    """List all available model names, optionally filtered by tag.

    Args:
        tag: If provided, only return models matching this tag
             (e.g. ``'SOTA'``, ``'attention'``, ``'convolutional'``).

    Returns:
        Sorted list of model names.

    Examples:
        >>> tfts.list_models()
        ['autoformer', 'bert', 'deep_ar', 'diffusion', 'dlinear', ...]

        >>> tfts.list_models(tag='SOTA')
        ['autoformer', 'dlinear', 'informer', 'itransformer', 'nbeats', 'patch_tst', 'tft', 'timemixer', 'timesfm']
    """
    if tag is None:
        return sorted(MODEL_REGISTRY.keys())
    return sorted(k for k, v in MODEL_REGISTRY.items() if tag in v.get("tags", []))


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get metadata for a specific model.

    Args:
        model_name: Name of the model as used in the registry.

    Returns:
        Dictionary with keys: class_name, config_class, description, paper, tags.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list_models()}")
    return dict(MODEL_REGISTRY[model_name])


def get_model_class_name(model_name: str) -> str:
    """Resolve a model name to its Python class name."""
    return MODEL_REGISTRY[model_name]["class_name"]


def get_config_class_name(model_name: str) -> str:
    """Resolve a model name to its Config class name."""
    return MODEL_REGISTRY[model_name]["config_class"]

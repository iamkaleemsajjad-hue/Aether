"""Complete synthetic HF Mamba-2 weights and an independent SSD matrix oracle.

The oracle uses Dao & Gu (2024), equations (2)-(3), rather than the runtime's
token scan. Section 7.1 places normalization after the multiplicative gate:
https://arxiv.org/html/2405.21060v1
Reference block: state-spaces/mamba v2.2.4, modules/mamba2_simple.py.
This small untrained fixture measures numerical fidelity, not corpus quality.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def checkpoint_data(
    *,
    expansion: int = 2,
    with_bias: bool = False,
    rms_norm: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return distinct learned norm scales and all real source parameter arrays."""
    hidden, heads, state, groups, width, vocab = 16, 4, 4, 2, 3, 17
    inner = expansion * hidden
    channels = inner + 2 * groups * state
    config = {
        "architectures": ["Mamba2ForCausalLM"],
        "model_type": "mamba2",
        "hidden_size": hidden,
        "expand": expansion,
        "num_heads": heads,
        "head_dim": inner // heads,
        "state_size": state,
        "n_groups": groups,
        "conv_kernel": width,
        "num_hidden_layers": 1,
        "vocab_size": vocab,
        "chunk_size": 4,
        "layer_norm_epsilon": 1e-5,
        "rms_norm": rms_norm,
        "hidden_act": "silu",
        "use_bias": with_bias,
        "use_conv_bias": True,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 16,
        "pad_token_id": 0,
        "torch_dtype": "float32",
    }
    rng = np.random.default_rng(20261020)

    def matrix(*shape: int) -> np.ndarray:
        return rng.normal(0, 0.1, shape).astype(np.float32)

    tensors = {
        "backbone.embeddings.weight": matrix(vocab, hidden),
        "backbone.norm_f.weight": np.linspace(0.8, 1.2, hidden, dtype=np.float32),
        "lm_head.weight": matrix(vocab, hidden),
        "backbone.layers.0.norm.weight": np.linspace(0.5, 1.5, hidden, dtype=np.float32),
        "backbone.layers.0.mixer.norm.weight": np.linspace(1.1, 2.1, inner, dtype=np.float32),
        "backbone.layers.0.mixer.in_proj.weight": matrix(
            2 * inner + 2 * groups * state + heads, hidden
        ),
        "backbone.layers.0.mixer.conv1d.weight": matrix(channels, 1, width),
        "backbone.layers.0.mixer.conv1d.bias": matrix(channels),
        "backbone.layers.0.mixer.A_log": np.log(np.arange(1, heads + 1, dtype=np.float32)),
        "backbone.layers.0.mixer.D": np.linspace(0.7, 1.3, heads, dtype=np.float32),
        "backbone.layers.0.mixer.dt_bias": np.linspace(-4, -2, heads, dtype=np.float32),
        "backbone.layers.0.mixer.out_proj.weight": matrix(hidden, inner),
    }
    if with_bias:
        tensors["backbone.layers.0.mixer.in_proj.bias"] = matrix(
            2 * inner + 2 * groups * state + heads
        )
        tensors["backbone.layers.0.mixer.out_proj.bias"] = matrix(hidden)
    return config, tensors


def write_checkpoint(directory: Path, **options: Any) -> dict[str, Any]:
    """Write a complete SafeTensors checkpoint and deterministic word tokenizer."""
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    config, tensors = checkpoint_data(**options)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    vocabulary = {"<unk>": 0, "hello": 1, **{f"tok{i}": i for i in range(2, config["vocab_size"])}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "model_max_length": 4096,
            }
        ),
        encoding="utf-8",
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "<unk>"}), encoding="utf-8"
    )
    return config


def matrix_logits(
    config: dict[str, Any],
    tensors: dict[str, np.ndarray],
    ids: np.ndarray,
) -> np.ndarray:
    """Evaluate the bounded full semiseparable matrix (paper equation 3), FP64."""
    eps = config["layer_norm_epsilon"]

    def rms(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return value / np.sqrt((value**2).mean(axis=-1, keepdims=True) + eps) * weight

    def silu(value: np.ndarray) -> np.ndarray:
        return value / (1 + np.exp(-value))

    def weight(name: str) -> np.ndarray:
        return tensors[f"backbone.layers.0.mixer.{name}"].astype(np.float64)

    hidden = tensors["backbone.embeddings.weight"][ids].astype(np.float64)
    normalized = rms(hidden, tensors["backbone.layers.0.norm.weight"])
    projected = normalized @ weight("in_proj.weight").T
    if config["use_bias"]:
        projected += weight("in_proj.bias")
    inner = config["expand"] * config["hidden_size"]
    heads, groups, state = config["num_heads"], config["n_groups"], config["state_size"]
    head_dim, length = config["head_dim"], len(ids)
    gate, xbc, dt = np.split(projected, [inner, 2 * inner + 2 * groups * state], axis=-1)
    filters = weight("conv1d.weight")[:, 0, :]
    convolved = np.column_stack(
        [
            np.convolve(xbc[:, channel], filters[channel, ::-1], mode="full")[:length]
            for channel in range(xbc.shape[-1])
        ]
    )
    convolved = silu(convolved + weight("conv1d.bias"))
    x, b, c = np.split(convolved, [inner, inner + groups * state], axis=-1)
    x = x.reshape(length, heads, head_dim)
    b, c = (
        np.repeat(value.reshape(length, groups, state), heads // groups, axis=1) for value in (b, c)
    )
    delta = np.logaddexp(0, dt + weight("dt_bias"))
    rates = delta * (-np.exp(weight("A_log")))
    cumulative = np.cumsum(rates, axis=0)
    log_decay = cumulative[:, None, :] - cumulative[None, :, :]
    log_decay[np.triu_indices(length, 1)] = -np.inf
    matrix = np.einsum("thn,shn->tsh", c, b) * np.exp(log_decay)
    y = np.einsum("tsh,shp->thp", matrix, x * delta[:, :, None])
    y += x * weight("D")[None, :, None]
    mixed = rms(y.reshape(length, inner) * silu(gate), weight("norm.weight"))
    hidden += mixed @ weight("out_proj.weight").T
    if config["use_bias"]:
        hidden += weight("out_proj.bias")
    return (
        rms(hidden, tensors["backbone.norm_f.weight"])
        @ tensors["lm_head.weight"].astype(np.float64).T
    )

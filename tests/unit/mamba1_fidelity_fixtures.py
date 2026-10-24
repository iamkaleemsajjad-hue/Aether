"""Synthetic HF Mamba-1 checkpoint and independent full-sequence FP64 oracle.

Gu & Dao (2023), section 2 equation (2), gives the recurrent linear system:
https://arxiv.org/html/2312.00752v2
The checkpoint contract here follows Mamba.step and selective_scan_ref in
state-spaces/mamba v2.2.4: Abar=exp(delta*A), Bbar=delta*B. That literal input
injection differs from equation (4)'s exact ZOH Bbar; replacing it changes the
trained checkpoint's function. References:
https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py
https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/selective_scan_interface.py
These untrained weights test fidelity, not trained-model corpus quality.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def checkpoint_data(
    *,
    expansion: int = 2,
    with_bias: bool = False,
    width: int = 3,
    rms_norm: bool = True,
    large_delta: bool = False,
    epsilon: float = 1e-5,
    activation: str = "silu",
    large_silu: str | None = None,
    norm_width: int | None = None,
    layers: int = 1,
    seed: int = 20261022,
    omit_activation: bool = False,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float32]]]:
    """Include nonzero B/C, distinct learned norm scales and optional real biases."""
    hidden, state, rank, vocab = 16, 4, 2, 17
    inner = expansion * hidden
    config = {
        "architectures": ["MambaForCausalLM"],
        "model_type": "mamba",
        "hidden_size": hidden,
        "expand": expansion,
        "state_size": state,
        "time_step_rank": rank,
        "conv_kernel": width,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "layer_norm_epsilon": epsilon,
        "rms_norm": rms_norm,
        "hidden_act": activation,
        "use_bias": with_bias,
        "use_conv_bias": True,
        "residual_in_fp32": True,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 16,
        "pad_token_id": 0,
        "torch_dtype": "float32",
    }
    if omit_activation:
        del config["hidden_act"]
    rng = np.random.default_rng(seed)

    def matrix(*shape: int) -> NDArray[np.float32]:
        return rng.normal(0, 0.1, shape).astype(np.float32)

    prefix = "backbone.layers.0."
    tensors = {
        "backbone.embeddings.weight": matrix(vocab, hidden),
        "backbone.norm_f.weight": np.linspace(0.8, 1.2, hidden, dtype=np.float32),
        "lm_head.weight": matrix(vocab, hidden),
        prefix + "norm.weight": np.linspace(
            0.5, 1.5, hidden if norm_width is None else norm_width, dtype=np.float32
        ),
        prefix + "mixer.in_proj.weight": matrix(2 * inner, hidden),
        prefix + "mixer.conv1d.weight": matrix(inner, 1, width),
        prefix + "mixer.conv1d.bias": matrix(inner),
        prefix + "mixer.x_proj.weight": matrix(rank + 2 * state, inner),
        prefix + "mixer.dt_proj.weight": matrix(inner, rank),
        prefix + "mixer.dt_proj.bias": (
            np.full(inner, 1000, np.float32)
            if large_delta
            else np.linspace(-4, -2, inner, dtype=np.float32)
        ),
        prefix + "mixer.A_log": np.log(
            np.linspace(0.7, 1.3, inner, dtype=np.float32)[:, None]
            * np.arange(1, state + 1, dtype=np.float32)[None, :]
        ),
        prefix + "mixer.D": np.linspace(0.7, 1.3, inner, dtype=np.float32),
        prefix + "mixer.out_proj.weight": matrix(hidden, inner),
    }
    if with_bias:
        tensors[prefix + "mixer.in_proj.bias"] = matrix(2 * inner)
        tensors[prefix + "mixer.out_proj.bias"] = matrix(hidden)
    if large_silu is not None:
        if large_silu == "gate" and with_bias:
            tensors[prefix + "mixer.in_proj.bias"][inner:] = -1e20
        elif large_silu == "conv":
            tensors[prefix + "mixer.conv1d.bias"][:] = -1e20
        else:
            raise ValueError("large_silu must be conv or a biased gate")
        tensors[prefix + "mixer.out_proj.weight"] *= 1e8
    for index in range(1, layers):
        _, additional = checkpoint_data(
            expansion=expansion,
            with_bias=with_bias,
            width=width,
            rms_norm=rms_norm,
            large_delta=large_delta,
            epsilon=epsilon,
            activation=activation,
            large_silu=large_silu,
            norm_width=norm_width,
            seed=seed + index,
        )
        for name, value in additional.items():
            if name.startswith(prefix):
                tensors[name.replace(prefix, f"backbone.layers.{index}.", 1)] = value
        tensors[f"backbone.layers.{index}.norm.weight"] *= 1.0 + 0.1 * index
    return config, tensors


def write_checkpoint(directory: Path, **options: Any) -> dict[str, Any]:
    """Write complete SafeTensors weights and the deterministic public tokenizer."""
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
        json.dumps({"unk_token": "<unk>"}),
        encoding="utf-8",
    )
    return config


def matrix_logits(
    config: dict[str, Any],
    tensors: dict[str, NDArray[np.float32]],
    ids: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Sum every causal source contribution; do not call a runtime token scan.

    For s<=t, h[t,i,n] includes exp(sum_{u=s+1}^t delta[u,i]*A[i,n])
    *delta[s,i]*B[s,n]*x[s,i]. C[t,n] reads this state before the D skip,
    SiLU gate, output projection and residual. The full matrix is deliberately
    bounded to small test prompts; it is not an inference implementation.
    """
    eps = config["layer_norm_epsilon"]

    def rms(
        value: NDArray[np.float64],
        weight: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        return np.asarray(
            value / np.sqrt((value**2).mean(axis=-1, keepdims=True) + eps) * weight,
            dtype=np.float64,
        )

    def silu(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(value * np.exp(-np.logaddexp(0, -value)), dtype=np.float64)

    hidden = tensors["backbone.embeddings.weight"][ids].astype(np.float64)
    length = len(ids)
    for index in range(config["num_hidden_layers"]):

        def weight(name: str, layer_index: int = index) -> NDArray[np.float64]:
            return tensors[f"backbone.layers.{layer_index}.mixer.{name}"].astype(np.float64)

        normalized = rms(hidden, tensors[f"backbone.layers.{index}.norm.weight"])
        projected = normalized @ weight("in_proj.weight").T
        if config["use_bias"]:
            projected += weight("in_proj.bias")
        x, gate = np.split(projected, 2, axis=-1)
        filters = weight("conv1d.weight")[:, 0, :]
        convolved = np.column_stack(
            [
                np.convolve(x[:, channel], filters[channel, ::-1], mode="full")[:length]
                for channel in range(x.shape[-1])
            ]
        )
        convolved = silu(convolved + weight("conv1d.bias"))
        rank, state = config["time_step_rank"], config["state_size"]
        delta_low, b, c = np.split(
            convolved @ weight("x_proj.weight").T, [rank, rank + state], axis=-1
        )
        delta = np.logaddexp(0, delta_low @ weight("dt_proj.weight").T + weight("dt_proj.bias"))
        cumulative = np.cumsum(delta[:, :, None] * -np.exp(weight("A_log"))[None, :, :], axis=0)
        decay = cumulative[:, None, :, :] - cumulative[None, :, :, :]
        decay[np.triu_indices(length, 1)] = -np.inf
        injection = delta[:, :, None] * convolved[:, :, None] * b[:, None, :]
        y = np.einsum("tsin,sin,tn->ti", np.exp(decay), injection, c)
        y += convolved * weight("D")[None, :]
        output = (y * silu(gate)) @ weight("out_proj.weight").T
        if config["use_bias"]:
            output += weight("out_proj.bias")
        hidden += output
    return (
        rms(hidden, tensors["backbone.norm_f.weight"])
        @ tensors["lm_head.weight"].astype(np.float64).T
    )

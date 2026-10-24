"""Offline FP32 checkpoints for pass_audit.py; no compiler mocks or downloads."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray


class Fixture(TypedDict):
    """Describe generated source tensors and the inputs required by opt-in passes."""

    name: str
    seed: int
    parameters: int
    source_dtype: str
    model: str
    merge_source: str
    adapters: list[str]
    synthetic_untrained: bool
    config: dict[str, Any]


FIXTURES = ("tiny", "mid", "moe", "mtp", "adapter")
PROMPTS = {"A": [1, 2, 3, 4], "B": [3, 4, 5, 6] * 16}
CONTINUATION = [7, 8, 9, 10]


def write_json(path: Path, value: object) -> None:
    """Persist readable JSON and reject nonfinite pseudo-measurements."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def geometry(name: str) -> dict[str, Any]:
    """Return the actual checkpoint dimensions, including the 27M mid model."""
    if name not in FIXTURES:
        raise ValueError(f"Unknown fixture: {name}")
    hidden, intermediate, layers, vocab, heads, kv = (
        (512, 1536, 8, 2048, 8, 4) if name == "mid" else (32, 64, 2, 128, 4, 2)
    )
    config: dict[str, Any] = {
        "architectures": ["MixtralForCausalLM" if name == "moe" else "LlamaForCausalLM"],
        "model_type": "mixtral" if name == "moe" else "llama",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": hidden // heads,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
    }
    if name == "moe":
        config.update(num_local_experts=4, num_experts_per_tok=2, router_aux_loss_coef=0.0)
    if name == "mtp":
        config.update(mtp_heads=2, num_nextn_predict_layers=2)
    return config


def parameter_count(name: str) -> int:
    """Count all real checkpoint coordinates before any expensive allocation."""
    c = geometry(name)
    h, i, v, n = (
        int(c[k]) for k in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers")
    )
    kv = int(c["num_key_value_heads"]) * int(c["head_dim"])
    ffn = 4 * 3 * h * i + 4 * h if name == "moe" else 3 * h * i
    return (
        2 * v * h
        + h
        + n * (2 * h * h + 2 * h * kv + 2 * h + ffn)
        + (2 * v * h if name == "mtp" else 0)
    )


def prompt_text(ids: list[int]) -> str:
    """Render the exact fixture vocabulary corresponding to fixed prompt IDs."""
    words = {0: "<unk>", 1: "hello", 2: "world", 3: "a", 4: "b", 5: "c"}
    return " ".join(words.get(token, f"t{token}") for token in ids)


def create_fixture(root: Path, name: str, seed: int = 1729) -> Fixture:
    """Write real SafeTensors, tokenizer, merge source, and optional adapter.

    Random data is synthetic and untrained. It establishes execution/numerical
    behavior only, never language quality or draft-head acceptance quality.
    """
    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    config = geometry(name)
    h, i, v, n = (
        int(config[k])
        for k in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers")
    )
    kv = int(config["num_key_value_heads"]) * int(config["head_dim"])
    rng = np.random.Generator(np.random.PCG64(seed))

    def matrix(*shape: int) -> NDArray[np.float32]:
        return rng.normal(0, 0.02, size=shape).astype("<f4")

    tensors = {
        "model.embed_tokens.weight": matrix(v, h),
        "model.norm.weight": np.ones(h, dtype="<f4"),
        "lm_head.weight": matrix(v, h),
    }
    for layer in range(n):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(h, dtype="<f4")
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(h, dtype="<f4")
        for projection, rows in (("q", h), ("k", kv), ("v", kv), ("o", h)):
            tensors[f"{prefix}.self_attn.{projection}_proj.weight"] = matrix(rows, h)
        if name == "moe":
            tensors[f"{prefix}.block_sparse_moe.gate.weight"] = matrix(4, h)
            for expert in range(4):
                ep = f"{prefix}.block_sparse_moe.experts.{expert}"
                tensors[f"{ep}.w1.weight"] = matrix(i, h)
                tensors[f"{ep}.w2.weight"] = matrix(h, i)
                tensors[f"{ep}.w3.weight"] = matrix(i, h)
        else:
            tensors[f"{prefix}.mlp.gate_proj.weight"] = matrix(i, h)
            tensors[f"{prefix}.mlp.up_proj.weight"] = matrix(i, h)
            tensors[f"{prefix}.mlp.down_proj.weight"] = matrix(h, i)
    if name == "mtp":
        for head in range(2):
            tensors[f"mtp_head_{head}.weight"] = matrix(v, h)

    model = root / "model"
    model.mkdir(parents=True, exist_ok=False)
    write_json(model / "config.json", config)
    # No metadata map with nondeterministic key ordering is passed to Rust.
    save_file(dict(sorted(tensors.items())), str(model / "model.safetensors"))
    vocab = {prompt_text([token]): token for token in range(v)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(model / "tokenizer.json"))
    write_json(
        model / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "model_max_length": 512,
        },
    )
    write_json(model / "special_tokens_map.json", {"unk_token": "<unk>"})

    # A real complete source with one changed tensor, so Pass 12 sees a task
    # vector in the same checkpoint coordinates as the base.
    merge = root / "merge_source"
    merge.mkdir()
    write_json(merge / "config.json", config)
    changed = "model.layers.0.self_attn.q_proj.weight"
    tensors[changed] = tensors[changed] + np.float32(0.005)
    save_file(dict(sorted(tensors.items())), str(merge / "model.safetensors"))
    adapters = []
    if name == "adapter":
        adapter = root / "adapter_demo"
        adapter.mkdir()
        write_json(
            adapter / "adapter_config.json", {"r": 2, "lora_alpha": 2, "target_modules": ["q_proj"]}
        )
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        save_file(
            {
                f"{prefix}.lora_A.weight": matrix(2, h),
                f"{prefix}.lora_B.weight": matrix(h, 2),
            },
            str(adapter / "adapter_model.safetensors"),
        )
        adapters.append(str(adapter))
    count = sum(int(tensor.size) for tensor in tensors.values())
    if count != parameter_count(name):
        raise ValueError(f"Fixture parameter accounting mismatch: {count}")
    return {
        "name": name,
        "seed": seed,
        "parameters": count,
        "source_dtype": "float32",
        "model": str(model),
        "merge_source": str(merge),
        "adapters": adapters,
        "synthetic_untrained": True,
        "config": config,
    }

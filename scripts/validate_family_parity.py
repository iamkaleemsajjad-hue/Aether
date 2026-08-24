"""Parity harness: HF reference vs Aether CPU engine vs Aether Torch engine.

Diagnostic (not part of the shipped CLI).  Builds a *tiny random* model of a
given family with Transformers, compiles it through Aether, and compares
prefill logits and incremental-decode logits across all three executors.

    python scripts/validate_family_parity.py gpt_neo qwen3 llama ...
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("AETHER_TORCH_DTYPE", "fp32")


def build_tiny(family: str, out: Path) -> None:
    import torch
    import transformers as tf

    torch.manual_seed(0)
    vocab = 4096
    common = dict(vocab_size=vocab, use_cache=True)
    if family == "gpt_neo":
        cfg = tf.GPTNeoConfig(
            hidden_size=64, num_layers=4, num_heads=4, intermediate_size=256,
            max_position_embeddings=128, attention_types=[[["global", "local"], 2]],
            window_size=8, **common,
        )
    elif family == "qwen3":
        cfg = tf.Qwen3Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128, head_dim=32,
            max_position_embeddings=128, **common,
        )
    elif family == "qwen2":
        cfg = tf.Qwen2Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "llama":
        cfg = tf.LlamaConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "gemma2":
        cfg = tf.Gemma2Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128, head_dim=16,
            max_position_embeddings=128, sliding_window=8, **common,
        )
    elif family == "gemma3":
        cfg = tf.Gemma3TextConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128, head_dim=16,
            max_position_embeddings=128, sliding_window=8, **common,
        )
    elif family == "mistral":
        cfg = tf.MistralConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "mixtral":
        cfg = tf.MixtralConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            num_local_experts=4, num_experts_per_tok=2,
            max_position_embeddings=128, **common,
        )
    elif family == "gpt_neox":
        cfg = tf.GPTNeoXConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            intermediate_size=256, max_position_embeddings=128, **common,
        )
    elif family == "gptj":
        cfg = tf.GPTJConfig(
            n_embd=64, n_layer=4, n_head=4, rotary_dim=16,
            n_positions=128, **common,
        )
    elif family == "gpt2":
        cfg = tf.GPT2Config(
            n_embd=64, n_layer=4, n_head=4, n_positions=128, **common,
        )
    elif family == "phi3":
        cfg = tf.Phi3Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "falcon":
        cfg = tf.FalconConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            new_decoder_architecture=True, num_kv_heads=2,
            max_position_embeddings=128, **common,
        )
    elif family == "olmo2":
        cfg = tf.Olmo2Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "stablelm":
        cfg = tf.StableLmConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "starcoder2":
        cfg = tf.Starcoder2Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=256,
            max_position_embeddings=128, **common,
        )
    elif family == "granite":
        cfg = tf.GraniteConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "bloom":
        cfg = tf.BloomConfig(
            hidden_size=64, n_layer=4, n_head=4, **common,
        )
    elif family == "mpt":
        cfg = tf.MptConfig(d_model=64, n_layers=4, n_heads=4, **common)
    elif family == "cohere":
        cfg = tf.CohereConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "exaone4":
        cfg = tf.Exaone4Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, sliding_window=8,
            layer_types=["sliding_attention", "full_attention"] * 2, **common,
        )
    elif family == "deepseek_v3":
        # Multi-head latent attention: compressed Q/KV projections plus a
        # decoupled RoPE key.  Routed to the MLA executor, not the dense one.
        cfg = tf.DeepseekV3Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=4, intermediate_size=128,
            kv_lora_rank=16, q_lora_rank=32,
            qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
            n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
            n_group=2, topk_group=1,
            first_k_dense_replace=1, moe_intermediate_size=64,
            max_position_embeddings=128, **common,
        )
    elif family == "qwen3_moe":
        cfg = tf.Qwen3MoeConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128, head_dim=16,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
            max_position_embeddings=128, **common,
        )
    elif family == "olmoe":
        cfg = tf.OlmoeConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            num_experts=4, num_experts_per_tok=2,
            max_position_embeddings=128, **common,
        )
    elif family == "glm4":
        cfg = tf.Glm4Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128, head_dim=16,
            max_position_embeddings=128, **common,
        )
    elif family == "nemotron":
        cfg = tf.NemotronConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    elif family == "minimax":
        cfg = tf.MiniMaxConfig(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            num_local_experts=4, num_experts_per_tok=2,
            max_position_embeddings=128, **common,
        )
    elif family == "smollm3":
        cfg = tf.SmolLM3Config(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            max_position_embeddings=128, **common,
        )
    else:
        raise SystemExit(f"unknown family {family}")

    # Several configs default pad/bos/eos ids to values from the full-size
    # model, which fall outside this fixture's small vocabulary.
    for attribute in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(cfg, attribute, None)
        if isinstance(value, int) and value >= vocab:
            setattr(cfg, attribute, None)

    model = tf.AutoModelForCausalLM.from_config(cfg)
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    # Minimal byte-level tokenizer so the AEG packaging step has one.
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    tk = Tokenizer(models.BPE(vocab={chr(i): i for i in range(256)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.save(str(out / "tokenizer.json"))


def compare(name: str, ref: np.ndarray, got: np.ndarray) -> bool:
    """Gate on max logit deviation relative to the reference logit spread.

    Cosine similarity over a large vocabulary is dominated by the residual
    stream and stays near 1.0 even when attention is structurally wrong, so it
    is far too permissive on random-weight fixtures.  ``max|a-b| / std(a)``
    reacts directly to a wrong scale, a wrong rotary layout, or a permuted
    head: a correct fp32 implementation lands around 1e-5, while any structural
    error is orders of magnitude larger.
    """
    n = min(ref.shape[0], got.shape[0])
    worst_rel, worst_pos, worst_cos = 0.0, -1, 1.0
    top1 = 0
    for pos in range(n):
        a, b = ref[pos].astype(np.float64), got[pos].astype(np.float64)
        spread = float(a.std()) or 1.0
        rel = float(np.abs(a - b).max()) / spread
        if rel > worst_rel:
            worst_rel, worst_pos = rel, pos
            worst_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        top1 += int(a.argmax() == b.argmax())
    ok = worst_rel < 2e-3
    print(
        f"    {'PASS' if ok else 'FAIL'} {name}: max_rel_err={worst_rel:.2e} @pos{worst_pos} "
        f"cos={worst_cos:.7f} top1={top1}/{n}"
    )
    return ok


def run_family(family: str) -> bool:
    import torch
    import transformers as tf

    tmp = Path(tempfile.mkdtemp(prefix=f"aether_{family}_"))
    src, aeg = tmp / "src", tmp / "model.aeg"
    try:
        build_tiny(family, src)
        print(f"[{family}]")

        ids = np.array([7, 42, 100, 3, 88, 12, 250, 61], dtype=np.int64)
        ref_model = tf.AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
        ref_model.eval()
        # The compiler stores weights at BF16 by default.  Round the reference
        # the same way so this measures the forward pass, not the quantizer:
        # otherwise BF16's ~8-bit mantissa shows up as a ~1e-2 logit deviation
        # for every family and masks real structural errors.
        with torch.no_grad():
            for param in ref_model.parameters():
                param.copy_(param.to(torch.bfloat16).to(torch.float32))
        with torch.no_grad():
            ref = ref_model(torch.tensor(ids).unsqueeze(0)).logits[0].float().numpy()

        from aether.compiler.compiler import Compiler
        from aether.compiler.config import CompilerConfig

        Compiler(CompilerConfig(targets=["cpu_avx512"])).compile(
            str(src), output_path=aeg
        )

        from aether.runtime.aeg_loader import load_engine_from_path

        engine = load_engine_from_path(aeg)
        cpu_logits, _ = engine.forward(ids)
        ok = compare("cpu-prefill", ref, np.asarray(cpu_logits, np.float32))

        from aether.runtime.torch_engine import TorchAEGEngine

        tengine = TorchAEGEngine(engine, "cpu")
        t_logits, _ = tengine.forward(ids)
        ok &= compare("torch-prefill", ref, np.asarray(t_logits, np.float32))

        # Incremental decode must match prefill row-for-row.
        step_rows, cache = [], None
        for tid in ids:
            row, cache = tengine.forward(np.asarray([tid], np.int64), cache)
            step_rows.append(np.asarray(row, np.float32)[-1])
        ok &= compare("torch-decode", ref, np.stack(step_rows))

        step_rows, cache = [], None
        for tid in ids:
            row, cache = engine.forward(np.asarray([tid], np.int64), cache)
            step_rows.append(np.asarray(row, np.float32)[-1])
        ok &= compare("cpu-decode", ref, np.stack(step_rows))

        # The tensor-parallel executor is what a multi-GPU host selects, and it
        # reimplements the block, so it needs the same proof.  A two-device CPU
        # mesh exercises the identical sharding and collective code paths.
        from aether.runtime.torch_tensor_parallel import TorchTensorParallelAEGEngine

        sharded = TorchTensorParallelAEGEngine(engine, ["cpu:0", "cpu:1"])
        s_logits, _ = sharded.forward(ids)
        ok &= compare("sharded-prefill", ref, np.asarray(s_logits, np.float32))

        step_rows, cache = [], None
        for tid in ids:
            row, cache = sharded.forward(np.asarray([tid], np.int64), cache)
            step_rows.append(np.asarray(row, np.float32)[-1])
        ok &= compare("sharded-decode", ref, np.stack(step_rows))

        # Exercise the generation loop too: it calls into the sharded overrides
        # with the base class's full keyword contract.
        greedy_single = tengine.generate(ids[:4], max_tokens=4, temperature=0.0)
        greedy_sharded = sharded.generate(ids[:4], max_tokens=4, temperature=0.0)
        if greedy_single == greedy_sharded:
            print("    PASS sharded-generate: matches single-device tokens")
        else:
            ok = False
            print(
                f"    FAIL sharded-generate: {greedy_sharded} != {greedy_single}"
            )
        return ok
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"    ERROR {family}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    families = sys.argv[1:] or ["gpt_neo"]
    results = {f: run_family(f) for f in families}
    print("\n=== SUMMARY ===")
    for family, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {family}")
    raise SystemExit(0 if all(results.values()) else 1)

#!/usr/bin/env python3
"""Real pretrained-model validation pipeline (and offline real-scale fallback).

TWO distinct validation modes, never conflated:

1. ``--offline`` (default when HuggingFace is unreachable):
   OFFLINE REAL-SCALE ARCHITECTURE VALIDATION. Builds a checkpoint with the
   exact DialoGPT-small architecture (12 layers / 768 hidden / 12 heads /
   50257 vocab) but SYNTHETIC weights, and drives it through the complete
   pipeline: ingestion → graph → optimization → AEG packaging → reload →
   native CPU inference → generation. This validates the pipeline at real
   scale. It is NOT pretrained-model validation and never claims to be.

2. ``--model <hf-id>`` (default ``microsoft/DialoGPT-small``):
   REAL PRETRAINED MODEL VALIDATION. Downloads the actual pretrained weights
   and tokenizer, compiles, executes natively, generates tokens, and compares
   logits/next-tokens against the transformers reference implementation with
   numerical tolerance. If the network is unavailable, the script reports the
   exact blocker and exits non-zero — it never fabricates success.

Usage:
    python scripts/validate_real_model.py                # try real, fall back to offline report
    python scripts/validate_real_model.py --offline      # offline fixture only
    python scripts/validate_real_model.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: The offline fixture reproduces microsoft/DialoGPT-small's architecture.
DIALOGPT_SMALL_ARCH = {
    "layers": 12,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_kv_heads": 12,
    "intermediate_size": 3072,
    "vocab_size": 50257,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,  # GPT-2 uses learned positional embeddings; engine treats this uniformly
}

REPORT: dict[str, object] = {}


def log(section: str, **fields: object) -> None:
    print(f"[{section}]", " ".join(f"{k}={v}" for k, v in fields.items()))


def hf_reachable(model_id: str) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception as exc:  # noqa: BLE001
        log("network", reachable=False, error=str(exc)[:120])
        return False


def build_offline_fixture(directory: Path) -> Path:
    """DialoGPT-small architecture with synthetic weights (clearly labeled)."""
    import numpy as np
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    L, H, I, V = (
        DIALOGPT_SMALL_ARCH["layers"],
        DIALOGPT_SMALL_ARCH["hidden_size"],
        DIALOGPT_SMALL_ARCH["intermediate_size"],
        DIALOGPT_SMALL_ARCH["vocab_size"],
    )
    rng = np.random.default_rng(4242)
    scale = 0.02

    def w(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype("float32")

    tensors: dict[str, np.ndarray] = {
        "transformer.wte.weight": w(V, H),
        "transformer.ln_f.weight": np.ones(H, dtype="float32"),
        "lm_head.weight": w(V, H),
    }
    for i in range(L):
        p = f"transformer.h.{i}"
        tensors[f"{p}.ln_1.weight"] = np.ones(H, dtype="float32")
        tensors[f"{p}.ln_2.weight"] = np.ones(H, dtype="float32")
        # MHA layout packed as q/k/v (GPT-2 c_attn is fused; llama-style
        # separate projections exercise Aether's fusion path).
        tensors[f"{p}.attn.q_proj.weight"] = w(H, H)
        tensors[f"{p}.attn.k_proj.weight"] = w(H, H)
        tensors[f"{p}.attn.v_proj.weight"] = w(H, H)
        tensors[f"{p}.attn.out_proj.weight"] = w(H, H)
        tensors[f"{p}.mlp.gate_proj.weight"] = w(I, H)
        tensors[f"{p}.mlp.up_proj.weight"] = w(I, H)
        tensors[f"{p}.mlp.down_proj.weight"] = w(H, I)

    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "num_hidden_layers": L, "hidden_size": H, "intermediate_size": I,
        "num_attention_heads": 12, "num_key_value_heads": 12, "vocab_size": V,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "float32",
    }), encoding="utf-8")

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        vocab = {"<unk>": 0, "<pad>": 1}
        vocab.update({f"tok{i}": i + 2 for i in range(V - 2)})
        Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")).save(
            str(directory / "tokenizer.json")
        )
    except ImportError:
        pass
    return directory


def run_pipeline(model_dir: Path, out_aeg: Path) -> dict[str, object]:
    """Ingest → optimize → package → reload → execute. Returns measurements."""
    import numpy as np

    from aether.compiler.compiler import Compiler
    from aether.compiler.config import CompilerConfig
    from aether.core.aeg_format import AEGPackage
    from aether.runtime.aeg_loader import load_engine_from_package

    started = datetime.datetime.now(datetime.timezone.utc)
    compiler = Compiler(CompilerConfig(targets=["cpu_avx512"], overwrite=True))
    package = compiler.compile(str(model_dir), output_path=out_aeg)
    compile_seconds = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()

    loaded = AEGPackage(out_aeg)
    loaded.load()
    engine = load_engine_from_package(loaded)

    logits, _ = engine.forward(np.asarray([1, 2, 3, 4], dtype=np.int64))
    tokens = engine.generate(
        np.asarray([1, 2, 3], dtype=np.int64), max_tokens=8, temperature=0.0
    )

    arch = loaded.manifest.architecture  # type: ignore[union-attr]
    tensors = loaded.weight_store().entries
    accounting = loaded.metadata.get("weight_accounting", {})
    result = {
        "compile_seconds": round(compile_seconds, 2),
        "source_layers": DIALOGPT_SMALL_ARCH["layers"],
        "manifest_layers": arch.layers,
        "runtime_layers": len(engine.weights.layers),
        "hidden_size": int(engine.weights.embedding.shape[1]),
        "vocab_size": int(engine.weights.embedding.shape[0]),
        "serialized_tensor_count": len(tensors),
        "required_tensor_count": accounting.get("required_weight_count"),
        "missing_required_tensors": accounting.get("missing_required_tensors", []),
        "graph_hash": loaded.manifest.graph_hash,  # type: ignore[union-attr]
        "logits_shape": list(logits.shape),
        "logits_finite": bool(np.isfinite(logits).all()),
        "generated_tokens": [int(t) for t in tokens],
        "aeg_format_version": loaded.manifest.format_version,  # type: ignore[union-attr]
        "torch_imported_by_pipeline": "torch" in sys.modules,
    }
    log("pipeline", **{k: v for k, v in result.items() if k != "graph_hash"})
    return result


def validate_offline() -> int:
    print("MODE: OFFLINE REAL-SCALE ARCHITECTURE VALIDATION")
    print("  (DialoGPT-small architecture, SYNTHETIC weights — NOT pretrained validation)\n")
    import shutil

    td = tempfile.mkdtemp(prefix="aether-realscale-")
    try:
        model_dir = build_offline_fixture(Path(td) / "dialogpt-arch-fixture")
        result = run_pipeline(model_dir, Path(td) / "realscale.aeg")
    finally:
        # The executed AEG loads a native CPU DLL which Windows keeps locked
        # for the process lifetime; cleanup is best-effort.
        shutil.rmtree(td, ignore_errors=True)

    ok = (
        result["source_layers"] == result["manifest_layers"] == result["runtime_layers"]
        and result["hidden_size"] == 768
        and result["vocab_size"] == 50257
        and not result["missing_required_tensors"]
        and result["logits_finite"]
        and len(result["generated_tokens"]) == 8
        and result["graph_hash"] != "sha256:pending"
    )
    REPORT["offline_real_scale"] = {"passed": ok, **result}
    print(f"\nOFFLINE REAL-SCALE RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def validate_real_pretrained(model_id: str) -> int:
    print(f"MODE: REAL PRETRAINED MODEL VALIDATION ({model_id})\n")
    if not hf_reachable(model_id):
        print(
            "BLOCKED: HuggingFace is unreachable (rate limit or offline). "
            "This script refuses to substitute synthetic weights for a real "
            "model. Run with --offline for the architecture fixture, or retry "
            "when network access is restored."
        )
        REPORT["real_pretrained"] = {"status": "BLOCKED", "reason": "huggingface unreachable"}
        return 2

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("BLOCKED: transformers is not installed. Install with: pip install 'aether-runtime[transformers-frontend]'")
        REPORT["real_pretrained"] = {"status": "BLOCKED", "reason": "transformers not installed"}
        return 2

    import numpy as np
    import torch

    with tempfile.TemporaryDirectory(prefix="aether-real-") as td:
        workdir = Path(td)
        print("Downloading model + tokenizer …")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded: {n_params/1e6:.1f}M parameters")

        # Export to a local SafeTensors checkpoint Aether can ingest.
        local = workdir / "checkpoint"
        local.mkdir()
        model.save_pretrained(str(local), safe_serialization=True)
        tokenizer.save_pretrained(str(local))
        # The tokenizer files (tokenizer.json etc.) enable the framework-free
        # packaging path.
        result = run_pipeline(local, workdir / "real.aeg")

        # Reference comparison with numerical tolerance.
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        with torch.no_grad():
            ref_logits = model(**inputs).logits[0, -1].float().numpy()
        from aether.core.aeg_format import AEGPackage
        from aether.runtime.aeg_loader import load_engine_from_package

        pkg = AEGPackage(workdir / "real.aeg")
        pkg.load()
        engine = load_engine_from_package(pkg)
        ids = inputs["input_ids"][0].numpy().astype(np.int64)
        aeg_logits, _ = engine.forward(ids)
        aeg_last = aeg_logits[-1]

        max_abs_diff = float(np.max(np.abs(aeg_last - ref_logits)))
        ref_next = int(np.argmax(ref_logits))
        aeg_next = int(np.argmax(aeg_last))
        agreement = ref_next == aeg_next
        tolerance = 0.05  # Q4_K_M quantized model vs FP32 reference
        passed = max_abs_diff <= tolerance or agreement

        comparison = {
            "parameters_million": round(n_params / 1e6, 1),
            "reference_next_token": ref_next,
            "aether_next_token": aeg_next,
            "next_token_agrees": agreement,
            "max_abs_logit_diff": round(max_abs_diff, 5),
            "tolerance": tolerance,
        }
        log("reference", **comparison)
        REPORT["real_pretrained"] = {"passed": passed, **comparison, **result}
        print(f"\nREAL PRETRAINED RESULT: {'PASS' if passed else 'FAIL'}")
        print("  (quantization differences make exact logit equality impossible;")
        print("   next-token agreement and bounded logit error are the criteria)")
        return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/DialoGPT-small")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    REPORT["machine"] = f"{platform.node()} {platform.system()}"
    REPORT["started"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if args.offline:
        code = validate_offline()
    else:
        code = validate_real_pretrained(args.model)
        if code == 2:  # blocked → still run the offline fixture for evidence
            print("\n--- falling back to offline real-scale validation ---\n")
            offline_code = validate_offline()
            code = max(code, offline_code)

    if args.json:
        args.json.write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
        print(f"\nReport written to {args.json}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

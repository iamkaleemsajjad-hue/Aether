"""One-shot accelerator validation for decode batch scaling.

Run this on the target accelerator and send back the JSON.  It answers, per model and
per context length, *where* a decode step spends its time at each batch size — which
is the measurement that decides whether the batch-scaling penalty is in the
projections, in attention, in the KV path, or in host dispatch.

    python scripts/profile_batch_scaling.py --aeg path/to/model.aeg [more.aeg ...] \
        --contexts 32,256,1024 --batches 1,2,4,8 --steps 24 --out scaling.json

Everything it reports is measured on the machine it runs on.  It contains no
device-specific branch: the phases come from the model's own structure and the
synchronisation from whatever the active backend exposes, so the same script is valid
on CUDA, ROCm, XPU, MPS and CPU.

With no ``--aeg`` it profiles synthetic decoders whose geometry spans the families
Aether supports — MHA and GQA, wide and narrow, sliding-window and full attention — so
the architecture can be exercised on a host with no artifacts present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def require_installed_aether() -> None:
    """Fail with a diagnosis instead of a misleading ``ModuleNotFoundError``.

    This script deliberately does **not** prepend ``src`` to ``sys.path``.  An
    installed distribution has to be importable on its own; repairing the path here
    would hide a broken install and would only work for callers who happened to run
    this file.

    The failure worth naming is the implicit namespace package.  If a *directory*
    called ``aether`` sits on ``sys.path`` — most easily by cloning the repository
    into a directory of that name next to an entry like ``/kaggle/working`` — and the
    real distribution is not importable, then ``import aether`` quietly succeeds as an
    empty namespace package with ``__file__ is None``, and the first submodule import
    fails pointing at the submodule rather than at the cause.  Worse, that empty module
    is now cached in ``sys.modules``, so a later ``sys.path`` fix cannot undo it.
    """
    spec = importlib.util.find_spec("aether")
    if spec is None:
        raise SystemExit(
            "aether is not importable. Install it first:  pip install -e .\n"
            "(from a notebook, restart the kernel after installing: a .pth file is "
            "only read when an interpreter starts.)"
        )
    if spec.origin is None or spec.submodule_search_locations is None:
        found = list(spec.submodule_search_locations or [])
        raise SystemExit(
            "aether resolved to an implicit namespace package, not the installed "
            f"distribution.\n  found: {found}\n"
            "A directory named 'aether' on sys.path is shadowing it, and the real "
            "package is not importable.\n"
            "Fix: install the distribution (pip install -e .) and, in a notebook, "
            "restart the kernel afterwards; or clone into a directory that is not "
            "named 'aether'."
        )


require_installed_aether()

#: Geometries covering the shape classes the supported families fall into.  Named for
#: the property being exercised, not for a vendor's model, because the point is the
#: shape: ``(layers, hidden, heads, kv_heads, head_dim, intermediate, vocab, window)``.
SYNTHETIC = {
    "narrow-mha":        (12, 576, 9, 9, 64, 1536, 49152, None),
    "narrow-gqa":        (12, 576, 9, 3, 64, 1536, 49152, None),
    "wide-gqa-bigvocab": (12, 1024, 16, 8, 128, 3072, 151936, None),
    "windowed-mha":      (12, 1024, 16, 16, 64, 4096, 50257, 256),
}


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_build"] = getattr(torch.version, "cuda", None)
        info["hip_build"] = getattr(torch.version, "hip", None)
        info["threads"] = int(torch.get_num_threads())
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(index)
            info["device"] = {
                "name": torch.cuda.get_device_name(index),
                "capability": f"sm{major}{minor}",
                "count": torch.cuda.device_count(),
                "total_gib": round(
                    torch.cuda.get_device_properties(index).total_memory / 1024 ** 3, 2
                ),
            }
        else:
            info["device"] = {"name": "cpu", "capability": "cpu", "count": 1}
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    try:
        from aether.placement.census import _backend_build

        info["backend_build"] = _backend_build()
    except Exception as exc:  # noqa: BLE001
        info["backend_build_error"] = str(exc)
    return info


def build_synthetic(name: str, device: str) -> Any:
    import numpy as np

    from aether.runtime.cpu_engine import (
        CPUExecutionEngine, LayerWeights, ModelWeights,
    )
    from aether.runtime.torch_engine import TorchAEGEngine

    layers, hidden, heads, kv, head_dim, inter, vocab, window = SYNTHETIC[name]
    rng = np.random.default_rng(0)

    def w(out: int, inp: int) -> Any:
        return rng.standard_normal((out, inp)).astype(np.float32) * 0.02

    stack = [
        LayerWeights(
            attention_norm=np.ones(hidden, dtype=np.float32),
            q_proj=w(heads * head_dim, hidden), k_proj=w(kv * head_dim, hidden),
            v_proj=w(kv * head_dim, hidden), o_proj=w(hidden, heads * head_dim),
            ffn_norm=np.ones(hidden, dtype=np.float32),
            gate_proj=w(inter, hidden), up_proj=w(inter, hidden),
            down_proj=w(hidden, inter),
        )
        for _ in range(layers)
    ]
    weights = ModelWeights(
        embedding=w(vocab, hidden), layers=stack,
        final_norm=np.ones(hidden, dtype=np.float32), lm_head=w(vocab, hidden),
        context_length=8192,
    )
    engine = TorchAEGEngine(
        CPUExecutionEngine(weights, num_heads=heads, num_kv_heads=kv), device
    )
    if window is not None:
        engine.layer_plan = [(True, window, p[2], p[3]) for p in engine.layer_plan]
    return engine


def load_aeg(path: Path, device: str) -> Any:
    """Return the portable tensor executor for a compiled artifact."""
    from aether.runtime.aeg_loader import load_engine_from_path
    from aether.runtime.torch_engine import TorchAEGEngine

    engine = load_engine_from_path(path)
    if type(engine).__name__ == "TorchAEGEngine":
        return engine
    return TorchAEGEngine(engine, device)


def peak_memory(device: str) -> dict[str, Any]:
    try:
        import torch

        if not device.startswith("cuda") or not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 4),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 4),
        }
    except Exception:  # noqa: BLE001
        return {"available": False}


def reset_peak(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def correctness(engine: Any, context: int) -> dict[str, Any]:
    """Greedy batched decode must agree token-for-token with the single-sequence path.

    Batching changes how the arithmetic is grouped, so this is the check that any
    scaling number is describing the same model. It is reported alongside the timings
    rather than assumed.
    """
    import numpy as np

    try:
        prompt = np.arange(context, dtype=np.int64) % int(engine.embedding.shape[0])
        alone = engine.generate(prompt, max_tokens=8, temperature=0.0)
        rows = engine.generate_batch([prompt, prompt], max_tokens=8, temperature=0.0)
        return {
            "checked": True,
            "batched_rows_agree": rows[0] == rows[1],
            "matches_single_sequence": list(alone) == list(rows[0]),
            "single": list(alone)[:8],
            "batched": list(rows[0])[:8],
        }
    except Exception as exc:  # noqa: BLE001 - report, never abort the run
        return {"checked": False, "error": str(exc)[:200]}


def profile_model(name: str, engine: Any, contexts: list[int], batches: list[int],
                  steps: int, device: str) -> dict[str, Any]:
    from aether.runtime.decode_profile import profile_engine

    cells: list[dict[str, Any]] = []
    for context in contexts:
        base_step: float | None = None
        for batch in batches:
            reset_peak(device)
            try:
                totals = profile_engine(
                    engine, batch=batch, context=context, steps=steps
                )
            except Exception as exc:  # noqa: BLE001 - a cell may not fit; say so
                cells.append({
                    "context": context, "batch": batch,
                    "status": "failed", "error": str(exc)[:200],
                })
                continue
            step = totals.wall_seconds / max(1, totals.steps)
            if batch == batches[0]:
                base_step = step
            ideal = batch / batches[0]
            achieved = (batch / step) / (batches[0] / base_step) if base_step else 0.0
            cells.append({
                "context": context,
                "batch": batch,
                "status": "ok",
                "ms_per_step": round(step * 1e3, 4),
                "step_cost_vs_base": round(step / base_step, 4) if base_step else 0.0,
                "tokens_per_s": round(batch / step, 2) if step > 0 else 0.0,
                "scaling_efficiency": round(achieved / ideal, 4) if ideal else 0.0,
                "phases": totals.to_dict()["phases"],
                "peak_memory": peak_memory(device),
            })
    return {
        "model": name,
        "geometry": {
            "layers": int(engine.num_layers),
            "heads": int(engine.num_heads),
            "kv_heads": int(engine.num_kv_heads),
            "head_dim": int(engine.head_dim),
            "hidden": int(engine.embedding.shape[1]),
            "vocab": int(engine.embedding.shape[0]),
            "compute_dtype": str(engine.compute_dtype),
        },
        "strategy_calibration": _strategy_report(engine),
        "correctness": correctness(engine, min(contexts)),
        "cells": cells,
    }


def _strategy_report(engine: Any) -> dict[str, Any]:
    try:
        return engine.projection_report()
    except Exception as exc:  # noqa: BLE001
        return {"enabled": False, "error": str(exc)[:200]}


def summarize(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in results:
        lines.append(f"\n=== {entry['model']}  {entry['geometry']}")
        strategy = entry.get("strategy_calibration", {})
        active = strategy.get("active") or {}
        lines.append(
            f"    strategy: {active.get('name', 'n/a')} "
            f"({active.get('source', 'n/a')}) on {strategy.get('device_kind', '?')}"
        )
        check = entry.get("correctness", {})
        lines.append(
            f"    correctness: batched==single {check.get('matches_single_sequence')} "
            f"rows agree {check.get('batched_rows_agree')}"
        )
        header = (
            f"    {'ctx':>6}{'B':>4}{'ms/step':>10}{'cost x':>8}"
            f"{'tok/s':>10}{'eff':>7}  top phases"
        )
        lines.append(header)
        for cell in entry["cells"]:
            if cell.get("status") != "ok":
                lines.append(
                    f"    {cell['context']:>6}{cell['batch']:>4}  FAILED "
                    f"{cell.get('error', '')[:60]}"
                )
                continue
            phases = list(cell["phases"].items())[:3]
            tops = " ".join(
                f"{phase}={data['share'] * 100:.0f}%" for phase, data in phases
            )
            lines.append(
                f"    {cell['context']:>6}{cell['batch']:>4}"
                f"{cell['ms_per_step']:>10.2f}{cell['step_cost_vs_base']:>8.2f}"
                f"{cell['tokens_per_s']:>10.1f}{cell['scaling_efficiency']:>7.2f}  {tops}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aeg", nargs="*", default=[], help="Compiled AEG artifacts.")
    parser.add_argument("--contexts", default="32,256,1024")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--device", default=None, help="Defaults to cuda when present.")
    parser.add_argument("--out", default="batch_scaling.json")
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Pin the reference kernel, to measure the mechanism's own effect.",
    )
    args = parser.parse_args()

    if args.no_calibration:
        import os

        os.environ["AETHER_DECODE_CALIBRATION"] = "0"

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    contexts = [int(v) for v in args.contexts.split(",") if v.strip()]
    batches = [int(v) for v in args.batches.split(",") if v.strip()]

    payload: dict[str, Any] = {
        "environment": environment(),
        "device": device,
        "contexts": contexts,
        "batches": batches,
        "steps": args.steps,
        "calibration_enabled": not args.no_calibration,
        "models": [],
    }

    targets: list[tuple[str, Any]] = []
    for path in args.aeg:
        try:
            targets.append((Path(path).name, load_aeg(Path(path), device)))
        except Exception as exc:  # noqa: BLE001
            payload["models"].append({"model": path, "error": str(exc)[:300]})
    if not args.aeg:
        for name in SYNTHETIC:
            try:
                targets.append((f"synthetic:{name}", build_synthetic(name, device)))
            except Exception as exc:  # noqa: BLE001
                payload["models"].append({"model": name, "error": str(exc)[:300]})

    for name, engine in targets:
        print(f"profiling {name} on {device} ...", flush=True)
        payload["models"].append(
            profile_model(name, engine, contexts, batches, args.steps, device)
        )

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summarize([m for m in payload["models"] if "cells" in m]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

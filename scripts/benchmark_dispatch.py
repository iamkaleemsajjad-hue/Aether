"""Compare per-token ATen dispatch counts: Aether vs HuggingFace.

Small-model decode is launch-bound, not compute-bound: a 350M model in FP16
reads ~700 MB of weights per token, which a V100 streams in under a millisecond,
so wall-clock is dominated by how many kernels are launched per token rather
than by arithmetic.  Counting dispatches is therefore the deterministic,
machine-independent way to compare the two runtimes' decode overhead.

    python scripts/benchmark_dispatch.py <family> [layers]
"""
from __future__ import annotations

import collections
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("AETHER_TORCH_DTYPE", "fp32")

sys.path.insert(0, str(Path(__file__).parent))


class DispatchCounter:
    """Count ATen calls, separating real kernels from metadata-only views."""

    #: Ops that only rewrite tensor metadata.  On an accelerator these launch
    #: no kernel, so they must not be counted against either runtime.
    VIEW_OPS = frozenset({
        "aten.slice.Tensor", "aten.view.default", "aten.reshape.default",
        "aten._unsafe_view.default", "aten.transpose.int", "aten.permute.default",
        "aten.unsqueeze.default", "aten.squeeze.default", "aten.squeeze.dim",
        "aten.expand.default", "aten.detach.default", "aten.alias.default",
        "aten.select.int", "aten.split.Tensor", "aten.split_with_sizes.default",
        "aten.flatten.using_ints", "aten.contiguous.default", "aten.t.default",
        "aten.chunk.default", "aten.narrow.default",
    })

    def __init__(self) -> None:
        self.by_op: collections.Counter[str] = collections.Counter()

    def __enter__(self) -> "DispatchCounter":
        from torch.utils._python_dispatch import TorchDispatchMode

        counter = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                counter.by_op[str(func)] += 1
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._mode.__exit__(*exc)

    @property
    def kernels(self) -> int:
        return sum(n for op, n in self.by_op.items() if op not in self.VIEW_OPS)

    @property
    def total(self) -> int:
        return sum(self.by_op.values())

    def top(self, limit: int = 12) -> list[tuple[str, int]]:
        return [
            (op, n) for op, n in self.by_op.most_common()
            if op not in self.VIEW_OPS
        ][:limit]


def main() -> int:
    from validate_family_parity import build_tiny

    family = sys.argv[1] if len(sys.argv) > 1 else "gpt_neo"
    steps = 8

    import torch
    import transformers as tf

    tmp = Path(tempfile.mkdtemp(prefix=f"dispatch_{family}_"))
    src, aeg = tmp / "src", tmp / "model.aeg"
    try:
        build_tiny(family, src)
        prompt = np.array([7, 42, 100, 3], dtype=np.int64)

        # ── HuggingFace reference ──────────────────────────────────────────
        ref = tf.AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
        ref.eval()
        ids = torch.tensor(prompt).unsqueeze(0)
        with torch.no_grad():
            warm = ref(ids, use_cache=True)
            past = warm.past_key_values
            next_id = warm.logits[:, -1:].argmax(-1)
            hf = DispatchCounter()
            with hf, torch.no_grad():
                for _ in range(steps):
                    out = ref(next_id, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    next_id = out.logits[:, -1:].argmax(-1)

        # ── Aether ─────────────────────────────────────────────────────────
        from aether.compiler.compiler import Compiler
        from aether.compiler.config import CompilerConfig
        from aether.runtime.aeg_loader import load_engine_from_path
        from aether.runtime.torch_engine import TorchAEGEngine

        Compiler(CompilerConfig(targets=["cpu_avx512"])).compile(str(src), output_path=aeg)
        engine = TorchAEGEngine(load_engine_from_path(aeg), "cpu")
        engine.generate(prompt, max_tokens=2, temperature=0.0)  # warm
        aether = DispatchCounter()
        with aether:
            engine.generate(prompt, max_tokens=steps, temperature=0.0)

        layers = engine.num_layers
        print(f"\n[{family}] layers={layers}, {steps} decode steps\n")
        print(f"{'runtime':10s} {'kernels/token':>14s} {'per layer':>10s} {'views/token':>12s}")
        for label, counter in (("HF", hf), ("Aether", aether)):
            per_token = counter.kernels / steps
            print(
                f"{label:10s} {per_token:14.1f} {per_token / layers:10.2f} "
                f"{(counter.total - counter.kernels) / steps:12.1f}"
            )
        ratio = (aether.kernels / max(hf.kernels, 1))
        print(f"\nAether launches {ratio:.2f}x HF's kernels per token\n")
        for label, counter in (("HF", hf), ("Aether", aether)):
            print(f"  {label} top ops per token:")
            for op, n in counter.top(10):
                print(f"     {n / steps:7.1f}  {op}")
            print()
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

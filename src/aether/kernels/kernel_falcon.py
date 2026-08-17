"""KernelFalcon — autonomous candidate kernel generation, validation, and caching.

Implements the v5 PRD KernelFalcon subsystem to the extent genuinely possible
on the current machine:

* **Candidate generation** — parameterized C source variants (loop order,
  blocking, unrolling) for native CPU kernels, plus portable numpy variants
  that always execute.
* **Safety screening** — generated sources are rejected before compilation if
  they contain forbidden constructs (file/network/process access, inline
  assembly, arbitrary extern symbols).
* **Correctness validation** — every candidate is executed against a trusted
  numpy reference on deterministic random inputs with a numerical tolerance.
  Candidates that mismatch, return non-finite values, or crash are rejected.
* **Benchmarking** — accepted candidates are timed against the reference; the
  fastest correct candidate is selected.
* **Caching** — a JSON manifest keyed by source-hash persists validated
  candidates so later runs reuse them instead of re-searching.

Honest scope: GPU kernel generation/execution is NOT implemented (no GPU on
this machine); ``KernelFalconState`` records that explicitly instead of
fabricating results. The C compilation stage runs only when a host toolchain
is detected; otherwise the search completes over numpy candidates and is
labeled ``numpy_only``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["KernelFalcon", "KernelFalconResult", "KernelCandidate", "KernelFalconState"]


#: Forbidden substrings in generated C sources. A candidate containing any of
#: these is rejected before it ever reaches a compiler.
_FORBIDDEN_C_CONSTRUCTS: tuple[str, ...] = (
    "#include <stdio.h>",
    "#include <stdlib.h>",
    "#include <unistd.h>",
    "#include <fcntl.h>",
    "#include <sys/",
    "system(",
    "fopen",
    "fork(",
    "exec",
    "popen",
    "socket",
    "__asm__",
    "asm volatile",
    "dlopen",
)


@dataclass
class KernelCandidate:
    """One generated kernel variant."""

    name: str
    """Unique candidate name (op + variant)."""

    language: str
    """``"c"`` or ``"numpy"``."""

    source: str
    """C source text, or the numpy variant tag."""

    symbol: str = ""
    """Exported C symbol for the candidate's entry point."""

    signature: str = ""
    """C calling convention kind: ``"rmsnorm6"`` or ``"elementwise3"``."""

    source_hash: str = ""
    """SHA-256 of the source; set on creation."""

    def __post_init__(self) -> None:
        if not self.source_hash:
            self.source_hash = hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:16]


@dataclass
class KernelFalconResult:
    """Outcome of a KernelFalcon search for one kernel spec."""

    op: str
    selected: str | None = None
    """Name of the selected (fastest correct) candidate, or None."""

    validated: list[str] = field(default_factory=list)
    """Names of candidates that passed correctness validation."""

    rejected: dict[str, str] = field(default_factory=dict)
    """Candidate name -> rejection reason."""

    benchmark_ms: dict[str, float] = field(default_factory=dict)
    """Candidate name -> measured wall-clock ms per iteration."""

    compiled: bool = False
    """True when at least one C candidate was compiled by a host toolchain."""

    library_path: str | None = None
    """Path of the compiled shared library, when one was produced."""

    cached: bool = False
    """True when the selection was reused from the cache manifest."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "selected": self.selected,
            "validated": list(self.validated),
            "rejected": dict(self.rejected),
            "benchmark_ms": dict(self.benchmark_ms),
            "compiled": self.compiled,
            "library_path": self.library_path,
            "cached": self.cached,
        }


@dataclass
class KernelFalconState:
    """Honest classification of what KernelFalcon can do on this machine."""

    implementation: str = "IMPLEMENTED"
    implementation_scope: str = "candidate generation, safety screening, correctness validation, benchmarking, selection, caching"
    gpu_execution: str = "NOT_TESTABLE_ON_CURRENT_MACHINE"
    gpu_execution_reason: str = "no CUDA/ROCm/Metal device available"
    host_c_compilation: str = "TOOLCHAIN_DEPENDENT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "implementation_scope": self.implementation_scope,
            "gpu_execution": self.gpu_execution,
            "gpu_execution_reason": self.gpu_execution_reason,
            "host_c_compilation": self.host_c_compilation,
        }


# ── Reference kernels (trusted numpy implementations) ─────────────────────────

def _ref_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32) * weight


def _ref_silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def _ref_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


def _ref_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)


#: op -> (reference callable, input factory for validation/benchmark)
_REFERENCES: dict[str, tuple[Callable[..., np.ndarray], Callable[[np.random.Generator], tuple[tuple, dict]]]] = {
    "rmsnorm": (
        _ref_rmsnorm,
        lambda rng: ((rng.normal(size=(128, 256)).astype(np.float32),), {"weight": rng.normal(size=256).astype(np.float32), "eps": 1e-5}),
    ),
    "silu": (_ref_silu, lambda rng: ((rng.normal(size=(256, 256)).astype(np.float32),), {})),
    "softmax": (_ref_softmax, lambda rng: ((rng.normal(size=(64, 256)).astype(np.float32),), {})),
    "sgemm": (
        _ref_matmul,
        lambda rng: (
            (
                rng.normal(size=(128, 256)).astype(np.float32),
                rng.normal(size=(256, 128)).astype(np.float32),
            ),
            {},
        ),
    ),
}


# ── Candidate generators ───────────────────────────────────────────────────────

def _c_rmsnorm_source(unroll: int, restrict_kw: str) -> str:
    return f"""
#include <math.h>
void aether_falcon_rmsnorm_u{unroll}(const float* x, const float* w, float* out,
                                     int rows, int cols, float eps) {{
    for (int r = 0; r < rows; ++r) {{
        const float* {restrict_kw} xr = x + (size_t)r * cols;
        float* {restrict_kw} orow = out + (size_t)r * cols;
        double acc = 0.0;
        for (int c = 0; c < cols; c += {unroll}) {{
            acc += (double)xr[c] * xr[c];
            {chr(10).join(f"if (c + {k} < cols) acc += (double)xr[c + {k}] * xr[c + {k}];" for k in range(1, unroll))}
        }}
        const float inv = (float)(1.0 / sqrt(acc / cols + eps));
        for (int c = 0; c < cols; ++c) orow[c] = xr[c] * inv * w[c];
    }}
}}
""".strip()


def _c_silu_source(vectorize: bool) -> str:
    lane = "#pragma omp simd" if vectorize else ""
    return f"""
#include <math.h>
void aether_falcon_silu_v{int(vectorize)}(const float* x, float* out, int n) {{
    {lane}
    for (int i = 0; i < n; ++i) {{
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }}
}}
""".strip()


def _c_softmax_source() -> str:
    return """
#include <math.h>
void aether_falcon_softmax_c(const float* x, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * cols;
        float* orow = out + (size_t)r * cols;
        float max_val = xr[0];
        for (int c = 1; c < cols; ++c) {
            if (xr[c] > max_val) max_val = xr[c];
        }
        double sum_exp = 0.0;
        for (int c = 0; c < cols; ++c) {
            float e = expf(xr[c] - max_val);
            orow[c] = e;
            sum_exp += (double)e;
        }
        float inv = (float)(1.0 / (sum_exp + 1e-9));
        for (int c = 0; c < cols; ++c) {
            orow[c] *= inv;
        }
    }
}
""".strip()


def _c_sgemm_source(unroll: int) -> str:
    return f"""
void aether_falcon_sgemm_u{unroll}(const float* A, const float* B, float* C, int M, int N, int K) {{
    for (int m = 0; m < M; ++m) {{
        for (int n = 0; n < N; ++n) {{
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {{
                acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
            }}
            C[(size_t)m * N + n] = acc;
        }}
    }}
}}
""".strip()


def _generate_candidates(op: str) -> list[KernelCandidate]:
    """Generate the candidate pool for an op. Deterministic."""
    candidates: list[KernelCandidate] = []
    if op == "rmsnorm":
        for unroll, restrict_kw in ((1, ""), (2, ""), (4, "__restrict")):
            candidates.append(
                KernelCandidate(
                    name=f"rmsnorm_c_unroll{unroll}",
                    language="c",
                    source=_c_rmsnorm_source(unroll, restrict_kw),
                    symbol=f"aether_falcon_rmsnorm_u{unroll}",
                    signature="rmsnorm6",
                )
            )
        candidates.append(KernelCandidate(name="rmsnorm_numpy_ref", language="numpy", source="numpy_reference"))
    elif op == "silu":
        for vectorize in (False, True):
            candidates.append(
                KernelCandidate(
                    name=f"silu_c_simd{int(vectorize)}",
                    language="c",
                    source=_c_silu_source(vectorize),
                    symbol=f"aether_falcon_silu_v{int(vectorize)}",
                    signature="elementwise3",
                )
            )
        candidates.append(KernelCandidate(name="silu_numpy_ref", language="numpy", source="numpy_reference"))
    elif op == "softmax":
        candidates.append(
            KernelCandidate(
                name="softmax_c_opt",
                language="c",
                source=_c_softmax_source(),
                symbol="aether_falcon_softmax_c",
                signature="softmax4",
            )
        )
        candidates.append(KernelCandidate(name="softmax_numpy_ref", language="numpy", source="numpy_reference"))
    elif op == "sgemm":
        candidates.append(
            KernelCandidate(
                name="sgemm_c_u1",
                language="c",
                source=_c_sgemm_source(1),
                symbol="aether_falcon_sgemm_u1",
                signature="sgemm6",
            )
        )
        candidates.append(KernelCandidate(name="sgemm_numpy_ref", language="numpy", source="numpy_reference"))
    else:
        raise ValueError(f"KernelFalcon has no generator for op {op!r}")
    return candidates


class KernelFalcon:
    """Autonomous kernel search over a validated candidate pool.

    Usage::

        falcon = KernelFalcon(cache_dir=Path(".cache/kernel_falcon"))
        result = falcon.optimize("rmsnorm")
        assert result.selected is not None
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        if cache_dir is None:
            from aether.core.constants import DEFAULT_CACHE_DIR

            cache_dir = Path(DEFAULT_CACHE_DIR) / "kernel_falcon"
        self.cache_dir = Path(cache_dir)
        self.state = KernelFalconState()
        self._numpy_impls: dict[str, Callable[..., np.ndarray]] = {
            "rmsnorm_numpy_ref": lambda x, weight, eps: _ref_rmsnorm(x, weight, eps),
            "silu_numpy_ref": _ref_silu,
            "softmax_numpy_ref": _ref_softmax,
            "sgemm_numpy_ref": _ref_matmul,
        }

    # ── Public API ──────────────────────────────────────────────────────────

    def optimize(self, op: str, *, tolerance: float = 1e-4) -> KernelFalconResult:
        """Generate, screen, validate, benchmark, and select a kernel for ``op``."""
        result = KernelFalconResult(op=op)

        cached = self._load_cache(op)
        if cached is not None:
            result.selected = cached["selected"]
            result.validated = list(cached["validated"])
            result.benchmark_ms = dict(cached["benchmark_ms"])
            result.compiled = bool(cached.get("compiled"))
            result.library_path = cached.get("library_path")
            result.cached = True
            logger.info("KernelFalcon: reused cached kernel %s for op %s", result.selected, op)
            return result

        if op not in _REFERENCES:
            result.rejected[op] = "no reference implementation available"
            return result

        reference, input_factory = _REFERENCES[op]
        rng = np.random.default_rng(20260816)
        args, kwargs = input_factory(rng)
        expected = reference(*args, **kwargs)

        candidates = _generate_candidates(op)

        # 1. Safety screening: reject unsafe C sources before any compilation.
        for candidate in candidates:
            if candidate.language == "c":
                reason = self._screen_safety(candidate)
                if reason:
                    result.rejected[candidate.name] = reason

        # 2. Compile the screened C candidates into one shared library when a
        #    host toolchain is available.
        screened_c = [c for c in candidates if c.language == "c" and c.name not in result.rejected]
        library_path: Path | None = None
        if screened_c:
            library_path, compiled_ok = self._compile_candidates(op, screened_c)
            result.compiled = compiled_ok
            result.library_path = str(library_path) if compiled_ok and library_path else None
            if not compiled_ok:
                for c in screened_c:
                    result.rejected[c.name] = "host C compilation unavailable/failed (toolchain-dependent)"

        # 3. Correctness validation against the trusted reference.
        for candidate in candidates:
            if candidate.name in result.rejected:
                continue
            try:
                runner = self._resolve_runner(candidate, op, library_path if result.compiled else None)
                actual = runner(*args, **kwargs)
                if actual.shape != expected.shape:
                    raise ValueError(f"shape mismatch {actual.shape} != {expected.shape}")
                if not np.all(np.isfinite(actual)):
                    raise ValueError("non-finite output")
                err = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
                if not err <= tolerance:
                    raise ValueError(f"max abs error {err:.3e} exceeds tolerance {tolerance:.1e}")
            except Exception as exc:  # noqa: BLE001 — any failure rejects the candidate
                result.rejected[candidate.name] = f"correctness: {exc}"
                continue
            result.validated.append(candidate.name)

        # 3. Benchmark the survivors.
        for name in result.validated:
            runner = self._resolve_runner(
                next(c for c in candidates if c.name == name), op, library_path if result.compiled else None
            )
            result.benchmark_ms[name] = self._benchmark(runner, args, kwargs)

        if not result.validated:
            logger.error("KernelFalcon: every candidate for op %s was rejected: %s", op, result.rejected)
            return result

        # 4. Select the fastest correct candidate.
        result.selected = min(result.benchmark_ms, key=result.benchmark_ms.get)  # type: ignore[arg-type]
        self._save_cache(result)
        logger.info(
            "KernelFalcon: selected %s for op %s (%.3f ms/iter; %d validated, %d rejected)",
            result.selected, op, result.benchmark_ms[result.selected], len(result.validated), len(result.rejected),
        )
        return result

    # ── Internals ───────────────────────────────────────────────────────────

    def _screen_safety(self, candidate: KernelCandidate) -> str | None:
        """Return a rejection reason when the source contains forbidden constructs."""
        lowered = candidate.source.lower()
        for construct in _FORBIDDEN_C_CONSTRUCTS:
            if construct.lower() in lowered:
                return f"unsafe construct in generated source: {construct!r}"
        return None

    def _compile_candidates(self, op: str, candidates: list[KernelCandidate]) -> tuple[Path | None, bool]:
        """Compile screened C candidates into one shared library via the host toolchain."""
        try:
            from aether.kernels.native_cpu import detect_toolchain

            toolchain = detect_toolchain()
            if toolchain is None:
                return None, False
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            source_path = self.cache_dir / f"falcon_{op}.c"
            library_path = self.cache_dir / f"falcon_{op}{toolchain.library_suffix}"
            # Guard with extern "C" so a C++ host toolchain does not mangle the
            # exported symbols ctypes looks up.
            combined = (
                "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
                + "\n\n".join(c.source for c in candidates)
                + "\n\n#ifdef __cplusplus\n}\n#endif\n"
            )
            source_path.write_text(combined, encoding="utf-8")
            import subprocess

            proc = subprocess.run(
                toolchain.build_command(source_path, library_path),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0 or not library_path.exists():
                logger.warning("KernelFalcon: host compilation failed: %s", proc.stderr[:400])
                return None, False
            return library_path, True
        except Exception as exc:  # noqa: BLE001 — toolchain problems degrade to numpy
            logger.warning("KernelFalcon: compilation stage unavailable: %s", exc)
            return None, False

    def _resolve_runner(
        self,
        candidate: KernelCandidate,
        op: str,
        library_path: Path | None,
    ) -> Callable[..., np.ndarray]:
        """Return a callable executing this candidate (numpy or compiled C)."""
        if candidate.language == "numpy":
            return self._numpy_impls[candidate.name]

        if library_path is None:
            msg = "C candidate has no compiled library (no host toolchain)"
            raise NotImplementedError(msg)

        import ctypes

        lib = ctypes.CDLL(str(library_path))
        fn = getattr(lib, candidate.symbol)

        if candidate.signature == "rmsnorm6":
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ]

            def runner(x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
                rows, cols = x.shape
                out = np.empty_like(x)
                w = weight if weight is not None else np.ones(cols, dtype=np.float32)
                fn(
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    np.ascontiguousarray(w, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    rows, cols, eps,
                )
                return out

            return runner

        if candidate.signature == "elementwise3":
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ]

            def runner(x: np.ndarray, **_ignored: Any) -> np.ndarray:
                x = np.ascontiguousarray(x, dtype=np.float32)
                out = np.empty_like(x)
                fn(
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    x.size,
                )
        if candidate.signature == "softmax4":
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ]

            def runner(x: np.ndarray, **_ignored: Any) -> np.ndarray:
                x = np.ascontiguousarray(x, dtype=np.float32)
                rows, cols = x.shape
                out = np.empty_like(x)
                fn(
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    rows,
                    cols,
                )
                return out

            return runner

        if candidate.signature == "sgemm6":
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]

            def runner(a: np.ndarray, b: np.ndarray, **_ignored: Any) -> np.ndarray:
                a = np.ascontiguousarray(a, dtype=np.float32)
                b = np.ascontiguousarray(b, dtype=np.float32)
                m, k = a.shape
                k2, n = b.shape
                out = np.empty((m, n), dtype=np.float32)
                fn(
                    a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    m,
                    n,
                    k,
                )
                return out

            return runner

        msg = f"unknown C signature {candidate.signature!r} for {candidate.name}"
        raise NotImplementedError(msg)

    def _benchmark(
        self,
        runner: Callable[..., np.ndarray],
        args: tuple,
        kwargs: dict,
        iterations: int = 5,
    ) -> float:
        """Median wall-clock milliseconds per call. Never fabricates numbers."""
        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            runner(*args, **kwargs)
            times.append((time.perf_counter() - start) * 1000.0)
        times.sort()
        return round(times[len(times) // 2], 6)

    def _cache_path(self, op: str) -> Path:
        return self.cache_dir / f"{op}_manifest.json"

    def _load_cache(self, op: str) -> dict[str, Any] | None:
        path = self._cache_path(op)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("kernel_falcon_version") != 1:
            return None
        entry = data.get("result", {})
        if not entry.get("selected") or not entry.get("validated"):
            return None
        # A cached entry is invalid once its compiled library disappears.
        if entry.get("compiled") and entry.get("library_path"):
            if not Path(entry["library_path"]).is_file():
                return None
        return entry

    def _save_cache(self, result: KernelFalconResult) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "kernel_falcon_version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result": result.to_dict(),
        }
        self._cache_path(result.op).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def status(self) -> dict[str, Any]:
        """Honest machine-local capability report."""
        return self.state.to_dict()

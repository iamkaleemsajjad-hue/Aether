"""
Native CPU kernels — real C++ source, compiled to a shared library at runtime.

Unlike :mod:`aether.targets.templates`, which holds descriptive template strings,
this module emits compilable C++, invokes an installed host compiler, loads the
resulting shared library through :mod:`ctypes`, and calls it on numpy buffers.

Design notes
------------
* **Graceful degradation.** Every kernel has a numpy reference implementation.
  When no compiler is present the reference runs instead, so importing and using
  this module never depends on a toolchain being installed.
* **Numerical parity.** ``-ffast-math`` is deliberately *not* used: kernels must
  agree with their numpy references to within float32 rounding, which is what the
  test suite asserts.
* **Build caching.** The library is keyed by a hash of the source and flags, so
  repeated runs reuse an existing binary instead of recompiling.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sysconfig
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CompilerToolchain",
    "NativeKernelError",
    "detect_toolchain",
    "NativeCPUKernels",
    "get_native_kernels",
    "CPU_KERNEL_SOURCE",
]


class NativeKernelError(RuntimeError):
    """Raised when kernel compilation or loading fails."""


@dataclass(frozen=True)
class CompilerToolchain:
    """A detected host C++ compiler and the flags needed to build a shared library."""

    name: str
    executable: str
    #: Optimisation and codegen flags, excluding output/shared-library flags.
    base_flags: tuple[str, ...] = ()
    #: Flag that produces a shared library.
    shared_flag: str = "-shared"
    #: Flag that names the output file, if it takes a separate argument.
    output_flag: str = "-o"
    #: True for MSVC, whose command line differs structurally from GCC's.
    is_msvc: bool = False

    @property
    def library_suffix(self) -> str:
        """Platform-appropriate shared library extension."""
        if os.name == "nt":
            return ".dll"
        return ".dylib" if _is_macos() else ".so"

    def build_command(self, source: Path, output: Path) -> list[str]:
        """Return the full command line to compile ``source`` into ``output``."""
        if self.is_msvc:
            return [
                self.executable,
                "/nologo",
                "/O2",
                "/LD",
                str(source),
                f"/Fe:{output}",
            ]
        return [
            self.executable,
            *self.base_flags,
            self.shared_flag,
            self.output_flag,
            str(output),
            str(source),
        ]


def _is_macos() -> bool:
    return sysconfig.get_platform().startswith("macosx")


#: Compilers tried in order of preference, with their flags.
_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # -ffast-math is intentionally omitted so results stay comparable to numpy.
    ("g++", ("-O3", "-march=native", "-funroll-loops", "-fPIC", "-std=c++17")),
    ("clang++", ("-O3", "-march=native", "-funroll-loops", "-fPIC", "-std=c++17")),
    ("c++", ("-O2", "-fPIC", "-std=c++17")),
)


def _find_executable(name: str) -> str | None:
    """Locate a compiler by name, searching ``PATH`` then known install roots.

    On Windows a working toolchain frequently is not on the interpreter's ``PATH``
    (WinGet, Qt, Strawberry Perl and MSYS2 all install outside it), so falling back
    to well-known locations is the difference between compiling and silently
    dropping to the numpy path.
    """
    found = shutil.which(name)
    if found is not None:
        return found

    if os.name != "nt":
        return None

    executable = f"{name}.exe"
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path("C:/msys64"),
        Path("C:/mingw64/bin"),
        Path("C:/Qt/Tools"),
        Path("C:/Strawberry/c/bin"),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / executable
        if direct.is_file():
            return str(direct)
        # WinGet and MSYS2 nest the toolchain a few levels down; bound the walk
        # so a large tree cannot stall import.
        try:
            for candidate in sorted(root.glob(f"*/mingw64/bin/{executable}")):
                return str(candidate)
            for candidate in sorted(root.glob(f"*/*/mingw64/bin/{executable}")):
                return str(candidate)
            for candidate in sorted(root.glob(f"*/bin/{executable}")):
                return str(candidate)
        except OSError:
            continue
    return None


def detect_toolchain() -> CompilerToolchain | None:
    """Locate a usable host C++ compiler.

    Returns:
        The first working toolchain found, or None when the host has no compiler.
        A found executable is verified by actually compiling a trivial program,
        since a name on ``PATH`` does not guarantee a working install.
    """
    for name, flags in _CANDIDATES:
        executable = _find_executable(name)
        if executable is None:
            continue
        toolchain = CompilerToolchain(name=name, executable=executable, base_flags=flags)
        if _verify_toolchain(toolchain):
            logger.debug("Using C++ toolchain: %s (%s)", name, executable)
            return toolchain
        # Retry without -march=native, which some cross-compilers reject.
        fallback = CompilerToolchain(
            name=name,
            executable=executable,
            base_flags=tuple(f for f in flags if f != "-march=native"),
        )
        if _verify_toolchain(fallback):
            logger.debug("Using C++ toolchain: %s (without -march=native)", name)
            return fallback

    if os.name == "nt":
        executable = _find_executable("cl")
        if executable is not None:
            msvc = CompilerToolchain(name="msvc", executable=executable, is_msvc=True)
            if _verify_toolchain(msvc):
                logger.debug("Using MSVC toolchain")
                return msvc

    logger.info("No host C++ compiler found; native CPU kernels will use numpy references")
    return None


def _verify_toolchain(toolchain: CompilerToolchain) -> bool:
    """Compile a trivial translation unit to confirm the toolchain works."""
    probe = 'extern "C" int aether_probe(void) { return 42; }\n'
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.cpp"
        source.write_text(probe, encoding="utf-8")
        output = Path(tmp) / f"probe{toolchain.library_suffix}"
        try:
            result = subprocess.run(  # noqa: S603  # nosec B603 - argv built from a fixed candidate list
                toolchain.build_command(source, output),
                capture_output=True,
                timeout=60,
                cwd=tmp,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and output.exists()


# ── Kernel source ─────────────────────────────────────────────────────────────

#: Real C++ implementations of the CPU-side LLM primitives. Written to be
#: auto-vectorizable by the compiler rather than using intrinsics directly, so the
#: same source builds on x86 (AVX-512) and ARM (NEON) targets.
CPU_KERNEL_SOURCE = r"""
// Aether native CPU kernels.
// Compiled at runtime by aether.kernels.native_cpu.

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#define AETHER_EXPORT extern "C" __declspec(dllexport)
#else
#define AETHER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ── RMSNorm ───────────────────────────────────────────────────────────────────
// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i], per row.
AETHER_EXPORT void aether_rmsnorm(
    const float* __restrict x,
    const float* __restrict weight,
    float* __restrict out,
    int rows, int cols, float eps)
{
    for (int r = 0; r < rows; ++r) {
        const float* row = x + (size_t)r * cols;
        float* dst = out + (size_t)r * cols;
        // Accumulate in double: long rows lose precision badly in float32.
        double sum_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum_sq += (double)row[c] * (double)row[c];
        }
        const float inv = 1.0f / std::sqrt((float)(sum_sq / (double)cols) + eps);
        for (int c = 0; c < cols; ++c) {
            dst[c] = row[c] * inv * weight[c];
        }
    }
}

// ── SiLU / Swish ──────────────────────────────────────────────────────────────
// out[i] = x[i] * sigmoid(x[i])
AETHER_EXPORT void aether_silu(
    const float* __restrict x, float* __restrict out, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// ── SwiGLU ────────────────────────────────────────────────────────────────────
// out[i] = silu(gate[i]) * up[i], the FFN activation used by Llama/Qwen/Mistral.
AETHER_EXPORT void aether_swiglu(
    const float* __restrict gate,
    const float* __restrict up,
    float* __restrict out,
    int64_t n)
{
    for (int64_t i = 0; i < n; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────
// Row-wise, max-shifted for numerical stability.
AETHER_EXPORT void aether_softmax(
    const float* __restrict x, float* __restrict out, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        const float* row = x + (size_t)r * cols;
        float* dst = out + (size_t)r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; ++c) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float e = std::exp(row[c] - max_val);
            dst[c] = e;
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (int c = 0; c < cols; ++c) {
            dst[c] *= inv;
        }
    }
}

// ── SGEMM ─────────────────────────────────────────────────────────────────────
// C = A * B, with A (M x K) row-major and B (K x N) row-major.
// Cache-blocked and written so the inner loop vectorizes over N.
AETHER_EXPORT void aether_sgemm(
    const float* __restrict a,
    const float* __restrict b,
    float* __restrict c,
    int m, int n, int k)
{
    const int BLOCK_M = 64;
    const int BLOCK_N = 256;
    const int BLOCK_K = 64;

    std::memset(c, 0, (size_t)m * (size_t)n * sizeof(float));

    for (int i0 = 0; i0 < m; i0 += BLOCK_M) {
        const int i_max = (i0 + BLOCK_M < m) ? i0 + BLOCK_M : m;
        for (int k0 = 0; k0 < k; k0 += BLOCK_K) {
            const int k_max = (k0 + BLOCK_K < k) ? k0 + BLOCK_K : k;
            for (int j0 = 0; j0 < n; j0 += BLOCK_N) {
                const int j_max = (j0 + BLOCK_N < n) ? j0 + BLOCK_N : n;
                for (int i = i0; i < i_max; ++i) {
                    float* crow = c + (size_t)i * n;
                    for (int kk = k0; kk < k_max; ++kk) {
                        // No zero-skip branch here: measured on 2:4-pruned
                        // weights it was ~45% *slower* than the straight loop,
                        // because scattered zeros mispredict more than they save.
                        const float aik = a[(size_t)i * k + kk];
                        const float* brow = b + (size_t)kk * n;
                        for (int j = j0; j < j_max; ++j) {
                            crow[j] += aik * brow[j];
                        }
                    }
                }
            }
        }
    }
}

// ── RoPE ──────────────────────────────────────────────────────────────────────
// Rotary position embedding, applied in-place over (seq, heads, head_dim).
// Pairs element d with d + half_dim, matching the HuggingFace layout.
AETHER_EXPORT void aether_rope(
    float* __restrict x,
    const float* __restrict cos_table,
    const float* __restrict sin_table,
    int seq_len, int num_heads, int head_dim, int position_offset)
{
    const int half = head_dim / 2;
    for (int s = 0; s < seq_len; ++s) {
        const int pos = s + position_offset;
        const float* cos_row = cos_table + (size_t)pos * half;
        const float* sin_row = sin_table + (size_t)pos * half;
        for (int h = 0; h < num_heads; ++h) {
            float* vec = x + ((size_t)s * num_heads + h) * head_dim;
            for (int d = 0; d < half; ++d) {
                const float lo = vec[d];
                const float hi = vec[d + half];
                vec[d]        = lo * cos_row[d] - hi * sin_row[d];
                vec[d + half] = hi * cos_row[d] + lo * sin_row[d];
            }
        }
    }
}

// ── Dequantize: symmetric block-scaled integer ────────────────────────────────
// codes are stored biased by `bias`; out = (code - bias) * scale[block].
AETHER_EXPORT void aether_dequantize_symmetric(
    const uint8_t* __restrict codes,
    const float* __restrict scales,
    float* __restrict out,
    int64_t n, int block_size, float bias)
{
    for (int64_t i = 0; i < n; ++i) {
        const int64_t block = i / block_size;
        out[i] = ((float)codes[i] - bias) * scales[block];
    }
}

// ── Dequantize: affine block-scaled integer ───────────────────────────────────
// out = (code - zero_point[block]) * scale[block].
AETHER_EXPORT void aether_dequantize_affine(
    const uint8_t* __restrict codes,
    const float* __restrict scales,
    const int16_t* __restrict zero_points,
    float* __restrict out,
    int64_t n, int block_size)
{
    for (int64_t i = 0; i < n; ++i) {
        const int64_t block = i / block_size;
        out[i] = ((float)codes[i] - (float)zero_points[block]) * scales[block];
    }
}

// ── Fused dequantize + GEMV ───────────────────────────────────────────────────
// y = W_dequant * x, for W (rows x cols) quantized affine per block along cols.
// Fusing avoids materialising the dequantized weight matrix, which is the whole
// point of quantized inference at decode time.
AETHER_EXPORT void aether_qgemv_affine(
    const uint8_t* __restrict codes,
    const float* __restrict scales,
    const int16_t* __restrict zero_points,
    const float* __restrict x,
    float* __restrict y,
    int rows, int cols, int block_size)
{
    for (int r = 0; r < rows; ++r) {
        const size_t base = (size_t)r * cols;
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) {
            const size_t idx = base + c;
            const int64_t block = (int64_t)(idx / block_size);
            const float w = ((float)codes[idx] - (float)zero_points[block]) * scales[block];
            acc += (double)w * (double)x[c];
        }
        y[r] = (float)acc;
    }
}

// ── Argmax ────────────────────────────────────────────────────────────────────
// Greedy token selection over a logit vector.
AETHER_EXPORT int64_t aether_argmax(const float* __restrict x, int64_t n)
{
    if (n <= 0) return -1;
    int64_t best = 0;
    float best_val = x[0];
    for (int64_t i = 1; i < n; ++i) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}
"""


# ── Compilation and dispatch ──────────────────────────────────────────────────

_F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
_I16 = np.ctypeslib.ndpointer(dtype=np.int16, flags="C_CONTIGUOUS")

#: ctypes signatures for every exported kernel.
_SIGNATURES: dict[str, tuple[type | object, list[type | object]]] = {
    "aether_rmsnorm": (None, [_F32, _F32, _F32, ctypes.c_int, ctypes.c_int, ctypes.c_float]),
    "aether_silu": (None, [_F32, _F32, ctypes.c_int64]),
    "aether_swiglu": (None, [_F32, _F32, _F32, ctypes.c_int64]),
    "aether_softmax": (None, [_F32, _F32, ctypes.c_int, ctypes.c_int]),
    "aether_sgemm": (None, [_F32, _F32, _F32, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
    "aether_rope": (
        None,
        [_F32, _F32, _F32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ),
    "aether_dequantize_symmetric": (
        None,
        [_U8, _F32, _F32, ctypes.c_int64, ctypes.c_int, ctypes.c_float],
    ),
    "aether_dequantize_affine": (
        None,
        [_U8, _F32, _I16, _F32, ctypes.c_int64, ctypes.c_int],
    ),
    "aether_qgemv_affine": (
        None,
        [_U8, _F32, _I16, _F32, _F32, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ),
    "aether_argmax": (ctypes.c_int64, [_F32, ctypes.c_int64]),
}


def _cache_dir() -> Path:
    """Directory holding compiled kernel libraries."""
    override = os.environ.get("AETHER_KERNEL_CACHE")
    base = Path(override) if override else Path(tempfile.gettempdir()) / "aether_kernels"
    base.mkdir(parents=True, exist_ok=True)
    return base


@dataclass
class NativeCPUKernels:
    """Compiles and dispatches the native CPU kernels.

    Every method falls back to a numpy reference when no compiler is available, so
    behaviour is identical with or without a toolchain — only speed differs. Check
    :attr:`is_native` to see which path is active.
    """

    toolchain: CompilerToolchain | None = None
    #: Populated after a successful build.
    library_path: Path | None = None
    _lib: ctypes.CDLL | None = field(default=None, repr=False)
    _build_error: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.toolchain is None:
            self.toolchain = detect_toolchain()

    # ── Build ────────────────────────────────────────────────────────────────

    @property
    def is_native(self) -> bool:
        """True when compiled kernels are loaded and in use."""
        return self._lib is not None

    @property
    def build_error(self) -> str | None:
        """Why compilation failed, when it did."""
        return self._build_error

    def ensure_compiled(self) -> bool:
        """Compile and load the kernel library, reusing a cached build if present.

        Returns:
            True when native kernels are available, False when the numpy
            reference path will be used instead.
        """
        if self._lib is not None:
            return True
        if self.toolchain is None:
            self._build_error = "no host C++ compiler found"
            return False

        digest = hashlib.sha256(
            (CPU_KERNEL_SOURCE + self.toolchain.name + " ".join(self.toolchain.base_flags)).encode()
        ).hexdigest()[:16]
        library = _cache_dir() / f"aether_cpu_{digest}{self.toolchain.library_suffix}"

        if not library.exists() and not self._compile(library):
            return False

        try:
            self._lib = ctypes.CDLL(str(library))
        except OSError as exc:
            self._build_error = f"could not load {library}: {exc}"
            logger.warning("Failed to load native kernels: %s", exc)
            return False

        self._bind_signatures()
        self.library_path = library
        logger.info("Native CPU kernels loaded from %s", library.name)
        return True

    def load_library(self, library_path: str | Path, expected_sha256: str | None = None) -> bool:
        """Load a verified native library carried by an AEG artifact.

        The AEG loader performs manifest validation before calling this method;
        the optional digest check is retained here as a second boundary for
        callers that use the kernel provider directly.  The library is bound
        through the same signatures as the compiler cache path, so packaged
        kernels are actually executed by the CPU engine rather than merely
        listed in metadata.
        """
        path = Path(library_path).resolve()
        if not path.is_file():
            self._build_error = f"native kernel library does not exist: {path}"
            return False
        if expected_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = expected_sha256.removeprefix("sha256:")
            if digest.lower() != expected.lower():
                self._build_error = f"native kernel library hash mismatch for {path}"
                return False
        try:
            library = ctypes.CDLL(str(path))
            previous = self._lib
            self._lib = library
            try:
                self._bind_signatures()
            except Exception:
                self._lib = previous
                raise
        except (OSError, NativeKernelError) as exc:
            self._build_error = f"could not load packaged native kernels from {path}: {exc}"
            return False
        self.library_path = path
        self._build_error = None
        logger.info("Loaded packaged native CPU kernels from %s", path.name)
        return True

    def _compile(self, output: Path) -> bool:
        """Compile the kernel source into ``output``."""
        assert self.toolchain is not None
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "aether_cpu_kernels.cpp"
            source.write_text(CPU_KERNEL_SOURCE, encoding="utf-8")
            staged = Path(tmp) / output.name
            command = self.toolchain.build_command(source, staged)
            try:
                result = subprocess.run(  # noqa: S603  # nosec B603 - argv from detected toolchain
                    command, capture_output=True, timeout=180, cwd=tmp, check=False, text=True
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self._build_error = f"compiler invocation failed: {exc}"
                logger.warning("Kernel compilation failed: %s", exc)
                return False

            if result.returncode != 0 or not staged.exists():
                self._build_error = (result.stderr or result.stdout or "unknown error").strip()[:500]
                logger.warning("Kernel compilation failed: %s", self._build_error)
                return False

            # Move into the cache only once the build succeeded, so a failed or
            # concurrent build never leaves a partial library behind.
            try:
                shutil.move(str(staged), str(output))
            except OSError:
                # Another process may have won the race; that is fine.
                if not output.exists():
                    self._build_error = "could not install compiled library into cache"
                    return False
        return True

    def _bind_signatures(self) -> None:
        """Attach argument and return types to every exported symbol."""
        assert self._lib is not None
        for symbol, (restype, argtypes) in _SIGNATURES.items():
            try:
                function = getattr(self._lib, symbol)
            except AttributeError:
                msg = f"compiled library is missing symbol '{symbol}'"
                raise NativeKernelError(msg) from None
            function.restype = restype
            function.argtypes = argtypes

    # ── Kernels ──────────────────────────────────────────────────────────────

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Root-mean-square layer normalisation over the last axis."""
        arr = np.ascontiguousarray(x, dtype=np.float32)
        w = np.ascontiguousarray(weight, dtype=np.float32)
        rows = int(arr.size // arr.shape[-1])
        cols = int(arr.shape[-1])
        if w.size != cols:
            msg = f"weight has {w.size} elements but last axis is {cols}"
            raise ValueError(msg)
        if not self.ensure_compiled():
            variance = np.mean(arr.astype(np.float64) ** 2, axis=-1, keepdims=True)
            return (arr / np.sqrt(variance + eps) * w).astype(np.float32)
        out = np.empty_like(arr)
        self._lib.aether_rmsnorm(arr.reshape(-1), w, out.reshape(-1), rows, cols, eps)  # type: ignore[union-attr]
        return out

    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU / Swish activation."""
        arr = np.ascontiguousarray(x, dtype=np.float32)
        if not self.ensure_compiled():
            return (arr / (1.0 + np.exp(-arr.astype(np.float64)))).astype(np.float32)
        out = np.empty_like(arr)
        self._lib.aether_silu(arr.reshape(-1), out.reshape(-1), arr.size)  # type: ignore[union-attr]
        return out

    def swiglu(self, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        """SwiGLU FFN activation: ``silu(gate) * up``."""
        g = np.ascontiguousarray(gate, dtype=np.float32)
        u = np.ascontiguousarray(up, dtype=np.float32)
        if g.shape != u.shape:
            msg = f"gate shape {g.shape} does not match up shape {u.shape}"
            raise ValueError(msg)
        if not self.ensure_compiled():
            activated = g / (1.0 + np.exp(-g.astype(np.float64)))
            return (activated * u).astype(np.float32)
        out = np.empty_like(g)
        self._lib.aether_swiglu(g.reshape(-1), u.reshape(-1), out.reshape(-1), g.size)  # type: ignore[union-attr]
        return out

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax over the last axis."""
        arr = np.ascontiguousarray(x, dtype=np.float32)
        rows = int(arr.size // arr.shape[-1])
        cols = int(arr.shape[-1])
        if not self.ensure_compiled():
            shifted = arr - arr.max(axis=-1, keepdims=True)
            exponentiated = np.exp(shifted)
            return (exponentiated / exponentiated.sum(axis=-1, keepdims=True)).astype(np.float32)
        out = np.empty_like(arr)
        self._lib.aether_softmax(arr.reshape(-1), out.reshape(-1), rows, cols)  # type: ignore[union-attr]
        return out

    def sgemm(self, a: np.ndarray, b: np.ndarray, force_native: bool = False) -> np.ndarray:
        """Single-precision matrix multiply ``a @ b``.

        GEMM is delegated to numpy, which dispatches to a tuned BLAS whose packed
        microkernels beat a portable C++ triple loop by a wide margin (measured
        0.63x for the hand-written kernel, on both dense and 2:4-pruned weights).
        The native path is retained for environments without a BLAS-backed numpy
        and for kernel-emission testing, and is selectable via ``force_native``.

        Args:
            a: Left matrix, shape ``(M, K)``.
            b: Right matrix, shape ``(K, N)``.
            force_native: Use the compiled kernel even when BLAS is available.

        Returns:
            The product as a float32 array of shape ``(M, N)``.
        """
        lhs = np.ascontiguousarray(a, dtype=np.float32)
        rhs = np.ascontiguousarray(b, dtype=np.float32)
        if lhs.ndim != 2 or rhs.ndim != 2:
            msg = f"sgemm requires 2-D inputs, got {lhs.ndim}-D and {rhs.ndim}-D"
            raise ValueError(msg)
        if lhs.shape[1] != rhs.shape[0]:
            msg = f"shapes {lhs.shape} and {rhs.shape} are not aligned for matmul"
            raise ValueError(msg)
        if not force_native or not self.ensure_compiled():
            return (lhs @ rhs).astype(np.float32)
        m, k = lhs.shape
        n = rhs.shape[1]
        out = np.empty((m, n), dtype=np.float32)
        self._lib.aether_sgemm(lhs.reshape(-1), rhs.reshape(-1), out.reshape(-1), m, n, k)  # type: ignore[union-attr]
        return out

    def rope(
        self,
        x: np.ndarray,
        cos_table: np.ndarray,
        sin_table: np.ndarray,
        position_offset: int = 0,
    ) -> np.ndarray:
        """Apply rotary position embeddings to ``(seq, heads, head_dim)``."""
        arr = np.ascontiguousarray(x, dtype=np.float32)
        if arr.ndim != 3:
            msg = f"rope expects (seq, heads, head_dim), got shape {arr.shape}"
            raise ValueError(msg)
        seq_len, num_heads, head_dim = arr.shape
        if head_dim % 2 != 0:
            msg = f"head_dim must be even for RoPE, got {head_dim}"
            raise ValueError(msg)
        cos = np.ascontiguousarray(cos_table, dtype=np.float32)
        sin = np.ascontiguousarray(sin_table, dtype=np.float32)
        half = head_dim // 2
        if not self.ensure_compiled():
            out = arr.copy()
            c = cos[position_offset : position_offset + seq_len, :half][:, None, :]
            s = sin[position_offset : position_offset + seq_len, :half][:, None, :]
            lo, hi = arr[..., :half], arr[..., half:]
            out[..., :half] = lo * c - hi * s
            out[..., half:] = hi * c + lo * s
            return out
        out = arr.copy()
        self._lib.aether_rope(  # type: ignore[union-attr]
            out.reshape(-1), cos.reshape(-1), sin.reshape(-1),
            seq_len, num_heads, head_dim, position_offset,
        )
        return out

    def argmax(self, logits: np.ndarray) -> int:
        """Index of the largest logit — greedy token selection."""
        arr = np.ascontiguousarray(logits, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            msg = "argmax on an empty array"
            raise ValueError(msg)
        if not self.ensure_compiled():
            return int(np.argmax(arr))
        return int(self._lib.aether_argmax(arr, arr.size))  # type: ignore[union-attr]

    def dequantize(self, tensor: object) -> np.ndarray:
        """Dequantize a :class:`~aether.quantization.formats.QuantizedTensor`.

        Falls back to the pure-Python codec path when native kernels are
        unavailable or the format is not one of the block-integer families.
        """
        from aether.quantization.codecs import AffineIntCodec, SymmetricIntCodec, get_codec
        from aether.quantization.formats import dequantize_tensor

        codec = get_codec(tensor.precision)  # type: ignore[attr-defined]
        native_eligible = (
            self.ensure_compiled()
            and isinstance(codec, (AffineIntCodec, SymmetricIntCodec))
            and getattr(tensor, "packed", False) is False
        )
        if not native_eligible:
            return dequantize_tensor(tensor)  # type: ignore[arg-type]

        codes = np.ascontiguousarray(tensor.data, dtype=np.uint8).reshape(-1)  # type: ignore[attr-defined]
        scales = np.ascontiguousarray(tensor.scales, dtype=np.float32)  # type: ignore[attr-defined]
        total = int(tensor.num_elements)  # type: ignore[attr-defined]
        out = np.empty(total, dtype=np.float32)
        block_size = int(tensor.block_size)  # type: ignore[attr-defined]

        if isinstance(codec, AffineIntCodec):
            zero_points = np.ascontiguousarray(tensor.zero_points, dtype=np.int16)  # type: ignore[attr-defined]
            self._lib.aether_dequantize_affine(  # type: ignore[union-attr]
                codes, scales, zero_points, out, total, block_size
            )
        else:
            bias = float(1 << (codec.bits - 1))
            self._lib.aether_dequantize_symmetric(  # type: ignore[union-attr]
                codes, scales, out, total, block_size, bias
            )
        return out.reshape(tensor.shape)  # type: ignore[attr-defined]

    def available_kernels(self) -> list[str]:
        """Names of the exported native kernels."""
        return sorted(_SIGNATURES)

    def __repr__(self) -> str:
        backend = f"native/{self.toolchain.name}" if self.is_native else "numpy-reference"
        return f"NativeCPUKernels({backend})"


#: Process-wide instance so the library is compiled at most once per run.
_INSTANCE: NativeCPUKernels | None = None


def get_native_kernels() -> NativeCPUKernels:
    """Return the shared :class:`NativeCPUKernels` instance."""
    global _INSTANCE  # noqa: PLW0603 - intentional process-wide kernel cache
    if _INSTANCE is None:
        _INSTANCE = NativeCPUKernels()
    return _INSTANCE

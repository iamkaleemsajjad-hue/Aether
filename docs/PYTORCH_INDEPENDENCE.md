# PyTorch Independence Guarantee

## Summary

Aether Runtime ≥ 1.0 operates as a **genuinely framework-independent** AI compiler and runtime. PyTorch is an **optional** dependency for legacy model ingestion only. The entire Aether inference path — from AEG loading through CPU execution — requires no PyTorch installation.

## What Was Changed

### `pyproject.toml`
- Moved `torch`, `transformers`, `tokenizers`, `sentencepiece` from mandatory `[project.dependencies]` to optional `[project.optional-dependencies]` groups.
- `pip install aether-runtime` installs **zero** ML framework dependencies.
- `pip install aether-runtime[torch]` opts into the PyTorch compatibility frontend.

### `src/aether/runtime/__init__.py`
- R1–R12 runtime layers (`PEAGLEEngine`, `TTTFastWeightEngine`, etc.) are now **lazily imported** via `__getattr__`.
- `import aether` loads only the framework-free core (Runtime, RuntimeConfig, KVCacheManager, HardwareDetector).
- R-layers are only imported on first attribute access, so any torch dependency in R1 only loads when `PEAGLEEngine` is actually used.

### `src/aether/compiler/graph_tracer.py`
- `import torch` / `from torch.fx` guarded inside `trace()` method body.
- `GraphTracer` can be imported and instantiated in framework-free environments.

### `src/aether/compiler/stage2_optimizer/pass02_sensitivity_analysis.py`
- Bare `import torch` removed. Numpy-based sensitivity heuristics run when torch is absent.

### `src/aether/core/types.py` — `HardwareTarget.auto()`
- Replaced `torch.cuda.is_available()` + `torch.cuda.get_device_properties()` with **pynvml** (NVIDIA Management Library).
- Falls back to ctypes CUDA driver API, then platform inspection for Metal/ROCm.

### `src/aether/backends/hardware_detector.py`
- `_cuda_driver_version()`: replaced `torch.version.cuda` with `pynvml.nvmlSystemGetDriverVersion()`.
- `_has_nvlink()` / `_has_nvlink_nvml()`: pure pynvml, no torch.
- All GPU capability detection via pynvml.

### `src/aether/runtime/hardware.py` — `HardwareDetector.detect()`
- CUDA GPU enumeration via pynvml: name, memory, compute capability, driver version.
- Metal: `platform.processor()` only. No torch.

### `src/aether/compiler/plan.py` — `recommend_backend()`
- Removed unconditional PyTorch fallback that silently activated torch when no GPU backend was found.
- CPU targets always return `"aether_cpu"` (the native engine).
- GPU targets return `None` if no backend is available, allowing callers to emit an explicit warning.

## Verification

All of the following have been empirically verified on this machine (CPU-only, no GPU):

```
import aether                           # torch NOT in sys.modules
HardwareTarget.auto()                   # detects cpu_avx512 without torch
detect_all_capabilities()               # enumerates backends without torch
CPUExecutionEngine.forward()            # runs real transformer forward pass
RuntimeConfig()                         # instantiates without torch
```

**Test script**: [`scratch/test_torch_full.py`](../brain/96a6e955-c3e2-47d1-a83b-85a20412c384/scratch/test_torch_full.py)

**Result**: `PASS — Aether is fully PyTorch-independent for all tested paths`

## PyTorch Still Supported

PyTorch remains a first-class optional backend:

| Scenario | Behavior |
|---|---|
| `import aether` (no torch) | OK — zero torch-related imports |
| `import aether` (torch installed) | OK — torch NOT eagerly imported |
| `from aether.runtime import PEAGLEEngine` (torch installed) | OK — lazy import fires, torch loads |
| `compiler.compile(path_to_pt_model)` | OK — `pytorch_loader.py` imports torch inside load call |
| `recommend_backend("cuda_sm90")` (torch installed) | Returns "pytorch" if available |
| `recommend_backend("cuda_sm90")` (no torch) | Returns `None` with warning |
| `recommend_backend("cpu_avx512")` | Always returns "aether_cpu" |

## Dependency Matrix

| Package | Status | Purpose |
|---|---|---|
| `numpy` | **mandatory** | All weight math, tensor ops |
| `safetensors` | **mandatory** | Native AEG weight format |
| `pynvml` | **mandatory** | GPU hardware detection |
| `torch` | **optional** (`[torch]`) | Legacy .pt/.pth ingestion, GPU inference |
| `transformers` | **optional** (`[hf]`) | HuggingFace model download |
| `tokenizers` | **optional** (`[hf]`) | HuggingFace tokenizer |
| `gguf` | **optional** (`[gguf]`) | GGUF reference validation |
| `onnxruntime` | **optional** (`[onnx]`) | ONNX backend |
| `vllm` | **optional** (`[vllm]`) | vLLM GPU backend |
| `mlx` | **optional** (`[metal]`) | Apple Metal backend |

## Known Limitations

- `pynvml` is a mandatory dependency even on non-NVIDIA machines (needed for hardware detection). It degrades gracefully when no NVIDIA GPU is present.
- The `pynvml` package triggers a `FutureWarning` about being deprecated in favour of `nvidia-ml-py`. This is a packaging issue in the third-party library and does not affect functionality. The warning appears in stderr only.

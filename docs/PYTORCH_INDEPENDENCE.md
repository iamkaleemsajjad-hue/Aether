# PyTorch Independence Guarantee

## Summary

Aether Runtime ≥ 1.0 operates as a **genuinely framework-independent** AI compiler and runtime. PyTorch is an **optional** dependency for legacy model ingestion only. The entire Aether inference path — from AEG loading through CPU execution — requires no PyTorch installation.

## What Was Changed

### `pyproject.toml`
- Kept `torch`, `transformers`, and `sentencepiece` in optional
  `[project.optional-dependencies]` groups; the small `tokenizers` package is
  a mandatory core dependency because packaged AEGs must tokenize without a
  model framework.
- `pip install aether-runtime` installs **zero** tensor/ML model frameworks.
- `pip install aether-runtime[pytorch]` opts into the PyTorch compatibility frontend.

### `src/aether/runtime/__init__.py`
- R1–R12 runtime layers (`PEAGLEEngine`, `TTTFastWeightEngine`, etc.) are now **lazily imported** via `__getattr__`.
- `import aether` loads only the framework-free core (Runtime, RuntimeConfig, KVCacheManager, HardwareDetector).
- R-layers are only imported on first attribute access, so any torch dependency in R1 only loads when `PEAGLEEngine` is actually used.

### `src/aether/compiler/stage1_ingestion/graph_tracer.py` *(removed)*
- The dead, torch.fx-placeholder conversion module was deleted; the live
  PyTorch ingestion path is `pytorch_loader.py`, which guards every torch
  import inside its load methods and fails closed without the framework.

### `src/aether/compiler/stage2_optimizer/pass02_sensitivity_analysis.py` *(removed)*
- The unregistered duplicate pass (random-number "sensitivity") was deleted.
  The live Pass 2 in `optimizer.py` + `calibration/sensitivity.py` is
  numpy-only and runs with or without torch.

### SafeTensors BF16 ingestion
- BF16 payloads are decoded directly from the SafeTensors header/data layout
  into FP32 when NumPy has no native BF16 dtype.
- Local SafeTensors compilation therefore has no PyTorch dtype fallback.

### `src/aether/backends/native_cpu_backend.py`
- The packaged-AEG backend implements the base backend contract directly; it
  does not subclass or construct the PyTorch backend while loading an AEG.

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

**Tests**: `tests/unit/test_adversarial.py` and
`tests/unit/test_framework_free_safetensors.py` block PyTorch and exercise the
native AEG and BF16 SafeTensors paths.

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
| `pynvml` / `nvidia-ml-py` | optional | NVIDIA telemetry and topology; CUDA driver API is the fallback |
| `torch` | **optional** (`[torch]`) | Legacy .pt/.pth ingestion, GPU inference |
| `transformers` | **optional** (`[transformers-frontend]`) | HuggingFace model download |
| `tokenizers` | mandatory core dependency | Packaged AEG tokenizer execution |
| `gguf` | **optional** (`[gguf]`) | GGUF reference validation |
| `onnxruntime` | **optional** (`[onnx]`) | ONNX backend |
| `vllm` | **optional** (`[vllm]`) | vLLM GPU backend |
| `mlx` | **optional** (`[metal]`) | Apple Metal backend |

## Known Limitations

- `pynvml`/`nvidia-ml-py` is optional, including on non-NVIDIA machines; CUDA
  driver API and vendor tools are used when available.
- The `pynvml` package triggers a `FutureWarning` about being deprecated in favour of `nvidia-ml-py`. This is a packaging issue in the third-party library and does not affect functionality. The warning appears in stderr only.

# Kaggle Benchmark Runbook — Aether Runtime vs All Engines

Complete, cell-by-cell instructions for a **Kaggle GPU notebook (2× Tesla T4)**.

> **Aether runs from the cloned source** — not from a pip release. Every
> benchmark number reflects exactly the code on `main`.
>
> **No kernel restart is required** after installing `onnxruntime-gpu`. Both
> the availability probe and the measurement workers run in subprocesses that
> always see the current disk state, so a stale in-process `.so` in the
> notebook kernel cannot affect any result.

---

## Notebook settings  *(before opening the first cell)*

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet access** | `On` |
| **Persistence** | `Files only` (reuses compiled artifacts across runs) |

---

## Cell 1 — Clone, install Aether from source, and install ORT GPU

Run this as the **very first cell** before any Python import of `onnxruntime`.
Because the kernel has not yet imported `onnxruntime` at startup, removing
the CPU build and installing the GPU build here means no restart is ever needed.

```python
# 1a. Clone the repo
!git clone https://github.com/iamkaleemsajjad-hue/Aether.git /kaggle/working/aether
%cd /kaggle/working/aether

# 1b. Install Aether from source (NOT the pip release — uses the cloned code).
#     [pytorch] pulls in torch + Aether CUDA kernels.
!pip install -q -e ".[pytorch]"

# 1c. Install the benchmark measurement stack.
#     Does NOT install competing engines — absent ones are reported with a reason.
!pip install -q -r benchmark/requirements.txt

# ── ONNX Runtime GPU build ────────────────────────────────────────────────
# Three steps, all in this cell, before any Python code imports onnxruntime.
# Doing it here (before any import) means NO kernel restart is needed.

# Step 1: Remove the pre-installed CPU build.
#   Kaggle ships onnxruntime (CPU). pip marks the dep satisfied and skips the
#   GPU wheel unless the CPU build is removed first.
!pip uninstall -q -y onnxruntime

# Step 2: Install the GPU build.
#   onnxruntime-gpu >=1.18 registers its metadata as "onnxruntime", so pip
#   treats subsequent "onnxruntime" requirements as already satisfied.
!pip install -q "onnxruntime-gpu>=1.18.0"

# Step 3: Install optimum WITH the [onnxruntime] extra.
#   The [onnxruntime] extra is REQUIRED — plain "optimum" does not include the
#   optimum.onnxruntime subpackage. pip sees onnxruntime as satisfied by the
#   GPU build above and does NOT install the CPU build.
!pip install -q "optimum[onnxruntime]>=1.20.0"
```

---

## Cell 2 — Install competing engines  *(order matters)*

### 2a. vLLM  *(install before the transformers pin)*

```python
%cd /kaggle/working/aether

# vLLM pins torch and transformers. Install it before the transformers pin
# so that the pin below (Cell 2b) wins.  ~5 min download.
!pip install -q vllm
```

### 2b. transformers version pin  *(mandatory for ONNX export)*

```python
%cd /kaggle/working/aether

# MANDATORY — without this pin the ONNX engine fails at load.
# Kaggle ships transformers 5.x. optimum's ONNX exporter needs transformers<4.58
# and imports get_parameter_dtype, which 5.x removed.
# Pin BEFORE any engine loads — not afterwards.
!pip install -q "transformers==4.57.1"
```

### 2c. DeepSpeed  *(optional)*

```python
# Kernel injection for architectures that have a policy.
# Reports NOT_SUPPORTED (with reason) where it has none.
!pip install -q deepspeed
```

### 2d. llama.cpp  *(optional — adds ~20 min for GGUF conversion)*

```python
# llama.cpp executes GGUF, not the HF checkpoint.
# Skip this cell to leave llama_cpp as NOT_INSTALLED.
!pip install -q llama-cpp-python gguf
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
```

> If you install llama.cpp, add `--gguf-convert-script /kaggle/working/llama.cpp/convert_hf_to_gguf.py`
> to every benchmark command in Cells 5–6.

---

## Cell 3 — Verify the GPU build is active

```python
%cd /kaggle/working/aether

import onnxruntime as ort
from importlib.metadata import version as pkg_version

print("ORT version        :", ort.__version__)
print("Available providers:", ort.get_available_providers())

assert "CUDAExecutionProvider" in ort.get_available_providers(), (
    "\n\nCUDAExecutionProvider is MISSING.\n"
    "Fix (no restart needed):\n"
    "  !pip uninstall -q -y onnxruntime\n"
    "  !pip install -q 'onnxruntime-gpu>=1.18.0'\n"
    "  !pip install -q 'optimum[onnxruntime]>=1.20.0'\n"
    "Then re-run this cell. No restart required."
)
print("OK — GPU build confirmed.")
```

---

## Cell 4 — Confirm the full environment

```python
%cd /kaggle/working/aether
from importlib.metadata import version as pkg_version

# GPU info
!nvidia-smi

# Package versions
import torch
import transformers
import onnxruntime as ort

print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
try:
    print(f"optimum      {pkg_version('optimum')}")
except Exception:
    print("optimum      (version unavailable)")
print(f"onnxruntime  {ort.__version__}")
print(f"CUDA avail   {torch.cuda.is_available()}")
print(f"GPU count    {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/1024**3:.1f} GiB  sm_{p.major}{p.minor}")

# Verify Aether is from the cloned source, not a pip release
!python -c "import aether; print('aether path:', aether.__file__)"
```

---

## Recovery cell  *(only if you are in a session where install already ran but ORT was wrong)*

If you already ran cells and ended up with `NOT_INSTALLED` for onnxruntime,
run this cell to fix without restarting:

```python
%cd /kaggle/working/aether

# Remove CPU build if still present
!pip uninstall -q -y onnxruntime

# Install GPU build
!pip install -q "onnxruntime-gpu>=1.18.0"

# Install optimum WITH the [onnxruntime] extra (required for optimum.onnxruntime subpackage)
!pip install -q "optimum[onnxruntime]>=1.20.0"

# No restart needed — probe and workers run in subprocesses that see fresh disk state.

# Verify in subprocess (this is exactly what the benchmark probe does)
import subprocess, sys
r = subprocess.run(
    [sys.executable, "-c",
     "import onnxruntime as ort; print(ort.get_available_providers())"],
    capture_output=True, text=True
)
print("Subprocess ORT providers:", r.stdout.strip())
assert "CUDAExecutionProvider" in r.stdout, "Still missing CUDAExecutionProvider"
print("OK — subprocess sees GPU build. No restart needed.")
```

---

## Cell 5 — Smoke run  *(survey engines, ~3–5 min)*

```python
%cd /kaggle/working/aether

!AETHER_ORT_REQUIRE_GPU=1 python benchmark.py \
    --smoke \
    --output-dir /kaggle/working/bench_smoke
```

Read the `engine availability` block. Every engine is `run` or `skip` with a
full reason. Fix anything unexpected before Cell 6.

**Expected availability on 2× T4:**

| Engine | Expected | Notes |
|---|---|---|
| `hf_transformers` | `run` | Reference baseline |
| `pytorch_native` | `run` | Same weights, hand-written loop |
| `onnxruntime` | `run` — CUDAExecutionProvider | GPU build required; fix message if CPU |
| `aether` | `run` | Compiled from source → `cuda_sm70` |
| `llama_cpp` | `run` or `NOT_INSTALLED` | Only if Cell 2d installed |
| `deepspeed` | `run` or `NOT_SUPPORTED` | Depends on architecture |
| `vllm` | `run` at fp16 | Refuses bf16 below sm_80 |
| `sglang` | `NOT_APPLICABLE` | Needs sm_80+; T4 is sm_75 |
| `tensorrt_llm` | `NOT_APPLICABLE` | Wheels target sm_80+ |
| `exllamav2` | `NOT_APPLICABLE` | Needs EXL2 weights |
| `mlc` | `NOT_APPLICABLE` | Needs TVM-compiled model |

---

## Cell 6 — Full benchmark run

`AETHER_ORT_REQUIRE_GPU=1` causes the run to fail fast with a clear message if
the ONNX engine is on CPU, instead of silently producing misleading results.

### Full run — all models, all matrix  *(~60–90 min on 2× T4)*

```python
%cd /kaggle/working/aether

!AETHER_ORT_REQUIRE_GPU=1 python benchmark.py \
    --output-dir /kaggle/working/benchmark_results
```

### Shorter run — fills every report section  *(~20–30 min)*

```python
%cd /kaggle/working/aether

!AETHER_ORT_REQUIRE_GPU=1 python benchmark.py \
    --batch-sizes 1,2,4,8 \
    --prompt-tokens 32,256 \
    --output-tokens 32,128 \
    --measure-iters 5 \
    --output-dir /kaggle/working/benchmark_results
```

### Resume a partial run  *(session crashed mid-way)*

```python
%cd /kaggle/working/aether

!AETHER_ORT_REQUIRE_GPU=1 python benchmark.py \
    --resume \
    --output-dir /kaggle/working/benchmark_results
```

### With llama.cpp *(if installed in Cell 2d)*

```python
%cd /kaggle/working/aether

!AETHER_ORT_REQUIRE_GPU=1 python benchmark.py \
    --batch-sizes 1,2,4,8 \
    --prompt-tokens 32,256 \
    --output-tokens 32,128 \
    --measure-iters 5 \
    --gguf-convert-script /kaggle/working/llama.cpp/convert_hf_to_gguf.py \
    --output-dir /kaggle/working/benchmark_results
```

> **For long runs** use **Save & Run All** (Kaggle batch execution) rather than
> the interactive session — the interactive kernel times out after ~9 hours.

---

## Cell 7 — Read the results

### Rendered Markdown report

```python
from IPython.display import Markdown, display
display(Markdown(open(
    "/kaggle/working/benchmark_results/reports/BENCHMARK_REPORT.md",
    encoding="utf-8"
).read()))
```

### Charts (all PNGs inline)

```python
from IPython.display import Image, display
import pathlib

for path in sorted(pathlib.Path("/kaggle/working/benchmark_results/graphs").glob("*.png")):
    print(f"\n── {path.name} ──")
    display(Image(str(path)))
```

### Raw data

```python
import pandas as pd

# Every (engine, model, batch, prompt, output) cell
df = pd.read_csv("/kaggle/working/benchmark_results/benchmark_results.csv")
print(f"Total cells: {len(df)}")
df.head()
```

```python
# Every pairwise engine comparison
pd.read_csv("/kaggle/working/benchmark_results/benchmark_comparisons.csv").head()
```

### Confirm ORT ran on GPU

```python
import json, pathlib

for path in sorted(pathlib.Path("/kaggle/working/benchmark_results/raw").glob("onnxruntime__*.json")):
    r = json.loads(path.read_text())
    d = r.get("describe", {})
    print(path.name)
    print("  execution_providers :", d.get("execution_providers", "not recorded"))
    print("  gpu_provider_active :", d.get("gpu_provider_active", "not recorded"))
    print("  status              :", r.get("status"))
    print()
```

> **`gpu_provider_active` must be `True`.** If it is `False`, the engine ran on
> CPU. Run the Recovery cell above, then re-run from Cell 5.

---

## Cell 8 — Save results

```python
# Copy to the Output tab
!cp -r /kaggle/working/benchmark_results /kaggle/working/aether/benchmark_results
```

---

## Reference: What runs on 2× T4

| Engine | Status | Notes |
|---|---|---|
| `hf_transformers` | ✅ runs | Reference baseline |
| `pytorch_native` | ✅ runs | Hand-written loop, same weights |
| `onnxruntime` | ✅ runs *(with GPU build + transformers pin)* | `onnxruntime-gpu` + `optimum[onnxruntime]` + `transformers==4.57.1` required |
| `aether` | ✅ runs | From source; compiles to `cuda_sm70` (T4 = sm_75 → sm70 tier) |
| `llama_cpp` | ✅ if installed | Executes GGUF (F16 auto-converted); CPU-only pip build |
| `deepspeed` | ⚠️ partial | `NOT_SUPPORTED` on architectures without a kernel injection policy |
| `vllm` | ✅ at fp16 | Refuses bf16 below sm_80; `--precision auto` resolves to fp16 on T4 |
| `sglang` | ❌ NOT_APPLICABLE | FlashAttention needs sm_80+; T4 is sm_75 |
| `tensorrt_llm` | ❌ NOT_APPLICABLE | Wheels require sm_80+ |
| `exllamav2` | ❌ NOT_APPLICABLE | Requires EXL2-quantized weights |
| `mlc` | ❌ NOT_APPLICABLE | Requires TVM-compiled model artifact |

NOT_APPLICABLE engines appear in the compatibility table with the deciding reason
— they are not failures and not zeros.

---

## T4-specific notes

**Precision resolves to fp16, not bf16.**
T4 is sm_75 and has no bf16 tensor cores. `--precision auto` resolves to **fp16**
on T4 because vLLM and others refuse bf16 below sm_80. The charter checkpoints are
published in bf16; each engine holds its own fp16 rendering of the same values, and
that storage difference is printed next to every comparison it affects. Pass
`--precision bf16` for the weight-exact configuration at the cost of the engines
that cannot run it.

**Every engine sees exactly one GPU.**
`--devices 1` is the default. The worker sets `CUDA_VISIBLE_DEVICES=0` before
any CUDA context exists, so each engine sees one device and none has its placement
logic altered. Pass `--devices 2` to measure multi-device execution deliberately.

**Aether compiles to `cuda_sm70` on T4.**
T4 is sm_75. Aether's target mapping rounds to the nearest supported tier:
sm_75 → `cuda_sm70` (Volta). The `.aeg` artifact is cached and reused on `--resume`.

**No session restart ever required for onnxruntime-gpu.**
The benchmark's availability probe and all measurement workers run in subprocesses.
Subprocesses always get a fresh Python import context and see what is actually on
disk. Installing `onnxruntime-gpu` then running the benchmark is sufficient — the
orchestrator process's in-memory stale `.so` is never involved in probing or timing.

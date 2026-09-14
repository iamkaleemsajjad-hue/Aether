# Kaggle Benchmark Runbook — Aether Runtime vs All Engines

Complete, cell-by-cell instructions for running the multi-engine benchmark on a
**Kaggle GPU notebook (2× Tesla T4, 4 vCPU, ~31 GiB RAM)**.

> **Aether is compiled from the cloned source code** — not from a pip release.
> Every number the report prints reflects exactly the code on `main`.

---

## Notebook settings (before opening a cell)

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x2` |
| **Internet access** | `On` |
| **Persistence** | `Files only` (lets a second run reuse compiled artifacts) |

---

## Cell 1 — Clone and install Aether from source

```python
# Clone the latest code from main
!git clone https://github.com/iamkaleemsajjad-hue/Aether.git /kaggle/working/aether
%cd /kaggle/working/aether

# Install Aether from source (editable install — uses the cloned code, NOT the pip release).
# The [pytorch] extra pulls torch + the Aether CUDA kernels.
!pip install -q -e ".[pytorch]"

# Install the benchmark measurement stack (telemetry, plotting, HF hub utils).
# This does NOT install competing engines.  Absent engines are reported as
# NOT_INSTALLED with a reason, never silently skipped.
!pip install -q -r benchmark/requirements.txt
```

---

## Cell 2 — Install competing engines  *(order matters)*

### 2a. vLLM  *(install first — it pins torch and transformers)*

```python
# vLLM moves torch and transformers to its own pinned versions.
# Install it FIRST so that the transformers pin below (Cell 2c) wins.
# Download ~5 min.
!pip install -q vllm
```

### 2b. ONNX Runtime — GPU build

```python
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  THREE MANDATORY STEPS — skipping any one leaves CUDAExecutionProvider  ║
# ║  absent and the engine silently running on CPU.                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# STEP 1 — Remove the pre-installed CPU build.
#   Kaggle ships onnxruntime (CPU) by default.  pip marks the onnxruntime
#   dependency satisfied and skips the GPU wheel unless this is removed first.
!pip uninstall -q -y onnxruntime

# STEP 2 — Install the GPU build.  Registers CUDAExecutionProvider.
!pip install -q "onnxruntime-gpu>=1.18.0"

# STEP 3 — Install optimum separately (NOT via optimum[onnxruntime-gpu]).
#   That alias does not install the GPU wheel when onnxruntime is present.
!pip install -q "optimum>=1.20.0"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REQUIRED: Run → Restart session after this cell.                        ║
# ║  The old CPU-build .so stays resident until the session restarts.        ║
# ║  CUDAExecutionProvider will still be absent without the restart.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
```

> **→ Run → Restart session now.**  Then start from the next cell.

### 2c. transformers pin  *(mandatory for ONNX export)*

```python
%cd /kaggle/working/aether

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MANDATORY — without this pin the ONNX engine fails at load.            ║
# ║  Kaggle ships transformers 5.x.  optimum's ONNX exporter needs <4.58    ║
# ║  and imports get_parameter_dtype, which 5.x removed.                    ║
# ║  Pin BEFORE any engine loads — not afterwards.                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
!pip install -q "transformers==4.57.1"
```

### 2d. DeepSpeed  *(optional)*

```python
# Kernel injection for architectures that have a policy.
# Reports NOT_SUPPORTED (with reason) where it has none.
!pip install -q deepspeed
```

### 2e. llama.cpp  *(optional — adds ~20 min for GGUF conversion)*

```python
# llama.cpp executes GGUF, not the HF checkpoint.
# A conversion script is needed for an F16 GGUF so the comparison is weight-exact.
# Skip this cell to leave llama_cpp as NOT_INSTALLED.
!pip install -q llama-cpp-python gguf
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
```

> If you install llama.cpp add `--gguf-convert-script /kaggle/working/llama.cpp/convert_hf_to_gguf.py`
> to every benchmark command below.

---

## Cell 3 — Verify the GPU build

```python
%cd /kaggle/working/aether

import onnxruntime as ort
print("ORT version       :", ort.__version__)
print("Available providers:", ort.get_available_providers())

assert "CUDAExecutionProvider" in ort.get_available_providers(), (
    "\n\nCUDAExecutionProvider is MISSING.\n"
    "Fix:\n"
    "  (1) pip uninstall -y onnxruntime\n"
    "  (2) pip install 'onnxruntime-gpu>=1.18.0'\n"
    "  (3) Run -> Restart session\n"
    "  (4) %cd /kaggle/working/aether and re-run this cell.\n"
    "The CPU-build .so stays resident until the session is restarted."
)
print("\nOK — GPU build confirmed.")
```

---

## Cell 4 — Confirm the full environment

```python
%cd /kaggle/working/aether

!nvidia-smi

!python -c "
import torch, transformers, optimum, onnxruntime as ort
print(f'torch        {torch.__version__}')
print(f'transformers {transformers.__version__}')
print(f'optimum      {optimum.__version__}')
print(f'onnxruntime  {ort.__version__}')
print(f'CUDA avail   {torch.cuda.is_available()}')
print(f'GPU count    {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  cuda:{i}  {p.name}  {p.total_memory/1024**3:.1f} GiB  sm_{p.major}{p.minor}')
"

# Verify Aether is loaded from the cloned source (not a site-packages release)
!python -c "import aether; print('aether path:', aether.__file__)"
```

---

## Cell 5 — Smoke run  *(engine survey, ~3 min)*

```python
%cd /kaggle/working/aether

!AETHER_ORT_REQUIRE_GPU=1 python benchmark.py \
    --smoke \
    --output-dir /kaggle/working/bench_smoke
```

Read the `engine availability` block. Every engine is `run` or `skip` with a
full reason.  Fix anything unexpected before Cell 6.

**Expected availability on 2× T4:**

| Engine | Expected | Notes |
|---|---|---|
| `hf_transformers` | `run` | Reference baseline |
| `pytorch_native` | `run` | Same weights, hand-written loop |
| `onnxruntime` | `run` — CUDAExecutionProvider | Must show CUDA, not CPU |
| `aether` | `run` | Compiled from cloned source to cuda_sm70 |
| `llama_cpp` | `run` or `NOT_INSTALLED` | Only if installed in Cell 2e |
| `deepspeed` | `run` or `NOT_SUPPORTED` | Depends on model architecture |
| `vllm` | `run` at fp16 | Refuses bf16 below sm_80 |
| `sglang` | `NOT_APPLICABLE` | Needs sm_80+; T4 is sm_75 |
| `tensorrt_llm` | `NOT_APPLICABLE` | Wheels target sm_80+ |
| `exllamav2` | `NOT_APPLICABLE` | Needs EXL2 weights |
| `mlc` | `NOT_APPLICABLE` | Needs TVM-compiled model |

---

## Cell 6 — Full benchmark run

`AETHER_ORT_REQUIRE_GPU=1` makes the run fail fast if ORT is on CPU, rather than
silently producing CPU numbers under the GPU engine label.

### Full run — all models, all matrix  *(~60–90 min)*

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

### With llama.cpp *(if installed)*

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

### Raw data as DataFrames

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

### Confirm ORT actually ran on GPU

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

> **`gpu_provider_active` must be `True`.**  If it is `False`, the engine ran on
> CPU.  Go back to Cell 2b, complete all three steps, restart the session, and
> rerun from Cell 3.

---

## Cell 8 — Save results

```python
# Outputs tab — copy results into the repo directory
!cp -r /kaggle/working/benchmark_results /kaggle/working/aether/benchmark_results
```

```python
# Optional: commit results back to GitHub
# !cd /kaggle/working/aether && git add benchmark_results && \
#     git commit -m "results: Kaggle 2xT4 $(date -u +%Y-%m-%d)" && \
#     git push origin main
```

---

## Reference: What runs on 2× T4

| Engine | Status on T4 | Notes |
|---|---|---|
| `hf_transformers` | ✅ runs | Reference baseline |
| `pytorch_native` | ✅ runs | Hand-written loop, same weights |
| `onnxruntime` | ✅ runs *(with transformers pin)* | GPU build required; exporter needs transformers<4.58 |
| `aether` | ✅ runs | Compiled from source to `cuda_sm70` (T4 = sm_75 → sm70 tier) |
| `llama_cpp` | ✅ if installed | Executes GGUF (F16 auto-converted); CPU-only unless built with CUDA |
| `deepspeed` | ⚠️ partial | NOT_SUPPORTED on architectures without an injection policy |
| `vllm` | ✅ at fp16 | Refuses bf16 below sm_80; suite pins fp16 on T4 |
| `sglang` | ❌ NOT_APPLICABLE | FlashAttention needs sm_80+; T4 is sm_75 |
| `tensorrt_llm` | ❌ NOT_APPLICABLE | Published wheels require sm_80+ |
| `exllamav2` | ❌ NOT_APPLICABLE | Requires EXL2-quantized weights (none exist for charter models) |
| `mlc` | ❌ NOT_APPLICABLE | Requires a TVM-compiled artifact (harness does not build one) |

NOT_APPLICABLE engines appear in the compatibility table with the deciding reason
— they are not failures and not zeros.

---

## T4-specific notes

**Precision resolves to fp16, not bf16.**
T4 is sm_75 and has no bf16 tensor cores.  Recent torch reports
`is_bf16_supported() == True` via software emulation, but vLLM and others refuse
bf16 below sm_80 outright.  `--precision auto` resolves to **fp16** on T4.
The charter checkpoints are published in bf16; each engine holds its own fp16
rendering of the same values, and that storage difference is printed next to
every comparison it affects.  Pass `--precision bf16` for the weight-exact
configuration at the cost of engines that cannot run it.

**Every engine sees exactly one GPU.**
`--devices 1` is the default.  The worker sets `CUDA_VISIBLE_DEVICES=0` before
any CUDA context is created, so each engine finds one device and does what it
does with one device — no engine's placement logic is modified.  Pass
`--devices 2` to measure multi-device execution deliberately.

**Aether compiles to `cuda_sm70` on T4.**
T4 is sm_75.  Aether's target mapping rounds to the nearest supported tier:
sm_75 → `cuda_sm70` (Volta).  The compiled `.aeg` artifact is cached and reused
on `--resume` runs.

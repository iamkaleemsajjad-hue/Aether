# Aether Runtime — Scripts

Developer tooling scripts for the Aether Runtime project.
All scripts are self-contained Python files that add `src/` to `sys.path`
automatically — no special installation step required beyond `pip install -e ".[dev]"`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| [`check_env.py`](#check_envpy) | Verify your environment before development |
| [`setup_dev.py`](#setup_devpy) | One-command dev environment setup |
| [`compile_model.py`](#compile_modelpy) | Compile any model to a `.aeg` artifact |
| [`run_inference.py`](#run_inferencepy) | Run inference on a compiled `.aeg` package |
| [`inspect_aeg.py`](#inspect_aegpy) | Inspect/validate an `.aeg` package |
| [`convert_weights.py`](#convert_weightspy) | Convert between weight formats |
| [`benchmark_kernels.py`](#benchmark_kernelspy) | Micro-benchmark native CPU kernels |
| [`profile_memory.py`](#profile_memorypy) | Memory profiling for compilation and inference |
| [`ci_smoke_test.py`](#ci_smoke_testpy) | Standalone CI smoke test (no pytest needed) |

---

## check_env.py

Diagnose your Aether development environment.

```bash
python scripts/check_env.py           # colored report
python scripts/check_env.py --json    # machine-readable JSON
python scripts/check_env.py --strict  # fail on missing optional deps
```

Checks: Python version, required packages, optional backends, CUDA/MPS/ROCm hardware,
C++ toolchain (g++/clang++/cl), and Aether native kernel compilation.

---

## setup_dev.py

Set up your development environment in one command after cloning.

```bash
python scripts/setup_dev.py
python scripts/setup_dev.py --extras vllm mlx onnxruntime
python scripts/setup_dev.py --no-hooks --no-tests   # fast path
python scripts/setup_dev.py --check-only            # verify without installing
```

Runs: `pip install -e ".[dev]"`, import check, smoke test (real engine forward pass),
pre-commit hook installation, and unit tests.

---

## compile_model.py

Compile any model to a self-contained `.aeg` package.

```bash
# Preview plan without compiling
python scripts/compile_model.py Qwen/Qwen3-0.6B --dry-run

# Compile with Q4_K_M (default)
python scripts/compile_model.py Qwen/Qwen3-0.6B --output ./qwen3-0.6b.aeg

# Compile with INT8, multiple targets, optimizer level 2
python scripts/compile_model.py meta-llama/Llama-3.1-8B \
    --output ./llama3-8b.aeg \
    --precision INT8 \
    --opt-level 2 \
    --targets cuda_sm90 cpu_avx512

# Profile per-stage timing
python scripts/compile_model.py Qwen/Qwen3-72B --profile
```

---

## run_inference.py

Run inference on a compiled `.aeg` package.

```bash
# Interactive REPL
python scripts/run_inference.py ./my-model.aeg

# Single prompt
python scripts/run_inference.py ./my-model.aeg \
    --prompt "Explain the AEG format" --max-tokens 128

# Batch prompts from file
python scripts/run_inference.py ./my-model.aeg --prompts-file prompts.txt

# Benchmark: 20 iterations, report throughput
python scripts/run_inference.py ./my-model.aeg \
    --prompt "Benchmark" --benchmark --iterations 20

# Sampling with temperature
python scripts/run_inference.py ./my-model.aeg \
    --prompt "Tell me a story" --temperature 0.7 --top-k 50
```

---

## inspect_aeg.py

Inspect and validate a compiled `.aeg` package.

```bash
# Full report
python scripts/inspect_aeg.py ./my-model.aeg

# Include per-tensor weight table
python scripts/inspect_aeg.py ./my-model.aeg --weights

# Print graph nodes
python scripts/inspect_aeg.py ./my-model.aeg --graph

# Verify content hashes
python scripts/inspect_aeg.py ./my-model.aeg --verify

# Machine-readable JSON output
python scripts/inspect_aeg.py ./my-model.aeg --json
```

---

## convert_weights.py

Convert between AI model weight formats.

```bash
# SafeTensors model dir → AEG
python scripts/convert_weights.py safetensors ./llama3-8b/ \
    --output ./llama3-8b.aeg --precision Q4_K_M

# GGUF file → AEG
python scripts/convert_weights.py gguf ./model.gguf \
    --output ./model.aeg

# AEG → dequantized SafeTensors (for debugging / fine-tuning)
python scripts/convert_weights.py aeg-to-safetensors ./my-model.aeg \
    --output ./recovered/

# Print precision distribution
python scripts/convert_weights.py analyze ./my-model.aeg
```

---

## benchmark_kernels.py

Micro-benchmark every Aether native CPU kernel.

```bash
# Default: 7B-class sizes, 50 iterations
python scripts/benchmark_kernels.py

# 70B-class sizes
python scripts/benchmark_kernels.py --size 70b

# Single kernel, 200 iterations, save JSON
python scripts/benchmark_kernels.py --kernel sgemm \
    --iterations 200 --output bench_sgemm.json

# All kernels, custom warmup
python scripts/benchmark_kernels.py --warmup 20 --iterations 100
```

Output: Markdown table with mean/min/max latency and throughput (GFLOP/s or GB/s).

---

## profile_memory.py

Track peak RSS memory at every compilation and inference stage.

```bash
# Profile compilation
python scripts/profile_memory.py --compile Qwen/Qwen3-0.6B

# Profile inference on a compiled package
python scripts/profile_memory.py --infer ./my-model.aeg

# Profile both in sequence
python scripts/profile_memory.py --compile Qwen/Qwen3-0.6B --infer-after

# Save JSON trace for CI memory regression tracking
python scripts/profile_memory.py --compile Qwen/Qwen3-0.6B \
    --output mem_trace.json
```

---

## ci_smoke_test.py

Standalone CI smoke test — no pytest, no network, no GPU required.

```bash
python scripts/ci_smoke_test.py
python scripts/ci_smoke_test.py --verbose
python scripts/ci_smoke_test.py --junit results.xml   # JUnit XML for CI
```

Covers 15 real tests: all quantization formats, all 7 CPU kernels, engine
forward/generate, full E2E compile→save→load→infer cycle, WeightStore round-trip.
Designed to complete in under 60 seconds on any development machine.

**Use in GitHub Actions:**
```yaml
- name: Aether smoke test
  run: python scripts/ci_smoke_test.py --junit ci_results.xml
- uses: actions/upload-artifact@v4
  with:
    name: test-results
    path: ci_results.xml
```

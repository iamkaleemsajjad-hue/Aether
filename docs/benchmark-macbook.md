# MacBook benchmark: Aether versus ONNX Runtime, Qwen3-8B only

This entry point schedules exactly **two engines (`aether`, `onnxruntime`)** and
**one model (`Qwen/Qwen3-8B`)**. It preserves the existing benchmark's batch,
prompt and output sweeps. It produces raw observations, CSV/JSON, a detailed
HTML/Markdown report, comparisons and eight graphs.

**A complete head-to-head cannot currently run on a roughly 16 GB MacBook.**
Aether's native CPU loader materializes floating-point weights: about 32 GB for
8.2B FP32 parameters alone, before other allocations. Its CPU forward method
also handles a single sequence, not a true batch. This harness does not fix those
runtime limitations or disguise another engine as Aether. It records
`SKIPPED_MEMORY` and `NOT_SUPPORTED`, and never turns missing results into wins.
An Aether result requires a prepared artifact and substantially more memory.
Full batching on both engines requires further Aether runtime development.

## What the project and this benchmark execute

Aether is a Python compiler/runtime for portable Aether Execution Graph (`.aeg`)
artifacts. Its setuptools build produces a wheel and source distribution.
The existing multi-engine benchmark and historical two-T4 results remain separate.

| Engine | Actual execution | Model representation | Current limit |
|---|---|---|---|
| Aether | `aether.runtime.cpu_engine.CPUExecutionEngine` loaded from compiled AEG | Source BF16 values stored in AEG, expanded to FP32 arrays | Substantially more than 16 GB; batch 1 |
| ONNX Runtime | `CPUExecutionProvider` with a NumPy greedy generation loop | Published Qwen3-8B CPU INT4 KLD, block 128; FP32 KV | Approximately 6.1 GB artifact; larger batches can exceed available RAM |

These are different numerical representations of the same named model, not an
equal-precision comparison. The converted ONNX checkpoint's exact upstream source
revision and accuracy equivalence are not independently established. Each
artifact's own repository revision is recorded. Cross-engine comparisons require
identical workload dimensions and prompt token hashes; tokenizers are checked for
matching vocabularies and corpus encodings. Generated continuations need not match
between the different representations.

Aether preparation disables optimization passes for a conservative baseline;
this is not a measurement of every Aether optimization or serving feature.
Both engines use direct executor calls, greedy decoding, fresh KV state, fixed
output lengths and no response cache. Aether's native-kernel availability and
any NumPy kernel fallback are disclosed in each result. There is no fallback to
ONNX, PyTorch, MLX or another inference engine under the Aether label.

## 1. Check the Mac and install tools

Use Terminal. Finish Apple's installer if it opens, then continue.

```bash
sw_vers
uname -m
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize
df -h .

if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install
fi
```

If Homebrew is missing, follow [Homebrew's installation instructions](https://brew.sh/)
and its printed shell setup. Then:

```bash
brew install git python@3.12
```

Use native Python on Apple Silicon; the benchmark rejects Rosetta. The pinned
`onnxruntime==1.23.2` has macOS 13+ wheels for arm64 and x86_64 with Python
3.10–3.13. Keep the Mac plugged in and close other memory-heavy applications.
Allow at least 10 GB disk for the ONNX artifact/environment, and substantially
more for source weights, the AEG, compiler temporary files and swap if preparing
Aether (plan at least 60 GB free disk).

## 2. Clone, install and build Aether

**Publish these changes yourself first, or transfer this working checkout to
the Mac.** No commit or push was made by the coding agent. A fresh clone of a
branch without these files will not contain this benchmark.

```bash
git clone https://github.com/iamkaleemsajjad-hue/Aether.git
cd Aether

# Check out your branch here if you published the changes on another branch.
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e .
python -m pip install -r benchmark/macbook/requirements.txt
python -m pip check
python -m build

python -c 'from pathlib import Path; import aether; print(aether.__file__); assert Path(aether.__file__).resolve().is_relative_to(Path.cwd() / "src")'
python -m benchmark.macbook doctor
```

`dist/` contains Aether's built distributions; the editable installation runs
this checkout's source. The benchmark itself is invoked from the repository
root. No CUDA installation is needed. `doctor` prints chip/RAM/package details
and executes tiny real ONNX `MatMulNBits`, `GatherBlockQuantized` and
`GroupQueryAttention` checks (rotary, prefill, decode, KV growth and batches 1/2).
These are compatibility checks, not Qwen3-8B performance or accuracy results.
If a kernel check fails, resolve that error before preparing the large artifact.

Optional local harness verification:

```bash
python -m pip install pytest pytest-cov
python -m pytest tests/benchmarks/test_macbook_benchmark.py -q --no-cov
```

## 3. Prepare the model artifacts

Prepare the ONNX representation on either size Mac:

```bash
python -m benchmark.macbook prepare
```

This downloads only the four necessary files from the published
[Qwen3-8B CPU INT4 artifact](https://huggingface.co/onnx-community/Qwen3-8B-ONNX/tree/main/onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128),
locks its Hub revision, verifies size/LFS hashes and writes local SHA-256 hashes
to `models/macbook-onnx-int4/model.lock.json`. The original BF16 weights are not
needed for this step. Every benchmark run rechecks the local hashes.

**Aether preparation requires a larger-memory host. Do not expect this command
to complete on 16 GB:**

```bash
# Only on a host with at least 56 GiB currently available RAM:
python -m benchmark.macbook prepare-aether
```

This resolves `Qwen/Qwen3-8B` to an immutable source revision, downloads the source
checkpoint, compiles `models/macbook-aether/model.aeg`, and writes
`aether.lock.json`. It targets `cpu_neon` on Apple Silicon and `cpu_avx2` on Intel.
The 56 GiB preparation guard and 48 GiB inference estimate are planning estimates,
not measured peak requirements or guarantees. A nominal 64 GB machine can still
have too little *available* memory. Preparation on another host is possible;
transfer the entire prepared `models/macbook-aether` directory to the benchmark
Mac. That Mac still needs enough RAM to load and execute the artifact.

Preparation refuses to overwrite existing artifacts. For reproducibility,
`prepare --revision SHA --model-dir models/onnx-repeat` and
`prepare-aether --revision SHA --aether-dir models/aether-repeat` accept the
respective revision from their locks. Supply the same directories to `run`.

## 4. Smoke check on the approximately 16 GB Mac

This loads the real Qwen3-8B ONNX artifact and records Aether's memory limitation.
It is an **incomplete** head-to-head, not a valid two-engine performance result.

```bash
caffeinate -i python -m benchmark.macbook run \
  --smoke --warmup-iters 1 --measure-iters 2 \
  --max-memory-gib 12 \
  --output-dir benchmark_results_macbook_smoke

open benchmark_results_macbook_smoke/reports/REPORT.html
```

Expected if sufficient RAM is available: ONNX `MEASURED`, Aether
`SKIPPED_MEMORY`, exit code 2. If ONNX is skipped, close other apps and retry in
a new output directory. Failures are logged in
`raw/onnxruntime/b1_p32_o4.log`. On a sufficiently large Mac with both artifacts
prepared, use a new output directory and `--max-memory-gib 52` for this check;
both engine rows must be `MEASURED` before treating it as a successful head-to-head.

## 5. Existing batch/token sweeps

The following command is appropriate for the approximately 16 GB Mac. Aether
rows will show the limitations described above; ONNX rows run when memory allows.

```bash
caffeinate -i python -m benchmark.macbook run \
  --batch-sizes 1,2,4,8,16 \
  --prompt-tokens 32,256,1024 \
  --output-tokens 32,128,512 \
  --warmup-iters 2 --measure-iters 5 \
  --max-memory-gib 12 --worker-timeout 7200 --cooldown 5 \
  --output-dir benchmark_results_macbook

python -m pip freeze > benchmark_results_macbook/environment.txt
```

As in the old suite, these are nine distinct workloads, now with two engine rows
each (18 scheduled measurements):

| Sweep | Batch | Prompt tokens | Output tokens |
|---|---|---|---|
| Batch | 1, 2, 4, 8, 16 | 256 | 128 |
| Prompt | 1 | 32, 256, 1024 | 128 |
| Output | 1 | 256 | 32, 128, 512 |

The shared baseline b1/p256/o128 is measured once per engine. Cases run
sequentially in fresh processes; engine order alternates between workloads.
Larger CPU workloads can take hours. No wall-clock estimate is claimed without
measuring your hardware. A 12 GiB RSS cap leaves memory for the OS. The actual
cap is also limited to 85% of physical RAM. Before spawning a worker, estimates
are checked against both the cap and available RAM minus 1 GiB. Memory pressure
and swap can still affect runs; RSS sampling cannot guarantee detection of every
transient peak. Over-cap workers are killed and recorded as `MEMORY_LIMIT`.

For a sufficiently large Mac with both artifacts prepared, the presently
supported **batch-1 head-to-head across token lengths** is:

```bash
caffeinate -i python -m benchmark.macbook run \
  --batch-sizes 1 --prompt-tokens 32,256,1024 --output-tokens 32,128,512 \
  --warmup-iters 2 --measure-iters 5 \
  --max-memory-gib 52 --worker-timeout 7200 --cooldown 5 \
  --output-dir benchmark_results_macbook_paired
```

For all 45 workload combinations (90 engine rows), including explicit unsupported
or memory-limited cases:

```bash
caffeinate -i python -m benchmark.macbook run \
  --matrix grid --batch-sizes 1,2,4,8,16 \
  --prompt-tokens 32,256,1024 --output-tokens 32,128,512 \
  --warmup-iters 2 --measure-iters 5 \
  --max-memory-gib 12 --worker-timeout 7200 \
  --output-dir benchmark_results_macbook_grid
```

Exit codes: **0** all requested engine/cases measured; **2** partial results;
**1** no measured cases or an error. Partial runs still get reports. Never reuse
an output directory containing prior run evidence.

## 6. Generate and open detailed reports, comparisons and graphs

```bash
python -m benchmark.macbook report --output-dir benchmark_results_macbook
open benchmark_results_macbook/reports/REPORT.html
open benchmark_results_macbook/reports/REPORT.md
open benchmark_results_macbook/graphs
open benchmark_results_macbook/benchmark_results.csv
open benchmark_results_macbook/benchmark_comparisons.csv
```

Change the output directory in these commands for smoke, paired or grid runs.
`report` rebuilds from saved observations without inference, and marks unfinished
cells `NOT_COMPLETED`. It does not fabricate results or interpolate missing cells.

```text
run.json                       model locks, host, versions, source hashes, settings
benchmark_results.json         observations, comparisons and graph list
benchmark_results.csv          engine/case metrics, statuses and reasons
benchmark_comparisons.csv       within-engine ratios and matched Aether/ONNX ratios
raw/aether/                    per-workload jobs, observations and worker logs
raw/onnxruntime/                per-workload jobs, observations and worker logs
reports/REPORT.html             offline report with graphs and full details
reports/REPORT.md               detailed report, provenance, samples and comparisons
graphs/                        eight batch/prompt/output plots
```

Throughput is batch × generated tokens / measured latency; per-request throughput
is generated tokens / latency. TTFT includes prefill and first greedy selection;
TPOT is the remaining decode duration divided by output tokens minus one.
Download, compile, loading and tokenization are excluded from generation timing.
Loading and cold calls are reported separately. Timings include synchronous
engine calls, Python generation, cache allocation and finite-logit checks. There
are discarded warmups, timed repetitions and a separate telemetry generation.

CSV/JSON include n, median, mean, standard deviation, p90, p95 and coefficient of
variation. Small-sample percentiles are descriptive, not production tail-latency
estimates. Raw token counts, prompt hashes, repeatability checks, generated samples,
RSS and swap observations are retained. No chat template or thinking-mode quality
evaluation is performed: the workload is exact-length repeated corpus tokens,
greedy continuation and ignored EOS, matching the fixed-length benchmark intent.

Within-engine comparisons hold the other workload dimensions fixed. Cross-engine
ratios are **Aether / ONNX**: throughput above 1 favors Aether; latency below 1
favors Aether. They are emitted only for measured pairs with identical prompt
hashes and are explicitly labeled as different representations. With no such
pairs the report says **INCOMPLETE HEAD-TO-HEAD**, and reports no winner.
No speedup against the historical T4 run is established.

Local validation exercises tiny real ONNX graphs and the Aether CPU executor,
not the complete 8B model. Full-model success, performance and Mac compatibility
remain unverified until the real smoke/benchmark commands run on the target Mac.
Operator details are documented in [ONNX Runtime's versioned operator reference](https://github.com/microsoft/onnxruntime/blob/v1.23.2/docs/ContribOperators.md).

## Validation performed during implementation

On the Windows development host: 22 dedicated benchmark tests passed, including
real ORT INT4/attention kernels, growing KV caches, actual tiny Aether CPU forward
passes, exact two-engine scheduling, memory/timeout/crash handling, artifact
checks and report generation. The related existing CPU/ingestion tests had 50
passes and 2 skips because source model weights were unavailable. Python syntax
and E/F lint checks passed; the Aether 1.3.0 wheel and source distribution built.
Graphs were visually reviewed using clearly labeled synthetic test data.

This is not a claim that the entire repository test suite is green: four existing
legacy benchmark tests expect engines absent from the current suite registry
(`torch_compile`, `vllm`, `exllamav2`) and failed independently of this entry point.
Those unrelated tests were not changed. No full Qwen3-8B inference, macOS run,
source-model accuracy comparison or real performance report was produced here.

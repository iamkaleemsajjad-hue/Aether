# Aether Runtime vs Hugging Face Transformers — Benchmark

> **Two benchmarks live here.** This document describes the controlled two-runtime
> A/B (`python benchmark/run_benchmark.py`), which measures Aether against
> Transformers in one process with execution order alternated per cell. The
> multi-engine competition — Aether against thirteen inference stacks, with
> per-engine process isolation, engine taxonomy, batch scaling to 16, win/loss
> matrices and the compile-once probe — is `python benchmark.py`; see
> [`suite/README.md`](suite/README.md) and, for a step-by-step notebook runbook,
> [`docs/benchmark-kaggle.md`](../docs/benchmark-kaggle.md). Both read the same
> locked model list from `config.py` and take their measurements with the same
> primitives in `runner.py`, so their numbers are directly comparable.

A neutral, reproducible comparison of two **runtimes** executing the **same model
architectures and the same weights**.

The purpose is not to show that either side wins. It is to measure where each is
faster or slower, what resources each uses, which kernels each executes, and
whether there is any accuracy cost. Results that favour Transformers are reported
exactly as prominently as results that favour Aether, and configurations that
fail are recorded rather than worked around.

## Models

Fixed by the benchmark's charter — three decoder families with deliberately
different structure, and a size range that spans half an order of magnitude
while fitting one 15 GiB accelerator at 16-bit weights:

| Model | Params | Layers | Positional scheme |
|-------|-------:|-------:|-------------------|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | 30 | RoPE |
| `Qwen/Qwen3-0.6B` | 0.6B | 28 | RoPE + per-head Q/K norm |
| `SummerSigh/GPTNeo350M-Instruct-SFT` | 350M | 24 | learned absolute (no RoPE) |

They are downloaded automatically from the Hub. Nothing is fetched by hand, and
both backends load the same repository at the same revision — which the report
records as a commit sha.

---

## Reproducing on a fresh Kaggle notebook

**Step 1 — create the notebook.** New Notebook, Python.

**Step 2 — enable the GPU.** *Settings → Accelerator → GPU T4 x2* (or *P100*).
The benchmark detects how many accelerators are actually visible; it does not
assume two.

**Step 3 — clone and install.**

```python
!git clone https://github.com/KaleemSajjad/Aether-Runtime.git /kaggle/working/aether
%cd /kaggle/working/aether
!pip install -q -e ".[pytorch]"
!pip install -q -r benchmark/requirements.txt
```

**Step 4 — confirm the environment.**

```python
!python benchmark/run_benchmark.py --quick --mode performance
```

A quick run touches one model, one prompt length and two iterations. It exists to
prove the pipeline works before spending GPU budget on the full matrix.

**Step 5 — run the full performance benchmark.**

```python
!python benchmark/run_benchmark.py --mode performance
```

**Step 6 — read the results.** Written to `benchmark/results/`:

```python
from IPython.display import Markdown, display
display(Markdown(open('benchmark/results/REPORT.md').read()))
```

**Step 7 — the remaining modes**, each independent so nothing is measured twice:

```python
!python benchmark/run_benchmark.py --mode correctness
!python benchmark/run_benchmark.py --mode profile
!python benchmark/run_benchmark.py --mode batch
!python benchmark/run_benchmark.py --mode mixed
!python benchmark/run_benchmark.py --mode multigpu
```

Or everything in sequence, which takes considerably longer:

```python
!python benchmark/run_benchmark.py --mode all
```

---

## Modes

| Mode | Question it answers |
|------|---------------------|
| `performance` | Throughput, latency, TTFT, prefill/decode split, memory, CPU/GPU utilization. |
| `memory` | Runs the performance path and surfaces its memory sections. |
| `correctness` | Do both runtimes compute the same logits, the same greedy tokens, the same text? |
| `profile` | Kernel counts and per-kernel time attribution. Instrumented — never feeds a throughput number. |
| `batch` | What batching buys, with aggregate throughput and per-request throughput reported separately. |
| `mixed` | Batches whose rows genuinely differ in length, with the padding overhead reported next to the throughput. |
| `multigpu` | How each runtime uses more than one accelerator, with configurations labelled so a 2-GPU run is never compared against a 1-GPU one. |

### Reading the two prefill columns

`performance` reports prefill twice, because the two answer different questions:

- **prefill (all logits)** — one forward pass returning logits at *every* prompt
  position. A like-for-like comparison of that operation.
- **prefill (serving)** — the same pass returning only the final position's logits,
  which is all generation reads. Both backends get the same option: Transformers via
  `logits_to_keep=1` (what its own `generate` uses), Aether via its last-position
  projection.

The `discarded` column is the ratio between them: work spent on logits nothing
reads. It grows with prompt length and vocabulary size — on Qwen3-0.6B, whose
`lm_head` is 156M of ~596M matmul parameters, a 1024-token prompt was projecting
1024 positions to a 151936-wide vocabulary and using one.

End-to-end throughput, TTFT and the decode column reflect the serving
configuration.

### Reading the `mixed` mode

Rows within a batch differ in length, so both runtimes pay for padding. `pad %` is
the fraction of padded slots holding no real token. `uniform-256` is the control —
same row count, zero padding — so the gap between it and the ragged profiles is
what raggedness costs. Latency percentiles describe the whole batched pass; every
row in a batch shares one wall time.

### Reading the `batch` mode

Batching is a **throughput** mechanism, not a latency one. The mode therefore
prints two rates per cell and never collapses them:

- **`batch tok/s`** — the whole pass's output over its wall time. This is the
  number batching is supposed to raise.
- **`per-request tok/s`** — one row's output over that same wall time: what a
  single caller waiting inside the batch experiences. This is *not* expected to
  improve, and usually falls.

`scaling` compares a cell against **that same backend's** batch-1 aggregate
throughput, so each runtime is measured against itself rather than against the
other.

Both backends replicate one prompt to the batch width (`tokenizer([prompt] * B)`
on the Transformers side, the same ids repeated on Aether's), so the two are given
identical work and neither batch carries padding. Aether decodes the rows
concurrently in one KV tensor; if the loaded engine cannot batch, the cell is
recorded as unsupported rather than being quietly serialized into a loop.


## Useful switches

```bash
--models Qwen/Qwen3-0.6B          # subset of the three
--precisions bf16,fp16,fp32       # default: bf16 only (see below)
--prompt-tokens 32,256,1024       # exact token counts, not characters
--batch-sizes 1,2,4               # both runtimes execute these as real batches
--max-new-tokens 128
--warmup-iters 2 --measure-iters 5
--devices 1                       # pin visibility to the first N GPUs
--cooldown 30                     # idle between sections so the GPU can cool
--output-dir benchmark/results
```

---

## Why `bf16` is the default comparison

All three checkpoints are published in BF16, and the Aether compiler's default
weight residency inside the `.aeg` is also BF16. At `bf16` the two backends
therefore hold the **same weight values**, and the comparison isolates execution.

At `fp16` and `fp32` they do not: Transformers loads the published checkpoint
directly, while Aether's weights have passed through BF16 storage. Those
configurations are still available and still reported — but the report states the
difference rather than presenting them as like-for-like.

## Fairness

Identical wherever technically possible: host, GPUs, model, revision, tokenizer
(verified, not assumed), prompt text, token counts, `max_new_tokens`, sampling
settings, seed, warm-up and measured iteration counts.

- **Greedy decoding** (`temperature=0`) for the primary comparison, so the
  measured work is deterministic.
- **Phases are separated.** Download, compile, load, warm-up and steady-state are
  distinct measurements; warm-up iterations are executed and discarded; the cold
  iteration is reported separately from the steady state.
- **Order is alternated.** Which backend runs first flips every repetition, so
  thermal drift or clock ramping cannot land preferentially on one of them.
- **CUDA is synchronized** on both edges of every timed region.
- **Telemetry is quarantined.** GPU and CPU sampling runs in dedicated extra
  iterations, never during the iterations whose latency is reported.
- **Profiling is quarantined.** Instrumentation perturbs a launch-bound decode
  loop, so no throughput figure comes from a profiled run.
- **Neither side is handicapped.** Transformers uses `model.generate` with its own
  KV cache and its default attention implementation for the device (which selects
  SDPA, and FlashAttention within SDPA where the GPU supports it). Aether uses
  `Compiler` and `Runtime` with default settings.

### One configuration flag is overridden, and here is why

Aether's `RuntimeConfig` enables a **semantic response cache** by default: an
identical prompt returns a stored completion without running the model. Measured
on this repository:

| `enable_semantic_cache` | 1st call | 2nd call | 3rd call |
|---|---|---|---|
| `True` (Aether's default) | 55.85 s | **0.001 s** | **0.000 s** |
| `False` (benchmark setting) | 66.38 s | 14.66 s | 17.95 s |

A benchmark issues the same prompt many times. With the cache on, every iteration
after the first measures a dictionary lookup rather than inference — and
Transformers has no equivalent, so the comparison would be meaningless rather than
merely flattering. The benchmark therefore sets `enable_semantic_cache=False`,
reports the override in `describe()`, and prints it in the report.

This is a flag on the public config, **not** a change to Aether. A test asserts
that Aether's own default remains `True`, so the benchmark cannot drift into
modifying the runtime it is measuring.

Aether Runtime was **not modified** for this benchmark.

## Known asymmetries, stated up front

- **Batching.** Aether's portable engine carries no batch dimension, verified by
  source inspection: a 2-D input is flattened into one sequence. Batch sizes above
  1 are therefore reported as `unsupported` rather than measured. Any batch>1 row
  is a Transformers-only observation.
- **Compilation.** Aether pays a one-time compile cost, timed separately as
  `prepare_s` and never amortized into throughput.
- **TTFT.** Measured through each library's own streaming API. Those are not
  identical code paths, so TTFT is a weaker comparison than throughput.

## Output

```
benchmark/results/
  REPORT.md         human-readable report
  results.json      complete raw record, including every failure
  results.csv       flat performance table
  throughput.png    tokens/second, axis from zero
  gpu_memory.png    peak reserved GPU memory, axis from zero
  aeg-cache/        compiled .aeg artifacts, reused across runs
```

## Files

| File | Role |
|------|------|
| `run_benchmark.py` | Entry point; one function per mode. |
| `config.py` | Every parameter, plus CLI parsing. Printed at the start of each run. |
| `system_info.py` | Hardware and software capture, including model revisions. |
| `prompts.py` | Prompts built to an exact token count; tokenizer-agreement check. |
| `metrics.py` | Statistics and the synchronize-timed-synchronize primitive. |
| `backends.py` | The contract both backends implement, so the runner cannot favour either. |
| `backend_transformers.py` | The Transformers baseline. |
| `backend_aether.py` | The Aether backend: compile, then run through `Runtime`. |
| `runner.py` | Warm-up, alternated repetitions, phase separation, failure capture. |
| `memory_monitor.py` | Process RSS and CPU sampling from OS APIs. |
| `gpu_monitor.py` | Allocator accounting plus NVML telemetry. |
| `correctness.py` | Logit, token and text comparison with an explicit tolerance. |
| `profiling.py` | Dispatch counting and profiler attribution. |
| `reporting.py` | `REPORT.md`, `results.json`, `results.csv`, charts. |

---

## Validating the harness offline

`--models` also accepts an existing local checkpoint directory. That is a
**pipeline check, not a benchmark result**: the run prints a note, and
`results.json` records `harness_validation_only`, so such a run can never be
mistaken for a charter measurement.

```bash
python benchmark/run_benchmark.py --models ./path/to/checkpoint \
    --prompt-tokens 16 --max-new-tokens 8 --measure-iters 2
```

## Further reading

`ARCHITECTURE.md` in this directory describes how each runtime executes a model,
based on source inspection. Every claim there is labelled `[source]`,
`[measured]`, or `[inferred]`, so a reader can see which statements rest on the
implementation, which on measurement, and which are unconfirmed hypotheses.

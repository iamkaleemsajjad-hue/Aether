# Running the multi-engine benchmark on Kaggle

A start-to-finish runbook for `python benchmark.py` on a Kaggle GPU notebook
(2x Tesla T4, 4 vCPU, ~31 GiB RAM). Every numbered step is one cell.

Read [What can actually run on 2x T4](#what-can-actually-run-on-2x-t4) first. All four
engines run here, but not all of them run on the GPU, and the suite says so with the
device each one reported rather than leaving a slow row to be misread.

[Exact versions](#exact-versions) lists the stack these runs were executed against, which
of it is pinned and which is inherited from the Kaggle image. [Same environment, same
input](#same-environment-same-input) is how the suite makes "all four engines, same
conditions" a checked property rather than a claim.

## 0. Push to `main` first

The notebook clones the repository, so the code has to be on GitHub before you start —
**including the four-engine field itself**. The engine list, the per-cell failure
handling and the position-ceiling fix all live in the working tree; a notebook run against
an older `main` measures whatever field that commit declared, which is how a run ends up
reporting more than four engines.

```bash
git add -A
git commit -m "bench: four-engine field; refuse positions past a learned table"
git push origin main
```

Confirm the clone got the right field before measuring anything (step 2 must have run):

```python
!python -c "from benchmark.suite import engines; print(list(engines.KEYS))"
# ['transformers', 'pytorch_native', 'openvino', 'aether']
```

If that prints anything else, the notebook is running older code — re-clone.

## 1. Notebook settings

- **Accelerator**: `GPU T4 x2`
- **Internet**: `On`
- **Persistence**: `Files only` is useful — a second run then reuses compiled artifacts

## 2. Clone and install the measurement stack

```python
!git clone https://github.com/iamkaleemsajjad-hue/Aether.git /kaggle/working/aether
%cd /kaggle/working/aether
!pip install -q -e ".[pytorch]"
!pip install -q -r benchmark/requirements.txt
```

## 3. Install the competing engine

Three of the four engines need nothing beyond step 2: `transformers`, `pytorch_native`
and `aether` all run on the stack already installed. One needs a package:

```python
# OpenVINO, through optimum's exporter. ~2 minutes.
!pip install -q "optimum[openvino]"
```

**Restart the session now** (Run -> Restart session), so every engine is measured
against one set of libraries. Then `%cd /kaggle/working/aether` again.

`optimum-intel` declares a `transformers` range. If the installed version falls outside
it the suite says so in the compatibility table, with both versions named, instead of
failing on a private symbol twenty minutes into the run. Pin into the declared range
before measuring if that happens - every engine in a run has to share one `transformers`
version, so pinning afterwards would mean the field was not measured against one stack:

```python
!pip install -q "transformers==4.57.1"
```

## 4. Confirm the environment

This cell prints every version the report will record, and fails loudly on the two
constraints that are not negotiable: a CUDA-enabled torch, and a `transformers` inside
the range `optimum-intel` declares. Everything else is recorded rather than enforced —
see [Exact versions](#exact-versions).

```python
import importlib.metadata as md, sys, torch
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| cuda", torch.version.cuda,
      "| devices", torch.cuda.device_count())
for name in ("transformers", "accelerate", "openvino", "optimum", "optimum-intel",
             "numpy", "huggingface-hub"):
    try:
        print(f"{name:16s}", md.version(name))
    except md.PackageNotFoundError:
        print(f"{name:16s} not installed")
import aether; print("aether          ", aether.__version__)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        print(f"gpu{i}         {torch.cuda.get_device_name(i)} sm_{major}{minor}",
              f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GiB")

assert torch.cuda.is_available(), "no CUDA device: set Accelerator to GPU T4 x2"
from benchmark.suite.engines.base import requirement_conflicts
for problem in requirement_conflicts("optimum-intel"):
    print("CONFLICT:", problem)
```

`requirement_conflicts` is the same function the report's compatibility table uses, so
what it prints here is what would appear there an hour later.

```python
!nvidia-smi
```

## 5. Check which engines the host accepts, before spending an hour

The smoke run surveys the field, measures one tiny cell per engine, and writes a full
report. Two to three minutes.

```python
!python benchmark.py --smoke --output-dir /kaggle/working/bench_smoke
```

Read the `engine availability` block it prints. Every engine is either `run` or `skip`
with a reason in full — no truncation. Fix anything surprising there before step 6.

`--smoke` runs the *first* model only (SmolLM2-135M), so it does not touch the other two.
One of them is worth a cell of its own before an hour-long run: GPT-Neo is the only
checkpoint in the set with a learned absolute position table, which makes its 2048
positions a hard ceiling rather than a rotary table that can be rebuilt taller. It is
also the model that used to take the run down at load time. Two minutes:

```python
!python benchmark.py \
    --models SummerSigh/GPTNeo350M-Instruct-SFT \
    --batch-sizes 1 --prompt-tokens 32 --output-tokens 16 \
    --primary-prompt-tokens 32 --primary-output-tokens 16 \
    --warmup-iters 1 --measure-iters 2 --no-charts \
    --output-dir /kaggle/working/bench_neo
```

All four engines should reach `MEASURED`. If one does not, the reason is in that run's
report and the failure is recorded per cell — a failed cell no longer ends the run.

## 6. The full run

```python
!python benchmark.py --output-dir /kaggle/working/benchmark_results
```

Three models, every engine the host accepts, batch 1/2/4/8/16, prompts 32/256/1024,
outputs 32/128/512, correctness validation and the compile-once probe. The largest
checkpoint in the set is 0.6B, so the run is a fraction of the wall-clock a
multi-billion-parameter model cost.

A shorter run that still fills every section of the report:

```python
!python benchmark.py \
    --batch-sizes 1,2,4,8 \
    --prompt-tokens 32,256 \
    --output-tokens 32,128 \
    --measure-iters 5 \
    --output-dir /kaggle/working/benchmark_results
```

If the session dies part way, `--resume` keeps what was written and measures only what
is missing:

```python
!python benchmark.py --resume --output-dir /kaggle/working/benchmark_results
```

For a long run prefer **Save & Run All** (batch execution) over the interactive session,
which times out.

## 7. Read the results

```python
from IPython.display import Markdown, display
display(Markdown(open(
    "/kaggle/working/benchmark_results/reports/BENCHMARK_REPORT.md", encoding="utf-8"
).read()))
```

```python
from IPython.display import Image, display
import pathlib
for path in sorted(pathlib.Path("/kaggle/working/benchmark_results/graphs").glob("*.png")):
    print(path.name)
    display(Image(str(path)))
```

```python
import pandas as pd
pd.read_csv("/kaggle/working/benchmark_results/benchmark_results.csv")       # every cell
pd.read_csv("/kaggle/working/benchmark_results/benchmark_comparisons.csv")   # every pairing
```

## 8. Keep the results

Everything is under `/kaggle/working/benchmark_results`, downloadable from the Output
tab. To bring it back into the repository:

```python
!cp -r /kaggle/working/benchmark_results /kaggle/working/aether/benchmark_results
```

---

## Exact versions

Two things are pinned, and everything else is inherited from the Kaggle image and
recorded. That split is deliberate: reinstalling torch on Kaggle is the single most
reliable way to break a run, because the wheel has to match the driver and the CUDA
runtime already in the image, and a mismatch surfaces as a CUDA error hours later
rather than as a failed install.

**Pinned, because the suite's correctness depends on it:**

| Package | Pin | Why |
| --- | --- | --- |
| `transformers` | `==4.57.1` | Three engines load through it, and `optimum-intel` declares a hard upper bound. One version for the whole field, chosen before anything is measured |
| `optimum[openvino]` | latest that satisfies the `transformers` pin | Supplies `openvino` and `optimum-intel`; the OpenVINO engine is absent without it |

**Inherited from the image, recorded in the report, never reinstalled:**

| Component | Observed | Note |
| --- | --- | --- |
| Python | `3.12.13` | The image's interpreter |
| torch | `2.10.0+cu128` / `2.13.0+cu130` | Both images this suite has run on. Any `>=2.5` CUDA build works |
| CUDA runtime | `12.8` / `13.0` | Whatever the torch wheel was built against |
| GPU | 2x Tesla T4, `sm_75`, 14.6 GiB each | Fixes precision to fp16; see below |
| `openvino` | `2026.3.1` | Pulled in by `optimum[openvino]` |
| `aether-runtime` | `1.2.8a0` | This repository, installed `-e` |
| Precision | `fp16` | Derived from `sm_75`, not chosen |

The full install, in order, is steps 2 and 3 above and nothing else:

```python
!pip install -q -e ".[pytorch]"
!pip install -q -r benchmark/requirements.txt
!pip install -q "optimum[openvino]"
!pip install -q "transformers==4.57.1"
# then: Run -> Restart session
```

The `transformers` pin goes **last** so it wins over anything `optimum[openvino]` pulled
in, and the restart is what makes one library set apply to every engine in the run.
`benchmark/requirements.txt` deliberately declares floors rather than pins, so pip keeps
the image's own newer torch, matplotlib and psutil instead of downgrading them.

The report records all of this per run under `environment` in `benchmark_results.json`,
so a comparison between two runs can be checked rather than assumed.

## Same environment, same input

"All four engines, same conditions" is enforced by the harness, not left to the operator:

- **One worker process per (engine, model)**, run one at a time. Two engines measured
  concurrently would contend for the same GPU, and an engine that claims all of a device
  cannot share a process with another that does the same.
- **The prompts are built once**, by the orchestrator, with the model's own tokenizer to
  an exact token count — then written to a workload file every worker reads. Each worker
  re-encodes those exact strings with its own engine's tokenizer and verifies the ids
  match. If each worker built its own prompts, a tokenizer difference would hand two
  engines different amounts of work while the table claimed they had the same.
- **One precision for the field**, resolved once from the hardware (`--precision auto`
  gives fp16 on T4) and applied to every engine.
- **One device per engine** — `--devices 1` by default, applied by restricting
  `CUDA_VISIBLE_DEVICES` in each worker before any CUDA context exists.
- **One thread budget**, pinned in every worker before torch initializes; when it is not
  set, the report says the budget was inherited rather than pretending it was controlled.
- **Same weights**: every engine loads the same checkpoint at the same revision, resolved
  once and recorded.

Where an engine genuinely cannot match the others — OpenVINO having no CUDA plugin, so
executing on CPU — the difference is labelled on every affected pairing rather than
averaged into a headline. The one deliberate configuration override in the field is
Aether's semantic response cache, which is turned off for measured runs so a repeated
prompt times inference instead of a cache hit; it is disclosed in the engine's notes.

## What can actually run on 2x T4

The suite attempts four engines. On this hardware:

| Engine | On 2x T4 | Executes on | Why |
| --- | --- | --- | --- |
| `transformers` | runs | GPU | — |
| `pytorch_native` | runs | GPU | — |
| `aether` | runs | GPU | — |
| `openvino` | runs | **CPU (2 cores)** | OpenVINO ships no CUDA plugin, so there is no NVIDIA device for it to target. Not a misconfiguration and not fixable with a flag |

OpenVINO's row is measured, ranked and disclosed, not dropped and not quietly compared:
it reports `CPU` as its execution device, every pairing against a GPU engine is labelled
`DEVICE_DIFFERENCE`, those pairings are counted separately in the standings, and no
percentage that crosses the boundary is presented as a difference between the two
stacks. It will be several times slower in wall-clock here, and that number is about
two T4 cores against a T4, not about OpenVINO against Aether.

On an Intel CPU or iGPU host, where OpenVINO does have a GPU plugin,
`--openvino-device GPU` targets it and the labelling follows the device the plugin
reports back. `--openvino-device auto` (the default) picks the fastest device OpenVINO
can actually see, which is the same courtesy the torch engines get from CUDA.

## Two things specific to T4 worth knowing

**Precision is fp16, not bf16.** T4 is compute capability 7.5 and has no bf16 tensor
cores. Recent torch answers `is_bf16_supported() == True` there anyway, because it can
emulate the format in software — which is why an earlier version of this suite chose a
precision part of the field could not execute natively. The suite now derives bf16 support from the
capability, so `--precision auto` resolves to fp16 on T4: the widest format the whole
field executes natively. The checkpoints are published in bf16, so each engine holds its
own 16-bit rendering of the same values; that storage difference is printed next to
every comparison it affects. `--precision bf16` gives the bit-exact configuration at the
cost of the engines that cannot run it.

**Every engine sees one GPU.** `--devices 1` is the default, applied by restricting
`CUDA_VISIBLE_DEVICES` in each worker before any CUDA context exists. Nothing in any
engine's placement logic is modified — each simply finds one device. Without this, a
runtime that shards a model across both T4s is measured on twice the hardware as one
that does not, which is a comparison of machines rather than of engines. Pass
`--devices 2` to measure multi-device execution deliberately.

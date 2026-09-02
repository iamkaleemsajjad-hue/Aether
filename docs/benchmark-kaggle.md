# Running the multi-engine benchmark on Kaggle

A start-to-finish runbook for `python benchmark.py` on a Kaggle GPU notebook
(2x Tesla T4, 4 vCPU, ~31 GiB RAM). Every numbered step is one cell.

Read [What can actually run on 2x T4](#what-can-actually-run-on-2x-t4) first. All four
engines run here, but not all of them run on the GPU, and the suite says so with the
device each one reported rather than leaving a slow row to be misread.

## 0. Push to `main` first

The notebook clones the repository, so the code has to be on GitHub before you start:

```bash
git add -A
git commit -m "bench: multi-engine benchmark suite"
git push origin main
```

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

```python
!nvidia-smi
!python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count())"
```

## 5. Check which engines the host accepts, before spending an hour

The smoke run surveys the field, measures one tiny cell per engine, and writes a full
report. Two to three minutes.

```python
!python benchmark.py --smoke --output-dir /kaggle/working/bench_smoke
```

Read the `engine availability` block it prints. Every engine is either `run` or `skip`
with a reason in full — no truncation. Fix anything surprising there before step 6.

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

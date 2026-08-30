# Running the multi-engine benchmark on Kaggle

A start-to-finish runbook for `python benchmark.py` on a Kaggle GPU notebook
(2x Tesla T4, 4 vCPU, ~31 GiB RAM). Every step is a cell you can paste as-is.

The same steps work on Colab or any single-host GPU box; only the accelerator
selection and the working directory differ.

## Before you start: push to `main`

The notebook clones the repository, so the code has to be on GitHub first. From your
working copy:

```bash
git status                       # confirm what you are about to commit
git add -A
git commit -m "bench: multi-engine benchmark suite"
git push origin main
```

## 1. Notebook setup

In the Kaggle notebook sidebar:

- **Accelerator**: `GPU T4 x2`
- **Internet**: `On` (model downloads and pip installs need it)
- **Persistence**: optional; `Files only` lets a second run reuse compiled artifacts

## 2. Clone and install

```python
!git clone https://github.com/<your-account>/<your-repo>.git /kaggle/working/aether
%cd /kaggle/working/aether
!git log --oneline -1
```

```python
# Aether itself, plus the measurement dependencies. The Kaggle image already ships
# torch, transformers, psutil and matplotlib; pip keeps whatever newer version is there.
!pip install -q -e ".[pytorch]"
!pip install -q -r benchmark/requirements.txt
```

Optional competing engines. Install only what you want measured; anything absent is
reported as `NOT_INSTALLED` with that reason, never as zero throughput.

```python
!pip install -q "optimum[onnxruntime-gpu]"      # ONNX Runtime on CUDA
!pip install -q vllm                             # vLLM (large download, ~5 min)
```

> Installing vLLM upgrades torch on most images. Do it **before** the run, then
> restart the session, so every engine is measured against one torch build. The
> report records the versions it actually saw.

## 3. Confirm the environment

```python
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 4. Smoke test first

Two minutes, one model, one cell per engine. It proves the whole pipeline —
measurement, analysis, figures, JSON, CSV, report — before you spend an hour on the
full matrix.

```python
!python benchmark.py --smoke --output-dir /kaggle/working/bench_smoke
```

## 5. The full run

```python
!python benchmark.py --output-dir /kaggle/working/benchmark_results
```

That runs every locked model against every engine the host can execute, at batch
1/2/4/8/16, prompt 32/256/1024, output 32/128/512, with correctness validation and the
compile-once probe. Expect roughly 1.5-3 hours on T4 x2, most of it Phi-3.5-mini.

Kaggle interactive sessions time out; if you are near the limit, either **Save & Run
All** to run it as a batch job, or narrow the matrix:

```python
# A shorter run that still produces every section of the report
!python benchmark.py \
    --batch-sizes 1,2,4,8 \
    --prompt-tokens 32,256 \
    --output-tokens 32,128 \
    --measure-iters 5 \
    --output-dir /kaggle/working/benchmark_results
```

If a session dies part-way, `--resume` reuses the raw records already written and
measures only what is missing:

```python
!python benchmark.py --resume --output-dir /kaggle/working/benchmark_results
```

## 6. Read the report

```python
from IPython.display import Markdown, display
display(Markdown(open(
    "/kaggle/working/benchmark_results/reports/BENCHMARK_REPORT.md", encoding="utf-8"
).read()))
```

The figures are separate PNGs:

```python
from IPython.display import Image, display
import pathlib
for path in sorted(pathlib.Path("/kaggle/working/benchmark_results/graphs").glob("*.png")):
    print(path.name)
    display(Image(str(path)))
```

The tables as a dataframe:

```python
import pandas as pd
pd.read_csv("/kaggle/working/benchmark_results/benchmark_results.csv")
pd.read_csv("/kaggle/working/benchmark_results/benchmark_comparisons.csv")
```

## 7. Keep the results

`/kaggle/working` is downloadable from the notebook's Output tab, and everything is
already there. To bring the numbers back into the repository:

```python
!cp -r /kaggle/working/benchmark_results /kaggle/working/aether/benchmark_results
```

Then commit that directory from your own machine, or zip it from the Output tab.

## Notes specific to this hardware

- **T4 has no native bf16.** The suite still selects bf16, because it is the only
  precision at which the published checkpoints and Aether's compiled artifact hold
  identical values — a same-weights comparison is worth more than a faster one. bf16
  arithmetic is emulated there and every engine pays that equally. `--precision fp16`
  gives the hardware-native comparison instead, and the report then labels Aether's
  bf16 weight residency as a representation difference.
- **Phi-3.5-mini is 3.8B.** At 16-bit weights it fits one T4 with room for a KV cache,
  but large batches may run out of memory. That is recorded per cell as `OOM` and the
  run continues.
- **Two GPUs are visible.** Aether shards only when a model does not fit the smallest
  device, so these models run single-device by default. Add `--engines` and the
  existing `benchmark/run_benchmark.py --mode multigpu` for the sharded comparison.
- **Kaggle is shared and virtualized.** Clocks and thermal state are not under the
  benchmark's control. Temperatures before and after are recorded so the report shows
  what happened rather than asserting parity.

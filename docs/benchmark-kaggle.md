# Running the multi-engine benchmark on Kaggle

A start-to-finish runbook for `python benchmark.py` on a Kaggle GPU notebook
(2x Tesla T4, 4 vCPU, ~31 GiB RAM). Every numbered step is one cell.

Read [What can actually run on 2x T4](#what-can-actually-run-on-2x-t4) first. Four of
the thirteen engines cannot run on this hardware at all, and the suite says so with the
capability that decided it rather than pretending otherwise.

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

## 3. Install the competing engines

Order matters. vLLM moves `torch` and `transformers`, so it goes first; the version pin
goes last so it wins.

```python
# vLLM. Large download, ~5 minutes. Moves torch/transformers, so install it first.
!pip install -q vllm
```

```python
# ONNX Runtime (CUDA) and OpenVINO, through optimum's exporters.
!pip install -q "optimum[onnxruntime-gpu]" "optimum[openvino]"
```

```python
# The version pin that makes the ONNX exporter work.
#
# Kaggle ships transformers 5.x. optimum's ONNX exporter declares
# transformers<4.58 and imports a symbol (get_parameter_dtype) that 5.x removed, so on
# the stock image ONNX Runtime fails at load with an ImportError. Every engine in the
# run has to share one transformers version, so the fix is to pin it into the range the
# exporter accepts before measuring - not to upgrade it afterwards.
!pip install -q "transformers==4.57.1"
```

```python
# DeepSpeed's kernel injection. Optional: it only has policies for some
# architectures, and the suite reports NOT_SUPPORTED (with that reason) where it has
# none rather than shipping a duplicate of the eager baseline.
!pip install -q deepspeed
```

**Restart the session now** (Run → Restart session), so every engine is measured
against one set of libraries. Then `%cd /kaggle/working/aether` again.

### Optional: llama.cpp

llama.cpp executes GGUF, not the published checkpoint, so it needs a conversion. This
adds roughly 20 minutes and the pip build is CPU-only unless you build with CUDA.

```python
!pip install -q llama-cpp-python gguf
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
```

Then add these flags to the benchmark command in step 6:

```
--gguf-convert-script /kaggle/working/llama.cpp/convert_hf_to_gguf.py
```

The suite converts each checkpoint to an **F16** GGUF — unquantized, so llama.cpp holds
a 16-bit rendering of the same weights as everything else and the comparison stays
same-representation. Supply your own quantized GGUF with
`--gguf-map <model>=<path.gguf>` instead if you want the quantized configuration; it is
then labelled `REPRESENTATION_DIFFERENCE` everywhere it appears.

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

The suite attempts thirteen engines. On this hardware:

| Engine | On 2x T4 | Why |
| --- | --- | --- |
| `transformers` | runs | — |
| `pytorch_native` | runs | — |
| `torch_compile` | runs | Whole-graph capture usually fails on `generate`; the suite falls back to graph-broken compilation and records which configuration compiled |
| `onnxruntime` | runs **with the transformers pin** | The exporter needs transformers<4.58 |
| `openvino` | runs, on **CPU** | OpenVINO has no CUDA target; the row is labelled with its device |
| `llama_cpp` | runs if you install it and supply a conversion | Executes GGUF, not the checkpoint |
| `vllm` | runs **at fp16** | vLLM refuses bf16 below compute capability 8.0 |
| `deepspeed` | runs only where it has an injection policy | Reports `NOT_SUPPORTED` where it has none, rather than duplicating the eager baseline |
| `aether` | runs | — |
| `sglang` | **NOT_APPLICABLE** | Attention kernels need capability 8.0+; T4 is 7.5 |
| `tensorrt_llm` | **NOT_APPLICABLE** | Published wheels target 8.0+ |
| `exllamav2` | **NOT_APPLICABLE** | Needs EXL2 weights, which do not exist for these checkpoints |
| `mlc` | **NOT_APPLICABLE** | Needs a TVM-compiled model, which this harness does not build |

Those four are not failures and are not zeros. They appear in the compatibility table
with the capability or the missing artifact that decided them.

## Two things specific to T4 worth knowing

**Precision is fp16, not bf16.** T4 is compute capability 7.5 and has no bf16 tensor
cores. Recent torch answers `is_bf16_supported() == True` there anyway, because it can
emulate the format in software — which is why an earlier version of this suite chose
bf16 and vLLM then refused to start. The suite now derives bf16 support from the
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

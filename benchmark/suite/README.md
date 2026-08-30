# Multi-engine inference benchmark

One command measures Aether Runtime against the field of inference stacks on the
same models, the same weights, the same hardware and the same workload:

```bash
python benchmark.py
```

The suite answers one question: **given everything else held fixed, how does Aether
compare against competing inference stacks — and exactly where does it lose?**

## What is compared

The field is thirteen stacks, and they are not all the same kind of system. The
report classifies each one, because a comparison only means what it says if the
reader knows what is being compared.

| Engine | What it is | Build phase | What the build leaves behind |
| --- | --- | --- | --- |
| `transformers` | inference framework, runtime (the reference) | no | nothing |
| `pytorch_native` | runtime, execution engine (hand-written decode loop) | no | nothing |
| `torch_compile` | JIT + graph compiler, kernel optimizer | yes | machine-local code cache |
| `onnxruntime` | runtime, graph compiler, kernel optimizer | yes | portable directory |
| `openvino` | AOT + graph compiler, runtime | yes | portable IR directory |
| `llama_cpp` | native runtime, quantized engine | yes | portable GGUF file |
| `vllm` | serving engine, runtime | yes | machine-local code cache |
| `sglang` | serving engine, runtime | yes | machine-local code cache |
| `tensorrt_llm` | AOT compiler, kernel optimizer, serving engine | yes | GPU-specific engine plan |
| `deepspeed` | kernel optimization system | yes | nothing (per process) |
| `exllamav2` | quantized inference engine | yes | portable EXL2 directory |
| `mlc` | AOT compiler (TVM), quantized engine | yes | portable compiled library |
| `aether` | **AOT compiler + runtime (the subject)** | yes | portable `.aeg` artifact |

Nothing is called a compiler that is not one. Transformers and the native PyTorch
loop interpret the checkpoint on every forward pass; vLLM and SGLang are serving
systems whose advantage is scheduling, not compilation.

## Models

Fixed in `benchmark/config.py` by the benchmark's charter and by the hardware budget
it was chosen for:

- `HuggingFaceTB/SmolLM2-135M-Instruct`
- `Qwen/Qwen3-0.6B`
- `SummerSigh/GPTNeo350M-Instruct-SFT`
- `microsoft/Phi-3.5-mini-instruct`

`--models` can only narrow that list. Results from two different model sets are not
comparable, so adding to it is a deliberate edit in two places (the tuple and the
test that pins it).

## How fairness is enforced

- **Prompts are built once per model, before any engine starts**, to an exact token
  count with the model's own tokenizer, and handed to every engine as the identical
  string. Each engine's tokenizer is then checked against the builder's, and any
  disagreement is printed in the compatibility table.
- **Precision is resolved from the hardware and disclosed with its reason.** On CUDA
  it is bf16: the checkpoints and Aether's artifact are both bf16, so it is the only
  precision at which every engine holds identical values. Pass `--precision fp16` for
  a hardware-native comparison, and Aether's bf16 weight residency is then reported
  as a representation difference.
- **Every engine is asked for a fixed number of tokens** with early stopping
  suppressed, so none can appear faster by generating less.
- **Threads are pinned** (OMP, MKL, OpenBLAS, NumExpr and torch) to the physical core
  count for every engine, and recorded.
- **Caches that would answer instead of computing are disabled**: Aether's semantic
  response cache and SGLang's prefix cache, both through public flags, both recorded.
  The suite issues one prompt repeatedly, so either would time a lookup.
- **Each engine runs in its own process**, one at a time. Failure isolation, real
  cold starts, and peak memory attributable to exactly one engine.
- **Engine order rotates per model**, so thermal drift over a long run cannot always
  penalize the same engine. The order used is recorded.

## How a missing measurement is handled

Never as a zero. Every cell carries one of `MEASURED`, `NOT_INSTALLED`,
`NOT_SUPPORTED`, `NOT_APPLICABLE`, `FAILED`, `OOM`, `SKIPPED`, and the reason. Charts
omit the point and name the engine as unavailable on the panel; the CSV keeps the row
with empty metric columns so a spreadsheet cannot average it in as zero.

## Outputs

```
benchmark_results/
├── benchmark_results.json        # raw payload plus every derived comparison
├── benchmark_results.csv         # one row per (engine, model, cell), measured or not
├── benchmark_comparisons.csv     # the win/loss matrix
├── raw/                          # per-worker records, the plan, the prompts
├── graphs/                       # figures
└── reports/BENCHMARK_REPORT.md   # the report
```

## Useful invocations

```bash
python benchmark.py                          # everything the host can run
python benchmark.py --smoke                  # smallest real run; proves the pipeline
python benchmark.py --engines aether,transformers,vllm
python benchmark.py --models Qwen/Qwen3-0.6B --batch-sizes 1,2,4
python benchmark.py --resume                 # reuse raw records already on disk
python benchmark.py --precision fp16         # hardware-native instead of weight-exact
python benchmark.py --gguf-map Qwen/Qwen3-0.6B=/path/model-f16.gguf   # enable llama.cpp
```

`--help` lists every switch, including the per-engine options.

## The compile-once question

Asked with a measurement rather than an argument. After the main run, a **brand-new
OS process** is started for every engine that claims a persistent build, holding
nothing but what the first process wrote to disk, and asked to load the artifact and
run once. The report prints, per engine: build time, artifact size, second-process
reload time, first inference after reload, the total cost of N requests cold and warm,
and the request count at which a build pays for itself against each competitor.

An engine with nothing to reuse says so in the same table.

# Multi-engine inference benchmark

One command measures a field of inference stacks against each other on the same
models, the same weights, the same hardware and the same workload:

```bash
python benchmark.py
```

The suite answers one question: **given everything else held fixed, how do these
inference stacks compare?**

It is engine-neutral by construction. Every engine is scored by the same code, the
pairwise matrix is computed for every ordered pair, and the standings, rankings and
per-engine win/loss sections come from one loop over the engine list. Aether is in the
field because the suite lives in Aether's repository; that is not why any number comes
out the way it does, and where Aether loses the report says so in the same words it uses
when Aether wins.

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
- **Precision is the widest 16-bit format the whole field can execute**, resolved from
  the device's compute capability and disclosed with its reason. bf16 tensor cores start
  at capability 8.0; below that the suite chooses fp16, because engines in this field
  refuse bf16 on older cards and choosing it would exclude them rather than measure them.
  Comparability is then judged on compute precision and storage width, and a 16-bit
  storage difference (fp16 tensors against Aether's bf16 artifact, both derived from the
  same published bf16 checkpoint) is printed beside every comparison it affects.
  `--precision bf16` gives the bit-exact configuration where the hardware allows it.
- **Every engine sees the same number of accelerators**, one by default. Enforced by
  restricting device visibility in each worker before any CUDA context exists, so no
  engine's placement logic is modified - each simply finds one device. Without it, a
  runtime that shards a model is measured on more hardware than one that does not, which
  compares machines rather than engines. `--devices 2` measures multi-device execution
  deliberately.
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
- **Every failure carries its reason in full.** An engine that could not run reports the
  compute capability, the version conflict or the missing artifact that decided it -
  never a truncated traceback, and never a zero.

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
python benchmark.py --precision bf16         # weight-exact, on hardware with bf16 cores
python benchmark.py --devices 2               # measure multi-device execution deliberately
python benchmark.py --focus vllm              # long-form drill-down for one engine only
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

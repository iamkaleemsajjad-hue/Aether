# Getting started with Aether

The [complete installation and execution guide](../GETTING_STARTED.md) covers
CPU and accelerator execution, precision, streaming, benchmarking, and serving.

## Install and verify

```bash
python -m pip install aether-runtime
aether version
aether doctor --json
aether hardware detect --json
```

The base package compiles supported SafeTensors models and executes CPU AEGs
without PyTorch or Transformers. Native CPU kernels use a C++ compiler when
available; otherwise the NumPy reference path remains functional. Install
`aether-runtime[pytorch]` for portable accelerator AEG execution and
`aether-runtime[server]` for API serving.

## Compile, inspect, and run

```bash
aether compile HuggingFaceTB/SmolLM2-135M-Instruct --target cpu_avx2 --output smollm2.aeg
aether inspect smollm2.aeg
aether run smollm2.aeg --prompt "Explain quantum entanglement." --max-tokens 128
aether run smollm2.aeg --prompt "Tell me a story." --max-tokens 128 --stream
aether bench smollm2.aeg --max-tokens 128 --output results.json
```

Use `cpu_neon` on ARM. Use the accelerator target reported by hardware
detection when requesting an accelerator variant. Automatic selection prefers a
ready GPU compatible with the artifact's executable or portable contracts and
otherwise uses CPU. A CPU-only contract stays on CPU. Explicit backend or device
requests constrain selection. Repeat `--target` for multiple variants.

Add `--precision q4_k_m` or `--precision q8_0` to request lossy layer
quantization explicitly, and evaluate model quality afterward. Automatic
precision assignment is the default. `aether compile --help` lists the
supported precision choices.

## Python

```python
from aether import Compiler, CompilerConfig, Runtime

artifact = Compiler(CompilerConfig(targets=["cpu_avx2"])).compile(
    "./my-model-directory", output_path="model.aeg"
)
response = Runtime().generate(
    str(artifact.root), "Hello!", max_tokens=64, apply_chat_template=True,
)
print(response.text)
print(response.metrics.to_dict())
```

Use `Runtime.generate_stream()` for text deltas. Direct backend access is
available through `from aether.backends import GenerationRequest, get_backend`.

## Serving and tests

```bash
python -m pip install "aether-runtime[server]"
aether serve smollm2.aeg --host 127.0.0.1 --port 8080
```

For an editable checkout:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The test extra includes pytest-timeout and API test dependencies. Optional
framework and hardware tests skip when their dependencies are absent.

See the [runtime](runtime.md), [API reference](api-reference.md), and
[execution contracts](research-backed-runtime.md) for details.

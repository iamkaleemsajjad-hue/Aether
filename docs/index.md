# Aether Runtime Documentation

Welcome to the Aether Runtime documentation. Aether is the compiler for AI models — not another inference wrapper. It compiles any model into a portable **Aether Execution Graph (AEG)** and runs it on any hardware with maximum performance.

## Getting Started

```{toctree}
:maxdepth: 2
:hidden:

getting-started
architecture
aeg-format
optimizer-passes
runtime
api-reference
research
roadmap
```

- [Quick start guide](getting-started.md)
- [Architecture overview](architecture.md)
- [AEG format specification](aeg-format.md)
- [Compiler optimizer passes](optimizer-passes.md)
- [Runtime and serving](runtime.md)
- [API reference](api-reference.md)
- [Research foundation](research.md)
- [Roadmap](roadmap.md)

## What is Aether?

Aether is an open-source AI model compiler. It takes any model format (SafeTensors, GGUF, ONNX, MLX, PyTorch) and produces a compiled AEG artifact that can run on:

- NVIDIA GPUs (V100, A100, RTX 4090, H100, B200)
- Apple Silicon (M1/M2/M3/M4/M5)
- AMD GPUs (RX 7000, MI300X)
- Intel NPUs and CPUs (OpenVINO, AVX-512)
- ARM CPUs (NEON)

The `.aeg` file is versioned, content-addressed, and stable: a model compiled today runs on all future Aether versions.

## One-line install

```bash
pip install aether-runtime
```

## Compile and run

```bash
aether compile Qwen/Qwen3-0.6B
aether run Qwen/Qwen3-0.6B
```

## Python SDK

```python
from aether import Runtime

rt = Runtime()
response = rt.generate("Qwen/Qwen3-0.6B", "Explain the AEG format.")
print(response.text)
print(response.metrics.throughput_tps)
```

## OpenAI-compatible server

```bash
aether serve Qwen/Qwen3-0.6B --port 11434
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="aether")
print(client.chat.completions.create(model="Qwen/Qwen3-0.6B", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

## Community

- GitHub: [github.com/iamkaleemsajjad-hue/Aether](https://github.com/iamkaleemsajjad-hue/Aether)
- Docs: [project documentation](https://github.com/iamkaleemsajjad-hue/Aether/tree/main/docs)

## License

Aether Runtime is released under the Apache License 2.0.

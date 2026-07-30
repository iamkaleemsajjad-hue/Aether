# Getting Started with Aether

This guide walks you through installing Aether, compiling your first model, and running it on your hardware.

## Installation

Install the base package:

```bash
pip install aether-runtime
```

### Optional backends

Install only the backend you need:

```bash
# NVIDIA
pip install aether-runtime[vllm]

# Apple Silicon
pip install aether-runtime[mlx]

# ONNX Runtime / Intel
pip install aether-runtime[onnxruntime]

# All development tools
pip install aether-runtime[dev]
```

## Verify installation

```bash
aether version
```

You should see the Aether version printed.

## Compile a model

Compile a small model from HuggingFace:

```bash
aether compile Qwen/Qwen3-0.6B
```

Aether will:

1. Download the model weights from HuggingFace.
2. Detect the architecture.
3. Trace the graph and produce AEG-IR.
4. Run the six optimizer passes.
5. Select backend plans for your hardware.
6. Save a `.aeg` artifact to your local cache.

This may take a few minutes on first run. Subsequent runs are instant if the AEG is cached.

## Inspect the compilation plan

Before compiling, you can preview what Aether will do:

```bash
aether compile Qwen/Qwen3-0.6B --dry-run
```

The plan shows:

- Target hardware
- Fusion opportunities
- Expected memory usage
- Estimated compile time
- Recommended backend

## Run a model

```bash
aether run Qwen/Qwen3-0.6B
```

Or pass a prompt directly:

```bash
aether run Qwen/Qwen3-0.6B --prompt "What is the AEG format?" --max-tokens 128
```

## Use the Python SDK

```python
from aether import Runtime

rt = Runtime()
response = rt.generate(
    model_id="Qwen/Qwen3-0.6B",
    prompt="Explain the AEG format in one paragraph.",
    max_tokens=128,
)
print(response.text)
print(f"TPS: {response.metrics.throughput_tps}")
print(f"TTFT: {response.metrics.ttft_ms}ms")
```

## Serve a model

Start the OpenAI-compatible server:

```bash
aether serve Qwen/Qwen3-0.6B --port 11434
```

Then use any OpenAI SDK client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="aether")
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Next steps

- Read the [architecture overview](architecture.md)
- Learn about the [AEG format](aeg-format.md)
- Explore the [optimizer passes](optimizer-passes.md)
- Check the [API reference](api-reference.md)

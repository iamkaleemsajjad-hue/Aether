# Aether Runtime — Supported Model Matrix

Aether detects model architecture from `config.json`, GGUF headers, SafeTensors metadata,
or structural graph analysis — **not** from the model name. Any model whose architecture
is recognized by the detector can be compiled to an AEG artifact and executed anywhere.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Detection + ingestion + AEG compile + runtime execution — validated by test suite |
| 🟡 | Detection + ingestion + AEG compile — runtime execution not yet checkpoint-validated |
| 🔍 | Detection only — full ingestion/execution requires real checkpoint fixture |
| ❌ | Not yet supported |

---

## Primary Model Families

| Family | Representative Models | Detect | Compile | Run | Architecture |
|--------|-----------------------|:------:|:-------:|:---:|-------------|
| **Llama 3.x** | Llama-3.1-8B, 3.2-1B, 3.3-70B | ✅ | ✅ | 🟡 | GQA + SwiGLU + RMSNorm + RoPE |
| **Qwen 3** | Qwen3-0.6B, 1.5B, 8B, 32B, 72B | ✅ | ✅ | 🟡 | GQA + QKNorm + SwiGLU + YaRN RoPE |
| **Qwen 2.5 / 2** | Qwen2.5-7B, Qwen2-72B, Qwen2-VL | ✅ | ✅ | 🟡 | GQA + SwiGLU |
| **Mistral 7B** | Mistral-7B-v0.1, v0.2, v0.3 | ✅ | ✅ | 🟡 | GQA + SwiGLU + RMSNorm |
| **Mixtral** | Mixtral-8x7B, Mixtral-8x22B | ✅ | ✅ | 🟡 | GQA + SwiGLU + MoE (8 experts, top-2) |
| **Gemma 2** | Gemma-2-2B, 9B, 27B | ✅ | ✅ | 🟡 | MQA + GeGLU + RMSNorm |
| **Gemma 3** | Gemma-3-4B, 12B, 27B | ✅ | 🟡 | 🟡 | MQA + GeGLU + RMSNorm |
| **DeepSeek V3** | DeepSeek-V3 (685B MoE) | ✅ | 🟡 | 🔍 | MLA + MoE (256 experts, top-8) |
| **DeepSeek R1** | DeepSeek-R1-671B | ✅ | 🟡 | 🔍 | MLA + MoE (256 experts, top-8) |
| **Phi-3 / Phi-4** | Phi-3 (3.8B), Phi-4 (14B) | ✅ | ✅ | 🟡 | GQA + GELU + LayerNorm |
| **Falcon** | Falcon-7B, Falcon-40B, Falcon-180B | ✅ | 🟡 | 🔍 | MQA + GELU + LayerNorm |
| **BERT** | BERT-base, BERT-large (uncased/cased) | ✅ | ✅ | ✅ | Bidirectional + GELU |
| **RoBERTa** | RoBERTa-base, RoBERTa-large | ✅ | ✅ | 🟡 | Bidirectional + GELU |
| **DeBERTa** | DeBERTa-v3-base, large, xlarge | ✅ | ✅ | 🟡 | Disentangled attention |
| **ELECTRA** | ELECTRA-base, large | ✅ | 🟡 | 🔍 | Bidirectional + GELU |
| **ALBERT** | ALBERT-base-v2, large, xlarge | ✅ | 🟡 | 🔍 | Shared layers + GELU |
| **GPT-2** | GPT-2 (117M, 345M, 762M, 1.5B) | ✅ | ✅ | 🟡 | MHA + GELU + absolute position |
| **GPT-NeoX** | GPT-NeoX-20B, Pythia (70M–12B) | ✅ | 🟡 | 🔍 | MHA + GELU + RoPE |
| **Mamba / Mamba-2** | Mamba-3-7B, Mamba-2 | ✅ | 🟡 | 🔍 | Selective scan SSM |
| **RWKV-7** | RWKV-7 (1.5B, 3B, 7B) | ✅ | 🟡 | 🔍 | Linear attention + state space |
| **Jamba** | Jamba-1.5-mini, Jamba-1.5-large | ✅ | 🟡 | 🔍 | Hybrid SSM + Attention + MoE |
| **T5 / mT5 / FLAN-T5** | T5-base through T5-11B | ✅ | 🟡 | 🔍 | Encoder-decoder + relative attention |
| **BART** | BART-base, BART-large | ✅ | 🟡 | 🔍 | Encoder-decoder |
| **Whisper** | Whisper tiny/base/small/medium/large-v3 | ✅ | 🟡 | 🔍 | Conv + cross-attention |

---

## GGUF Format

Any GGUF file whose `general.architecture` field maps to a known family is directly supported.
Architecture is read from the GGUF header — the filename is ignored.

| Quantization | Support |
|-------------|---------|
| Q4_0, Q4_1 | ✅ Detected + dequantized |
| Q4_K_M, Q4_K_S | ✅ Detected + dequantized |
| Q8_0 | ✅ Detected + dequantized |
| Q5_K_M | ✅ Detected + dequantized |
| F16, F32 | ✅ Detected + loaded |

---

## Vision-Language Models

| Family | Models | Support |
|--------|--------|---------|
| LLaVA | LLaVA-1.5, LLaVA-NeXT | 🟡 Detection + planning |
| InternVL | InternVL-2, InternVL-2.5 | 🟡 Detection + planning |
| PaliGemma | PaliGemma-3B, 10B | 🟡 Detection + planning |
| Qwen2-VL / Qwen3-VL | Qwen2-VL-7B, Qwen2.5-VL-7B | 🟡 Detection + planning |
| Pixtral | Pixtral-12B, Pixtral-Large | 🟡 Detection + planning |

---

## Generic Decoder Family (100+ aliases)

The following models are detected through the capability-driven `generic_decoder_family` path.
Architecture dimensions come from the model's `config.json`; the runtime contract
is the standard decoder (RMSNorm/LayerNorm, GQA/MQA/MHA, RoPE/ALiBi, gated FFN).

OLMo / OLMoE · Granite / Granite-Code / Granite-3 / Granite-4 · Command R / Command A / Command R+ ·
Yi / Yi-Coder · InternLM-2 / InternLM-3 · MiniCPM / MiniCPM-3 · SmolLM / SmolLM-2 ·
StarCoder / StarCoder2 · Code Llama / CodeGemma / CodeQwen / DeepSeek-Coder / Codestral ·
CodeGeeX · WizardCoder / WizardLM · BLOOM / BLOOMZ · MPT · GPT-J / GPT-OSS ·
RedPajama · OpenELM · StableLM · Pythia · OPT · XGen · Vicuna ·
GLM-4 / GLM-5 / ChatGLM · Kimi / Kimi-K2 · Hunyuan · MiniMax ·
EXAONE · HyperCLOVA X · Solar · JAIS · SeaLLM · Aya / Aya Expanse ·
Nous Hermes · OpenChat · Zephyr · Dolphin · Tulu · TinyLlama ·
MobileLLM / MobileLLM-2 / MobileLLM-3 · BitNet b1.58 · Liquid LFM ·
RecurrentGemma · Nemotron / Nvidia-Nemotron · Megatron ·
Arctic / Snowflake Arctic · Grok · Apertus · Sarvam · StepFun / Step-1 ·
DBRX · Zamba / Zamba-2 · Hymba · Falcon-H1 · Bamba

---

## Execution Paths

| Execution Path | Framework | Supported Hardware |
|---------------|-----------|-------------------|
| `aether_cpu` | Pure NumPy + native C++ kernels (no PyTorch) | Any CPU with AVX2/AVX-512/NEON |
| `vllm` | vLLM | NVIDIA CUDA, AMD ROCm |
| `mlx` | Apple MLX | Apple Silicon (M1–M5) |
| `onnxruntime` | ONNX Runtime | CPU, OpenVINO, QNN, RISC-V |
| `tensorrt-llm` | TensorRT-LLM | NVIDIA CUDA (sm80+) |
| `llama.cpp` | llama.cpp shared library | CPU, Metal, CUDA |
| `bitnet.cpp` | bitnet.cpp | CPU (AVX2/NEON ternary), FPGA |

---

## Notes

- Detection means the architecture is recognized and `ModelArchitecture` is populated correctly.
- Compilation means the 5-stage compiler pipeline produces a valid `.aeg/` artifact.
- Runtime execution means the AEG can be loaded and decoded via at least one execution path.
- A `config.json` in a local checkpoint directory is always authoritative over the name-based lookup.
- Families marked 🔍 fail with a clear error message — they never silently produce synthetic weights.
- For multi-GPU deployment, the AEG artifact contains pre-computed sharding plans weighted by VRAM for 1–8 GPUs (Pass 6: Parallelism Discovery).

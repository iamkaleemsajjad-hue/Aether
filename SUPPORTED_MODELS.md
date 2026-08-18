# Aether Runtime model support matrix

This matrix describes executable support that is currently validated by the
repository. A detector entry is not a promise that every checkpoint in a
family can run; a family is marked runnable only when ingestion, AEG packaging,
reload, and runtime execution are covered by tests.

| Family | Detection | Local ingestion | AEG/runtime execution | Validation |
|---|---:|---:|---:|---|
| BERT / SBERT / MiniLM | Yes | Yes | Yes for BERT-style encoder layouts | `tests/integration/test_encoder_aeg_roundtrip.py` |
| RoBERTa / DeBERTa / ELECTRA / ALBERT | Yes | Partial | Not fully validated on real checkpoints | architecture matrix tests |
| DistilBERT | Yes | Partial | Not fully validated; checkpoint layout differs | architecture matrix tests |
| MPNet | Yes | Partial | Not fully validated; checkpoint layout differs | architecture matrix tests |
| GPT-2 / GPT-Neo / GPT-NeoX | Yes | Partial | Decoder runtime requires checkpoint-layout validation | architecture matrix tests |
| Llama / Qwen / Mistral / Gemma / Phi / Falcon | Yes | Partial | CPU path depends on exact checkpoint and weights | ingestion/compiler tests |
| GGUF | Yes | Yes for tested quantization layouts | CPU execution depends on model architecture | GGUF test suite |
| MoE | Yes | Partial | Real Mixtral/DeepSeek execution remains unverified | MoE tests |
| Mamba / RWKV / hybrid SSM | Yes | Partial | Real model execution remains unverified | specialised-loader tests |
| VLM / video / diffusion | Detection and planning only | Partial | Not production-validated | specialised-loader/pass tests |

Unsupported or hardware-dependent families must fail with an explicit error;
they must not silently produce synthetic weights or fabricated model output.

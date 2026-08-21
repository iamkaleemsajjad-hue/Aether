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

## Family-name and architecture-variant coverage

The architecture detector also recognizes the complete public compatibility
alias set requested by the PRDs: Qwen, DeepSeek, Llama, Gemma, Mistral,
Mixtral, Phi, OLMo/OLMoE, Falcon, Command R/A, Granite and Granite Code/3/4,
Yi, InternLM, InternVL, MiniCPM, SmolLM, Pythia, GPT-2/Neo/J/NeoX/OSS,
BLOOM/BLOOMZ, MPT, RedPajama, OpenELM, StableLM, BERT/T5/FLAN-T5/mT5/ByT5/
UL2, ALBERT, RoBERTa, DeBERTa, XLNet, ELECTRA, RWKV, Mamba, Jamba,
StarCoder/2, Code Llama/Gemma/Qwen, DeepSeek-Coder, Codestral, CodeGeeX,
WizardCoder/LM, Vicuna, XGen, OPT, GLM/4/5, Kimi/K2, Hunyuan, MiniMax,
EXAONE, HyperCLOVA X, Solar, DBRX, Grok, Apertus, Sarvam, StepFun/Step-1,
Nemotron, Megatron, Arctic, Jais, SeaLLM, Aya/Expanse, Nous Hermes,
OpenChat, Zephyr, Dolphin, Tulu, TinyLlama, MobileLLM/2/3, BitNet, Liquid,
RecurrentGemma, and related aliases.

This is dispatch coverage, not a claim that every checkpoint layout is
already validated. A local `config.json` and its actual tensor names remain
authoritative. The current validated executable paths are the native AEG CPU
engine and the optional PyTorch portable engine; real checkpoint fixtures are
still required before marking a family production-validated.

For a multi-device mesh, the planner and dense decoder tensor-parallel engine
use one model copy, lossless tensor partitions, and collectives for activation
reduction/gather. Encoder, seq2seq, MLA, and state-space specialized engines
currently fail closed rather than pretending to support an unimplemented
cross-device state contract.

Unsupported or hardware-dependent families must fail with an explicit error;
they must not silently produce synthetic weights or fabricated model output.

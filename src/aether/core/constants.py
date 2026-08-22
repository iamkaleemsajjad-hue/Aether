"""
Aether core constants — version strings, target IDs, defaults, and architecture patterns.

This module defines all version strings, valid target identifiers, supported model
architecture patterns, default directories, and other runtime-visible constants
shared across the entire Aether codebase.
"""

from __future__ import annotations

from typing import ClassVar

# ── Aether version ────────────────────────────────────────────────────────────
# NOTE: the Aether software version and the AEG format version are SEPARATE
# version namespaces and are intentionally not synchronized (see
# docs/PRD_COMPLIANCE_MATRIX.md § AEG versioning). Aether 1.2.0 reads and
# writes AEG/1.1 by default (AEG/2.0 and AEG/3.0 when v4/v5 optimizer passes
# are applied).

AETHER_VERSION: str = "1.2.0"
"""Current version of the Aether Runtime package (matches pyproject.toml)."""

AETHER_VERSION_TUPLE: tuple[int, int, int] = (1, 2, 0)
"""Machine-readable version tuple (major, minor, patch)."""

# ── AEG format version ─────────────────────────────────────────────────────────

AEG_FORMAT_VERSION: str = "AEG/1.1"
"""Current v3.1 AEG format version embedded in every manifest."""

AEG_FORMAT_VERSION_MAJOR: int = 1
"""AEG format major version. Changed only on breaking changes."""

AEG_FORMAT_VERSION_MINOR: int = 1
"""AEG format minor version. Incremented for backward-compatible additions."""

AEG_MINIMUM_COMPATIBLE_VERSION: str = "AEG/1.0"
"""Oldest AEG format version the current runtime can read."""

AEG_SUPPORTED_FORMAT_VERSIONS: tuple[str, ...] = ("AEG/1.0", "AEG/1.1", "AEG/2.0", "AEG/3.0")
"""Format versions understood by this runtime's package reader."""

# ── File extensions ────────────────────────────────────────────────────────────

AEG_FILE_EXTENSION: str = ".aeg"
"""Standard file extension for compiled AEG artifacts."""

AEG_IR_FILE_EXTENSION: str = ".aeg-ir"
"""Extension for standalone AEG-IR graph files."""

AEG_QUANT_FILE_EXTENSION: str = ".aeg-quant"
"""Extension for AEG weight files."""

AEG_MANIFEST_FILENAME: str = "manifest.json"
"""Canonical name of the AEG manifest file inside a package."""

AEG_GRAPH_FILENAME: str = "computation_graph.aeg-ir"
"""Canonical name of the AEG-IR graph file inside a package."""

AEG_PRECISION_MAP_FILENAME: str = "precision_map.json"
"""Canonical name of the precision map file."""

# ── Hardware target identifiers ────────────────────────────────────────────────

SUPPORTED_TARGETS: dict[str, str] = {
    # ── v3.1 NVIDIA ────────────────────────────────────────────────────────────
    "cuda_sm70": "NVIDIA V100 (Volta)",
    "cuda_sm80": "NVIDIA A100 (Ampere)",
    "cuda_sm89": "NVIDIA RTX 4090 (Ada)",
    "cuda_sm90": "NVIDIA H100 (Hopper)",
    "cuda_sm100": "NVIDIA B200 (Blackwell)",
    "cuda_sm120": "NVIDIA Rubin R100 (sm_120)",
    # ── v3.1 Apple ────────────────────────────────────────────────────────────
    "metal_m1": "Apple M1/M2",
    "metal_m3": "Apple M3/M4/M5",
    # ── v3.1 AMD ──────────────────────────────────────────────────────────────
    "rocm_rdna3": "AMD RX 7000 Series (RDNA3)",
    "rocm_cdna3": "AMD MI300X (CDNA3)",
    # ── v3.1 Intel / Qualcomm / CPU ───────────────────────────────────────────
    "openvino_npu": "Intel Arc NPU (OpenVINO)",
    "openvino_gpu": "Intel GPU (OpenVINO)",
    "qualcomm_qnn": "Qualcomm Snapdragon NPU (QNN)",
    "cpu_avx512": "x86_64 (AVX-512)",
    "cpu_avx2": "x86_64 (AVX2)",
    "cpu_neon": "ARM (NEON SIMD)",
    # ── v4.0 new targets ──────────────────────────────────────────────────────
    "cuda_sm130": "NVIDIA Rubin Ultra (sm_130, dual-core ~100 PFLOPS FP4)",
    "cuda_sm100_tee": "NVIDIA B200 Confidential Computing (CC mode)",
    "riscv_mips_s8200": "MIPS S8200 NPU (RISC-V agentic, sub-10W)",
    "riscv_sifive_x160": "SiFive Intelligence X160 (scalar+vector+matrix)",
    "riscv_xuantie_c930": "Alibaba XuanTie C930 (RISC-V + integrated NPU)",
    "fpga_xilinx_vu9p": "Xilinx VU9P FPGA (decode-only, 10x cheaper/token)",
    "amd_mi350x": "AMD MI350X (CDNA4, HBM3e, successor to MI300X)",
    "qualcomm_cloud_ai100": "Qualcomm Cloud AI 100 Ultra (data center NPU)",
    # ── v5.0 new targets ──────────────────────────────────────────────────────
    "cuda_sm100_gb300": "NVIDIA GB300 Blackwell Ultra (1.5x B200 FP4, HBM3e+)",
    "rocm_cdna5_mi455x": "AMD MI455X (CDNA5, 432 GB HBM4, 23.3 TB/s, MXFP6)",
    "cpu_avx512_ternary": "x86_64 AVX2 Ternary (BitNet b1.58, ADD-only)",
    "cpu_neon_ternary": "ARM NEON Ternary (BitNet b1.58, mobile/Apple M-series)",
    "fpga_ternary": "FPGA BTC-LLM Ternary (0.8-1.58 bit, purpose-built circuits)",
    "riscv_cervell": "Semidynamics Cervell (unified scalar/vector/tensor RISC-V NPU)",
}
"""Mapping of target IDs to human-readable hardware descriptions."""

SUPPORTED_TARGET_IDS: frozenset[str] = frozenset(SUPPORTED_TARGETS.keys())
"""Set of all valid target IDs for validation."""

BACKEND_BY_TARGET: dict[str, list[str]] = {
    # v3.1
    "cuda_sm70": ["pytorch", "tensorrt-llm"],
    "cuda_sm80": ["vllm", "pytorch", "tensorrt-llm"],
    "cuda_sm89": ["vllm", "pytorch", "tensorrt-llm"],
    "cuda_sm90": ["vllm", "pytorch", "tensorrt-llm"],
    "cuda_sm100": ["vllm", "pytorch", "tensorrt-llm"],
    "cuda_sm120": ["vllm", "pytorch", "tensorrt-llm"],
    "metal_m1": ["mlx", "llama.cpp", "pytorch"],
    "metal_m3": ["mlx", "llama.cpp", "pytorch"],
    "rocm_rdna3": ["pytorch", "llama.cpp"],
    "rocm_cdna3": ["vllm", "pytorch"],
    "openvino_npu": ["onnxruntime", "pytorch"],
    "openvino_gpu": ["onnxruntime", "pytorch"],
    "qualcomm_qnn": ["onnxruntime", "pytorch"],
    "cpu_avx512": ["aether_cpu", "llama.cpp", "onnxruntime", "pytorch"],
    "cpu_avx2": ["llama.cpp", "onnxruntime", "aether_cpu"],
    "cpu_neon": ["aether_cpu", "llama.cpp", "onnxruntime", "pytorch"],
    # v4.0
    "cuda_sm130": ["vllm", "pytorch", "tensorrt-llm"],
    "cuda_sm100_tee": ["pytorch"],   # TEE requires dedicated backend
    "riscv_mips_s8200": ["onnxruntime"],
    "riscv_sifive_x160": ["onnxruntime"],
    "riscv_xuantie_c930": ["onnxruntime", "pytorch"],
    "fpga_xilinx_vu9p": ["onnxruntime"],
    "amd_mi350x": ["vllm", "pytorch"],
    "qualcomm_cloud_ai100": ["onnxruntime", "pytorch"],
    # v5.0
    "cuda_sm100_gb300": ["vllm", "pytorch", "tensorrt-llm"],
    "rocm_cdna5_mi455x": ["vllm", "pytorch"],
    "cpu_avx512_ternary": ["aether_cpu", "bitnet.cpp", "llama.cpp"],
    "cpu_neon_ternary": ["aether_cpu", "bitnet.cpp", "llama.cpp"],
    "fpga_ternary": ["bitnet.cpp"],
    "riscv_cervell": ["onnxruntime"],
}
"""Priority-ordered backend candidates for each target."""

# ── Supported model architecture families ──────────────────────────────────────

SUPPORTED_ARCHITECTURES: dict[str, dict[str, str | bool]] = {
    "llama_family": {
        "attn": "GQA",
        "ffn": "SwiGLU",
        "norm": "RMSNorm",
        "rope": True,
        "is_moe": False,
    },
    "qwen_family": {
        "attn": "GQA_QKNorm",
        "ffn": "SwiGLU",
        "norm": "RMSNorm",
        "rope": "YaRN",
        "is_moe": False,
    },
    "gemma_family": {
        "attn": "MQA",
        "ffn": "GeGLU",
        "norm": "RMSNorm",
        "rope": True,
        "is_moe": False,
    },
    "deepseek_family": {
        "attn": "MLA",
        "ffn": "MoE",
        "norm": "RMSNorm",
        "rope": "NTK_aware",
        "is_moe": True,
    },
    "moe_family": {
        "ffn": "MoE",
        "router": "TopK",
        "is_moe": True,
    },
    "mistral_family": {
        "attn": "GQA",
        "ffn": "SwiGLU",
        "norm": "RMSNorm",
        "rope": True,
        "is_moe": False,
    },
    "phi_family": {
        "attn": "GQA",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "is_moe": False,
    },
    "falcon_family": {
        "attn": "MQA",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "is_moe": False,
    },
    "gpt_family": {
        "attn": "Causal_Self_Attention",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "rope": False,
        "is_moe": False,
    },
    "vision_family": {
        "encoder": "ViT",
        "cross_attn": True,
    },
    "whisper_family": {
        "encoder": "Conv1D_Transformer",
        "decoder": "Transformer",
        "cross_attn": True,
    },
    "hybrid_ssm_family": {
        "attn": "hybrid",
        "ssm": "selective_scan",
        "stateful": True,
        "is_moe": False,
    },
    "bert_family": {
        "attn": "Bidirectional_Self_Attention",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "rope": False,
        "is_moe": False,
        "is_encoder": True,
    },
    "roberta_family": {
        "attn": "Bidirectional_Self_Attention",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "rope": False,
        "is_moe": False,
        "is_encoder": True,
    },
    "deberta_family": {
        "attn": "Disentangled_Attention",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "rope": False,
        "is_moe": False,
        "is_encoder": True,
    },
    "electra_family": {
        "attn": "Bidirectional_Self_Attention",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "rope": False,
        "is_moe": False,
        "is_encoder": True,
    },
    "albert_family": {
        "attn": "Bidirectional_Self_Attention",
        "ffn": "GELU",
        "norm": "LayerNorm",
        "rope": False,
        "is_moe": False,
        "is_encoder": True,
    },
    # Capability-based family for standard decoder-only checkpoints whose
    # names/configs differ but whose computation contract is the ordinary
    # transformer decoder (RMSNorm/LayerNorm, MHA/GQA/MQA, RoPE/ALiBi, and a
    # gated or classic FFN).  The concrete tensor geometry still comes from
    # config.json and the checkpoint, never from this registry entry.
    "generic_decoder_family": {
        "attn": "config_declared_attention",
        "ffn": "config_declared_ffn",
        "norm": "config_declared_norm",
        "rope": "config_declared_position_encoding",
        "is_moe": False,
    },
    "encoder_decoder_family": {
        "encoder": True,
        "decoder": True,
        "cross_attn": True,
    },
}
"""Architecture detection patterns keyed by family name."""

ARCHITECTURE_BY_MODEL_PREFIX: dict[str, str] = {
    "llama": "llama_family",
    "qwen": "qwen_family",
    "gemma": "gemma_family",
    "deepseek": "deepseek_family",
    "mixtral": "moe_family",
    "mistral": "mistral_family",
    "phi": "phi_family",
    "falcon": "falcon_family",
    "whisper": "whisper_family",
    "vit": "vision_family",
    "llava": "vision_family",
    "internvl": "vision_family",
    "paligemma": "vision_family",
    "pixtral": "vision_family",
    "qwen2_vl": "vision_family",
    "qwen2_5_vl": "vision_family",
    "qwen3_vl": "vision_family",
    "mamba": "hybrid_ssm_family",
    "jamba": "hybrid_ssm_family",
    "bamba": "hybrid_ssm_family",
    "rwkv": "hybrid_ssm_family",
    "bert": "bert_family",
    "roberta": "roberta_family",
    "deberta": "deberta_family",
    "electra": "electra_family",
    "albert": "albert_family",
    "gpt2": "gpt_family",
    "gpt_neo": "gpt_family",
    "gpt_neox": "gpt_family",
    "gpt": "gpt_family",
    # Standard decoder families covered by the capability-driven path.
    "olmo": "generic_decoder_family",
    "olmoe": "generic_decoder_family",
    "command": "generic_decoder_family",
    "command_r": "generic_decoder_family",
    "command_a": "generic_decoder_family",
    "cohere": "generic_decoder_family",
    "hyperclova": "generic_decoder_family",
    "granite": "generic_decoder_family",
    "granite_code": "generic_decoder_family",
    "granite3": "generic_decoder_family",
    "granite4": "generic_decoder_family",
    "dbrx": "generic_decoder_family",
    "yi": "generic_decoder_family",
    "internlm": "generic_decoder_family",
    "minicpm": "generic_decoder_family",
    "smollm": "generic_decoder_family",
    "pythia": "generic_decoder_family",
    "gptj": "generic_decoder_family",
    "gpt_bigcode": "generic_decoder_family",
    "bloom": "generic_decoder_family",
    "mpt": "generic_decoder_family",
    "redpajama": "generic_decoder_family",
    "openelm": "generic_decoder_family",
    "stablelm": "generic_decoder_family",
    "starcoder": "generic_decoder_family",
    "starcoder2": "generic_decoder_family",
    "codegen": "generic_decoder_family",
    "codegeex": "generic_decoder_family",
    "codegemma": "generic_decoder_family",
    "codeqwen": "generic_decoder_family",
    "deepseek_coder": "generic_decoder_family",
    "codelama": "llama_family",
    "code_llama": "llama_family",
    "codestral": "generic_decoder_family",
    "wizard": "generic_decoder_family",
    "wizardcoder": "generic_decoder_family",
    "wizardlm": "generic_decoder_family",
    "vicuna": "generic_decoder_family",
    "xgen": "generic_decoder_family",
    "opt": "generic_decoder_family",
    "gpt_oss": "generic_decoder_family",
    "glm": "generic_decoder_family",
    "glm4": "generic_decoder_family",
    "glm5": "generic_decoder_family",
    "kimi": "generic_decoder_family",
    "kimi_k2": "generic_decoder_family",
    "hunyuan": "generic_decoder_family",
    "minimax": "generic_decoder_family",
    "exaone": "generic_decoder_family",
    "solar": "generic_decoder_family",
    "jais": "generic_decoder_family",
    "seallm": "generic_decoder_family",
    "aya": "generic_decoder_family",
    "aya_expanse": "generic_decoder_family",
    "nous": "generic_decoder_family",
    "nous_hermes": "generic_decoder_family",
    "openchat": "generic_decoder_family",
    "zephyr": "generic_decoder_family",
    "dolphin": "generic_decoder_family",
    "tulu": "generic_decoder_family",
    "tinyllama": "generic_decoder_family",
    "tiny_llama": "generic_decoder_family",
    "mobilellm": "generic_decoder_family",
    "mobilellm2": "generic_decoder_family",
    "mobilellm3": "generic_decoder_family",
    "liquid": "generic_decoder_family",
    "lfm": "generic_decoder_family",
    "recurrentgemma": "generic_decoder_family",
    "recurrent_gemma": "generic_decoder_family",
    "bitnet": "generic_decoder_family",
    "nemotron": "generic_decoder_family",
    "nvidia_nemotron": "generic_decoder_family",
    "megatron": "generic_decoder_family",
    "apertus": "generic_decoder_family",
    "sarvam": "generic_decoder_family",
    "step": "generic_decoder_family",
    "stepfun": "generic_decoder_family",
    "arctic": "generic_decoder_family",
    "snowflake_arctic": "generic_decoder_family",
    "grok": "generic_decoder_family",
    "nvidia": "generic_decoder_family",
    # Hybrid SSM / state space extensions
    "mamba2": "hybrid_ssm_family",
    "zamba": "hybrid_ssm_family",
    "zamba2": "hybrid_ssm_family",
    "hymba": "hybrid_ssm_family",
    "falcon_h1": "hybrid_ssm_family",
    "plamo": "hybrid_ssm_family",
    # Encoder-decoder families (must not enter decoder-only runtime)
    "t5": "encoder_decoder_family",
    "mt5": "encoder_decoder_family",
    "byt5": "encoder_decoder_family",
    "ul2": "encoder_decoder_family",
    "flan_t5": "encoder_decoder_family",
    "bart": "encoder_decoder_family",
    "mbart": "encoder_decoder_family",
    "pegasus": "encoder_decoder_family",
    "marian": "encoder_decoder_family",
    "xlnet": "bert_family",
    # Hybrid SSM / state space extensions
    "mamba2": "hybrid_ssm_family",
    "zamba": "hybrid_ssm_family",
    "zamba2": "hybrid_ssm_family",
    "hymba": "hybrid_ssm_family",
    "falcon_h1": "hybrid_ssm_family",
    "plamo": "hybrid_ssm_family",
    # Additional instruction-tuned families
    "openhermes": "generic_decoder_family",
    "capybara": "generic_decoder_family",
    "starling": "generic_decoder_family",
    "nous": "generic_decoder_family",
    "nous_hermes": "generic_decoder_family",
    # Additional code model families (2025)
    "yi_coder": "generic_decoder_family",
    "qwen_coder": "generic_decoder_family",
    "deepseek_coder_v2": "generic_decoder_family",
    "deepseek_coder_v3": "generic_decoder_family",
    "starchat": "generic_decoder_family",
    # Multilingual / regional families (2025+)
    "internlm2": "generic_decoder_family",
    "internlm3": "generic_decoder_family",
    "baichuan": "generic_decoder_family",
    "baichuan2": "generic_decoder_family",
    "chatglm": "generic_decoder_family",
    "chatglm2": "generic_decoder_family",
    "chatglm3": "generic_decoder_family",
    "glm_4": "generic_decoder_family",
    "seallm3": "generic_decoder_family",
    "typhoon": "generic_decoder_family",
    "polylm": "generic_decoder_family",
    "megrez": "generic_decoder_family",
    "telechat": "generic_decoder_family",
    "skywork": "generic_decoder_family",
    "orion": "generic_decoder_family",
    # Additional MoE entries
    "qwen2_moe": "moe_family",
    "qwen3_moe": "moe_family",
    "deepseek_r2": "deepseek_family",
    "deepseek_prover": "deepseek_family",
    # Tiny / edge models (2025)
    "smollm2": "generic_decoder_family",
    "smollm3": "generic_decoder_family",
    "openlm": "generic_decoder_family",
    "microllama": "generic_decoder_family",
}
"""Model name prefix to architecture family mapping."""

# ── Default paths ──────────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR: str = "~/.aether"
"""Default Aether cache directory. Override with AETHER_CACHE_DIR env var."""

DEFAULT_MODEL_CACHE_SUBDIR: str = "models"
"""Subdirectory under cache for downloaded AEG artifacts."""

DEFAULT_KERNEL_CACHE_SUBDIR: str = "kernels"
"""Subdirectory under cache for compiled kernel blobs."""
DEFAULT_CONFIG_SUBDIR: str = "config"
"""Subdirectory under cache for user configuration."""

DEFAULT_LOG_SUBDIR: str = "logs"
"""Subdirectory under cache for runtime logs."""

DEFAULT_HUB_CACHE_SUBDIR: str = "hub"
"""Subdirectory under cache for Hub metadata."""

# ── Aether Hub ─────────────────────────────────────────────────────────────────

DEFAULT_HUB_URL: str = "https://hub.aether.dev"
"""Default URL for the Aether Hub API."""

HUB_API_VERSION: str = "v1"
"""Hub API version string."""

HUB_UPLOAD_TIMEOUT_S: int = 300
"""Upload timeout in seconds."""

HUB_DOWNLOAD_TIMEOUT_S: int = 120
"""Download timeout in seconds."""

HUB_RETRY_ATTEMPTS: int = 3
"""Number of retries for Hub API calls."""

HUB_RETRY_BACKOFF_S: float = 2.0
"""Exponential backoff base in seconds."""

# ── Compiler defaults ──────────────────────────────────────────────────────────

DEFAULT_QUALITY_BUDGET: float = 0.02
"""Default maximum allowed perplexity increase (2%)."""

DEFAULT_CALIBRATION_DATASET: str = "wikitext-2"
"""Default calibration dataset for sensitivity analysis."""

DEFAULT_CALIBRATION_TOKENS: int = 131072
"""Default number of tokens to draw from the calibration dataset."""

DEFAULT_OPTIMIZATION_LEVEL: int = 2
"""Default optimization level: 0=none, 1=basic, 2=full, 3=aggressive."""

DEFAULT_FUSION_PASS: bool = True
"""Whether operator fusion is enabled by default."""

DEFAULT_SENSITIVITY_PASS: bool = True
"""Whether sensitivity analysis is enabled by default."""

DEFAULT_PRECISION_PASS: bool = True
"""Whether precision assignment is enabled by default."""

DEFAULT_KV_CACHE_PASS: bool = True
"""Whether KV cache structuring is enabled by default."""

DEFAULT_MOE_ROUTING_PASS: bool = True
"""Whether MoE routing optimization is enabled by default."""

DEFAULT_PARALLELISM_PASS: bool = True
"""Whether parallelism discovery is enabled by default."""

DEFAULT_REASONING_GRAPH_PASS: bool = True
"""Whether reasoning graph compilation is enabled by default."""

DEFAULT_SPARSE_ATTENTION_PASS: bool = True
"""Whether long-context sparse-attention planning is enabled by default.

The runtime activates the persisted plan only at its recorded context
threshold; ordinary short prompts therefore retain dense-reference behavior.
"""

DEFAULT_PRUNING_PASS: bool = False
"""Whether lossy pruning/sparsity masks are enabled by default.

Pruning is opt-in until task/perplexity validation certifies the generated mask.
"""

# ── PRD v4.0 compiler pass defaults ───────────────────────────────────────────

DEFAULT_MTP_HEAD_PASS: bool = False
"""Pass 10: Native MTP head compilation is opt-in."""

DEFAULT_GRAMMAR_CONSTRAINT_PASS: bool = False
"""Pass 11: Grammar constraint FSM pre-compilation — opt-in (schema required)."""

DEFAULT_MODEL_MERGING_PASS: bool = False
"""Pass 12: Model merging / task vector fusion — opt-in."""

DEFAULT_TTT_PASS: bool = False
"""Pass 13: TTT fast-weight injection — opt-in (requires fast-weight slots)."""

DEFAULT_SEMANTIC_KV_PASS: bool = False
"""Pass 14: Semantic KV compression is opt-in."""

DEFAULT_CROSS_LAYER_KV_PASS: bool = False
"""Pass 15: Cross-layer KV sharing is opt-in."""

DEFAULT_GREEN_ENERGY_PASS: bool = False
"""Pass 16: Green energy-aware compilation — opt-in (requires carbon API)."""

DEFAULT_TEE_PASS: bool = False
"""Pass 17: TEE enclave emission — opt-in (requires TEE hardware)."""

# ── PRD v5.0 compiler pass defaults ───────────────────────────────────────────

DEFAULT_MDLM_DRAFTER_PASS: bool = False
"""Pass 18: MDLM diffusion drafter compilation — opt-in."""

DEFAULT_SUB2BIT_PASS: bool = False
"""Pass 19: Sub-2-bit ternary quantization — opt-in (requires BitNet checkpoint)."""

DEFAULT_VIDEO_COMPRESSION_PASS: bool = False
"""Pass 20: Video token compression is opt-in for VLMs."""

DEFAULT_ADVANCED_PEFT_PASS: bool = False
"""Pass 21: Advanced PEFT compilation is opt-in."""

DEFAULT_RLVR_VERIFIER_PASS: bool = False
"""Pass 22: RLVR verifier head injection — opt-in (training workflow only)."""

# ── PRD v4.0 runtime defaults ─────────────────────────────────────────────────

DEFAULT_P_EAGLE_ENGINE: bool = True
"""R1: Enable P-EAGLE hardware-parallel speculative decoding."""

DEFAULT_MULTI_AGENT_KV: bool = False
"""R2: Multi-agent KV coordination — opt-in (requires session registry)."""

DEFAULT_GRAMMAR_FSM_ENGINE: bool = False
"""R3: Structured output grammar FSM engine — opt-in (requires pre-compiled FSM)."""

DEFAULT_SLO_SCHEDULER: bool = True
"""R4: SLO-aware adaptive scheduler — enabled by default."""

DEFAULT_TTT_ENGINE: bool = False
"""R5: TTT fast-weight engine — opt-in (requires compiled TTT slots)."""

DEFAULT_MCP_INTEGRATION: bool = False
"""R6: MCP native integration layer — opt-in (requires MCP server config)."""

DEFAULT_GREEN_POWER_MANAGER: bool = False
"""R7: Green inference power manager — opt-in."""

DEFAULT_TEE_RUNTIME: bool = False
"""R8: Confidential inference TEE runtime — opt-in (requires CC-mode hardware)."""

# ── PRD v5.0 runtime defaults ─────────────────────────────────────────────────

DEFAULT_DIFFUSION_SPEC_ENGINE: bool = False
"""R9: Diffusion speculative decoding engine — opt-in."""

DEFAULT_KV_TRANSFER_ENGINE: str = "nixl"
"""R10: KV network transfer engine: nixl|uccl|rdma|nvlink|cxl."""

DEFAULT_NIKA_POLICY: bool = True
"""R10: NIKA adaptive transfer-vs-recompute policy — enabled by default."""

DEFAULT_SEMANTIC_CACHE: bool = False
"""R11: Semantic request cache — opt-in."""

DEFAULT_SEMANTIC_CACHE_THRESHOLD: float = 0.92
"""R11: Cosine similarity threshold for semantic cache hit (0.0-1.0)."""

DEFAULT_SEMANTIC_CACHE_SIZE: int = 100_000
"""R11: Maximum HNSW index entries for semantic cache."""

DEFAULT_CXL_KV_POOL: bool = False
"""R12: CXL rack-scale KV pool — opt-in (requires CXL 3.0 hardware)."""

DEFAULT_CXL_POOL_SIZE_GB: int = 512
"""R12: Default CXL pool size in GB."""

# ── Runtime defaults ───────────────────────────────────────────────────────────

DEFAULT_OPTIMIZE_FOR: str = "latency"
"""Default optimization objective: latency, throughput, or quality."""

DEFAULT_SPECULATIVE_DECODING: bool = True
"""Whether tree-speculative decoding is enabled by default."""

DEFAULT_SPECULATIVE_TREE_DEPTH: int = 4
"""Default draft tree depth for speculative decoding."""

DEFAULT_PREFILL_CHUNK_SIZE: int = 2048
"""Default maximum tokens per prefill chunk."""

DEFAULT_MAX_BATCH_SIZE: int = 256
"""Default maximum number of concurrent requests per batch."""

DEFAULT_KV_CACHE_DTYPE: str = "fp8"
"""Default KV cache numeric format."""

DEFAULT_KV_CACHE_CPU_GB: int = 32
"""Default CPU DRAM budget for KV cache (GB)."""

DEFAULT_KV_CACHE_NVME_GB: int = 200
"""Default NVMe SSD budget for KV cache (GB)."""

DEFAULT_DYNAMIC_PRECISION: bool = True
"""Whether dynamic precision adjustment is enabled by default."""

DEFAULT_DISAGGREGATE_SERVE: bool = False
"""Whether disaggregated prefill/decode is enabled by default."""

DEFAULT_SERVER_PORT: int = 11434
"""Default port for the Aether REST server."""

DEFAULT_SERVER_HOST: str = "localhost"
"""Default host for the Aether REST server."""

DEFAULT_TEMPERATURE: float = 0.7
"""Default generation temperature."""

DEFAULT_MAX_TOKENS: int = 1024
"""Default maximum generation tokens."""

DEFAULT_TOP_P: float = 0.9
"""Default top-p sampling parameter."""

# ── Precision constants ────────────────────────────────────────────────────────

PRECISION_SIZES_BYTES: dict[str, float] = {
    "BF16": 2.0,
    "FP16": 2.0,
    "FP32": 4.0,
    "FP8": 1.0,
    "Q8_0": 1.0,
    "Q6_K": 0.75,
    "MXFP6": 0.75,     # 6 bits / 8 = 0.75 bytes (PRD v5.0 AMD MI455X)
    "Q4_K_M": 0.5,
    "Q4_0": 0.5,
    "Q3_K": 0.375,
    "Q3_K_S": 0.375,
    "IQ3_XS": 0.375,
    "IQ2_XXS": 0.25,
    "Q2_K": 0.25,
    "Q2_K_S": 0.25,
    "INT4": 0.5,
    "FP4": 0.5,
    "NVFP4": 0.5,
    "MXFP4": 0.5,
    "INT8": 1.0,
    "INT16": 2.0,
    # Sub-2-bit (PRD v5.0 Pass 19)
    "TERNARY": 0.25,   # 2 bits packed storage per ternary weight (4 per byte)
    "BINARY": 0.125,   # ~1 bit effective; stored as codebook index
    "NANOQ": 0.125,    # sub-1-bit trellis; stored as codebook index
}
"""Memory size in bytes per element for each precision format."""

PRECISION_BITS: dict[str, float] = {
    "BF16": 16,
    "FP16": 16,
    "FP32": 32,
    "FP8": 8,
    "Q8_0": 8,
    "MXFP6": 6,        # AMD MI455X CDNA5 native format (PRD v5.0)
    "Q6_K": 6,
    "Q4_K_M": 4,
    "Q4_0": 4,
    "Q3_K": 3,
    "Q3_K_S": 3,
    "IQ3_XS": 3,
    "IQ2_XXS": 2,
    "Q2_K": 2,
    "Q2_K_S": 2,
    "INT4": 4,
    "FP4": 4,
    "NVFP4": 4,
    "MXFP4": 4,
    "INT8": 8,
    "INT16": 16,
    # Sub-2-bit (PRD v5.0 Pass 19)
    "TERNARY": 1.58,   # log2(3) — information content of one ternary symbol
    "BINARY": 1.0,     # 1 bit per weight (binary codebook)
    "NANOQ": 0.9,      # approximate; codebook size dependent
}
"""Effective bit width of each precision format."""

# ── Sensitivity thresholds ─────────────────────────────────────────────────────

SENSITIVITY_CRITICAL_THRESHOLD: float = 0.9
"""Layers with sensitivity above this are assigned BF16."""

SENSITIVITY_HIGH_THRESHOLD: float = 0.7
"""Layers with sensitivity above this are assigned FP8 or Q6_K."""

SENSITIVITY_MEDIUM_THRESHOLD: float = 0.4
"""Layers with sensitivity above this are assigned Q4_K_M."""

# ── MoE thresholds ─────────────────────────────────────────────────────────────

MOE_HOT_THRESHOLD: float = 0.05
"""Experts with activation rate above this are hot (GPU HBM)."""

MOE_WARM_THRESHOLD: float = 0.001
"""Experts with activation rate above this are warm (CPU DRAM + prefetch)."""

MOE_HOT_ACTIVATION_LIMIT: int = 1024
"""Number of calibration tokens per expert for hot classification."""

# ── Speculative decoding ───────────────────────────────────────────────────────

DRAFT_FAMILIES: dict[str, str] = {}
"""Deprecated compatibility symbol; draft models are explicit configuration.

An installed target checkpoint does not imply a compatible draft checkpoint.
The speculative runtime therefore refuses to infer one from model-family
names.  Keep this empty mapping for callers that imported the old symbol.
"""

MINIMUM_ACCEPTANCE_RATE: float = 0.70
"""Minimum draft acceptance rate before falling back to standard decoding."""

# ── Performance and memory ─────────────────────────────────────────────────────

PREFILL_MEMORY_OVERHEAD_FACTOR: float = 1.2
"""Multiplier for prefill memory estimation."""

DECODE_MEMORY_OVERHEAD_FACTOR: float = 1.1
"""Multiplier for decode memory estimation."""

KV_CACHE_BLOCK_SIZE: int = 16
"""Number of KV slots per block in the paged cache."""

KV_CACHE_ALLOCATION_ALIGNMENT: int = 256
"""Memory alignment for KV block allocations."""

MAX_TENSOR_PARALLEL_DEGREE: int = 8
"""Maximum supported tensor parallelism degree."""

MAX_PIPELINE_PARALLEL_STAGES: int = 8
"""Maximum supported pipeline parallelism stages."""

MAX_CONTEXT_PARALLEL_DEGREE: int = 8
"""Maximum supported context parallelism degree."""

MAX_EXPERT_PARALLEL_DEGREE: int = 8
"""Maximum supported expert parallelism degree."""

# ── Bounds and limits ──────────────────────────────────────────────────────────

MAX_CONTEXT_LENGTH: int = 262144
"""Maximum supported context length in tokens."""

MAX_BATCH_SIZE_HARD_LIMIT: int = 4096
"""Hard limit on batch size regardless of configuration."""

MAX_PREFILL_CHUNK_SIZE: int = 65536
"""Maximum prefill chunk size in tokens."""

MAX_MODEL_NAME_LENGTH: int = 256
"""Maximum model name/ID length."""

MAX_CACHE_KEY_LENGTH: int = 128
"""Maximum length of a content-addressed cache key."""

# ── Error messages ─────────────────────────────────────────────────────────────

ERR_MODEL_NOT_FOUND: str = "Model '{}' not found in local cache or on Hub."
ERR_MODEL_NOT_COMPILED: str = "Model '{}' has not been compiled yet. Run `aether compile {}` first."
ERR_UNSUPPORTED_FORMAT: str = "Unsupported model format: '{}'. Supported: {}."
ERR_UNSUPPORTED_TARGET: str = "Unsupported target '{}'. Supported targets: {}."
ERR_BACKEND_NOT_FOUND: str = "No backend available for target '{}'. Install the required backend package."
ERR_TARGET_NOT_FOUND: str = "No kernel or backend plan found for target '{}'. Compile with this target first."
ERR_CACHE_MISS: str = "Cache miss for key '{}'. Kernel must be compiled locally."
ERR_AEG_VERSION_MISMATCH: str = "AEG format '{}' is not compatible with runtime version '{}'."
ERR_AEG_INTEGRITY: str = "AEG integrity check failed for file '{}'. Expected hash: {} != actual: {}."
ERR_DOWNLOAD_FAILED: str = "Failed to download '{}' from '{}'. Status: {}."
ERR_UPLOAD_FAILED: str = "Failed to upload to '{}'. Status: {}."
ERR_CALIBRATION_FAILED: str = "Calibration failed for model '{}' on dataset '{}'. {}"
ERR_SENSITIVITY_FAILED: str = "Sensitivity analysis failed at layer {}. {}"
ERR_COMPILATION_FAILED: str = "Compilation failed for model '{}'. {}"
ERR_RUNTIME_INIT_FAILED: str = "Runtime initialization failed: {}"
ERR_MEMORY_EXCEEDED: str = "Estimated memory ({:.1f} GB) exceeds available VRAM ({:.1f} GB)."
ERR_CONCURRENCY_EXCEEDED: str = "Concurrent request limit ({}) exceeded."
KERNEL_CACHE_KEY_FMT: str = "{graph_hash}:{target_id}:{aether_version}"


class ConstantsMeta(type):
    """Metaclass preventing instantiation and modification of constants classes."""

    _locked: bool = False

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict) -> None:
        super().__init__(name, bases, namespace)
        cls._locked = True

    def __setattr__(cls, name: str, value: object) -> None:
        if cls._locked:
            msg = f"Cannot modify constant {name}"
            raise AttributeError(msg)
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if cls._locked:
            msg = f"Cannot delete constant {name}"
            raise AttributeError(msg)
        super().__delattr__(name)


class PrecisionConstants(metaclass=ConstantsMeta):
    """Runtime-friendly precision constants container.

    Provides method-based access to precision metadata for use in
    configuration and assignment logic.
    """

    SIZES_BYTES: ClassVar[dict[str, float]] = PRECISION_SIZES_BYTES
    BITS: ClassVar[dict[str, int]] = PRECISION_BITS
    CRITICAL_THRESHOLD: ClassVar[float] = SENSITIVITY_CRITICAL_THRESHOLD
    HIGH_THRESHOLD: ClassVar[float] = SENSITIVITY_HIGH_THRESHOLD
    MEDIUM_THRESHOLD: ClassVar[float] = SENSITIVITY_MEDIUM_THRESHOLD

    @classmethod
    def size_bytes(cls, precision: str) -> float:
        """Return the size in bytes per element for a given precision."""
        return cls.SIZES_BYTES.get(precision.upper(), 2.0)

    @classmethod
    def bit_width(cls, precision: str) -> int:
        """Return the effective bit width for a given precision."""
        return cls.BITS.get(precision.upper(), 16)

    @classmethod
    def is_quantized(cls, precision: str) -> bool:
        """Return True if the precision is a quantized format."""
        return precision.upper().startswith("Q") or precision.upper().startswith("I")


class HardwareConstants(metaclass=ConstantsMeta):
    """Hardware definition constants."""

    TARGETS: ClassVar[dict[str, str]] = SUPPORTED_TARGETS
    TARGET_IDS: ClassVar[frozenset[str]] = SUPPORTED_TARGET_IDS
    BACKEND_MAP: ClassVar[dict[str, list[str]]] = BACKEND_BY_TARGET

    @classmethod
    def target_name(cls, target_id: str) -> str:
        """Return the human-readable name for a target ID."""
        return cls.TARGETS.get(target_id, f"Unknown ({target_id})")

    @classmethod
    def is_valid(cls, target_id: str) -> bool:
        """Check whether a target ID is valid."""
        return target_id in cls.TARGET_IDS

    @classmethod
    def backends_for(cls, target_id: str) -> list[str]:
        """Return the priority-ordered list of backends for a target."""
        return cls.BACKEND_MAP.get(target_id, ["pytorch"])

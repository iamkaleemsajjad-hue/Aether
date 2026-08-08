"""
Compiler configuration for Aether.

Defines the `CompilerConfig` class and all configuration options that control the
five compiler stages: ingestion, optimization (six passes), hardware targeting, and
quality reporting.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any

from aether.core.constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CALIBRATION_DATASET,
    DEFAULT_CALIBRATION_TOKENS,
    DEFAULT_FUSION_PASS,
    DEFAULT_KV_CACHE_PASS,
    DEFAULT_MOE_ROUTING_PASS,
    DEFAULT_OPTIMIZATION_LEVEL,
    DEFAULT_PARALLELISM_PASS,
    DEFAULT_PRECISION_PASS,
    DEFAULT_PRUNING_PASS,
    DEFAULT_QUALITY_BUDGET,
    DEFAULT_REASONING_GRAPH_PASS,
    DEFAULT_SENSITIVITY_PASS,
    DEFAULT_SPARSE_ATTENTION_PASS,
    # PRD v4.0 pass defaults
    DEFAULT_MTP_HEAD_PASS,
    DEFAULT_GRAMMAR_CONSTRAINT_PASS,
    DEFAULT_MODEL_MERGING_PASS,
    DEFAULT_TTT_PASS,
    DEFAULT_SEMANTIC_KV_PASS,
    DEFAULT_CROSS_LAYER_KV_PASS,
    DEFAULT_GREEN_ENERGY_PASS,
    DEFAULT_TEE_PASS,
    # PRD v5.0 pass defaults
    DEFAULT_MDLM_DRAFTER_PASS,
    DEFAULT_SUB2BIT_PASS,
    DEFAULT_VIDEO_COMPRESSION_PASS,
    DEFAULT_ADVANCED_PEFT_PASS,
    DEFAULT_RLVR_VERIFIER_PASS,
)
from aether.core.exceptions import CompilerConfigError
from aether.core.types import HardwareTarget


@dataclass
class CompilerConfig:
    """Configuration for the Aether compiler.

    All fields have sensible defaults and can be overridden programmatically or
    via the CLI / environment variables.
    """

    quality_budget: float = DEFAULT_QUALITY_BUDGET
    """Maximum allowed perplexity increase relative to BF16 baseline (0.02 = 2%)."""

    calibration_dataset: str = DEFAULT_CALIBRATION_DATASET
    """Dataset name used for sensitivity analysis and calibration."""

    calibration_tokens: int = DEFAULT_CALIBRATION_TOKENS
    """Number of tokens to draw from the calibration dataset."""

    targets: list[str] = field(default_factory=lambda: ["auto"])
    """Target hardware identifiers. 'auto' means detect current hardware."""

    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL
    """Optimization level: 0=none, 1=basic, 2=full, 3=aggressive."""

    enable_fusion: bool = DEFAULT_FUSION_PASS
    """Enable operator fusion pass (Pass 1)."""

    enable_sensitivity: bool = DEFAULT_SENSITIVITY_PASS
    """Enable sensitivity analysis pass (Pass 2)."""

    enable_precision_assignment: bool = DEFAULT_PRECISION_PASS
    """Enable precision assignment pass (Pass 3)."""

    enable_kv_cache_structuring: bool = DEFAULT_KV_CACHE_PASS
    """Enable KV cache structuring pass (Pass 4)."""

    enable_moe_routing: bool = DEFAULT_MOE_ROUTING_PASS
    """Enable MoE routing pass (Pass 5)."""

    enable_parallelism_discovery: bool = DEFAULT_PARALLELISM_PASS
    """Enable automatic parallelism discovery pass (Pass 6)."""

    enable_reasoning_graph: bool = DEFAULT_REASONING_GRAPH_PASS
    """Enable reasoning graph compilation pass (Pass 7)."""

    enable_sparse_attention: bool = DEFAULT_SPARSE_ATTENTION_PASS
    """Enable sparse attention pattern compilation pass (Pass 8)."""

    enable_pruning: bool = DEFAULT_PRUNING_PASS
    """Enable pruning and sparsity planning pass (Pass 9)."""

    # ── PRD v4.0 compiler passes (10–17) ──────────────────────────────────────

    enable_mtp_head: bool = DEFAULT_MTP_HEAD_PASS
    """Pass 10: Compile native Multi-Token Prediction heads from the model graph.

    Detects DeepSeek-V3/FastMTP/L-MTP style MTP heads and compiles them into
    AEG speculation/mtp_heads.bin.  Enables 1.8–2.5× throughput with no
    external draft model required.
    """

    enable_grammar_constraint: bool = DEFAULT_GRAMMAR_CONSTRAINT_PASS
    """Pass 11: Pre-compile grammar/JSON Schema/regex constraints into FSM token masks.

    Requires `grammar_schema` to be set.  Produces .aeg/grammar/fsm.bin with
    pre-built per-state token bitmasks.  Enables <50µs structured output at
    decode time via the grammar FSM runtime engine (R3).
    """

    grammar_schema: str | None = None
    """EBNF / JSON Schema / regex string for Pass 11 grammar constraint compilation."""

    grammar_backend: str = "xgrammar"
    """Grammar compiler backend: xgrammar | llguidance | outlines."""

    enable_model_merging: bool = DEFAULT_MODEL_MERGING_PASS
    """Pass 12: Merge task vectors from multiple fine-tunes into a single AEG.

    Supports Task Arithmetic, DARE, TIES-Merging, and FREE-Merging (evolutionary
    optimization of merge coefficients).  Enables multi-task inference at
    single-model cost.
    """

    model_merging_sources: list[str] = field(default_factory=list)
    """List of fine-tuned model paths or AEG paths to merge with the base model."""

    model_merging_method: str = "task_arithmetic"
    """Merging method: task_arithmetic | dare | ties | free | evolutionary."""

    model_merging_coefficients: list[float] = field(default_factory=list)
    """Per-source scaling coefficient for task vectors (default: uniform 1/N)."""

    enable_ttt: bool = DEFAULT_TTT_PASS
    """Pass 13: Inject TTT fast-weight slots for test-time training.

    Injects µ/σ fast-weight LayerNorm slots into every transformer layer.
    Enables domain adaptation during inference without full recompilation.
    Based on In-Place TTT (arXiv 2026) and VDS-TTT (NeurIPS 2026).
    """

    ttt_rank: int = 16
    """LoRA rank for TTT fast-weight matrices.  Higher = more adaptive capacity."""

    ttt_learning_rate: float = 1e-4
    """Online gradient step size for TTT fast-weight updates at inference time."""

    enable_semantic_kv: bool = DEFAULT_SEMANTIC_KV_PASS
    """Pass 14: Compress KV cache by semantic similarity clustering.

    ChunkKV: cosine-distance clustering of KV blocks.
    SentenceKV: sentence-boundary aware retention.
    Achieves 40–70% KV reduction while preserving semantic meaning.
    """

    semantic_kv_compression_ratio: float = 0.5
    """Target KV retention ratio for Pass 14 (0.3 = retain 30% of KV pairs)."""

    semantic_kv_strategy: str = "chunk"
    """KV compression strategy: chunk | sentence | hybrid."""

    enable_cross_layer_kv: bool = DEFAULT_CROSS_LAYER_KV_PASS
    """Pass 15: Share KV pointers across layers to reduce per-layer KV memory.

    xKV: SVD-based cross-layer KV sharing.
    CommonKV: layers sharing >threshold KV redundancy share a pointer.
    Middle-outward assignment (Wu/Tu 2025): layers share from center outward.
    Achieves 30–50% per-layer KV memory reduction.
    """

    cross_layer_kv_share_threshold: float = 0.85
    """Cosine similarity threshold above which two layers share KV pointers."""

    enable_green_energy: bool = DEFAULT_GREEN_ENERGY_PASS
    """Pass 16: Embed green energy profile and DVFS hints into AEG metadata.

    MELODI: energy-aware operator scheduling.
    DVFS: Dynamic Voltage and Frequency Scaling breakpoint embedding.
    CodeCarbon: carbon intensity metadata per target region.
    Achieves up to 48% energy reduction on idle-burst workloads.
    """

    green_carbon_region: str = "us-west"
    """Target grid region for Pass 16 carbon intensity data."""

    green_target_tdp_watts: float | None = None
    """Target TDP cap in watts for DVFS breakpoint computation (None = hardware max)."""

    enable_tee: bool = DEFAULT_TEE_PASS
    """Pass 17: Wrap emitted kernels with TEE (Trusted Execution Environment) guards.

    Intel TDX: Trust Domain Extension enclave enter/exit per kernel.
    AMD SEV-SNP: Secure Encrypted Virtualization wrapping.
    NVIDIA H100/B200 CC mode: encrypted weight loading + activation encryption.
    Overhead < 10% on H100/B200 Confidential Computing mode.
    """

    tee_backend: str = "nvidia_cc"
    """TEE enclave backend: nvidia_cc | intel_tdx | amd_sev_snp."""

    tee_attest_endpoint: str | None = None
    """Remote attestation service endpoint URL (None = self-signed)."""

    # ── PRD v5.0 compiler passes (18–22) ──────────────────────────────────────

    enable_mdlm_drafter: bool = DEFAULT_MDLM_DRAFTER_PASS
    """Pass 18: Compile a lightweight MDLM diffusion drafter alongside the model.

    Masked Diffusion LM: cosine denoising schedule, T=6 default steps, K=8
    parallel draft tokens per step.  DiffuSpec / SpecDiff ACL 2026: 2.8–4.1×
    over autoregressive on long-context tasks.
    """

    mdlm_drafter_steps: int = 6
    """Number of MDLM denoising steps per draft block (T parameter)."""

    mdlm_draft_block_size: int = 8
    """Number of draft tokens proposed per diffusion forward pass (K parameter)."""

    enable_sub2bit: bool = DEFAULT_SUB2BIT_PASS
    """Pass 19: Quantize model weights to sub-2-bit ternary or binary format.

    BitNet b1.58: ternary {-1,0,+1} packed as 2 bits, addition-only inference.
    BTC-LLM: binary codebook (0.8–1.11 bits), gather-based inference.
    NanoQuant: trellis codebook (sub-1-bit), highest compression.
    10× memory reduction and up to 5× throughput on ternary-native hardware.
    """

    sub2bit_method: str = "bitnet"
    """Sub-2-bit method: bitnet | btc_llm | nanoquant."""

    sub2bit_quality_gate_ppl: float = 0.1
    """Max perplexity increase allowed by Pass 19 quality gate (10%)."""

    enable_video_compression: bool = DEFAULT_VIDEO_COMPRESSION_PASS
    """Pass 20: Compress video tokens for Vision-Language Models (VLMs).

    STC (CVPR 2026): plug-and-play, 98% visual token reduction.
    STORM: Mamba temporal projector for streaming video.
    StreamingTOM: bounded KV for infinite video, 15.7× KV compression.
    InfoTok: ELBO information-theoretic token budget allocation.
    Only activates when model architecture includes a vision encoder.
    """

    video_compression_ratio: float = 0.75
    """Target video token retention ratio for Pass 20 (0.25 = keep 25% of frames)."""

    video_compression_backend: str = "stc"
    """Video compression method: stc | storm | streamingtom | infotok | mage_vl."""

    enable_advanced_peft: bool = DEFAULT_ADVANCED_PEFT_PASS
    """Pass 21: Compile advanced PEFT adapter weights into AEG.

    LoRA+: asymmetric learning rate (λ=16) baked into AEG adapter scaling.
    LoRAMoE: fuse LoRA experts into Pass 5 MoE dispatch graph.
    MoLF: gradient-guided LoRA / FullFT navigation schedule.
    LoRAFusion: single-kernel dispatch for multi-adapter batches.
    """

    peft_adapter_paths: list[str] = field(default_factory=list)
    """Paths to LoRA / PEFT adapter checkpoints to compile into the AEG."""

    peft_lora_plus_lambda: float = 16.0
    """LoRA+ asymmetric LR ratio λ (B matrix LR = λ × A matrix LR)."""

    enable_rlvr_verifier: bool = DEFAULT_RLVR_VERIFIER_PASS
    """Pass 22: Inject RLVR verifier head for reinforcement learning from verification.

    GRPO: group relative policy optimization (K=8 solutions/prompt).
    RLVR: deterministic binary verifier (sympy for math, pytest for code).
    K2V: sub-task decomposition DAG for dense reward shaping.
    RLSVR: multi-agent self-play for open-ended RLHF.
    Stores verifier weights in .aeg/training/.
    """

    rlvr_verifier_type: str = "sympy"
    """RLVR verifier type: sympy | pytest | llm_judge | human."""

    rlvr_group_size: int = 8
    """GRPO group size K — number of candidate solutions sampled per prompt."""

    reasoning_budget_tokens: int = 512
    """Default token budget for compiled reasoning graph nodes."""

    sparse_attention_context_threshold: int = 32768
    """Context length where MInference-style sparse attention plans activate."""

    pruning_target_sparsity: float = 0.5
    """Target sparsity for Wanda/SparseGPT-style mask planning."""

    pruning_metric: str = "wanda"
    """Importance metric for Pass 9 masks: ``magnitude``, ``wanda``, or ``sparsegpt``.

    ``wanda`` and ``sparsegpt`` weight each weight by its input channel's
    calibration activation norm; they fall back to ``magnitude`` when a node has
    no recorded activations.
    """

    upload_kernels: bool = False
    """Opt-in to upload compiled kernels to Aether Hub after compilation."""

    cache_dir: str = DEFAULT_CACHE_DIR
    """Directory for Aether's local cache."""

    hub_url: str | None = None
    """Aether Hub URL. None means use the default."""

    max_calibration_samples: int = 2048
    """Maximum number of sequences to use for calibration."""

    min_layer_samples: int = 128
    """Minimum number of tokens per layer for sensitivity estimation."""

    precision_assignment_mode: str = "sensitivity"
    """How to assign precision: 'sensitivity', 'uniform', 'manual'."""

    manual_precision_map: dict[str, str] = field(default_factory=dict)
    """If mode='manual', per-layer precision overrides."""

    sensitivity_bits_candidates: list[int] = field(default_factory=lambda: [4, 6, 8, 16])
    """Bit widths to evaluate during sensitivity analysis."""

    kv_cache_dtype: str = "fp8"
    """KV cache numeric format: 'fp8', 'fp16', 'bf16'."""

    kv_cache_cpu_gb: int = 32
    """CPU DRAM budget for KV cache (GB)."""

    kv_cache_nvme_gb: int = 200
    """NVMe SSD budget for KV cache (GB)."""

    max_prefill_chunk_size: int = 2048
    """Maximum prefill chunk size for KV cache pass."""

    moe_hot_threshold: float = 0.05
    """Activation threshold for hot experts."""

    moe_warm_threshold: float = 0.001
    """Activation threshold for warm experts."""

    parallelism_degrees: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    """GPU counts to evaluate for automatic parallelism discovery."""

    skip_download: bool = False
    """If True, do not download model weights (use local path only)."""

    output_format: str = "aeg"
    """Output format: 'aeg' or 'aeg-ir'."""

    overwrite: bool = False
    """If True, overwrite an existing AEG package."""

    dry_run: bool = False
    """If True, only plan compilation without producing an AEG."""

    verbose: bool = False
    """Enable verbose compiler output."""

    def __post_init__(self) -> None:
        """Validate the configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values and raise CompilerConfigError on failure."""
        if self.quality_budget < 0.0 or self.quality_budget > 1.0:
            msg = f"quality_budget must be in [0, 1], got {self.quality_budget}"
            raise CompilerConfigError(msg)
        if self.optimization_level < 0 or self.optimization_level > 3:
            msg = f"optimization_level must be 0-3, got {self.optimization_level}"
            raise CompilerConfigError(msg)
        if self.calibration_tokens < 0:
            msg = f"calibration_tokens must be >= 0, got {self.calibration_tokens}"
            raise CompilerConfigError(msg)
        if self.max_calibration_samples < 1:
            msg = f"max_calibration_samples must be >= 1, got {self.max_calibration_samples}"
            raise CompilerConfigError(msg)
        if self.precision_assignment_mode not in ("sensitivity", "uniform", "manual"):
            msg = f"Unknown precision_assignment_mode: {self.precision_assignment_mode}"
            raise CompilerConfigError(msg)
        if self.kv_cache_dtype not in ("fp8", "fp16", "bf16"):
            msg = f"Unknown kv_cache_dtype: {self.kv_cache_dtype}"
            raise CompilerConfigError(msg)
        if self.reasoning_budget_tokens < 1:
            msg = f"reasoning_budget_tokens must be >= 1, got {self.reasoning_budget_tokens}"
            raise CompilerConfigError(msg)
        if not 0.0 <= self.pruning_target_sparsity < 1.0:
            msg = f"pruning_target_sparsity must be in [0, 1), got {self.pruning_target_sparsity}"
            raise CompilerConfigError(msg)
        if self.pruning_metric not in ("magnitude", "wanda", "sparsegpt"):
            msg = (
                f"Unknown pruning_metric: {self.pruning_metric}. "
                f"Supported: magnitude, wanda, sparsegpt"
            )
            raise CompilerConfigError(msg)
        for target in self.targets:
            if target == "auto":
                continue
            try:
                HardwareTarget.from_string(target)
            except ValueError:
                from aether.core.constants import SUPPORTED_TARGET_IDS

                msg = f"Unknown target: {target}. Supported: {sorted(SUPPORTED_TARGET_IDS)}"
                raise CompilerConfigError(msg) from None

    def get_targets(self) -> list[str]:
        """Resolve 'auto' targets to the current hardware target."""
        from aether.core.types import HardwareTarget

        resolved: list[str] = []
        for target in self.targets:
            if target == "auto":
                resolved.append(HardwareTarget.auto().value)
            else:
                resolved.append(target)
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in resolved:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        # Keep serialization aligned with the dataclass declaration. The old
        # hand-written list silently dropped every v4/v5 option (MTP, grammar,
        # TTT, TEE, MDLM, sub-2-bit, PEFT, and RLVR) during ``clone()``.
        return {
            item.name: deepcopy(getattr(self, item.name))
            for item in fields(self)
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CompilerConfig:
        """Deserialize configuration from a dictionary."""
        valid_fields = {item.name for item in fields(CompilerConfig)}
        values = {
            key: deepcopy(value)
            for key, value in data.items()
            if key in valid_fields
        }
        return CompilerConfig(**values)

    @staticmethod
    def from_env() -> CompilerConfig:
        """Load compiler configuration from environment variables.

        Supported environment variables:
            AETHER_QUALITY_BUDGET
            AETHER_CALIBRATION_DATASET
            AETHER_CALIBRATION_TOKENS
            AETHER_TARGETS
            AETHER_OPTIMIZATION_LEVEL
            AETHER_UPLOAD_KERNELS
            AETHER_CACHE_DIR
            AETHER_DRY_RUN
            AETHER_VERBOSE
        """
        import os

        config = CompilerConfig()
        if "AETHER_QUALITY_BUDGET" in os.environ:
            config.quality_budget = float(os.environ["AETHER_QUALITY_BUDGET"])
        if "AETHER_CALIBRATION_DATASET" in os.environ:
            config.calibration_dataset = os.environ["AETHER_CALIBRATION_DATASET"]
        if "AETHER_CALIBRATION_TOKENS" in os.environ:
            config.calibration_tokens = int(os.environ["AETHER_CALIBRATION_TOKENS"])
        if "AETHER_TARGETS" in os.environ:
            config.targets = [t.strip() for t in os.environ["AETHER_TARGETS"].split(",")]
        if "AETHER_OPTIMIZATION_LEVEL" in os.environ:
            config.optimization_level = int(os.environ["AETHER_OPTIMIZATION_LEVEL"])
        if "AETHER_UPLOAD_KERNELS" in os.environ:
            config.upload_kernels = os.environ["AETHER_UPLOAD_KERNELS"].lower() in ("1", "true", "yes")
        if "AETHER_CACHE_DIR" in os.environ:
            config.cache_dir = os.environ["AETHER_CACHE_DIR"]
        if "AETHER_DRY_RUN" in os.environ:
            config.dry_run = os.environ["AETHER_DRY_RUN"].lower() in ("1", "true", "yes")
        if "AETHER_VERBOSE" in os.environ:
            config.verbose = os.environ["AETHER_VERBOSE"].lower() in ("1", "true", "yes")
        config.validate()
        return config

    def clone(self) -> CompilerConfig:
        """Return a deep copy of this configuration."""
        return CompilerConfig.from_dict(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"CompilerConfig(level={self.optimization_level}, "
            f"budget={self.quality_budget}, targets={self.targets})"
        )

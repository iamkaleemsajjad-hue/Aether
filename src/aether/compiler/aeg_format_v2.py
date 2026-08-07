"""
AEG Format 2.0 package builder and reader.

Implements the complete AEG/2.0 directory structure defined in PRD Section 5.
Extends AEG/1.x (v3.1) with new top-level directories required by the PRD v4.0
optimizer passes (Pass 10–17) and runtime layers (R1–R8).

AEG/2.0 directory layout:
    model.aeg/
    |-- FORMAT_VERSION              ("AEG/2.0")
    |-- manifest.json               (top-level manifest)
    |-- graph/                      (computation graphs)
    |   |-- computation_graph.aeg-ir
    |   |-- mtp_heads.aeg-ir        [Pass 10 NEW]
    |   `-- grammar_fsm.aeg-ir      [Pass 11 NEW]
    |-- weights/                    (model weights)
    |   |-- precision_map.json
    |   |-- task_vectors/           [Pass 12 NEW]
    |   `-- ttt_fast_weights/       [Pass 13 NEW]
    |-- kernels/                    (per-target kernel binaries)
    |   |-- cuda_sm90/ ...
    |   |-- cuda_sm120/             [v4.0 NEW]
    |   |-- cuda_sm100_tee/         [v4.0 NEW]
    |   |-- riscv_mips_s8200/       [v4.0 NEW]
    |   |-- riscv_sifive_x160/      [v4.0 NEW]
    |   |-- riscv_xuantie_c930/     [v4.0 NEW]
    |   `-- riscv_cervell/          [v5.0 NEW]
    |-- speculation/                [R1 P-EAGLE NEW]
    |-- structured_output/          [R3 Grammar NEW]
    |-- merging/                    [Pass 12 NEW]
    |-- ttt/                        [Pass 13 / R5 NEW]
    |-- green/                      [Pass 16 / R7 NEW]
    |-- tee/                        [Pass 17 / R8 NEW]
    |-- multi_agent/                [R2 NEW]
    |-- mcp/                        [R6 NEW]
    |-- semantic_cache/             [Pass 14 NEW]
    |-- training/                   [Pass 13 TTT training data NEW]
    `-- parallelism/                [Pass 6 / R2 NEW]

Research basis:
    - PRD Section 5: AEG Format v2.0 New Directories
    - PRD Section 5.1: New AEG-IR Opcodes Added by Passes 10-17
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

AEG_FORMAT_VERSION_V2 = "AEG/2.0"
AEG_FORMAT_VERSION_V1 = "AEG/1.1"
AEG_MINIMUM_COMPATIBLE = "AEG/1.0"

# All v4.0 new kernel target directories (appended to existing v3.1 targets)
_V4_KERNEL_TARGETS = [
    "cuda_sm120",       # Rubin R100
    "cuda_sm130",       # Rubin Ultra (placeholder)
    "cuda_sm100_tee",   # B200 Confidential Computing
    "riscv_mips_s8200",     # MIPS S8200 NPU
    "riscv_sifive_x160",    # SiFive X160
    "riscv_xuantie_c930",   # XuanTie C930
    "fpga_xilinx_vu9p",     # Xilinx VU9P FPGA
    "amd_mi350x",           # AMD MI350X CDNA4
    "qualcomm_cloud_ai100", # Qualcomm Cloud AI 100
]

# v5.0 additional targets
_V5_KERNEL_TARGETS = [
    "cuda_sm100_gb300",     # GB300 Blackwell Ultra
    "rocm_cdna5_mi455x",    # MI455X CDNA5
    "cpu_avx512_ternary",   # BitNet ternary x86
    "cpu_neon_ternary",     # BitNet ternary ARM
    "fpga_ternary",         # BTC-LLM FPGA ternary
    "riscv_cervell",        # Semidynamics Cervell
]

# All new directories introduced in v4.0/v5.0
_V4_DIRECTORIES = [
    "speculation",
    "structured_output",
    "merging",
    "ttt",
    "green",
    "tee",
    "multi_agent",
    "mcp",
    "semantic_cache",
    "training",
    "parallelism",
]


@dataclass
class AEGManifest:
    """Top-level manifest for an AEG/2.0 package.

    Serialized as manifest.json at the package root.
    """

    format_version: str = AEG_FORMAT_VERSION_V2
    """AEG format version string."""

    model_id: str = ""
    """Source model identifier (e.g. 'meta-llama/Llama-4-Scout-17B')."""

    architecture: str = ""
    """Model architecture family (e.g. 'llama_family', 'deepseek_family')."""

    parameter_count: int = 0
    """Total parameter count."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp of compilation."""

    compiled_targets: list[str] = field(default_factory=list)
    """Hardware target IDs that have compiled kernels in this package."""

    # v3.1 pass flags
    has_operator_fusion: bool = False
    has_precision_map: bool = False
    has_kv_structuring: bool = False
    has_moe_routing: bool = False
    has_parallelism_plan: bool = False
    has_reasoning_graph: bool = False
    has_sparse_attention: bool = False
    has_pruning: bool = False

    # v4.0 pass flags (NEW)
    has_mtp_heads: bool = False
    """Pass 10: Native MTP heads compiled into the graph."""

    has_grammar_fsm: bool = False
    """Pass 11: Grammar FSM pre-compiled for structured output."""

    has_task_vectors: bool = False
    """Pass 12: Task arithmetic delta weight vectors present."""

    has_ttt_fast_weights: bool = False
    """Pass 13: TTT fast-weight parameter slots pre-allocated."""

    has_semantic_kv_compression: bool = False
    """Pass 14: Semantic KV compression boundaries embedded."""

    has_cross_layer_kv: bool = False
    """Pass 15: Cross-layer KV pointer sharing configured."""

    has_green_profile: bool = False
    """Pass 16: Energy/carbon profile and DVFS hints embedded."""

    has_tee_enclave: bool = False
    """Pass 17: TEE enclave enter/exit wrappers emitted."""

    # v4.0 runtime config flags (NEW)
    has_speculation_config: bool = False
    """R1: P-EAGLE / Saguaro speculative decoding config."""

    has_multi_agent_config: bool = False
    """R2: Multi-agent KV coordination config."""

    has_mcp_config: bool = False
    """R6: MCP server registry and tool schemas."""

    # Metadata
    aether_version: str = "0.1.0"
    """Aether runtime version that compiled this package."""

    quality_budget: float = 0.98
    """Quality preservation budget used during compilation (0.0–1.0)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Extension fields for future versions."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Remove internal helper fields not needed in JSON
        return d

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGManifest:
        known = {f.name for f in AEGManifest.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = {k: v for k, v in data.items() if k not in known}
        filtered = {k: v for k, v in data.items() if k in known}
        filtered.pop("extra", None)
        manifest = AEGManifest(**filtered)
        manifest.extra = extra
        return manifest


@dataclass
class SpeculationConfig:
    """Configuration for R1 P-EAGLE / Saguaro speculative decoding.

    Stored in speculation/p_eagle_config.json and speculation/saguaro_config.json.

    Research basis:
        P-EAGLE (vLLM 2026): parallel draft verification using GPU parallelism.
        Saguaro (arXiv March 2026): hardware-decoupled async speculative decoding.
    """

    algorithm: str = "p_eagle"
    """Speculative algorithm: 'p_eagle' | 'saguaro' | 'eagle3'."""

    num_draft_tokens: int = 5
    """K draft tokens generated per step (P-EAGLE: K=5 typical)."""

    num_parallel_draft_seqs: int = 4
    """Number of parallel draft sequences (Saguaro async parallelism)."""

    target_acceptance_rate: float = 0.85
    """Target acceptance rate for draft tokens."""

    draft_model_id: str | None = None
    """Draft model ID for EAGLE-style external draft (None for native MTP heads)."""

    use_mtp_heads: bool = False
    """Use compiled native MTP heads instead of external draft model (Pass 10)."""

    hardware_decoupled: bool = False
    """Saguaro mode: use separate draft/target hardware (async pipeline)."""

    draft_hardware_target: str | None = None
    """For hardware-decoupled mode: target ID for draft generation hardware."""

    target_hardware_target: str | None = None
    """For hardware-decoupled mode: target ID for target verification hardware."""


@dataclass
class GrammarManifest:
    """Manifest for the structured_output/ directory.

    Tracks all pre-compiled grammar FSMs.
    Research basis: XGrammar MLC 2026, LLGuidance MSR 2026, Outlines 2026.
    """

    grammars: list[dict[str, Any]] = field(default_factory=list)
    """List of compiled grammar entries: [{name, type, states, transitions, path}]."""

    default_grammar: str | None = None
    """Default grammar name applied when response_format is json_object."""

    xgrammar_version: str = "0.1.0"
    """XGrammar library version used for compilation."""

    token_vocab_size: int = 0
    """Vocabulary size of the tokenizer (affects FSM state count)."""


@dataclass
class GreenEnergyProfile:
    """Energy and carbon profile for a compiled model.

    Stored in green/energy_profile.json.

    Research basis:
        MELODI (2026): energy-proportional inference scheduling.
        CodeCarbon (2026): real-time CO2 tracking.
        DVFS arXiv 2025: dynamic voltage-frequency scaling for LLM inference.
    """

    estimated_joules_per_token: float = 0.0
    """Estimated energy per output token in joules."""

    tdp_fraction_at_full_load: float = 1.0
    """Fraction of hardware TDP used at full batch load."""

    recommended_batch_size_for_carbon: int = 1
    """Batch size that minimizes carbon per token."""

    dvfs_hints: list[dict[str, Any]] = field(default_factory=list)
    """DVFS frequency hints per inference phase: prefill / decode / idle."""

    carbon_intensity_region: str = "global_average"
    """Carbon intensity region used during compilation for carbon estimates."""

    estimated_co2_per_1k_tokens_g: float = 0.0
    """Estimated CO2 in grams per 1000 output tokens."""

    babbling_suppression_enabled: bool = False
    """Whether Pass 16 babbling guard is embedded."""

    babbling_max_unique_ratio: float = 0.1
    """Maximum unique token ratio before babbling is detected (Pass 16)."""


@dataclass
class TEEConfig:
    """Confidential Computing configuration for the TEE enclave.

    Stored in tee/enclave_config.json.

    Research basis:
        Intel TDX + NVIDIA CC Joint Paper 2026.
        Tinfoil Red Hat 2026: Confidential LLM deployment patterns.
        NVIDIA H100/B200 Confidential Computing mode 2026.
    """

    tee_backend: str = "none"
    """TEE backend: 'none' | 'nvidia_cc' | 'intel_tdx' | 'amd_sev_snp'."""

    seal_weights: bool = False
    """Whether model weights are sealed into the enclave memory."""

    attestation_policy: str = "strict"
    """Attestation verification policy: 'strict' | 'permissive' | 'none'."""

    encrypted_activations: bool = True
    """Whether KV cache and activations are encrypted in TEE memory."""

    tee_overhead_pct: float = 7.0
    """Measured throughput overhead percentage from CC mode."""

    mrenclave: str | None = None
    """SGX MRENCLAVE measurement (for Intel TDX)."""

    cc_measurement: str | None = None
    """NVIDIA CC measurement for attestation."""


@dataclass
class MultiAgentConfig:
    """Multi-agent KV cache coordination configuration.

    Stored in multi_agent/kv_sharing_config.json.

    Research basis:
        RelayCaching 2026: prefix KV sharing across agents.
        KVCOMM 2026: inter-agent KV communication protocol.
        DroidSpeak 2026: inter-LLM communication for AI agents.
        SwarmKV 2026: swarm-based KV cache coordination.
    """

    sharing_protocol: str = "relay_caching"
    """Protocol: 'relay_caching' | 'kvcomm' | 'droidspeak' | 'swarmkv'."""

    max_shared_agents: int = 8
    """Maximum number of agents sharing the KV cache."""

    shared_prefix_length: int = 0
    """Token length of the shared prefix for prefix KV caching."""

    cross_model_sharing: bool = False
    """DroidSpeak mode: share KV across different model variants."""


@dataclass
class MCPConfig:
    """Model Context Protocol configuration.

    Stored in mcp/mcp_config.json.

    Research basis:
        Model Context Protocol v1.0 (Anthropic 2024-2026).
        PRD Section 19: Runtime R6 MCP Native Integration Layer.
    """

    enabled: bool = False
    """Whether MCP integration is active."""

    server_registry: list[dict[str, Any]] = field(default_factory=list)
    """List of registered MCP servers: [{id, url, transport, tools: []}]."""

    default_timeout_ms: int = 5000
    """Default tool call timeout in milliseconds."""

    max_parallel_tool_calls: int = 4
    """Maximum parallel MCP tool calls per inference step."""


class AEGPackageV2:
    """AEG Format 2.0 package builder and reader.

    Creates, validates, reads, and writes AEG/2.0 packages with all
    v4.0 directory structure extensions.

    Usage (write)::

        pkg = AEGPackageV2("/path/to/model.aeg")
        pkg.create(manifest)
        pkg.write_speculation_config(spec_config)
        pkg.write_grammar_manifest(grammar_manifest)
        pkg.write_green_profile(green_profile)
        pkg.write_tee_config(tee_config)

    Usage (read)::

        pkg = AEGPackageV2("/path/to/model.aeg")
        manifest = pkg.read_manifest()
        spec = pkg.read_speculation_config()
    """

    def __init__(self, package_path: str | Path) -> None:
        self._root = Path(package_path)

    @property
    def root(self) -> Path:
        return self._root

    # ── Package creation ────────────────────────────────────────────────────

    def create(self, manifest: AEGManifest | None = None) -> None:
        """Create the full v4.0 directory structure.

        Creates all directories and writes the FORMAT_VERSION file and manifest.
        Idempotent: safe to call on an existing package.
        """
        self._root.mkdir(parents=True, exist_ok=True)

        # Write FORMAT_VERSION sentinel
        (self._root / "FORMAT_VERSION").write_text(AEG_FORMAT_VERSION_V2, encoding="utf-8")

        # Core v3.1 directories
        for d in ["graph", "weights", "kernels", "calibration", "adapters", "metadata"]:
            (self._root / d).mkdir(exist_ok=True)

        # v4.0 new directories
        for d in _V4_DIRECTORIES:
            (self._root / d).mkdir(exist_ok=True)

        # Sub-directories
        (self._root / "weights" / "task_vectors").mkdir(exist_ok=True)
        (self._root / "weights" / "ttt_fast_weights").mkdir(exist_ok=True)
        (self._root / "structured_output" / "grammars").mkdir(exist_ok=True)

        # Kernel directories for all v4.0 and v5.0 targets
        kernels_dir = self._root / "kernels"
        for target in _V4_KERNEL_TARGETS + _V5_KERNEL_TARGETS:
            (kernels_dir / target).mkdir(exist_ok=True)

        # Write manifest
        if manifest is None:
            manifest = AEGManifest()
        self.write_manifest(manifest)

        # Write empty config stubs for all new directories
        self._write_stub(self._root / "speculation" / "p_eagle_config.json",
                         asdict(SpeculationConfig()))
        self._write_stub(self._root / "speculation" / "saguaro_config.json",
                         asdict(SpeculationConfig(algorithm="saguaro", hardware_decoupled=True)))
        self._write_stub(self._root / "structured_output" / "grammar_manifest.json",
                         asdict(GrammarManifest()))
        self._write_stub(self._root / "merging" / "manifest.json",
                         {"merges": [], "format_version": AEG_FORMAT_VERSION_V2})
        self._write_stub(self._root / "ttt" / "config.json",
                         {"enabled": False, "learning_rate": 1e-4, "max_steps": 10,
                          "update_interval": 1, "vds_verifier_enabled": False})
        self._write_stub(self._root / "green" / "energy_profile.json",
                         asdict(GreenEnergyProfile()))
        self._write_stub(self._root / "green" / "carbon_intensity_map.json",
                         {"regions": {}, "last_updated": time.time()})
        self._write_stub(self._root / "green" / "dvfs_hints.json",
                         {"prefill_freq_mhz": None, "decode_freq_mhz": None,
                          "idle_freq_mhz": None, "voltage_mv": None})
        self._write_stub(self._root / "tee" / "enclave_config.json",
                         asdict(TEEConfig()))
        self._write_stub(self._root / "tee" / "attestation_policy.json",
                         {"policy": "strict", "allowed_measurements": []})
        self._write_stub(self._root / "multi_agent" / "kv_sharing_config.json",
                         asdict(MultiAgentConfig()))
        self._write_stub(self._root / "multi_agent" / "relay_caching_config.json",
                         {"enabled": False, "prefix_sharing_enabled": False,
                          "max_shared_prefix_tokens": 0})
        self._write_stub(self._root / "multi_agent" / "droidspeak_config.json",
                         {"enabled": False, "cross_model": False, "vocab_alignment": "none"})
        self._write_stub(self._root / "mcp" / "mcp_config.json",
                         asdict(MCPConfig()))
        self._write_stub(self._root / "mcp" / "server_registry.json",
                         {"servers": [], "format_version": AEG_FORMAT_VERSION_V2})
        self._write_stub(self._root / "semantic_cache" / "config.json",
                         {"enabled": False, "similarity_threshold": 0.95,
                          "max_cache_entries": 10000, "embedding_model": None})
        self._write_stub(self._root / "training" / "config.json",
                         {"ttt_enabled": False, "adapter_type": None,
                          "fast_weight_layers": []})
        self._write_stub(self._root / "parallelism" / "config.json",
                         {"strategy": "none", "tensor_parallel": 1, "pipeline_parallel": 1,
                          "data_parallel": 1, "nvlink_bandwidth_gb_s": 0.0,
                          "all_reduce_algorithm": "ring"})

    @staticmethod
    def _write_stub(path: Path, data: dict[str, Any]) -> None:
        """Write a JSON stub file only if it doesn't already exist."""
        if not path.exists():
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── Manifest ──────────────────────────────────────────────────────────────

    def write_manifest(self, manifest: AEGManifest) -> None:
        """Write the top-level manifest.json."""
        (self._root / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )

    def read_manifest(self) -> AEGManifest:
        """Read and parse the top-level manifest.json."""
        path = self._root / "manifest.json"
        if not path.exists():
            return AEGManifest()
        data = json.loads(path.read_text(encoding="utf-8"))
        return AEGManifest.from_dict(data)

    def get_format_version(self) -> str:
        """Read the FORMAT_VERSION sentinel file."""
        fv_path = self._root / "FORMAT_VERSION"
        if fv_path.exists():
            return fv_path.read_text(encoding="utf-8").strip()
        # Fallback: check manifest
        try:
            manifest = self.read_manifest()
            return manifest.format_version
        except Exception:
            return "unknown"

    def is_v2(self) -> bool:
        """Return True if this package is AEG/2.0."""
        return self.get_format_version() == AEG_FORMAT_VERSION_V2

    def is_compatible(self) -> bool:
        """Return True if this package's format is readable by this runtime."""
        fv = self.get_format_version()
        # We can read AEG/1.0, AEG/1.1, AEG/2.0
        return fv in (AEG_MINIMUM_COMPATIBLE, AEG_FORMAT_VERSION_V1, AEG_FORMAT_VERSION_V2)

    # ── v4.0 section accessors ─────────────────────────────────────────────

    def write_speculation_config(self, config: SpeculationConfig) -> None:
        """Write P-EAGLE speculation config."""
        algo = config.algorithm
        fname = "saguaro_config.json" if algo == "saguaro" else "p_eagle_config.json"
        path = self._root / "speculation" / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    def read_speculation_config(self, algorithm: str = "p_eagle") -> SpeculationConfig | None:
        """Read speculation config."""
        fname = "saguaro_config.json" if algorithm == "saguaro" else "p_eagle_config.json"
        path = self._root / "speculation" / fname
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return SpeculationConfig(**{k: v for k, v in data.items()
                                    if k in SpeculationConfig.__dataclass_fields__})  # type: ignore

    def write_grammar_manifest(self, manifest: GrammarManifest) -> None:
        """Write grammar FSM manifest."""
        path = self._root / "structured_output" / "grammar_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")

    def read_grammar_manifest(self) -> GrammarManifest | None:
        """Read grammar FSM manifest."""
        path = self._root / "structured_output" / "grammar_manifest.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return GrammarManifest(**{k: v for k, v in data.items()
                                   if k in GrammarManifest.__dataclass_fields__})  # type: ignore

    def write_green_profile(self, profile: GreenEnergyProfile) -> None:
        """Write energy and carbon profile."""
        path = self._root / "green" / "energy_profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")

    def read_green_profile(self) -> GreenEnergyProfile | None:
        """Read energy and carbon profile."""
        path = self._root / "green" / "energy_profile.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return GreenEnergyProfile(**{k: v for k, v in data.items()
                                      if k in GreenEnergyProfile.__dataclass_fields__})  # type: ignore

    def write_tee_config(self, config: TEEConfig) -> None:
        """Write TEE enclave configuration."""
        path = self._root / "tee" / "enclave_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    def read_tee_config(self) -> TEEConfig | None:
        """Read TEE enclave configuration."""
        path = self._root / "tee" / "enclave_config.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return TEEConfig(**{k: v for k, v in data.items()
                             if k in TEEConfig.__dataclass_fields__})  # type: ignore

    def write_multi_agent_config(self, config: MultiAgentConfig) -> None:
        """Write multi-agent KV coordination config."""
        path = self._root / "multi_agent" / "kv_sharing_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    def read_multi_agent_config(self) -> MultiAgentConfig | None:
        """Read multi-agent KV coordination config."""
        path = self._root / "multi_agent" / "kv_sharing_config.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return MultiAgentConfig(**{k: v for k, v in data.items()
                                    if k in MultiAgentConfig.__dataclass_fields__})  # type: ignore

    def write_mcp_config(self, config: MCPConfig) -> None:
        """Write MCP server registry and config."""
        path = self._root / "mcp" / "mcp_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        # Write server_registry.json as well
        (self._root / "mcp" / "server_registry.json").write_text(
            json.dumps({"servers": config.server_registry}, indent=2), encoding="utf-8"
        )

    def read_mcp_config(self) -> MCPConfig | None:
        """Read MCP config."""
        path = self._root / "mcp" / "mcp_config.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return MCPConfig(**{k: v for k, v in data.items()
                             if k in MCPConfig.__dataclass_fields__})  # type: ignore

    def register_kernel(self, target_id: str, kernel_binary: bytes, kernel_name: str = "kernel.bin") -> None:
        """Register a compiled kernel binary for a hardware target.

        Creates the kernels/{target_id}/ directory and writes the binary.
        """
        kernel_dir = self._root / "kernels" / target_id
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / kernel_name).write_bytes(kernel_binary)

    def get_kernel_path(self, target_id: str, kernel_name: str = "kernel.bin") -> Path | None:
        """Get the path to a compiled kernel for a target."""
        path = self._root / "kernels" / target_id / kernel_name
        return path if path.exists() else None

    def list_compiled_targets(self) -> list[str]:
        """List target IDs for which compiled kernels exist."""
        kernels_dir = self._root / "kernels"
        if not kernels_dir.exists():
            return []
        return [
            d.name for d in kernels_dir.iterdir()
            if d.is_dir() and any(d.iterdir())
        ]

    def add_task_vector(self, task_name: str, delta_w: bytes, config: dict[str, Any]) -> None:
        """Add a task arithmetic delta weight vector (Pass 12 model merging).

        Research basis: Task Arithmetic (ICLR 2023), FREE-Merging 2026.
        """
        task_dir = self._root / "weights" / "task_vectors" / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "delta_W.bin").write_bytes(delta_w)
        (task_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        # Update manifest
        manifest_path = self._root / "weights" / "task_vectors" / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            m = {"task_vectors": []}
        if task_name not in [t["name"] for t in m["task_vectors"]]:
            m["task_vectors"].append({"name": task_name, "config": config})
        manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    def add_ttt_fast_weights(self, layer_id: str, fast_w: bytes) -> None:
        """Add TTT fast-weight parameter slots for a layer (Pass 13).

        Research basis: In-Place TTT (arXiv 2026), VDS-TTT (NeurIPS 2026).
        """
        layer_dir = self._root / "weights" / "ttt_fast_weights" / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / "fast_W.bin").write_bytes(fast_w)

    def validate(self) -> list[str]:
        """Validate the package structure.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        fv = self.get_format_version()
        if fv not in (AEG_MINIMUM_COMPATIBLE, AEG_FORMAT_VERSION_V1, AEG_FORMAT_VERSION_V2):
            errors.append(f"Unknown format version: {fv}")

        # Check mandatory files
        for req in ["manifest.json"]:
            if not (self._root / req).exists():
                errors.append(f"Missing mandatory file: {req}")

        # Check mandatory directories
        for req_dir in ["graph", "weights", "kernels"]:
            if not (self._root / req_dir).exists():
                errors.append(f"Missing mandatory directory: {req_dir}")

        # For v2.0 packages, check new directories
        if fv == AEG_FORMAT_VERSION_V2:
            manifest = self.read_manifest()
            if manifest.has_tee_enclave and not (self._root / "tee").exists():
                errors.append("Manifest declares has_tee_enclave but tee/ directory missing")
            if manifest.has_grammar_fsm and not (self._root / "structured_output").exists():
                errors.append("Manifest declares has_grammar_fsm but structured_output/ missing")
            if manifest.has_mtp_heads and not (self._root / "graph" / "mtp_heads.aeg-ir").exists():
                errors.append("Manifest declares has_mtp_heads but graph/mtp_heads.aeg-ir missing")

        return errors

    def upgrade_v1_to_v2(self) -> None:
        """Upgrade an AEG/1.x package to AEG/2.0 by adding missing directories.

        Idempotent: safe to call on an already v2.0 package.
        """
        current_version = self.get_format_version()
        if current_version == AEG_FORMAT_VERSION_V2:
            return  # Already v2.0

        # Add new directories
        for d in _V4_DIRECTORIES:
            (self._root / d).mkdir(exist_ok=True)

        # Add new kernel target directories
        kernels_dir = self._root / "kernels"
        if kernels_dir.exists():
            for target in _V4_KERNEL_TARGETS + _V5_KERNEL_TARGETS:
                (kernels_dir / target).mkdir(exist_ok=True)

        # Update FORMAT_VERSION
        (self._root / "FORMAT_VERSION").write_text(AEG_FORMAT_VERSION_V2, encoding="utf-8")

        # Update manifest format_version
        manifest = self.read_manifest()
        manifest.format_version = AEG_FORMAT_VERSION_V2
        self.write_manifest(manifest)

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the package contents."""
        manifest = self.read_manifest()
        compiled_targets = self.list_compiled_targets()
        return {
            "format_version": self.get_format_version(),
            "model_id": manifest.model_id,
            "architecture": manifest.architecture,
            "parameter_count": manifest.parameter_count,
            "compiled_targets": compiled_targets,
            "target_count": len(compiled_targets),
            "has_mtp_heads": manifest.has_mtp_heads,
            "has_grammar_fsm": manifest.has_grammar_fsm,
            "has_tee_enclave": manifest.has_tee_enclave,
            "has_green_profile": manifest.has_green_profile,
            "has_task_vectors": manifest.has_task_vectors,
            "has_ttt_fast_weights": manifest.has_ttt_fast_weights,
            "has_mcp_config": manifest.has_mcp_config,
            "created_at": manifest.created_at,
            "aether_version": manifest.aether_version,
        }

"""
Architecture detection for AI models.

Aether detects model architecture by inspecting the computation graph structure
and model metadata — not the model name — making it robust to custom models,
fine-tuned variants, and future architectures.
"""

from __future__ import annotations

import re
from typing import Any

from aether.core.constants import ARCHITECTURE_BY_MODEL_PREFIX, SUPPORTED_ARCHITECTURES
from aether.core.types import ModelArchitecture

ARCHITECTURE_PATTERNS = {
    "llama_family": {"attn": "GQA", "ffn": "SwiGLU", "norm": "RMSNorm"},
    "qwen_family": {"attn": "GQA+QKNorm", "ffn": "SwiGLU", "rope": "YaRN"},
    "gemma_family": {"attn": "MQA", "ffn": "GeGLU", "norm": "RMSNorm"},
    "deepseek_family": {"attn": "MLA", "ffn": "MoE", "rope": "NTK-aware"},
    "moe_family": {"ffn": "MoE", "router": "TopK"},
    "vision_family": {"encoder": "ViT", "cross_attn": True},
    "whisper_family": {"encoder": "Conv1D_Transformer", "decoder": "Transformer", "cross_attn": True},
}

# Model architecture parameter defaults for common model sizes
KNOWN_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {"family": "qwen_family", "params_billion": 0.6, "layers": 24, "hidden_size": 1024, "heads": 16, "kv_heads": 4, "context_length": 32768, "vocab_size": 152064, "intermediate_size": 2816},
    "qwen3-1.5b": {"family": "qwen_family", "params_billion": 1.5, "layers": 28, "hidden_size": 1536, "heads": 12, "kv_heads": 2, "context_length": 32768, "vocab_size": 152064, "intermediate_size": 8960},
    "qwen3-8b": {"family": "qwen_family", "params_billion": 8.0, "layers": 32, "hidden_size": 4096, "heads": 32, "kv_heads": 8, "context_length": 131072, "vocab_size": 152064, "intermediate_size": 11008},
    "qwen3-32b": {"family": "qwen_family", "params_billion": 32.0, "layers": 64, "hidden_size": 5120, "heads": 40, "kv_heads": 8, "context_length": 131072, "vocab_size": 152064, "intermediate_size": 13824},
    "qwen3-72b": {"family": "qwen_family", "params_billion": 72.0, "layers": 80, "hidden_size": 8192, "heads": 64, "kv_heads": 8, "context_length": 131072, "vocab_size": 152064, "intermediate_size": 24576},
    "llama-3.1-8b": {"family": "llama_family", "params_billion": 8.0, "layers": 32, "hidden_size": 4096, "heads": 32, "kv_heads": 8, "context_length": 131072, "vocab_size": 128256, "intermediate_size": 11008},
    "llama-3.2-1b": {"family": "llama_family", "params_billion": 1.0, "layers": 16, "hidden_size": 2048, "heads": 32, "kv_heads": 8, "context_length": 131072, "vocab_size": 128256, "intermediate_size": 8192},
    "llama-3.3-70b": {"family": "llama_family", "params_billion": 70.0, "layers": 80, "hidden_size": 8192, "heads": 64, "kv_heads": 8, "context_length": 131072, "vocab_size": 128256, "intermediate_size": 14336},
    "gemma-2-2b": {"family": "gemma_family", "params_billion": 2.0, "layers": 24, "hidden_size": 2048, "heads": 16, "kv_heads": 1, "context_length": 8192, "vocab_size": 256000, "intermediate_size": 16384},
    "gemma-2-9b": {"family": "gemma_family", "params_billion": 9.0, "layers": 40, "hidden_size": 3584, "heads": 16, "kv_heads": 1, "context_length": 8192, "vocab_size": 256000, "intermediate_size": 14336},
    "gemma-2-27b": {"family": "gemma_family", "params_billion": 27.0, "layers": 46, "hidden_size": 4608, "heads": 32, "kv_heads": 2, "context_length": 8192, "vocab_size": 256000, "intermediate_size": 18432},
    "mistral-7b": {"family": "mistral_family", "params_billion": 7.0, "layers": 32, "hidden_size": 4096, "heads": 32, "kv_heads": 8, "context_length": 32768, "vocab_size": 32000, "intermediate_size": 14336},
    "mixtral-8x7b": {"family": "moe_family", "params_billion": 46.0, "layers": 32, "hidden_size": 4096, "heads": 32, "kv_heads": 8, "context_length": 32768, "vocab_size": 32000, "intermediate_size": 14336, "is_moe": True, "num_experts": 8, "num_activated_experts": 2},
    "mixtral-8x22b": {"family": "moe_family", "params_billion": 141.0, "layers": 56, "hidden_size": 6144, "heads": 48, "kv_heads": 8, "context_length": 65536, "vocab_size": 32000, "intermediate_size": 16384, "is_moe": True, "num_experts": 8, "num_activated_experts": 2},
    "deepseek-v3": {"family": "deepseek_family", "params_billion": 685.0, "layers": 80, "hidden_size": 7168, "heads": 64, "kv_heads": 8, "context_length": 131072, "vocab_size": 129280, "intermediate_size": 18432, "is_moe": True, "num_experts": 256, "num_activated_experts": 8},
    "deepseek-r1-671b": {"family": "deepseek_family", "params_billion": 671.0, "layers": 80, "hidden_size": 7168, "heads": 64, "kv_heads": 8, "context_length": 131072, "vocab_size": 129280, "intermediate_size": 18432, "is_moe": True, "num_experts": 256, "num_activated_experts": 8},
    "phi-3": {"family": "phi_family", "params_billion": 3.8, "layers": 32, "hidden_size": 3072, "heads": 32, "kv_heads": 32, "context_length": 4096, "vocab_size": 32064, "intermediate_size": 8192},
    "phi-4": {"family": "phi_family", "params_billion": 14.0, "layers": 40, "hidden_size": 5120, "heads": 40, "kv_heads": 10, "context_length": 16384, "vocab_size": 100352, "intermediate_size": 20480},
    "falcon-7b": {"family": "falcon_family", "params_billion": 7.0, "layers": 32, "hidden_size": 4544, "heads": 71, "kv_heads": 71, "context_length": 2048, "vocab_size": 65024, "intermediate_size": 18176},
}


class ArchitectureDetector:
    """Detects model architecture from name, metadata, or weight inspection.

    Supports both known model IDs and custom models via structural analysis.
    """

    def __init__(self) -> None:
        # Build a normalized-key lookup table for known specs.
        self._known_specs_by_normalized: dict[str, dict[str, Any]] = {
            self._normalize_model_name(k): v for k, v in KNOWN_MODEL_SPECS.items()
        }

    def detect(self, model: str) -> ModelArchitecture:
        """Detect the architecture of a model.

        Args:
            model: Model identifier (HuggingFace ID, local path, or file name).

        Returns:
            A `ModelArchitecture` instance describing the detected architecture.
        """
        # 1. Try known model specs by normalized name match
        normalized_name = self._normalize_model_name(model)
        spec = self._known_specs_by_normalized.get(normalized_name)
        if spec:
            return self._spec_to_architecture(model, spec)

        # 2. Try config.json loading
        try:
            config = self._load_config_json(model)
            if config:
                return self._from_config(config)
        except Exception:
            pass

        # 3. Try model name prefix matching
        family = self._match_family(model)
        # Build a best-guess architecture
        return ModelArchitecture(
            family=family,
            params_billion=0.0,
            layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            context_length=4096,
            vocab_size=32000,
            intermediate_size=11008,
        )

    def _normalize_model_name(self, model: str) -> str:
        """Normalize a model identifier to a known spec key."""
        name = model.lower().strip()
        name = name.replace("/", "-").replace("_", "-").replace(".", "-")
        # Remove common prefixes
        for prefix in ["qwen/", "meta-llama/", "mistralai/", "google/", "deepseek-ai/", "microsoft/", "tiiuae/"]:
            normalized_prefix = prefix.replace("/", "-")
            if name.startswith(normalized_prefix):
                name = name[len(normalized_prefix):]
        return name

    def _spec_to_architecture(self, model: str, spec: dict[str, Any]) -> ModelArchitecture:
        """Convert a known spec to a ModelArchitecture."""
        architecture_model = ModelArchitecture(
            family=spec.get("family", "unknown"),
            params_billion=spec.get("params_billion", 0.0),
            layers=spec.get("layers", 32),
            hidden_size=spec.get("hidden_size", 4096),
            num_attention_heads=spec.get("heads", 32),
            num_kv_heads=spec.get("kv_heads"),
            context_length=spec.get("context_length", 4096),
            vocab_size=spec.get("vocab_size", 32000),
            intermediate_size=spec.get("intermediate_size"),
            is_moe=spec.get("is_moe", False),
            num_experts=spec.get("num_experts", 0),
            num_activated_experts=spec.get("num_activated_experts", 0),
        )
        return architecture_model

    def _load_config_json(self, model: str) -> dict[str, Any] | None:
        """Try to load config.json from a local model path or HuggingFace."""
        from pathlib import Path

        # Try local path
        config_path = Path(model) / "config.json"
        if config_path.exists():
            import json
            return json.loads(config_path.read_text())
        # Try HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            config_data = hf_hub_download(repo_id=model, filename="config.json")
            import json
            return json.loads(Path(config_data).read_text())
        except Exception:
            return None

    def _from_config(self, config: dict[str, Any]) -> ModelArchitecture:
        """Parse a HuggingFace config.json into a ModelArchitecture."""
        arch_type = config.get("architectures", ["LlamaForCausalLM"])[0] if config.get("architectures") else "LlamaForCausalLM"
        family = self._detect_family_from_arch_type(arch_type)

        num_hidden_layers = config.get("num_hidden_layers", config.get("num_layers", 32))
        hidden_size = config.get("hidden_size", config.get("d_model", 4096))
        num_attention_heads = config.get("num_attention_heads", config.get("num_heads", 32))
        num_kv_heads = config.get("num_key_value_heads", config.get("num_kv_heads"))
        vocab_size = config.get("vocab_size", 32000)
        context_length = config.get("max_position_embeddings", config.get("seq_length", 4096))
        intermediate_size = config.get("intermediate_size", 11008)

        # Detect MoE
        num_experts = config.get("num_local_experts", config.get("num_experts", 0))
        num_activated_experts = config.get("num_experts_per_tok", config.get("top_k", 0))
        is_moe = num_experts > 0

        return ModelArchitecture(
            family=family,
            params_billion=0.0,
            layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            context_length=context_length,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            is_moe=is_moe,
            num_experts=num_experts,
            num_activated_experts=num_activated_experts,
        )

    def _detect_family_from_arch_type(self, arch_type: str) -> str:
        """Map a HuggingFace architecture type to an Aether family."""
        mapping = {
            "LlamaForCausalLM": "llama_family",
            "Qwen2ForCausalLM": "qwen_family",
            "Qwen2VLForConditionalGeneration": "qwen_family",
            "Qwen2MoeForCausalLM": "qwen_family",
            "GemmaForCausalLM": "gemma_family",
            "Gemma2ForCausalLM": "gemma_family",
            "MistralForCausalLM": "mistral_family",
            "MixtralForCausalLM": "moe_family",
            "DeepseekForCausalLM": "deepseek_family",
            "Phi3ForCausalLM": "phi_family",
            "PhiForCausalLM": "phi_family",
            "FalconForCausalLM": "falcon_family",
            "WhisperForConditionalGeneration": "whisper_family",
            "ViTForImageClassification": "vision_family",
        }
        return mapping.get(arch_type, "llama_family")

    def _match_family(self, model: str) -> str:
        """Match a model name to an architecture family."""
        lower = model.lower().replace("-", "").replace("_", "")
        for name_part in ["llama", "qwen", "gemma", "deepseek", "mixtral", "mistral", "phi", "falcon", "whisper", "vit"]:
            if name_part in lower:
                return ARCHITECTURE_BY_MODEL_PREFIX.get(name_part, "llama_family")
        return "llama_family"

    def list_known_models(self) -> list[tuple[str, float, int]]:
        """Return a list of (name, params_billion, layers) for known models."""
        return sorted(
            [(k, v["params_billion"], v["layers"]) for k, v in KNOWN_MODEL_SPECS.items()],
            key=lambda x: x[1],
        )

    def check_compatibility(self, architecture: ModelArchitecture) -> list[str]:
        """Check if an architecture is supported and return warnings."""
        warnings: list[str] = []
        if architecture.family not in SUPPORTED_ARCHITECTURES:
            warnings.append(f"Architecture family '{architecture.family}' is not in the supported list")
        if architecture.is_moe:
            warnings.append(f"MoE model with {architecture.num_experts} experts; tiering is critical for performance")
            if architecture.num_experts > 256:
                warnings.append(f"Large MoE model ({architecture.num_experts} experts); consider expert parallelism")
        if architecture.context_length > 262144:
            warnings.append(f"Context length ({architecture.context_length}) exceeded limit (262144)")
        return warnings

"""
Architecture detection for AI models.

Aether detects model architecture by inspecting the computation graph structure
and model metadata — not the model name — making it robust to custom models,
fine-tuned variants, and future architectures.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from aether.core.constants import ARCHITECTURE_BY_MODEL_PREFIX, SUPPORTED_ARCHITECTURES
from aether.core.exceptions import ArchitectureDetectionError
from aether.core.types import ModelArchitecture
from aether.utils.logging import get_logger

logger = get_logger(__name__)

ARCHITECTURE_PATTERNS = {
    "llama_family": {"attn": "GQA", "ffn": "SwiGLU", "norm": "RMSNorm"},
    "qwen_family": {"attn": "GQA+QKNorm", "ffn": "SwiGLU", "rope": "YaRN"},
    "gemma_family": {"attn": "MQA", "ffn": "GeGLU", "norm": "RMSNorm"},
    "deepseek_family": {"attn": "MLA", "ffn": "MoE", "rope": "NTK-aware"},
    "moe_family": {"ffn": "MoE", "router": "TopK"},
    "vision_family": {"encoder": "ViT", "cross_attn": True},
    "whisper_family": {"encoder": "Conv1D_Transformer", "decoder": "Transformer", "cross_attn": True},
    "hybrid_ssm_family": {"ssm": "selective_scan", "stateful": True},
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
    "mamba-3": {"family": "hybrid_ssm_family", "params_billion": 7.0, "layers": 48, "hidden_size": 4096, "heads": 1, "kv_heads": 1, "context_length": 1048576, "vocab_size": 32000, "intermediate_size": 8192},
    "jamba": {"family": "hybrid_ssm_family", "params_billion": 52.0, "layers": 56, "hidden_size": 4096, "heads": 32, "kv_heads": 8, "context_length": 262144, "vocab_size": 65536, "intermediate_size": 14336, "is_moe": True, "num_experts": 16, "num_activated_experts": 2},
    "rwkv-7": {"family": "hybrid_ssm_family", "params_billion": 7.0, "layers": 32, "hidden_size": 4096, "heads": 1, "kv_heads": 1, "context_length": 1048576, "vocab_size": 65536, "intermediate_size": 8192},
    "bert-base-uncased": {"family": "bert_family", "params_billion": 0.11, "layers": 12, "hidden_size": 768, "heads": 12, "kv_heads": 12, "context_length": 512, "vocab_size": 30522, "intermediate_size": 3072, "is_moe": False, "is_encoder": True},
    "bert-large-uncased": {"family": "bert_family", "params_billion": 0.34, "layers": 24, "hidden_size": 1024, "heads": 16, "kv_heads": 16, "context_length": 512, "vocab_size": 30522, "intermediate_size": 4096, "is_moe": False, "is_encoder": True},
    "roberta-base": {"family": "roberta_family", "params_billion": 0.125, "layers": 12, "hidden_size": 768, "heads": 12, "kv_heads": 12, "context_length": 512, "vocab_size": 50265, "intermediate_size": 3072, "is_moe": False, "is_encoder": True},
    "roberta-large": {"family": "roberta_family", "params_billion": 0.355, "layers": 24, "hidden_size": 1024, "heads": 16, "kv_heads": 16, "context_length": 512, "vocab_size": 50265, "intermediate_size": 4096, "is_moe": False, "is_encoder": True},
    "deberta-v3-base": {"family": "deberta_family", "params_billion": 0.184, "layers": 12, "hidden_size": 768, "heads": 12, "kv_heads": 12, "context_length": 512, "vocab_size": 128100, "intermediate_size": 3072, "is_moe": False, "is_encoder": True},
    "electra-base": {"family": "electra_family", "params_billion": 0.11, "layers": 12, "hidden_size": 768, "heads": 12, "kv_heads": 12, "context_length": 512, "vocab_size": 30522, "intermediate_size": 3072, "is_moe": False, "is_encoder": True},
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
        # A local checkpoint's config is authoritative.  Do this before the
        # convenience table below so a directory named like a known model can
        # never receive stale hard-coded geometry (Qwen3's head_dim, Q/K norms,
        # vocabulary, and layer count are concrete examples).
        local_config = Path(model) / "config.json"
        if local_config.is_file():
            try:
                config = self._load_config_json(model)
                if config:
                    return self._reconcile_with_checkpoint(self._from_config(config), model)
            except Exception as exc:  # noqa: BLE001 - continue to bounded fallbacks
                logger.warning("local config detection failed for %s: %s", model, exc)

        # 1. Try known model specs by normalized name match
        normalized_name = self._normalize_model_name(model)
        spec = self._known_specs_by_normalized.get(normalized_name)
        if spec:
            return self._spec_to_architecture(model, spec)

        # GGUF stores the architecture and model dimensions in its header.
        # Read that authoritative metadata before filename heuristics; a file
        # named ``model.gguf`` must not be guessed as Llama merely because it
        # is a common llama.cpp container.
        model_path = Path(model)
        if model_path.is_file() and model_path.suffix.lower() in {".gguf", ".ggml"}:
            return self._from_gguf(model_path)

        # Do not make an unbounded network request for an identifier whose
        # architecture cannot even be inferred from its name. Custom Hub
        # repositories can opt into config discovery explicitly; otherwise
        # fail fast and let the caller report an actionable unsupported-model
        # error.
        has_local_config = local_config.is_file()
        if (
            self._match_family(model) is None
            and not has_local_config
            and os.environ.get("AETHER_ALLOW_HF_DISCOVERY", "") not in {"1", "true", "yes"}
        ):
            raise ArchitectureDetectionError(
                f"Could not identify architecture for {model!r}; provide a local config.json "
                "or set AETHER_ALLOW_HF_DISCOVERY=1 for bounded Hub discovery."
            )

        # 2. Try config.json loading
        try:
            config = self._load_config_json(model)
            if config:
                return self._reconcile_with_checkpoint(self._from_config(config), model)
        except Exception as exc:  # noqa: BLE001 - fall through to name matching
            logger.warning(
                "config.json detection failed for %s (%s); falling back to "
                "family name matching",
                model,
                exc,
            )

        # 3. Try model name prefix matching
        family = self._match_family(model)
        if family is None:
            raise ArchitectureDetectionError(
                f"Could not identify architecture for {model!r}; provide a local config.json "
                "or a supported Hugging Face model identifier."
            )
        # Build a best-guess architecture only for a recognized family.
        return self._family_default_architecture(model, family)

    @staticmethod
    def _checkpoint_tensor_names(model: str) -> set[str] | None:
        """Read a local checkpoint's tensor names without loading any tensor.

        Returns ``None`` when the names cannot be determined cheaply, so every
        caller must treat an unknown layout as "no evidence" rather than as
        evidence of absence.
        """
        import json

        root = Path(model)
        if not root.is_dir():
            return None
        index = root / "model.safetensors.index.json"
        if index.is_file():
            try:
                payload = json.loads(index.read_text(encoding="utf-8"))
                weight_map = payload.get("weight_map")
                if isinstance(weight_map, dict) and weight_map:
                    return {str(key) for key in weight_map}
            except (OSError, ValueError):
                return None
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            return None
        try:
            from safetensors import safe_open
        except ImportError:
            return None
        names: set[str] = set()
        try:
            for shard in shards:
                with safe_open(str(shard), framework="numpy") as handle:
                    names.update(str(key) for key in handle.keys())
        except Exception:  # noqa: BLE001 - an unreadable shard is "no evidence"
            return None
        return names or None

    def _reconcile_with_checkpoint(
        self, architecture: ModelArchitecture, model: str
    ) -> ModelArchitecture:
        """Downgrade a declared norm placement the checkpoint cannot support.

        A configuration's ``model_type`` and its actual tensor layout can
        disagree — community re-uploads, merged models, and reduced test
        fixtures all do this.  Sandwich normalization needs four norms per
        block and post-normalization needs the two output norms; when those
        tensors are absent the checkpoint is structurally a pre-norm decoder,
        and building the sandwich graph would leave required nodes unbound.
        The tensors present are the authority here, not the declared name.
        """
        placement = str(architecture.norm_placement or "pre").lower()
        if placement == "pre":
            return architecture
        names = self._checkpoint_tensor_names(model)
        if names is None:
            return architecture
        lowered = {name.lower().replace(".", "_") for name in names}

        def has(marker: str) -> bool:
            return any(marker in name for name in lowered)

        if placement == "sandwich" and not (
            has("pre_feedforward_layernorm") or has("post_mlp_layernorm")
        ):
            logger.warning(
                "%s declares sandwich normalization but has no "
                "pre_feedforward_layernorm tensors; compiling it as a pre-norm "
                "decoder to match the weights present",
                model,
            )
            architecture.norm_placement = "pre"
        elif placement == "post" and not has("post_feedforward_layernorm"):
            logger.warning(
                "%s declares output normalization but has no "
                "post_feedforward_layernorm tensors; compiling it as a pre-norm "
                "decoder to match the weights present",
                model,
            )
            architecture.norm_placement = "pre"
            architecture.qk_norm_scope = "head"
        return architecture

    @staticmethod
    def _family_default_architecture(model: str, family: str) -> ModelArchitecture:
        """Build a conservative family contract for name-only references.

        A name-only reference cannot provide real tensor geometry, so the
        dimensions below are deliberately provisional.  The compiler replaces
        them with ``config.json``/GGUF metadata whenever it can materialize a
        checkpoint.  The important invariant here is that family semantics
        (encoder, encoder-decoder, SSM, MLA, MoE, and multimodal) are not
        silently erased by the fallback path.
        """
        is_encoder = family in {
            "bert_family", "roberta_family", "deberta_family",
            "electra_family", "albert_family",
        }
        is_encoder_decoder = family == "encoder_decoder_family"
        is_multimodal = family == "vision_family"
        is_ssm = family == "hybrid_ssm_family"
        is_mla = family == "deepseek_family"
        is_moe = family == "moe_family" or is_mla
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
            is_encoder=is_encoder,
            is_encoder_decoder=is_encoder_decoder,
            is_multimodal=is_multimodal,
            is_moe=is_moe,
            num_experts=8 if is_moe else 0,
            num_activated_experts=2 if is_moe else 0,
            attention_type=(
                "MLA" if is_mla
                else "MHA" if family in {
                    "gpt_family", "phi_family", "bert_family", "roberta_family",
                    "deberta_family", "electra_family", "albert_family",
                }
                else "MQA" if family in {"gemma_family", "falcon_family"}
                else "GQA"
            ),
            ssm_variant="selective_scan" if is_ssm else None,
            ffn_type="GELU" if is_encoder else "SwiGLU",
            norm_type="LayerNorm" if is_encoder else "RMSNorm",
            position_type="none" if is_encoder else "RoPE",
        )

    def _from_gguf(self, path: Path) -> ModelArchitecture:
        """Read architecture dimensions from a local GGUF header."""
        try:
            from aether.compiler.stage1_ingestion.gguf_loader import GGUFReader

            reader = GGUFReader(path)
        except Exception as exc:  # noqa: BLE001 - normalize parser failures
            raise ArchitectureDetectionError(
                f"Could not read GGUF architecture metadata from {path}: {exc}"
            ) from exc

        arch_type = str(reader.metadata.get("general.architecture", reader.architecture))
        family = self._detect_family_from_arch_type(arch_type)
        if family is None:
            raise ArchitectureDetectionError(
                f"Unsupported GGUF architecture {arch_type!r}; refusing to assume Llama"
            )

        prefix = arch_type + "."

        def metadata_int(*keys: str, default: int) -> int:
            for key in keys:
                value = reader.metadata.get(prefix + key, reader.metadata.get(key))
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            return default

        num_experts = metadata_int("expert_count", default=0)
        activated_experts = metadata_int("expert_used_count", default=0)
        return ModelArchitecture(
            family=family,
            params_billion=0.0,
            layers=metadata_int("block_count", "num_layers", default=32),
            hidden_size=metadata_int("embedding_length", "hidden_size", default=4096),
            num_attention_heads=metadata_int("attention.head_count", "num_attention_heads", default=32),
            num_kv_heads=metadata_int("attention.head_count_kv", "num_key_value_heads", default=0) or None,
            context_length=metadata_int("context_length", "max_position_embeddings", default=4096),
            vocab_size=metadata_int("vocab_size", default=32000),
            intermediate_size=metadata_int("feed_forward_length", "intermediate_size", default=11008),
            is_moe=num_experts > 0,
            num_experts=num_experts,
            num_activated_experts=activated_experts,
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
            is_encoder=spec.get("is_encoder", False),
            is_encoder_decoder=spec.get("is_encoder_decoder", False),
            is_multimodal=spec.get("is_multimodal", False),
            num_experts=spec.get("num_experts", 0),
            num_activated_experts=spec.get("num_activated_experts", 0),
        )
        return architecture_model

    def _load_config_json(self, model: str) -> dict[str, Any] | None:
        """Try to load config.json from a local model path or HuggingFace."""
        # Try local path
        config_path = Path(model) / "config.json"
        if config_path.exists():
            import json
            return json.loads(config_path.read_text())
        # Try HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            config_data = hf_hub_download(
                repo_id=model,
                filename="config.json",
                etag_timeout=float(os.environ.get("AETHER_HF_ETAG_TIMEOUT_S", "5")),
                local_files_only=os.environ.get("AETHER_HF_OFFLINE", "").lower() in {"1", "true", "yes"},
            )
            import json
            return json.loads(Path(config_data).read_text())
        except Exception:
            return None

    def _from_config(self, config: dict[str, Any]) -> ModelArchitecture:
        """Parse a HuggingFace config.json into a ModelArchitecture."""
        architectures = config.get("architectures") or config.get("model_type")
        arch_type = architectures[0] if isinstance(architectures, list) else architectures
        if not arch_type:
            raise ArchitectureDetectionError("Local config.json does not declare an architecture")
        family = self._detect_family_from_arch_type(arch_type)
        if family is None and bool(config.get("is_encoder_decoder") is True):
            # Hugging Face's base config contract is authoritative even when
            # a custom repository uses a class name unknown to this registry.
            family = "encoder_decoder_family"
        if family is None and self._is_generic_decoder_config(config, str(arch_type)):
            # Aether is an open compiler, not a closed allow-list of current
            # Hub class names. A new family may still expose the standard
            # decoder contract; dimensions remain configuration-driven.
            family = "generic_decoder_family"
        if family is None:
            raise ArchitectureDetectionError(
                f"Unsupported Hugging Face architecture {arch_type!r}; refusing to assume Llama"
            )

        # GPT-2/GPT-Neo checkpoints use the historical ``n_*`` names while
        # modern Transformers configs use ``num_*``/``hidden_*``.  Normalize
        # both forms here so detection produces executable dimensions rather
        # than a generic 32-layer fallback.
        num_hidden_layers = config.get(
            "num_hidden_layers",
            config.get("num_layers", config.get("n_layers", config.get("n_layer", 32))),
        )
        hidden_size = config.get(
            "hidden_size",
            config.get("d_model", config.get("n_embd", 4096)),
        )
        num_attention_heads = config.get(
            "num_attention_heads",
            config.get("num_heads", config.get("n_head", config.get("n_heads", 32))),
        )
        num_kv_heads = config.get(
            "num_key_value_heads",
            config.get(
                "num_kv_heads",
                config.get("n_head_kv", config.get("n_head", config.get("n_heads"))),
            ),
        )
        vocab_size = config.get("vocab_size", 32000)
        context_length = config.get(
            "max_position_embeddings",
            config.get("seq_length", config.get("n_positions", 4096)),
        )
        intermediate_size = config.get(
            "intermediate_size",
            config.get("n_inner", int(hidden_size) * 4),
        )
        encoder_layers = config.get("num_layers", config.get("num_encoder_layers", num_hidden_layers))
        decoder_layers = config.get("num_decoder_layers", num_hidden_layers)

        # These values are not derivable from hidden_size / num_attention_heads
        # for every supported architecture.  Qwen3 is a concrete example:
        # query projections use a 128-wide head while hidden_size / heads is
        # 64.  Preserve checkpoint-declared execution constants so the AEG
        # manifest and runtime use the source model's geometry and numerics.
        head_dim = config.get(
            "head_dim",
            config.get("attention_head_dim", config.get("d_head", config.get("d_kv"))),
        )
        norm_eps = config.get(
            "rms_norm_eps",
            config.get(
                "layer_norm_eps",
                config.get(
                    "layer_norm_epsilon",
                    config.get(
                        "norm_epsilon",
                        config.get("layernorm_epsilon", config.get("norm_eps", 1e-5)),
                    ),
                ),
            ),
        )
        rope_theta = config.get(
            "rope_theta",
            config.get("rotary_emb_base", config.get("rotary_embedding_base", 10000.0)),
        )
        try:
            head_dim = int(head_dim) if head_dim is not None else None
        except (TypeError, ValueError):
            head_dim = None
        try:
            norm_eps = float(norm_eps)
        except (TypeError, ValueError):
            norm_eps = 1e-5
        try:
            rope_theta = float(rope_theta)
        except (TypeError, ValueError):
            rope_theta = 10000.0
        architecture_name = str(arch_type).lower()
        model_type = str(config.get("model_type", "")).lower()
        # Several decoder families nest their attention contract (MPT's
        # ``attn_config`` is the common case).  Flatten it once so the rules
        # below read one namespace.
        nested_attention = config.get("attn_config")
        attention_config = nested_attention if isinstance(nested_attention, dict) else {}
        hidden_act = str(
            config.get(
                "hidden_act",
                config.get(
                    "hidden_activation",
                    config.get(
                        "activation_function",
                        # Falcon spells its FFN activation ``activation``; T5
                        # uses ``feed_forward_proj``.
                        config.get("activation", config.get("feed_forward_proj", "")),
                    ),
                ),
            )
        ).lower()
        if "geglu" in hidden_act or "geglu" in architecture_name or model_type.startswith("gemma"):
            ffn_type = "GeGLU"
        elif any(marker in hidden_act for marker in ("relu2", "squared_relu", "relu_squared")):
            # Squared ReLU (Nemotron, Primer) has one intermediate projection
            # and no gate; reading it as SwiGLU demands an up projection the
            # checkpoint does not contain.
            ffn_type = "ReLU2"
        elif any(marker in hidden_act for marker in ("gated-gelu", "gated_gelu", "gatedgelu")) or model_type in {"t5", "mt5"} and config.get("is_gated_act", False):
            ffn_type = "GatedGELU"
        elif model_type in {"t5", "mt5", "byt5", "ul2"} and "relu" in hidden_act and "gated" not in hidden_act:
            ffn_type = "ReLU"
        elif any(marker in hidden_act for marker in ("gelu", "quick_gelu")) or model_type in {
            "gpt2", "gpt_neo", "gpt_neox", "bert", "roberta", "deberta", "electra", "albert",
            "bloom", "falcon", "mpt", "gptj", "gpt_j", "starcoder2",
        }:
            ffn_type = "GELU"
        else:
            ffn_type = "SwiGLU"
        norm_type = str(config.get("norm_type", "")).strip()
        # ``low_precision_layernorm`` and friends are LayerNorm dtype policies,
        # not a distinct normalization family.
        if norm_type and "layernorm" in norm_type.lower().replace("_", ""):
            norm_type = "LayerNorm"
        elif norm_type and "rmsnorm" in norm_type.lower().replace("_", ""):
            norm_type = "RMSNorm"
        elif norm_type:
            norm_type = ""
        if not norm_type:
            # The epsilon field name is the checkpoint's own structural
            # declaration of its normalization family and is far more reliable
            # than matching class names: every RMSNorm family publishes
            # ``rms_norm_eps`` while LayerNorm families publish one of the
            # ``layer_norm_*``/``norm_epsilon`` spellings.
            if config.get("rms_norm_eps") is not None:
                norm_type = "RMSNorm"
            elif any(
                config.get(key) is not None
                for key in ("layer_norm_eps", "layer_norm_epsilon", "norm_epsilon", "layernorm_epsilon")
            ):
                norm_type = "LayerNorm"
            elif model_type == "nemotron" or "nemotron" in architecture_name:
                # Nemotron's published LayerNorm1P is a LayerNorm whose scale is
                # (1 + weight); its config spells the epsilon plainly as
                # ``norm_eps``, matching neither family's usual key.
                norm_type = "LayerNorm"
            else:
                norm_type = "LayerNorm" if any(
                    marker in model_type or marker in architecture_name
                    for marker in ("bert", "roberta", "deberta", "electra", "albert", "gpt2", "gpt_neo", "gpt_neox")
                ) else "RMSNorm"
        position_type = str(config.get("position_type", "")).strip()
        if not position_type:
            if bool(config.get("alibi", False)) or bool(attention_config.get("alibi", False)):
                position_type = "ALiBi"
            # BLOOM has no ``alibi`` switch: the architecture is defined with
            # ALiBi, so a missing field must not be read as "absolute".
            elif model_type == "bloom" or "bloom" in architecture_name:
                position_type = "ALiBi"
            # Match complete model types here.  ``gpt_neo`` is a prefix of
            # ``gpt_neox``; a substring test silently turns GPT-NeoX's RoPE
            # contract into GPT-Neo's learned absolute positions and then
            # requires a position table that the checkpoint does not contain.
            elif model_type in {"gpt2", "gpt_neo", "opt"} or architecture_name in {
                "gpt2", "gpt2lmheadmodel", "gptneoforcausallm", "optforcausallm",
            }:
                position_type = "absolute"
            else:
                position_type = "RoPE"
        raw_attention_layers = config.get("attention_layers")
        attention_layers: list[str] | None = None
        if isinstance(raw_attention_layers, list) and len(raw_attention_layers) == int(num_hidden_layers):
            if all(not isinstance(value, (list, tuple, dict)) for value in raw_attention_layers):
                attention_layers = [str(value) for value in raw_attention_layers]
        # Modern Transformers configs publish an explicit per-layer schedule as
        # ``layer_types`` with the values ``sliding_attention``/``full_attention``
        # (Gemma-2/3, EXAONE-4, Starcoder2, Ministral, ...).  Normalize it to the
        # AEG vocabulary so the runtime never infers a window from a family name.
        if attention_layers is None:
            raw_layer_types = config.get("layer_types")
            if isinstance(raw_layer_types, list) and len(raw_layer_types) == int(num_hidden_layers):
                attention_layers = [
                    "local" if "sliding" in str(value).lower() or "local" in str(value).lower()
                    else "global"
                    for value in raw_layer_types
                ]
        # GPT-Neo stores the same contract as repeated groups, for example
        # ``[[["global", "local"], 12]]``.  Expand it once at ingestion so
        # every executor receives an unambiguous per-layer schedule.
        if attention_layers is None:
            expanded: list[str] = []
            raw_attention_types = config.get("attention_types")
            if isinstance(raw_attention_types, list):
                try:
                    for group in raw_attention_types:
                        if not isinstance(group, (list, tuple)) or len(group) != 2:
                            raise ValueError
                        pattern, count = group
                        if isinstance(pattern, str):
                            pattern_values = [pattern]
                        elif isinstance(pattern, (list, tuple)) and pattern:
                            pattern_values = [str(value) for value in pattern]
                        else:
                            raise ValueError
                        for layer_index in range(int(count)):
                            expanded.append(pattern_values[layer_index % len(pattern_values)])
                    if len(expanded) == int(num_hidden_layers):
                        attention_layers = expanded
                except (TypeError, ValueError):
                    attention_layers = None
        raw_attention_window = config.get(
            "window_size", config.get("attention_window", config.get("sliding_window"))
        )
        try:
            attention_window = int(raw_attention_window) if raw_attention_window is not None else None
        except (TypeError, ValueError):
            attention_window = None
        # A declared window only takes effect where the schedule asks for it.
        # Qwen2 and Mistral publish a ``sliding_window`` value together with an
        # all-global schedule (or ``use_sliding_window: false``); applying it
        # would truncate attention the source model never truncated.
        if config.get("use_sliding_window") is False:
            attention_window = None
            if attention_layers is not None:
                attention_layers = ["global"] * len(attention_layers)
        if attention_window is not None and attention_layers is None:
            attention_window = None
        embedding_norm = bool(
            config.get("embedding_norm", False)
            or model_type == "bloom"
            or "bloom" in architecture_name
        )
        attention_type = str(config.get("attention_type", "")).strip()
        if not attention_type:
            if config.get("kv_lora_rank") is not None or "mla" in architecture_name or "mla" in model_type:
                attention_type = "MLA"
            elif num_kv_heads == num_attention_heads:
                attention_type = "MHA"
            else:
                attention_type = "GQA"
        qk_norm = bool(
            config.get("qk_norm", config.get("use_qk_norm", config.get("qk_layernorm", False)))
            or "qwen3" in architecture_name
            or model_type == "qwen3"
            # These families always normalize Q and K; the contract is
            # structural rather than switchable, so their configs declare no
            # flag for it.
            or model_type in {"olmo2", "olmo_2", "exaone4", "olmoe"}
            or model_type.startswith("gemma3")
            or any(
                marker in architecture_name
                for marker in ("olmo2", "exaone4", "gemma3", "olmoe")
            )
        )
        # Falcon's legacy multi-query attention shares a single K/V head across
        # every query head but publishes no ``num_kv_heads``.  Without this the
        # fused projection is split with the wrong group width.
        if (
            not config.get("new_decoder_architecture", False)
            and config.get("multi_query") is True
            and config.get("num_kv_heads") is None
            and config.get("num_key_value_heads") is None
        ):
            num_kv_heads = 1
        # GPT-J's published block computes attention and MLP from one shared
        # LayerNorm output and adds both branches to the residual together.
        # Preserve this structural capability in the AEG contract.
        parallel_residual = bool(
            config.get("parallel_residual", False)
            or config.get("use_parallel_residual", False)
            # Falcon spells the same block ``parallel_attn``; its new decoder
            # architecture is always parallel and carries two input norms.
            or config.get("parallel_attn", False)
            or config.get("new_decoder_architecture", False)
            or model_type in {"gptj", "gpt-j", "cohere", "cohere2"}
            or "gptj" in architecture_name.lower().replace("_", "")
            or "cohere" in architecture_name.lower()
        )

        numerics = self._detect_execution_numerics(
            config=config,
            attention_config=attention_config,
            model_type=model_type,
            architecture_name=architecture_name,
            hidden_size=int(hidden_size),
            num_attention_heads=int(num_attention_heads or 1),
            num_hidden_layers=int(num_hidden_layers),
            head_dim=head_dim,
            hidden_activation=hidden_act,
            attention_schedule=attention_layers,
        )

        # Detect MoE
        # HF model families use several names for the routed expert bank.
        # DeepSeek uses ``n_routed_experts`` while Mixtral/OLMoE commonly use
        # ``num_local_experts``.  Normalize them at the capability boundary so
        # downstream graph/runtime code never needs a family-name branch.
        num_experts = config.get(
            "num_local_experts",
            config.get("n_routed_experts", config.get("num_experts", 0)),
        )
        num_activated_experts = config.get(
            "num_experts_per_tok",
            config.get("num_experts_per_token", config.get("top_k", 0)),
        )
        try:
            num_experts = max(0, int(num_experts or 0))
        except (TypeError, ValueError):
            num_experts = 0
        try:
            num_activated_experts = max(0, int(num_activated_experts or 0))
        except (TypeError, ValueError):
            num_activated_experts = 0
        is_moe = num_experts > 0
        # Mixtral-style checkpoints are all-MoE; Jamba/DeepSeek-style
        # checkpoints can replace only a subset of layers.  Preserve the
        # declared pattern in the architecture contract instead of making
        # the runtime guess from a family name.
        if is_moe:
            first_dense = int(config.get("first_k_dense_replace", 0) or 0)
            frequency = int(config.get("moe_layer_frequency", 1) or 1)
            explicit_layers = config.get("moe_layer_indices", config.get("moe_layers"))
            if isinstance(explicit_layers, list):
                moe_layer_indices = [int(index) for index in explicit_layers]
            elif frequency > 1:
                moe_layer_indices = [
                    index for index in range(int(num_hidden_layers))
                    if index >= first_dense and index % frequency == 0
                ]
            else:
                moe_layer_indices = list(range(first_dense, int(num_hidden_layers)))
        else:
            moe_layer_indices = None
        is_encoder = (
            family in ("bert_family", "roberta_family", "deberta_family", "electra_family", "albert_family")
        ) or bool(config.get("is_encoder") is True)
        is_encoder_decoder = bool(
            family == "encoder_decoder_family"
            or config.get("is_encoder_decoder") is True
        )
        is_multimodal = bool(
            family == "vision_family"
            or family == "whisper_family"
            or
            config.get("vision_config") is not None
            or config.get("visual_config") is not None
            or any(marker in architecture_name for marker in ("vision", "vl", "visual", "audio"))
            or any(marker in model_type for marker in ("vision", "vl", "visual", "audio"))
        )
        mtp_declared = config.get(
            "mtp_heads",
            config.get("num_mtp_heads", config.get("num_nextn_predict_layers", 0)),
        )
        if isinstance(mtp_declared, dict):
            mtp_declared = mtp_declared.get("n_heads", mtp_declared.get("num_heads", 0))
        try:
            mtp_heads = max(0, int(mtp_declared or 0))
        except (TypeError, ValueError):
            mtp_heads = 0

        ssm_variant: str | None = None
        ssm_state_size: int | None = None
        ssm_inner_size: int | None = None
        ssm_dt_rank: int | None = None
        ssm_conv_kernel: int | None = None
        ssm_num_heads: int | None = None
        ssm_num_groups: int | None = None
        ssm_head_dim: int | None = None
        hybrid_layer_types: list[str] | None = None
        if family == "hybrid_ssm_family" or any(
            marker in model_type for marker in ("mamba", "rwkv", "jamba")
        ):
            if "rwkv" in model_type:
                ssm_variant = "rwkv_time_mix"
            elif "mamba2" in model_type or "mamba_2" in model_type:
                ssm_variant = "ssd"
            else:
                ssm_variant = "selective_scan"
            if "jamba" in model_type or "jamba" in architecture_name:
                # Jamba publishes either an explicit block schedule or an
                # attention period/offset. Preserve the schedule in the AEG;
                # the runtime must never guess a layer type from a family
                # name after compilation.
                explicit_schedule = config.get("layers_block_type", config.get("layer_types"))
                attention_indices = config.get("attention_layer_indices", config.get("attn_layer_indices"))
                if isinstance(explicit_schedule, list) and len(explicit_schedule) == int(num_hidden_layers):
                    hybrid_layer_types = [
                        "attention" if str(value).lower() in {"attention", "attn", "transformer"} else "ssm"
                        for value in explicit_schedule
                    ]
                else:
                    if isinstance(attention_indices, list):
                        attention_set = {int(value) for value in attention_indices}
                    else:
                        period = int(config.get("attn_layer_period", config.get("attention_layer_period", 8)) or 8)
                        offset = int(config.get("attn_layer_offset", max(0, period - 1)) or 0)
                        attention_set = set(range(offset, int(num_hidden_layers), max(period, 1)))
                    hybrid_layer_types = [
                        "attention" if index in attention_set else "ssm"
                        for index in range(int(num_hidden_layers))
                    ]
                ssm_variant = "hybrid_selective_scan"
            ssm_state_size = int(config.get("d_state", config.get("state_size", 16)) or 16)
            # Mamba-2 names its expanded channel geometry ``n_heads`` ×
            # ``headdim``; Mamba-1 uses the simpler ``d_inner`` contract.
            # Preserve both forms in the artifact instead of making the
            # runtime infer a Qwen/transformer-shaped dimension.
            ssm_num_heads = int(config.get("n_heads", config.get("num_ssm_heads", num_attention_heads)) or 0)
            ssm_num_groups = int(config.get("n_groups", config.get("num_ssm_groups", 1)) or 1)
            ssm_head_dim = config.get("headdim", config.get("ssm_head_dim"))
            try:
                ssm_head_dim = int(ssm_head_dim) if ssm_head_dim is not None else None
            except (TypeError, ValueError):
                ssm_head_dim = None
            configured_inner = config.get("d_inner")
            if configured_inner is None and ssm_head_dim and ssm_num_heads:
                configured_inner = ssm_head_dim * ssm_num_heads
            if configured_inner is None:
                configured_inner = config.get("intermediate_size")
            ssm_inner_size = int(configured_inner or int(hidden_size) * 2)
            if ssm_head_dim is None and ssm_num_heads > 0 and ssm_inner_size % ssm_num_heads == 0:
                ssm_head_dim = ssm_inner_size // ssm_num_heads
            raw_dt_rank = config.get("dt_rank", config.get("time_step_rank", "auto"))
            ssm_dt_rank = (int(hidden_size) + 15) // 16 if raw_dt_rank == "auto" else int(raw_dt_rank)
            ssm_conv_kernel = int(config.get("d_conv", config.get("conv_kernel", 4)) or 4)

        return ModelArchitecture(
            family=family,
            params_billion=0.0,
            layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            context_length=context_length,
            vocab_size=vocab_size,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            parallel_residual=parallel_residual,
            intermediate_size=intermediate_size,
            is_moe=is_moe,
            is_encoder=is_encoder,
            is_encoder_decoder=is_encoder_decoder,
            is_multimodal=is_multimodal,
            encoder_layers=int(encoder_layers) if is_encoder_decoder else None,
            decoder_layers=int(decoder_layers) if is_encoder_decoder else None,
            tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
            relative_attention_num_buckets=int(config.get("relative_attention_num_buckets", 32)),
            num_experts=num_experts,
            num_activated_experts=num_activated_experts,
            moe_layer_indices=moe_layer_indices,
            mtp_heads=mtp_heads,
            attention_type=attention_type,
            mla_kv_lora_rank=(
                int(config["kv_lora_rank"]) if config.get("kv_lora_rank") is not None else None
            ),
            mla_q_lora_rank=(
                int(config["q_lora_rank"]) if config.get("q_lora_rank") is not None else None
            ),
            mla_qk_nope_head_dim=(
                int(config["qk_nope_head_dim"])
                if config.get("qk_nope_head_dim") is not None else None
            ),
            mla_qk_rope_head_dim=(
                int(config["qk_rope_head_dim"])
                if config.get("qk_rope_head_dim") is not None else None
            ),
            mla_v_head_dim=(
                int(config["v_head_dim"]) if config.get("v_head_dim") is not None else None
            ),
            ssm_variant=ssm_variant,
            ssm_state_size=ssm_state_size,
            ssm_inner_size=ssm_inner_size,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv_kernel=ssm_conv_kernel,
            ssm_num_heads=ssm_num_heads,
            ssm_num_groups=ssm_num_groups,
            ssm_head_dim=ssm_head_dim,
            hybrid_layer_types=hybrid_layer_types,
            ffn_type=ffn_type,
            norm_type=norm_type,
            position_type=position_type,
            attention_layers=attention_layers,
            attention_window=attention_window,
            embedding_norm=embedding_norm,
            **numerics,
        )

    @staticmethod
    def _detect_execution_numerics(
        *,
        config: dict[str, Any],
        attention_config: dict[str, Any],
        model_type: str,
        architecture_name: str,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int | None,
        hidden_activation: str = "",
        attention_schedule: list[str] | None = None,
    ) -> dict[str, Any]:
        """Derive the scalar and structural execution constants of a decoder.

        Every value here is read from the source configuration.  These are not
        tuning knobs: each one is part of the model's definition, and omitting
        one changes every logit the artifact produces.  Keeping them in the AEG
        manifest is what lets a single executor run any family without
        consulting a model class at runtime.

        References are the published architectures: unscaled attention in
        GPT-Neo (Black et al. 2021), partial rotary embeddings in GPT-NeoX (Black
        et al. 2022) and the interleaved rotary convention in GPT-J (Wang & Komatsuzaki
        2021), logit soft-capping and sandwich normalization in Gemma 2 (Gemma
        Team 2024), local/global interleaving and a separate local rotary base in
        Gemma 3 (Gemma Team 2025), output-normalized blocks in OLMo 2 (OLMo Team
        2024), and the explicit multiplier set in Granite (IBM 2024).
        """
        effective_head_dim = int(head_dim or max(hidden_size // max(num_attention_heads, 1), 1))

        def numeric(*keys: str) -> float | None:
            for key in keys:
                value = config.get(key, attention_config.get(key))
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        # ── Attention softmax scale ────────────────────────────────────────
        attention_scale: float | None = None
        explicit_softmax_scale = numeric("softmax_scale", "attention_softmax_scale")
        query_pre_attn_scalar = numeric("query_pre_attn_scalar")
        attention_multiplier = numeric("attention_multiplier")
        if explicit_softmax_scale is not None and explicit_softmax_scale > 0:
            attention_scale = explicit_softmax_scale
        elif query_pre_attn_scalar is not None and query_pre_attn_scalar > 0:
            # Gemma-2/3 normalize queries by a configured scalar that is
            # deliberately decoupled from the head width.
            attention_scale = float(query_pre_attn_scalar) ** -0.5
        elif attention_multiplier is not None and attention_multiplier > 0:
            attention_scale = float(attention_multiplier)
        elif config.get("scale_attn_weights") is False:
            attention_scale = 1.0
        elif model_type == "gpt_neo" or architecture_name in {
            "gptneoforcausallm", "gptneomodel",
        }:
            # GPT-Neo's published attention omits the 1/sqrt(d) factor
            # entirely; it trained that way and depends on it.  Match the
            # complete name: ``gpt_neo`` is a prefix of ``gpt_neox``, which
            # does use the standard scale.
            attention_scale = 1.0
        if attention_scale is not None and attention_scale <= 0:
            attention_scale = None

        # ── Embedding / residual / logit multipliers ───────────────────────
        embedding_scale = numeric("embedding_multiplier")
        if embedding_scale is not None and embedding_scale == 1.0:
            embedding_scale = None
        if embedding_scale is None and (
            model_type.startswith("gemma") or "gemma" in architecture_name
        ):
            # Gemma scales the embedding table by sqrt(hidden_size), applied in
            # the model's compute dtype.
            embedding_scale = float(hidden_size) ** 0.5
        residual_scale = numeric("residual_multiplier")
        if residual_scale is not None and residual_scale == 1.0:
            residual_scale = None
        logit_scale = numeric("logit_scale")
        logits_scaling = numeric("logits_scaling")
        if logits_scaling is not None and logits_scaling not in (0.0, 1.0):
            # Granite divides the logits; express it as one multiplier.
            logit_scale = (1.0 / logits_scaling) * (logit_scale or 1.0)
        if logit_scale is not None and logit_scale == 1.0:
            logit_scale = None

        attn_logit_softcap = numeric("attn_logit_softcapping")
        final_logit_softcap = numeric("final_logit_softcapping")
        if attn_logit_softcap is not None and attn_logit_softcap <= 0:
            attn_logit_softcap = None
        if final_logit_softcap is not None and final_logit_softcap <= 0:
            final_logit_softcap = None

        # ── Normalization form and placement ──────────────────────────────
        # Gemma stores normalization weights as offsets from unity, so the
        # scale applied is (1 + w).  Reading them as plain weights zeroes the
        # residual stream.
        norm_offset_one = bool(
            model_type.startswith("gemma")
            or "gemma" in architecture_name
            # Nemotron's LayerNorm1P applies (1 + weight) exactly as Gemma's
            # RMSNorm does, so one offset flag covers both.
            or model_type == "nemotron"
            or "nemotron" in architecture_name
            or config.get("norm_offset_one", False)
        )
        declared_placement = str(config.get("norm_placement", "") or "").strip().lower()
        if declared_placement in {"pre", "post", "sandwich"}:
            norm_placement = declared_placement
        elif model_type.startswith("glm4") or "glm4" in architecture_name:
            # Same four-norm block as Gemma-2, different tensor spelling; the
            # variant is recorded so ingestion binds the right norm to the
            # right slot.
            norm_placement = "sandwich_glm"
        elif (
            model_type.startswith("gemma2")
            or model_type.startswith("gemma3")
            or "gemma2" in architecture_name
            or "gemma3" in architecture_name
        ):
            # Gemma-2/3 normalize both the input and the output of each
            # sublayer (four norms per block).
            norm_placement = "sandwich"
        elif (
            model_type in {"olmo2", "olmo_2", "exaone4"}
            or "olmo2" in architecture_name
            or "exaone4" in architecture_name
        ):
            # OLMo-2 and EXAONE-4 removed the input norms and normalize each
            # sublayer's output before the residual add (two norms per block).
            norm_placement = "post"
        else:
            norm_placement = "pre"

        # OLMo-2 normalizes the whole Q/K projection rather than each head.
        # EXAONE-4 shares the placement but keeps per-head Q/K norms.
        qk_norm_scope = "full" if (
            model_type in {"olmo2", "olmo_2", "olmoe"}
            or "olmo2" in architecture_name
            or "olmoe" in architecture_name
        ) else "head"

        # ── Rotary geometry ───────────────────────────────────────────────
        rope_partial_dim: int | None = None
        rotary_dim = config.get("rotary_dim")
        partial_factor = numeric("partial_rotary_factor", "rotary_pct")
        if rotary_dim is not None:
            try:
                rope_partial_dim = int(rotary_dim)
            except (TypeError, ValueError):
                rope_partial_dim = None
        elif partial_factor is not None and 0.0 < partial_factor < 1.0:
            rope_partial_dim = int(effective_head_dim * partial_factor)
        if rope_partial_dim is not None:
            # An even count is required: RoPE rotates channel pairs.
            rope_partial_dim = max(2, (min(rope_partial_dim, effective_head_dim) // 2) * 2)
            if rope_partial_dim >= effective_head_dim:
                rope_partial_dim = None

        # GPT-J pairs adjacent channels; GPT-NeoX/Llama pair the two halves.
        # Cohere/Command-R and GLM/GLM-4 also publish the interleaved form
        # (their reference implementations build the table with
        # repeat_interleave rather than concatenation).
        rope_interleaved = bool(
            config.get("rope_interleaved", False)
            or model_type in {"gptj", "gpt-j", "gpt_j", "cohere", "cohere2", "glm", "glm4"}
            or "gptj" in architecture_name.replace("_", "").replace("-", "")
            or "cohere" in architecture_name
            or "glm4" in architecture_name
            or architecture_name.startswith("glm")
        )
        rope_local_theta = numeric("rope_local_base_freq", "rope_theta_local")

        # ── Fused QKV memory layout ───────────────────────────────────────
        if config.get("new_decoder_architecture"):
            # Falcon's new architecture stores each KV group as
            # [q * heads_per_group, k, v].
            fused_qkv_layout = "group_interleaved"
        elif model_type in {"gpt_neox", "bloom", "falcon", "refinedweb", "refinedwebmodel"} or (
            "gptneox" in architecture_name.replace("_", "")
            or "bloom" in architecture_name
            or "falcon" in architecture_name
        ):
            # GPT-NeoX and BLOOM interleave [q, k, v] per head.
            fused_qkv_layout = "head_interleaved"
        else:
            fused_qkv_layout = "contiguous"

        # ── NoPE layers ───────────────────────────────────────────────────
        no_rope_layers: list[int] | None = None
        raw_no_rope = config.get("no_rope_layers")
        if isinstance(raw_no_rope, list) and len(raw_no_rope) == num_hidden_layers:
            # The published field is a per-layer *enable* flag: a falsy entry
            # marks a layer that applies no rotation at all.
            skipped = [index for index, flag in enumerate(raw_no_rope) if not flag]
            no_rope_layers = skipped or None
        if no_rope_layers is None and attention_schedule and config.get("sliding_window"):
            # EXAONE-4's published hybrid attention rotates only its
            # sliding-window layers and leaves the global layers without any
            # positional encoding.  The indices follow from the schedule.
            if model_type == "exaone4" or "exaone4" in architecture_name:
                global_layers = [
                    index
                    for index, kind in enumerate(attention_schedule)
                    if kind != "local"
                ]
                no_rope_layers = global_layers or None

        return {
            "attention_scale": attention_scale,
            "attention_scale_by_layer_index": bool(
                config.get("scale_attn_by_inverse_layer_idx", False)
            ),
            "embedding_scale": embedding_scale,
            "residual_scale": residual_scale,
            "logit_scale": logit_scale,
            "attn_logit_softcap": attn_logit_softcap,
            "final_logit_softcap": final_logit_softcap,
            "norm_offset_one": norm_offset_one,
            "rope_partial_dim": rope_partial_dim,
            "rope_interleaved": rope_interleaved,
            "rope_local_theta": rope_local_theta,
            "norm_placement": norm_placement,
            "qk_norm_scope": qk_norm_scope,
            "fused_qkv_layout": fused_qkv_layout,
            "no_rope_layers": no_rope_layers,
            # ``gelu`` is the exact error-function form; every ``gelu_new`` /
            # ``gelu_pytorch_tanh`` / ``gelu_fast`` spelling is the tanh
            # approximation.  BLOOM's built-in activation is also the tanh form
            # even though its config names no activation at all.
            # Mixtral and DeepSeek renormalize the selected top-k routing
            # weights; Qwen3-MoE and OLMoE publish ``norm_topk_prob: false``
            # and keep the full-softmax probabilities.
            "moe_renormalize_topk": bool(
                config.get("norm_topk_prob", config.get("normalize_expert_weights", True))
            ),
            "gelu_approximate": bool(
                any(
                    marker in hidden_activation
                    for marker in ("new", "tanh", "fast", "quick")
                )
                or model_type == "bloom"
                or "bloom" in architecture_name
                or not hidden_activation
            ),
        }

    @staticmethod
    def _is_generic_decoder_config(config: dict[str, Any], arch_type: str) -> bool:
        """Return whether an unknown config explicitly declares a decoder LM.

        This is intentionally conservative.  It accepts only causal language
        model class names or an explicit decoder flag, and rejects multimodal
        and encoder/decoder class contracts that require a different graph.
        """
        normalized_arch = arch_type.lower().replace("_", "")
        model_type = str(config.get("model_type", "")).lower()
        if any(marker in normalized_arch for marker in ("vision", "vl", "conditionalgeneration", "encoderdecoder")):
            return False
        return bool(
            "causallm" in normalized_arch
            or normalized_arch.endswith("lmheadmodel")
            or config.get("is_decoder") is True
            or config.get("is_encoder_decoder") is False and model_type.endswith("lm")
        )

    def _detect_family_from_arch_type(self, arch_type: str) -> str | None:
        """Map a HuggingFace architecture type or GGUF type to an Aether family."""
        mapping = {
            "LlamaForCausalLM": "llama_family",
            "Qwen2ForCausalLM": "qwen_family",
            "Qwen3ForCausalLM": "qwen_family",
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
            "MambaForCausalLM": "hybrid_ssm_family",
            "JambaForCausalLM": "hybrid_ssm_family",
            "RwkvForCausalLM": "hybrid_ssm_family",
            "GPT2Model": "gpt_family",
            "GPT2LMHeadModel": "gpt_family",
            "GPTNeoModel": "gpt_family",
            "GPTNeoForCausalLM": "gpt_family",
            "GPTNeoXModel": "gpt_family",
            "GPTNeoXForCausalLM": "gpt_family",
            # Encoder architectures:
            "BertModel": "bert_family",
            "BertForMaskedLM": "bert_family",
            "BertForSequenceClassification": "bert_family",
            "RobertaModel": "roberta_family",
            "RobertaForMaskedLM": "roberta_family",
            "RobertaForSequenceClassification": "roberta_family",
            "DebertaModel": "deberta_family",
            "DebertaV2Model": "deberta_family",
            "ElectraModel": "electra_family",
            "ElectraForMaskedLM": "electra_family",
            "AlbertModel": "albert_family",
            "MPNetModel": "bert_family",
            "MPNetForMaskedLM": "bert_family",
            "MPNetForSequenceClassification": "bert_family",
            # Short / model_type aliases:
            "llama": "llama_family",
            "llama2": "llama_family",
            "llama3": "llama_family",
            "qwen": "qwen_family",
            "qwen2": "qwen_family",
            "qwen3": "qwen_family",
            "qwen2_moe": "moe_family",
            "qwen2_vl": "qwen_family",
            "qwen2_5_vl": "qwen_family",
            "qwen3_vl": "qwen_family",
            "gemma": "gemma_family",
            "gemma2": "gemma_family",
            "gemma3": "gemma_family",
            "mistral": "mistral_family",
            "mixtral": "moe_family",
            "deepseek": "deepseek_family",
            "deepseek_v2": "deepseek_family",
            "deepseek_v3": "deepseek_family",
            "deepseek_vl": "deepseek_family",
            "phi": "phi_family",
            "phi3": "phi_family",
            "phi4": "phi_family",
            "falcon": "falcon_family",
            "whisper": "whisper_family",
            "vit": "vision_family",
            "llava": "vision_family",
            "internvl": "vision_family",
            "mamba": "hybrid_ssm_family",
            "jamba": "hybrid_ssm_family",
            "rwkv": "hybrid_ssm_family",
            "bert": "bert_family",
            "roberta": "roberta_family",
            "deberta": "deberta_family",
            "deberta_v2": "deberta_family",
            "deberta-v2": "deberta_family",
            "electra": "electra_family",
            "albert": "albert_family",
            "mpnet": "bert_family",
            "distilbert": "bert_family",
            "gpt2": "gpt_family",
            "gpt_neo": "gpt_family",
            "gpt-neox": "gpt_family",
        }
        if arch_type in mapping:
            return mapping[arch_type]
        normalized = arch_type.lower().replace("-", "_").replace(".", "_")
        if normalized in mapping:
            return mapping[normalized]
        # Keep the registry extensible without duplicating every Hugging Face
        # class spelling (for example OLMoForCausalLM, GraniteForCausalLM,
        # Starcoder2ForCausalLM, and code-model variants).  The family is a
        # capability classification; all dimensions still come from config.
        normalized_compact = normalized.replace("_", "")
        for key, fam in sorted(
            ARCHITECTURE_BY_MODEL_PREFIX.items(),
            key=lambda item: len(item[0].replace("_", "")),
            reverse=True,
        ):
            if key.replace("_", "") in normalized_compact:
                return fam
        # Keep explicit class-name aliases as a last fallback.  Broad aliases
        # such as ``phi`` must not match an unrelated class name accidentally.
        for key, fam in mapping.items():
            key_compact = key.lower().replace("-", "").replace("_", "")
            if key_compact and key_compact in normalized_compact:
                return fam
        return None

    def _match_family(self, model: str) -> str | None:
        """Match a model name to an architecture family."""
        lower = model.lower().replace("-", "").replace("_", "")
        for name_part, family in sorted(
            ARCHITECTURE_BY_MODEL_PREFIX.items(),
            key=lambda item: len(item[0].replace("_", "")),
            reverse=True,
        ):
            if name_part.replace("_", "") in lower:
                return family
        return None

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

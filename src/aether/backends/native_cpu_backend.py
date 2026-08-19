"""Framework-free execution backend for packaged AEG artifacts.

The native CPU engine is part of Aether's core runtime contract. It must not
be hidden behind the optional PyTorch backend: a user who installs the base
package and receives a compiled AEG must be able to load that artifact without
importing PyTorch or Transformers.

This module reuses the compiled-AEG request handling in ``TorchBackend`` (the
module does not import torch at import time), while providing its own model
loader and tokenizer adapter. The forward pass is always performed by
``CPUExecutionEngine``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aether.backends.base import BackendInfo
from aether.backends.torch_backend import CompiledAEGHandle, TorchBackend
from aether.core.exceptions import BackendError
from aether.runtime.aeg_loader import load_engine_from_path, package_is_runnable


class PackagedTokenizer:
    """Adapter around a serialized Hugging Face ``tokenizer.json``."""

    def __init__(self, tokenizer_path: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - packaging error path
            raise BackendError(
                "The AEG tokenizer requires the 'tokenizers' package; install "
                "aether-runtime with its base dependencies.",
                backend_name="aether_cpu",
            ) from exc
        if not tokenizer_path.is_file():
            raise BackendError(
                f"AEG tokenizer file is missing: {tokenizer_path}",
                backend_name="aether_cpu",
            )
        try:
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:  # noqa: BLE001 - normalize untrusted artifact errors
            raise BackendError(
                f"Unable to load packaged tokenizer {tokenizer_path}: {exc}",
                backend_name="aether_cpu",
            ) from exc
        config = self._read_json(tokenizer_path.with_name("tokenizer_config.json"))
        special = self._read_json(tokenizer_path.with_name("special_tokens_map.json"))
        self.eos_token_id = self._special_id(config.get("eos_token"), special.get("eos_token"))
        self.pad_token_id = self._special_id(config.get("pad_token"), special.get("pad_token"))
        self.bos_token_id = self._special_id(config.get("bos_token"), special.get("bos_token"))
        self.chat_template = config.get("chat_template")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _special_id(self, *values: Any) -> int | None:
        for value in values:
            token = value
            if isinstance(value, dict):
                token = value.get("content") or value.get("token")
            if not isinstance(token, str):
                continue
            token_id = self._tokenizer.token_to_id(token)
            if token_id is not None:
                return int(token_id)
        return None

    def __len__(self) -> int:
        """Return the serialized tokenizer vocabulary size.

        ``len(tokenizer)`` is the common Hugging Face/tokenizers contract and
        is also the identity input used by the grammar compiler's tokenizer
        fingerprint.  Keeping this on the adapter ensures a grammar compiled
        before process restart is verified against the exact packaged
        vocabulary rather than being accepted on vocabulary size alone.
        """
        return int(self._tokenizer.get_vocab_size())

    def get_vocab_size(self, *, with_added_tokens: bool = True) -> int:
        """Expose the tokenizers vocabulary-size API used by runtime passes."""
        return int(self._tokenizer.get_vocab_size(with_added_tokens=with_added_tokens))

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str = "np",
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        if return_tensors not in {"np", "numpy"}:
            raise ValueError("PackagedTokenizer supports return_tensors='np' only")
        texts = text if isinstance(text, list) else [text]
        encodings = [self._tokenizer.encode(value, add_special_tokens=True) for value in texts]
        ids = [encoding.ids for encoding in encodings]
        if truncation and max_length is not None:
            ids = [row[:max_length] for row in ids]
        if padding or len(ids) > 1:
            width = max((len(row) for row in ids), default=0)
            if self.pad_token_id is None:
                raise ValueError("batched packaged-tokenizer input requires a pad token")
            ids = [row + [self.pad_token_id] * (width - len(row)) for row in ids]
        if not ids:
            ids = [[]]
        return {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(
                [[1 if token != self.pad_token_id else 0 for token in row] for row in ids],
                dtype=np.int64,
            ),
        }

    def decode(
        self,
        ids: Any,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        values = np.asarray(ids, dtype=np.int64).reshape(-1).tolist()
        try:
            return self._tokenizer.decode(
                [int(value) for value in values],
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
        except TypeError:
            # ``tokenizers.Tokenizer.decode`` versions before 0.20 do not
            # expose the Transformers-only cleanup keyword.
            return self._tokenizer.decode(
                [int(value) for value in values],
                skip_special_tokens=skip_special_tokens,
            )


class NativeCPUBackend(TorchBackend):
    """Aether-native CPU backend for self-contained AEG execution."""

    def __init__(self) -> None:
        # Bypass TorchBackend.__init__: optional torch device discovery must not
        # determine whether the framework-free backend is available.
        from aether.backends.base import Backend

        Backend.__init__(
            self,
            BackendInfo(
                name="aether_cpu",
                version="1.0.0",
                supported_targets=[
                    "cpu_avx2", "cpu_avx512", "cpu_neon",
                    "cpu_avx512_ternary", "cpu_neon_ternary",
                ],
                capabilities=[
                    "generate", "chat", "stream", "kv_cache",
                    "grammar_constraints", "structured_output",
                    "packaged_aeg", "framework_free",
                ],
            ),
        )
        self._models: dict[str, CompiledAEGHandle] = {}
        self._tokenizers: dict[str, PackagedTokenizer] = {}
        self._device = "cpu"
        self._allow_remote_code = False

    def is_available(self) -> bool:
        """The backend is available whenever the base NumPy runtime is usable."""
        return True

    def load_model(self, model_id: str, aeg_path: str | None = None, **_: Any) -> Any:
        """Load and authenticate a self-contained AEG package."""
        if model_id in self._models:
            return self._models[model_id]
        if aeg_path is None:
            raise BackendError(
                "aether_cpu executes compiled .aeg artifacts only; compile the "
                "source model first or select an installed frontend backend",
                backend_name=self.name,
            )
        root = Path(aeg_path).resolve()
        try:
            from aether.core.aeg_format import AEGPackage
            from aether.adapters.lora import load_compiled_lora_adapters

            package = AEGPackage(root).load()
            package.verify_integrity()
            eval_report_path = root / "observability" / "eval_report.json"
            if eval_report_path.is_file():
                eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
                gate = eval_report.get("gate", {})
                if gate.get("passed") is False:
                    failing = gate.get("failing_benchmarks", [])
                    raise BackendError(
                        "AEG artifact is rejected by its persisted evaluation gate: "
                        + ", ".join(str(item) for item in failing),
                        backend_name=self.name,
                    )
            if not package_is_runnable(package):
                raise BackendError(f"AEG artifact {root} is not executable", backend_name=self.name)
            engine = load_engine_from_path(root)
            tokenizer = PackagedTokenizer(root / "tokenizer" / "tokenizer.json")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            metadata_path = root / "graph" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            precision_path = root / "weights" / "quantized" / "precision_map.json"
            precision_map = json.loads(precision_path.read_text(encoding="utf-8")) if precision_path.is_file() else {}
            handle = CompiledAEGHandle(
                model_id=model_id,
                aeg_path=root,
                manifest=manifest,
                metadata=metadata,
                precision_map=precision_map,
                engine=engine,
                tokenizer=tokenizer,
                lora_adapters=load_compiled_lora_adapters(root),
            )
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed at artifact boundary
            raise BackendError(
                f"AEG artifact {root} failed native CPU integrity/load validation: {exc}",
                backend_name=self.name,
            ) from exc
        self._models[model_id] = handle
        self._tokenizers[model_id] = tokenizer
        return handle

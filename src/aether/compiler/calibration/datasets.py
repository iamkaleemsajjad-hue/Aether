"""
Calibration datasets for sensitivity analysis and perplexity evaluation.

Provides small, permissive datasets (e.g., WikiText-2, Hellaswag) and support
for custom JSONL datasets. The actual dataset content is downloaded lazily or
provided by the user; this module provides a unified interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Callable, Iterator

from aether.core.exceptions import CalibrationError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


def _load_named_dataset(
    *,
    name: str,
    path: Path | None,
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    max_tokens: int,
    cache: Any,
    formatter: Callable[[Any], str] | None = None,
) -> Iterator[str]:
    """Load a named corpus from a local file or an already cached HF dataset.

    Network access is intentionally not attempted by compiler calibration.
    Reproducible/offline builds must receive a local corpus or have the exact
    dataset cached ahead of time.
    """
    if cache._samples is None:
        samples: list[str] = []
        if path is not None:
            if not path.is_file():
                raise CalibrationError(f"Calibration dataset file not found: {path}")
            try:
                if path.suffix.lower() in {".jsonl", ".json"}:
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            value = json.loads(line)
                            text = formatter(value) if formatter is not None else (
                                value.get("text", "") if isinstance(value, dict) else str(value)
                            )
                            if text.strip():
                                samples.append(text.strip())
                else:
                    samples = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CalibrationError(f"Unable to read {name} calibration file {path}: {exc}") from exc
        else:
            try:
                # Avoid the HF client's retry/backoff path when this process
                # is offline and no dataset cache exists.  This keeps a
                # compiler invocation deterministic and fast on air-gapped
                # hosts.
                hf_cache = Path(
                    os.environ.get(
                        "HF_DATASETS_CACHE",
                        Path(os.environ.get("HF_HOME", Path.home() / ".cache")) / "huggingface" / "datasets",
                    )
                )
                if not hf_cache.exists() or not any(hf_cache.iterdir()):
                    raise CalibrationError(
                        f"no local Hugging Face dataset cache at {hf_cache}"
                    )
                from datasets import DownloadConfig, load_dataset  # type: ignore[import]

                dataset = load_dataset(
                    dataset_id,
                    name=dataset_config,
                    split=split,
                    download_config=DownloadConfig(local_files_only=True, max_retries=0),
                )
                for row in dataset:
                    text = formatter(row) if formatter is not None else str(row.get("text", ""))
                    if text.strip():
                        samples.append(text.strip())
            except Exception as exc:  # noqa: BLE001
                raise CalibrationError(
                    f"Real {name} data is unavailable locally; provide a calibration file or cache "
                    f"the {dataset_id!r} dataset before compiling ({exc})"
                ) from exc
        if not samples:
            raise CalibrationError(f"Calibration dataset {name!r} produced no samples")
        cache._samples = samples

    emitted = 0
    for sample in cache._samples:
        remaining = max_tokens - emitted
        if remaining <= 0:
            break
        words = sample.split()
        if len(words) > remaining:
            yield " ".join(words[:remaining])
            break
        yield sample
        emitted += len(words)


def _format_hellaswag_row(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    context = str(row.get("ctx", "")).strip()
    endings = row.get("endings")
    label = row.get("label")
    if not context or not isinstance(endings, (list, tuple)):
        return ""
    try:
        ending = endings[int(label)]
    except (TypeError, ValueError, IndexError):
        return ""
    return f"{context} {ending}".strip()


class CalibrationDataset:
    """Base class for calibration datasets."""

    def __init__(self, name: str, max_tokens: int = 131072) -> None:
        self.name = name
        self.max_tokens = max_tokens

    def iter_text(self) -> Iterator[str]:
        """Yield calibration text samples."""
        raise NotImplementedError

    def iter_limited_text(self) -> Iterator[str]:
        """Yield samples until the configured token budget is exhausted.

        Calibration passes need deterministic, bounded input so dry-run and CI
        compilation never accidentally consume a full external dataset. The
        tokenization here intentionally uses whitespace because this package
        cannot assume a model tokenizer is available during graph-only passes.
        """
        emitted_tokens = 0
        for text in self.iter_text():
            words = text.split()
            if not words:
                continue
            remaining = self.max_tokens - emitted_tokens
            if remaining <= 0:
                break
            if len(words) > remaining:
                yield " ".join(words[:remaining])
                break
            yield text
            emitted_tokens += len(words)

    def token_count(self) -> int:
        """Return the bounded whitespace token count used by calibration."""
        return sum(len(text.split()) for text in self.iter_limited_text())

    def __repr__(self) -> str:
        return f"CalibrationDataset({self.name})"


class WikiText2Dataset(CalibrationDataset):
    """WikiText-2 calibration dataset."""

    def __init__(self, max_tokens: int = 131072, path: str | Path | None = None) -> None:
        super().__init__("wikitext-2", max_tokens=max_tokens)
        self.path = Path(path) if path is not None else None
        self._samples: list[str] | None = None

    def iter_text(self) -> Iterator[str]:
        """Yield real WikiText-2 samples from a local file or HF cache.

        Calibration must never silently substitute hand-written prose for a
        named benchmark.  ``path`` may point to JSONL (``text`` field) or a
        plain-text file.  Without a local corpus, the failure explicitly asks
        the caller to provide one or pre-cache the dataset.
        """
        yield from _load_named_dataset(
            name="wikitext-2",
            path=self.path,
            dataset_id="wikitext",
            dataset_config="wikitext-2-raw-v1",
            split="train",
            max_tokens=self.max_tokens,
            cache=self,
        )


class HellaswagDataset(CalibrationDataset):
    """Hellaswag calibration dataset for commonsense evaluation."""

    def __init__(self, max_tokens: int = 131072, path: str | Path | None = None) -> None:
        super().__init__("hellaswag", max_tokens=max_tokens)
        self.path = Path(path) if path is not None else None
        self._samples: list[str] | None = None

    def iter_text(self) -> Iterator[str]:
        """Yield real HellaSwag contexts and labelled endings."""
        yield from _load_named_dataset(
            name="hellaswag",
            path=self.path,
            dataset_id="Rowan/hellaswag",
            dataset_config=None,
            split="train",
            max_tokens=self.max_tokens,
            cache=self,
            formatter=lambda row: _format_hellaswag_row(row),
        )


class InlineCalibrationDataset(CalibrationDataset):
    """Calibration dataset backed by an in-memory iterable of text samples."""

    def __init__(self, samples: Iterable[str], name: str = "inline", max_tokens: int = 131072) -> None:
        super().__init__(name, max_tokens=max_tokens)
        self.samples = [sample for sample in samples if sample.strip()]

    def iter_text(self) -> Iterator[str]:
        """Yield user-provided text samples."""
        yield from self.samples


class CustomJsonlDataset(CalibrationDataset):
    """Custom calibration dataset from a JSONL file."""

    def __init__(self, path: str | Path, max_tokens: int = 131072, text_key: str = "text") -> None:
        super().__init__(f"custom:{path}", max_tokens=max_tokens)
        self.path = Path(path)
        self.text_key = text_key
        if not self.path.exists():
            msg = f"Calibration dataset file not found: {self.path}"
            raise CalibrationError(msg)

    def iter_text(self) -> Iterator[str]:
        """Yield text samples from the JSONL file."""
        emitted_tokens = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"Invalid JSON in calibration dataset: {line}"
                    raise CalibrationError(msg) from exc
                if isinstance(obj, dict):
                    text = obj.get(self.text_key, "")
                elif isinstance(obj, str):
                    text = obj
                else:
                    text = ""
                if not text:
                    continue
                words = text.split()
                remaining = self.max_tokens - emitted_tokens
                if remaining <= 0:
                    break
                if len(words) > remaining:
                    yield " ".join(words[:remaining])
                    break
                yield text
                emitted_tokens += len(words)


def get_dataset(name: str, max_tokens: int = 131072, path: str | Path | None = None) -> CalibrationDataset:
    """Factory for calibration datasets."""
    if name == "wikitext-2":
        return WikiText2Dataset(max_tokens=max_tokens, path=path)
    if name == "hellaswag":
        return HellaswagDataset(max_tokens=max_tokens, path=path)
    if name == "custom" and path is not None:
        return CustomJsonlDataset(path, max_tokens=max_tokens)
    if path is not None:
        return CustomJsonlDataset(path, max_tokens=max_tokens)
    msg = f"Unknown calibration dataset: {name}"
    raise CalibrationError(msg)

"""
Calibration datasets for sensitivity analysis and perplexity evaluation.

Provides small, permissive datasets (e.g., WikiText-2, Hellaswag) and support
for custom JSONL datasets. The actual dataset content is downloaded lazily or
provided by the user; this module provides a unified interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Iterator

from aether.core.exceptions import CalibrationError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


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

    def __init__(self, max_tokens: int = 131072) -> None:
        super().__init__("wikitext-2", max_tokens=max_tokens)

    def iter_text(self) -> Iterator[str]:
        """Yield synthetic samples representing WikiText-2 passages."""
        samples = [
            "The quick brown fox jumps over the lazy dog. This sentence is a well-known pangram.",
            "Machine learning models are trained on large datasets to recognize patterns and make predictions.",
            "The transformer architecture has revolutionized natural language processing and many other fields.",
            "Efficient inference requires careful optimization of memory bandwidth, compute utilization, and scheduling.",
            "Quantization reduces model size by representing weights with fewer bits, often with minimal quality loss.",
            "Compiler intermediate representations preserve semantic structure while allowing hardware specific lowering.",
            "Autoregressive decoding alternates memory bound key value cache reads with small matrix multiplications.",
            "Graph level optimizations are most effective when operation intent is retained until late lowering.",
            "Mixed precision inference protects sensitive layers and compresses layers that tolerate quantization noise.",
            "Portable model artifacts need versioned metadata, stable integrity hashes, and reproducible compilation plans.",
        ]
        for sample in samples:
            yield sample


class HellaswagDataset(CalibrationDataset):
    """Hellaswag calibration dataset for commonsense evaluation."""

    def __init__(self, max_tokens: int = 131072) -> None:
        super().__init__("hellaswag", max_tokens=max_tokens)

    def iter_text(self) -> Iterator[str]:
        """Yield synthetic samples representing Hellaswag-style completions."""
        samples = [
            "A person is baking a cake. They mix flour and sugar, then add eggs and milk. The next step is to put the batter in the oven.",
            "A dog sees a squirrel and runs after it. The dog barks loudly and the squirrel climbs up a tree.",
            "She opened the book and began to read. The story was engaging and she finished the chapter quickly.",
            "The engineer reviews a failing benchmark. After isolating the slow kernel, they measure memory traffic and update the fusion plan.",
            "A server receives several long prompts. The scheduler separates prefill work from decoding so each phase gets suitable hardware.",
        ]
        for sample in samples:
            yield sample


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
        return WikiText2Dataset(max_tokens=max_tokens)
    if name == "hellaswag":
        return HellaswagDataset(max_tokens=max_tokens)
    if name == "custom" and path is not None:
        return CustomJsonlDataset(path, max_tokens=max_tokens)
    if path is not None:
        return CustomJsonlDataset(path, max_tokens=max_tokens)
    msg = f"Unknown calibration dataset: {name}"
    raise CalibrationError(msg)

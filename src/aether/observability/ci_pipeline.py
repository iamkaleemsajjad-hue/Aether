"""CI/CD eval pipeline for AEG quality gating.

Runs benchmark suites against compiled AEG artifacts before production rollout.
Integrates with EvalGate from gates.py to block deployment on regression.

Research: Eval-Driven Compilation (Aether PRD §19), HellaSwag/MMLU/GSM8K benchmarks.
"""

from __future__ import annotations

import json
import math
import re
import csv
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

from aether.observability.gates import EvalGate, EvalGateDecision, EvalResult
from aether.core.exceptions import BenchmarkError


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "hellaswag": {
        "task": "multiple_choice",
        "metric": "accuracy",
        "num_questions": 10042,
        "baseline_score": 0.892,
        "higher_is_better": True,
    },
    "mmlu": {
        "task": "multiple_choice",
        "metric": "accuracy",
        "num_questions": 14042,
        "baseline_score": 0.847,
        "higher_is_better": True,
    },
    "gsm8k": {
        "task": "math",
        "metric": "exact_match",
        "num_questions": 1319,
        "baseline_score": 0.913,
        "higher_is_better": True,
    },
    "math-500": {
        "task": "math",
        "metric": "exact_match",
        "num_questions": 500,
        "baseline_score": 0.721,
        "higher_is_better": True,
    },
    "humaneval": {
        "task": "code",
        "metric": "pass@1",
        "num_questions": 164,
        "baseline_score": 0.812,
        "higher_is_better": True,
    },
    "aime": {
        "task": "math",
        "metric": "exact_match",
        "num_questions": 30,
        "baseline_score": 0.467,
        "higher_is_better": True,
    },
}


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    """Outcome of running one benchmark suite."""

    benchmark: str
    score: float
    num_correct: int
    num_total: int
    perplexity: float | None
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "score": round(self.score, 6),
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "perplexity": round(self.perplexity, 4) if self.perplexity is not None else None,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


# A configured evaluator owns dataset loading, prompt construction, model
# execution, and answer scoring.  The pipeline only accepts its measured
# result; it never derives a score from non-empty text or from a model name.
BenchmarkEvaluator = Callable[
    [str, Mapping[str, Any]], BenchmarkResult | Mapping[str, Any]
]


class JsonlBenchmarkEvaluator:
    """Evaluate measured exact-match/accuracy examples from a JSONL file.

    Each record must contain ``prompt`` (or ``question``) and one of
    ``answer``, ``target``, or ``expected``.  Optional ``answer_regex`` allows
    a benchmark-specific, declarative extractor for answers embedded in model
    explanations.  This class intentionally does not infer correctness from
    response length or from successful generation.

    The supplied ``generate_fn`` is the model execution boundary and must
    accept ``prompt=``, ``benchmark=``, and ``max_tokens=`` keyword arguments.
    Dataset paths and scoring rules are explicit so the resulting score is
    reproducible and auditable.
    """

    def __init__(
        self,
        dataset_paths: Mapping[str, str | Path],
        generate_fn: Callable[..., str],
        *,
        max_tokens: int = 256,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.dataset_paths = {str(name): Path(path) for name, path in dataset_paths.items()}
        self.generate_fn = generate_fn
        self.max_tokens = max_tokens

    def __call__(self, benchmark: str, spec: Mapping[str, Any]) -> BenchmarkResult:
        path = self.dataset_paths.get(benchmark)
        if path is None:
            raise BenchmarkError(f"No JSONL dataset configured for benchmark {benchmark!r}")
        if not path.is_file():
            raise BenchmarkError(f"Benchmark dataset not found: {path}")

        correct = 0
        total = 0
        started = time.perf_counter()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(
                        f"Invalid JSON in {path} at line {line_number}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise BenchmarkError(
                        f"Benchmark record at {path}:{line_number} must be an object"
                    )
                prompt = record.get("prompt", record.get("question"))
                expected = next(
                    (record[key] for key in ("answer", "target", "expected") if key in record),
                    None,
                )
                if not isinstance(prompt, str) or not prompt.strip():
                    raise BenchmarkError(
                        f"Benchmark record at {path}:{line_number} has no non-empty prompt"
                    )
                if expected is None:
                    raise BenchmarkError(
                        f"Benchmark record at {path}:{line_number} has no expected answer"
                    )

                response = self.generate_fn(
                    prompt=prompt,
                    benchmark=benchmark,
                    max_tokens=self.max_tokens,
                )
                if not isinstance(response, str):
                    raise BenchmarkError(
                        f"generate_fn returned {type(response).__name__} at {path}:{line_number}; expected str"
                    )
                if self._matches(response, expected, record):
                    correct += 1
                total += 1

        if total == 0:
            raise BenchmarkError(f"Benchmark dataset {path} contains no examples")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return BenchmarkResult(
            benchmark=benchmark,
            score=correct / total,
            num_correct=correct,
            num_total=total,
            perplexity=None,
            latency_ms=elapsed_ms / total,
            metadata={
                "dataset": str(path),
                "metric": str(spec.get("metric", "exact_match")),
                "evaluator": "jsonl_exact_match",
            },
        )

    @classmethod
    def _matches(
        cls, response: str, expected: Any, record: Mapping[str, Any]
    ) -> bool:
        answer_regex = record.get("answer_regex")
        if answer_regex is not None:
            if not isinstance(answer_regex, str):
                raise BenchmarkError("answer_regex must be a string")
            return re.search(answer_regex, response, flags=re.IGNORECASE) is not None and bool(
                re.search(answer_regex, str(expected), flags=re.IGNORECASE)
            )

        choices = record.get("choices")
        if isinstance(choices, list) and choices:
            if isinstance(expected, int):
                if not 0 <= expected < len(choices):
                    raise BenchmarkError("choice answer index is outside choices")
                expected_values = [choices[expected], chr(ord("A") + expected)]
            else:
                expected_values = [expected]
            normalized_response = cls._normalize(response)
            for value in expected_values:
                normalized = cls._normalize(value)
                if normalized and (
                    normalized_response == normalized
                    or normalized_response.startswith(normalized + " ")
                    or normalized_response.startswith(normalized + ")")
                    or normalized_response.startswith(normalized + ".")
                ):
                    return True
            return False

        return cls._normalize(response) == cls._normalize(expected)

    @staticmethod
    def _normalize(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value).strip().casefold()).strip(" .,:;\"'`[]()")


class DatasetBenchmarkEvaluator:
    """Evaluate common PRD benchmark file formats against a real model callback.

    This adapter deliberately accepts local dataset files only.  It supports
    the public dataset layouts used by HellaSwag, MMLU, GSM8K/Math-500/AIME,
    and HumanEval without bundling or downloading copyrighted benchmark data.
    The callback is the only model execution boundary; no score is inferred
    from a non-empty response.

    HumanEval execution is disabled by default because generated code is
    untrusted.  Callers must explicitly set ``allow_code_execution=True`` and
    accept that this is an evaluation sandbox boundary, not a security sandbox.
    """

    _SUPPORTED = frozenset({"hellaswag", "mmlu", "gsm8k", "math-500", "humaneval", "aime"})

    def __init__(
        self,
        dataset_paths: Mapping[str, str | Path],
        generate_fn: Callable[..., str],
        *,
        max_tokens: int = 256,
        max_examples: int | None = None,
        allow_code_execution: bool = False,
        code_timeout_s: float = 5.0,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive when provided")
        if code_timeout_s <= 0:
            raise ValueError("code_timeout_s must be positive")
        self.dataset_paths = {str(name).lower(): Path(path) for name, path in dataset_paths.items()}
        self.generate_fn = generate_fn
        self.max_tokens = max_tokens
        self.max_examples = max_examples
        self.allow_code_execution = allow_code_execution
        self.code_timeout_s = code_timeout_s

    def __call__(self, benchmark: str, spec: Mapping[str, Any]) -> BenchmarkResult:
        benchmark = str(benchmark).lower()
        if benchmark not in self._SUPPORTED:
            raise BenchmarkError(f"Unsupported dataset benchmark {benchmark!r}")
        path = self.dataset_paths.get(benchmark)
        if path is None:
            raise BenchmarkError(f"No dataset configured for benchmark {benchmark!r}")
        if not path.is_file():
            raise BenchmarkError(f"Benchmark dataset not found: {path}")
        if benchmark == "humaneval" and not self.allow_code_execution:
            raise BenchmarkError(
                "HumanEval requires allow_code_execution=True because it executes generated code"
            )

        records = self._load_records(benchmark, path)
        if self.max_examples is not None:
            records = records[: self.max_examples]
        if not records:
            raise BenchmarkError(f"Benchmark dataset {path} contains no examples")

        correct = 0
        started = time.perf_counter()
        for line_number, record in enumerate(records, start=1):
            if benchmark == "humaneval":
                passed = self._run_humaneval(record, path, line_number)
            else:
                prompt, expected, choices = self._prompt_and_answer(benchmark, record, path, line_number)
                response = self.generate_fn(
                    prompt=prompt,
                    benchmark=benchmark,
                    max_tokens=self.max_tokens,
                )
                if not isinstance(response, str):
                    raise BenchmarkError(
                        f"generate_fn returned {type(response).__name__} for {benchmark!r}; expected str"
                    )
                passed = self._matches(benchmark, response, expected, choices)
            correct += int(passed)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return BenchmarkResult(
            benchmark=benchmark,
            score=correct / len(records),
            num_correct=correct,
            num_total=len(records),
            perplexity=None,
            latency_ms=elapsed_ms / len(records),
            metadata={
                "dataset": str(path),
                "evaluator": "standard_local_dataset",
                "code_execution": benchmark == "humaneval",
                "max_examples": len(records),
            },
        )

    @staticmethod
    def _load_records(benchmark: str, path: Path) -> list[Mapping[str, Any]]:
        if benchmark == "mmlu" or path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        if benchmark == "humaneval" and path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise BenchmarkError("HumanEval JSON root must be an array")
            return [row for row in data if isinstance(row, Mapping)]

        records: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(f"Invalid JSON in {path} at line {line_number}") from exc
                if not isinstance(row, Mapping):
                    raise BenchmarkError(f"Dataset record at {path}:{line_number} must be an object")
                records.append(row)
        return records

    @classmethod
    def _prompt_and_answer(
        cls,
        benchmark: str,
        record: Mapping[str, Any],
        path: Path,
        line_number: int,
    ) -> tuple[str, Any, list[Any] | None]:
        if benchmark == "hellaswag":
            prompt = record.get("ctx") or record.get("context")
            choices = record.get("endings") or record.get("choices")
            expected = record.get("label")
            if not isinstance(prompt, str) or not isinstance(choices, list) or expected is None:
                raise BenchmarkError(f"Invalid HellaSwag record at {path}:{line_number}")
            try:
                expected = int(expected)
            except (TypeError, ValueError) as exc:
                raise BenchmarkError(f"Invalid HellaSwag label at {path}:{line_number}") from exc
            return prompt, expected, choices
        if benchmark == "mmlu":
            prompt = record.get("question")
            choices = [record.get(letter) for letter in ("A", "B", "C", "D")]
            expected = record.get("answer")
            if not isinstance(prompt, str) or any(not isinstance(choice, str) for choice in choices):
                raise BenchmarkError(f"Invalid MMLU record at {path}:{line_number}")
            return prompt, expected, choices

        prompt = record.get("question") or record.get("problem")
        expected = record.get("answer", record.get("target", record.get("expected")))
        if benchmark == "humaneval":
            prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or expected is None:
            raise BenchmarkError(f"Invalid {benchmark} record at {path}:{line_number}")
        return prompt, expected, None

    @classmethod
    def _matches(
        cls,
        benchmark: str,
        response: str,
        expected: Any,
        choices: list[Any] | None,
    ) -> bool:
        if choices is not None:
            normalized_response = JsonlBenchmarkEvaluator._normalize(response)
            if isinstance(expected, int):
                if not 0 <= expected < len(choices):
                    return False
                values = [choices[expected], chr(ord("A") + expected), str(expected)]
            else:
                values = [expected]
            return any(
                (normalized := JsonlBenchmarkEvaluator._normalize(value))
                and (
                    normalized_response == normalized
                    or normalized_response.startswith(normalized + " ")
                    or normalized_response.startswith(normalized + ")")
                    or normalized_response.startswith(normalized + ".")
                )
                for value in values
            )
        if benchmark in {"gsm8k", "math-500", "aime"}:
            return cls._math_normalize(cls._extract_math_answer(response)) == cls._math_normalize(
                cls._extract_math_answer(str(expected))
            )
        return JsonlBenchmarkEvaluator._normalize(response) == JsonlBenchmarkEvaluator._normalize(expected)

    @staticmethod
    def _extract_math_answer(value: str) -> str:
        matches = re.findall(r"####\s*([^\n]+)", value)
        if matches:
            return matches[-1].strip()
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", value)
        if boxed:
            return boxed[-1].strip()
        return value.strip().splitlines()[-1] if value.strip() else ""

    @staticmethod
    def _math_normalize(value: str) -> str:
        value = value.strip().replace(",", "")
        value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
        value = re.sub(r"\s+", "", value).casefold()
        try:
            return f"{float(value):.12g}"
        except ValueError:
            return value.strip(" .:;,$")

    def _run_humaneval(self, record: Mapping[str, Any], path: Path, line_number: int) -> bool:
        prompt = record.get("prompt")
        tests = record.get("test")
        entry_point = record.get("entry_point")
        if not all(isinstance(value, str) and value for value in (prompt, tests, entry_point)):
            raise BenchmarkError(f"Invalid HumanEval record at {path}:{line_number}")
        response = self.generate_fn(prompt=prompt, benchmark="humaneval", max_tokens=self.max_tokens)
        if not isinstance(response, str):
            raise BenchmarkError("HumanEval generate_fn must return source text")
        completion = re.sub(
            r"^```(?:python)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE
        )
        # HumanEval prompts contain the function signature.  The model may
        # return only the body or a complete function; support both without
        # dropping the prompt from the executable candidate.
        source = completion if completion.startswith("def ") else str(prompt) + completion
        script = source + "\n\n" + tests + f"\n\ncheck({entry_point})\n"
        with tempfile.TemporaryDirectory(prefix="aether-humaneval-") as tmp:
            script_path = Path(tmp) / "candidate.py"
            script_path.write_text(script, encoding="utf-8")
            # Windows Python needs system variables such as SystemRoot for
            # runtime initialization.  Preserve the host environment while
            # removing import/code injection hooks; this is an explicit
            # evaluation subprocess, not a security sandbox.
            secret_prefixes = (
                "AETHER_",
                "AWS_",
                "AZURE_",
                "GOOGLE_",
                "GCP_",
                "HF_",
                "OPENAI_",
            )
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith(secret_prefixes)
            }
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            env["PYTHONNOUSERSITE"] = "1"
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.code_timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False
        return completed.returncode == 0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs benchmark evaluation against a compiled AEG model.

    In production this must call a registered dataset evaluator against the AEG
    runtime. This class fails closed until such an evaluator is configured; it
    never invents benchmark scores.

    Research basis: lm-evaluation-harness (EleutherAI), AEG quality gates (PRD §19).
    """

    def __init__(
        self,
        aeg_path: str | Path | None = None,
        seed: int = 42,
        evaluator: BenchmarkEvaluator | None = None,
    ) -> None:
        self.aeg_path = Path(aeg_path) if aeg_path else None
        self._seed = seed
        self._evaluator = evaluator

    def run(
        self,
        benchmark: str,
        score_override: float | None = None,
        perplexity: float | None = None,
    ) -> BenchmarkResult:
        """
        Run one benchmark suite.

        Args:
            benchmark: Name of benchmark from BENCHMARK_REGISTRY.
            score_override: If provided, use this score (for testing / CI replay).
            perplexity: Optional measured perplexity associated with an
                evaluator result; it is not used to manufacture a score.

        Returns:
            BenchmarkResult with score, correct counts, and latency.
        """
        if benchmark not in BENCHMARK_REGISTRY:
            raise ValueError(f"Unknown benchmark: {benchmark}. Known: {list(BENCHMARK_REGISTRY)}")

        spec = BENCHMARK_REGISTRY[benchmark]
        n = spec["num_questions"]
        if score_override is not None:
            score = float(score_override)
            if not 0.0 <= score <= 1.0:
                raise ValueError("score_override must be between 0 and 1")
            num_correct = round(score * n)
            latency_ms = 0.0
            return BenchmarkResult(
                benchmark=benchmark,
                score=score,
                num_correct=num_correct,
                num_total=n,
                perplexity=perplexity,
                latency_ms=latency_ms,
                metadata={
                    "aeg_path": str(self.aeg_path),
                    "seed": str(self._seed),
                    "replay": True,
                },
            )

        if self._evaluator is None:
            raise BenchmarkError(
                f"No evaluator is configured for {benchmark!r}. Provide a real dataset/evaluator; "
                "synthetic scores are disabled."
            )

        measured = self._evaluator(benchmark, spec)
        if isinstance(measured, BenchmarkResult):
            result = measured
        elif isinstance(measured, Mapping):
            required = {"score", "num_correct", "num_total", "latency_ms"}
            missing = sorted(required.difference(measured))
            if missing:
                raise BenchmarkError(
                    f"Evaluator result for {benchmark!r} is missing measured fields: {missing}"
                )
            metadata = measured.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise BenchmarkError("Evaluator metadata must be a mapping")
            result = BenchmarkResult(
                benchmark=str(measured.get("benchmark", benchmark)),
                score=float(measured["score"]),
                num_correct=int(measured["num_correct"]),
                num_total=int(measured["num_total"]),
                perplexity=(
                    None
                    if measured.get("perplexity") is None
                    else float(measured["perplexity"])
                ),
                latency_ms=float(measured["latency_ms"]),
                metadata=dict(metadata),
            )
        else:
            raise BenchmarkError(
                f"Evaluator for {benchmark!r} returned {type(measured).__name__}; "
                "expected BenchmarkResult or a measured mapping"
            )

        self._validate_measured_result(result, benchmark)
        if perplexity is not None and result.perplexity is None:
            result = BenchmarkResult(
                benchmark=result.benchmark,
                score=result.score,
                num_correct=result.num_correct,
                num_total=result.num_total,
                perplexity=perplexity,
                latency_ms=result.latency_ms,
                metadata=result.metadata,
            )
        return result

    @staticmethod
    def _validate_measured_result(result: BenchmarkResult, requested: str) -> None:
        """Reject malformed evaluator output before it reaches the gate."""
        if result.benchmark != requested:
            raise BenchmarkError(
                f"Evaluator returned benchmark {result.benchmark!r}, expected {requested!r}"
            )
        if not 0.0 <= result.score <= 1.0:
            raise BenchmarkError("Evaluator score must be between 0 and 1")
        if result.num_total <= 0:
            raise BenchmarkError("Evaluator num_total must be positive")
        if not 0 <= result.num_correct <= result.num_total:
            raise BenchmarkError("Evaluator num_correct must be within [0, num_total]")
        measured_score = result.num_correct / result.num_total
        if abs(measured_score - result.score) > 1e-6:
            raise BenchmarkError(
                f"Evaluator score {result.score} disagrees with measured counts "
                f"{result.num_correct}/{result.num_total}"
            )
        if result.latency_ms < 0.0:
            raise BenchmarkError("Evaluator latency_ms cannot be negative")

    def run_suite(
        self,
        benchmarks: list[str],
        score_overrides: dict[str, float] | None = None,
        perplexity: float | None = None,
    ) -> list[BenchmarkResult]:
        """Run multiple benchmarks and return all results."""
        overrides = score_overrides or {}
        return [
            self.run(b, score_override=overrides.get(b), perplexity=perplexity)
            for b in benchmarks
        ]


# ---------------------------------------------------------------------------
# CI eval pipeline
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Structured quality report produced by CIEvalPipeline."""

    aeg_path: str
    benchmark_results: list[BenchmarkResult]
    gate_decision: EvalGateDecision
    compiler_version: str = "aether/3.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "aeg_path": self.aeg_path,
            "compiler_version": self.compiler_version,
            "gate": self.gate_decision.to_dict(),
            "benchmarks": [r.to_dict() for r in self.benchmark_results],
            "summary": {
                "total_benchmarks": len(self.benchmark_results),
                "passed": self.gate_decision.passed,
                "max_regression_pct": round(self.gate_decision.max_relative_regression * 100, 3),
                "failing": list(self.gate_decision.failing_benchmarks),
            },
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out


class CIEvalPipeline:
    """
    CI/CD eval pipeline: runs benchmarks → EvalGate → blocks/allows deployment.

    Usage:
        pipeline = CIEvalPipeline(aeg_path="./model.aeg", max_regression=0.02)
        report = pipeline.run(["hellaswag", "mmlu", "gsm8k"])
        if not report.gate_decision.passed:
            raise SystemExit("Eval gate FAILED — blocking rollout")
    """

    def __init__(
        self,
        aeg_path: str | Path,
        max_regression: float = 0.02,
        required_benchmarks: tuple[str, ...] = ("hellaswag", "mmlu", "gsm8k"),
        evaluator: BenchmarkEvaluator | None = None,
    ) -> None:
        self.aeg_path = Path(aeg_path)
        self.runner = BenchmarkRunner(aeg_path=aeg_path, evaluator=evaluator)
        self.gate = EvalGate(
            max_relative_regression=max_regression,
            required_benchmarks=required_benchmarks,
        )

    def run(
        self,
        benchmarks: list[str] | None = None,
        baselines: dict[str, float] | None = None,
        score_overrides: dict[str, float] | None = None,
        perplexity: float | None = None,
    ) -> QualityReport:
        """
        Run full CI eval: benchmark → compare to baseline → EvalGate decision.

        Args:
            benchmarks: Which benchmarks to run. Defaults to required_benchmarks.
            baselines: Override baseline scores per benchmark. Defaults to BENCHMARK_REGISTRY values.
            score_overrides: Force specific scores (for CI replay / testing).
            perplexity: Perplexity from calibration for proxy-based scoring.

        Returns:
            QualityReport with gate decision.
        """
        suites = benchmarks or list(self.gate.required_benchmarks)
        bench_results = self.runner.run_suite(suites, score_overrides=score_overrides, perplexity=perplexity)

        eval_results = []
        for br in bench_results:
            spec = BENCHMARK_REGISTRY.get(br.benchmark, {})
            baseline = (baselines or {}).get(br.benchmark, spec.get("baseline_score", br.score))
            higher = spec.get("higher_is_better", True)
            eval_results.append(EvalResult(
                benchmark=br.benchmark,
                baseline_score=baseline,
                candidate_score=br.score,
                higher_is_better=higher,
            ))

        decision = self.gate.evaluate(eval_results)
        return QualityReport(
            aeg_path=str(self.aeg_path),
            benchmark_results=bench_results,
            gate_decision=decision,
        )

    def run_and_save(
        self,
        output_path: str | Path,
        benchmarks: list[str] | None = None,
        score_overrides: dict[str, float] | None = None,
    ) -> QualityReport:
        """Run pipeline and save JSON report to disk."""
        report = self.run(benchmarks=benchmarks, score_overrides=score_overrides)
        report.save(output_path)
        return report

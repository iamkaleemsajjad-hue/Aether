"""Repeatable compiler pass audit. Run --help; measured execution is opt-in.

All work is serial. Compile, load, numerics, public Runtime, HF reference and
timing use fresh subprocesses. This file does not alter compiler algorithms.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psutil

JsonObject = dict[str, Any]

if __package__:
    from .pass_audit_fixtures import (
        CONTINUATION,
        FIXTURES,
        PROMPTS,
        create_fixture,
        parameter_count,
        prompt_text,
        write_json,
    )
else:
    from pass_audit_fixtures import (
        CONTINUATION,
        FIXTURES,
        PROMPTS,
        create_fixture,
        parameter_count,
        prompt_text,
        write_json,
    )


@dataclass(frozen=True)
class Pass:
    """Identify one canonical compiler flag and the evidence it requires."""

    number: int
    name: str
    flag: str
    trigger: str
    consumer: str


PASSES = (
    Pass(
        1,
        "operator_fusion",
        "enable_fusion",
        "Dense connected RMSNorm/QKV/RoPE graph",
        "IR rewrites; fused execution is not established by an opcode",
    ),
    Pass(
        2,
        "sensitivity_analysis",
        "enable_sensitivity",
        "Real attached layer tensors",
        "Compile-time reconstruction scores, not model perplexity",
    ),
    Pass(
        3,
        "precision_assignment",
        "enable_precision_assignment",
        "Explicit uniform Q4_K_M",
        "Weight codec decoded by AEG loader; CPU executes FP32",
    ),
    Pass(
        4,
        "kv_cache_structuring",
        "enable_kv_cache_structuring",
        "Transformer layers",
        "Paged pool helper exists; ordinary loader generation does not exercise it",
    ),
    Pass(
        5,
        "moe_routing",
        "enable_moe_routing",
        "moe: four experts, top-2 router and real expert weights",
        "Routed FFN executes; hot/cold tier hints need separate evidence",
    ),
    Pass(
        6,
        "parallelism_discovery",
        "enable_parallelism_discovery",
        "Sharding plans for 1/2/4/8 devices",
        "Single CPU audit cannot prove multi-device benefit",
    ),
    Pass(
        7,
        "reasoning_graph",
        "enable_reasoning_graph",
        "Reasoning budget 32",
        "Reasoning plan; ordinary greedy decode does not prove reasoning execution",
    ),
    Pass(
        8,
        "sparse_attention",
        "enable_sparse_attention",
        "Threshold 16; prompt B has 64 tokens",
        "CPU sparse plan; default 4096-token windows may cause no sparsity here",
    ),
    Pass(
        9,
        "pruning_sparsity",
        "enable_pruning",
        "Attached weights, magnitude metric, target sparsity 0.5",
        "Mask emission alone does not prove sparse kernel dispatch",
    ),
    Pass(
        10,
        "mtp_head_compilation",
        "enable_mtp_head",
        "mtp: two real FP32 vocabulary projection heads",
        "PEAGLE required; ordinary loader path does not activate heads",
    ),
    Pass(
        11,
        "grammar_constraint_compilation",
        "enable_grammar_constraint",
        'Tokenizer-backed builtin root ::= "hello"',
        "Public Runtime may consume FSM; unconstrained loader metrics are separate",
    ),
    Pass(
        12,
        "model_merging",
        "enable_model_merging",
        "Real complete matching source with changed Q projection, coefficient 0.5",
        "Merged weights consumed by loader; task quality is unevaluated",
    ),
    Pass(
        13,
        "ttt_fast_weight_injection",
        "enable_ttt",
        "Two or more real layers, rank 2",
        "Initialized slots; this audit does not request online training",
    ),
    Pass(
        14,
        "semantic_kv_compression",
        "enable_semantic_kv",
        "Chunk strategy, retention 0.5, prompt B length 64",
        "CPU semantic_kv_plan and generation KV cache",
    ),
    Pass(
        15,
        "cross_layer_kv_sharing",
        "enable_cross_layer_kv",
        "Two or more layers; threshold 0.1 to produce actual groups",
        "CPU cross_layer_kv_plan and generation KV cache",
    ),
    Pass(
        16,
        "green_energy_compilation",
        "enable_green_energy",
        "CPU target, us-west region",
        "Profile hints; energy/DVFS not measured by this harness",
    ),
    Pass(
        17,
        "tee_kernel_wrapping",
        "enable_tee",
        "Requires real backend-emitted TEE kernels (absent)",
        "Expected skip/refusal; hashes do not establish enclave execution",
    ),
    Pass(
        18,
        "mdlm_drafter_compilation",
        "enable_mdlm_drafter",
        "Requires trained drafter bundle (not synthesized)",
        "Expected skip/refusal; no trained drafter fixture",
    ),
    Pass(
        19,
        "sub2bit_quantization",
        "enable_sub2bit",
        "Real source tensors; BitNet codec",
        "Packed TERNARY decoded to FP32; reconstruction is not a quality gate",
    ),
    Pass(
        20,
        "video_token_compression",
        "enable_video_compression",
        "Requires vision/video encoder (absent in text fixtures)",
        "Expected skip/refusal; no visual runtime coverage",
    ),
    Pass(
        21,
        "advanced_peft_compilation",
        "enable_advanced_peft",
        "adapter: real rank-2 SafeTensors q_proj LoRA A/B",
        "Compiled adapter explicitly selected in loader worker",
    ),
    Pass(
        22,
        "rlvr_verifier_head_injection",
        "enable_rlvr_verifier",
        "Sympy verifier config; group size 4",
        "Training artifacts; inference is not a training/verifier test",
    ),
)
COMBINATIONS = ((1, 2, 3), (3, 9), (4, 8, 14), (14, 15), (10, 11), (5, 21))
MIB = 1024 * 1024
MEMORY_SCHEMA = "aether-pass-audit-memory/2"
ROOT = Path(__file__).resolve().parents[1]


def compiler_options(
    numbers: list[int], fixture: Mapping[str, Any], cache: Path, *, precision_mode: str = "uniform"
) -> JsonObject:
    """Do not inherit even one enabled pass from CompilerConfig defaults."""
    selected = set(numbers)
    if selected - set(range(1, 23)):
        raise ValueError("Pass numbers must be in 1..22")
    options: JsonObject = {p.flag: p.number in selected for p in PASSES}
    options.update(
        targets=["cpu_generic"],
        optimization_level=0,
        skip_download=True,
        reproducible_builds=True,
        overwrite=False,
        cache_dir=str(cache),
        calibration_tokens=16,
        max_calibration_samples=2,
        upload_kernels=False,
        reasoning_budget_tokens=32,
        sparse_attention_context_threshold=16,
        pruning_metric="magnitude",
        pruning_target_sparsity=0.5,
        semantic_kv_strategy="chunk",
        semantic_kv_compression_ratio=0.5,
        cross_layer_kv_share_threshold=0.1,
        ttt_rank=2,
        grammar_backend="builtin",
        grammar_schema='root ::= "hello"',
        model_merging_sources=[fixture["merge_source"]],
        model_merging_coefficients=[0.5],
        model_merging_method="task_arithmetic",
        peft_adapter_paths=fixture["adapters"],
        sub2bit_method="bitnet",
        rlvr_verifier_type="sympy",
        rlvr_group_size=4,
    )
    if 3 in selected and precision_mode == "uniform":
        options.update(precision="Q4_K_M", precision_assignment_mode="uniform")
    return options


def inventory(root: Path) -> dict[str, JsonObject]:
    """Hash *every* file, including manifest, metadata, binaries and tokenizer."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[path.relative_to(root).as_posix()] = {
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
            }
    return result


def json_changes(before: Any, after: Any, pointer: str = "") -> list[JsonObject]:
    """Compare JSON leaves without removing volatile or structural fields."""
    if type(before) is type(after) and isinstance(before, dict):
        out = []
        for key in sorted(before.keys() | after.keys()):
            escaped = key.replace("~", "~0").replace("/", "~1")
            path = f"{pointer}/{escaped}"
            if key not in before or key not in after:
                out.append(
                    {
                        "path": path,
                        "before": before.get(key),
                        "after": after.get(key),
                        "presence_changed": True,
                    }
                )
            else:
                out.extend(json_changes(before[key], after[key], path))
        return out
    if type(before) is type(after) and isinstance(before, list) and len(before) == len(after):
        return [
            change
            for i, (a, b) in enumerate(zip(before, after, strict=True))
            for change in json_changes(a, b, f"{pointer}/{i}")
        ]
    return [] if before == after else [{"path": pointer, "before": before, "after": after}]


def diff_artifacts(
    a: Path,
    b: Path,
    left: dict[str, JsonObject] | None = None,
    right: dict[str, JsonObject] | None = None,
) -> JsonObject:
    """Retain every raw file digest and separately describe JSON differences."""
    left = inventory(a) if left is None else left
    right = inventory(b) if right is None else right
    rows = []
    volatile = []
    for name in sorted(left.keys() | right.keys()):
        x, y = left.get(name), right.get(name)
        state = (
            "added"
            if x is None
            else "removed"
            if y is None
            else "identical"
            if x == y
            else "changed"
        )
        row: JsonObject = {"file": name, "status": state, "before": x, "after": y}
        if state == "changed" and name.endswith(".json") and a == b:
            row["json_diff_unavailable"] = (
                "Only pre-runtime hashes retained; original JSON not rewritten or normalized"
            )
        elif state == "changed" and name.endswith(".json"):
            try:
                changes = json_changes(
                    json.loads((a / name).read_text(encoding="utf-8")),
                    json.loads((b / name).read_text(encoding="utf-8")),
                )
                row["json_changes"] = changes
                for change in changes:
                    leaf = change["path"].rsplit("/", 1)[-1]
                    if (
                        "timestamp" in leaf
                        or "duration" in leaf
                        or leaf in {"compiled_at", "compile_date", "created_at"}
                    ):
                        volatile.append({"file": name, **change})
            except (ValueError, OSError, UnicodeError) as exc:
                row["json_diff_error"] = str(exc)
        rows.append(row)
    return {
        "byte_identical": all(row["status"] == "identical" for row in rows),
        "files": rows,
        "volatile_metadata_changes": volatile,
        "normalization_applied": False,
    }


def summarize(values: list[float]) -> JsonObject:
    """Describe observed spread, retaining the original finite samples."""
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Expected finite nonempty measurements")
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "pstdev": statistics.pstdev(values),
    }


def continuation_ppl(log_ref: Any, log_actual: Any) -> JsonObject:
    """Score only the four fixed next-token labels, without clipping losses.

    PPL = exp(-mean(log p(label | preceding tokens))). The relative change is
    expm1(candidate_mean_nll - reference_mean_nll). Shift logits against their
    next-token labels as in the primary GPTQ evaluation implementation:
    https://github.com/IST-DASLab/gptq/blob/main/llama.py (llama_eval).
    Definition: https://huggingface.co/docs/transformers/en/perplexity.
    These references define the metric, not a borrowed performance claim.
    """
    import numpy as np

    labels = np.asarray(CONTINUATION, dtype=np.int64)
    count = len(labels)
    if log_ref.shape[0] < count or log_ref.shape[1] <= int(labels.max()):
        return {"status": "unavailable", "reason": "Not enough rows or vocabulary for labels"}
    positions = np.arange(count)
    ref_nll = float(-log_ref[positions, labels].mean())
    actual_nll = float(-log_actual[positions, labels].mean())
    difference = actual_nll - ref_nll
    relative: float | None
    try:
        relative = math.expm1(difference)
    except OverflowError:
        relative = None
    return {
        "status": "ok",
        "labels": CONTINUATION,
        "label_count": count,
        "reference_mean_nll": ref_nll,
        "candidate_mean_nll": actual_nll,
        "mean_nll_difference": difference,
        "relative_ppl_change": relative,
        "relative_ppl_overflow": relative is None,
        "ppl_regression_within_one_percent": difference <= math.log1p(0.01),
        "scope": "Fixed synthetic continuation; not corpus or natural-language quality",
    }


def compare_numerics(reference: JsonObject, actual: JsonObject) -> JsonObject:
    """Compare real logits and IDs, plus explicitly scoped numerical thresholds."""
    import numpy as np

    if reference.get("status") != "ok" or actual.get("status") != "ok":
        return {"status": "unavailable", "reason": "Both numerical workers must succeed"}
    rows = {}
    with (
        np.load(reference["logits"], allow_pickle=False) as ref,
        np.load(actual["logits"], allow_pickle=False) as got,
    ):
        for prompt in PROMPTS:
            a, b = (
                np.asarray(ref[prompt], dtype=np.float64),
                np.asarray(got[prompt], dtype=np.float64),
            )
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                return {
                    "status": "failed",
                    "reason": f"{prompt}: shape mismatch or nonfinite logits",
                }

            def logsoftmax(x: Any) -> Any:
                shifted = x - x.max(axis=-1, keepdims=True)
                return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

            la, lb = logsoftmax(a), logsoftmax(b)
            kl = np.maximum(0, (np.exp(la) * (la - lb)).sum(axis=-1))
            na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
            cosine = np.divide(
                (a * b).sum(axis=-1), na * nb, out=np.zeros_like(na), where=na * nb > 0
            )
            cosine[(na == 0) & (nb == 0)] = 1
            expected, observed = reference["tokens"][prompt], actual["tokens"][prompt]
            agreement = sum(x == y for x, y in zip(expected, observed, strict=False)) / max(
                1, len(expected), len(observed)
            )
            ppl = continuation_ppl(la, lb)
            checks = {
                "kl_mean_le_0.01": bool(kl.mean() <= 0.01),
                "cosine_min_ge_0.999": bool(cosine.min() >= 0.999),
                "greedy_agreement_ge_0.95": agreement >= 0.95,
                "ppl_regression_le_0.01": ppl.get("ppl_regression_within_one_percent"),
            }
            rows[prompt] = {
                "teacher_forced_positions": int(a.shape[0]),
                "kl_reference_to_candidate_mean": float(kl.mean()),
                "kl_reference_to_candidate_max": float(kl.max()),
                "logit_cosine_mean": float(cosine.mean()),
                "logit_cosine_min": float(cosine.min()),
                "max_abs_logit_error": float(np.abs(a - b).max()),
                "teacher_forced_argmax_agreement": float(np.mean(a.argmax(-1) == b.argmax(-1))),
                "greedy_token_agreement": agreement,
                "greedy_exact_match": expected == observed,
                "reference_token_ids": expected,
                "candidate_token_ids": observed,
                "continuation_ppl": ppl,
                "numerical_gate": {
                    "status": "evaluated" if ppl["status"] == "ok" else "unavailable",
                    "passed": all(checks.values()) if ppl["status"] == "ok" else None,
                    "checks": checks,
                    "compiler_quality_certification": False,
                },
            }
    return {"status": "ok", "prompts": rows}


def reserve_bytes(fixture_name: str, action: str, passes: Sequence[int] = ()) -> int:
    """Estimate admission reserve rather than advertise estimated RSS as measured."""
    raw = parameter_count(fixture_name) * 4
    # Conservative admission estimates, not claimed measurements. Pass 19 and
    # merging materialize Python float lists; allow much more than tensor size.
    multiplier = (
        (40 if 12 in passes else 24 if 19 in passes else 8)
        if action in {"compile", "idempotency"}
        else 5
        if action == "fixture"
        else 4
    )
    # The fixed small HF fixtures peaked below 300 MiB aggregate tree RSS in
    # recorded CPU runs. Keep a 512 MiB estimate plus the independent live
    # headroom guard; the unmeasured 27M oracle retains its 1 GiB estimate.
    overhead = (1024 if fixture_name == "mid" else 512) * MIB if action == "hf" else 384 * MIB
    return int(overhead + raw * multiplier)


def admission(available: int, reserve: int, headroom: int) -> JsonObject:
    """Decide whether a worker may start with the requested memory headroom."""
    return {
        "status": "ok" if available >= reserve + headroom else "blocked",
        "available_bytes": available,
        "estimated_worker_bytes": reserve,
        "required_headroom_bytes": headroom,
        "reason": "RAM admission estimate; no worker launched when blocked",
    }


def memory_point(phase: str) -> JsonObject:
    """Identify the actual Python worker and record its own RSS at a boundary."""
    import psutil

    process = psutil.Process()
    return {
        "schema": MEMORY_SCHEMA,
        "phase": phase,
        "pid": int(process.pid),
        "create_time": float(process.create_time()),
        "rss_bytes": int(process.memory_info().rss),
    }


class WorkerMemorySampler:
    """Separate verified worker RSS from a launcher and aggregate descendants."""

    def __init__(self, root: psutil.Process) -> None:
        self.root = root
        self.root_identity = (int(root.pid), float(root.create_time()))
        self.worker_identity: tuple[int, float] | None = None
        self.phase = "imports"
        self.phase_peaks: dict[str, int] = {}
        self.tree_peaks: dict[str, int] = {}
        self.samples = 0
        self.tree_samples = 0
        self.rejected_points = 0
        self.final_point_status = "not_received"

    @staticmethod
    def _identity(point: JsonObject) -> tuple[int, float] | None:
        if point.get("schema") != MEMORY_SCHEMA:
            return None
        pid, created = point.get("pid"), point.get("create_time")
        if type(pid) is not int or pid <= 0:
            return None
        if not isinstance(created, (int, float)) or isinstance(created, bool):
            return None
        if not math.isfinite(created):
            return None
        return pid, float(created)

    def _owned_processes(self) -> dict[tuple[int, float], psutil.Process]:
        import psutil

        owned: dict[tuple[int, float], psutil.Process] = {}
        try:
            if not self.root.is_running():
                return owned
            if (self.root.pid, self.root.create_time()) != self.root_identity:
                return owned
            processes = [self.root, *self.root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return owned
        for process in processes:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                if process.is_running():
                    owned[(int(process.pid), float(process.create_time()))] = process
        return owned

    def _record_point(self, point: JsonObject) -> bool:
        identity = self._identity(point)
        phase, rss = point.get("phase"), point.get("rss_bytes")
        if (
            identity is None
            or identity != self.worker_identity
            or not isinstance(phase, str)
            or not phase
            or type(rss) is not int
            or rss < 0
        ):
            self.rejected_points += 1
            return False
        self.phase = phase
        self.phase_peaks[phase] = max(self.phase_peaks.get(phase, 0), rss)
        return True

    def sample(self, marker: JsonObject | None) -> None:
        """Sample a marked, live owned worker; never substitute launcher RSS."""
        import psutil

        owned = self._owned_processes()
        if marker is not None:
            identity = self._identity(marker)
            if identity in owned and (
                self.worker_identity is None or self.worker_identity == identity
            ):
                # The first marker comes from worker_main before compiler or
                # runtime imports. Once selected, compiler children cannot
                # replace this identity, even if a later marker names them.
                if self.worker_identity is None:
                    self.worker_identity = identity
                self._record_point(marker)
            else:
                self.rejected_points += 1
        worker = owned.get(self.worker_identity) if self.worker_identity is not None else None
        if worker is not None:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                rss = int(worker.memory_info().rss)
                self.phase_peaks[self.phase] = max(self.phase_peaks.get(self.phase, 0), rss)
                self.samples += 1
        tree_rss = 0
        tree_sampled = False
        for process in owned.values():
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                tree_rss += int(process.memory_info().rss)
                tree_sampled = True
        if tree_sampled:
            self.tree_peaks[self.phase] = max(self.tree_peaks.get(self.phase, 0), tree_rss)
            self.tree_samples += 1

    def finish(self, point: JsonObject | None) -> None:
        """Accept a final boundary only for a previously verified worker identity."""
        if point is None:
            return
        accepted = self.worker_identity is not None and self._record_point(point)
        self.final_point_status = "verified_identity" if accepted else "unverified_identity"

    def summary(self) -> JsonObject:
        """Expose distinct main-worker and aggregate-tree measurements and identity."""
        worker = (
            {"pid": self.worker_identity[0], "create_time": self.worker_identity[1]}
            if self.worker_identity is not None
            else None
        )
        return {
            "marker_schema": MEMORY_SCHEMA,
            "samples": self.samples,
            "tree_samples": self.tree_samples,
            "phase_peak_rss_bytes": self.phase_peaks,
            "phase_tree_peak_rss_bytes": self.tree_peaks,
            "worker_identity": worker,
            "worker_identity_status": "verified_owned" if worker else "unavailable",
            "launcher_identity": {
                "pid": self.root_identity[0],
                "create_time": self.root_identity[1],
            },
            "rejected_memory_points": self.rejected_points,
            "final_point_status": self.final_point_status,
        }


def run_worker(job: JsonObject, directory: Path, args: argparse.Namespace) -> JsonObject:
    """Run one admitted subprocess and sample only its own phase memory."""
    import psutil

    directory.mkdir(parents=True, exist_ok=False)
    check = admission(
        psutil.virtual_memory().available,
        reserve_bytes(job["fixture_name"], job["action"], job.get("passes", ())),
        int(args.min_free_mib * MIB),
    )
    write_json(directory / "admission.json", check)
    if check["status"] == "blocked":
        return check
    job.update(work=str(directory), native=args.native)
    write_json(directory / "job.json", job)
    env = dict(os.environ)
    env.update(
        {
            "SOURCE_DATE_EPOCH": str(args.epoch),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "NUMEXPR_NUM_THREADS": str(args.threads),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "AETHER_HF_OFFLINE": "1",
            "AETHER_CACHE_DIR": str(directory / "cache"),
            "AETHER_KERNEL_CACHE": str(directory / "kernels"),
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
            "TEMP": str(directory / "tmp"),
            "TMP": str(directory / "tmp"),
            "TMPDIR": str(directory / "tmp"),
        }
    )
    (directory / "tmp").mkdir()
    termination: str | None = None
    host_available_min = int(check["available_bytes"])
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                str(directory / "job.json"),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        observed = psutil.Process(process.pid)
        sampler = WorkerMemorySampler(observed)
        started = time.monotonic()
        while process.poll() is None:
            marker = None
            with contextlib.suppress(OSError, ValueError):
                marker = json.loads((directory / "phase.json").read_text(encoding="utf-8"))
            sampler.sample(marker)
            host_available = int(psutil.virtual_memory().available)
            host_available_min = min(host_available_min, host_available)
            if host_available < args.min_free_mib * MIB:
                termination = "memory_headroom_exhausted"
            elif time.monotonic() - started > args.timeout:
                termination = "timeout"
            if termination:
                for child in reversed(observed.children(recursive=True)):
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.kill()
                process.kill()
                break
            time.sleep(0.005)
        process.wait()
    result_file = directory / "result.json"
    result: JsonObject
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {"status": "failed", "reason": termination or "Worker exited without valid result"}
    if process.returncode or termination:
        result.update(
            status="blocked" if termination == "memory_headroom_exhausted" else "failed",
            exit_code=process.returncode,
        )
        if termination:
            result["reason"] = termination
    sampler.finish(result.get("final_memory_point"))
    result["memory"] = {
        "method": "psutil RSS of marker-identified owned Python worker, fresh worker per action",
        "sample_interval_ms": 5,
        **sampler.summary(),
        "admission": check,
        "host_available_min_bytes": host_available_min,
        "configured_headroom_mib": args.min_free_mib,
        "measurement_environment": (
            "memory_constrained_diagnostic; timing is not clean-host performance evidence"
            if args.min_free_mib < 512 or host_available_min < 1024 * MIB
            else "standard_headroom_guard; exclusive host resources are not proven"
        ),
        "limitation": (
            "Sampling can miss brief peaks; tree RSS can double-count shared pages; "
            "no GPU memory measurement"
        ),
    }
    result["worker_log"] = str(directory / "worker.log")
    write_json(directory / "observed.json", result)
    return result


def mark_phase(work: Path, phase: str) -> int:
    """Publish a phase plus actual worker PID, creation time and boundary RSS."""
    point = memory_point(phase)
    write_json(work / "phase.json", point)
    return int(point["rss_bytes"])


def canonical_graph_value(value: Any) -> Any:
    """Preserve all graph state, hashing full tensor bytes instead of array reprs."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return {
            "tensor_shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, enum.Enum):
        return canonical_graph_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_graph_value(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): canonical_graph_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_graph_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported graph snapshot value: {type(value).__name__}")


def audit_idempotency(job: JsonObject, work: Path) -> JsonObject:
    """Apply one actual pipeline pass twice to the same ingested weighted graph."""
    from aether.compiler.config import CompilerConfig
    from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
    from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
    from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline

    config = CompilerConfig(**compiler_options(job["passes"], job["fixture"], work / "cache"))
    architecture = ArchitectureDetector().detect(job["fixture"]["model"])
    graph = IngestionPipeline(config).ingest(job["fixture"]["model"], architecture)
    staged = work / "pass_output"
    staged.mkdir()
    graph.output_dir = str(staged)
    pipeline = OptimizerPipeline(config)
    number = job["passes"][0]
    states: list[JsonObject] = []
    reports: list[JsonObject] = []
    first_files: dict[str, JsonObject] = {}
    for iteration in (1, 2):
        mark_phase(work, f"same_graph_pass_{iteration}")
        graph, pass_reports = pipeline.run(graph, architecture)
        state = canonical_graph_value(graph.to_dict())
        snapshot = work / f"graph_after_{iteration}.json"
        write_json(snapshot, state)
        states.append(state)
        reports.append(pass_reports[number - 1].to_dict())
        if iteration == 1:
            first_files = inventory(staged)
    return {
        "status": "failed" if any(row["status"] == "failed" for row in reports) else "ok",
        "scope": "Same graph instance, same selected pass twice; separate from rebuild determinism",
        "pass_reports": reports,
        "graph_changes": json_changes(states[0], states[1]),
        "graph_idempotent": states[0] == states[1],
        "emitted_files": diff_artifacts(staged, staged, first_files, inventory(staged)),
        "runtime_benefit_claim": False,
    }


def _worker(job: JsonObject) -> JsonObject:
    work = Path(job["work"])
    action = job["action"]
    if action == "idempotency":
        return audit_idempotency(job, work)
    if action == "fixture":
        mark_phase(work, "fixture")
        first = create_fixture(work / "first", job["fixture_name"])
        create_fixture(work / "second", job["fixture_name"])
        return {
            "status": "ok",
            "fixture": first,
            "determinism": diff_artifacts(work / "first", work / "second"),
        }
    if action == "compile":
        from aether import Compiler, CompilerConfig

        if "compiler_config" in job:
            options = dict(job["compiler_config"])
            options["cache_dir"] = str(work / "compiler_cache")
        else:
            options = compiler_options(
                job["passes"],
                job["fixture"],
                work / "compiler_cache",
                precision_mode=job.get("precision_mode", "uniform"),
            )
        config = CompilerConfig(**options)
        enabled = [p.number for p in PASSES if getattr(config, p.flag)]
        if enabled != sorted(job["passes"]):
            raise ValueError(
                "Declared pass selection differs from the actual compiler configuration"
            )
        mark_phase(work, "compile")
        started = time.perf_counter()
        package = Compiler(config).compile(job["fixture"]["model"], output_path=work / "model.aeg")
        package.verify_integrity()
        compile_and_verify_seconds = time.perf_counter() - started
        transformations = package.metadata.get("provenance", {}).get("transformations", [])
        statuses = {
            entry["pass"]: entry.get("parameters", {}).get("status", "unknown")
            for entry in transformations
        }
        if set(statuses) != {p.name for p in PASSES}:
            raise ValueError(
                "Compiler pass registry differs from audit catalog; update catalog before auditing"
            )
        if not job["passes"] and any(state != "skipped" for state in statuses.values()):
            raise ValueError("All-off baseline unexpectedly executed a pass")
        return {
            "status": "ok",
            "artifact": str(package.root),
            "compile_and_verify_seconds": compile_and_verify_seconds,
            "passes": statuses,
            "config": config.to_dict(),
            "evaluation_status": package.metadata.get("evaluation_status", "unavailable"),
            "evaluation_evaluator_supplied": False,
            "quality_gate": "not_evaluated",
            "metadata": {
                key: package.metadata.get(key)
                for key in (
                    "sub2bit_runtime",
                    "weight_accounting",
                    "sensitivity_method",
                    "fusion_accounting",
                )
            },
        }

    import numpy as np

    if action == "hf":
        if job["fixture_name"] not in {"tiny", "mid", "adapter"}:
            return {"status": "skipped", "reason": "HF oracle supports dense Llama fixtures only"}
        missing = [
            name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None
        ]
        if missing:
            return {
                "status": "skipped",
                "reason": "Missing optional HF dependencies",
                "missing": missing,
            }
        import torch
        from safetensors.torch import load_file
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.set_num_threads(job["threads"])
        torch.manual_seed(0)
        mark_phase(work, "hf_load")
        config = LlamaConfig(**job["fixture"]["config"])
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).to(dtype=torch.float32).eval()
        # Strictly load original FP32 checkpoint. No BF16 rounding/matching of
        # the oracle to the compiler's lossy packaging is allowed here.
        model.load_state_dict(
            load_file(str(Path(job["fixture"]["model"]) / "model.safetensors")), strict=True
        )
        logits, tokens = {}, {}
        mark_phase(work, "hf_numerics")
        with torch.inference_mode():
            for label, ids in PROMPTS.items():
                teacher = torch.tensor([ids + CONTINUATION], dtype=torch.long)
                logits[label] = model(teacher).logits[0, len(ids) - 1 :].float().numpy()
                generated = list(ids)
                for _ in range(job["tokens"]):
                    token = int(
                        model(torch.tensor([generated], dtype=torch.long)).logits[0, -1].argmax()
                    )
                    generated.append(token)
                tokens[label] = generated[len(ids) :]
        np.savez(work / "logits.npz", **logits)
        return {
            "status": "ok",
            "logits": str(work / "logits.npz"),
            "tokens": tokens,
            "reference": "Hugging Face LlamaForCausalLM, original source FP32; no weight rounding",
            "dtype": "float32",
        }

    if action == "public":
        from aether import Runtime, RuntimeConfig

        config = RuntimeConfig(
            backend_name="aether_cpu",
            execution_devices=["cpu"],
            hf_offline=True,
            speculative_decoding=False,
            enable_diffusion_spec=False,
            enable_semantic_cache=False,
            enable_prefix_caching=False,
            enable_continuous_batching=False,
            enable_telemetry=False,
            dynamic_precision=False,
            disaggregate_prefill_decode=False,
            ttt_enabled=False,
            model_cache_dir=str(work / "runtime_cache"),
        )
        mark_phase(work, "public_runtime")
        outputs = {}
        with Runtime(config) as runtime:
            for label, ids in PROMPTS.items():
                response = runtime.generate(
                    job["artifact"], prompt_text(ids), max_tokens=job["tokens"], temperature=0.0
                )
                outputs[label] = {"text": response.text, "usage": response.usage}
        return {
            "status": "ok",
            "api": "Runtime.generate",
            "outputs": outputs,
            "scope": "Public text API smoke; feature requests such as TTT/MTP are not activated",
        }

    from aether.runtime.aeg_loader import load_engine_from_path

    before_load = mark_phase(work, "load")
    engine = load_engine_from_path(job["artifact"], prepare_native_kernels=job["native"])
    resident = mark_phase(work, "loaded")
    if action == "load":
        return {
            "status": "ok",
            "rss_before_load_bytes": before_load,
            "rss_loaded_bytes": resident,
            "rss_load_delta_bytes": resident - before_load,
            "engine": type(engine).__name__,
            "consumer_plans": {
                name: bool(getattr(engine, name, None))
                for name in (
                    "sparse_attention_plan",
                    "semantic_kv_plan",
                    "cross_layer_kv_plan",
                )
            },
            "execution_storage": (
                "CPU decoded projections, typically FP32; mmap is not zero-copy evidence"
            ),
        }
    adapter_id = None
    if 21 in job["passes"] and job["fixture"]["adapters"]:
        from aether.adapters.lora import load_compiled_lora_adapters

        adapters = load_compiled_lora_adapters(Path(job["artifact"]))
        if not adapters:
            raise ValueError("Adapter fixture requested but no compiled adapter was loadable")
        adapter_id = sorted(adapters)[0]
        engine = engine.with_lora_adapter(adapters, adapter_id)
    kwargs = {"adapter_id": adapter_id} if adapter_id else {}
    if action == "numerics":
        mark_phase(work, "numerics")
        logits, tokens = {}, {}
        for label, ids in PROMPTS.items():
            teacher = np.asarray(ids + CONTINUATION, dtype=np.int64)
            rows, _ = engine.forward(teacher, **kwargs)
            logits[label] = np.asarray(rows)[len(ids) - 1 :]
            if not np.isfinite(logits[label]).all():
                raise ValueError(f"Nonfinite {label} logits")
            tokens[label] = [
                int(token)
                for token in engine.generate_iter(
                    np.asarray(ids, dtype=np.int64),
                    max_tokens=job["tokens"],
                    temperature=0.0,
                    seed=0,
                    **kwargs,
                )
            ]
        np.savez(work / "logits.npz", **logits)
        return {
            "status": "ok",
            "logits": str(work / "logits.npz"),
            "tokens": tokens,
            "adapter_id": adapter_id,
            "api": "load_engine_from_path -> forward/generate_iter",
        }
    if action != "timing":
        raise ValueError(f"Unknown worker action {action}")
    samples: dict[str, dict[str, list[float]]] = {
        label: {
            "ttft_ms": [],
            "elapsed_ms": [],
            "tokens_per_second": [],
            "decode_tokens_per_second": [],
        }
        for label in PROMPTS
    }
    for index in range(job["warmup"] + job["repeats"]):
        mark_phase(work, "warmup" if index < job["warmup"] else "generation")
        # Alternate A/B order to avoid making only B the warmed request.
        for label in list(PROMPTS) if index % 2 == 0 else list(reversed(PROMPTS)):
            ids = PROMPTS[label]
            stream = iter(
                engine.generate_iter(
                    np.asarray(ids, dtype=np.int64),
                    max_tokens=job["tokens"],
                    temperature=0.0,
                    seed=0,
                    **kwargs,
                )
            )
            started = time.perf_counter()
            first = next(stream)
            first_time = time.perf_counter()
            generated = [int(first)] + [int(token) for token in stream]
            ended = time.perf_counter()
            if generated != job["expected_tokens"][label]:
                raise ValueError(f"{label}: timed output differs from numerical worker")
            if index >= job["warmup"]:
                samples[label]["ttft_ms"].append((first_time - started) * 1000)
                samples[label]["elapsed_ms"].append((ended - started) * 1000)
                samples[label]["tokens_per_second"].append(len(generated) / (ended - started))
                if len(generated) > 1:
                    samples[label]["decode_tokens_per_second"].append(
                        (len(generated) - 1) / (ended - first_time)
                    )
    return {
        "status": "ok",
        "prompts": {
            label: {
                metric: summarize(values) if values else None for metric, values in metrics.items()
            }
            for label, metrics in samples.items()
        },
        "warmup_requests_per_prompt": job["warmup"],
        "repeats_per_prompt": job["repeats"],
        "rss_loaded_bytes": resident,
        "ttft_definition": "generate_iter start through first yielded ID, including prefill",
        "elapsed_definition": "iterator fully exhausted, including final cache update",
        "adapter_id": adapter_id,
    }


def worker_main(path: Path) -> None:
    """Persist explicit outcomes; unexpected import failures are genuine failures."""
    job = json.loads(path.read_text(encoding="utf-8"))
    work = Path(job["work"])
    mark_phase(work, "imports")
    try:
        result = _worker(job)
    except Exception as exc:
        traceback.print_exc()
        status = "refused" if getattr(exc, "stage", None) else "failed"
        result = {
            "status": status,
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "stage": getattr(exc, "stage", None),
        }
    try:
        phase = json.loads((work / "phase.json").read_text(encoding="utf-8"))["phase"]
    except (OSError, ValueError):
        phase = "imports"
    result["final_memory_point"] = memory_point(phase)
    write_json(work / "result.json", result)


def measured_deltas(baseline: JsonObject, actual: JsonObject) -> JsonObject:
    """Observations, never attribute a timing delta to an unconsumed IR pass."""
    out: JsonObject = {
        "artifact_bytes_delta": actual["builds"][0]["artifact_bytes"]
        - baseline["builds"][0]["artifact_bytes"],
        "speed_attribution": (
            "Unattributed A/B observations; consumer evidence is required for pass speed claims"
        ),
    }
    for metric in ("rss_loaded_bytes", "rss_load_delta_bytes"):
        a, b = baseline["runtime"].get("load", {}), actual["runtime"].get("load", {})
        if a.get("status") == b.get("status") == "ok":
            out[metric + "_delta"] = b[metric] - a[metric]
    a, b = baseline["runtime"].get("timing", {}), actual["runtime"].get("timing", {})
    if a.get("status") == b.get("status") == "ok":
        out["timing"] = {}
        for label in PROMPTS:
            left, right = a["prompts"][label], b["prompts"][label]
            out["timing"][label] = {
                "elapsed_speed_ratio_baseline_over_candidate": left["elapsed_ms"]["median"]
                / right["elapsed_ms"]["median"],
                "ttft_ms_median_delta": right["ttft_ms"]["median"] - left["ttft_ms"]["median"],
                "elapsed_ranges_overlap": max(left["elapsed_ms"]["min"], right["elapsed_ms"]["min"])
                <= min(left["elapsed_ms"]["max"], right["elapsed_ms"]["max"]),
            }
        left = a["memory"]["phase_peak_rss_bytes"].get("generation")
        right = b["memory"]["phase_peak_rss_bytes"].get("generation")
        if left is not None and right is not None:
            out["generation_peak_rss_bytes_delta"] = right - left
    return out


def case_evidence(
    numbers: list[int], compile_result: JsonObject, runtime: JsonObject, comparison: JsonObject
) -> list[JsonObject]:
    """Distinguish compile status, consumer evidence and unestablished benefits."""
    rows = []
    for number in numbers:
        p = PASSES[number - 1]
        state = compile_result.get("passes", {}).get(p.name, compile_result["status"])
        load = runtime.get("load", {})
        plan = {8: "sparse_attention_plan", 14: "semantic_kv_plan", 15: "cross_layer_kv_plan"}.get(
            number
        )
        consumer = "not_exercised"
        if runtime.get("numerics", {}).get("status") == "ok":
            if number in {3, 9, 12, 19} and state == "applied":
                consumer = "decoded_weight_transform_executed"
            elif plan and load.get("consumer_plans", {}).get(plan):
                consumer = "plan_attached_and_generation_executed"
            elif number == 21 and runtime["numerics"].get("adapter_id"):
                consumer = "compiled_adapter_selected_and_executed"
        rows.append(
            {
                "number": number,
                "pass": p.name,
                "compile_status": state,
                "runtime_consumer": consumer,
                "consumer_scope": p.consumer,
                "benefit_status": "not_established",
                "working_claim": False,
                "reason": (
                    "Compile status and successful generation alone do not establish pass benefit"
                ),
                "numerical_comparison_status": comparison.get("status", "unavailable"),
                "numerical_gate_passed": (
                    all(
                        metrics.get("numerical_gate", {}).get("passed") is True
                        for metrics in comparison["prompts"].values()
                    )
                    if comparison.get("prompts")
                    else None
                ),
            }
        )
    return rows


def parse_passes(values: list[str]) -> list[int]:
    """Accept canonical pass numbers while refusing ambiguous selections."""
    if values == ["all"]:
        return list(range(1, 23))
    try:
        numbers = sorted({int(number) for value in values for number in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Passes must be 'all' or numbers 1..22") from exc
    if not numbers or min(numbers) < 1 or max(numbers) > 22:
        raise argparse.ArgumentTypeError("Passes must be in 1..22")
    return numbers


def main(argv: list[str] | None = None) -> int:
    """Inspect a plan or execute an explicitly selected serial audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", nargs="+", choices=(*FIXTURES, "all"), default=["tiny"])
    parser.add_argument("--passes", nargs="+", default=["all"], metavar="N")
    parser.add_argument(
        "--output", type=Path, help="New directory; existing output is never overwritten"
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print catalog/configuration only; no fixtures, workers or tests",
    )
    parser.add_argument(
        "--timing", action="store_true", help="Opt in after other suites/benchmarks finish"
    )
    parser.add_argument(
        "--hf-reference",
        action="store_true",
        help="Optional original FP32 HF Llama oracle, no downloads",
    )
    parser.add_argument(
        "--no-public-runtime", action="store_true", help="Explicitly skip public text API smoke"
    )
    parser.add_argument("--no-combinations", action="store_true")
    parser.add_argument(
        "--no-idempotency", action="store_true", help="Skip same-graph diagnostic workers"
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Use normal native-kernel loader; default uses public NumPy reference loader option",
    )
    parser.add_argument("--builds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-free-mib", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--epoch", type=int, default=1700000000)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker_main(args.worker)
        return 0
    try:
        numbers = parse_passes(args.passes)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if (
        args.builds < 2
        or args.repeats < 2
        or args.warmup < 0
        or args.tokens < 2
        or args.threads < 1
        or args.min_free_mib < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error(
            "Need builds/repeats/tokens >=2, threads >=1, warmup/headroom >=0 "
            "and finite positive timeout"
        )
    fixtures = list(FIXTURES) if "all" in args.fixtures else list(dict.fromkeys(args.fixtures))
    cases: list[list[int]] = [[]] + [[number] for number in numbers]
    if not args.no_combinations:
        cases += [list(combo) for combo in COMBINATIONS if set(combo) <= set(numbers)]
        if {2, 3} <= set(numbers):
            cases.append([2, 3])
    report: JsonObject = {
        "schema": "aether-pass-audit/1",
        "status": "planned",
        "pass_matrix": [{**asdict(p), "status": "not_run", "observations": []} for p in PASSES],
        "selected_fixtures": fixtures,
        "selected_cases": cases,
        "source_date_epoch": args.epoch,
        "reference_labels": {
            "primary": (
                "unoptimized AEG with all 22 passes explicitly off; packaging may still store BF16"
            ),
            "hf": "original checkpoint FP32 HF Llama oracle"
            if args.hf_reference
            else "not_requested",
        },
        "quality_scope": (
            "Synthetic numerical regression, not task/perplexity quality certification; "
            "no evaluator supplied to compile"
        ),
        "sparse_scope": (
            "Forced threshold 16 differs from default 32768; "
            "64-token prompt is not long-context validation"
        ),
        "precision_scope": (
            "Pass 3 and (1,2,3) use uniform Q4_K_M. The (2,3) combination uses actual "
            "sensitivity/default precision selection; corpus quality remains not audited."
        ),
        "determinism_scope": (
            "Repeated compile builds and separate same-graph idempotency workers for single passes"
        ),
        "prompts_token_ids": PROMPTS,
        "teacher_forced_continuation_ids": CONTINUATION,
        "timing": "requested_serial_workers" if args.timing else "skipped_not_requested",
        "native_loader_requested": args.native,
        "fixture_parameters": {name: parameter_count(name) for name in fixtures},
        "results": [],
        "fixtures": {},
    }
    if args.plan:
        print(json.dumps(report, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for execution; use --plan for a dry inspection")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in (
        "aether-runtime",
        "numpy",
        "safetensors",
        "psutil",
        "tokenizers",
        "torch",
        "transformers",
        "onnxruntime",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    report["environment"] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
        "threads": args.threads,
        "configured_headroom_mib": args.min_free_mib,
        "worker_io_encoding": "utf-8",
        "seed": 1729,
        "cwd": str(ROOT),
    }
    source_files = [
        ROOT / "src/aether/compiler/compiler.py",
        ROOT / "src/aether/compiler/config.py",
        ROOT / "src/aether/runtime/aeg_loader.py",
        ROOT / "src/aether/runtime/cpu_engine.py",
        ROOT / "src/aether/runtime/runtime.py",
        *sorted((ROOT / "src/aether/compiler/stage2_optimizer").glob("*.py")),
    ]
    report["environment"]["source_sha256"] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    report["status"] = "running"

    def save() -> None:
        write_json(output / "report.json", report)

    save()
    for fixture_name in fixtures:
        print(f"fixture {fixture_name}", flush=True)
        prepared = run_worker(
            {"action": "fixture", "fixture_name": fixture_name},
            output / fixture_name / "fixture",
            args,
        )
        report["fixtures"][fixture_name] = prepared
        save()
        if prepared["status"] != "ok":
            for number in numbers:
                catalog = report["pass_matrix"][number - 1]
                catalog["status"] = prepared["status"]
                catalog["observations"].append(
                    {
                        "fixture": fixture_name,
                        "case": "fixture_setup",
                        "status": prepared["status"],
                        "compile_status": "not_run",
                    }
                )
            save()
            continue
        fixture = prepared["fixture"]
        hf = (
            run_worker(
                {
                    "action": "hf",
                    "fixture_name": fixture_name,
                    "fixture": fixture,
                    "tokens": args.tokens,
                    "threads": args.threads,
                },
                output / fixture_name / "hf",
                args,
            )
            if args.hf_reference
            else {"status": "skipped", "reason": "not requested"}
        )
        prepared["hf_reference"] = hf
        baseline: JsonObject | None = None
        for enabled in cases:
            name = "baseline" if not enabled else "pass_" + "_".join(map(str, enabled))
            precision_mode = "sensitivity" if enabled == [2, 3] else "uniform"
            if precision_mode == "sensitivity":
                name += "_sensitivity_default"
            print(f"  {fixture_name}/{name}", flush=True)
            directory = output / fixture_name / name
            row: JsonObject = {
                "fixture": fixture_name,
                "case": name,
                "enabled_passes": enabled,
                "builds": [],
                "runtime": {},
            }
            report["results"].append(row)
            base_job = {
                "fixture_name": fixture_name,
                "fixture": fixture,
                "passes": enabled,
                "tokens": args.tokens,
                "precision_mode": precision_mode,
            }
            for build in range(args.builds):
                compiled = run_worker(
                    {**base_job, "action": "compile"}, directory / f"build_{build}", args
                )
                artifact = Path(
                    compiled.get("artifact", directory / f"build_{build}" / "model.aeg")
                )
                compiled["artifact_files"] = inventory(artifact)
                compiled["artifact_bytes"] = sum(
                    item["bytes"] for item in compiled["artifact_files"].values()
                )
                row["builds"].append(compiled)
                save()
                if compiled["status"] != "ok":
                    break
            first = row["builds"][0]
            if first["status"] != "ok":
                row["status"] = first["status"]
                row["evidence"] = case_evidence(enabled, first, {}, {})
                save()
                continue
            a = Path(first["artifact"])
            row["repeat_diffs"] = [
                diff_artifacts(a, Path(b["artifact"]), first["artifact_files"], b["artifact_files"])
                for b in row["builds"][1:]
                if b["status"] == "ok"
            ]
            row["repeatability_status"] = (
                "incomplete"
                if len(row["repeat_diffs"]) != args.builds - 1
                else "identical"
                if all(d["byte_identical"] for d in row["repeat_diffs"])
                else "different"
            )
            if baseline:
                row["versus_baseline_artifacts"] = diff_artifacts(
                    Path(baseline["builds"][0]["artifact"]),
                    a,
                    baseline["builds"][0]["artifact_files"],
                    first["artifact_files"],
                )
                row["artifact_bytes_delta"] = (
                    first["artifact_bytes"] - baseline["builds"][0]["artifact_bytes"]
                )
            row["same_graph_idempotency"] = (
                run_worker({**base_job, "action": "idempotency"}, directory / "idempotency", args)
                if len(enabled) == 1 and not args.no_idempotency
                else {
                    "status": "skipped",
                    "reason": "Singleton diagnostic disabled or not applicable",
                }
            )
            for action in ("load", "numerics", "public"):
                if action == "public" and args.no_public_runtime:
                    row["runtime"][action] = {"status": "skipped", "reason": "disabled by CLI"}
                    continue
                row["runtime"][action] = run_worker(
                    {**base_job, "action": action, "artifact": str(a)}, directory / action, args
                )
                save()
            numerical = row["runtime"]["numerics"]
            if not enabled:
                baseline = row
            if baseline:
                row["versus_unoptimized_aeg"] = compare_numerics(
                    baseline["runtime"]["numerics"], numerical
                )
            else:
                row["versus_unoptimized_aeg"] = {
                    "status": "unavailable",
                    "reason": "All-off baseline compilation did not succeed",
                }
            if hf.get("status") == "ok":
                row["versus_original_fp32_hf"] = compare_numerics(hf, numerical)
            else:
                row["versus_original_fp32_hf"] = {
                    "status": hf["status"],
                    "reason": hf.get("reason", "HF reference unavailable"),
                }
            if args.timing and numerical["status"] == "ok":
                row["runtime"]["timing"] = run_worker(
                    {
                        **base_job,
                        "action": "timing",
                        "artifact": str(a),
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "expected_tokens": numerical["tokens"],
                    },
                    directory / "timing",
                    args,
                )
            else:
                row["runtime"]["timing"] = {
                    "status": "skipped",
                    "reason": "Timing not requested or numerical worker unavailable",
                }
            row["after_runtime_artifact_diff"] = diff_artifacts(
                a, a, first["artifact_files"], inventory(a)
            )
            row["evidence"] = case_evidence(
                enabled, first, row["runtime"], row.get("versus_unoptimized_aeg", {})
            )
            if baseline:
                row["measured_deltas_versus_unoptimized_aeg"] = measured_deltas(baseline, row)
                for evidence in row["evidence"]:
                    evidence["unattributed_observed_deltas"] = row[
                        "measured_deltas_versus_unoptimized_aeg"
                    ]
            requested_states = [first["passes"][PASSES[n - 1].name] for n in enabled]
            action_states = [value["status"] for value in row["runtime"].values()]
            action_states.append(row["same_graph_idempotency"]["status"])
            row["status"] = next(
                (
                    state
                    for state in ("failed", "refused", "blocked")
                    if state in requested_states + action_states
                ),
                "skipped"
                if requested_states and all(s == "skipped" for s in requested_states)
                else "observed"
                if numerical["status"] == "ok"
                else numerical["status"],
            )
            save()
    for row in report["results"]:
        for number in row["enabled_passes"]:
            catalog = report["pass_matrix"][number - 1]
            catalog["status"] = "observed"
            catalog["observations"].append(
                {
                    "fixture": row["fixture"],
                    "case": row["case"],
                    "status": row["status"],
                    "compile_status": row["builds"][0]
                    .get("passes", {})
                    .get(PASSES[number - 1].name, row["builds"][0]["status"]),
                }
            )
    report["status"] = "complete"
    report["limitations"] = [
        "Untrained fixtures do not certify real-model quality or workload performance.",
        "Compiler provenance exposes statuses, not every PassReport reason; "
        "worker logs preserve emitted explanations.",
        "Graph-only passes, specialized training, TEE, video and unrequested speculative/TTT "
        "paths have no benefit claim.",
        "RSS samples are observed process memory, not per-weight/KV/activation "
        "attribution or GPU VRAM.",
        "All-off AEG can be BF16; only a successful optional HF worker is an "
        "original-precision reference.",
    ]
    save()
    failures = [
        r
        for r in report["results"]
        if r["status"] in {"failed", "refused", "blocked"}
        or r.get("repeatability_status") in {"different", "incomplete"}
    ]
    fixture_failures = any(
        item["status"] != "ok"
        or not item.get("determinism", {}).get("byte_identical", False)
        or item.get("hf_reference", {}).get("status") in {"failed", "refused", "blocked"}
        for item in report["fixtures"].values()
    )
    print(str(output / "report.json"))
    return 1 if failures or fixture_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

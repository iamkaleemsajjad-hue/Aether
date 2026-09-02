"""Tests for the multi-engine benchmark suite.

These target the properties that make the suite trustworthy rather than its
plumbing: that a missing measurement can never become a zero, that a comparison
Aether loses is produced by the same code path as one it wins, and that the
representation label is attached wherever the two sides are not holding the same
weights.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from benchmark.suite import analysis, charts, hardware, plan, report
from benchmark.suite import engines as registry
from benchmark.suite import status as status_mod

# ── Status vocabulary ───────────────────────────────────────────────────────

def test_missing_measurement_is_never_measured() -> None:
    """Every non-MEASURED status must fail the predicate aggregation gates on."""
    for value in status_mod.ALL_STATUSES:
        assert status_mod.is_measured(value) is (value == status_mod.MEASURED)
    assert status_mod.is_measured({"status": status_mod.OOM}) is False
    assert status_mod.is_measured(None) is False
    assert status_mod.is_measured({}) is False


def test_exceptions_classify_into_distinguishable_statuses() -> None:
    """OOM, unsupported and a plain defect must not collapse into one label."""
    from benchmark.backends import UnsupportedConfiguration

    assert status_mod.from_exception(UnsupportedConfiguration("no"))[0] == \
        status_mod.NOT_SUPPORTED
    assert status_mod.from_exception(RuntimeError("CUDA out of memory"))[0] == \
        status_mod.OOM
    assert status_mod.from_exception(MemoryError())[0] == status_mod.OOM
    assert status_mod.from_exception(ImportError("no module"))[0] == \
        status_mod.NOT_INSTALLED
    assert status_mod.from_exception(ValueError("boom"))[0] == status_mod.FAILED


def test_runner_statuses_map_onto_the_suite_vocabulary() -> None:
    assert status_mod.from_runner({"status": "ok"}) == status_mod.MEASURED
    assert status_mod.from_runner({"status": "oom"}) == status_mod.OOM
    assert status_mod.from_runner({"status": "unsupported"}) == status_mod.NOT_SUPPORTED
    # An unrecognized status must not be optimistically read as a measurement.
    assert status_mod.from_runner({"status": "something new"}) == status_mod.FAILED


# ── Precision resolution ────────────────────────────────────────────────────

def _hardware(**overrides: Any) -> hardware.Hardware:
    base = hardware.Hardware(
        platform="x86_64", os_name="Linux", logical_cores=8, physical_cores=4,
        ram_bytes=32 * 1024 ** 3,
    )
    return replace(base, **overrides)


def test_precision_is_the_widest_format_the_whole_field_can_execute() -> None:
    """bf16 only where the hardware has bf16 tensor cores; fp16 otherwise.

    A precision only some engines support is not a fair benchmark precision, it is a
    way of excluding engines - an engine can refuse bf16 below compute capability
    8.0 outright. So a pre-Ampere host resolves to fp16 and the reason says why.
    """
    native = _hardware(nvidia=True, bf16_native=True, compute_capabilities=["8.0"])
    assert hardware.resolve_precision("auto", native)[0] == "bf16"

    turing = _hardware(nvidia=True, bf16_native=False, torch_reports_bf16=True,
                       compute_capabilities=["7.5", "7.5"])
    resolved, reason = hardware.resolve_precision("auto", turing)
    assert resolved == "fp16"
    assert "8.0 or newer" in reason
    assert "--precision bf16" in reason, "the weight-exact option must stay documented"
    assert "software emulation" in reason, (
        "when torch claims bf16 support on a pre-Ampere card, say what that claim is"
    )


def test_bf16_nativeness_comes_from_the_capability_not_from_torch() -> None:
    """torch answers True on cards that only emulate bf16; the capability decides."""
    turing = _hardware(nvidia=True, compute_capabilities=["7.5"], torch_reports_bf16=True)
    assert hardware.meets_capability(turing, (8, 0)) is False
    ampere = _hardware(nvidia=True, compute_capabilities=["8.6"])
    assert hardware.meets_capability(ampere, (8, 0)) is True
    assert hardware.min_compute_capability(turing) == (7, 5)
    assert hardware.min_compute_capability(_hardware()) is None


def test_cpu_resolves_to_fp32_and_an_explicit_request_is_honoured() -> None:
    resolved, reason = hardware.resolve_precision("auto", _hardware())
    assert resolved == "fp32"
    assert "no accelerator" in reason
    assert hardware.resolve_precision("fp16", _hardware())[0] == "fp16"


def test_weight_fit_guard_uses_the_smallest_visible_device() -> None:
    """A guard against a doomed load, sized from the smallest card, not the largest."""
    box = _hardware(nvidia=True, gpu_vram_bytes=[16 * 1024 ** 3, 40 * 1024 ** 3])
    assert hardware.can_hold_weights(box, 3_800_000_000, "bf16")
    assert not hardware.can_hold_weights(box, 3_800_000_000, "fp32")
    # An unknown parameter count must not be turned into a refusal to try.
    assert hardware.can_hold_weights(box, None, "fp32")


def test_thread_pinning_reports_what_it_pinned(monkeypatch: Any) -> None:
    for name in hardware.THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    record = hardware.pin_threads(3)
    assert record["controlled"] is True
    assert record["env"]["OMP_NUM_THREADS"] == "3"
    inherited = hardware.pin_threads(None)
    assert inherited["controlled"] is False


# ── Engine registry ─────────────────────────────────────────────────────────

def test_every_engine_declares_the_full_adapter_contract() -> None:
    """A registered engine must be describable and probeable without running."""
    assert len(registry.KEYS) == len(set(registry.KEYS)), "engine keys must be unique"
    for key in registry.KEYS:
        module = registry.module_for(key)
        spec = module.SPEC
        assert spec.key == key
        assert spec.taxonomy, f"{key} must declare what kind of system it is"
        assert spec.summary
        assert callable(module.probe)
        assert callable(module.build)
        assert spec.artifact_persistence in {
            "none", "process-local", "on-disk-cache", "portable-artifact",
        }
        # A build phase and a persistence class have to agree: something that
        # compiles nothing cannot leave an artifact behind.
        if not spec.has_build_phase:
            assert spec.artifact_persistence == "none"


def test_reference_and_subject_are_distinct_registered_engines() -> None:
    assert registry.REFERENCE in registry.KEYS
    assert registry.SUBJECT in registry.KEYS
    assert registry.REFERENCE != registry.SUBJECT


def test_the_field_is_the_four_engines_and_none_of_them_is_privileged() -> None:
    """The set is pinned, and pinned in the order a table reads it.

    Reducing the field and removing an engine are the same edit, so what the field is
    has to be a stated fact rather than whichever modules happen to import. Every
    property asserted here is one an engine could fail while still being perfectly
    installable, and any of them would put a rank in the standings for an experiment
    the rest of the field did not run.
    """
    from benchmark.suite.engines import base

    assert registry.KEYS == ("transformers", "pytorch_native", "openvino", "aether")
    # Report order, not rank order: the reference first, the subject last.
    assert registry.KEYS[0] == registry.REFERENCE
    assert registry.KEYS[-1] == registry.SUBJECT
    for key in registry.KEYS:
        spec = registry.spec_for(key)
        assert not spec.requires_cuda, f"{key} would be excluded by host, not measured"
        assert not spec.alters_representation, (
            f"{key} would be measured on weights the rest of the field does not load"
        )
        assert base.SERVING_ENGINE not in spec.taxonomy, (
            f"{key} would be measuring a serving loop against single-process latency"
        )


def test_taxonomy_does_not_call_eager_frameworks_compilers() -> None:
    """The classification has to be accurate, not flattering or convenient."""
    from benchmark.suite.engines import base

    compilers = {base.AOT_COMPILER, base.JIT_COMPILER, base.GRAPH_COMPILER}
    for key in ("transformers", "pytorch_native"):
        assert not (set(registry.spec_for(key).taxonomy) & compilers)
    for key in ("openvino", "aether"):
        assert set(registry.spec_for(key).taxonomy) & compilers
    # The subject is described in the same vocabulary as everything else and holds no
    # label of its own: a term coined for Aether would read as a capability the
    # competing compiler lacks, when all it would actually mark is who wrote it.
    others = set().union(*(
        set(registry.spec_for(key).taxonomy)
        for key in registry.KEYS if key != registry.SUBJECT
    ))
    assert set(registry.spec_for(registry.SUBJECT).taxonomy) <= others


def test_no_engine_needs_hardware_or_an_artifact_the_others_are_not_given() -> None:
    """Four engines, one checkpoint, one host: every probe answers from the model id.

    An engine that can only run from a file it was handed, or only on a card this host
    may not have, cannot be measured on the same cells as the rest of the field. So the
    only reason any of the four may decline is that it is not installed - never
    NOT_APPLICABLE, which is the status the analysis excludes from every percentage.
    """
    config = plan.SuiteConfig()
    hosts = {
        "cpu": (_hardware(), "fp32"),
        "gpu": (_hardware(nvidia=True, bf16_native=True, compute_capabilities=["8.0"],
                          gpu_vram_bytes=[16 * 1024 ** 3]), "bf16"),
    }
    for name, (host, precision) in hosts.items():
        for key in registry.KEYS:
            result = registry.probe(key, host, "Qwen/Qwen3-0.6B", precision, config)
            assert result.status in {status_mod.MEASURED, status_mod.NOT_INSTALLED}, (
                f"{key} declined the {name} host as {result.status}: {result.reason}"
            )
            if result.status != status_mod.MEASURED:
                assert result.reason, "a declined engine must say why"


def test_a_hardware_requirement_is_declined_before_installation_is_considered() -> None:
    """"This machine could never run it" outranks "it is not installed here".

    Nothing in the four-engine field carries a hardware requirement - that is part of
    what makes it the field - so the check is exercised against a synthetic spec rather
    than deleted along with the engines that used to need it. The distinction still has
    to hold: NOT_APPLICABLE is excluded from the percentages, while NOT_INSTALLED on a
    host that could have run the engine is a different fact about a different run.
    """
    from benchmark.suite.engines import base

    cuda_only = replace(registry.spec_for("aether"), key="synthetic",
                        requires_cuda=True, requires=())
    declined = base.generic_probe(cuda_only, _hardware())
    assert declined.status == status_mod.NOT_APPLICABLE
    assert "NVIDIA" in declined.reason
    reachable = base.generic_probe(cuda_only, _hardware(nvidia=True))
    assert reachable.status == status_mod.MEASURED

    ampere_only = replace(cuda_only, min_capability=(8, 0))
    turing = base.generic_probe(
        ampere_only, _hardware(nvidia=True, compute_capabilities=["7.5"])
    )
    assert turing.status == status_mod.NOT_APPLICABLE
    assert "7.5" in turing.reason and "8.0" in turing.reason


#: Every hardware class the suite can name, plus the honest answer when a name is not
#: one of them. Asserted against as a set so a new class of accelerator is added in one
#: place rather than in each test that happens to touch a device string.
_DEVICE_CLASS_VOCABULARY = {
    "nvidia-gpu", "amd-gpu", "apple-gpu", "intel-gpu", "npu", "cpu", None,
}


def test_device_class_names_hardware_in_one_vocabulary_and_never_guesses() -> None:
    """Two engines on two kinds of chip have to be detectable as exactly that.

    The classes carry more than they look: OpenVINO's ``GPU`` is its Intel plugin, so a
    row reporting GPU on a host whose accelerator is an NVIDIA card did not run on the
    accelerator the rest of the field used. An unrecognised name is reported as unknown
    and never defaulted either way - defaulting to the CPU would invent a hardware
    difference, and defaulting to the accelerator would hide one.
    """
    from benchmark.suite.engines import base

    assert base.device_class("cuda:0") == "nvidia-gpu"
    assert base.device_class("CPU") == "cpu"
    assert base.device_class("GPU") == "intel-gpu", "OpenVINO's GPU plugin is Intel's"
    assert base.device_class("xpu") == "intel-gpu"
    assert base.device_class("mps") == "apple-gpu"
    assert base.device_class("hip:1") == "amd-gpu"
    assert base.device_class("NPU") == "npu"
    for unknown in (None, "", "   ", "auto", "something-new"):
        assert base.device_class(unknown) is None
    assert set(_DEVICE_CLASS_VOCABULARY) >= {
        label for _, label in base._DEVICE_CLASSES
    }


def test_every_engine_reports_the_device_and_thread_budget_it_actually_used() -> None:
    """Not what the plan asked for: what the engine says it did.

    A controlled variable an engine can decline is not controlled unless the engine
    reports what it did instead. An engine that fell back to the host CPU and one that
    took every core on the machine are both invisible in a throughput number - the
    first just looks slow and the second just looks fast - so each of the four states
    the device it ran on, the class of that device and its thread count, in the same
    vocabulary, and the report prints them beside what was requested.
    """
    from benchmark.suite.engines import openvino_engine

    gpu = _hardware(nvidia=True, compute_capabilities=["7.5"],
                    gpu_vram_bytes=[16 * 1024 ** 3])
    config = plan.SuiteConfig(threads=2)
    for key in registry.KEYS:
        described = registry.build(key, gpu, "Qwen/Qwen3-0.6B", "fp16",
                                   config).describe()
        for name in ("execution_device", "execution_device_class", "threads"):
            assert name in described, f"{key} reports no {name}"
        assert described["execution_device_class"] in _DEVICE_CLASS_VOCABULARY

    # And the class is derived from the device rather than from the host: OpenVINO
    # compiled for its own GPU plugin is on Intel hardware, which is not the NVIDIA
    # card the torch engines were placed on, and the label has to say so.
    assert openvino_engine.Engine(device="GPU").describe()[
        "execution_device_class"] == "intel-gpu"
    assert openvino_engine.Engine(device="CPU").describe()[
        "execution_device_class"] == "cpu"


def test_the_thread_budget_reaches_the_pool_each_engine_schedules_on() -> None:
    """One budget, two mechanisms, because the two runtimes read different ones.

    torch takes its thread count from the environment, which is where the worker pins
    it. OpenVINO schedules on TBB and ignores ``OMP_NUM_THREADS`` entirely: pinning
    through the environment alone holds every torch engine to the budget and leaves
    OpenVINO the whole machine, which is a difference in resources wearing the clothes
    of a difference in engines. It has to be handed the count through its own property.
    """
    from benchmark.suite.engines import openvino_engine

    assert "OMP_NUM_THREADS" in hardware.THREAD_ENV_VARS
    pinned = openvino_engine.Engine(threads=2)
    assert pinned._runtime_config()["INFERENCE_NUM_THREADS"] == 2
    assert pinned.describe()["threads"] == 2
    # No budget pinned means no property set: an absent budget must not become a 1.
    assert "INFERENCE_NUM_THREADS" not in openvino_engine.Engine()._runtime_config()


def test_an_unsupported_keyword_costs_the_keyword_not_the_engine() -> None:
    """A renamed loader argument must not remove a whole engine from the field.

    Third-party loaders rename and retire keywords between releases. Passing one
    unconditionally turns a spelling into an absent row on every version that renamed
    it; passing none means the run's configuration is never applied at all. So exactly
    the keyword the callee named in its own TypeError is dropped and the call retried,
    while a TypeError raised from inside the call still propagates.
    """
    from benchmark.suite.engines import base

    def loader(model: str, *, dtype: str) -> str:
        return f"{model}:{dtype}"

    assert base.call_with_supported_kwargs(
        loader, "qwen", dtype="fp16", torch_dtype="fp16", export=True
    ) == "qwen:fp16"

    def strict(model: str) -> str:
        raise TypeError("the callee's own complaint, from inside the call")

    with pytest.raises(TypeError, match="own complaint"):
        base.call_with_supported_kwargs(strict, "qwen")


# ── What OpenVINO reports about its own export ───────────────────────────────

class _FakeNode:
    """One node of an OpenVINO graph, with only what the reader touches."""

    def __init__(self, type_name: str, element: str, shape: tuple[int, ...]) -> None:
        self._type_name, self._element, self._shape = type_name, element, shape

    def get_type_name(self) -> str:
        return self._type_name

    def get_element_type(self) -> str:
        return self._element

    def get_output_shape(self, index: int) -> tuple[int, ...]:
        return self._shape


class _FakeIR:
    def __init__(self, nodes: list[_FakeNode]) -> None:
        self.model = type("Graph", (), {"get_ordered_ops": lambda _self: list(nodes)})()


def test_the_ir_element_type_comes_from_a_weight_not_from_the_input_ids() -> None:
    """The stored width is read off a weight, which is the only node that carries it.

    Reading the first graph *parameter* instead reports every export as integer-typed,
    because the first parameter is ``input_ids`` - an int64 tensor. That is what made
    the report print an int64 storage format for a float16 export, and with the width
    unreadable the comparison against a 16-bit engine came out labelled like for like.
    A scale factor is float and also not the answer, so the largest float constant is
    what is read: in a decoder graph that is a weight matrix, never a scalar.
    """
    from benchmark.suite.engines import openvino_engine

    nodes = [
        _FakeNode("Parameter", "<Type: 'int64_t'>", (1, 128)),
        _FakeNode("Constant", "<Type: 'int64_t'>", (1, 128)),
        _FakeNode("Constant", "<Type: 'float16'>", (1,)),
        _FakeNode("Constant", "<Type: 'float16'>", (151936, 1024)),
    ]
    assert openvino_engine._weight_element_type(_FakeIR(nodes)) == "<Type: 'float16'>"
    # No float weight to read means unknown, which the report prints as unknown.
    assert openvino_engine._weight_element_type(_FakeIR(nodes[:2])) is None
    assert openvino_engine._weight_element_type(object()) is None


def test_element_widths_and_precision_names_are_read_longest_first() -> None:
    """``float32`` is not an ``f32`` prefix match and ``bfloat16`` is not ``f16``.

    OpenVINO renders a type as ``<Type: 'float32'>``, so both readers match substrings.
    Shortest-first ordering reads that as an f32 *and* would read ``bfloat16`` as f16 -
    one of those is the right width by luck and the other is the wrong format name.
    """
    from benchmark.suite.engines import openvino_engine

    assert openvino_engine._element_bits("<Type: 'float32'>") == 32
    assert openvino_engine._element_bits("<Type: 'bfloat16'>") == 16
    assert openvino_engine._element_bits("f16") == 16
    assert openvino_engine._element_bits("i4") == 4
    assert openvino_engine._element_bits(None) is None
    assert openvino_engine._element_bits("something new") is None

    assert openvino_engine._precision_name("<Type: 'float32'>") == "fp32"
    assert openvino_engine._precision_name("<Type: 'bfloat16'>") == "bf16"
    assert openvino_engine._precision_name("<Type: 'float16'>") == "fp16"
    assert openvino_engine._precision_name("f16") == "fp16"
    assert openvino_engine._precision_name(None) is None
    assert openvino_engine._precision_name("<Type: 'int64_t'>") is None


def test_every_precision_the_suite_can_resolve_has_a_defined_storage_element() -> None:
    """The export cannot quietly store a width the run did not ask for.

    The requested precision used to be recorded and then dropped, so the IR was written
    at the exporter's default while the rest of the field executed 16-bit weights - and
    since the width was never read back, the report called the representations equal.
    Every precision the suite can resolve to now maps to an explicit element type, and
    a precision with no defined mapping is declined by the probe instead of converted
    at whatever the exporter would have picked.
    """
    from benchmark.suite.engines import openvino_engine

    resolvable = {choice for choice in plan.PRECISION_CHOICES if choice != "auto"}
    assert set(openvino_engine._STORAGE_ELEMENT) == resolvable
    # bf16 stores as f16: the CPU plugin has no bf16 weight container, the width is the
    # same one, and the difference in kind travels in the label rather than in silence.
    assert openvino_engine._STORAGE_ELEMENT["bf16"] == "f16"
    for precision, element in openvino_engine._STORAGE_ELEMENT.items():
        assert openvino_engine._element_bits(element) == (
            32 if precision == "fp32" else 16
        )


# ── Plan validation ─────────────────────────────────────────────────────────

def test_plan_rejects_models_outside_the_charter() -> None:
    with pytest.raises(SystemExit) as caught:
        plan.parse_args(["--models", "meta-llama/Llama-3-70B"])
    assert "fixed by the charter" in str(caught.value)


def test_plan_rejects_an_unknown_engine_before_the_run_starts() -> None:
    with pytest.raises(SystemExit) as caught:
        plan.parse_args(["--engines", "definitely-not-an-engine"])
    assert "unknown engine" in str(caught.value)


def test_plan_restores_the_measurements_every_derived_figure_needs() -> None:
    """Batch 1 and the primary lengths are denominators, not preferences."""
    config = plan.parse_args([
        "--batch-sizes", "4,8", "--prompt-tokens", "64", "--output-tokens", "64",
        "--primary-prompt-tokens", "256", "--primary-output-tokens", "128",
    ])
    assert config.batch_sizes[0] == 1
    assert 256 in config.prompt_tokens
    assert 128 in config.output_tokens


def test_plan_refuses_a_single_measured_iteration() -> None:
    with pytest.raises(SystemExit) as caught:
        plan.parse_args(["--measure-iters", "1"])
    assert "dispersion" in str(caught.value)


def test_workload_signature_covers_everything_that_defines_the_work() -> None:
    signature = plan.SuiteConfig().workload_signature()
    for field in ("precision", "batch_sizes", "prompt_tokens", "output_tokens",
                  "warmup_iters", "measure_iters", "seed", "temperature", "threads"):
        assert field in signature


# ── Comparison arithmetic ───────────────────────────────────────────────────

def test_compare_reports_both_operands_with_every_ratio() -> None:
    """A ratio must always be checkable against the numbers it came from."""
    faster = analysis.compare(150.0, 100.0)
    assert faster["ratio"] == pytest.approx(1.5)
    assert faster["subject_improvement_percent"] == pytest.approx(50.0)
    assert faster["subject"] == 150.0 and faster["other"] == 100.0


def test_compare_inverts_direction_for_metrics_where_lower_is_better() -> None:
    latency = analysis.compare(0.5, 1.0, lower_is_better=True)
    assert latency["subject_improvement_percent"] == pytest.approx(50.0)
    assert latency["ratio"] == pytest.approx(2.0)
    worse = analysis.compare(2.0, 1.0, lower_is_better=True)
    assert worse["subject_improvement_percent"] == pytest.approx(-100.0)


def test_a_missing_operand_never_becomes_a_zero() -> None:
    for subject, other in ((None, 100.0), (100.0, None), (0.0, 100.0), (100.0, 0.0)):
        result = analysis.compare(subject, other)
        assert result["comparable"] is False
        assert result["ratio"] is None
        assert result["subject_improvement_percent"] is None


def test_verdict_is_symmetric_around_the_stated_tie_threshold() -> None:
    threshold = analysis.TIE_THRESHOLD * 100.0
    assert analysis.verdict(threshold + 0.1) == "subject"
    assert analysis.verdict(-(threshold + 0.1)) == "competitor"
    assert analysis.verdict(0.0) == "tie"
    assert analysis.verdict(threshold - 0.1) == "tie"
    assert analysis.verdict(-(threshold - 0.1)) == "tie"
    assert analysis.verdict(None) == "no comparison"


def test_representation_difference_is_labelled_on_both_sides() -> None:
    same = {"quantized": False, "representation": "published checkpoint",
            "precision": "fp16", "weight_storage_bits": 16,
            "weight_storage_format": "fp16"}
    assert analysis.comparability(same, dict(same))[0] == analysis.SAME_REPRESENTATION

    quantized = {**same, "quantized": True, "weight_storage_bits": 4,
                 "weight_storage_format": "Q4_K_M"}
    assert analysis.comparability(same, quantized)[0] == \
        analysis.REPRESENTATION_DIFFERENCE
    other_precision = {**same, "precision": "bf16"}
    assert analysis.comparability(same, other_precision)[0] == \
        analysis.REPRESENTATION_DIFFERENCE

    # A 32-bit export against 16-bit tensors is a different amount of memory traffic,
    # so it is labelled even though neither side is quantized.
    exported = {**same, "weight_storage_bits": 32, "weight_storage_format": "fp32",
                "representation": "ONNX graph, float32 weights"}
    assert analysis.comparability(same, exported)[0] == \
        analysis.REPRESENTATION_DIFFERENCE

    # Two different 16-bit containers at the same compute precision is a disclosed
    # storage detail, not a different experiment: both are one rounding step from the
    # same published bf16 checkpoint. The note has to say so.
    bf16_storage = {
        "quantized": False, "precision": "fp16",
        "representation": "compiled AEG artifact, bf16 weight storage, fp16 compute",
        "weight_storage_bits": 16, "weight_storage_format": "bf16",
    }
    label, note = analysis.comparability(bf16_storage, same)
    assert label == analysis.SAME_REPRESENTATION
    assert "bf16" in note and "fp16" in note
    assert "rounding step" in note


#: A row every engine in the field can produce: the published checkpoint, executed on
#: the accelerator at the run's precision. Each test below changes exactly one field of
#: it, so the verdict it asserts is attributable to that field alone.
_LIKE_FOR_LIKE_ROW = {
    "quantized": False, "representation": "published checkpoint",
    "precision": "fp16", "weight_storage_bits": 16, "weight_storage_format": "fp16",
    "execution_device": "cuda:0", "execution_device_class": "nvidia-gpu",
    "completion_tokens": 128,
}


def test_two_kinds_of_hardware_are_not_a_representation_difference() -> None:
    """Judged before anything about the weights, and reported as its own kind of gap.

    OpenVINO ships no CUDA plugin, so on an NVIDIA host it executes on the CPU cores
    while the rest of the field is on the card. The percentage from such a pair is a
    real measurement of two machines, and filing it under weight formats would put a
    hardware gap where a reader expects a rounding detail.
    """
    on_host = {**_LIKE_FOR_LIKE_ROW, "execution_device": "CPU",
               "execution_device_class": "cpu"}
    label, note = analysis.comparability(_LIKE_FOR_LIKE_ROW, on_host)
    assert label == analysis.DEVICE_DIFFERENCE
    assert "different hardware" in note and "cpu" in note
    assert "cannot be read as a difference between the stacks" in note
    # It outranks every other difference present, including a quantization: two stacks
    # on two kinds of chip are not being compared with each other at all.
    heavier = {**on_host, "quantized": True, "weight_storage_bits": 4}
    assert analysis.comparability(_LIKE_FOR_LIKE_ROW, heavier)[0] == \
        analysis.DEVICE_DIFFERENCE
    # A second card of the same class is the same hardware; the index is not a gap.
    same_class = {**_LIKE_FOR_LIKE_ROW, "execution_device": "cuda:1"}
    assert analysis.comparability(_LIKE_FOR_LIKE_ROW, same_class)[0] == \
        analysis.SAME_REPRESENTATION


def test_generations_of_different_lengths_are_not_a_like_for_like_rate() -> None:
    """The same cell is not the same work if one side stopped early.

    Every engine is asked for the same token count and pinned to it wherever the API
    takes a minimum length. Aether's public generate does not, so a short generation is
    caught here from the count instead: one prefill amortized over 64 tokens against the
    same prefill over 128 is a different rate, and it flatters whoever produced fewer.
    """
    short = {**_LIKE_FOR_LIKE_ROW, "completion_tokens": 64}
    label, note = analysis.comparability(_LIKE_FOR_LIKE_ROW, short)
    assert label == analysis.WORK_DIFFERENCE
    assert "128 tokens against 64" in note
    # Both directions, from the same code path: whoever generated fewer, it is labelled.
    assert analysis.comparability(short, _LIKE_FOR_LIKE_ROW)[0] == \
        analysis.WORK_DIFFERENCE
    assert analysis.comparability(_LIKE_FOR_LIKE_ROW, dict(_LIKE_FOR_LIKE_ROW))[0] == \
        analysis.SAME_REPRESENTATION
    # And it outranks a weight-format difference, which would otherwise absorb it.
    quantized_and_short = {**short, "quantized": True, "weight_storage_bits": 4}
    assert analysis.comparability(_LIKE_FOR_LIKE_ROW, quantized_and_short)[0] == \
        analysis.WORK_DIFFERENCE


def test_the_severest_difference_present_is_the_one_a_summary_reports() -> None:
    """An aggregate row is summarised by its worst verdict, not by its commonest.

    A set of pairings that crossed a hardware boundary and also differs in weight format
    must not collapse to "format differs": the collapsed verdict is the whole of what a
    reader sees on an aggregated row. The order comes from the module's own severity
    list, so a verdict added to the vocabulary cannot drift out of step with the way
    sets of verdicts are reduced.
    """
    order = analysis.COMPARABILITY_LABELS
    assert order[0] == analysis.DEVICE_DIFFERENCE
    assert order.index(analysis.WORK_DIFFERENCE) < \
        order.index(analysis.REPRESENTATION_DIFFERENCE)
    assert order[-1] == analysis.SAME_REPRESENTATION, "clean is the least severe"

    assert analysis._dominant_label([]) == analysis.SAME_REPRESENTATION
    mixed = [analysis.REPRESENTATION_DIFFERENCE, analysis.SAME_REPRESENTATION,
             analysis.WORK_DIFFERENCE, analysis.DEVICE_DIFFERENCE]
    assert analysis._dominant_label(mixed) == analysis.DEVICE_DIFFERENCE
    assert analysis._dominant_label(mixed[:3]) == analysis.WORK_DIFFERENCE
    assert analysis._dominant_label(mixed[:2]) == analysis.REPRESENTATION_DIFFERENCE


# ── End-to-end analysis over a synthetic payload ────────────────────────────

def _cell(batch: int, prompt: int, output: int, throughput: float, latency: float,
          *, primary: bool = False, sweeps: list[str] | None = None,
          status: str = status_mod.MEASURED, reason: str = "",
          produced: int | None = None) -> dict[str, Any]:
    """One planned cell and its measurement.

    ``produced`` is what the engine actually generated when that differs from what the
    cell asked for: the planned length keys the cell, the produced length is what a
    rate is derived from, and a run where an engine stopped early is the case the two
    have to be kept apart for.
    """
    record: dict[str, Any] = {
        "kind": "primary" if primary else "batch",
        "batch_size": batch, "prompt_tokens": prompt, "output_tokens": output,
        "sweeps": sweeps or ["batch"], "is_primary": primary, "status": status,
        "reason": reason,
    }
    if status != status_mod.MEASURED:
        return record
    record["measurement"] = {
        "status": "ok", "prompt_tokens": prompt,
        "completion_tokens": output if produced is None else produced,
        "latency_s": {"n": 10, "median": latency, "mean": latency, "stdev": 0.01,
                      "min": latency, "max": latency, "p95": latency, "p99": latency,
                      "coefficient_of_variation": 0.01},
        "tokens_per_s": {"median": throughput},
        "host_during_inference": {"rss_peak_bytes": 1_000_000_000},
        "gpu_peak": {"available": True,
                     "devices": [{"peak_reserved_bytes": 2_000_000_000}]},
    }
    record["derived"] = {
        "total_tokens_per_s": throughput, "per_request_tokens_per_s": throughput / batch,
        "decode_tokens_per_s": throughput, "prompt_tokens_per_s": 500.0,
        "ttft_s": 0.05, "tpot_ms": 10.0, "end_to_end_latency_s": latency,
        "cold_latency_s": latency * 2, "iterations": 10,
        "coefficient_of_variation": 0.01,
        "latency_stats": record["measurement"]["latency_s"],
        "throughput_stats": record["measurement"]["tokens_per_s"],
    }
    return record


def _run(engine: str, cells: list[dict[str, Any]], *, build_s: float | None,
         load_s: float, quantized: bool = False,
         representation: str = "published checkpoint",
         persistence: str = "none",
         describe_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One engine's run record, as the worker writes it.

    ``describe_extra`` carries what the engine reported about its own execution - the
    device, its class, the precision it ended up in, the thread count - which is the
    half of the record every parity check is made against.
    """
    return {
        "engine": engine, "model": "Qwen/Qwen3-0.6B", "precision": "bf16",
        "status": status_mod.MEASURED, "cells": cells,
        "spec": {"has_build_phase": build_s is not None, "taxonomy": ["runtime"],
                 "artifact_persistence": persistence},
        "describe": {"representation": representation, "quantized": quantized,
                     **(describe_extra or {})},
        "load": {"status": "ok", "prepare_s": build_s, "load_s": load_s,
                 "total_s": (build_s or 0.0) + load_s, "notes": {}},
        "artifact": {"has_build_phase": build_s is not None, "persistence": persistence,
                     "build_s": build_s, "load_s": load_s,
                     "total_startup_s": (build_s or 0.0) + load_s,
                     "artifact_bytes": 1024 if build_s else None},
        "correctness_sample": {
            "status": status_mod.MEASURED, "token_ids": [1, 2, 3, 4],
            "text": "hello world", "completion_tokens": 4,
        },
    }


def _payload(runs: list[dict[str, Any]], reuse: list[dict[str, Any]] | None = None,
             ) -> dict[str, Any]:
    return {
        "suite_version": "test", "generated_at": "now",
        "plan": {"temperature": 0.0, "amortization_runs": [1, 100],
                 "resolved_precision": "bf16", "precision_reason": "test",
                 "models": ["Qwen/Qwen3-0.6B"], "engines": [r["engine"] for r in runs],
                 "warmup_iters": 3, "measure_iters": 10, "seed": 1, "top_p": 1.0,
                 "top_k": 0, "threads": 4, "invocation": "pytest"},
        "workload_signature": {}, "hardware": {"accelerator": "cuda"},
        "environment": {"software": {}}, "engine_catalogue": {},
        "models": {}, "runs": runs, "reuse_runs": reuse or [],
        "worker_processes": [],
    }


def test_unmeasured_cells_survive_flattening_with_their_reason() -> None:
    """A row that was not measured has to reach the report, not vanish from it."""
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(8, 256, 128, 0.0, 0.0, status=status_mod.OOM,
                  reason="CUDA out of memory"),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
    ]
    rows = analysis.flatten(_payload(runs))
    assert len(rows) == 2
    unmeasured = next(row for row in rows if row["batch_size"] == 8)
    assert unmeasured["status"] == status_mod.OOM
    assert unmeasured["total_tokens_per_s"] is None
    assert "out of memory" in unmeasured["reason"]


def test_wins_and_losses_are_both_reported_from_the_same_cells() -> None:
    """The anti-bias property: a loss is recorded exactly like a win."""
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 90.0, 5.7),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [
            _cell(1, 256, 128, 25.0, 5.1, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 120.0, 4.3),
        ], build_s=None, load_s=6.0),
    ]
    result = analysis.analyze(_payload(runs))
    comparisons = result["comparisons"]
    assert len(comparisons) == 2
    by_batch = {item["batch_size"]: item for item in comparisons}
    assert by_batch[1]["winner"] == "subject"
    assert by_batch[1]["throughput"]["subject_improvement_percent"] == pytest.approx(60.0)
    assert by_batch[4]["winner"] == "competitor"
    assert by_batch[4]["throughput"]["subject_improvement_percent"] == pytest.approx(-25.0)

    win_loss = result["win_loss"]["all"]
    assert win_loss["wins"] == 1
    assert win_loss["losses"] == 1
    assert win_loss["compared"] == 2
    # The extremes must name the real best and worst case, not the best twice.
    assert result["win_loss"]["largest_advantage"]["batch_size"] == 1
    assert result["win_loss"]["largest_disadvantage"]["batch_size"] == 4


def test_scaling_efficiency_is_measured_against_the_engine_own_batch_one() -> None:
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 80.0, 6.4),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
    ]
    scaling = {
        (entry["batch_size"]): entry
        for entry in analysis.analyze(_payload(runs))["batch_scaling"]
    }
    assert scaling[1]["scaling_efficiency_percent"] == pytest.approx(100.0)
    # Double the throughput at four times the width is 50% of linear.
    assert scaling[4]["scaling_vs_batch1"] == pytest.approx(2.0)
    assert scaling[4]["scaling_efficiency_percent"] == pytest.approx(50.0)
    assert scaling[4]["per_request_tokens_per_s"] == pytest.approx(20.0)


def test_break_even_solves_the_crossing_point_and_names_the_never_case() -> None:
    """Aether pays 16s more to start and saves 1.9s per request: about 8 requests."""
    runs = [
        _run("aether", [_cell(1, 256, 128, 40.0, 3.2, primary=True,
                              sweeps=["batch", "prompt", "output"])],
             build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [_cell(1, 256, 128, 25.0, 5.1, primary=True,
                                    sweeps=["batch", "prompt", "output"])],
             build_s=None, load_s=8.0),
    ]
    reuse = [{
        "engine": "aether", "model": "Qwen/Qwen3-0.6B", "mode": "reuse",
        "status": status_mod.MEASURED, "load": {"total_s": 4.5},
        "first_inference": {"cold_latency_s": 3.6},
    }]
    economics = analysis.analyze(_payload(runs, reuse))["compile_economics"]
    entry = next(item for item in economics["entries"] if item["engine"] == "aether")
    assert entry["second_process_load_s"] == pytest.approx(4.5)
    assert entry["total_cost_s"]["100"]["cold_first_deployment"] == pytest.approx(
        24.0 + 100 * 3.2
    )
    assert entry["total_cost_s"]["100"]["warm_reused_artifact"] == pytest.approx(
        4.5 + 100 * 3.2
    )
    crossing = next(
        item for item in economics["break_even"]
        if item["subject"] == "aether" and item["competitor"] == "transformers"
    )
    assert crossing["break_even_runs"] == pytest.approx(16.0 / 1.9, rel=1e-6)

    # Reverse it: when Aether is slower per request, no run count repays the build.
    slower = [
        _run("aether", [_cell(1, 256, 128, 20.0, 6.0, primary=True,
                              sweeps=["batch", "prompt", "output"])],
             build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [_cell(1, 256, 128, 25.0, 5.1, primary=True,
                                    sweeps=["batch", "prompt", "output"])],
             build_s=None, load_s=8.0),
    ]
    never = next(
        item for item in
        analysis.analyze(_payload(slower))["compile_economics"]["break_even"]
        if item["subject"] == "aether" and item["competitor"] == "transformers"
    )
    assert never["break_even_runs"] is None
    assert "no number of requests" in never["interpretation"]


def test_correctness_classes_separate_rounding_from_a_different_computation() -> None:
    identical = {"identical": True, "matching_prefix_fraction": 1.0}
    text_same = {"identical": False, "matching_prefix_fraction": 0.0,
                 "first_divergence_index": 0}
    late_split = {"identical": False, "matching_prefix_fraction": 0.9,
                  "first_divergence_index": 57}
    early_split = {"identical": False, "matching_prefix_fraction": 0.02,
                   "first_divergence_index": 1}

    assert analysis._classify(identical, {"identical": True}, True)[0] == \
        analysis.EXACT_MATCH
    # Identical decoded text with different ids is still an exact match, and the
    # basis has to say which observable it rests on.
    label, basis = analysis._classify(text_same, {"identical": True}, True)
    assert label == analysis.EXACT_MATCH
    assert "text" in basis
    assert analysis._classify(late_split, {"identical": False}, True)[0] == \
        analysis.NUMERICALLY_EQUIVALENT
    assert analysis._classify(early_split, {"identical": False}, True)[0] == \
        analysis.DIFFERENT_OUTPUT
    assert analysis._classify(early_split, {"identical": False}, False)[0] == \
        analysis.EXPECTED_SAMPLING_DIFFERENCE


# ── Output artifacts ────────────────────────────────────────────────────────

def _analyzed() -> tuple[dict[str, Any], dict[str, Any]]:
    runs = [
        _run("aether", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 90.0, 5.7),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact"),
        _run("transformers", [
            _cell(1, 256, 128, 25.0, 5.1, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 120.0, 4.3),
            _cell(8, 256, 128, 0.0, 0.0, status=status_mod.OOM, reason="out of memory"),
        ], build_s=None, load_s=6.0),
    ]
    payload = _payload(runs)
    return payload, analysis.analyze(payload)


def test_csv_keeps_unmeasured_rows_so_nothing_averages_them_as_zero(
    tmp_path: Path,
) -> None:
    import csv

    _, analyzed = _analyzed()
    target = tmp_path / "results.csv"
    report.write_csv(analyzed, target)
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert len(rows) == len(analyzed["rows"])
    oom = next(row for row in rows if row["status"] == status_mod.OOM)
    assert oom["total_tokens_per_s"] == ""
    assert oom["batch_size"] == "8"


def test_comparison_csv_records_the_sign_of_every_result(tmp_path: Path) -> None:
    import csv

    _, analyzed = _analyzed()
    target = tmp_path / "comparisons.csv"
    report.write_comparison_csv(analyzed, target)
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    signs = {row["winner"] for row in rows}
    assert signs == {"subject", "competitor"}
    losing = next(row for row in rows if row["winner"] == "competitor")
    assert float(losing["subject_improvement_percent"]) < 0
    # Ordered pairs: every comparison appears from both sides.
    assert {row["subject"] for row in rows} == {"aether", "transformers"}


def test_report_states_both_the_win_and_the_loss(tmp_path: Path) -> None:
    payload, analyzed = _analyzed()
    text = report.build_report(payload, analyzed, {"written": [], "skipped": [],
                                                   "directory": "graphs"})
    assert "## Head-to-head results" in text
    assert "#### Cells `aether` lost" in text
    assert "#### Cells `transformers` lost" in text, (
        "every measured engine must get the same treatment"
    )
    assert "-25.0%" in text, "the losing comparison must appear with its sign"
    assert "+60.0%" in text
    # The unmeasured cell has to be visible as unmeasured, not as absence.
    assert "OOM" in text
    for heading in ("## Compilation economics", "## Correctness",
                    "## Statistical quality", "## Final rankings",
                    "## Limitations", "## Reproducibility", "## Methodology"):
        assert heading in text


def test_charts_skip_rather_than_invent_when_data_is_absent(tmp_path: Path) -> None:
    empty = {
        "rows": [], "primary_metric": "total_tokens_per_s",
        "primary_metric_label": "tok/s", "batch_scaling": [], "per_competitor": {},
        "compile_economics": {"entries": []},
    }
    manifest = charts.write_all(empty, tmp_path)
    assert manifest["written"] == []
    assert manifest["skipped"], "a skipped figure must be recorded with a reason"
    assert not list(tmp_path.glob("*.png"))


def test_charts_are_written_for_measured_data(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _, analyzed = _analyzed()
    manifest = charts.write_all(analyzed, tmp_path)
    assert manifest["written"]
    for name in manifest["written"]:
        assert (tmp_path / name).stat().st_size > 0


# ── Worker planning and derivation ──────────────────────────────────────────

def test_the_primary_cell_is_planned_once_and_serves_every_sweep() -> None:
    """Measuring it three times would spend budget to sample the same thing."""
    from benchmark.suite import worker

    cells = worker._plan_cells(
        {"primary_prompt_tokens": 256, "primary_output_tokens": 128,
         "batch_sizes": [1, 2, 4], "output_tokens": [32, 128, 512]},
        {"32": {}, "256": {}, "1024": {}},
    )
    primaries = [cell for cell in cells if cell.get("is_primary")]
    assert len(primaries) == 1
    assert set(primaries[0]["sweeps"]) == {"batch", "prompt", "output"}
    keys = [(c["batch_size"], c["prompt_tokens"], c["output_tokens"]) for c in cells]
    assert len(keys) == len(set(keys)), "no configuration may be planned twice"
    assert (2, 256, 128) in keys and (4, 256, 128) in keys
    assert (1, 32, 128) in keys and (1, 1024, 128) in keys
    assert (1, 256, 32) in keys and (1, 256, 512) in keys


def test_derivation_separates_prefill_from_decode_and_admits_when_it_cannot() -> None:
    from benchmark.suite import worker

    measurement = {
        "prompt_tokens": 100, "completion_tokens": 11,
        "latency_s": {"median": 2.0, "n": 10},
        "tokens_per_s": {"median": 5.5},
    }
    derived = worker._derive(measurement, batch=1, prefill_s=1.0, ttft_s=1.05)
    # Ten of the eleven tokens come out of the decode loop, in the second left
    # after prefill.
    assert derived["decode_tokens_per_s"] == pytest.approx(10.0)
    assert derived["tpot_ms"] == pytest.approx(100.0)
    assert derived["prompt_tokens_per_s"] == pytest.approx(100.0)
    assert derived["ttft_s"] == pytest.approx(1.05)

    # With no prefill measurement, decode must stay undefined rather than borrow
    # the end-to-end rate under a name that means something narrower.
    without = worker._derive(measurement, batch=1, prefill_s=None, ttft_s=None)
    assert without["decode_tokens_per_s"] is None
    assert without["prompt_tokens_per_s"] is None
    assert without["total_tokens_per_s"] == pytest.approx(5.5)


def test_batch_throughput_is_aggregate_and_per_request_is_derived_from_it() -> None:
    from benchmark.suite import worker

    measurement = {
        "prompt_tokens": 100, "completion_tokens": 10,
        "latency_s": {"median": 2.0, "n": 10},
        "tokens_per_s": {"median": 40.0},
    }
    derived = worker._derive(measurement, batch=8, prefill_s=None, ttft_s=None)
    assert derived["total_tokens_per_s"] == pytest.approx(40.0)
    assert derived["per_request_tokens_per_s"] == pytest.approx(5.0)
    assert derived["generated_tokens_total"] == 80


def test_a_crashed_worker_leaves_a_record_with_no_measurement_in_it() -> None:
    from benchmark.suite import orchestrate

    record = orchestrate._orphan_record(
        "openvino", "Qwen/Qwen3-0.6B", "bf16",
        {"returncode": None, "elapsed_s": 900.0, "timed_out": True},
    )
    assert record["status"] == status_mod.FAILED
    assert record["cells"] == []
    assert "timeout" in record["reason"]
    assert not status_mod.is_measured(record)


# ── Device parity ───────────────────────────────────────────────────────────

def test_every_engine_sees_one_accelerator_by_default() -> None:
    """A runtime that shards would otherwise be measured on more hardware."""
    assert plan.SuiteConfig().devices == 1
    assert plan.parse_args([]).devices == 1
    assert plan.parse_args(["--devices", "2"]).devices == 2
    assert plan.parse_args([]).workload_signature()["devices"] == 1


#: Environment variables :func:`hardware.visible_devices` writes for real.  It has
#: to write them: it runs in the worker before torch opens a CUDA context, which is
#: the last moment ``CUDA_VISIBLE_DEVICES`` is still read.  A test that calls it in
#: the pytest process is therefore editing the session's own environment.
_VISIBILITY_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "AETHER_EXECUTION_DEVICES",
    "AETHER_FORCE_TENSOR_PARALLEL",
)

#: Those variables as this module found them, captured at import time - before any
#: test in it has run - so the guard below compares against the environment the
#: developer actually has.  A multi-GPU host exports ``CUDA_VISIBLE_DEVICES``
#: legitimately, and asserting it is empty would fail there for the wrong reason.
_VISIBILITY_ON_IMPORT = {name: os.environ.get(name) for name in _VISIBILITY_VARIABLES}


def test_device_restriction_is_applied_through_visibility(monkeypatch: Any) -> None:
    """Visibility, not a patched placement path: no engine's own logic is changed."""
    for name in _VISIBILITY_VARIABLES:
        # Registering the current value is what gives the function's writes an undo.
        # ``delenv`` alone records nothing when the variable is already absent, so the
        # ``cuda:0`` this test provokes used to outlive it and make every later
        # ``load_model`` in the session demand a GPU this host need not have.
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name, raising=False)

    record = hardware.visible_devices(1)
    assert record["restricted"] is True
    assert record["CUDA_VISIBLE_DEVICES"] == "0"
    assert record["AETHER_EXECUTION_DEVICES"] == "cuda:0"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    unrestricted = hardware.visible_devices(0)
    assert unrestricted["restricted"] is False


def test_restricting_visibility_does_not_outlive_the_call_that_asked_for_it() -> None:
    """The suite's own tests must not leave a GPU demand in the session's environment.

    This is the guard for the failure it replaces: three unrelated ``load_model``
    tests failed with "requested CUDA device 'cuda:0' is unavailable" purely because
    they ran after the test above in the same process.
    """
    for name, original in _VISIBILITY_ON_IMPORT.items():
        assert os.environ.get(name) == original, (
            f"{name} was left as {os.environ.get(name)!r} by an earlier test in this "
            f"process; it was {original!r} before the module ran"
        )


def test_aether_adapter_names_the_single_device_it_was_given() -> None:
    """Belt and braces: the execution device is recorded, not left to be inferred."""
    from benchmark.suite.engines import aether_engine

    gpu = _hardware(nvidia=True, gpu_count=2, compute_capabilities=["7.5", "7.5"])
    engine = aether_engine.build(gpu, "Qwen/Qwen3-0.6B", "fp16",
                                plan.SuiteConfig(devices=1))
    assert engine.execution_devices == ["cuda:0"]
    two = aether_engine.build(gpu, "Qwen/Qwen3-0.6B", "fp16",
                              plan.SuiteConfig(devices=2))
    assert two.execution_devices == ["cuda:0", "cuda:1"]


# ── Neutrality ──────────────────────────────────────────────────────────────

def test_the_pairwise_matrix_holds_every_ordered_pair() -> None:
    _, analyzed = _analyzed()
    pairs = {(item["subject"], item["competitor"]) for item in analyzed["pairwise"]
             if item["throughput"].get("comparable")}
    assert ("aether", "transformers") in pairs
    assert ("transformers", "aether") in pairs


def test_pairwise_comparisons_are_antisymmetric() -> None:
    """A wins over B by x% must be recorded as B losing to A, from the same numbers."""
    _, analyzed = _analyzed()
    by_pair = {
        (item["subject"], item["competitor"], item["batch_size"]): item
        for item in analyzed["pairwise"] if item["throughput"].get("comparable")
    }
    for (subject, competitor, batch), item in by_pair.items():
        mirror = by_pair.get((competitor, subject, batch))
        assert mirror is not None, "every pairing must appear in both directions"
        assert item["throughput"]["subject"] == pytest.approx(
            mirror["throughput"]["other"]
        )
        # A won by +60% means B lost; the mirrored percentage is not the negation
        # (percentages are asymmetric) but the verdicts must be opposite.
        if item["winner"] == "subject":
            assert mirror["winner"] == "competitor"
        elif item["winner"] == "competitor":
            assert mirror["winner"] == "subject"
        else:
            assert mirror["winner"] == "tie"


def test_standings_score_every_engine_with_the_same_measure() -> None:
    _, analyzed = _analyzed()
    standings = analyzed["standings"]
    assert {entry["engine"] for entry in standings} == {"aether", "transformers"}
    assert [entry["rank"] for entry in standings] == [1, 2]
    for entry in standings:
        assert entry["compared"] > 0
        assert 0.0 <= entry["win_rate_percent"] <= 100.0
        assert entry["median_percent_of_best"] is not None
        # A share of the fastest engine in a cell cannot exceed the fastest engine.
        assert entry["median_percent_of_best"] <= 100.0
    # Ordering follows the score, and the score alone.
    assert standings[0]["median_percent_of_best"] >= standings[1]["median_percent_of_best"]


def test_per_engine_views_exist_for_every_measured_engine() -> None:
    _, analyzed = _analyzed()
    assert set(analyzed["per_engine"]) == set(analyzed["engines_measured"])
    for view in analyzed["per_engine"].values():
        assert "win_loss" in view and "per_competitor" in view


def test_report_gives_every_engine_the_same_treatment() -> None:
    payload, analyzed = _analyzed()
    text = report.build_report(payload, analyzed, {"written": [], "skipped": [],
                                                   "directory": "graphs"})
    assert "# Inference Engine Benchmark Report" in text
    for engine in analyzed["engines_measured"]:
        assert f"#### Cells `{engine}` won" in text
        assert f"#### Cells `{engine}` lost" in text
        assert f"#### `{engine}` against each opponent, aggregated" in text
    assert "## Head-to-head results" in text
    assert "### Pairwise matrix" in text


def test_focus_narrows_the_drill_down_without_changing_any_number() -> None:
    payload, analyzed = _analyzed()
    focused = dict(analyzed, focus="transformers")
    text = report.build_report(payload, focused, {"written": [], "skipped": [],
                                                  "directory": "graphs"})
    assert "#### Cells `transformers` won" in text
    assert "#### Cells `aether` won" not in text
    # The standings still cover the whole field.
    assert "`aether`" in text


def _field_payload() -> dict[str, Any]:
    """One model, two cells, three engines: two on the card, one on the host CPU.

    The shape a real run has on an NVIDIA host, in miniature. It is the smallest payload
    in which the device label, the exhaustive bucketing, the cross-device column of the
    standings and the report's execution-parity table can all be checked, and in which a
    short generation coexists with a hardware difference so their order of severity is
    exercised by the pipeline rather than only by the label function.
    """
    on_card = {"execution_device": "cuda:0", "execution_device_class": "nvidia-gpu",
               "precision": "bf16", "weight_storage_bits": 16,
               "weight_storage_format": "bf16", "threads": 2,
               "ttft_method": "streaming"}
    on_host = {**on_card, "execution_device": "CPU", "execution_device_class": "cpu",
               "precision": "fp32", "weight_storage_format": "f16"}
    runs = [
        _run("transformers", [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 120.0, 4.3),
        ], build_s=None, load_s=6.0, describe_extra=on_card),
        _run("openvino", [
            _cell(1, 256, 128, 6.0, 21.0, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 18.0, 28.0),
        ], build_s=30.0, load_s=8.0, persistence="portable-artifact",
            describe_extra=on_host),
        _run("aether", [
            _cell(1, 256, 128, 52.0, 2.5, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            # The same cell, stopped early: 100 tokens where the field produced 128.
            _cell(4, 256, 128, 150.0, 3.4, produced=100),
        ], build_s=20.0, load_s=4.0, persistence="portable-artifact",
            describe_extra=on_card),
    ]
    return _payload(runs)


def test_every_comparison_lands_in_exactly_one_bucket() -> None:
    """The buckets partition the comparable set, so classifying cannot hide anything.

    A verdict that exists in the analysis but not in the summary drops its comparisons
    out of the report while every printed count stays internally consistent - the worst
    kind of omission, because nothing looks wrong. The buckets are built from the
    module's own label list, and here they are asserted to add up.
    """
    analyzed = analysis.analyze(_field_payload())
    labelled = {
        item["comparability"] for item in analyzed["pairwise"]
        if item["throughput"].get("comparable")
    }
    assert analysis.DEVICE_DIFFERENCE in labelled, "the CPU-bound engine must be flagged"
    assert analysis.WORK_DIFFERENCE in labelled, "the short generation must be flagged"
    assert analysis.SAME_REPRESENTATION in labelled, "and a clean pair must survive"

    for engine, view in analyzed["per_engine"].items():
        summary = view["win_loss"]
        parts = ("same_representation", "representation_difference",
                 "device_difference", "work_difference")
        assert sum(summary[name]["compared"] for name in parts) == \
            summary["all"]["compared"], f"{engine}'s buckets do not sum to its total"
        for name in parts:
            counted = summary[name]
            assert counted["wins"] + counted["losses"] + counted["ties"] == \
                counted["compared"]


def test_a_rank_earned_on_other_hardware_travels_with_that_fact() -> None:
    """An engine that ran on the CPU still has a rank, and the rank says what it is.

    Dropping it from the standings would be the other error: the run happened and the
    numbers are real. What must not happen is a table where a row measured on two CPU
    cores sits beside rows measured on a T4 with nothing to distinguish them, because
    then the reader reads a runtime comparison off a hardware comparison.
    """
    analyzed = analysis.analyze(_field_payload())
    by_engine = {entry["engine"]: entry for entry in analyzed["standings"]}
    assert set(by_engine) == {"transformers", "openvino", "aether"}
    host_bound = by_engine["openvino"]
    assert host_bound["cross_device"] == host_bound["compared"], (
        "every pairing this engine had crossed a hardware boundary"
    )
    assert host_bound["same_representation"] == 0
    for engine in ("transformers", "aether"):
        entry = by_engine[engine]
        assert 0 < entry["cross_device"] < entry["compared"], (
            f"{engine} was paired both with the CPU-bound engine and against the card"
        )

    # The aggregated per-opponent verdict collapses to the severest label present, so
    # the pairing against the CPU-bound engine is never summarised as a format detail.
    against = analyzed["per_engine"]["aether"]["per_competitor"]
    assert against["openvino"]["comparability"] == analysis.DEVICE_DIFFERENCE
    assert against["openvino"]["device_difference_cells"] == \
        against["openvino"]["cells"]
    assert against["openvino"]["same_representation_cells"] == 0
    assert against["openvino"][
        "median_improvement_percent_same_representation"] is None, (
        "there is no like-for-like median to quote against an engine on other hardware"
    )
    assert against["transformers"]["work_difference_cells"] == 1, (
        "the cell Aether stopped early in is counted, not averaged away"
    )


def test_the_report_prints_what_each_engine_did_beside_what_it_was_asked_for() -> None:
    """The controlled variables are only controlled if the report shows who kept them.

    A precision and a device are requested once for the whole run, and an engine can
    decline either: a plugin with no driver for the accelerator falls back to the host,
    and one that cannot execute a format widens it. Neither shows up in a throughput
    number - a row that ran on the CPU at fp32 just looks slow - so both are printed
    next to the request, which is also the evidence behind the labels in the standings.
    """
    payload = _field_payload()
    analyzed = analysis.analyze(payload)
    text = report.build_report(payload, analyzed, {"written": [], "skipped": [],
                                                   "directory": "graphs"})
    assert "**What each engine executed on**" in text
    for column in ("Execution device", "Device class", "Precision asked for",
                   "Precision reported", "CPU threads", "TTFT method"):
        assert column in text, f"the parity table must print {column!r}"
    parity = text.split("**What each engine executed on**", 1)[1]
    host_row = next(line for line in parity.splitlines() if "`openvino`" in line)
    assert "cpu" in host_row and "fp32" in host_row and "bf16" in host_row, (
        "the CPU fallback and the widened precision must both be readable in one row"
    )
    card_row = next(line for line in parity.splitlines() if "`aether`" in line)
    assert "nvidia-gpu" in card_row and "cuda:0" in card_row
    # The instrument behind the one metric an engine can be measured for differently.
    assert "streaming" in card_row and "streaming" in host_row

    # And the caveat is stated, not left for the reader to deduce from the table.
    assert "no CUDA plugin" in text
    assert "hardware differs" in text, "the label has a reading in the tables"


# ── One stopwatch for time to first token ───────────────────────────────────

def test_the_first_token_is_timed_by_one_stopwatch_wherever_an_engine_can_share_it() -> None:
    """A ranked metric must not be measured by a different instrument per engine.

    Time to first token is ranked head-to-head and turned into a percentage against
    every competitor, so the machinery behind it is part of the comparison. Every
    engine whose generate accepts a streamer is measured by the same function object -
    not by its own copy of the same idea - and the one engine that has no stream to
    subscribe to overrides the method itself, so what its spec declares is what its
    code does.
    """
    from benchmark import backend_transformers, backends
    from benchmark.suite.engines import base, hf_transformers, openvino_engine, pytorch_native

    assert (backend_transformers.stream_first_token_latency
            is backends.stream_first_token_latency
            is openvino_engine.stream_first_token_latency), (
        "two engines reporting ttft_method='streaming' must share the implementation"
    )
    for module in (hf_transformers, openvino_engine, registry.spec_for("aether")):
        spec = module if isinstance(module, base.EngineSpec) else module.SPEC
        assert spec.ttft_method == "streaming", f"{spec.key} drifted off the shared method"

    # The exception is declared, and it is declared by an engine that really does
    # measure something else: the raw loop has no stream, and timing its generate
    # would make this row Transformers' generate for one metric and not for the rest.
    assert pytorch_native.SPEC.ttft_method == "single_token_call"
    assert (pytorch_native.Engine.first_token_latency
            is not base.BackendAdapterMixin.first_token_latency)
    for key in registry.KEYS:
        assert registry.spec_for(key).ttft_method, f"{key} declares no ttft method"


def test_each_streaming_engine_reaches_that_stopwatch_with_its_own_model(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Sharing the timer is only neutrality if both engines actually route to it.

    Checked without a model: each adapter is handed a stand-in for the model and the
    tokenizer it would have loaded, and the shared timer is replaced by a recorder. What
    must match between the two calls is the request - the same token budget, its own
    model, its own tokenizer - because that is what makes the two figures comparable.
    """
    from benchmark import backend_transformers
    from benchmark.suite.engines import openvino_engine

    class _Encoding(dict):
        def items(self) -> Any:  # so _encode's device move is exercised
            return super().items()

    class _Tensor:
        def to(self, _device: str) -> _Tensor:
            return self

    class _Tokenizer:
        pad_token_id = 0

        def __call__(self, prompts: Any, **_: Any) -> Any:
            return _Encoding(input_ids=_Tensor(), attention_mask=_Tensor())

    calls: list[dict[str, Any]] = []

    def _recorder(model: Any, tokenizer: Any, inputs: Any, *,
                  max_new_tokens: int) -> float:
        calls.append({"model": model, "tokenizer": tokenizer,
                      "inputs": dict(inputs), "max_new_tokens": max_new_tokens})
        return 0.25

    monkeypatch.setattr(backend_transformers, "stream_first_token_latency", _recorder)
    monkeypatch.setattr(openvino_engine, "stream_first_token_latency", _recorder)

    reference = backend_transformers.TransformersBackend(device="cpu")
    reference._model, reference._tokenizer = object(), _Tokenizer()
    subject = openvino_engine.Engine(device="CPU")
    subject._model, subject._tokenizer = object(), _Tokenizer()

    for engine in (reference, subject):
        assert engine.first_token_latency("hi", max_new_tokens=64, seed=7) == 0.25

    assert len(calls) == 2, "both engines must go through the shared timer"
    assert {call["max_new_tokens"] for call in calls} == {64}
    assert {tuple(sorted(call["inputs"])) for call in calls} == {
        ("attention_mask", "input_ids")
    }, "both engines must hand the same encoded fields to the same timer"
    assert calls[0]["model"] is not calls[1]["model"], "each times its own model"


def test_the_shared_stopwatch_stops_at_the_first_token_and_never_invents_one() -> None:
    """What the figure includes, and what happens when there is no figure to give.

    Driven with a stand-in model that streams one token, works for a while longer, then
    finishes: the number returned has to be the wait for the first token and not the cost
    of the rest, or every engine's time-to-first-token would silently be a function of
    the output length the plan happened to ask for. A generation that fails must reach
    the runner as a failure, because a cell recorded as unmeasured is honest and a
    latency for a token that never arrived is not.
    """
    import time

    import torch
    from benchmark.backends import UnsupportedConfiguration, stream_first_token_latency

    class _Tokenizer:
        pad_token_id = 0

        def decode(self, ids: Any, **_: Any) -> str:
            # A trailing space, so the streamer's word-boundary flush releases the token
            # rather than holding it back for the next one.
            return " ".join(f"t{int(value)}" for value in ids) + " "

    class _Model:
        def __init__(self, behaviour: str) -> None:
            self.behaviour = behaviour

        def generate(self, *, input_ids: Any, streamer: Any, **_: Any) -> Any:
            streamer.put(input_ids)  # the prompt, which skip_prompt drops
            if self.behaviour == "raises":
                raise RuntimeError("CUDA out of memory")
            if self.behaviour == "silent":
                # A generation that stopped immediately: the streamer still emits
                # once when it ends, and the returned sequence is the prompt.
                streamer.end()
                return input_ids
            streamer.put(torch.tensor([[5]]))
            time.sleep(0.4)  # the tokens after the first must not be in the figure
            streamer.put(torch.tensor([[6]]))
            streamer.end()
            return torch.cat([input_ids, torch.tensor([[5, 6]])], dim=-1)

    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    started = time.perf_counter()
    elapsed = stream_first_token_latency(
        _Model("streams"), _Tokenizer(), inputs, max_new_tokens=32
    )
    total = time.perf_counter() - started
    assert total >= 0.4, "the call still waits for the generation it started"
    assert elapsed < 0.2, (
        f"time to first token ({elapsed:.3f}s) counted work done after the first token"
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        stream_first_token_latency(
            _Model("raises"), _Tokenizer(), inputs, max_new_tokens=32
        )
    with pytest.raises(UnsupportedConfiguration, match="no token beyond the prompt"):
        stream_first_token_latency(
            _Model("silent"), _Tokenizer(), inputs, max_new_tokens=32
        )


def _timed_field(methods: dict[str, str]) -> dict[str, Any]:
    """The whole field measured at batch 1, each engine's stopwatch declared.

    Time to first token is the one headline metric an engine can be measured for by a
    different mechanism than its neighbour, so the payload carries the mechanism per
    engine and the values are distinct, which makes both the ordering and the caveat
    checkable from the same run.
    """
    seconds = {"transformers": 0.09, "pytorch_native": 0.04,
               "openvino": 0.62, "aether": 0.03}
    runs = []
    for key in registry.KEYS:
        run = _run(key, [
            _cell(1, 256, 128, 40.0, 3.2, primary=True,
                  sweeps=["batch", "prompt", "output"]),
            _cell(4, 256, 128, 120.0, 4.3),
        ], build_s=None, load_s=6.0, describe_extra={
            "execution_device": "cuda:0", "execution_device_class": "nvidia-gpu",
            "precision": "bf16", "threads": 2, "ttft_method": methods[key],
        })
        for cell in run["cells"]:
            cell["derived"]["ttft_s"] = seconds[key]
        runs.append(run)
    return _payload(runs)


def test_a_ranking_of_figures_taken_two_ways_says_so_wherever_it_is_printed(
        tmp_path: Path) -> None:
    """The field really does mix stopwatches, so the mixing has to be disclosed.

    Three engines expose a stream and are timed by the shared implementation; the raw
    loop has no stream and times the work before its first token exists. That is a
    defensible difference and an undisclosed one would not be, so the method travels
    from each engine's own description into the ranking, the note under it, the summary
    line that quotes the winner, the terminal output, and the per-pair export - the
    same discipline the representation and device labels already get.
    """
    declared = {key: registry.spec_for(key).ttft_method for key in registry.KEYS}
    assert len(set(declared.values())) > 1, (
        "this test exists because the field mixes methods; if it no longer does, "
        "the disclosure is no longer needed and this test should be the one that fails"
    )
    payload = _timed_field(declared)
    analyzed = analysis.analyze(payload)
    ranking = analyzed["rankings"]["ttft"]

    assert ranking["mixed_methods"] is True
    assert ranking["methods"] == declared, "every ranked engine names its own stopwatch"
    assert [entry["engine"] for entry in ranking["order"]] == [
        "aether", "pytorch_native", "transformers", "openvino"
    ], "the ordering itself is by measured value, lowest first"

    # A percentage is only like-for-like when both sides were timed the same way, and
    # the flag is set from the pair rather than from either engine's identity.
    pairs = {
        (item["subject"], item["competitor"]): item["ttft"]
        for item in analyzed["pairwise"] if item["is_primary"]
    }
    assert pairs[("aether", "transformers")]["same_method"] is True
    assert pairs[("transformers", "aether")]["same_method"] is True
    assert pairs[("aether", "pytorch_native")]["same_method"] is False
    assert pairs[("pytorch_native", "aether")]["same_method"] is False
    assert pairs[("aether", "pytorch_native")]["subject_method"] == "streaming"
    assert pairs[("aether", "pytorch_native")]["competitor_method"] == "single_token_call"

    text = report.build_report(payload, analyzed, {"written": [], "skipped": [],
                                                  "directory": "graphs"})
    section = text.split("### Lowest time to first token", 1)[1]
    assert "not like-for-like" in section
    assert "`pytorch_native` by single token call" in section
    assert "`aether` by streaming" in section
    assert "measured by more than one method" in text.split("## Executive summary", 1)[1]
    assert "[mixed methods]" in report.terminal_summary(payload, analyzed, {})

    written = tmp_path / "pairs.csv"
    report.write_comparison_csv(analyzed, written)
    exported = written.read_text(encoding="utf-8").splitlines()
    assert "ttft_same_method" in exported[0]
    assert any("False" in line for line in exported[1:]), (
        "a spreadsheet filtered on the TTFT column has to be able to see which "
        "percentages are between two different measurements"
    )


def test_one_stopwatch_across_the_field_leaves_no_caveat_to_print() -> None:
    """The disclosure is conditional, not decorative.

    If every engine were measured the same way there would be nothing to warn about,
    and a warning printed anyway would train a reader to ignore it.
    """
    payload = _timed_field(dict.fromkeys(registry.KEYS, "streaming"))
    analyzed = analysis.analyze(payload)
    assert analyzed["rankings"]["ttft"]["mixed_methods"] is False
    assert all(item["ttft"]["same_method"] for item in analyzed["pairwise"])

    text = report.build_report(payload, analyzed, {"written": [], "skipped": [],
                                                  "directory": "graphs"})
    assert "not like-for-like" not in text.split(
        "### Lowest time to first token", 1)[1].split("###", 1)[0]
    assert "measured by more than one method" not in text
    assert "[mixed methods]" not in report.terminal_summary(payload, analyzed, {})

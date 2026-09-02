"""OpenVINO: Intel's AOT graph compiler and CPU/iGPU/NPU inference runtime.

The checkpoint is converted once into OpenVINO IR - a real ahead-of-time build
producing a portable directory - and executed by the OpenVINO runtime, which
compiles the IR for the specific device at load time and dispatches its own
kernels. That makes it the closest thing in the field to what Aether does, and
the comparison the local-inference story actually hinges on.

Three things this adapter does that a naive one does not, each because getting it
wrong silently biases the result rather than failing:

* **The run's precision is applied to the IR.** OpenVINO's exporter defaults to
  fp32, so an adapter that only records the requested precision measures a 32-bit
  engine against a field of 16-bit ones and never says so. Decode is memory bound;
  that is roughly a factor of two, attributed to the runtime.
* **The measured IR is the saved IR.** The export is written out, then read back
  and executed. Otherwise a fresh run executes the in-memory graph while a resumed
  run executes the file, and the two are not the same weights.
* **The thread budget is applied through OpenVINO's own knob.** OpenVINO schedules
  on TBB, which does not read ``OMP_NUM_THREADS``; the suite's pinning therefore
  reaches every other engine and misses this one, handing it every core on the box
  while torch is held to the pinned count.

What this adapter cannot fix is that OpenVINO has no CUDA plugin. On an NVIDIA
host it executes on the CPU, which is a different class of hardware from the one
the torch engines get. That is disclosed rather than papered over: the device
actually used is read back from the compiled model and recorded, and
:func:`benchmark.suite.analysis.comparability` labels every comparison that
crosses a device class.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from benchmark.backends import (
    GenerationOutcome,
    LoadOutcome,
    UnsupportedConfiguration,
    set_seed,
    stream_first_token_latency,
)
from benchmark.suite.engines import base

#: Preference order when the operator asks for ``auto``: dedicated matrix hardware
#: before the CPU. Declared as data so the choice is auditable, and applied only to
#: devices OpenVINO reports it can actually see on this host.
DEVICE_PREFERENCE: tuple[str, ...] = ("GPU", "NPU", "CPU")

#: IR weight element type for each precision the suite runs. fp16 and bf16 both map
#: to a 16-bit IR because that is the only 16-bit float OpenVINO's serializer
#: compresses to; the container difference is reported and is the same disclosed
#: difference that already exists between the fp16 framework engines and Aether's
#: bf16 artifact. What matters for a memory-bound decode is the width, and the width
#: is the run's.
_STORAGE_ELEMENT: dict[str, str] = {"fp16": "f16", "bf16": "f16", "fp32": "f32"}

#: Read back from the compiled model, mapped onto the suite's precision names so a
#: comparison of compute precision is a comparison of the same vocabulary.
_PRECISION_FROM_ELEMENT: dict[str, str] = {
    "f32": "fp32", "float32": "fp32", "f16": "fp16", "float16": "fp16",
    "bf16": "bf16", "bfloat16": "bf16",
}

SPEC = base.EngineSpec(
    key="openvino",
    display="OpenVINO",
    taxonomy=(
        base.AOT_COMPILER, base.GRAPH_COMPILER, base.RUNTIME, base.EXECUTION_ENGINE,
        base.KERNEL_OPTIMIZER,
    ),
    summary=(
        "Ahead-of-time conversion to OpenVINO IR, then execution by the OpenVINO "
        "runtime, which compiles the IR for the host device at load time. The IR is "
        "a portable directory another process or machine can load."
    ),
    package="optimum-intel",
    requires=("openvino", "optimum"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    ttft_method="streaming",
    notes=(
        "The IR is written at the run's precision and the element type is read back "
        "from a weight constant after conversion, not assumed. When the stored width "
        "or the executed precision differs from another engine's, every percentage "
        "against this engine is labelled.",
        "OpenVINO has no CUDA plugin. On an NVIDIA host it executes on the CPU, so "
        "its rows are a CPU runtime measured against GPU runtimes; the device is read "
        "back from the compiled model and every comparison that crosses a device "
        "class is labelled DEVICE_DIFFERENCE.",
        "Scheduling is OpenVINO's own, which ignores OMP_NUM_THREADS, so the suite's "
        "thread budget is applied through INFERENCE_NUM_THREADS instead. The latency "
        "performance hint is set because the suite measures one request at a time, "
        "which is the configuration every other engine is also run in.",
        "Time to first token is taken with the shared streaming implementation, the "
        "same function that times the reference engine, because the OpenVINO runtime "
        "is driven through Transformers' own generate and can therefore be measured "
        "the same way rather than by timing a one-token call.",
    ),
)


class Engine(base.BackendAdapterMixin):
    """Convert once to OpenVINO IR, cache it, then generate through the runtime."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, device: str = "auto", cache_dir: str | None = None,
                 threads: int | None = None, **_: Any) -> None:
        self.requested_device = device or "auto"
        self.device = self.requested_device
        self.threads = threads
        self.cache_dir = Path(cache_dir or "benchmark_results/artifacts/openvino")
        self._model: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._requested_precision: str | None = None
        self._convert_s: float = 0.0
        self._reused: bool | None = None
        self._artifact: Path | None = None
        self._ir_element_type: str | None = None
        self._available_devices: list[str] = []
        self._execution_devices: list[str] = []
        self._execution_precision: str | None = None
        self._applied_config: dict[str, Any] = {}

    # ── Device and runtime configuration ────────────────────────────────────

    def _resolve_device(self) -> str:
        """The device to compile for, from what OpenVINO reports it can see.

        ``auto`` is resolved here rather than left to OpenVINO's own ``AUTO`` plugin
        so that the chosen device is a recorded fact instead of an internal decision
        the report would have to infer.
        """
        try:
            import openvino as ov

            self._available_devices = [str(name) for name in ov.Core().available_devices]
        except Exception:  # noqa: BLE001 - reported as unknown rather than guessed
            self._available_devices = []
        if self.requested_device.lower() != "auto":
            return self.requested_device
        families = {name.split(".")[0] for name in self._available_devices}
        for candidate in DEVICE_PREFERENCE:
            if candidate in families:
                return candidate
        return "CPU"

    def _runtime_config(self) -> dict[str, Any]:
        """The properties the compiled model is given, and why each one is there.

        Only two: the thread budget the suite pinned, and the hint that matches how
        the suite measures. Neither changes the arithmetic; both exist to stop this
        engine being run in a configuration the rest of the field is not.
        """
        config: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
        if self.threads:
            config["INFERENCE_NUM_THREADS"] = int(self.threads)
        return config

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "requested_device": self.requested_device,
            "openvino_available_devices": list(self._available_devices),
            "execution_device": ", ".join(self._execution_devices) or self.device,
            "execution_device_class": base.device_class(
                self._execution_devices[0] if self._execution_devices else self.device
            ),
            # The compute precision the plugin reports, which is what a precision
            # comparison has to be made on. An x86 CPU plugin executes fp32 whatever
            # the IR stores, so recording the request here would assert parity the
            # run does not have.
            "precision": self._precision,
            "requested_precision": self._requested_precision,
            "execution_precision_reported": self._execution_precision,
            "precision_source": (
                "read back from the compiled model" if self._execution_precision
                else "the run's requested precision; the plugin did not report one"
            ),
            "ir_element_type": self._ir_element_type,
            "runtime_config": dict(self._applied_config),
            "threads": self.threads,
            "artifact": str(self._artifact) if self._artifact else None,
            "artifact_reused": self._reused,
            "conversion_s": self._convert_s,
            "generation": "OVModelForCausalLM.generate",
            "representation": (
                f"OpenVINO IR, {self._ir_element_type or 'unknown'} weight storage, "
                f"{self._precision or '?'} compute"
            ),
            "weight_storage_bits": _element_bits(self._ir_element_type),
            "weight_storage_format": self._ir_element_type,
            "quantized": False,
            "ttft_method": SPEC.ttft_method,
            "version": base.package_version("openvino"),
        }

    # ── Load ────────────────────────────────────────────────────────────────

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer

        self._requested_precision = precision
        self._precision = precision
        self.device = self._resolve_device()
        self._applied_config = self._runtime_config()

        download_start = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        download_s = time.perf_counter() - download_start

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.cache_dir / f"{model_id.replace('/', '--')}--{precision}"
        self._artifact = artifact
        reuse = (artifact / "openvino_model.xml").exists()
        self._reused = reuse

        start = time.perf_counter()
        try:
            if not reuse:
                self._convert(OVModelForCausalLM, model_id, artifact, precision)
                self._convert_s = time.perf_counter() - start
                start = time.perf_counter()
            # Both paths load the IR from disk, so a fresh run and a resumed run
            # execute the identical file. An in-memory export would differ from the
            # saved one by exactly the compression applied on the way out.
            self._model = base.call_with_supported_kwargs(
                OVModelForCausalLM.from_pretrained, artifact,
                device=self.device, ov_config=dict(self._applied_config),
            )
            load_s = time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"OpenVINO conversion/load failed for {model_id}: "
                f"{type(exc).__name__}: {exc}"[:400]
            ) from exc

        self._ir_element_type = _weight_element_type(self._model)
        self._read_back_execution()
        return LoadOutcome(
            download_s=download_s,
            prepare_s=self._convert_s or None,
            load_s=load_s,
            total_s=download_s + self._convert_s + load_s,
            notes={
                "converted_this_run": not reuse,
                "artifact_bytes": _tree_size(artifact),
                "ir_element_type": self._ir_element_type,
                "openvino_device": self.device,
                "openvino_available_devices": list(self._available_devices),
                "openvino_execution_devices": list(self._execution_devices),
                "openvino_runtime_config": dict(self._applied_config),
                "execution_precision": self._execution_precision,
            },
        )

    def _convert(self, model_class: Any, model_id: str, artifact: Path,
                 precision: str) -> None:
        """Export to IR at the run's precision and write it out.

        The export is not compiled: the graph produced here is serialized and then
        read back, so compiling it would be paying for a device compilation of a
        model that is about to be replaced by the one on disk.
        """
        import openvino as ov

        element = _STORAGE_ELEMENT.get(precision)
        if element is None:
            raise UnsupportedConfiguration(
                f"no OpenVINO IR element type is defined for precision {precision!r}; "
                f"known: {', '.join(sorted(_STORAGE_ELEMENT))}"
            )
        exported = base.call_with_supported_kwargs(
            model_class.from_pretrained, model_id, export=True, compile=False,
        )
        exported.save_pretrained(artifact)
        # Written again, deliberately: save_pretrained's own compression choice is a
        # property of the installed optimum-intel, and the width the weights are
        # stored at is the run's decision, not the library's.
        ov.save_model(
            exported.model, str(artifact / "openvino_model.xml"),
            compress_to_fp16=(element == "f16"),
        )

    def _read_back_execution(self) -> None:
        """Ask the compiled model which device and precision it actually got.

        Nothing here falls back to an assumption. When a property is unavailable the
        field stays empty and ``describe`` says the precision came from the request,
        so a reader can tell a measured fact from a plan.
        """
        compiled = _compiled_model(self._model)
        if compiled is None:
            return
        devices = _property(compiled, "EXECUTION_DEVICES")
        if devices:
            self._execution_devices = (
                [str(devices)] if isinstance(devices, str)
                else [str(name) for name in devices]
            )
        reported = _property(compiled, "INFERENCE_PRECISION_HINT")
        if reported is not None:
            self._execution_precision = str(reported)
            mapped = _precision_name(self._execution_precision)
            if mapped:
                self._precision = mapped

    def tokenizer(self) -> Any:
        return self._tokenizer

    # ── Generate ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        batch_size: int = 1,
    ) -> GenerationOutcome:
        set_seed(seed)
        encoded = self._tokenizer([prompt] * batch_size, return_tensors="pt")
        prompt_len = int(encoded["input_ids"].shape[1])
        sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,  # fixed work per iteration
            "do_sample": sample,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if sample:
            kwargs.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                kwargs["top_k"] = top_k
        output = self._model.generate(**encoded, **kwargs)
        generated = output[0, prompt_len:].tolist()
        return GenerationOutcome(
            text=self._tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=[int(value) for value in generated],
            prompt_tokens=prompt_len,
            completion_tokens=len(generated),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": int(output.shape[0]),
                "engine": "openvino",
                "openvino_device": self.device,
                "openvino_execution_devices": list(self._execution_devices),
            },
        )

    def first_token_latency(self, prompt: str, *, max_new_tokens: int, seed: int) -> float:
        """Time to the first token through the same streamer the reference engine uses.

        ``OVModelForCausalLM.generate`` is Transformers' own generate, so this row is
        under no obligation to fall back to timing a one-token call: it can be measured
        by the shared streaming implementation, and is. Reporting a different method
        here would put a stopwatch difference inside a number the report ranks.
        """
        set_seed(seed)
        return stream_first_token_latency(
            self._model, self._tokenizer,
            self._tokenizer([prompt], return_tensors="pt"),
            max_new_tokens=max_new_tokens,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        super().unload()


def _compiled_model(model: Any) -> Any:
    """The compiled model behind an optimum-intel wrapper, or None.

    Several attribute names have carried it across optimum-intel versions, and an
    infer request can hand it back as well. All of them are tried because the
    alternative is reporting the device as unknown on a working installation.
    """
    for attribute in ("compiled_model", "_compiled_model"):
        candidate = getattr(model, attribute, None)
        if candidate is not None and hasattr(candidate, "get_property"):
            return candidate
    request = getattr(model, "request", None)
    if request is not None:
        if hasattr(request, "get_compiled_model"):
            try:
                return request.get_compiled_model()
            except Exception:  # noqa: BLE001
                return None
        if hasattr(request, "get_property"):
            return request
    return None


def _property(compiled: Any, name: str) -> Any:
    try:
        return compiled.get_property(name)
    except Exception:  # noqa: BLE001 - an unsupported property is not an error here
        return None


def _weight_element_type(model: Any) -> str | None:
    """The element type of the largest weight constant in the graph.

    Not the first graph *parameter*: those are the inputs, and the first of them is
    ``input_ids``, an int64 tensor. Reading it reports the IR as integer-typed for
    every model, which makes the stored width unreadable and silences the label that
    exists to flag a representation difference. The largest floating-point constant
    is a weight matrix in any decoder graph, and never a scalar scale factor.
    """
    try:
        graph = getattr(model, "model", None)
        ops = getattr(graph, "get_ordered_ops", None)
        if ops is None:
            return None
        best_size, best_type = -1, None
        for node in ops():
            if node.get_type_name() != "Constant":
                continue
            element = str(node.get_element_type())
            if _element_bits(element) is None or "int" in element.lower():
                continue
            size = 1
            for dimension in node.get_output_shape(0):
                size *= int(dimension)
            if size > best_size:
                best_size, best_type = size, element
        return best_type
    except Exception:  # noqa: BLE001 - reported as unknown rather than guessed
        return None


#: Element-type names, longest first, so ``float32`` is not read as an ``f32``
#: prefix match and ``bf16`` is not read as ``f16``.
_ELEMENT_WIDTHS: tuple[tuple[str, int], ...] = (
    ("bfloat16", 16), ("float32", 32), ("float16", 16), ("bf16", 16),
    ("f32", 32), ("f16", 16), ("i8", 8), ("u8", 8), ("i4", 4), ("u4", 4),
)


def _element_bits(element_type: str | None) -> int | None:
    """Width, in bits, of the element type OpenVINO stored the weights as.

    Read from the IR rather than assumed from the requested precision, because the
    conversion decides this and the comparability label depends on it.
    """
    if not element_type:
        return None
    text = str(element_type).lower()
    for name, bits in _ELEMENT_WIDTHS:
        if name in text:
            return bits
    return None


def _precision_name(reported: str | None) -> str | None:
    """The suite's precision name for an OpenVINO element type, or None.

    OpenVINO renders a type as ``<Type: 'float32'>``, so this matches on substrings
    in a fixed longest-first order rather than parsing the wrapper text.
    """
    if not reported:
        return None
    text = str(reported).lower()
    for name in ("bfloat16", "float32", "float16", "bf16", "f32", "f16"):
        if name in text:
            return _PRECISION_FROM_ELEMENT[name]
    return None


def _tree_size(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    ok, reason = base.module_importable("optimum.intel")
    if not ok:
        return base.not_installed(
            f"openvino is installed but optimum.intel is not ({reason}); "
            "install optimum[openvino]"
        )
    if precision not in _STORAGE_ELEMENT:
        return base.not_supported(
            f"OpenVINO IR has no defined weight element type for {precision}; "
            f"the suite converts at {', '.join(sorted(_STORAGE_ELEMENT))}"
        )
    return base.available(base.package_version("openvino"))


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(
        device=getattr(options, "openvino_device", None) or "auto",
        cache_dir=getattr(options, "openvino_cache_dir", None),
        # The same budget the worker pinned for torch. OpenVINO schedules on its own
        # thread pool, so without this it silently gets every core on the machine
        # while every other engine is held to the pinned count.
        threads=getattr(options, "threads", None),
    )

"""ONNX Runtime: an exported graph executed by a cross-platform runtime.

The checkpoint is exported to ONNX once - a real build phase that leaves a real
directory on disk - and then executed by ONNX Runtime's own graph optimizer and
kernels. Like Aether, and unlike torch.compile, what the build leaves behind is a
portable artifact: the same folder loads in another process, and on another
machine with the same runtime.

The honest caveat, recorded and printed rather than buried: an ONNX export of a
16-bit checkpoint is float32 unless it is explicitly converted, so this row is
normally a representation difference and is labelled as one.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

from benchmark.backends import (
    GenerationOutcome,
    LoadOutcome,
    UnsupportedConfiguration,
    set_seed,
)
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="onnxruntime",
    display="ONNX Runtime",
    taxonomy=(
        base.RUNTIME, base.EXECUTION_ENGINE, base.GRAPH_COMPILER, base.KERNEL_OPTIMIZER,
    ),
    summary=(
        "The model exported to an ONNX graph ahead of time, then executed by ONNX "
        "Runtime with its own graph-level optimizations and kernel library. The "
        "export is a separate build step that produces a portable directory."
    ),
    package="optimum",
    requires=("torch", "optimum", "onnxruntime"),
    has_build_phase=True,
    artifact_persistence=base.ARTIFACT_PORTABLE,
    alters_representation=True,
    ttft_method="single_token_call",
    notes=(
        "Exported through optimum.onnxruntime. The exported graph carries float32 "
        "weights unless a conversion pass is applied, so unless the benchmark "
        "precision is fp32 this row is a representation difference, not a "
        "same-weights comparison, and every derived percentage against it is "
        "labelled REPRESENTATION_DIFFERENCE.",
        "The execution provider actually selected is recorded per run. CUDA "
        "execution requires the onnxruntime-gpu build; the CPU build reports "
        "CPUExecutionProvider even on a GPU host.",
    ),
)


class Engine(base.BackendAdapterMixin):
    """Export once to ONNX, cache the export, then generate through ORT."""

    spec = SPEC
    name = SPEC.key

    def __init__(self, device: str = "cpu", cache_dir: str | None = None, **_: Any) -> None:
        self.device = device
        self.cache_dir = Path(cache_dir or "benchmark_results/artifacts/onnx")
        self._model: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._export_s: float = 0.0
        self._export_reused: bool | None = None
        self._artifact: Path | None = None
        self._providers: list[str] = []

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine_key": SPEC.key,
            "taxonomy": list(SPEC.taxonomy),
            "device": self.device,
            "precision": self._precision,
            "execution_providers": self._providers,
            # Explicit flag so the report can colour-code CPU-fallback runs without
            # having to parse the providers list.
            "gpu_provider_active": "CUDAExecutionProvider" in self._providers,
            "artifact": str(self._artifact) if self._artifact else None,
            "artifact_reused": self._export_reused,
            "export_s": self._export_s,
            "generation": "ORTModelForCausalLM.generate (io-binding managed by optimum)",
            "representation": "ONNX graph exported from the checkpoint; float32 weights",
            "weight_storage_bits": 32,
            "weight_storage_format": "fp32",
            "quantized": False,
            "ttft_method": SPEC.ttft_method,
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
        except ImportError as exc:
            # The export stack is installed but does not agree with the transformers
            # version this run is measuring everything else on. That is an
            # environment incompatibility, not a defect in the engine, so it is
            # reported as unsupported with the conflicts spelled out rather than as a
            # bare traceback.
            conflicts = [
                problem
                for distribution in _EXPORT_DISTRIBUTIONS
                for problem in base.requirement_conflicts(distribution)
            ]
            raise UnsupportedConfiguration(
                f"optimum.onnxruntime could not be imported: {exc}. "
                + ("Declared conflicts: " + "; ".join(conflicts) if conflicts
                   else "No declared version conflict was found, so this is likely an "
                        "incompatibility the packages do not declare.")
            ) from exc
        from transformers import AutoTokenizer

        self._precision = precision
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"

        download_start = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        download_s = time.perf_counter() - download_start

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.cache_dir / model_id.replace("/", "--")
        self._artifact = artifact
        reuse = (artifact / "model.onnx").exists() or any(artifact.glob("*.onnx"))
        self._export_reused = reuse

        export_start = time.perf_counter()
        try:
            if reuse:
                self._model = _load_ort_model(ORTModelForCausalLM, artifact,
                                              provider, export=False)
            else:
                self._model = _load_ort_model(ORTModelForCausalLM, model_id,
                                              provider, export=True)
                self._model.save_pretrained(artifact)
            self._export_s = 0.0 if reuse else time.perf_counter() - export_start
        except UnsupportedConfiguration:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedConfiguration(
                f"ONNX export/load failed for {model_id}: "
                f"{type(exc).__name__}: {exc}"[:400]
            ) from exc
        load_s = time.perf_counter() - export_start - self._export_s
        # optimum renamed the internal ORT session: try both attribute names so the
        # provider readback works with both optimum 1.x (.model) and 2.x
        # (.model_session or .sessions).
        self._providers = _read_providers(self._model)

        # Post-load provider verification: confirm the session is actually using the
        # requested provider. ORT can silently fall back to CPUExecutionProvider when
        # CUDA initialisation fails (e.g. driver/library version mismatch). Catching
        # this here means the LoadOutcome carries an explicit warning rather than the
        # mismatch going unnoticed until the throughput numbers look wrong.
        provider_mismatch: str | None = None
        if provider not in self._providers:
            provider_mismatch = (
                f"requested {provider!r} but the session is running with "
                f"{self._providers}. Likely cause: the onnxruntime-gpu build is not "
                "installed, CUDA initialisation failed, or the session fell back to "
                "CPUExecutionProvider silently. Check 'ort.get_available_providers()' "
                "and verify the onnxruntime-gpu wheel is installed (not the CPU build)."
            )
            _log.warning("ORT provider mismatch: %s", provider_mismatch)

        load_notes: dict[str, Any] = {
            "exported_this_run": not reuse,
            "artifact_bytes": _tree_size(artifact),
            "execution_providers": self._providers,
            "requested_provider": provider,
        }
        if provider_mismatch is not None:
            load_notes["provider_mismatch"] = provider_mismatch

        return LoadOutcome(
            download_s=download_s,
            prepare_s=self._export_s,
            load_s=max(load_s, 0.0),
            total_s=download_s + self._export_s + max(load_s, 0.0),
            notes=load_notes,
        )

    def tokenizer(self) -> Any:
        return self._tokenizer

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
        # Move tokenizer outputs to the execution device so ORT's IO-binding path
        # receives tensors on the correct device. Without this, optimum passes CPU
        # tensors to the session even when use_io_binding=True, causing a silent
        # fallback to CPUExecutionProvider regardless of the configured provider.
        #
        # Important: we iterate all items (not just Tensors) so that non-Tensor
        # values (e.g. token-type ids stored as Python ints) are preserved in the
        # mapping. Only torch.Tensor instances are moved; everything else is kept
        # as-is. We reassign ``encoded`` to a plain dict because BatchEncoding's
        # constructor signature varies across transformers versions and is not part
        # of the guaranteed public API.
        if self.device == "cuda":
            import torch
            encoded = {
                key: (value.to("cuda") if isinstance(value, torch.Tensor) else value)
                for key, value in encoded.items()
            }
        sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "do_sample": sample,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if sample:
            kwargs.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                kwargs["top_k"] = top_k
        output = self._model.generate(**encoded, **kwargs)
        # output may be on GPU; move to CPU for decoding.
        if hasattr(output, "cpu"):
            output = output.cpu()
        generated = output[0, prompt_len:].tolist()
        return GenerationOutcome(
            text=self._tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=[int(value) for value in generated],
            prompt_tokens=prompt_len,
            completion_tokens=len(generated),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": int(output.shape[0]),
                "engine": "onnxruntime",
                "execution_providers": self._providers,
            },
        )

    def prefill(self, prompt: str) -> Any:
        """One forward pass through the ONNX graph, logits at every position."""
        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt")
        # Same device-placement fix as in generate(): the session's IO-binding
        # path requires tensors already on the execution device.
        if self.device == "cuda":
            # Same device-placement logic as generate(): move Tensor values and
            # preserve all other values so the full dict reaches the session.
            encoded = {
                key: (value.to("cuda") if isinstance(value, torch.Tensor) else value)
                for key, value in encoded.items()
            }
        with torch.no_grad():
            output = self._model(**encoded)
        return output.logits[0, -1].detach().float().cpu()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        super().unload()


def _tree_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


#: Distributions whose declared requirements decide whether the ONNX export path
#: works at all. optimum's exporter lives in optimum-onnx on current releases and in
#: optimum itself on older ones, and both pin a transformers range: when the host
#: framework has moved past it, importing the exporter fails on a private symbol
#: (``get_parameter_dtype`` is the usual one). Checking the declared range is how that
#: becomes a stated incompatibility instead of a mid-run crash.
_EXPORT_DISTRIBUTIONS = ("optimum-onnx", "optimum")


def _optimum_ort_importable() -> tuple[bool, str]:
    """Check whether ``optimum.onnxruntime`` can be imported in a fresh subprocess.

    Using a subprocess rather than ``importlib.util.find_spec`` avoids two
    failure modes that are common on Kaggle and similar cloud notebook hosts:

    1. **Stale in-process .so**: if the user replaced the CPU ORT build with
       the GPU build inside the same session (pip uninstall + pip install
       without restart), ``find_spec('optimum.onnxruntime')`` imports the
       ``optimum`` parent package which may trigger the stale C extension,
       causing a spurious failure. A fresh subprocess is unaffected.

    2. **Missing [onnxruntime] extra**: ``optimum.onnxruntime`` may not be
       discoverable if ``optimum`` was installed without the ``[onnxruntime]``
       extra (which installs the integration subpackage). A real import attempt
       surfaces this clearly instead of a confusing find_spec None.
    """
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import optimum.onnxruntime"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, ""
        stderr = result.stderr.strip()
        last_line = stderr.split("\n")[-1] if stderr else "import failed"
        return False, last_line
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _cuda_provider_available() -> bool:
    """True when the installed onnxruntime build exposes CUDAExecutionProvider.

    Runs in a **subprocess** rather than importing onnxruntime in-process for
    two reasons:

    1. ``onnxruntime-gpu`` 1.18+ registers its metadata as ``onnxruntime``, so
       ``importlib.metadata.version('onnxruntime-gpu')`` returns ``None`` even
       on a fully-functional CUDA build. Checking the provider list is
       definitive regardless of how the package was named.

    2. If the user replaced the CPU build with the GPU build inside the same
       Python session (e.g. ran ``pip uninstall onnxruntime && pip install
       onnxruntime-gpu`` without restarting), the old CPU-build ``.so`` is
       still resident in the orchestrator process's address space. A subprocess
       always starts clean and sees whatever is actually on disk, so the check
       is accurate even without a session restart.
    """
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import onnxruntime as ort; "
             "print('CUDAExecutionProvider' in ort.get_available_providers())"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and "True" in result.stdout
    except Exception:  # noqa: BLE001
        return False


def _load_ort_model(cls: Any, model_id_or_path: Any, provider: str,
                    export: bool = False) -> Any:
    """Load or export an ORTModelForCausalLM with the correct execution provider.

    The optimum API uses ``provider=`` (singular string) in ``from_pretrained``.
    For CUDA, two extra flags are required:

    * ``use_io_binding=True`` — without it, ORT receives CPU tensors and falls
      back to CPU execution silently even when CUDAExecutionProvider is set.
    * ``provider_options=[{"device_id": 0}]`` — pins the session to device 0,
      which is always the first *visible* device after the worker restricts
      ``CUDA_VISIBLE_DEVICES``. Without this pin, ORT may open a context on a
      different device than the one the rest of the benchmark is using, making
      GPU memory attribution incorrect.

    After loading, ``.to(device)`` is called as a final backstop: it calls
    ``session.set_providers([provider])`` internally, so even if a cached artifact
    was originally loaded with the wrong provider, this corrects it.
    """
    from benchmark.backends import UnsupportedConfiguration

    kwargs: dict[str, Any] = {}
    if export:
        kwargs["export"] = True

    # use_io_binding is required for CUDAExecutionProvider to actually use the GPU.
    # Without it, optimum passes CPU tensors directly to the ORT session and the
    # session silently falls back to CPU even when CUDAExecutionProvider is set.
    #
    # provider_options pins the session to device 0 — the first device visible
    # after the worker sets CUDA_VISIBLE_DEVICES. This ensures benchmark GPU memory
    # attribution is correct even when the host has multiple GPUs.
    use_cuda = provider == "CUDAExecutionProvider"
    if use_cuda:
        kwargs["use_io_binding"] = True
        kwargs["provider_options"] = [{"device_id": 0}]

    try:
        model = cls.from_pretrained(
            str(model_id_or_path),
            provider=provider,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedConfiguration(
            f"ORTModelForCausalLM.from_pretrained failed with provider={provider!r}: "
            f"{type(exc).__name__}: {exc}"[:400]
        ) from exc

    # Call .to() as a final backstop. This calls session.set_providers([provider])
    # internally, correcting the provider even when a cached artifact was loaded
    # with a different provider from a previous run.
    if use_cuda:
        try:
            model = model.to("cuda")
        except Exception as _move_exc:  # noqa: BLE001
            # .to() is not available on all optimum versions; the provider= kwarg
            # above is the primary mechanism; this is belt-and-suspenders.
            # Log rather than silently swallow: a failure here is diagnostic when
            # the session later reports an unexpected CPUExecutionProvider.
            _log.warning(
                "ORTModelForCausalLM.to('cuda') failed (non-fatal, provider= kwarg "
                "is the primary mechanism): %s: %s",
                type(_move_exc).__name__, _move_exc,
            )

    return model


def _read_providers(model: Any) -> list[str]:
    """Read the active execution providers from an ORT model wrapper.

    In optimum, the ORT InferenceSession is stored as ``model.model`` (confirmed
    from optimum source: ``self.model.set_providers(...)``).  ``self.providers``
    is a list attribute that mirrors ``model.get_providers()``.

    Try the instance attribute first (cheapest), then fall back to the session.
    """
    # Fastest: the instance-level providers list that optimum keeps in sync
    providers_attr = getattr(model, "providers", None)
    if isinstance(providers_attr, list) and providers_attr:
        return list(providers_attr)
    # Session attribute confirmed from optimum source (self.model = ort.InferenceSession)
    for attr in ("model", "model_session"):
        session = getattr(model, attr, None)
        if session is not None and callable(getattr(session, "get_providers", None)):
            return list(session.get_providers())
    # optimum may expose a dict of sessions (decoder / decoder_with_past split)
    sessions = getattr(model, "sessions", None)
    if isinstance(sessions, dict):
        for session in sessions.values():
            if callable(getattr(session, "get_providers", None)):
                return list(session.get_providers())
    return []


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    generic = base.generic_probe(SPEC, hardware)
    if not generic.usable:
        return generic
    # Use a subprocess rather than find_spec / in-process import to check
    # optimum.onnxruntime. In-process checks fail when:
    #   a) onnxruntime was replaced in the same session without restart (stale .so)
    #   b) optimum was installed without the [onnxruntime] extra (missing subpackage)
    # A subprocess always gets a fresh import context and sees the current disk state.
    ok, reason = _optimum_ort_importable()
    if not ok:
        return base.not_installed(
            "optimum.onnxruntime is not importable. "
            f"Reason: {reason}. "
            "Fix: pip install 'optimum[onnxruntime]>=1.20.0' (the [onnxruntime] "
            "extra is required; plain 'optimum' does not include the integration "
            "subpackage). On GPU hosts install onnxruntime-gpu FIRST so that pip "
            "sees onnxruntime as satisfied by the GPU build: "
            "pip uninstall -y onnxruntime && "
            "pip install 'onnxruntime-gpu>=1.18.0' && "
            "pip install 'optimum[onnxruntime]>=1.20.0'"
        )
    conflicts = [
        problem
        for distribution in _EXPORT_DISTRIBUTIONS
        for problem in base.requirement_conflicts(distribution)
        if "transformers" in problem or "onnx" in problem
    ]
    if conflicts:
        return base.not_supported(
            "the installed ONNX export stack does not support this environment: "
            + "; ".join(conflicts)
            + ". Install a compatible pair before the run - the whole field has to "
            "share one transformers version, so pin transformers into the range this "
            "exporter accepts rather than upgrading it afterwards."
        )
    version = base.package_version("optimum")
    # onnxruntime-gpu 1.18+ registers its metadata as "onnxruntime", so
    # package_version("onnxruntime-gpu") returns None even when CUDA providers
    # are fully available. Check the provider list directly instead.
    if hardware.nvidia and not _cuda_provider_available():
        # CUDAExecutionProvider is absent — the CPU build of onnxruntime is
        # installed instead of the GPU build. The most common cause on Kaggle and
        # cloud notebook environments: the pre-installed CPU build satisfies the
        # "onnxruntime" dependency, so pip never installs onnxruntime-gpu.
        #
        # AETHER_ORT_REQUIRE_GPU=1 opts the operator into a hard fail (NOT_INSTALLED)
        # that excludes this engine entirely, preventing CPU results from appearing
        # under the onnxruntime label in a GPU benchmark. Without the variable the
        # engine is allowed to run on CPU and its active provider is recorded in
        # every result row — the data is honest, but the comparison is unfair.
        _gpu_fix_message = (
            "CUDAExecutionProvider is NOT available on this GPU host. "
            "The CPU build of onnxruntime is installed; it does not expose CUDA. "
            "Fix (Kaggle / any Linux GPU host, NO session restart required): "
            "(1) pip uninstall -y onnxruntime  "
            "(2) pip install 'onnxruntime-gpu>=1.18.0'  "
            "(3) pip install 'optimum[onnxruntime]>=1.20.0'  "
            "No restart is needed because the probe and measurement workers run in "
            "subprocesses that always see the current disk state. "
            "Verify: python -c \"import onnxruntime as ort; "
            "print(ort.get_available_providers())\" "
            "— must print CUDAExecutionProvider. "
            "Set AETHER_ORT_REQUIRE_GPU=1 to treat this as NOT_INSTALLED "
            "and exclude the engine from a GPU benchmark run."
        )
        if os.environ.get("AETHER_ORT_REQUIRE_GPU", "").strip() in ("1", "true", "yes"):
            return base.not_installed(_gpu_fix_message)
        avail = base.available(version, _gpu_fix_message)
        avail.detail["gpu_provider_available"] = False
        avail.detail["cuda_provider_check"] = "ort.get_available_providers() did not include CUDAExecutionProvider"
        return avail
    return base.available(version)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    # Use the provider-list check rather than the distribution name: onnxruntime-gpu
    # 1.18+ registers its metadata as "onnxruntime", so the old name-based check
    # always returned None and forced every engine onto CPU even on GPU hosts.
    cuda_available = hardware.nvidia and _cuda_provider_available()
    return Engine(
        device="cuda" if cuda_available else "cpu",
        cache_dir=getattr(options, "onnx_cache_dir", None),
    )



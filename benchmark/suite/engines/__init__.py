"""The field of engines, and how the suite talks to any one of them.

Adding an engine means adding a module here with three names: ``SPEC``, ``probe``
and ``build``. Nothing else in the suite needs to change - the runner already
knows how to measure anything that satisfies
:class:`benchmark.backends.Backend`, and the report already knows how to print
anything that carries an :class:`~benchmark.suite.engines.base.EngineSpec`.

Order matters only for presentation. The reference baseline comes first, Aether
last, so a table reads from "the stack everyone starts with" to "the stack under
test" - and so no ordering can be mistaken for a ranking.

The field is four engines, deliberately: the reference framework, the same
framework's kernels driven by a hand-written loop, one competing ahead-of-time
compiler, and Aether. Every one of them runs the published checkpoint at the
run's precision with no weight format of its own, which is what makes the
comparison a comparison of execution rather than of representations. An engine
that can only be measured on a quantization it brings with it, or that could not
be measured on the same cells as the rest of the field, is not in the set: a row
that ran a different experiment is worse than an absent row, because it still
takes a rank.
"""

from __future__ import annotations

from typing import Any

from benchmark.suite import status as status_mod
from benchmark.suite.engines import (
    aether_engine,
    base,
    hf_transformers,
    openvino_engine,
    pytorch_native,
)

#: Every engine module, in report order.
MODULES: tuple[Any, ...] = (
    hf_transformers,
    pytorch_native,
    openvino_engine,
    aether_engine,
)

#: The engine every correctness comparison is made against. Chosen because it is
#: the reference implementation of the published checkpoints, not because it is
#: fastest or slowest.
REFERENCE = hf_transformers.SPEC.key

#: The engine under test. Named explicitly so the analysis code never has to
#: infer which column the report is about.
SUBJECT = aether_engine.SPEC.key

BY_KEY: dict[str, Any] = {module.SPEC.key: module for module in MODULES}
KEYS: tuple[str, ...] = tuple(BY_KEY)


def module_for(key: str) -> Any:
    """The engine module registered under ``key``."""
    if key not in BY_KEY:
        raise KeyError(f"unknown engine {key!r}; known: {', '.join(KEYS)}")
    return BY_KEY[key]


def spec_for(key: str) -> base.EngineSpec:
    return module_for(key).SPEC


def specs() -> dict[str, base.EngineSpec]:
    return {module.SPEC.key: module.SPEC for module in MODULES}


def probe(key: str, hardware: Any, model_id: str, precision: str,
          options: Any) -> base.Availability:
    """Ask one engine whether it can run this model here.

    A probe that itself raises is reported as a failure of the probe, not of the
    engine: a broken installation should say so rather than crashing the field
    survey that every other engine's row depends on.
    """
    module = module_for(key)
    try:
        return module.probe(hardware, model_id, precision, options)
    except BaseException as exc:  # noqa: BLE001
        return base.Availability(
            status=status_mod.FAILED,
            reason=f"availability probe raised: {type(exc).__name__}: {exc}"[:300],
        )


def probe_all(hardware: Any, model_id: str, precision: str, options: Any,
              keys: list[str] | None = None) -> dict[str, base.Availability]:
    """Survey the whole field for one model, in report order."""
    return {
        key: probe(key, hardware, model_id, precision, options)
        for key in (keys or KEYS)
    }


def build(key: str, hardware: Any, model_id: str, precision: str, options: Any) -> Any:
    """Construct the adapter for ``key``. Raises if the engine cannot be built."""
    return module_for(key).build(hardware, model_id, precision, options)

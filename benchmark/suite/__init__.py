"""The multi-engine inference benchmark suite.

The original :mod:`benchmark` package compares two stacks — Aether Runtime and
Hugging Face Transformers — inside one process, which is the right shape for a
controlled A/B. This package generalizes that comparison to a field of inference
stacks (frameworks, runtimes, graph compilers, serving engines) without changing
how anything is measured: every engine adapter here implements the same
:class:`benchmark.backends.Backend` protocol, and every measurement is taken by
the same :mod:`benchmark.runner` primitives.

Two structural differences from the two-backend harness:

* **one worker process per (engine, model)**. A serving engine that claims all
  device memory, a native library that segfaults, or a build that cannot import
  cannot then take the rest of the suite with it — and a process boundary makes
  peak host memory attributable to exactly one engine rather than to whichever
  engines happened to be resident.
* **status is data**. A configuration that was not measured carries the reason it
  was not: ``NOT_INSTALLED``, ``NOT_SUPPORTED``, ``NOT_APPLICABLE``, ``FAILED``,
  ``OOM`` or ``SKIPPED``. Nothing absent is ever rendered as zero.
"""

from __future__ import annotations

#: Bumped when the measurement methodology changes in a way that makes results
#: from two versions non-comparable. Recorded in every result file.
SUITE_VERSION = "2.0.0"

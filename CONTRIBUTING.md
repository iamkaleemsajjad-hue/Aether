# Contributing to Aether Runtime

Thank you for your interest in contributing to Aether Runtime! This document covers everything you need to know to get started, from setting up your development environment to submitting a pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Project Structure](#project-structure)
5. [Coding Standards](#coding-standards)
6. [Type Hints and Static Analysis](#type-hints-and-static-analysis)
7. [Testing](#testing)
8. [Writing Documentation](#writing-documentation)
9. [Pull Request Process](#pull-request-process)
10. [Commit Message Guidelines](#commit-message-guidelines)
11. [Issue Reporting](#issue-reporting)
12. [Security](#security)
13. [Areas Where Help is Needed](#areas-where-help-is-needed)
14. [Community](#community)
15. [License](#license)

---

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please be respectful, constructive, and inclusive in all interactions. Harassment, discrimination, and abusive behavior are not tolerated.

If you experience or witness unacceptable behavior, please report it to the maintainers at `conduct@aether.dev`.

---

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Create a new branch for your work.
4. Set up the development environment (see below).
5. Make your changes, add tests, and update documentation.
6. Run the full test suite and linters.
7. Push your branch and open a pull request.

---

## Development Environment

### Prerequisites

- Python 3.10 or newer
- Git
- A C++ compiler (for some optional dependencies)
- (Optional) CUDA 12.x for NVIDIA backend testing
- (Optional) An Apple Silicon Mac for MLX testing

### Clone the repository

```bash
git clone https://github.com/aether-dev/aether-runtime.git
cd aether-runtime
```

### Install in editable mode

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The `dev` extra installs all lint, test, docs, and benchmark dependencies. If you only want to work on a specific backend, install the relevant extras:

```bash
pip install -e ".[dev,vllm]"
pip install -e ".[dev,llamacpp]"
pip install -e ".[dev,mlx]"
```

### Pre-commit hooks (optional but recommended)

We use `pre-commit` for automated checks before commits. Install it with:

```bash
pip install pre-commit
pre-commit install
```

If you do not use `pre-commit`, run the same checks manually:

```bash
make lint
make typecheck
make test
```

---

## Project Structure

```
aether-runtime/
├── src/aether/              # Main source code
│   ├── core/                # AEG format, AEG-IR, graph, types
│   ├── compiler/            # Compiler pipeline, passes, calibration
│   ├── backends/            # Backend plugins (vLLM, llama.cpp, etc.)
│   ├── runtime/             # Runtime, scheduler, KV cache, executor
│   ├── server/              # REST API and OpenAI-compatible routes
│   ├── hub/                 # Aether Hub client and local cache
│   ├── targets/             # Target profiles and kernel templates
│   ├── parallelism/         # Tensor/pipeline/expert/context parallelism
│   ├── moe/                 # MoE routing and expert management
│   ├── quantization/        # Quantization formats and precision assignment
│   ├── kernels/             # Kernel dispatch and base classes
│   └── utils/               # Logging, profiling, telemetry, utilities
├── tests/                   # Unit and integration tests
├── docs/                    # Sphinx + Markdown documentation
├── examples/                # Usage examples
├── benchmarks/              # Benchmark suite and comparisons
├── scripts/                 # Helper scripts
├── research/                # Research foundation summaries
├── .github/workflows/       # CI/CD pipelines
└── pyproject.toml           # Build, dependencies, and tool config
```

---

## Coding Standards

### Python style

- Follow PEP 8 with a line length of 100 characters.
- Use `ruff` for formatting and linting.
- Use Google-style docstrings.
- Keep functions focused and short when possible, but do not sacrifice clarity for artificial brevity.
- Prefer composition over inheritance for runtime components.
- Avoid heavy framework dependencies in the core package.

### Naming conventions

- `snake_case` for functions, variables, and file names.
- `PascalCase` for classes.
- `UPPER_CASE` for module-level constants.
- Private helpers start with `_`.

### Imports

Order imports as follows, separated by blank lines:

1. Standard library imports
2. Third-party imports
3. First-party `aether` imports

Use absolute imports within `src/aether` and relative imports only when it significantly improves readability.

### Error handling

- Use Aether-specific exceptions from `aether.exceptions`.
- Do not silently swallow exceptions; log or propagate them with context.
- Validate inputs at public API boundaries.
- Avoid `except Exception:` unless re-raising or logging at a high level.

### Logging

Use `structlog` via the logging utilities in `aether.utils.logging`. Avoid `print` statements in library code; use `rich` only for CLI output.

---

## Type Hints and Static Analysis

Aether uses `mypy` in strict mode. All public functions and classes must have type annotations. Type-only imports should be guarded with `if TYPE_CHECKING:`.

### Example

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aether.core.graph import AEGGraph

def optimize_graph(graph: AEGGraph) -> AEGGraph:
    ...
```

Run `mypy`:

```bash
make typecheck
```

---

## Testing

### Test organization

- `tests/unit/` — fast, deterministic tests for individual modules.
- `tests/integration/` — end-to-end tests that may download small models or require backends.
- `tests/fixtures/` — synthetic models and test helpers.
- `tests/data/` — small static test data files.

### Running tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# Integration tests
pytest tests/integration -m integration

# Skip slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=aether --cov-report=html
```

### Writing tests

- Use `pytest` fixtures and parametrization.
- Place shared fixtures in `tests/conftest.py`.
- Mark slow tests with `@pytest.mark.slow`.
- Mark tests requiring network with `@pytest.mark.network`.
- Mark tests requiring a GPU with `@pytest.mark.gpu`.
- Mark tests requiring a specific backend with `@pytest.mark.requires_backend("vllm")`.

### Integration tests with real models

Some integration tests download small HuggingFace models (e.g., `Qwen/Qwen3-0.6B`). These are marked with `@pytest.mark.integration` and `@pytest.mark.network`. They are skipped in the default `pytest` configuration if the network is unavailable. CI runs them on a scheduled basis.

---

## Writing Documentation

Documentation lives in `docs/` and is built with Sphinx + MyST Parser. We write most docs in Markdown.

### Build documentation

```bash
make docs
```

The built docs are placed in `docs/_build/html/`.

### Documentation style

- Use clear, concise explanations.
- Include code examples for every feature.
- Cross-reference related sections.
- Update `docs/api-reference.md` when adding public APIs.
- Keep the AEG format specification (`docs/aeg-format.md`) synchronized with the implementation.

### README and examples

When adding a major feature, update the README with a short example and add a dedicated example script under `examples/` if appropriate.

---

## Pull Request Process

1. **Open an issue first** for large changes or new features so the design can be discussed.
2. **Create a branch** from `main` with a descriptive name: `feature/disaggregated-scheduler`, `fix/kv-cache-eviction`, `docs/aeg-format-update`.
3. **Make focused commits** with clear messages.
4. **Add tests and documentation** for your changes.
5. **Run the full check suite** before pushing:
   ```bash
   make check
   ```
6. **Open a pull request** against `main` and fill out the template.
7. **Address review feedback** promptly and respectfully.
8. **Squash merge** is not required; maintainers will use the merge strategy that keeps history clean.

### PR checklist

- [ ] Tests added or updated.
- [ ] Documentation updated.
- [ ] `make check` passes locally.
- [ ] No new warnings from `mypy` or `ruff`.
- [ ] Commit messages follow the guidelines.
- [ ] PR description explains the change and why it matters.

---

## Commit Message Guidelines

We use conventional commit style for clarity and automated changelog generation.

```
<type>(<scope>): <short summary>

<body>

<footer>
```

### Types

- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation changes
- `style` — formatting, no code change
- `refactor` — code change that neither fixes a bug nor adds a feature
- `perf` — performance improvement
- `test` — adding or correcting tests
- `chore` — build, CI, dependency updates
- `ci` — continuous integration changes

### Scopes

Common scopes: `core`, `compiler`, `runtime`, `server`, `backends`, `quantization`, `moe`, `parallelism`, `hub`, `cli`, `docs`, `tests`.

### Examples

```
feat(compiler): add sensitivity analysis pass

Computes d(perplexity)/d(precision) per layer using a calibration
dataset and stores the resulting sensitivity map in the AEG.

Closes #123
```

```
fix(runtime): restore BF16 weights when memory pressure drops

Previously the precision manager only downgraded weights under
pressure but never restored them. This change restores high-sensitivity
layers when VRAM returns below 70%.
```

---

## Issue Reporting

We use GitHub Issues for bug reports, feature requests, and design discussions.

### Bug report template

- A clear description of the bug.
- Steps to reproduce.
- Expected vs. actual behavior.
- Environment details: OS, Python version, Aether version, GPU/backend used.
- Relevant logs or error messages (with secrets removed).

### Feature request template

- A clear description of the feature.
- The problem it solves.
- How it fits with Aether's architecture and roadmap.
- Proposed API or CLI changes, if any.

---

## Security

If you discover a security vulnerability, please report it privately to `security@aether.dev` rather than opening a public issue. We will respond promptly and coordinate disclosure.

---

## Areas Where Help is Needed

We especially welcome contributions in these areas:

### High priority

- **New backend plugins** — AMD ROCm, Intel OpenVINO, Qualcomm, WebGPU, etc.
- **Compiler passes** — better operator fusion, more advanced sensitivity analysis, improved MoE routing.
- **Runtime optimizations** — smarter scheduling, better KV cache eviction, more speculative decoding strategies.
- **Quantization formats** — support for new bit-packing schemes and per-expert quantization.
- **Documentation** — tutorials, examples, API docs, research summaries.

### Medium priority

- **Benchmarks** — add more models and hardware combinations to the benchmark suite.
- **Hub integrations** — HuggingFace AEG variant detection, registry features.
- **SDK bindings** — Rust, Go, JavaScript/TypeScript bindings.
- **Cross-platform CI** — improve Windows and macOS test coverage.

### Good first issues

- Typo fixes and documentation improvements.
- Adding more unit tests.
- Improving error messages.
- Adding small CLI conveniences.

Look for issues labeled `good first issue` and `help wanted` on GitHub.

---

## Community

- GitHub Discussions: [github.com/aether-dev/aether-runtime/discussions](https://github.com/aether-dev/aether-runtime/discussions)
- Discord: [discord.gg/aether-runtime](https://discord.gg/aether-runtime)
- Twitter/X: [@aether_runtime](https://twitter.com/aether_runtime)
- Newsletter: [aether.dev/newsletter](https://aether.dev/newsletter)

---

## License

By contributing to Aether Runtime, you agree that your contributions will be licensed under the Apache License 2.0. See [LICENSE](LICENSE).

---

## Thank You

Every contribution, whether code, documentation, bug reports, or community support, makes Aether better. We appreciate your time and effort!


## PRD v3.1 Contribution Requirements

New PRD-layer features must include all of the following before review:

- A functional reference implementation or an explicit metadata contract in the AEG artifact.
- Unit tests for deterministic behavior and package serialization.
- Documentation updates in `README.md`, `docs/aeg-format.md`, and the relevant architecture/runtime doc.
- A research note in `docs/research.md` when the feature is paper-derived.
- CI-safe behavior without requiring network, GPU, or paid services unless the test is marked accordingly.


### v3.1 Layer Contributions

When changing agentic, observability, fleet, distillation, CUDA graph, MLA, EAGLE-3, or multimodal code, contributors must:

- Add deterministic unit tests for planner output and AEG artifact serialization.
- Keep generated package contracts backward-compatible with `AEGPackage.load()`.
- Avoid network-dependent tests; use local synthetic traces, architectures, and telemetry snapshots.
- Update `docs/aeg-format.md` when a new artifact path is added to `manifest.artifacts`.


---

## Authoring PRD v4.0 + v5.0 Optimizer Passes (Passes 10–22)

Each new optimizer pass follows the pattern established by passes 10–22.

### Pass File Structure

Create `src/aether/compiler/stage2_optimizer/pass{N}_{name}.py`:

```python
"""
Pass N — Short Title.

Module-level docstring with:
  - Algorithm description referencing specific research papers.
  - AEG artifacts written (paths under .aeg/).
  - Performance targets from benchmarks.

Research basis: cite arXiv IDs or paper titles.
"""

from __future__ import annotations
from aether.utils.logging import get_logger
logger = get_logger(__name__)

class MyNewPass:
    """Single-sentence summary.

    Longer description with key algorithm details.
    """

    PASS_NAME = "my_new_pass"          # Must be unique across all passes.
    PASS_VERSION = "1.0"

    def run(self, graph, arch: dict, config) -> tuple[Any, PassReport]:
        """Required signature — all passes must implement this."""
        if not getattr(config, "enable_my_pass", False):
            return graph, PassReport(pass_name=self.PASS_NAME, status="skipped", ...)

        try:
            # Implementation here.
            ...
            return graph, PassReport(pass_name=self.PASS_NAME, status="ok", ...)
        except Exception as exc:
            return graph, PassReport(pass_name=self.PASS_NAME, status="failed", ...)
```

### Required Elements for Every New Pass

1. **Config flag**: Add `enable_my_pass: bool = False` to `CompilerConfig` (opt-in by default).
2. **Skip path**: Return `status="skipped"` when `enable_my_pass=False`.
3. **AEG artifact**: Write JSON/binary output to `{output_dir}/<category>/<artifact>.json`.
4. **Opcode emission**: Add `aeg.my_opcode` metadata to the graph.
5. **PassReport**: Always return a `PassReport` — never raise exceptions from `run()`.
6. **Registration**: Register in `OptimizerPipeline.__init__()` with the correct ordering.
7. **Tests**: Add tests to `tests/test_passes_v2.py`:
   - Smoke test (runs without exception).
   - Skip-when-disabled test.
   - Core algorithm correctness test (pure-Python, no GPU).
   - AEG artifact written test.

### AEG Artifact Conventions

| Category | Path pattern | Used by |
|----------|-------------|---------|
| Speculation | `.aeg/speculation/` | R1 P-EAGLE |
| Grammar | `.aeg/grammar/` | R3 FSM |
| Graph plans | `.aeg/graph/` | R2, R10 |
| Diffusion | `.aeg/diffusion/` | — |
| Quantization | `.aeg/quantization/` | R9 |
| Adapters | `.aeg/adapters/` | — |
| Metadata | `.aeg/metadata/` | R7 |
| Security | `.aeg/security/` | R8 |
| Training | `.aeg/training/` | R12 |
| TTT | `.aeg/ttt/` | R5 |

All artifact JSON files should contain at minimum:
```json
{
  "pass_name": "my_new_pass",
  "pass_version": "1.0",
  "generated_at": "ISO-8601 timestamp",
  ...
}
```

---

## Authoring PRD v4.0 + v5.0 Runtime Layers (R1–R12 Pattern)

### Runtime Layer File Structure

Create `src/aether/runtime/r{N}_{name}.py`:

```python
"""
R{N} — Short Title.

Module-level docstring covering:
  - What the layer does.
  - Which AEG artifact it loads (if any).
  - Performance targets.
  - Research basis.
"""

from __future__ import annotations
from aether.utils.logging import get_logger
logger = get_logger(__name__)

class MyRuntimeLayer:
    def __init__(self, config_path: str | None = None) -> None:
        self._config: dict = {}
        self._stats = _MyLayerStats()
        if config_path:
            self._load_config(config_path)

    def _load_config(self, path: str) -> None:
        """Load AEG artifact config. Fail gracefully if not found."""
        ...

    # Public API methods go here.

    @property
    def stats(self) -> "_MyLayerStats":
        return self._stats

    def summary(self) -> dict:
        return {... }  # JSON-serializable dict.


class _MyLayerStats:
    """Internal stats (use __slots__ for memory efficiency)."""
    __slots__ = ("field_a", "field_b")

    def __init__(self) -> None:
        self.field_a = 0
        self.field_b = 0.0
```

### Required Elements for Every New Runtime Layer

1. **AEG config loading**: Load from the AEG artifact if path is provided; fail gracefully.
2. **Thread safety**: Use `threading.RLock()` for any shared mutable state.
3. **Stats object**: `_Stats` class with `__slots__` tracking key metrics.
4. **`summary()` method**: Returns a JSON-serializable dict for telemetry.
5. **Export**: Add to `src/aether/runtime/__init__.py` `__all__`.
6. **Tests**: Add to `tests/test_runtime_v2.py`:
   - Config loading test.
   - Core algorithm test (pure-Python inputs, no GPU).
   - Edge case tests (empty inputs, missing config, etc.).
   - `summary()` key presence test.

### Algorithm Standards

All algorithm implementations must:

- Be **self-contained**: no mandatory external dependencies. Optional dependencies
  (e.g., `hnswlib`, `sympy`) are welcome but must degrade gracefully.
- Have **pure-Python correctness tests**: the algorithm must be testable without GPU.
- Reference the **research paper** in module docstring with arXiv ID or full citation.
- Match the **mathematical specification** in the PRD (not just approximate it).

---

## Writing a New Compiler Pass (v4.0+ Guide)

All compiler passes (Passes 10-22 and beyond) follow the same structure.

### Step 1: Create the pass re-export file

Create `src/aether/compiler/stage2_optimizer/passN_name.py`:

- First non-docstring line: `from __future__ import annotations`
- Docstring must cite the research paper (full citation or arXiv ID)
- Re-export the pass class from `optimizer.py` using `__all__`

### Step 2: Implement the pass class in optimizer.py

Add to `src/aether/compiler/stage2_optimizer/optimizer.py`:

```python
class MyNewPass(BasePass):
    name = "my_new_pass"
    description = "Longer description for reports."

    def run(self, graph, architecture, config):
        if not config.enable_my_new_pass:
            return graph, self._skipped()
        try:
            graph.metadata["my_pass_result"] = {}
            return graph, self._applied({"key": "value"})
        except Exception as exc:
            return graph, self._failed(str(exc))
```

### Step 3: Add a CompilerConfig gate (opt-in)

Add to `src/aether/compiler/config.py`:

```python
enable_my_new_pass: bool = False  # opt-in, default disabled
```

### Step 4: Register in OptimizerPipeline.__init__()

```python
self._passes.append(MyNewPass())
self._pass_enabled["my_new_pass"] = config.enable_my_new_pass
```

### Step 5: AEG Format 2.0 artifacts (if pass emits files)

```python
from aether.compiler.aeg_format_v2 import AEGPackageV2
pkg = AEGPackageV2(config.output_path)
dir_ = pkg.root / "my_pass"
dir_.mkdir(exist_ok=True)
(dir_ / "config.json").write_text(json.dumps(result), encoding="utf-8")
```

Add `has_my_pass: bool = False` field to `AEGManifest` in `aeg_format_v2.py`.

### Step 6: Write tests

```python
class TestPassNMyNewPass:
    def test_skipped_when_disabled(self, graph, architecture):
        _, report = MyNewPass().run(graph, architecture, CompilerConfig(enable_my_new_pass=False))
        assert report.status == "skipped"

    def test_applied_when_enabled(self, graph, architecture):
        _, report = MyNewPass().run(graph, architecture, CompilerConfig(enable_my_new_pass=True))
        assert report.status == "applied"

    def test_core_algorithm(self):
        # Pure-Python, no GPU required
        assert my_fn(inputs) == expected
```

### Mandatory Checklist for New Passes

- [ ] `from __future__ import annotations` is first non-docstring line
- [ ] Docstring cites the research paper with arXiv ID or full citation
- [ ] Gated behind `CompilerConfig.enable_X: bool = False` (opt-in)
- [ ] `graph.metadata` annotated with results for downstream passes
- [ ] `_skipped()` returned when gate is off; `_failed(msg)` on errors (never raise)
- [ ] At least one pure-Python correctness test (no GPU required)
- [ ] Exported from `src/aether/compiler/stage2_optimizer/__init__.py`
- [ ] CHANGELOG.md entry added
- [ ] AEG Format 2.0 manifest flag added if pass emits artifacts
- [ ] CI job added in `.github/workflows/ci.yml`

---

## AEG Format 2.0 Directory Contract

Always use `AEGPackageV2` — never create directories manually:

```python
from aether.compiler.aeg_format_v2 import AEGPackageV2, AEGManifest
pkg = AEGPackageV2("/path/to/model.aeg")
pkg.create()                   # idempotent
pkg.upgrade_v1_to_v2()         # migrate AEG/1.x to AEG/2.0
manifest = pkg.read_manifest()
```

Key `AEGManifest` boolean flags (`has_X: bool`) must match the physical files
present in the package. `AEGPackageV2.validate()` checks consistency.

---

## RISC-V NPU Backend Development

To add a new RISC-V NPU vendor backend:

1. Create `src/aether/compiler/stage3_targeting/target_riscv_{vendor}.py`.
2. Implement `AetherRISCVNPUBackend` protocol:
   - `family: str` (e.g. `mips_npu`)
   - `supported_tiling_dims: list[str]`
   - `lower(program) -> list[str]` — emit abstract ISA instructions
   - `emit_asm(program) -> str` — emit final assembly/qdIR
   - `register()` — call `RISCV_NPU_BACKEND_REGISTRY.register(self)`
3. Add hardware profile in `hardware_profile.py` with `is_riscv_npu=True`.
4. Add kernel directory entry in `aeg_format_v2._V4_KERNEL_TARGETS`.

Tiling invariant (ALL backends must satisfy):
```
3 * T * T * dtype_bytes <= scratchpad_bytes,  T must be power of 2
```

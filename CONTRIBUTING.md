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

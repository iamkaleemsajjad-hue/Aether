# Aether Runtime — Makefile

.PHONY: help install install-dev install-all lint format typecheck test test-unit test-integration test-coverage docs docs-serve clean build upload check pre-commit bench fix fix-all

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
RUFF ?= ruff
MYPY ?= mypy
SPHINX ?= sphinx-build

help: ## Show this help message
	@echo "Aether Runtime development commands"
	@echo "==================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install the package in editable mode
	$(PIP) install -e "."

install-dev: ## Install with all development dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install with all optional backends and development tools
	$(PIP) install -e ".[dev,vllm,llamacpp,trtllm,mlx,onnxruntime,triton]"

lint: ## Run ruff linter
	$(RUFF) check src tests examples benchmarks scripts

format: ## Run ruff formatter
	$(RUFF) format src tests examples benchmarks scripts

format-check: ## Check formatting without modifying files
	$(RUFF) format --check src tests examples benchmarks scripts

typecheck: ## Run mypy static analysis
	$(MYPY) src/aether

test: ## Run all tests
	$(PYTEST) tests

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit

test-integration: ## Run integration tests (may require network)
	$(PYTEST) tests/integration -m integration

test-coverage: ## Run tests with coverage report
	$(PYTEST) --cov=aether --cov-report=html --cov-report=term-missing tests

bench: ## Run the benchmark suite
	$(PYTHON) -m benchmarks.bench_suite

bench-smoke: ## Run a quick benchmark smoke test
	$(PYTHON) -m benchmarks.bench_suite --smoke

docs: ## Build Sphinx documentation
	$(SPHINX) -W -b html docs docs/_build/html

docs-serve: ## Serve documentation locally
	$(PYTHON) -m http.server 8000 --directory docs/_build/html

clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info .eggs
	rm -rf .pytest_cache .coverage .coverage.* htmlcov
	rm -rf .mypy_cache .ruff_cache
	rm -rf docs/_build
	find src tests examples benchmarks scripts -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find src tests examples benchmarks scripts -type f -name '*.pyc' -delete 2>/dev/null || true

build: ## Build source and wheel distributions
	$(PYTHON) -m build

upload: ## Upload to PyPI (requires credentials)
	$(PYTHON) -m twine upload dist/*

check: ## Run full check suite: lint, format-check, typecheck, and unit tests
	$(RUFF) check src tests examples benchmarks scripts
	$(RUFF) format --check src tests examples benchmarks scripts
	$(MYPY) src/aether
	$(PYTEST) tests/unit

fix: ## Auto-fix lint and format issues
	$(RUFF) check --fix src tests examples benchmarks scripts
	$(RUFF) format src tests examples benchmarks scripts

fix-all: ## Aggressive auto-fix (unsafe fixes included)
	$(RUFF) check --fix --unsafe-fixes src tests examples benchmarks scripts
	$(RUFF) format src tests examples benchmarks scripts

pre-commit: ## Install pre-commit hooks
	pre-commit install

integration-smoke: ## Quick smoke test with a tiny model
	$(PYTHON) -m aether compile --dry-run Qwen/Qwen3-0.6B
	$(PYTHON) -m aether run Qwen/Qwen3-0.6B --max-tokens 16 --non-interactive

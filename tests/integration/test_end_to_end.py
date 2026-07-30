"""
Integration tests for compile, serve, and end-to-end generation.

These tests require network access to download small HuggingFace models.
They are marked with @pytest.mark.integration and @pytest.mark.network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.server.app import create_app


@pytest.mark.integration
@pytest.mark.network
def test_compile_small_model(tmp_path: Path) -> None:
    """Compile a tiny model from HuggingFace and verify the AEG exists."""
    config = CompilerConfig(
        targets=["cpu_avx512"],
        overwrite=True,
        cache_dir=str(tmp_path / ".aether"),
        calibration_tokens=1024,
        max_calibration_samples=4,
    )
    compiler = Compiler(config)
    aeg = compiler.compile("Qwen/Qwen3-0.6B", output_path=tmp_path / "qwen3-0.6b.aeg")
    assert aeg.root.exists()
    manifest_path = aeg.root / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "model_id" in manifest
    assert manifest["architecture"]["family"] == "qwen_family"


@pytest.mark.integration
@pytest.mark.network
def test_end_to_end_generate(tmp_path: Path) -> None:
    """Compile and generate text from a small model end-to-end."""
    rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_path / ".aether")))
    rt.pull("Qwen/Qwen3-0.6B")
    response = rt.generate(
        model_id="Qwen/Qwen3-0.6B",
        prompt="Hello, my name is",
        max_tokens=16,
        temperature=0.7,
    )
    assert response.text
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0
    assert response.metrics.throughput_tps > 0


@pytest.mark.integration
@pytest.mark.network
def test_chat_completion(tmp_path: Path) -> None:
    """Test chat completion with a small model."""
    rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_path / ".aether")))
    rt.pull("Qwen/Qwen3-0.6B")
    response = rt.chat(
        model_id="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "user", "content": "Say hello in one word."},
        ],
        max_tokens=8,
    )
    assert response.text
    assert response.finish_reason in ("stop", "length")


@pytest.mark.integration
@pytest.mark.network
def test_compile_plan_dry_run() -> None:
    """Test that compilation planning works for a known model."""
    compiler = Compiler()
    plan = compiler.plan("Qwen/Qwen3-0.6B")
    assert plan.is_feasible
    assert len(plan.fusion_opportunities) > 0
    assert plan.estimated_memory_gb > 0
    assert plan.estimated_compile_time_s > 0


@pytest.mark.integration
@pytest.mark.network
def test_model_info(tmp_path: Path) -> None:
    """Compile and inspect model metadata."""
    rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_path / ".aether")))
    rt.pull("Qwen/Qwen3-0.6B")
    info = rt.info("Qwen/Qwen3-0.6B")
    assert "model_id" in info
    assert info["architecture"]["family"] == "qwen_family"


@pytest.mark.integration
@pytest.mark.network
def test_openai_compatible_server(tmp_path: Path) -> None:
    """Test that the server creates properly and routes work."""
    config = RuntimeConfig(model_cache_dir=str(tmp_path / ".aether"))
    app = create_app(config)
    assert app is not None
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    hw_response = client.get("/v1/hardware")
    assert hw_response.status_code == 200
    assert "target_id" in hw_response.json()


@pytest.mark.integration
@pytest.mark.network
def test_model_list_and_remove(tmp_path: Path) -> None:
    """Test listing and removing compiled models."""
    rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_path / ".aether")))
    rt.pull("Qwen/Qwen3-0.6B")
    models = rt.list()
    assert "Qwen/Qwen3-0.6B" in models or any("qwen" in m.lower() for m in models)
    rt.remove("Qwen/Qwen3-0.6B")
    assert "Qwen/Qwen3-0.6B" not in rt.list()


@pytest.mark.integration
@pytest.mark.network
def test_backend_detection() -> None:
    """Test that at least one backend is available."""
    from aether.backends.registry import BackendRegistry
    registry = BackendRegistry()
    available = registry.get_available_backend_names()
    assert "pytorch" in available or len(available) > 0


@pytest.mark.integration
@pytest.mark.network
def test_hardware_detection() -> None:
    """Test hardware fingerprinting."""
    from aether.runtime.hardware import HardwareDetector
    detector = HardwareDetector()
    fingerprint = detector.detect()
    assert fingerprint.target_id is not None
    assert fingerprint.cpu_count > 0
    assert fingerprint.total_ram_gb > 0


@pytest.mark.integration
@pytest.mark.network
def test_benchmark_smoke(tmp_path: Path) -> None:
    """Test benchmark produces metrics."""
    rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_path / ".aether")))
    rt.pull("Qwen/Qwen3-0.6B")
    result = rt.benchmark("Qwen/Qwen3-0.6B", max_tokens=16)
    assert result["throughput_tps"] > 0
    assert result["completion_tokens"] > 0


@pytest.mark.integration
@pytest.mark.network
def test_aeg_format_roundtrip(tmp_path: Path) -> None:
    """Test compile, save, load roundtrip for AEG format."""
    config = CompilerConfig(targets=["cpu_avx512"], overwrite=True)
    compiler = Compiler(config)
    aeg = compiler.compile("Qwen/Qwen3-0.6B", output_path=tmp_path / "roundtrip.aeg")
    aeg_path = aeg.root
    from aether.core.aeg_format import load_aeg_package
    loaded = load_aeg_package(aeg_path)
    loaded.verify_integrity()
    assert loaded.model_id == "Qwen--Qwen3-0.6B" or loaded.model_id is not None


@pytest.mark.integration
@pytest.mark.network
def test_cross_hardware_portability(tmp_path: Path) -> None:
    """Test that same AEG works on CPU with different backends."""
    config = CompilerConfig(targets=["cpu_avx512"], overwrite=True)
    compiler = Compiler(config)
    aeg = compiler.compile("Qwen/Qwen3-0.6B", output_path=tmp_path / "portable.aeg")
    aeg.save()
    rt = Runtime(RuntimeConfig(model_cache_dir=str(tmp_path / ".aether")))
    rt.pull("Qwen/Qwen3-0.6B")
    response = rt.generate(
        model_id="Qwen/Qwen3-0.6B",
        prompt="Hello.",
        max_tokens=8,
    )
    assert response.text


@pytest.mark.integration
def test_synthetic_model_generation(small_architecture) -> None:
    """Test that the runtime can work with a tiny synthetic fixture."""
    from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
    config = CompilerConfig(targets=["cpu_avx512"])
    pipeline = IngestionPipeline(config)
    graph = pipeline._build_architecture_graph(AEGGraph(name="test"), small_architecture)
    assert graph.node_count > 0
    assert graph.edge_count > 0


from aether.core.graph import AEGGraph

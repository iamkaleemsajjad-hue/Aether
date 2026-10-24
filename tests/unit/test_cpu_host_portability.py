"""CPU deployment contracts; mocked checks never publish execution evidence."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from aether.kernels import native_cpu as native


def _kernels(monkeypatch, flags=("-O3", "-march=native")):
    # These tests exercise a concrete host-specialized build contract, on any
    # operating system running the suite. Anonymous-host portability is separate.
    monkeypatch.setattr(native, "_normalized_machine", lambda: "x86_64")
    monkeypatch.setattr(
        native.platform, "processor", lambda: "Intel64 Family 6 Model 186 Stepping 3, GenuineIntel"
    )
    monkeypatch.setattr(
        "aether.backends.capabilities.cpu_runtime_features",
        lambda: {"avx": True, "avx2": True, "sse2": True},
    )
    return native.NativeCPUKernels(native.CompilerToolchain("g++", "compiler.exe", flags))


def test_msvc_build_command_preserves_precision_standard_and_parallel_flags():
    toolchain = native.CompilerToolchain(
        "clang-cl", "clang-cl", ("/std:c++17", "/fp:precise", "/openmp"), is_msvc=True
    )
    command = toolchain.build_command(Path("kernel.cpp"), Path("kernel.dll"))
    assert {"/LD", "/std:c++17", "/fp:precise", "/openmp"}.issubset(command)


@pytest.mark.parametrize("compiler", ["cl", "clang-cl"])
def test_windows_msvc_and_clang_cl_discovery(compiler, monkeypatch):
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt", environ=dict(os.environ)))
    monkeypatch.setattr(
        native, "_find_executable", lambda name: "usable.exe" if name == compiler else None
    )
    monkeypatch.setattr(native, "_visual_studio_environments", lambda: [])
    monkeypatch.setattr(native, "_verify_toolchain", lambda toolchain: toolchain.is_msvc)
    toolchain = native.detect_toolchain()
    assert toolchain is not None
    assert toolchain.name == ("msvc" if compiler == "cl" else "clang-cl")
    assert "/openmp" in toolchain.base_flags


def test_visual_studio_sdk_environment_used_without_global_mutation(monkeypatch):
    sdk_environment = (("PATH", "sdk-bin"), ("INCLUDE", "sdk-headers"), ("LIB", "sdk-libs"))
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt", environ=dict(os.environ)))
    monkeypatch.setattr(native, "_find_executable", lambda _: None)
    monkeypatch.setattr(
        native, "_visual_studio_environments", lambda: [(Path("vs"), sdk_environment)]
    )
    monkeypatch.setattr(
        native.shutil,
        "which",
        lambda name, path=None: "sdk-cl.exe" if name == "cl" and path == "sdk-bin" else None,
    )
    monkeypatch.setattr(
        native,
        "_verify_toolchain",
        lambda toolchain: toolchain.subprocess_environment().get("INCLUDE") == "sdk-headers",
    )
    prior = dict(native.os.environ)
    toolchain = native.detect_toolchain()
    assert toolchain is not None
    assert dict(toolchain.environment) == dict(sdk_environment)
    assert native.os.environ == prior


def test_vswhere_initializes_matching_visual_studio_developer_environment(tmp_path, monkeypatch):
    installer_root = tmp_path / "program-files-x86"
    finder = installer_root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    finder.parent.mkdir(parents=True)
    finder.write_bytes(b"finder")
    installation = tmp_path / "Visual Studio"
    script = installation / "Common7" / "Tools" / "VsDevCmd.bat"
    script.parent.mkdir(parents=True)
    script.write_text("developer setup")
    monkeypatch.setenv("ProgramFiles(x86)", str(installer_root))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "program-files"))
    monkeypatch.delenv("VSINSTALLDIR", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda _: None)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == str(finder):
            import json

            return SimpleNamespace(
                returncode=0, stdout=json.dumps([{"installationPath": str(installation)}])
            )
        return SimpleNamespace(
            returncode=0,
            stdout="PATH=sdk-bin\nINCLUDE=sdk-headers\nLIB=sdk-libs\nUNRELATED_SECRET=do-not-capture\n",
        )

    monkeypatch.setattr(native.subprocess, "run", run)
    environments = native._visual_studio_environments()
    assert environments == [
        (installation, (("INCLUDE", "sdk-headers"), ("LIB", "sdk-libs"), ("PATH", "sdk-bin")))
    ]
    assert "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in commands[0]
    assert "VsDevCmd.bat" in commands[1][-1]
    assert "-arch=" in commands[1][-1]


def test_compiler_probe_uses_race_free_openmp_reduction_and_sdk_environment(monkeypatch):
    source = []
    sdk = (("INCLUDE", "sdk-headers"),)
    toolchain = native.CompilerToolchain("g++", "g++", ("-fopenmp",), environment=sdk)

    def run(command, **kwargs):
        source.append(Path(command[-1]).read_text())
        Path(command[command.index("-o") + 1]).write_bytes(b"probe")
        assert kwargs["env"]["INCLUDE"] == "sdk-headers"
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    class Probe:
        def __call__(self):
            return 42

    monkeypatch.setattr(native.subprocess, "run", run)
    monkeypatch.setattr(
        native, "_open_shared_library", lambda *args: SimpleNamespace(aether_probe=Probe())
    )
    monkeypatch.setattr(native, "_release_library", lambda _: None)
    assert native._verify_toolchain(toolchain)
    assert "reduction(+:total)" in source[0]
    assert "#include <cmath>" in source[0]


def test_failed_installed_compiler_retains_real_diagnostics(monkeypatch):
    toolchain = native.CompilerToolchain("g++", "broken-g++")
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"missing C++ headers"
        ),
    )
    assert not native._verify_toolchain(toolchain)
    assert "missing C++ headers" in native.compiler_probe_failures()["broken-g++"]
    kernels = native.NativeCPUKernels(toolchain)
    kernels.toolchain = None
    kernels._discovery_failures = native.compiler_probe_failures()
    assert not kernels.ensure_compiled()
    assert "missing C++ headers" in kernels.build_error


def test_native_identity_changes_with_os_usable_isa_compiler_and_environment(tmp_path, monkeypatch):
    compiler = tmp_path / "compiler.exe"
    compiler.write_bytes(b"version1")
    kernels = _kernels(monkeypatch)
    kernels.toolchain = native.CompilerToolchain("g++", str(compiler), ("-march=native",))
    first = kernels.build_identity()
    monkeypatch.setattr(
        "aether.backends.capabilities.cpu_runtime_features",
        lambda: {"avx": True, "avx2": False, "sse2": True},
    )
    changed_features = kernels.build_identity()
    assert first != changed_features
    monkeypatch.setenv("CL", "/arch:AVX2")
    changed_environment = kernels.build_identity()
    assert changed_environment["environment_sha256"] != changed_features["environment_sha256"]
    assert "/arch:AVX2" not in str(changed_environment)
    compiler.write_bytes(b"version2-upgraded")
    assert kernels.build_identity()["compiler"] != changed_environment["compiler"]


def test_cached_native_build_is_not_reused_across_cpu_isa(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_KERNEL_CACHE", str(tmp_path))
    built = []

    def compile_library(self, output):
        built.append(output)
        output.write_bytes(b"test library")
        return True

    monkeypatch.setattr(native.NativeCPUKernels, "_compile", compile_library)
    monkeypatch.setattr(native.NativeCPUKernels, "_bind_signatures", lambda self: None)
    monkeypatch.setattr(native, "_open_shared_library", lambda *args: object())
    _kernels(monkeypatch).ensure_compiled()
    second = _kernels(monkeypatch)
    monkeypatch.setattr("aether.backends.capabilities.cpu_runtime_features", lambda: {"sse2": True})
    second.ensure_compiled()
    assert len(built) == 2
    assert built[0] != built[1]


@pytest.mark.parametrize(
    "changed",
    [
        "platform",
        "machine",
        "pointer_bits",
        "required_cpu_features",
        "host_cpu_signature",
        "isa_policy",
        "schema_version",
    ],
)
def test_packaged_library_requires_compatible_process_and_instruction_contract(
    changed, monkeypatch
):
    identity = _kernels(monkeypatch).build_identity()
    assert native.library_compatibility_error(identity) is None
    replacements = {
        "platform": "other-os",
        "machine": "other-arch",
        "pointer_bits": 0,
        "required_cpu_features": ["unavailable_extension"],
        "host_cpu_signature": "different-cpu",
        "isa_policy": "unknown",
        "schema_version": 999,
    }
    identity[changed] = replacements[changed]
    assert native.library_compatibility_error(identity) is not None


def test_baseline_library_does_not_require_same_host_cpu(monkeypatch):
    identity = _kernels(monkeypatch, flags=("-O3",)).build_identity()
    assert identity["isa_policy"] == "architecture_baseline"
    assert identity["required_cpu_features"] == []
    monkeypatch.setattr("aether.backends.capabilities.cpu_runtime_features", lambda: {})
    assert native.library_compatibility_error(identity) is None


@pytest.mark.parametrize("identity", [None, {"schema_version": 999}])
def test_legacy_or_unknown_packaged_native_library_falls_back_before_loading(
    tmp_path, monkeypatch, identity
):
    from aether.runtime.aeg_loader import _load_packaged_native_kernels

    path = tmp_path / "legacy.dll"
    path.write_bytes(b"do not execute")
    descriptor = {"backend": "native_cpu", "path": "legacy.dll", "sha256": "hash"}
    if identity is not None:
        descriptor["build_identity"] = identity
    package = SimpleNamespace(root=tmp_path, metadata={"kernel_artifacts": [descriptor]})
    attempted = []
    kernels = SimpleNamespace(
        load_library=lambda *args, **kwargs: pytest.fail(
            "unknown ISA must not reach the OS loader"
        ),
        ensure_compiled=lambda: attempted.append("host-recompile") or False,
    )
    _load_packaged_native_kernels(package, kernels)
    assert attempted == ["host-recompile"]


def test_generic_cpu_is_registered_and_framework_free():
    from aether.backends.native_cpu_backend import NativeCPUBackend
    from aether.compiler.stage3_targeting.target_registry import TargetRegistry
    from aether.core.constants import BACKEND_BY_TARGET
    from aether.core.types import HardwareTarget

    profile = TargetRegistry().get_profile("cpu_generic")
    assert profile.recommended_backend == "aether_cpu"
    assert not profile.supports_bf16
    assert profile.memory_gb == 0
    assert HardwareTarget.CPU_GENERIC.vendor == "CPU"
    assert BACKEND_BY_TARGET["cpu_generic"][0] == "aether_cpu"
    assert "cpu_generic" in NativeCPUBackend().info.supported_targets


def test_reference_provider_never_discovers_or_builds_a_compiler(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        native,
        "detect_toolchain",
        lambda: pytest.fail("reference execution must not discover compilers"),
    )
    reference = native.NativeCPUKernels.reference()
    assert reference.toolchain is None
    assert not reference.is_native
    np.testing.assert_allclose(
        reference.sgemv(np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32)), np.ones(3)
    )
    assert not reference.is_native


def test_native_codegen_does_not_enable_unconfirmed_process_simd(monkeypatch):
    monkeypatch.setattr(native, "_normalized_machine", lambda: "x86_64")
    monkeypatch.setattr(
        native.platform, "processor", lambda: "Intel64 Family 6 Model 186 Stepping 3, GenuineIntel"
    )
    monkeypatch.setattr("aether.backends.capabilities.cpu_runtime_features", lambda: {})
    assert native._process_codegen_flags(("-O3", "-march=native")) == ("-O3", "-march=x86-64")
    monkeypatch.setattr(
        "aether.backends.capabilities.cpu_runtime_features",
        lambda: {"sse2": True, "avx": False, "avx2": False, "avx512f": False},
    )
    flags = native._process_codegen_flags(("-O3", "-march=native"))
    assert {"-mno-avx", "-mno-avx2", "-mno-avx512f", "-mno-fma", "-mno-f16c"}.issubset(flags)


@pytest.mark.parametrize("processor", ["", "x86_64", "AMD64", "unknown", "Intel"])
def test_anonymous_cpu_retains_only_explicit_os_usable_vector_isa(monkeypatch, processor):
    monkeypatch.setattr(native, "_normalized_machine", lambda: "x86_64")
    monkeypatch.setattr(native.platform, "processor", lambda: processor)
    monkeypatch.delenv("PROCESSOR_IDENTIFIER", raising=False)
    features = {"sse2": True, "avx": True, "avx2": True, "fma3": True,
                "f16c": False, "avx512f": False}
    monkeypatch.setattr("aether.backends.capabilities.cpu_runtime_features", lambda: features)
    kernels = native.NativeCPUKernels(
        native.CompilerToolchain("g++", "compiler.exe", ("-O3", "-march=native"))
    )
    flags = kernels.toolchain.base_flags
    assert "-march=native" not in flags and "-mcpu=native" not in flags
    assert {"-march=x86-64", "-mavx", "-mavx2", "-mfma"}.issubset(flags)
    assert "-mf16c" not in flags and "-mavx512f" not in flags
    identity = kernels.build_identity()
    assert identity["isa_policy"] == "architecture_baseline"
    assert identity["host_cpu_signature"] is None
    assert set(identity["required_cpu_features"]) == {"sse2", "avx", "avx2", "fma3"}
    assert native.library_compatibility_error(identity) is None
    monkeypatch.setattr(native.platform, "processor", lambda: "another generic CPU")
    assert native.library_compatibility_error(identity) is None
    features["avx2"] = False
    assert "avx2" in native.library_compatibility_error(identity)


def test_anonymous_codegen_does_not_enable_extension_dependencies(monkeypatch):
    monkeypatch.setattr(native, "_normalized_machine", lambda: "x86_64")
    monkeypatch.setattr(native.platform, "processor", lambda: "x86_64")
    monkeypatch.delenv("PROCESSOR_IDENTIFIER", raising=False)
    monkeypatch.setattr(
        "aether.backends.capabilities.cpu_runtime_features",
        lambda: {"avx": False, "avx2": True, "avx512f": True, "fma3": True},
    )
    flags = native._process_codegen_flags(("-O3", "-march=native"))
    assert flags == ("-O3", "-march=x86-64")


def test_packaged_host_native_identity_is_rejected_on_anonymous_cpu(monkeypatch):
    identity = _kernels(monkeypatch).build_identity()
    assert identity["isa_policy"] == "host_native"
    monkeypatch.setattr(native.platform, "processor", lambda: "x86_64")
    monkeypatch.delenv("PROCESSOR_IDENTIFIER", raising=False)
    assert "concrete CPU identity" in native.library_compatibility_error(identity)
    identity["host_cpu_signature"] = None
    assert "concrete CPU identity" in native.library_compatibility_error(identity)


def test_unverifiable_msvc_environment_never_reaches_native_compilation(monkeypatch):
    monkeypatch.setattr(native, "_normalized_machine", lambda: "x86_64")
    monkeypatch.setattr(native.platform, "processor", lambda: "")
    monkeypatch.delenv("PROCESSOR_IDENTIFIER", raising=False)
    monkeypatch.setenv("CL", "/arch:AVX2")
    monkeypatch.setattr("aether.backends.capabilities.cpu_runtime_features", lambda: {"avx2": True})
    kernels = native.NativeCPUKernels(
        native.CompilerToolchain("msvc", "cl", ("/std:c++17",), is_msvc=True)
    )
    monkeypatch.setattr(kernels, "_compile", lambda *_: pytest.fail("unverified ISA must not compile"))
    assert not kernels.ensure_compiled()
    assert "concrete CPU identity" in kernels.build_error


def test_concrete_windows_cpu_keeps_native_codegen(monkeypatch):
    kernels = _kernels(monkeypatch)
    assert "-march=native" in kernels.toolchain.base_flags
    assert kernels.build_identity()["isa_policy"] == "host_native"
    assert kernels.build_identity()["host_cpu_signature"]


def test_real_explicit_isa_build_executes_without_cpu_identity(monkeypatch):
    import numpy as np

    monkeypatch.setattr(native.platform, "processor", lambda: "")
    monkeypatch.delenv("PROCESSOR_IDENTIFIER", raising=False)
    kernels = native.NativeCPUKernels()
    if kernels.toolchain is None:
        pytest.skip("a usable compiler is needed for real explicit-ISA execution")
    assert "-march=native" not in kernels.toolchain.base_flags
    assert kernels.ensure_compiled(), kernels.build_error
    assert kernels.is_native
    assert kernels.build_identity()["isa_policy"] == "architecture_baseline"
    matrix = np.arange(24, dtype=np.float32).reshape(3, 8) / 32
    vector = np.linspace(-0.75, 0.5, 8, dtype=np.float32)
    np.testing.assert_allclose(kernels.sgemv(matrix, vector), matrix @ vector, atol=1e-6)


def test_auto_target_uses_executable_canonical_evidence(monkeypatch):
    from aether.core.types import HardwareTarget

    def capability(target_id, available=True, implemented=True):
        return SimpleNamespace(target_id=target_id, available=available, implemented=implemented)

    monkeypatch.setattr(
        "aether.backends.hardware_detector.detect_all_capabilities",
        lambda: [
            capability("cpu_generic"),
            capability("cuda_sm89", available=False),
            capability("cuda_sm90", implemented=False),
            capability("unregistered_gpu"),
        ],
    )
    assert HardwareTarget.auto() is HardwareTarget.CPU_GENERIC
    monkeypatch.setattr(
        "aether.backends.hardware_detector.detect_all_capabilities",
        lambda: [
            capability("cpu_avx2"),
            capability("cuda_sm89"),
        ],
    )
    assert HardwareTarget.auto() is HardwareTarget.CUDA_SM89


def test_generic_artifact_compiles_reloads_and_generates(tiny_local_safetensors_model, tmp_path):
    from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig

    artifact = Compiler(
        CompilerConfig(
            quality_mode="diagnostic", targets=["cpu_generic"], cache_dir=str(tmp_path / "cache")
        )
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "generic.aeg")
    artifact.verify_integrity()
    assert artifact.manifest.kernels.targets == ["cpu_generic"]
    for descriptor in artifact.metadata.get("kernel_artifacts", []):
        assert native.library_compatibility_error(descriptor["build_identity"]) is None
    response = Runtime(
        RuntimeConfig(
            allow_uncertified=True,
            backend_name="aether_cpu",
            execution_devices=["cpu"],
            hf_offline=True,
        )
    ).generate(str(artifact.root), "hello", max_tokens=2, temperature=0.0)
    assert response.usage["completion_tokens"] == 2
    assert response.metrics.backend_name == "aether_cpu"


def test_windows_packaged_dll_loads_verified_cache_copy(tmp_path, monkeypatch):
    import hashlib

    kernels = _kernels(monkeypatch)
    identity = kernels.build_identity()
    monkeypatch.setattr(
        native,
        "os",
        SimpleNamespace(name="nt", environ=dict(os.environ), replace=os.replace, fsync=os.fsync),
    )
    native.os.environ["AETHER_KERNEL_CACHE"] = str(tmp_path / "cache")
    artifact = tmp_path / "artifact.dll"
    artifact.write_bytes(b"verified test library")
    expected_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    loaded = []
    monkeypatch.setattr(
        native, "_open_shared_library", lambda path, toolchain: loaded.append(path) or object()
    )
    monkeypatch.setattr(kernels, "_bind_signatures", lambda: None)
    assert kernels.load_library(artifact, expected_hash, identity)
    assert loaded[0] != artifact.resolve()
    assert loaded[0].name.startswith("aether_packaged_")
    assert loaded[0].read_bytes() == artifact.read_bytes()
    first_loaded = loaded[0]
    assert kernels.load_library(artifact, expected_hash, identity)
    assert loaded[1] == first_loaded
    artifact.unlink()
    assert not artifact.exists()
    assert first_loaded.is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows DLL lifetime regression")
def test_real_packaged_windows_aeg_can_be_removed_while_engine_is_live(tmp_path, monkeypatch):
    import tempfile
    import numpy as np

    from aether import Compiler, CompilerConfig
    from aether.backends.hardware_validation import _write_probe_checkpoint
    from aether.runtime.aeg_loader import load_engine_from_path

    monkeypatch.setenv("AETHER_KERNEL_CACHE", str(tmp_path / "persistent-cache"))
    if native.detect_toolchain() is None:
        pytest.skip("A usable compiler is required to reproduce a packaged Windows DLL")
    with tempfile.TemporaryDirectory(dir=tmp_path, prefix="removable-aeg-") as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        _write_probe_checkpoint(checkpoint)
        artifact = Compiler(
            CompilerConfig(
                quality_mode="diagnostic",
                targets=["cpu_generic"],
                precision="FP32",
                cache_dir=str(root / "cache"),
            )
        ).compile(str(checkpoint), output_path=root / "compiled.aeg")
        assert artifact.metadata.get("kernel_artifacts")
        engine = load_engine_from_path(artifact.root, allow_uncertified=True)
        logits, _ = engine.forward(np.array([1, 2]))
        assert engine.kernels.is_native
        assert np.isfinite(logits).all()
        assert not engine.kernels.library_path.is_relative_to(root)
    assert not root.exists()
    logits, _ = engine.forward(np.array([3]))
    assert np.isfinite(logits).all()


@pytest.mark.parametrize("target", ["cpu_generic", "cpu_avx2", "cpu_avx512", "cpu_neon"])
def test_cpu_family_profiles_do_not_infer_bf16_from_baseline_simd(target):
    from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile

    assert not HardwareProfile.from_target_id(target).supports_bf16

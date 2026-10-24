"""Native build reproducibility, including execution of two independent cold builds."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from aether.kernels import native_cpu as native


@pytest.mark.parametrize(
    "banner,returncode,expected",
    [
        ("GNU ld (GNU Binutils) 2.44", 0, True),
        ("LLD 20.1.0", 0, False),
        ("GNU gold 1.16", 0, False),
        ("GNU ld", 1, False),
    ],
)
def test_linker_policy_identifies_selected_linker(
    banner: str, returncode: int, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply GNU options only to a successfully identified GNU PE linker."""
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt", environ=dict(os.environ)))
    commands = []
    toolchain = native.CompilerToolchain(
        "clang++",
        "driver",
        ("-O2", "-fuse-ld=selected"),
        environment=(("LIB", "sdk-libraries"),),
        link_flags=("-Wl,--as-needed",),
    )

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        assert kwargs["env"]["LIB"] == "sdk-libraries"
        return SimpleNamespace(returncode=returncode, stdout="", stderr=banner)

    monkeypatch.setattr(native.subprocess, "run", run)
    configured = native._configure_linker_policy(toolchain)
    assert commands == [
        [
            "driver",
            "-O2",
            "-fuse-ld=selected",
            "-Wl,--as-needed",
            "-Wl,--version",
        ]
    ]
    if expected:
        assert configured.linker_policy == "gnu-pe-deterministic-v1"
        command = configured.build_command(Path("kernel.cpp"), Path("kernel.dll"))
        assert "-Wl,--disable-auto-image-base" in command
        assert "-Wl,--no-insert-timestamp" in command
        assert "-Wl,--as-needed" in command
        assert not any(
            flag in command
            for flag in (
                "-Wl,--disable-dynamicbase",
                "-Wl,--disable-reloc-section",
            )
        )
    else:
        assert configured == toolchain


@pytest.mark.parametrize("system,is_msvc", [("posix", False), ("nt", True)])
def test_other_platform_and_msvc_commands_remain_unchanged(
    system: str, is_msvc: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not send GNU PE arguments to another platform or MSVC driver."""
    monkeypatch.setattr(native, "os", SimpleNamespace(name=system, environ=dict(os.environ)))
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected probe"))
    toolchain = native.CompilerToolchain("compiler", "driver", ("standard",), is_msvc=is_msvc)
    source, output = Path("kernel.cpp"), Path("kernel.library")
    assert native._configure_linker_policy(toolchain) is toolchain
    assert toolchain.build_command(source, output) == (
        ["driver", "/nologo", "/O2", "/LD", "standard", str(source), f"/Fe:{output}"]
        if is_msvc
        else ["driver", "standard", "-shared", "-o", str(output), str(source)]
    )


def test_discovery_probes_effective_gnu_pe_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiler discovery must validate the same linker policy the build uses."""
    monkeypatch.setattr(native, "os", SimpleNamespace(name="nt", environ=dict(os.environ)))
    monkeypatch.setattr(native, "_CANDIDATES", (("g++", ("-O2",)),))
    monkeypatch.setattr(native, "_find_executable", lambda _: "driver")
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0,
            stdout="GNU ld (GNU Binutils) 2.44",
            stderr="",
        ),
    )
    probed = []
    monkeypatch.setattr(native, "_verify_toolchain", lambda tc: probed.append(tc) or True)
    toolchain = native.detect_toolchain()
    assert probed == [toolchain]
    assert toolchain.link_flags == (
        "-Wl,--disable-auto-image-base",
        "-Wl,--no-insert-timestamp",
    )


def test_legacy_cache_is_not_reused_and_policy_recipe_change_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic cache routing only; actual native execution is tested separately."""
    monkeypatch.setenv("AETHER_KERNEL_CACHE", str(tmp_path))
    kernels = native.NativeCPUKernels.reference()
    kernels.toolchain = native.CompilerToolchain("g++", "driver", ("-O2",))
    identity = kernels.build_identity()
    legacy = {key: value for key, value in identity.items() if key != "build_recipe"}
    legacy["compiler"] = {
        key: value
        for key, value in identity["compiler"].items()
        if key not in {"link_flags", "linker_policy"}
    }
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:16]
    old_library = tmp_path / f"aether_cpu_{digest}{kernels.toolchain.library_suffix}"
    old_library.write_bytes(b"old recipe: never execute")
    attempted = []
    monkeypatch.setattr(kernels, "_compile", lambda output: attempted.append(output) or False)
    monkeypatch.setattr(native, "_open_shared_library", lambda *a: pytest.fail("old cache loaded"))
    assert not kernels.ensure_compiled()
    assert len(attempted) == 1 and attempted[0] != old_library
    assert old_library.read_bytes() == b"old recipe: never execute"
    assert native.library_compatibility_error(legacy) is None

    kernels.toolchain = replace(
        kernels.toolchain,
        link_flags=("-Wl,--no-insert-timestamp",),
        linker_policy="gnu-pe-deterministic-v1",
    )
    policy_identity = kernels.build_identity()
    assert policy_identity != identity
    monkeypatch.setattr(native, "_BUILD_RECIPE_VERSION", native._BUILD_RECIPE_VERSION + 1)
    assert kernels.build_identity() != policy_identity


@pytest.mark.skipif(os.name != "nt", reason="GNU PE linker regression")
def test_two_cold_gnu_pe_builds_are_identical_and_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two real cold DLLs must match byte for byte and execute correct kernels."""
    toolchain = native.detect_toolchain()
    if toolchain is None or toolchain.is_msvc:
        pytest.skip("GNU PE toolchain unavailable")
    version = subprocess.run(
        [toolchain.executable, *toolchain.base_flags, *toolchain.link_flags, "-Wl,--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=toolchain.subprocess_environment(),
    )
    if version.returncode or "GNU ld" not in version.stdout + version.stderr:
        pytest.skip("Selected linker is not GNU ld")

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    rng = np.random.default_rng(712)
    lhs = rng.normal(size=(7, 19)).astype(np.float32)
    rhs = rng.normal(size=(19, 11)).astype(np.float32)
    vector = rng.normal(size=19).astype(np.float32)
    payloads = []
    identities = []
    paths = []
    for job in ("first", "second"):
        cache = tmp_path / job / "cold-cache"
        assert not cache.exists()
        monkeypatch.setenv("AETHER_KERNEL_CACHE", str(cache))
        kernels = native.NativeCPUKernels(toolchain)
        try:
            assert kernels.ensure_compiled(), kernels.build_error
            assert kernels.is_native
            paths.append(kernels.library_path)
            identities.append(kernels.build_identity())
            payloads.append(kernels.library_path.read_bytes())
            np.testing.assert_allclose(
                kernels.sgemm(lhs, rhs, force_native=True),
                lhs @ rhs,
                rtol=2e-5,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                kernels.sgemv(rhs.T, vector),
                rhs.T @ vector,
                rtol=2e-5,
                atol=2e-5,
            )
        finally:
            native._release_library(kernels._lib)
            kernels._lib = None

    assert paths[0].parent != paths[1].parent
    assert paths[0].name == paths[1].name
    assert identities[0] == identities[1]
    assert payloads[0] == payloads[1], [hashlib.sha256(data).hexdigest() for data in payloads]
    for data in payloads:
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        assert data[pe : pe + 4] == b"PE\0\0"
        assert struct.unpack_from("<I", data, pe + 8)[0] == 0  # COFF timestamp
        assert not struct.unpack_from("<H", data, pe + 22)[0] & 1  # relocations retained
        assert struct.unpack_from("<H", data, pe + 24 + 70)[0] & 0x40  # ASLR enabled

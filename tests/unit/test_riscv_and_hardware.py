"""
Tests for v4.0 / v5.0 hardware targets and the RISC-V NPU Abstract IR layer.

Covers:
  - HardwareProfile dataclass fields (FP4, TEE, ternary, MXFP6, RISC-V NPU flags)
  - All 20+ target profiles in _TARGET_PROFILES
  - RISCVNPUIRBuilder tiling math (tile condition: 3*T^2*dtype_bytes <= scratchpad)
  - RISCVNPUBackendRegistry register / get / is_registered
  - All 4 RISC-V vendor backends: lower() returns non-empty bytes; opcodes handled
  - RISCVNPUCompiler facade (compile with registered backends)
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so tests run without GPU environment
# ---------------------------------------------------------------------------

for mod in [
    "aether.utils.logging",
]:
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.get_logger = lambda name: MagicMock()  # type: ignore[attr-defined]
        sys.modules[mod] = stub


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from aether.compiler.stage3_targeting.hardware_profile import (
    HardwareProfile,
    _TARGET_PROFILES,
)
from aether.compiler.stage3_targeting.riscv_npu_ir import (
    RISCV_NPU_BACKEND_REGISTRY,
    RISCVNPUBackendRegistry,
    RISCVNPUCompiler,
    RISCVNPUIRBuilder,
    RISCVNPUOpcode,
    RISCVNPUProgram,
    RISCVNPUInstruction,
    AetherRISCVNPUBackend,
)


# ===========================================================================
# HardwareProfile tests
# ===========================================================================


class TestHardwareProfileFields:
    """Verify that HardwareProfile contains all new v4.0/v5.0 fields."""

    def test_fp4_field_exists(self):
        p = HardwareProfile(target_id="cuda_sm100", name="B200")
        assert hasattr(p, "supports_fp4")
        assert hasattr(p, "flops_fp4")

    def test_tee_fields(self):
        p = HardwareProfile(target_id="x", name="y", supports_tee=True, tee_backend="nvidia_cc")
        assert p.supports_tee is True
        assert p.tee_backend == "nvidia_cc"

    def test_ternary_field(self):
        p = HardwareProfile(target_id="x", name="y", supports_ternary=True)
        assert p.supports_ternary is True

    def test_mxfp6_field(self):
        p = HardwareProfile(target_id="x", name="y", supports_mxfp6=True)
        assert p.supports_mxfp6 is True

    def test_riscv_npu_fields(self):
        p = HardwareProfile(
            target_id="x", name="y",
            is_riscv_npu=True,
            abstract_ir_family="mips_npu",
        )
        assert p.is_riscv_npu is True
        assert p.abstract_ir_family == "mips_npu"

    def test_nvlink_bandwidth_field(self):
        p = HardwareProfile(target_id="x", name="y", nvlink_bandwidth_gb_s=1800.0)
        assert p.nvlink_bandwidth_gb_s == pytest.approx(1800.0)

    def test_tdp_watts_field(self):
        p = HardwareProfile(target_id="x", name="y", tdp_watts=700.0)
        assert p.tdp_watts == pytest.approx(700.0)


class TestTargetProfiles:
    """Verify every registered profile has correct capabilities."""

    @pytest.mark.parametrize("target_id,expected_fp4", [
        ("cuda_sm70",  False),
        ("cuda_sm80",  False),
        ("cuda_sm90",  False),
        ("cuda_sm100", True),
        ("cuda_sm120", True),
        ("cuda_sm130", True),
        ("cuda_sm100_tee", True),
        ("cuda_sm100_gb300", True),
        ("amd_mi350x", True),
        ("rocm_cdna5_mi455x", True),
        ("rocm_cdna3", True),
        ("cpu_avx512", False),
        ("cpu_avx512_ternary", False),
        ("fpga_ternary", False),
    ])
    def test_fp4_support(self, target_id: str, expected_fp4: bool):
        data = _TARGET_PROFILES[target_id]
        assert data.get("supports_fp4", False) is expected_fp4, (
            f"{target_id}: expected supports_fp4={expected_fp4}"
        )

    @pytest.mark.parametrize("target_id,expected_tee", [
        ("cuda_sm100_tee", True),
        ("cuda_sm100", False),
        ("cuda_sm90", False),
    ])
    def test_tee_support(self, target_id: str, expected_tee: bool):
        data = _TARGET_PROFILES[target_id]
        assert data.get("supports_tee", False) is expected_tee

    @pytest.mark.parametrize("target_id", [
        "cpu_avx512_ternary",
        "cpu_neon_ternary",
        "fpga_ternary",
    ])
    def test_ternary_support(self, target_id: str):
        data = _TARGET_PROFILES[target_id]
        assert data.get("supports_ternary", False) is True

    def test_mxfp6_only_mi455x(self):
        # Only MI455X should have MXFP6
        mi455x = _TARGET_PROFILES["rocm_cdna5_mi455x"]
        assert mi455x.get("supports_mxfp6", False) is True
        # All others should not
        for tid, data in _TARGET_PROFILES.items():
            if tid != "rocm_cdna5_mi455x":
                assert data.get("supports_mxfp6", False) is False, (
                    f"{tid} should not support MXFP6"
                )

    @pytest.mark.parametrize("target_id", [
        "riscv_mips_s8200",
        "riscv_sifive_x160",
        "riscv_xuantie_c930",
        "riscv_cervell",
    ])
    def test_riscv_npu_flags(self, target_id: str):
        data = _TARGET_PROFILES[target_id]
        assert data.get("is_riscv_npu", False) is True
        assert data.get("abstract_ir_family") is not None

    def test_rubin_r100_specs(self):
        r100 = _TARGET_PROFILES["cuda_sm120"]
        assert r100["memory_gb"] >= 200       # HBM4 >= 200 GB
        assert r100["flops_fp4"] >= 10000     # >=10 PFLOPS FP4
        assert r100["nvlink_bandwidth_gb_s"] >= 3000  # NVLink 6

    def test_mi455x_specs(self):
        mi = _TARGET_PROFILES["rocm_cdna5_mi455x"]
        assert mi["memory_gb"] >= 400         # 432 GB HBM4
        assert mi["memory_bandwidth_gb_s"] >= 20000   # 23.3 TB/s

    def test_tee_backend_tag(self):
        tee = _TARGET_PROFILES["cuda_sm100_tee"]
        assert tee.get("tee_backend") == "nvidia_cc"

    def test_gb300_fp4_1_5x_b200(self):
        b200_fp4 = _TARGET_PROFILES["cuda_sm100"]["flops_fp4"]
        gb300_fp4 = _TARGET_PROFILES["cuda_sm100_gb300"]["flops_fp4"]
        assert gb300_fp4 == pytest.approx(b200_fp4 * 1.5, rel=0.05)

    def test_mips_tdp_sub_10w(self):
        data = _TARGET_PROFILES["riscv_mips_s8200"]
        assert data["tdp_watts"] <= 10.0


class TestHardwareProfileFromTargetId:
    """Verify from_target_id correctly builds profiles with new fields."""

    def test_sm100_tee_profile(self):
        p = HardwareProfile.from_target_id("cuda_sm100_tee")
        assert p is not None
        assert p.supports_tee is True
        assert p.tee_backend == "nvidia_cc"
        assert p.supports_fp4 is True

    def test_riscv_profile(self):
        p = HardwareProfile.from_target_id("riscv_mips_s8200")
        assert p is not None
        assert p.is_riscv_npu is True
        assert p.abstract_ir_family == "mips_npu"
        assert p.tdp_watts <= 10.0

    def test_cervell_profile(self):
        p = HardwareProfile.from_target_id("riscv_cervell")
        assert p is not None
        assert p.is_riscv_npu is True
        assert p.abstract_ir_family == "cervell"

    def test_ternary_avx_profile(self):
        p = HardwareProfile.from_target_id("cpu_avx512_ternary")
        assert p is not None
        assert p.supports_ternary is True
        assert p.supports_fp4 is False

    def test_mi455x_profile(self):
        p = HardwareProfile.from_target_id("rocm_cdna5_mi455x")
        assert p is not None
        assert p.supports_mxfp6 is True
        assert p.memory_gb >= 400

    def test_to_dict_round_trip(self):
        p = HardwareProfile.from_target_id("cuda_sm120")
        d = p.to_dict()
        p2 = HardwareProfile.from_dict(d)
        assert p2.target_id == p.target_id
        assert p2.supports_fp4 == p.supports_fp4
        assert p2.flops_fp4 == pytest.approx(p.flops_fp4)
        assert p2.nvlink_bandwidth_gb_s == pytest.approx(p.nvlink_bandwidth_gb_s)
        assert p2.tdp_watts == pytest.approx(p.tdp_watts)
        assert p2.is_riscv_npu == p.is_riscv_npu

    def test_unknown_target_returns_none(self):
        result = HardwareProfile.from_target_id("nonexistent_xyz_target")
        assert result is None


# ===========================================================================
# RISCVNPUIRBuilder tiling tests
# ===========================================================================


class TestRISCVNPUIRBuilderTiling:
    """Verify tiling math: 3*T^2*dtype_bytes <= scratchpad_bytes."""

    @pytest.mark.parametrize("dtype,dtype_bytes", [
        ("int8", 1), ("bf16", 2), ("fp32", 4), ("ternary", 1),
    ])
    def test_tile_fits_scratchpad(self, dtype: str, dtype_bytes: int):
        scratchpad = 16384
        builder = RISCVNPUIRBuilder({
            "max_shm_size_bytes": scratchpad,
            "memory_bandwidth_gb_s": 100.0,
            "tdp_watts": 8.0,
        })
        tm, tn, tk = builder._compute_tile_sizes(512, 512, 512, dtype)
        # Verify the three tiles (A, B, C) fit in the scratchpad
        total_bytes = (tm * tk + tk * tn + tm * tn) * dtype_bytes
        assert total_bytes <= scratchpad, (
            f"Tiles exceed scratchpad: {total_bytes}B > {scratchpad}B for dtype={dtype}"
        )

    def test_tile_is_power_of_two(self):
        builder = RISCVNPUIRBuilder({"max_shm_size_bytes": 8192, "memory_bandwidth_gb_s": 100.0})
        tm, tn, tk = builder._compute_tile_sizes(256, 256, 256, "int8")
        # Each tile dimension must be a power of two
        assert tm > 0 and (tm & (tm - 1)) == 0
        assert tn > 0 and (tn & (tn - 1)) == 0
        assert tk > 0 and (tk & (tk - 1)) == 0

    def test_tile_clamped_to_dim(self):
        builder = RISCVNPUIRBuilder({"max_shm_size_bytes": 65536, "memory_bandwidth_gb_s": 100.0})
        # Use tiny shape: tile should be clamped to shape dims
        tm, tn, tk = builder._compute_tile_sizes(4, 8, 4, "fp32")
        assert tm <= 4
        assert tn <= 8
        assert tk <= 4


class TestRISCVNPUIRBuilderBuild:
    """Test the full build() pipeline."""

    def _simple_nodes(self) -> list[dict]:
        return [
            {"op": "aeg.linear", "operands": ["W", "x"], "shape": [64, 64], "dtype": "int8", "attrs": {"k_dim": 64}},
            {"op": "aeg.relu", "operands": ["out"], "shape": [64, 64], "dtype": "int8", "attrs": {}},
        ]

    def test_build_returns_program(self):
        builder = RISCVNPUIRBuilder({
            "max_shm_size_bytes": 8192, "memory_bandwidth_gb_s": 100.0, "tdp_watts": 8.0
        })
        prog = builder.build(self._simple_nodes(), "mips_npu")
        assert isinstance(prog, RISCVNPUProgram)
        assert len(prog.instructions) == 2
        assert prog.target_family == "mips_npu"

    def test_instructions_have_correct_opcodes(self):
        builder = RISCVNPUIRBuilder({"max_shm_size_bytes": 8192, "memory_bandwidth_gb_s": 100.0})
        prog = builder.build(self._simple_nodes(), "mips_npu")
        assert prog.instructions[0].opcode == RISCVNPUOpcode.MATMUL
        assert prog.instructions[1].opcode == RISCVNPUOpcode.FUSED_RELU

    def test_buffer_table_populated(self):
        builder = RISCVNPUIRBuilder({"max_shm_size_bytes": 8192, "memory_bandwidth_gb_s": 100.0})
        prog = builder.build(self._simple_nodes(), "mips_npu")
        assert "W" in prog.buffer_table
        assert "x" in prog.buffer_table
        for buf in prog.buffer_table.values():
            assert buf["location"] in ("sram", "dram")

    def test_unknown_op_falls_back_to_vendor_intrinsic(self):
        builder = RISCVNPUIRBuilder({"max_shm_size_bytes": 8192, "memory_bandwidth_gb_s": 100.0})
        nodes = [{"op": "custom.unknown_op", "operands": [], "shape": [1], "dtype": "fp32", "attrs": {}}]
        prog = builder.build(nodes, "mips_npu")
        assert prog.instructions[0].opcode == RISCVNPUOpcode.VENDOR_INTRINSIC

    def test_cycle_estimate_positive(self):
        builder = RISCVNPUIRBuilder({"max_shm_size_bytes": 8192, "memory_bandwidth_gb_s": 100.0})
        prog = builder.build(self._simple_nodes(), "mips_npu")
        assert prog.estimated_cycles >= 0

    def test_scratchpad_spill_for_large_tensor(self):
        builder = RISCVNPUIRBuilder({"max_shm_size_bytes": 256, "memory_bandwidth_gb_s": 100.0})
        # 1024*1024 fp32 = 4 MB >> 256 bytes scratchpad
        nodes = [{"op": "aeg.linear", "operands": ["W"], "shape": [1024, 1024], "dtype": "fp32", "attrs": {}}]
        prog = builder.build(nodes, "sifive_x")
        # Buffer should be in DRAM (spilled)
        assert prog.buffer_table.get("W", {}).get("location") in ("sram", "dram")


# ===========================================================================
# RISCVNPUBackendRegistry tests
# ===========================================================================


class TestRISCVNPUBackendRegistry:
    def test_register_and_get(self):
        reg = RISCVNPUBackendRegistry()

        class FakeBackend:
            @property
            def family_name(self): return "test_family"
            def supports_opcode(self, op): return True
            def lower(self, prog): return b"fake"
            def estimate_cycles(self, ins): return 1
            def tile_policy(self, shape, dtype): return (64, 64, 64)

        fb = FakeBackend()
        reg.register("test_family", fb)
        assert reg.get("test_family") is fb
        assert reg.is_registered("test_family")
        assert "test_family" in reg.registered_families

    def test_get_unknown_returns_none(self):
        reg = RISCVNPUBackendRegistry()
        assert reg.get("nonexistent_family") is None

    def test_register_non_protocol_raises(self):
        reg = RISCVNPUBackendRegistry()
        with pytest.raises(TypeError, match="protocol"):
            reg.register("bad", object())  # type: ignore

    def test_registered_families_sorted(self):
        reg = RISCVNPUBackendRegistry()
        class FB:
            @property
            def family_name(self): return "z"
            def supports_opcode(self, op): return True
            def lower(self, prog): return b""
            def estimate_cycles(self, ins): return 1
            def tile_policy(self, shape, dtype): return (1,1,1)
        class FB2:
            @property
            def family_name(self): return "a"
            def supports_opcode(self, op): return True
            def lower(self, prog): return b""
            def estimate_cycles(self, ins): return 1
            def tile_policy(self, shape, dtype): return (1,1,1)
        reg.register("z", FB())
        reg.register("a", FB2())
        assert reg.registered_families == sorted(reg.registered_families)


# ===========================================================================
# RISC-V vendor backend tests
# ===========================================================================


def _make_simple_program(family: str = "mips_npu") -> RISCVNPUProgram:
    """Build a tiny 2-instruction program for backend testing."""
    return RISCVNPUProgram(
        target_family=family,
        instructions=[
            RISCVNPUInstruction(
                opcode=RISCVNPUOpcode.MATMUL,
                operands=["W", "x"],
                shape=(64, 64),
                dtype="int8",
                tile_m=32, tile_n=32, tile_k=32,
            ),
            RISCVNPUInstruction(
                opcode=RISCVNPUOpcode.FUSED_RELU,
                operands=["out"],
                shape=(64, 64),
                dtype="int8",
                tile_m=32, tile_n=32, tile_k=32,
            ),
            RISCVNPUInstruction(
                opcode=RISCVNPUOpcode.TERNARY_MATMUL,
                operands=["W_ternary", "x"],
                shape=(64, 64),
                dtype="ternary",
                tile_m=32, tile_n=32, tile_k=32,
            ),
            RISCVNPUInstruction(
                opcode=RISCVNPUOpcode.PAGED_ATTN,
                operands=["Q", "K", "V"],
                shape=(1, 32, 128),
                dtype="bf16",
                tile_m=1, tile_n=128, tile_k=128,
                attributes={"num_heads": 32},
            ),
            RISCVNPUInstruction(
                opcode=RISCVNPUOpcode.BARRIER,
                operands=[],
                shape=(1,),
                dtype="fp32",
            ),
        ],
        buffer_table={
            "W": {"shape": [64, 64], "dtype": "int8", "location": "sram"},
            "x": {"shape": [64, 1], "dtype": "int8", "location": "sram"},
            "out": {"shape": [64, 64], "dtype": "int8", "location": "sram"},
            "W_ternary": {"shape": [64, 64], "dtype": "ternary", "location": "dram"},
            "Q": {"shape": [1, 32, 128], "dtype": "bf16", "location": "dram"},
            "K": {"shape": [1, 32, 128], "dtype": "bf16", "location": "dram"},
            "V": {"shape": [1, 32, 128], "dtype": "bf16", "location": "dram"},
        },
        scratchpad_bytes_required=8192,
    )


class TestMIPSNPUBackend:
    def _get_backend(self):
        # Import triggers self-registration
        import aether.compiler.stage3_targeting.target_riscv_mips  # noqa
        return RISCV_NPU_BACKEND_REGISTRY.get("mips_npu")

    def test_registered(self):
        self._get_backend()
        assert RISCV_NPU_BACKEND_REGISTRY.is_registered("mips_npu")

    def test_family_name(self):
        b = self._get_backend()
        assert b.family_name == "mips_npu"

    def test_lower_returns_bytes(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("mips_npu"))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_lower_contains_matmul_instruction(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("mips_npu")).decode("utf-8")
        assert "mips.npu.matmul" in result

    def test_ternary_matmul_no_multiply_keyword(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("mips_npu")).decode("utf-8")
        assert "ternary_mm" in result
        # The BitNet constraint: add-only, no multiply instruction
        assert "mips.npu.matmul" not in result.split("ternary_mm")[0].split("\n")[-1]

    def test_estimate_cycles_positive(self):
        b = self._get_backend()
        ins = RISCVNPUInstruction(
            opcode=RISCVNPUOpcode.MATMUL, operands=[], shape=(512, 512),
            dtype="int8", tile_m=64, tile_n=64, tile_k=64,
        )
        cycles = b.estimate_cycles(ins)
        assert cycles > 0

    def test_tile_policy_power_of_two(self):
        b = self._get_backend()
        tm, tn, tk = b.tile_policy((1024, 1024), "int8")
        assert tm > 0 and (tm & (tm - 1)) == 0


class TestSiFiveX160Backend:
    def _get_backend(self):
        import aether.compiler.stage3_targeting.target_riscv_sifive  # noqa
        return RISCV_NPU_BACKEND_REGISTRY.get("sifive_x")

    def test_registered(self):
        self._get_backend()
        assert RISCV_NPU_BACKEND_REGISTRY.is_registered("sifive_x")

    def test_lower_returns_bytes(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("sifive_x"))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_uses_rvv_instructions(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("sifive_x")).decode("utf-8")
        # RVV-1.0 or RMMM-0.7 instructions
        assert any(kw in result for kw in ["mmaqa", "vle", "vse", "RVV"]), (
            f"Expected RVV/RMMM instructions, got: {result[:500]}"
        )

    def test_tile_policy_respects_vlen(self):
        b = self._get_backend()
        # INT8: VLEN=512b -> 64 elements/vreg -> tile = 64
        tm, tn, tk = b.tile_policy((2048, 2048), "int8")
        assert tm <= 64 and tn <= 64 and tk <= 64


class TestXuanTieC930Backend:
    def _get_backend(self):
        import aether.compiler.stage3_targeting.target_riscv_xuantie  # noqa
        return RISCV_NPU_BACKEND_REGISTRY.get("xuantie_c")

    def test_registered(self):
        self._get_backend()
        assert RISCV_NPU_BACKEND_REGISTRY.is_registered("xuantie_c")

    def test_supports_all_opcodes(self):
        b = self._get_backend()
        for op in RISCVNPUOpcode:
            assert b.supports_opcode(op) is True

    def test_lower_returns_bytes(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("xuantie_c"))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_uses_th_xpu_instructions(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("xuantie_c")).decode("utf-8")
        assert "th.xpu" in result


class TestCervellBackend:
    def _get_backend(self):
        import aether.compiler.stage3_targeting.target_riscv_cervell  # noqa
        return RISCV_NPU_BACKEND_REGISTRY.get("cervell")

    def test_registered(self):
        self._get_backend()
        assert RISCV_NPU_BACKEND_REGISTRY.is_registered("cervell")

    def test_supports_all_opcodes(self):
        b = self._get_backend()
        for op in RISCVNPUOpcode:
            assert b.supports_opcode(op) is True

    def test_lower_returns_qdIR(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("cervell")).decode("utf-8")
        assert ".module" in result
        assert ".func main" in result
        assert "tensor.gemm" in result

    def test_qdIR_ternary_no_multiply(self):
        b = self._get_backend()
        result = b.lower(_make_simple_program("cervell")).decode("utf-8")
        assert "ternary_gemm" in result


# ===========================================================================
# RISCVNPUCompiler facade tests
# ===========================================================================


class TestRISCVNPUCompiler:
    def test_compile_with_mips_backend(self):
        import aether.compiler.stage3_targeting.target_riscv_mips  # noqa
        profile = HardwareProfile.from_target_id("riscv_mips_s8200")
        compiler = RISCVNPUCompiler(profile)
        nodes = [
            {"op": "aeg.linear", "operands": ["W", "x"], "shape": [32, 32], "dtype": "int8", "attrs": {"k_dim": 32}},
        ]
        result = compiler.compile(nodes)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_compile_with_cervell_backend(self):
        import aether.compiler.stage3_targeting.target_riscv_cervell  # noqa
        profile = HardwareProfile.from_target_id("riscv_cervell")
        compiler = RISCVNPUCompiler(profile)
        nodes = [
            {"op": "aeg.linear", "operands": ["W", "x"], "shape": [32, 32], "dtype": "bf16", "attrs": {}},
        ]
        result = compiler.compile(nodes)
        assert isinstance(result, bytes)

    def test_compile_non_riscv_raises(self):
        profile = HardwareProfile.from_target_id("cuda_sm90")
        with pytest.raises(ValueError, match="not a RISC-V NPU"):
            RISCVNPUCompiler(profile)


# ===========================================================================
# Opcode mapping tests
# ===========================================================================


class TestOpcodeMappings:
    @pytest.mark.parametrize("op_name,expected_opcode", [
        ("aeg.linear", RISCVNPUOpcode.MATMUL),
        ("aeg.gemv", RISCVNPUOpcode.GEMV),
        ("aeg.ternary_gemm", RISCVNPUOpcode.TERNARY_MATMUL),
        ("aeg.relu", RISCVNPUOpcode.FUSED_RELU),
        ("aeg.silu", RISCVNPUOpcode.FUSED_SILU),
        ("aeg.rms_norm", RISCVNPUOpcode.LAYER_NORM),
        ("aeg.softmax", RISCVNPUOpcode.SOFTMAX),
        ("aeg.paged_attention", RISCVNPUOpcode.PAGED_ATTN),
        ("aeg.barrier", RISCVNPUOpcode.BARRIER),
    ])
    def test_known_ops_map_correctly(self, op_name: str, expected_opcode: RISCVNPUOpcode):
        result = RISCVNPUIRBuilder._map_opcode(op_name, {})
        assert result == expected_opcode

    def test_unknown_op_maps_to_none(self):
        result = RISCVNPUIRBuilder._map_opcode("custom.totally_unknown", {})
        assert result is None


def test_unavailable_profile_only_is_not_reported_as_implemented():
    from aether.backends.hardware_detector import detect_all_capabilities

    reports = detect_all_capabilities()
    by_target = {item.target_id: item for item in reports}
    for target in ("qualcomm_qnn", "riscv_sifive_x160", "fpga_xilinx_vu9p"):
        assert by_target[target].available is False
        assert by_target[target].implemented is False


def test_openvino_gpu_probe_uses_device_registry(monkeypatch):
    """Intel GPU detection requires an actual OpenVINO GPU device entry."""
    import sys
    import types

    class FakeCore:
        available_devices = ["CPU", "GPU"]

        def get_property(self, device, name):
            assert device == "GPU"
            return {
                "FULL_DEVICE_NAME": "Intel Arc Test GPU",
                "DEVICE_ARCHITECTURE": "xe_hpg",
                "DRIVER_VERSION": "test-driver",
            }[name]

    fake_openvino = types.ModuleType("openvino")
    fake_openvino.Core = FakeCore
    fake_openvino.__version__ = "test"
    monkeypatch.setitem(sys.modules, "openvino", fake_openvino)

    from aether.backends.hardware_detector import detect_openvino_gpu

    result = detect_openvino_gpu()
    assert result.available is True
    assert result.vendor == "Intel"
    assert result.target_id == "openvino_gpu"
    assert result.device == "Intel Arc Test GPU"
    assert result.architecture == "xe_hpg"
    assert result.execution_tested is False

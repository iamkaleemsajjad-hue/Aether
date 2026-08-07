"""
RISC-V NPU Abstract IR Layer - Aether Stage 3 targeting.

This module implements the Abstract IR layer that sits between the architecture-
independent AEG-IR (optimizer output) and the vendor-specific RISC-V NPU backends.

The layer addresses ISA fragmentation across RISC-V NPU vendors (PRD Section 3.2)
by defining a shared set of abstract IR opcodes that every vendor backend can lower
into its own ISA. The four supported families are:

    mips_npu   -- MIPS S8200 (sub-10W agentic edge, RISC-V)
    sifive_x   -- SiFive Intelligence X160 (scalar+vector+matrix)
    xuantie_c  -- Alibaba XuanTie C930 (RISC-V + integrated NPU)
    cervell    -- Semidynamics Cervell (unified scalar/vector/tensor)

Research basis:
    - PRD Section 3.2 RISC-V NPU Abstract IR
    - MIPS S8200 product brief 2026
    - SiFive X160 product brief 2026 (RVV-1.0, RMMM-0.7 matrix extension)
    - XuanTie C930 datasheet 2026 (RVV-1.0 + integrated NPU co-processor)
    - Semidynamics Cervell brief 2026 (Quadric DevStudio toolchain compatible)
    - Quadric DevStudio RISC-V AI compiler toolchain 2026
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Protocol, runtime_checkable

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class RISCVNPUOpcode(str, enum.Enum):
    """Abstract IR opcodes shared by all RISC-V NPU backends."""

    MATMUL = "matmul"
    GEMV = "gemv"
    TERNARY_MATMUL = "ternary_matmul"
    ELEMENTWISE_ADD = "elementwise_add"
    ELEMENTWISE_MUL = "elementwise_mul"
    FUSED_RELU = "fused_relu"
    FUSED_SILU = "fused_silu"
    LAYER_NORM = "layer_norm"
    SOFTMAX = "softmax"
    LOAD_TILE = "load_tile"
    STORE_TILE = "store_tile"
    BARRIER = "barrier"
    PAGED_ATTN = "paged_attn"
    DOT_PRODUCT_ATTN = "dot_product_attn"
    DEQUANT_INT8 = "dequant_int8"
    QUANT_INT8 = "quant_int8"
    LOOP_BEGIN = "loop_begin"
    LOOP_END = "loop_end"
    CONDITIONAL = "conditional"
    VENDOR_INTRINSIC = "vendor_intrinsic"


@dataclasses.dataclass
class RISCVNPUInstruction:
    """A single abstract IR instruction for a RISC-V NPU target."""

    opcode: RISCVNPUOpcode
    operands: list[str]
    shape: tuple[int, ...]
    dtype: str
    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 64
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RISCVNPUProgram:
    """A compiled RISC-V NPU program in abstract IR form."""

    target_family: str
    instructions: list[RISCVNPUInstruction]
    buffer_table: dict[str, dict[str, Any]]
    scratchpad_bytes_required: int = 0
    estimated_cycles: int = 0


@runtime_checkable
class AetherRISCVNPUBackend(Protocol):
    """Protocol that every RISC-V NPU vendor plugin must implement."""

    @property
    def family_name(self) -> str: ...

    def supports_opcode(self, opcode: RISCVNPUOpcode) -> bool: ...

    def lower(self, program: RISCVNPUProgram) -> bytes: ...

    def estimate_cycles(self, instruction: RISCVNPUInstruction) -> int: ...

    def tile_policy(self, shape: tuple[int, ...], dtype: str) -> tuple[int, int, int]: ...


class RISCVNPUBackendRegistry:
    """Registry of RISC-V NPU vendor backends."""

    def __init__(self) -> None:
        self._backends: dict[str, AetherRISCVNPUBackend] = {}

    def register(self, family: str, backend: AetherRISCVNPUBackend) -> None:
        if not isinstance(backend, AetherRISCVNPUBackend):
            msg = f"Backend for family {family!r} does not implement AetherRISCVNPUBackend protocol"
            raise TypeError(msg)
        self._backends[family] = backend
        logger.debug(f"Registered RISC-V NPU backend for family {family!r}: {type(backend).__name__}")

    def get(self, family: str) -> AetherRISCVNPUBackend | None:
        return self._backends.get(family)

    def is_registered(self, family: str) -> bool:
        return family in self._backends

    @property
    def registered_families(self) -> list[str]:
        return sorted(self._backends.keys())


RISCV_NPU_BACKEND_REGISTRY: RISCVNPUBackendRegistry = RISCVNPUBackendRegistry()


class RISCVNPUIRBuilder:
    """Translates AEG-IR graph nodes into a RISCVNPUProgram (abstract IR).

    Tiling strategy: choose tile_m, tile_n, tile_k such that the A-tile,
    B-tile, and C-tile all fit in the SRAM scratchpad simultaneously.
    Tile condition: dtype_bytes * (tm*tk + tk*tn + tm*tn) <= scratchpad_bytes.
    We solve for the largest power-of-two T satisfying 3*T^2*dtype_bytes <= scratchpad_bytes.

    Research basis:
        - Software-managed scratchpad tiling: RISC-V Vector Extension RVV-1.0
        - Quadric DevStudio tiling heuristics 2026
        - PRD Section 20.3 DVFS hints for energy-efficient tile sizes
    """

    def __init__(self, profile_data: dict[str, Any]) -> None:
        self._scratchpad_bytes = profile_data.get("max_shm_size_bytes", 16384)
        self._memory_bandwidth_gb_s = profile_data.get("memory_bandwidth_gb_s", 100.0)
        self._tdp_watts = profile_data.get("tdp_watts", 10.0)
        self._family = profile_data.get("abstract_ir_family", "unknown")

    def _compute_tile_sizes(self, m: int, n: int, k: int, dtype: str) -> tuple[int, int, int]:
        dtype_bytes = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(dtype, 2)
        max_t = int(math.sqrt(self._scratchpad_bytes / (3 * dtype_bytes)))
        if max_t >= 2:
            max_t = 2 ** int(math.log2(max_t))
        else:
            max_t = 1
        return min(max_t, m), min(max_t, n), min(max_t, k)

    def build(self, aeg_ir_nodes: list[dict[str, Any]], target_family: str) -> RISCVNPUProgram:
        instructions: list[RISCVNPUInstruction] = []
        buffer_table: dict[str, dict[str, Any]] = {}
        total_cycles = 0

        for node in aeg_ir_nodes:
            op_name = node.get("op", "")
            operands = node.get("operands", [])
            shape = tuple(node.get("shape", [1]))
            dtype = node.get("dtype", "fp32")
            attrs = node.get("attrs", {})

            opcode = self._map_opcode(op_name, attrs)
            if opcode is None:
                logger.warning(f"RISC-V NPU IR: no mapping for op {op_name!r}; using VENDOR_INTRINSIC")
                opcode = RISCVNPUOpcode.VENDOR_INTRINSIC

            if len(shape) >= 2:
                m, n = shape[-2], shape[-1]
                k = attrs.get("k_dim", n)
                tm, tn, tk = self._compute_tile_sizes(m, n, k, dtype)
            else:
                tm, tn, tk = 1, 1, 1

            for operand in operands:
                if operand not in buffer_table:
                    dtype_bytes = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(dtype, 2)
                    tensor_bytes = math.prod(shape) * dtype_bytes
                    location = "sram" if tensor_bytes <= self._scratchpad_bytes // 4 else "dram"
                    buffer_table[operand] = {"shape": list(shape), "dtype": dtype, "location": location}

            instructions.append(RISCVNPUInstruction(
                opcode=opcode, operands=operands, shape=shape,
                dtype=dtype, tile_m=tm, tile_n=tn, tile_k=tk, attributes=attrs,
            ))

            if len(shape) >= 2:
                dtype_bytes = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(dtype, 2)
                tile_bytes = (tm * tk + tk * tn + tm * tn) * dtype_bytes
                bw_bytes_per_ns = self._memory_bandwidth_gb_s
                cycles_per_tile = max(1, int(tile_bytes / bw_bytes_per_ns))
                num_tiles = max(1, math.prod(shape) // (tm * tn))
                total_cycles += cycles_per_tile * num_tiles

        max_scratchpad = 0
        for buf in buffer_table.values():
            if buf["location"] == "sram":
                dtype_bytes = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(buf["dtype"], 2)
                sz = math.prod(buf["shape"]) * dtype_bytes
                max_scratchpad = max(max_scratchpad, sz)

        if max_scratchpad > self._scratchpad_bytes:
            logger.warning(
                f"RISC-V NPU IR: scratchpad requirement {max_scratchpad}B exceeds "
                f"hardware limit {self._scratchpad_bytes}B; spilling to DRAM."
            )
            for buf in buffer_table.values():
                if buf["location"] == "sram":
                    dtype_bytes = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(buf["dtype"], 2)
                    if math.prod(buf["shape"]) * dtype_bytes > self._scratchpad_bytes:
                        buf["location"] = "dram"

        program = RISCVNPUProgram(
            target_family=target_family,
            instructions=instructions,
            buffer_table=buffer_table,
            scratchpad_bytes_required=min(max_scratchpad, self._scratchpad_bytes),
            estimated_cycles=total_cycles,
        )
        logger.info(
            f"RISC-V NPU IR build complete: {len(instructions)} instructions, "
            f"~{total_cycles:,} estimated cycles, {len(buffer_table)} buffers"
        )
        return program

    @staticmethod
    def _map_opcode(op_name: str, attrs: dict[str, Any]) -> RISCVNPUOpcode | None:
        _OPCODE_MAP: dict[str, RISCVNPUOpcode] = {
            "aeg.linear": RISCVNPUOpcode.MATMUL,
            "aeg.gemv": RISCVNPUOpcode.GEMV,
            "aeg.ternary_gemm": RISCVNPUOpcode.TERNARY_MATMUL,
            "aeg.add": RISCVNPUOpcode.ELEMENTWISE_ADD,
            "aeg.mul": RISCVNPUOpcode.ELEMENTWISE_MUL,
            "aeg.relu": RISCVNPUOpcode.FUSED_RELU,
            "aeg.silu": RISCVNPUOpcode.FUSED_SILU,
            "aeg.rms_norm": RISCVNPUOpcode.LAYER_NORM,
            "aeg.layer_norm": RISCVNPUOpcode.LAYER_NORM,
            "aeg.softmax": RISCVNPUOpcode.SOFTMAX,
            "aeg.paged_attention": RISCVNPUOpcode.PAGED_ATTN,
            "aeg.dot_product_attn": RISCVNPUOpcode.DOT_PRODUCT_ATTN,
            "aeg.dequantize": RISCVNPUOpcode.DEQUANT_INT8,
            "aeg.quantize": RISCVNPUOpcode.QUANT_INT8,
            "aeg.load": RISCVNPUOpcode.LOAD_TILE,
            "aeg.store": RISCVNPUOpcode.STORE_TILE,
            "aeg.barrier": RISCVNPUOpcode.BARRIER,
        }
        return _OPCODE_MAP.get(op_name)


class RISCVNPUCompiler:
    """High-level facade that orchestrates the abstract IR pipeline."""

    def __init__(self, profile: Any) -> None:
        if not profile.is_riscv_npu:
            msg = f"Profile {profile.target_id!r} is not a RISC-V NPU target"
            raise ValueError(msg)
        self._profile = profile
        self._family = profile.abstract_ir_family or "unknown"
        self._builder = RISCVNPUIRBuilder(profile.to_dict())

    def compile(self, aeg_ir_nodes: list[dict[str, Any]]) -> bytes:
        """Compile AEG-IR nodes to RISC-V NPU binary via the abstract IR layer."""
        program = self._builder.build(aeg_ir_nodes, self._family)

        backend = RISCV_NPU_BACKEND_REGISTRY.get(self._family)
        if backend is None:
            backend = self._try_import_backend(self._family)

        if backend is None:
            msg = (
                f"No RISC-V NPU backend registered for family {self._family!r}. "
                f"Registered: {RISCV_NPU_BACKEND_REGISTRY.registered_families}"
            )
            raise RuntimeError(msg)

        logger.info(
            f"Lowering RISC-V NPU abstract IR ({len(program.instructions)} instrs) "
            f"via backend {type(backend).__name__!r}"
        )
        return backend.lower(program)

    @staticmethod
    def _try_import_backend(family: str) -> AetherRISCVNPUBackend | None:
        module_map = {
            "mips_npu": "aether.compiler.stage3_targeting.target_riscv_mips",
            "sifive_x": "aether.compiler.stage3_targeting.target_riscv_sifive",
            "xuantie_c": "aether.compiler.stage3_targeting.target_riscv_xuantie",
            "cervell": "aether.compiler.stage3_targeting.target_riscv_cervell",
        }
        module_path = module_map.get(family)
        if module_path is None:
            return None
        try:
            import importlib
            importlib.import_module(module_path)
            return RISCV_NPU_BACKEND_REGISTRY.get(family)
        except ImportError as e:
            logger.warning(f"Could not import RISC-V NPU backend for {family!r}: {e}")
            return None

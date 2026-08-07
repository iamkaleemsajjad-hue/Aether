from __future__ import annotations
import math
from aether.compiler.stage3_targeting.riscv_npu_ir import (
    RISCV_NPU_BACKEND_REGISTRY, RISCVNPUOpcode, RISCVNPUInstruction, RISCVNPUProgram,
)
from aether.utils.logging import get_logger
logger = get_logger(__name__)

# MIPS S8200 product brief 2026: 64 TOPS INT8 | sub-10W | RISC-V ISA
# PRD Section 3.2 RISC-V NPU Abstract IR | PRD Section B.11
_TOPS = 64.0
_CLK = 1.0
_OPC = _TOPS * 1e12 / (_CLK * 1e9)   # ops-per-cycle

_SUPPORTED = frozenset({
    RISCVNPUOpcode.MATMUL, RISCVNPUOpcode.GEMV, RISCVNPUOpcode.TERNARY_MATMUL,
    RISCVNPUOpcode.ELEMENTWISE_ADD, RISCVNPUOpcode.ELEMENTWISE_MUL,
    RISCVNPUOpcode.FUSED_RELU, RISCVNPUOpcode.FUSED_SILU, RISCVNPUOpcode.LAYER_NORM,
    RISCVNPUOpcode.SOFTMAX, RISCVNPUOpcode.LOAD_TILE, RISCVNPUOpcode.STORE_TILE,
    RISCVNPUOpcode.BARRIER, RISCVNPUOpcode.PAGED_ATTN,
    RISCVNPUOpcode.DEQUANT_INT8, RISCVNPUOpcode.QUANT_INT8,
    RISCVNPUOpcode.LOOP_BEGIN, RISCVNPUOpcode.LOOP_END,
})


class MIPSNPUBackend:
    @property
    def family_name(self) -> str:
        return "mips_npu"

    def supports_opcode(self, op: RISCVNPUOpcode) -> bool:
        return op in _SUPPORTED

    def tile_policy(self, shape: tuple, dtype: str) -> tuple:
        # S8200: 8 MB total / 4 cores = 2 MB per core; fit A+B+C tiles
        db = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(dtype, 2)
        max_t = int(math.sqrt(2 * 1024 * 1024 // (3 * db)))
        max_t = 2 ** int(math.log2(max(1, max_t)))
        if len(shape) >= 2:
            return min(max_t, shape[-2]), min(max_t, shape[-1]), min(max_t, shape[-1])
        return 64, 64, 64

    def estimate_cycles(self, ins: RISCVNPUInstruction) -> int:
        if len(ins.shape) < 2:
            return 1
        ds = {"int8": 1.0, "bf16": 0.5, "fp32": 0.25, "ternary": 1.2}.get(ins.dtype, 0.5)
        return max(1, int(2 * math.prod(ins.shape) * ds / _OPC))

    def lower(self, prog: RISCVNPUProgram) -> bytes:
        out = [
            "# Aether MIPS S8200 NPU Assembly (RV32IM + MIPS.NPU extension)",
            "# Research: MIPS S8200 product brief 2026; PRD Section 3.2",
            f"# {len(prog.instructions)} instructions | {prog.scratchpad_bytes_required}B scratchpad",
            ".section .npu_text", ".align 4", ".global _npu_main", "_npu_main:",
        ]
        sp, ba = 0, {}
        for n, b in prog.buffer_table.items():
            if b["location"] == "sram":
                ba[n] = sp
                db = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(b["dtype"], 2)
                sp += math.prod(b["shape"]) * db
            else:
                ba[n] = -1
        for i, ins in enumerate(prog.instructions):
            out.append(f"  # {i}: {ins.opcode.value}")
            out.extend(f"  {ln}" for ln in self._asm(ins, ba))
        out.append("  ret")
        return "\n".join(out).encode("utf-8")

    def _asm(self, ins: RISCVNPUInstruction, ba: dict) -> list:
        op = ins.opcode
        M, N, K = ins.tile_m, ins.tile_n, ins.tile_k
        if op == RISCVNPUOpcode.LOAD_TILE:
            s = ins.operands[0] if ins.operands else "src"
            return [f"mips.npu.dma_load t0, {ba.get(s, 0)}, {M * K}"]
        if op == RISCVNPUOpcode.STORE_TILE:
            d = ins.operands[0] if ins.operands else "dst"
            return [f"mips.npu.dma_store {ba.get(d, 0)}, t0, {M * N}"]
        if op in (RISCVNPUOpcode.MATMUL, RISCVNPUOpcode.GEMV):
            df = {"int8": "i8", "bf16": "bf16", "fp32": "f32", "ternary": "tern"}.get(ins.dtype, "i8")
            return [f"mips.npu.matmul.{df} t0, t1, t2, {M}, {N}, {K}"]
        if op == RISCVNPUOpcode.TERNARY_MATMUL:
            # BitNet b1.58 {-1,0,+1}: add-only, no multiply — Microsoft Research 2024/2026
            return ["# BitNet b1.58: add-only, no multiply instruction",
                    f"mips.npu.ternary_mm t0, t1, t2, {M}, {N}, {K}"]
        simple = {
            RISCVNPUOpcode.FUSED_RELU:   "mips.npu.relu t0, t0",
            RISCVNPUOpcode.FUSED_SILU:   "mips.npu.silu t0, t0",
            RISCVNPUOpcode.LAYER_NORM:   "mips.npu.rmsnorm t0, t0",
            RISCVNPUOpcode.BARRIER:      "mips.npu.sync",
            RISCVNPUOpcode.DEQUANT_INT8: "mips.npu.dequant.i8 t0, t1, t2",
            RISCVNPUOpcode.QUANT_INT8:   "mips.npu.quant.i8 t0, t1, t2",
        }
        if op in simple:
            return [simple[op]]
        if op == RISCVNPUOpcode.SOFTMAX:
            return [f"mips.npu.softmax t0, t0, {ins.shape[-1]}"]
        if op == RISCVNPUOpcode.PAGED_ATTN:
            return [f"mips.npu.paged_attn a0, a1, a2, {ins.attributes.get('num_heads', 1)}"]
        return [f"call __aether_scalar_{op.value}"]


RISCV_NPU_BACKEND_REGISTRY.register("mips_npu", MIPSNPUBackend())
logger.info("MIPS S8200 RISC-V NPU backend registered")

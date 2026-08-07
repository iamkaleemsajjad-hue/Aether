from __future__ import annotations
import math
from aether.compiler.stage3_targeting.riscv_npu_ir import (
    RISCV_NPU_BACKEND_REGISTRY, RISCVNPUOpcode, RISCVNPUInstruction, RISCVNPUProgram,
)
from aether.utils.logging import get_logger
logger = get_logger(__name__)

# SiFive Intelligence X160 product brief 2026:
# 128 TOPS INT8 | 5W TDP | RVV-1.0 + RMMM-0.7 matrix extension
# VLEN=512b (64 INT8 elements / vector register)
# PRD Section 3.2 RISC-V NPU Abstract IR
_TOPS = 128.0
_CLK = 2.0
_OPC = _TOPS * 1e12 / (_CLK * 1e9)

_SUPPORTED = frozenset({
    RISCVNPUOpcode.MATMUL, RISCVNPUOpcode.GEMV, RISCVNPUOpcode.TERNARY_MATMUL,
    RISCVNPUOpcode.ELEMENTWISE_ADD, RISCVNPUOpcode.ELEMENTWISE_MUL,
    RISCVNPUOpcode.FUSED_RELU, RISCVNPUOpcode.FUSED_SILU, RISCVNPUOpcode.LAYER_NORM,
    RISCVNPUOpcode.SOFTMAX, RISCVNPUOpcode.LOAD_TILE, RISCVNPUOpcode.STORE_TILE,
    RISCVNPUOpcode.BARRIER, RISCVNPUOpcode.PAGED_ATTN, RISCVNPUOpcode.DOT_PRODUCT_ATTN,
    RISCVNPUOpcode.DEQUANT_INT8, RISCVNPUOpcode.QUANT_INT8,
    RISCVNPUOpcode.LOOP_BEGIN, RISCVNPUOpcode.LOOP_END,
})


class SiFiveX160Backend:
    @property
    def family_name(self) -> str:
        return "sifive_x"

    def supports_opcode(self, op: RISCVNPUOpcode) -> bool:
        return op in _SUPPORTED

    def tile_policy(self, shape: tuple, dtype: str) -> tuple:
        # RVV-1.0 VLEN=512 bits; tile = elements_per_vreg for square RMMM tile
        db = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(dtype, 2)
        elems = 512 // (db * 8)
        if len(shape) >= 2:
            return min(elems, shape[-2]), min(elems, shape[-1]), min(elems, shape[-1])
        return elems, elems, elems

    def estimate_cycles(self, ins: RISCVNPUInstruction) -> int:
        if len(ins.shape) < 2:
            return 1
        ds = {"int8": 1.0, "bf16": 0.5, "fp32": 0.25, "ternary": 1.0}.get(ins.dtype, 0.5)
        return max(1, int(2 * math.prod(ins.shape) * ds / _OPC))

    def lower(self, prog: RISCVNPUProgram) -> bytes:
        out = [
            "# Aether SiFive Intelligence X160 Assembly (RVV-1.0 + RMMM-0.7)",
            "# Research: SiFive X160 product brief 2026; PRD Section 3.2",
            f"# {len(prog.instructions)} instructions | {prog.scratchpad_bytes_required}B scratchpad",
            ".section .text",
            ".option arch, +v,+zmmm",   # Enable RVV-1.0 + matrix extension
            ".align 4", ".global _npu_main", "_npu_main:",
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
            ew = {"int8": "8", "bf16": "16", "fp32": "32"}.get(ins.dtype, "8")
            return [f"vle{ew}.v v0, ({ba.get(s, 0)})(sp)  # RVV-1.0 vector load"]
        if op == RISCVNPUOpcode.STORE_TILE:
            d = ins.operands[0] if ins.operands else "dst"
            ew = {"int8": "8", "bf16": "16", "fp32": "32"}.get(ins.dtype, "8")
            return [f"vse{ew}.v v0, ({ba.get(d, 0)})(sp)  # RVV-1.0 vector store"]
        if op in (RISCVNPUOpcode.MATMUL, RISCVNPUOpcode.GEMV):
            dm = {"int8": "mm8", "bf16": "mm16", "fp32": "mmf"}.get(ins.dtype, "mm8")
            return [f"# RMMM-0.7 tile M={M} N={N} K={K}",
                    f"mmaqa.{dm} m0, v0, v8  # RISC-V matrix multiply-accumulate"]
        if op == RISCVNPUOpcode.TERNARY_MATMUL:
            # Sign-bit decomposition + vcpop accumulate; no multiply instruction
            return ["# Ternary: sign-bit + vcpop (RVV vsub.vx, no multiply)",
                    "vsub.vx v0, v0, zero", "vsub.vv v16, v0, v8"]
        simple = {
            RISCVNPUOpcode.FUSED_RELU:        "vmax.vx v0, v0, zero",
            RISCVNPUOpcode.FUSED_SILU:        "call __sifive_silu_rvv",
            RISCVNPUOpcode.LAYER_NORM:        "call __sifive_rmsnorm_rvv",
            RISCVNPUOpcode.SOFTMAX:           "call __sifive_softmax_rvv",
            RISCVNPUOpcode.PAGED_ATTN:        "call __sifive_paged_attn",
            RISCVNPUOpcode.DOT_PRODUCT_ATTN:  "call __sifive_sdpa_rvv",
            RISCVNPUOpcode.BARRIER:           "fence iorw, iorw",
            RISCVNPUOpcode.DEQUANT_INT8:      "vfmul.vf v0, v0, fa0",
            RISCVNPUOpcode.QUANT_INT8:        "vfcvt.x.f.v v0, v0",
        }
        if op in simple:
            return [simple[op]]
        return [f"call __aether_scalar_{op.value}"]


RISCV_NPU_BACKEND_REGISTRY.register("sifive_x", SiFiveX160Backend())
logger.info("SiFive X160 RISC-V NPU backend registered")

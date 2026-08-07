from __future__ import annotations
import math
from aether.compiler.stage3_targeting.riscv_npu_ir import (
    RISCV_NPU_BACKEND_REGISTRY, RISCVNPUOpcode, RISCVNPUInstruction, RISCVNPUProgram,
)
from aether.utils.logging import get_logger
logger = get_logger(__name__)

# T-Head XuanTie C930 datasheet 2026:
# 256 TOPS INT8 | 25W TDP | RVV-1.0 + XPU co-processor | FP8 support
# PRD Section 3.2 RISC-V NPU Abstract IR
_TOPS = 256.0
_CLK = 2.5
_OPC = _TOPS * 1e12 / (_CLK * 1e9)


class XuanTieC930Backend:
    @property
    def family_name(self) -> str:
        return "xuantie_c"

    def supports_opcode(self, op: RISCVNPUOpcode) -> bool:
        return True  # Full NPU co-processor supports all opcodes

    def tile_policy(self, shape: tuple, dtype: str) -> tuple:
        # C930 integrated NPU has 32 MB SRAM
        db = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(dtype, 2)
        max_t = int(math.sqrt(32 * 1024 * 1024 // (3 * db)))
        max_t = 2 ** int(math.log2(max(1, max_t)))
        if len(shape) >= 2:
            return min(max_t, shape[-2]), min(max_t, shape[-1]), min(max_t, shape[-1])
        return 128, 128, 128

    def estimate_cycles(self, ins: RISCVNPUInstruction) -> int:
        if len(ins.shape) < 2:
            return 1
        ds = {"int8": 1.0, "bf16": 0.5, "fp32": 0.25, "ternary": 0.9}.get(ins.dtype, 0.5)
        return max(1, int(2 * math.prod(ins.shape) * ds / _OPC))

    def lower(self, prog: RISCVNPUProgram) -> bytes:
        out = [
            "# Aether XuanTie C930 Assembly (RVV-1.0 + XPU co-processor)",
            "# Research: T-Head XuanTie C930 datasheet 2026; PRD Section 3.2",
            f"# {len(prog.instructions)} instructions | {prog.scratchpad_bytes_required}B scratchpad",
            ".section .text", ".option arch, +v", ".align 4",
            ".global _npu_main", "_npu_main:",
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
        if op in (RISCVNPUOpcode.MATMUL, RISCVNPUOpcode.GEMV):
            ds = {"int8": "s8", "bf16": "bf16", "fp32": "f32"}.get(ins.dtype, "s8")
            return [f"# XuanTie XPU GEMM M={M} N={N} K={K}",
                    f"th.xpu.gemm.{ds} x0, x1, x2, {M}, {N}, {K}"]
        if op == RISCVNPUOpcode.TERNARY_MATMUL:
            return ["# XuanTie ternary GEMM (add-only, BitNet b1.58)",
                    f"th.xpu.ternary_gemm x0, x1, x2, {M}, {N}, {K}"]
        simple = {
            RISCVNPUOpcode.FUSED_RELU:   "vmax.vx v0, v0, zero",
            RISCVNPUOpcode.FUSED_SILU:   "th.xpu.silu v0, v0",
            RISCVNPUOpcode.LAYER_NORM:   "th.xpu.rmsnorm v0, v0",
            RISCVNPUOpcode.PAGED_ATTN:   "th.xpu.paged_attn a0, a1, a2",
            RISCVNPUOpcode.BARRIER:      "fence iorw, iorw",
            RISCVNPUOpcode.DEQUANT_INT8: "th.xpu.quant.i8 v0, v0, fa0, fa1",
            RISCVNPUOpcode.QUANT_INT8:   "th.xpu.quant.i8 v0, v0, fa0, fa1",
        }
        if op in simple:
            return [simple[op]]
        if op == RISCVNPUOpcode.SOFTMAX:
            return [f"th.xpu.softmax v0, v0, {ins.shape[-1]}"]
        return [f"call __aether_scalar_{op.value}"]


RISCV_NPU_BACKEND_REGISTRY.register("xuantie_c", XuanTieC930Backend())
logger.info("XuanTie C930 RISC-V NPU backend registered")

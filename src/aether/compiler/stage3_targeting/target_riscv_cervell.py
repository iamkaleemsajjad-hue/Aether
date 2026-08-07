from __future__ import annotations
import math
from aether.compiler.stage3_targeting.riscv_npu_ir import (
    RISCV_NPU_BACKEND_REGISTRY, RISCVNPUOpcode, RISCVNPUInstruction, RISCVNPUProgram,
)
from aether.utils.logging import get_logger
logger = get_logger(__name__)

# Semidynamics Cervell brief 2026:
# 512 TOPS INT8 (est.) | 45W TDP | unified scalar+vector+tensor execution unit
# Quadric DevStudio toolchain compatible (emits qdIR for Cervell native binary)
# PRD Section 3.2 RISC-V NPU Abstract IR; PRD B.11
_TOPS = 512.0
_CLK = 3.0
_OPC = _TOPS * 1e12 / (_CLK * 1e9)


class CervellBackend:
    @property
    def family_name(self) -> str:
        return "cervell"

    def supports_opcode(self, op: RISCVNPUOpcode) -> bool:
        return True  # Unified execution handles all opcodes without co-proc dispatch

    def tile_policy(self, shape: tuple, dtype: str) -> tuple:
        # Cervell: 65536 bytes SRAM per execution cluster
        db = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(dtype, 2)
        max_t = int(math.sqrt(65536 // (3 * db)))
        max_t = 2 ** int(math.log2(max(1, max_t)))
        if len(shape) >= 2:
            return min(max_t, shape[-2]), min(max_t, shape[-1]), min(max_t, shape[-1])
        return max_t, max_t, max_t

    def estimate_cycles(self, ins: RISCVNPUInstruction) -> int:
        if len(ins.shape) < 2:
            return 1
        ds = {"int8": 1.0, "bf16": 0.5, "fp32": 0.25, "ternary": 1.0}.get(ins.dtype, 0.5)
        return max(1, int(2 * math.prod(ins.shape) * ds / _OPC))

    def lower(self, prog: RISCVNPUProgram) -> bytes:
        # Emit Quadric DevStudio IR (qdIR) — DevStudio JIT-compiles to Cervell native
        out = [
            "; Aether Cervell NPU (Quadric DevStudio IR / qdIR)",
            "; Unified scalar+vector+tensor execution — no co-processor dispatch",
            "; Research: Semidynamics Cervell brief 2026; Quadric DevStudio 2026",
            f"; {len(prog.instructions)} instructions | {prog.scratchpad_bytes_required}B scratchpad",
            ".module aether_cervell_module",
            ".target cervell_1.0",
            ".func main() -> void {",
        ]
        sp, bd = 0, {}
        for n, b in prog.buffer_table.items():
            if b["location"] == "sram":
                di = {"int8": "i8", "bf16": "bf16", "fp32": "f32", "ternary": "i2"}.get(b["dtype"], "f32")
                ss = "x".join(str(d) for d in b["shape"])
                out.append(f"  %{n} = alloca [{ss}x{di}], spad+{sp}")
                bd[n] = f"%{n}"
                db = {"int8": 1, "bf16": 2, "fp32": 4, "ternary": 1}.get(b["dtype"], 2)
                sp += math.prod(b["shape"]) * db
            else:
                bd[n] = f"%{n}_dram"
                out.append(f"  %{n}_dram = global_ptr i8* @{n}")
        for i, ins in enumerate(prog.instructions):
            out.append(f"  ; {i}: {ins.opcode.value}")
            out.extend(f"  {ln}" for ln in self._qdIR(ins, bd))
        out += ["}  ; end func main", ".end_module"]
        return "\n".join(out).encode("utf-8")

    def _qdIR(self, ins: RISCVNPUInstruction, bd: dict) -> list:
        op = ins.opcode
        M, N, K = ins.tile_m, ins.tile_n, ins.tile_k
        ops = [bd.get(o, f"%{o}") for o in ins.operands[:3]] + ["%0", "%1", "%2"]
        if op in (RISCVNPUOpcode.MATMUL, RISCVNPUOpcode.GEMV):
            di = {"int8": "i8", "bf16": "bf16", "fp32": "f32"}.get(ins.dtype, "f32")
            return [f"; Cervell tensor GEMM [{M}x{K}]@[{K}x{N}]",
                    f"%r = tensor.gemm.{di} {ops[0]}, {ops[1]}, tile=[{M},{N},{K}]"]
        if op == RISCVNPUOpcode.TERNARY_MATMUL:
            return ["; Cervell ternary GEMM (add-only, BitNet b1.58)",
                    f"%r = tensor.ternary_gemm {ops[0]}, {ops[1]}, tile=[{M},{N},{K}]"]
        qdIR_map = {
            RISCVNPUOpcode.FUSED_RELU:        f"%r = vector.relu {ops[0]}",
            RISCVNPUOpcode.FUSED_SILU:        f"%r = vector.silu {ops[0]}",
            RISCVNPUOpcode.LAYER_NORM:        f"%r = vector.rmsnorm {ops[0]}",
            RISCVNPUOpcode.BARRIER:           "memory.fence",
            RISCVNPUOpcode.LOAD_TILE:         f"%t = memory.load {ops[0]}, [{M},{K}]",
            RISCVNPUOpcode.STORE_TILE:        f"memory.store {ops[0]}, {ops[1]}, [{M},{N}]",
            RISCVNPUOpcode.DEQUANT_INT8:      f"%r = scalar.quant_i8 {ops[0]}, {ops[1]}, {ops[2]}",
            RISCVNPUOpcode.QUANT_INT8:        f"%r = scalar.quant_i8 {ops[0]}, {ops[1]}, {ops[2]}",
        }
        if op in qdIR_map:
            return [qdIR_map[op]]
        if op == RISCVNPUOpcode.SOFTMAX:
            return [f"%r = vector.softmax {ops[0]}, dim={ins.shape[-1]}"]
        if op == RISCVNPUOpcode.PAGED_ATTN:
            return [f"%r = attn.paged {ops[0]}, {ops[1]}, {ops[2]}"]
        if op == RISCVNPUOpcode.DOT_PRODUCT_ATTN:
            return [f"%r = attn.sdpa {ops[0]}, {ops[1]}, {ops[2]}"]
        return [f"call @__aether_scalar_{op.value}({ops[0]})"]


RISCV_NPU_BACKEND_REGISTRY.register("cervell", CervellBackend())
logger.info("Semidynamics Cervell RISC-V NPU backend registered")

"""
AEG-IR: Aether Execution Graph Intermediate Representation.

AEG-IR is Aether's portable, hardware-agnostic IR for transformer-family models.
It is inspired by MLIR but specialized for the tensor operations found in LLMs.
The IR preserves high-level semantics (GQA, SwiGLU, RoPE) through the optimizer
passes and only lowers to target-specific operations in Stage 3.

The IR is organized as a module containing functions, blocks, instructions, and
operands — similar to LLVM IR but with operations at a higher level of
abstraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from aether.core.graph import AEGGraph


class AEGOpCode:
    """AEG-IR operation codes for high-level tensor operations.

    These define the vocabulary of the AEG Intermediate Representation.
    Each op code corresponds to a distinct tensor operation or control structure
    that the Aether compiler recognizes and can optimize across.

    This is a class of string constants, not an enum, to allow the set to be
    extensible by third-party hardware targets and custom passes.
    """

    # ── Data flow ─────────────────────────────────────────────────────────────
    CONSTANT = "aeg.constant"
    PARAMETER = "aeg.parameter"
    TENSOR = "aeg.tensor"

    # ── Normalization ──────────────────────────────────────────────────────────
    RMS_NORM = "aeg.rmsnorm"
    LAYER_NORM = "aeg.layernorm"
    GROUP_NORM = "aeg.groupnorm"
    BATCH_NORM = "aeg.batchnorm"

    # ── Projection and linear ─────────────────────────────────────────────────
    LINEAR = "aeg.linear"
    QKV_PROJ = "aeg.qkv_proj"
    Q_PROJ = "aeg.q_proj"
    K_PROJ = "aeg.k_proj"
    V_PROJ = "aeg.v_proj"
    O_PROJ = "aeg.o_proj"
    GATE_PROJ = "aeg.gate_proj"
    UP_PROJ = "aeg.up_proj"
    DOWN_PROJ = "aeg.down_proj"

    # ── Attention ──────────────────────────────────────────────────────────────
    GQA = "aeg.gqa"
    MLA = "aeg.mla"
    MHA = "aeg.mha"
    CROSS_ATTN = "aeg.cross_attn"
    SELF_ATTN = "aeg.self_attn"
    ATTN_MASK = "aeg.attn_mask"
    FLASH_ATTN = "aeg.flash_attn"

    # ── Positional embeddings ─────────────────────────────────────────────────
    ROPE = "aeg.rope"
    YAWN_ROPE = "aeg.yarn_rope"
    NTK_ROPE = "aeg.ntk_rope"
    ALIBI = "aeg.alibi"
    POSITION_IDS = "aeg.position_ids"

    # ── Feed-forward ──────────────────────────────────────────────────────────
    SWIGLU_FFN = "aeg.swiglu_ffn"
    GEGLU_FFN = "aeg.geglu_ffn"
    GELU_FFN = "aeg.gelu_ffn"
    RELU_FFN = "aeg.relu_ffn"
    FFN = "aeg.ffn"
    SILU = "aeg.silu"
    GELU = "aeg.gelu"
    RELU = "aeg.relu"
    SIGMOID = "aeg.sigmoid"

    # ── Arithmetic ────────────────────────────────────────────────────────────
    ADD = "aeg.add"
    SUB = "aeg.sub"
    MUL = "aeg.mul"
    DIV = "aeg.div"
    MATMUL = "aeg.matmul"
    BMM = "aeg.bmm"
    SOFTMAX = "aeg.softmax"
    LOG_SOFTMAX = "aeg.log_softmax"

    # ── Tensor manipulation ───────────────────────────────────────────────────
    RESHAPE = "aeg.reshape"
    TRANSPOSE = "aeg.transpose"
    PERMUTE = "aeg.permute"
    SLICE = "aeg.slice"
    CONCAT = "aeg.concat"
    SPLIT = "aeg.split"
    EXPAND = "aeg.expand"
    SQUEEZE = "aeg.squeeze"
    PAD = "aeg.pad"
    GATHER = "aeg.gather"
    SCATTER = "aeg.scatter"

    # ── Memory ─────────────────────────────────────────────────────────────────
    KV_CACHE_STORE = "aeg.kv_cache_store"
    KV_CACHE_LOAD = "aeg.kv_cache_load"
    KV_CACHE_TRANSFER = "aeg.kv_cache_transfer"
    KV_CACHE_EVICT = "aeg.kv_cache_evict"

    # ── MoE operations ─────────────────────────────────────────────────────────
    MOE_ROUTER = "aeg.moe_router"
    MOE_DISPATCH = "aeg.moe_dispatch"
    MOE_COMBINE = "aeg.moe_combine"
    EXPERT_FFN = "aeg.expert_ffn"
    TOP_K_ROUTING = "aeg.topk_routing"
    THRESHOLD_ROUTING = "aeg.threshold_routing"

    # ── Quantization ──────────────────────────────────────────────────────────
    QUANTIZE = "aeg.quantize"
    DEQUANTIZE = "aeg.dequantize"
    QUANTIZED_LINEAR = "aeg.quantized_linear"
    QUANTIZED_GEMM = "aeg.quantized_gemm"

    # ── Fused operations ──────────────────────────────────────────────────────
    FUSED_QKV_ROPE_NORM = "aeg.fused_qkv_rope_norm"
    FUSED_ATTN_OUT_PROJ = "aeg.fused_attn_out_proj"
    FUSED_FFN_RESIDUAL = "aeg.fused_ffn_residual"
    FUSED_LAYER = "aeg.fused_layer"

    # ── Control flow ───────────────────────────────────────────────────────────
    COND = "aeg.cond"
    LOOP = "aeg.loop"
    BRANCH = "aeg.branch"
    CALL = "aeg.call"
    RETURN = "aeg.return"

    # ── Embedding ──────────────────────────────────────────────────────────────
    EMBEDDING = "aeg.embedding"
    POSITION_EMBEDDING = "aeg.position_embedding"
    TOKEN_EMBEDDING = "aeg.token_embedding"

    # ── Output ─────────────────────────────────────────────────────────────────
    LM_HEAD = "aeg.lm_head"
    LOGITS = "aeg.logits"
    SOFTMAX_WITH_TEMP = "aeg.softmax_with_temp"

    # ── Vision and multimodal ─────────────────────────────────────────────────
    VIT_ENCODER = "aeg.vit_encoder"
    VIT_PATCH_EMBED = "aeg.vit_patch_embed"
    VIT_POS_EMBED = "aeg.vit_pos_embed"
    CROSS_ATTN_LAYER = "aeg.cross_attn_layer"
    IMAGE_ENCODER = "aeg.image_encoder"
    AUDIO_ENCODER = "aeg.audio_encoder"
    MULTIMODAL_FUSION = "aeg.multimodal_fusion"

    # ── Parallelism and distributed ────────────────────────────────────────────
    ALL_REDUCE = "aeg.all_reduce"
    ALL_GATHER = "aeg.all_gather"
    REDUCE_SCATTER = "aeg.reduce_scatter"
    SEND = "aeg.send"
    RECV = "aeg.recv"
    SHARD = "aeg.shard"
    RESHARD = "aeg.reshard"

    # ── Speculative decoding ───────────────────────────────────────────────────
    DRAFT_VERIFY = "aeg.draft_verify"
    MTP_HEAD = "aeg.mtp_head"
    TREE_ATTN = "aeg.tree_attn"
    TREE_MASK = "aeg.tree_mask"
    ACCEPT_TOKENS = "aeg.accept_tokens"

    ARGUMENT = "aeg.argument"


    @classmethod
    def is_valid(cls, op_code: str) -> bool:
        """Check if a string is a valid AEG op code."""
        return any(op_code == v for v in vars(cls).values() if isinstance(v, str) and v.startswith("aeg."))

    @classmethod
    def known_ops(cls) -> list[str]:
        """Return all known op code strings."""
        return sorted(v for v in vars(cls).values() if isinstance(v, str) and v.startswith("aeg."))

    @classmethod
    def fused_ops(cls) -> list[str]:
        """Return all fused operation codes."""
        return [v for v in cls.known_ops() if v.startswith("aeg.fused_")]

    @classmethod
    def attention_ops(cls) -> list[str]:
        """Return all attention-related operation codes."""
        return [v for v in cls.known_ops() if "attn" in v or v in ("aeg.gqa", "aeg.mla", "aeg.mha")]

    @classmethod
    def norm_ops(cls) -> list[str]:
        """Return all normalization operation codes."""
        return [v for v in cls.known_ops() if "norm" in v]

    @classmethod
    def moe_ops(cls) -> list[str]:
        """Return all MoE operation codes."""
        return [v for v in cls.known_ops() if "moe" in v or "expert" in v]

    @classmethod
    def communication_ops(cls) -> list[str]:
        """Return all distributed communication operation codes."""
        return [v for v in cls.known_ops() if v.startswith(("aeg.all_", "aeg.send", "aeg.recv", "aeg.shard"))]

    @classmethod
    def speculative_ops(cls) -> list[str]:
        """Return all speculative decoding operation codes."""
        return [v for v in cls.known_ops() if "draft" in v or "tree" in v or "accept" in v]


AttributeDict = dict[str, Any]


@dataclass
class AEGVariable:
    """A named variable reference in AEG-IR.

    A variable is a higher-level name that may refer to an operand, parameter,
    or constant. It is used by the public API for naming tensors and model
    weights.
    """

    name: str
    type_str: str = "tensor<*xbf16>"
    attributes: AttributeDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_str,
            "attributes": self.attributes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGVariable:
        return AEGVariable(
            name=data["name"],
            type_str=data.get("type", "tensor<*xbf16>"),
            attributes=dict(data.get("attributes", {})),
        )

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type_str}"


@dataclass
class AEGOperand:
    """An operand in an AEG-IR instruction.

    An operand is a named reference produced by a prior instruction, a function
    parameter, or a constant.
    """

    name: str
    """Unique name within the function scope."""

    type_str: str
    """Type string (e.g., 'tensor<*xbf16>', 'i64', 'tensor<64x128xfp8>')."""

    attributes: AttributeDict = field(default_factory=dict)
    """Attributes of this operand (e.g., shape hints, layout info)."""

    def __post_init__(self) -> None:
        if not self.name:
            msg = "Operand name cannot be empty"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_str,
            "attributes": self.attributes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGOperand:
        return AEGOperand(
            name=data["name"],
            type_str=data.get("type", "tensor<*xbf16>"),
            attributes=dict(data.get("attributes", {})),
        )

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type_str}"


@dataclass
class AEGInstruction:
    """A single instruction in AEG-IR.

    Instructions follow the SSA (Static Single Assignment) form: each
    instruction defines a set of result operands and uses prior operands
    as inputs.
    """

    results: list[AEGOperand]
    """Operands defined by this instruction (may be empty)."""

    op_code: str
    """The operation code, e.g. 'aeg.rmsnorm'."""

    inputs: list[str | AEGOperand] = field(default_factory=list)
    """Names of input operands or operands themselves."""

    attributes: AttributeDict = field(default_factory=dict)
    """Instruction-specific attributes (e.g., eps=1e-6, num_heads=64)."""

    comment: str | None = None
    """Optional human-readable comment."""

    def __post_init__(self) -> None:
        if not self.op_code:
            msg = "Instruction op_code cannot be empty"
            raise ValueError(msg)
        if not isinstance(self.op_code, str) or not self.op_code.startswith("aeg."):
            msg = f"Invalid op code: {self.op_code}; must start with 'aeg.'"
            raise ValueError(msg)

    @property
    def result_names(self) -> list[str]:
        """Return the names of result operands."""
        return [r.name for r in self.results]

    @property
    def input_names(self) -> list[str]:
        """Return the names of input operands."""
        return [i.name if isinstance(i, AEGOperand) else i for i in self.inputs]

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an instruction attribute."""
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an instruction attribute with a default."""
        return self.attributes.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "op_code": self.op_code,
            "inputs": [i.to_dict() if isinstance(i, AEGOperand) else i for i in self.inputs],
            "attributes": self.attributes,
            "comment": self.comment,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGInstruction:
        return AEGInstruction(
            results=[AEGOperand.from_dict(r) for r in data.get("results", [])],
            op_code=data["op_code"],
            inputs=[AEGOperand.from_dict(i) if isinstance(i, dict) else i for i in data.get("inputs", [])],
            attributes=dict(data.get("attributes", {})),
            comment=data.get("comment"),
        )

    def __repr__(self) -> str:
        results_str = ", ".join(r.name for r in self.results) if self.results else ""
        inputs_str = ", ".join(self.input_names)
        if results_str:
            return f"    %{results_str} = {self.op_code}({inputs_str})"
        return f"    {self.op_code}({inputs_str})"


@dataclass
class Block:
    """A basic block in an AEG-IR function.

    Blocks contain a linear sequence of instructions and end with a terminator
    (e.g., return or branch).
    """

    name: str
    """Block label."""

    instructions: list[AEGInstruction] = field(default_factory=list)
    """Instructions in this block."""

    arguments: list[AEGOperand] = field(default_factory=list)
    """Block arguments (for control flow)."""

    @property
    def instruction_count(self) -> int:
        """Return the number of instructions in the block."""
        return len(self.instructions)

    def add_instruction(self, instruction: AEGInstruction) -> AEGInstruction:
        """Add an instruction to the end of this block."""
        self.instructions.append(instruction)
        return instruction

    def insert_instruction(self, index: int, instruction: AEGInstruction) -> None:
        """Insert an instruction at a specific position."""
        self.instructions.insert(index, instruction)

    def remove_instruction(self, index: int) -> None:
        """Remove an instruction by index."""
        self.instructions.pop(index)

    def find_instructions_by_op(self, op_code: str) -> list[AEGInstruction]:
        """Find all instructions with a given op code."""
        return [inst for inst in self.instructions if inst.op_code == op_code]

    def find_instruction(self, result_name: str) -> AEGInstruction | None:
        """Find the instruction that defines a given result name."""
        for inst in self.instructions:
            if result_name in inst.result_names:
                return inst
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": [a.to_dict() for a in self.arguments],
            "instructions": [i.to_dict() for i in self.instructions],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Block:
        return Block(
            name=data["name"],
            arguments=[AEGOperand.from_dict(a) for a in data.get("arguments", [])],
            instructions=[AEGInstruction.from_dict(i) for i in data.get("instructions", [])],
        )

    def __repr__(self) -> str:
        return f"Block('{self.name}', {len(self.instructions)} instructions)"

    def __iter__(self) -> Any:
        return iter(self.instructions)


@dataclass
class Function:
    """A function in AEG-IR, containing one or more blocks.

    A function may represent a single transformer layer, the full model, or
    a sub-computation like the embedding or LM head.
    """

    name: str
    """Function name (e.g., 'transformer_layer', 'embedding', 'model')."""

    parameters: list[AEGOperand] = field(default_factory=list)
    """Input parameters to the function."""

    results: list[AEGOperand] = field(default_factory=list)
    """Return values of the function."""

    blocks: list[Block] = field(default_factory=list)
    """Basic blocks composing the function body."""

    attributes: AttributeDict = field(default_factory=dict)
    """Function-level attributes (e.g., layer_index, precision hints)."""

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def instruction_count(self) -> int:
        return sum(b.instruction_count for b in self.blocks)

    def add_block(self, block: Block) -> Block:
        """Add a block to this function."""
        self.blocks.append(block)
        return block

    def first_block(self) -> Block | None:
        """Return the first (entry) block, or None if the function is empty."""
        return self.blocks[0] if self.blocks else None

    def find_block(self, name: str) -> Block | None:
        """Find a block by name."""
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def find_instructions_by_op(self, op_code: str) -> list[AEGInstruction]:
        """Find all instructions with a given op code across all blocks."""
        results: list[AEGInstruction] = []
        for block in self.blocks:
            results.extend(block.find_instructions_by_op(op_code))
        return results

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
            "results": [r.to_dict() for r in self.results],
            "blocks": [b.to_dict() for b in self.blocks],
            "attributes": self.attributes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Function:
        return Function(
            name=data["name"],
            parameters=[AEGOperand.from_dict(p) for p in data.get("parameters", [])],
            results=[AEGOperand.from_dict(r) for r in data.get("results", [])],
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
            attributes=dict(data.get("attributes", {})),
        )

    def __repr__(self) -> str:
        return f"Function('{self.name}', {len(self.blocks)} blocks, {self.instruction_count} instructions)"


@dataclass
class AEGIRModule:
    """Top-level AEG-IR module containing functions and metadata.

    An AEG-IR module is the serialized form of a compiled model's computation
    graph. It may contain one or more functions (e.g., one per transformer
    layer type, plus the forward pass).
    """

    version: str
    """AEG-IR version string (e.g., 'AEG-IR/1.0')."""

    functions: list[Function] = field(default_factory=list)
    """Functions in this module."""

    metadata: AttributeDict = field(default_factory=dict)
    """Module-level metadata (model name, total params, etc.)."""

    @property
    def function_count(self) -> int:
        return len(self.functions)

    @property
    def instruction_count(self) -> int:
        return sum(f.instruction_count for f in self.functions)

    def add_function(self, function: Function) -> Function:
        """Add a function to this module."""
        self.functions.append(function)
        return function

    def find_function(self, name: str) -> Function | None:
        """Find a function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None

    def find_instructions_by_op(self, op_code: str) -> list[AEGInstruction]:
        """Find all instructions with a given op code across all functions."""
        results: list[AEGInstruction] = []
        for func in self.functions:
            results.extend(func.find_instructions_by_op(op_code))
        return results

    def compute_instruction_count_by_op(self) -> dict[str, int]:
        """Return a histogram of op codes in this module."""
        counts: dict[str, int] = {}
        for func in self.functions:
            for block in func.blocks:
                for inst in block.instructions:
                    counts[inst.op_code] = counts.get(inst.op_code, 0) + 1
        return counts

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "functions": [f.to_dict() for f in self.functions],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_text(self) -> str:
        """Return a human-readable textual representation of the IR.

        This is the format shown by `aether graph <model>`.
        """
        lines: list[str] = []
        lines.append(f"; AEG-IR v{self.version}")
        lines.append(f"; metadata: {json.dumps(self.metadata, default=str)}")
        lines.append("")
        for func in self.functions:
            params_str = ", ".join(str(p) for p in func.parameters)
            results_str = ", ".join(str(r) for r in func.results)
            lines.append(f"func @{func.name}({params_str}) -> ({results_str}) {{")
            for attr_key, attr_val in func.attributes.items():
                lines.append(f"  // @{func.name}.{attr_key} = {attr_val}")
            if func.attributes:
                lines.append("")
            for block in func.blocks:
                if func.block_count > 1:
                    lines.append(f"^{block.name}:")
                for inst in block.instructions:
                    lines.append(str(inst))
            lines.append("}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGIRModule:
        return AEGIRModule(
            version=data.get("version", "AEG-IR/1.0"),
            functions=[Function.from_dict(f) for f in data.get("functions", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @staticmethod
    def from_json(json_str: str) -> AEGIRModule:
        data = json.loads(json_str)
        return AEGIRModule.from_dict(data)

    @staticmethod
    def from_graph(graph: AEGGraph, version: str = "AEG-IR/1.0") -> AEGIRModule:
        """Convert an AEGGraph to an AEGIRModule.

        This is the bridge between Stage 1 (graph extraction) and the
        optimizer passes.
        """
        module = AEGIRModule(
            version=version,
            metadata={
                "name": graph.name,
                "architecture": graph.architecture.family if graph.architecture else "unknown",
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
            },
        )
        # Create a single function for the model forward pass
        func = Function(
            name="model",
            parameters=[
                AEGOperand(name="input", type_str="tensor<*xi64>"),
            ],
            results=[
                AEGOperand(name="logits", type_str="tensor<*xbf16>"),
            ],
        )
        block = Block(name="entry")
        func.add_block(block)
        # Convert graph nodes into AEG-IR instructions in topological order
        op_mapping: dict[str, str] = {}
        mapping = {
            "input": AEGOpCode.ARGUMENT,
            "parameter": AEGOpCode.PARAMETER,
            "output": AEGOpCode.LOGITS,
            "kv_cache": AEGOpCode.KV_CACHE_STORE,
            "expert_router": AEGOpCode.MOE_ROUTER,
            "fused": AEGOpCode.FUSED_LAYER,
            "embedding": AEGOpCode.EMBEDDING,
            "rmsnorm": AEGOpCode.RMS_NORM,
            "layernorm": AEGOpCode.LAYER_NORM,
            "linear": AEGOpCode.LINEAR,
            # Encoder-only and generic attention graph nodes.  These entries
            # are deliberately explicit: falling through to the raw graph
            # op_type would create an invalid AEG-IR opcode without the
            # required ``aeg.`` namespace prefix.
            "mha": AEGOpCode.MHA,
            "pooler": AEGOpCode.LINEAR,
            "gelu": AEGOpCode.GELU,
            "qkv_proj": AEGOpCode.QKV_PROJ,
            "gate_proj": AEGOpCode.GATE_PROJ,
            "up_proj": AEGOpCode.UP_PROJ,
            "down_proj": AEGOpCode.DOWN_PROJ,
            "gqa": AEGOpCode.GQA,
            "rope": AEGOpCode.ROPE,
            "ffn": AEGOpCode.FFN,
            "swiglu_ffn": AEGOpCode.SWIGLU_FFN,
            "expert_ffn": AEGOpCode.EXPERT_FFN,
            "gemm": AEGOpCode.QUANTIZED_GEMM,
            "add": AEGOpCode.ADD,
            "matmul": AEGOpCode.MATMUL,
            "softmax": AEGOpCode.SOFTMAX,
            "lm_head": AEGOpCode.LM_HEAD,
            "mtp_head": AEGOpCode.MTP_HEAD,
        }
        for node in graph:
            aeg_op = mapping.get(node.op_type or "", node.op_type or "aeg.tensor")
            results = []
            out_name = f"{node.id}_out"
            op_result = AEGOperand(name=out_name, type_str="tensor<*xbf16>")
            results.append(op_result)
            op_mapping[node.id] = out_name
            inputs = [op_mapping.get(in_id, in_id) for in_id in node.inputs]
            # Weight payloads belong to the content-addressed weight store,
            # not to the textual IR.  Convert compiler-side rich metadata
            # (numpy arrays, pruning masks, enums) to deterministic wire data
            # so save/load never falls back to object repr strings.
            from aether.core.hash_utils import _canonicalize_for_hash

            safe_attributes = _canonicalize_for_hash(dict(node.attributes))
            inst = AEGInstruction(
                results=results,
                op_code=aeg_op,
                inputs=inputs,
                attributes=safe_attributes,
                comment=node.name,
            )
            block.add_instruction(inst)
        module.add_function(func)
        return module

    def __repr__(self) -> str:
        return f"AEGIRModule(v{self.version}, {len(self.functions)} functions)"


def parse_aeg_ir(text: str) -> AEGIRModule:
    """Parse AEG-IR text format into an AEGIRModule.

    This is a simple parser for the human-readable AEG-IR syntax shown by
    `aether graph`. It handles single-block functions and a subset of
    operation types.

    Args:
        text: AEG-IR text.

    Returns:
        Parsed AEGIRModule.
    """
    module = AEGIRModule(version="AEG-IR/1.0")
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith(";") and not line.strip().startswith("//")]
    current_function: Function | None = None
    current_block: Block | None = None
    for line in lines:
        if line.startswith("func @"):
            # Parse function signature
            name_part = line.split("(")[0].replace("func @", "").strip()
            params_part = line.split("(")[1].split(")")[0] if "(" in line else ""
            if params_part:
                params = [AEGOperand(name=p.split(":")[0].strip().lstrip("%"), type_str=p.split(":")[1].strip()) for p in params_part.split(",") if p.strip()]
            else:
                params = []
            current_function = Function(name=name_part, parameters=params)
            current_block = Block(name="entry")
            current_function.add_block(current_block)
            module.add_function(current_function)
        elif line == "}":
            current_function = None
            current_block = None
        elif current_block is not None and "=" in line:
            # Instruction line
            result_part = line.split("=")[0].strip()
            op_part = line.split("=")[1].strip()
            op_code = op_part.split("(")[0].strip()
            args_part = op_part.split("(")[1].split(")")[0] if "(" in op_part else ""
            input_names = [a.strip().lstrip("%") for a in args_part.split(",") if a.strip() and a.strip() != "%"]
            result_names = [r.strip().lstrip("%") for r in result_part.split(",")]
            results = [AEGOperand(name=rn, type_str="tensor<*xbf16>") for rn in result_names]
            inst = AEGInstruction(results=results, op_code=op_code, inputs=input_names)
            current_block.add_instruction(inst)
    return module

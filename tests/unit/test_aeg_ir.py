"""
Tests for AEG-IR data structures and serialization.
"""

from __future__ import annotations

import pytest

from aether.core.aeg_ir import (
    AEGInstruction,
    AEGIRModule,
    AEGOpCode,
    AEGOperand,
    Block,
    Function,
    parse_aeg_ir,
)


class TestAEGOpCode:
    """Tests for AEG-IR op code constants."""

    def test_is_valid_for_known_op(self) -> None:
        assert AEGOpCode.is_valid("aeg.rmsnorm")
        assert AEGOpCode.is_valid("aeg.gqa")
        assert AEGOpCode.is_valid("aeg.swiglu_ffn")

    def test_is_valid_for_unknown_op(self) -> None:
        assert not AEGOpCode.is_valid("aeg.unknown_op")
        assert not AEGOpCode.is_valid("invalid")

    def test_known_ops_contains_core_ops(self) -> None:
        ops = AEGOpCode.known_ops()
        assert "aeg.rmsnorm" in ops
        assert "aeg.linear" in ops
        assert "aeg.add" in ops

    def test_attention_ops(self) -> None:
        assert "aeg.gqa" in AEGOpCode.attention_ops()
        assert "aeg.mla" in AEGOpCode.attention_ops()

    def test_norm_ops(self) -> None:
        assert "aeg.rmsnorm" in AEGOpCode.norm_ops()
        assert "aeg.layernorm" in AEGOpCode.norm_ops()


class TestAEGOperand:
    """Tests for AEG-IR operands."""

    def test_create_operand(self) -> None:
        op = AEGOperand(name="x", type_str="tensor<*xbf16>")
        assert op.name == "x"
        assert op.type_str == "tensor<*xbf16>"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            AEGOperand(name="", type_str="tensor<*xbf16>")

    def test_to_dict(self) -> None:
        op = AEGOperand(name="x", type_str="tensor<*xbf16>", attributes={"shape": [1, 2, 3]})
        assert op.to_dict()["name"] == "x"
        assert op.to_dict()["attributes"]["shape"] == [1, 2, 3]

    def test_from_dict(self) -> None:
        data = {"name": "x", "type": "tensor<*xbf16>", "attributes": {}}
        op = AEGOperand.from_dict(data)
        assert op.name == "x"
        assert op.type_str == "tensor<*xbf16>"


class TestAEGInstruction:
    """Tests for AEG-IR instructions."""

    def test_create_instruction(self) -> None:
        inst = AEGInstruction(
            results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.RMS_NORM,
            inputs=["x"],
            attributes={"eps": 1e-6},
        )
        assert inst.op_code == "aeg.rmsnorm"
        assert inst.result_names == ["y"]
        assert inst.input_names == ["x"]

    def test_invalid_op_code_raises(self) -> None:
        with pytest.raises(ValueError):
            AEGInstruction(
                results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
                op_code="invalid",
                inputs=["x"],
            )

    def test_set_attribute(self) -> None:
        inst = AEGInstruction(
            results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.RMS_NORM,
            inputs=["x"],
        )
        inst.set_attribute("eps", 1e-6)
        assert inst.get_attribute("eps") == 1e-6

    def test_to_dict(self) -> None:
        inst = AEGInstruction(
            results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.RMS_NORM,
            inputs=["x"],
            attributes={"eps": 1e-6},
        )
        data = inst.to_dict()
        assert data["op_code"] == "aeg.rmsnorm"
        assert data["inputs"] == ["x"]
        assert data["attributes"]["eps"] == 1e-6

    def test_from_dict(self) -> None:
        data = {
            "results": [{"name": "y", "type": "tensor<*xbf16>", "attributes": {}}],
            "op_code": "aeg.rmsnorm",
            "inputs": ["x"],
            "attributes": {"eps": 1e-6},
            "comment": "rmsnorm",
        }
        inst = AEGInstruction.from_dict(data)
        assert inst.op_code == "aeg.rmsnorm"
        assert inst.comment == "rmsnorm"


class TestBlock:
    """Tests for AEG-IR blocks."""

    def test_add_instruction(self) -> None:
        block = Block(name="entry")
        inst = AEGInstruction(
            results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.RMS_NORM,
            inputs=["x"],
        )
        block.add_instruction(inst)
        assert block.instruction_count == 1

    def test_find_instruction(self) -> None:
        block = Block(name="entry")
        inst = AEGInstruction(
            results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
            op_code=AEGOpCode.RMS_NORM,
            inputs=["x"],
        )
        block.add_instruction(inst)
        found = block.find_instruction("y")
        assert found is inst

    def test_find_instruction_by_op(self) -> None:
        block = Block(name="entry")
        block.add_instruction(
            AEGInstruction(
                results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
                op_code=AEGOpCode.RMS_NORM,
                inputs=["x"],
            )
        )
        block.add_instruction(
            AEGInstruction(
                results=[AEGOperand(name="z", type_str="tensor<*xbf16>")],
                op_code=AEGOpCode.LINEAR,
                inputs=["y"],
            )
        )
        linear = block.find_instructions_by_op("aeg.linear")
        assert len(linear) == 1


class TestFunction:
    """Tests for AEG-IR functions."""

    def test_function_instruction_count(self) -> None:
        func = Function(name="test")
        block = Block(name="entry")
        block.add_instruction(
            AEGInstruction(
                results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
                op_code=AEGOpCode.RMS_NORM,
                inputs=["x"],
            )
        )
        func.add_block(block)
        assert func.instruction_count == 1

    def test_find_function(self) -> None:
        module = AEGIRModule(version="AEG-IR/1.0")
        func = Function(name="model")
        func.add_block(Block(name="entry"))
        module.add_function(func)
        assert module.find_function("model") is func

    def test_find_instruction_by_op(self) -> None:
        func = Function(name="test")
        block = Block(name="entry")
        block.add_instruction(
            AEGInstruction(
                results=[AEGOperand(name="y", type_str="tensor<*xbf16>")],
                op_code=AEGOpCode.RMS_NORM,
                inputs=["x"],
            )
        )
        block.add_instruction(
            AEGInstruction(
                results=[AEGOperand(name="z", type_str="tensor<*xbf16>")],
                op_code=AEGOpCode.LINEAR,
                inputs=["y"],
            )
        )
        func.add_block(block)
        results = func.find_instructions_by_op("aeg.linear")
        assert len(results) == 1


class TestAEGIRModule:
    """Tests for AEG-IR modules."""

    def test_module_json_roundtrip(self, minimal_aeg_ir: AEGIRModule) -> None:
        json_str = minimal_aeg_ir.to_json()
        loaded = AEGIRModule.from_json(json_str)
        assert loaded.version == minimal_aeg_ir.version
        assert loaded.function_count == minimal_aeg_ir.function_count
        assert loaded.instruction_count == minimal_aeg_ir.instruction_count

    def test_module_to_text(self, minimal_aeg_ir: AEGIRModule) -> None:
        text = minimal_aeg_ir.to_text()
        assert "AEG-IR" in text
        assert "func @model" in text
        assert "aeg.embedding" in text

    def test_compute_instruction_count_by_op(self, minimal_aeg_ir: AEGIRModule) -> None:
        counts = minimal_aeg_ir.compute_instruction_count_by_op()
        assert counts["aeg.embedding"] == 1
        assert counts["aeg.add"] == 1

    def test_parse_aeg_ir(self) -> None:
        text = """
        func @model(%input: tensor<*xi64>) -> (%logits: tensor<*xbf16>) {
            %emb = aeg.embedding(%input)
            %logits = aeg.linear(%emb)
        }
        """
        module = parse_aeg_ir(text)
        assert module.function_count == 1
        func = module.find_function("model")
        assert func is not None
        assert func.instruction_count == 2


class TestOpCodeCategorization:
    """Tests for op code categorization helpers."""

    def test_fused_ops(self) -> None:
        fused = AEGOpCode.fused_ops()
        assert "aeg.fused_qkv_rope_norm" in fused

    def test_moe_ops(self) -> None:
        moe = AEGOpCode.moe_ops()
        assert "aeg.moe_router" in moe
        assert "aeg.expert_ffn" in moe

    def test_speculative_ops(self) -> None:
        spec = AEGOpCode.speculative_ops()
        assert "aeg.draft_verify" in spec

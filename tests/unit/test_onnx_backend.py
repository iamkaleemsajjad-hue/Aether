"""Real ONNX Runtime backend integration tests."""

from __future__ import annotations

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime

from aether.backends.base import GenerationRequest
from aether.backends.onnx_backend import ONNXBackend


class _TinyTokenizer:
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert text
        return [1]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join({2: "A", 3: "B"}.get(int(token), "?") for token in token_ids)


def _write_constant_logits_model(path) -> None:
    input_info = oh.make_tensor_value_info(
        "input_ids", onnx.TensorProto.INT64, [1, "sequence"]
    )
    output_info = oh.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, [1, 1, 4]
    )
    # Token 2 is the unique greedy choice. The dynamic input shape allows the
    # backend to execute the same real session for multiple decode steps.
    values = np.asarray([[[0.0, 0.5, 4.0, 1.0]]], dtype=np.float32)
    constant = oh.make_tensor(
        "constant_logits", onnx.TensorProto.FLOAT, values.shape, values.ravel().tolist()
    )
    node = oh.make_node("Constant", inputs=[], outputs=["logits"], value=constant)
    graph = oh.make_graph([node], "aether_onnx_decode", [input_info], [output_info])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    # The installed ONNX Runtime in the audit environment supports IR <= 11.
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_onnx_runtime_executes_real_autoregressive_loop(tmp_path) -> None:
    model_path = tmp_path / "constant_logits.onnx"
    _write_constant_logits_model(model_path)

    backend = ONNXBackend()
    backend.load_model(
        str(model_path),
        providers=["CPUExecutionProvider"],
        tokenizer=_TinyTokenizer(),
    )
    result = backend.generate(
        GenerationRequest(
            model_id=str(model_path),
            prompt="prompt",
            max_tokens=2,
            temperature=0.0,
        )
    )

    assert result.text == "AA"
    assert result.prompt_tokens == 1
    assert result.completion_tokens == 2
    assert result.finish_reason == "length"
    assert result.metrics["device"] == "onnxruntime"
    assert "CPUExecutionProvider" in result.metrics["providers"]

    chunks = list(
        backend.generate_stream(
            GenerationRequest(
                model_id=str(model_path),
                prompt="prompt",
                max_tokens=2,
                temperature=0.0,
                stream=True,
            )
        )
    )
    assert chunks == ["A", "A"]


def test_onnx_runtime_requires_tokenizer_adapter(tmp_path) -> None:
    model_path = tmp_path / "constant_logits.onnx"
    _write_constant_logits_model(model_path)
    backend = ONNXBackend()
    backend.load_model(str(model_path), providers=["CPUExecutionProvider"])

    from aether.core.exceptions import BackendError

    try:
        backend.generate(
            GenerationRequest(model_id=str(model_path), prompt="prompt", max_tokens=1)
        )
    except BackendError as exc:
        assert "refusing fabricated output" in str(exc)
    else:
        raise AssertionError("ONNX generation must require a tokenizer adapter")


def test_runtime_routes_local_onnx_to_onnx_backend(tmp_path) -> None:
    model_path = tmp_path / "constant_logits.onnx"
    _write_constant_logits_model(model_path)

    from aether.runtime.config import RuntimeConfig
    from aether.runtime.runtime import Runtime

    runtime = Runtime(
        RuntimeConfig(
            backend_name="onnxruntime",
            hf_offline=True,
            speculative_decoding=False,
        )
    )
    response = runtime.generate(
        str(model_path),
        "prompt",
        max_tokens=2,
        temperature=0.0,
        tokenizer=_TinyTokenizer(),
    )

    assert response.text == "AA"
    assert response.metrics.backend_name == "onnx"

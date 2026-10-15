"""Tiny real CPU operator checks, not model accuracy or performance measurements."""

from __future__ import annotations

import numpy as np


def check_ort_kernels() -> dict[str, str]:
    """Exercise the quantized and cached-attention kernels required by the artifact."""
    import onnxruntime as ort
    from onnx import TensorProto as T, helper as h

    def session(nodes, inputs, outputs, initializers=()):
        graph = h.make_graph(
            nodes, "kernel-compatibility-only", inputs, outputs, list(initializers)
        )
        model = h.make_model(
            graph,
            opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        runtime = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        runtime.disable_fallback()
        if runtime.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("Kernel probe selected an unexpected provider")
        return runtime

    vi = h.make_tensor_value_info
    matmul = session(
        [
            h.make_node(
                "MatMulNBits",
                ["x", "w", "scales"],
                ["y"],
                domain="com.microsoft",
                K=128,
                N=2,
                bits=4,
                block_size=128,
            )
        ],
        [vi("x", T.FLOAT, [2, 128])],
        [vi("y", T.FLOAT, [2, 2])],
        [
            h.make_tensor("w", T.UINT8, [2, 1, 64], [0x99] * 128),
            h.make_tensor("scales", T.FLOAT, [2, 1], [1, 2]),
        ],
    )
    np.testing.assert_allclose(
        matmul.run(None, {"x": np.ones((2, 128), np.float32)})[0],
        [[128, 256], [128, 256]],
        rtol=1e-5,
    )
    # INT4 values 1 and 2: packed low/high nibbles, explicit ONNX raw storage.
    gather = session(
        [
            h.make_node(
                "GatherBlockQuantized",
                ["w", "ids", "s"],
                ["y"],
                domain="com.microsoft",
                gather_axis=0,
                quantize_axis=1,
                block_size=128,
            )
        ],
        [vi("ids", T.INT64, [2])],
        [vi("y", T.FLOAT, [2, 128])],
        [
            h.make_tensor("w", T.INT4, [2, 128], bytes([0x11] * 64 + [0x22] * 64), raw=True),
            h.make_tensor("s", T.FLOAT, [2, 1], [1, 1]),
        ],
    )
    np.testing.assert_allclose(
        gather.run(None, {"ids": np.array([1, 0], np.int64)})[0],
        np.repeat([[2.0], [1.0]], 128, axis=1),
    )
    attention = session(
        [
            h.make_node(
                "GroupQueryAttention",
                ["q", "k", "v", "pk", "pv", "lengths", "total", "cos", "sin"],
                ["y", "nk", "nv"],
                domain="com.microsoft",
                num_heads=4,
                kv_num_heads=2,
                do_rotary=1,
                rotary_interleaved=0,
            )
        ],
        [
            vi("q", T.FLOAT, ["b", "s", 64]),
            vi("k", T.FLOAT, ["b", "s", 32]),
            vi("v", T.FLOAT, ["b", "s", 32]),
            vi("pk", T.FLOAT, ["b", 2, "p", 16]),
            vi("pv", T.FLOAT, ["b", 2, "p", 16]),
            vi("lengths", T.INT32, ["b"]),
            vi("total", T.INT32, []),
        ],
        [
            vi("y", T.FLOAT, ["b", "s", 64]),
            vi("nk", T.FLOAT, ["b", 2, "t", 16]),
            vi("nv", T.FLOAT, ["b", 2, "t", 16]),
        ],
        [
            h.make_tensor("cos", T.FLOAT, [8, 8], [1] * 64),
            h.make_tensor("sin", T.FLOAT, [8, 8], [0] * 64),
        ],
    )
    for batch in (1, 2):
        pk = pv = np.empty((batch, 2, 0, 16), np.float32)
        for sequence, total, value, expected in ((2, 2, 1, 1), (1, 3, 4, 2)):
            y, pk, pv = attention.run(
                None,
                {
                    "q": np.zeros((batch, sequence, 64), np.float32),
                    "k": np.zeros((batch, sequence, 32), np.float32),
                    "v": np.full((batch, sequence, 32), value, np.float32),
                    "pk": pk,
                    "pv": pv,
                    "lengths": np.full(batch, total - 1, np.int32),
                    "total": np.array(total, np.int32),
                },
            )
            np.testing.assert_allclose(y, expected, rtol=1e-5)
            if pk.shape != (batch, 2, total, 16) or pv.shape != pk.shape:
                raise ValueError("GroupQueryAttention did not grow its KV cache")
    return {
        "MatMulNBits": "passed",
        "GatherBlockQuantized": "passed",
        "GroupQueryAttention": "passed: rotary, prefill, decode, growing KV, batch 1/2",
        "scope": "tiny operator checks only; Qwen3-8B smoke run still required",
    }

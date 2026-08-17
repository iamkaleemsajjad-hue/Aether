"""
Complete GGUF loader implementation with full metadata support.

Enhances the GGUF loader to extract all metadata, embedded tokenizers,
and properly handle quantization formats.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class GGUFReader:
    """Complete GGUF file reader with full metadata extraction."""

    # GGUF magic number
    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    GGUF_VERSION = 3

    # Quantization types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S = 21
    GGML_TYPE_IQ2_S = 22
    GGML_TYPE_IQ4_XS = 23

    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata: dict[str, Any] = {}
        self.architecture: str = ""
        self.tensors: dict[str, dict[str, Any]] = {}
        self.tokenizer: dict[str, Any] = {}
        self._file_handle = None

        if not self.path.exists():
            raise IngestionError(f"GGUF file not found: {path}")

        self._load()

    def _load(self):
        """Load GGUF file header and metadata."""
        with open(self.path, 'rb') as f:
            self._file_handle = f

            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise IngestionError(f"Invalid GGUF magic number: {hex(magic)}")

            version = struct.unpack('<I', f.read(4))[0]
            if version not in {2, 3}:
                raise IngestionError(f"Unsupported GGUF version: {version}")

            # Read tensor count and metadata count
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata key-value pairs
            for _ in range(metadata_count):
                key = self._read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value

            # Extract architecture
            self.architecture = str(self.metadata.get("general.architecture", "llama"))

            # Read tensor information
            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
                tensor_type = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]

                self.tensors[name] = {
                    "shape": dims,
                    "type": tensor_type,
                    "offset": offset,
                }

            # Extract tokenizer if present
            self._extract_tokenizer()

            logger.info("Loaded GGUF", path=str(self.path),
                       architecture=self.architecture,
                       tensors=len(self.tensors),
                       metadata_keys=len(self.metadata))

    def _read_string(self, f) -> str:
        """Read a string from GGUF file."""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, value_type: int) -> Any:
        """Read a value based on its type."""
        # Type mappings
        GGUF_TYPE_UINT8 = 0
        GGUF_TYPE_INT8 = 1
        GGUF_TYPE_UINT16 = 2
        GGUF_TYPE_INT16 = 3
        GGUF_TYPE_UINT32 = 4
        GGUF_TYPE_INT32 = 5
        GGUF_TYPE_FLOAT32 = 6
        GGUF_TYPE_BOOL = 7
        GGUF_TYPE_STRING = 8
        GGUF_TYPE_ARRAY = 9
        GGUF_TYPE_UINT64 = 10
        GGUF_TYPE_INT64 = 11
        GGUF_TYPE_FLOAT64 = 12

        if value_type == GGUF_TYPE_UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == GGUF_TYPE_INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == GGUF_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == GGUF_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == GGUF_TYPE_UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == GGUF_TYPE_INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == GGUF_TYPE_FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == GGUF_TYPE_UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == GGUF_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == GGUF_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == GGUF_TYPE_BOOL:
            return struct.unpack('<?', f.read(1))[0]
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]
        else:
            raise IngestionError(f"Unknown GGUF value type: {value_type}")

    def _extract_tokenizer(self):
        """Extract embedded tokenizer metadata."""
        prefix = "tokenizer."

        # Extract tokenizer type
        model_key = f"{prefix}ggml.model"
        self.tokenizer["type"] = self.metadata.get(model_key, "llama")

        # Extract vocabulary
        tokens = self.metadata.get(f"{prefix}ggml.tokens", [])
        scores = self.metadata.get(f"{prefix}ggml.scores", [])
        token_type = self.metadata.get(f"{prefix}ggml.token_type", [])

        if tokens:
            self.tokenizer["vocab"] = {
                "tokens": tokens,
                "scores": scores if scores else [0.0] * len(tokens),
                "types": token_type if token_type else [0] * len(tokens),
            }
            self.tokenizer["vocab_size"] = len(tokens)

        # Extract special tokens
        special_tokens = {}
        for special in ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]:
            key = f"{prefix}ggml.{special}_token_id"
            if key in self.metadata:
                special_tokens[special] = self.metadata[key]

        if special_tokens:
            self.tokenizer["special_tokens"] = special_tokens

        # Extract BPE merges if present
        merges = self.metadata.get(f"{prefix}ggml.merges", [])
        if merges:
            self.tokenizer["merges"] = merges

        # Extract added tokens
        added_tokens = self.metadata.get(f"{prefix}ggml.added_tokens", [])
        if added_tokens:
            self.tokenizer["added_tokens"] = added_tokens

    def dequantize_tensor(self, tensor_name: str) -> np.ndarray:
        """Dequantize a tensor to FP32.

        Args:
            tensor_name: Name of the tensor to dequantize.

        Returns:
            Dequantized numpy array in FP32.
        """
        if tensor_name not in self.tensors:
            raise IngestionError(f"Tensor not found: {tensor_name}")

        tensor_info = self.tensors[tensor_name]
        tensor_type = tensor_info["type"]
        shape = tensor_info["shape"]
        offset = tensor_info["offset"]

        # Read raw tensor data
        with open(self.path, 'rb') as f:
            # Skip to tensor data (after all metadata and tensor headers)
            f.seek(offset)

            if tensor_type == self.GGML_TYPE_F32:
                # Already FP32
                size = np.prod(shape)
                data = np.fromfile(f, dtype=np.float32, count=size)
                return data.reshape(shape)

            elif tensor_type == self.GGML_TYPE_F16:
                # FP16 -> FP32
                size = np.prod(shape)
                data = np.fromfile(f, dtype=np.float16, count=size)
                return data.astype(np.float32).reshape(shape)

            elif tensor_type == self.GGML_TYPE_Q4_0:
                return self._dequantize_q4_0(f, shape)

            elif tensor_type == self.GGML_TYPE_Q4_1:
                return self._dequantize_q4_1(f, shape)

            elif tensor_type == self.GGML_TYPE_Q5_0:
                return self._dequantize_q5_0(f, shape)

            elif tensor_type == self.GGML_TYPE_Q5_1:
                return self._dequantize_q5_1(f, shape)

            elif tensor_type == self.GGML_TYPE_Q8_0:
                return self._dequantize_q8_0(f, shape)

            elif tensor_type in {self.GGML_TYPE_Q2_K, self.GGML_TYPE_Q3_K,
                                 self.GGML_TYPE_Q4_K, self.GGML_TYPE_Q5_K,
                                 self.GGML_TYPE_Q6_K}:
                return self._dequantize_k_quant(f, shape, tensor_type)

            else:
                raise IngestionError(
                    f"Unsupported quantization type for dequantization: {tensor_type}"
                )

    def _dequantize_q4_0(self, f, shape: tuple) -> np.ndarray:
        """Dequantize Q4_0 format (4-bit with delta), ggml-compatible layout.

        Byte j holds element j (low nibble) and element j+16 (high nibble),
        per dequantize_row_q4_0 in ggml-quanta.c.
        """
        block_size = 32
        num_blocks = (np.prod(shape) + block_size - 1) // block_size

        result = []
        for _ in range(num_blocks):
            delta = np.frombuffer(f.read(2), dtype=np.float16)[0]
            quants = np.frombuffer(f.read(16), dtype=np.uint8)

            values = np.empty(32, dtype=np.float32)
            values[0:16] = (quants & 0x0F).astype(np.float32) - 8.0
            values[16:32] = (quants >> 4).astype(np.float32) - 8.0

            result.extend(values * delta)

        return np.array(result[:np.prod(shape)], dtype=np.float32).reshape(shape)

    def _dequantize_q4_1(self, f, shape: tuple) -> np.ndarray:
        """Dequantize Q4_1 format (4-bit with delta and min), ggml layout.

        Byte j holds element j (low nibble) and element j+16 (high nibble).
        """
        block_size = 32
        num_blocks = (np.prod(shape) + block_size - 1) // block_size

        result = []
        for _ in range(num_blocks):
            delta = np.frombuffer(f.read(2), dtype=np.float16)[0]
            min_val = np.frombuffer(f.read(2), dtype=np.float16)[0]
            quants = np.frombuffer(f.read(16), dtype=np.uint8)

            values = np.empty(32, dtype=np.float32)
            values[0:16] = (quants & 0x0F).astype(np.float32)
            values[16:32] = (quants >> 4).astype(np.float32)

            result.extend(values * delta + min_val)

        return np.array(result[:np.prod(shape)], dtype=np.float32).reshape(shape)

    def _dequantize_q5_0(self, f, shape: tuple) -> np.ndarray:
        """Dequantize Q5_0 format (5-bit with delta), ggml-compatible layout.

        Element j = (qs[j] low nibble | qh bit j << 4) - 16 and
        element j+16 = (qs[j] high nibble | qh bit (j+16) << 4) - 16.
        """
        block_size = 32
        num_blocks = (np.prod(shape) + block_size - 1) // block_size

        result = []
        for _ in range(num_blocks):
            delta = np.frombuffer(f.read(2), dtype=np.float16)[0]
            qh = np.frombuffer(f.read(4), dtype=np.uint32)[0]  # High bits
            quants = np.frombuffer(f.read(16), dtype=np.uint8)

            values = np.empty(32, dtype=np.float32)
            j = np.arange(16)
            xh0 = ((qh >> j) & 1) << 4
            xh1 = ((qh >> (j + 16)) & 1) << 4
            values[0:16] = ((quants & 0x0F) | xh0).astype(np.float32) - 16.0
            values[16:32] = ((quants >> 4) | xh1).astype(np.float32) - 16.0

            result.extend(values * delta)

        return np.array(result[:np.prod(shape)], dtype=np.float32).reshape(shape)

    def _dequantize_q5_1(self, f, shape: tuple) -> np.ndarray:
        """Dequantize Q5_1 format (5-bit with delta and min), ggml layout."""
        block_size = 32
        num_blocks = (np.prod(shape) + block_size - 1) // block_size

        result = []
        for _ in range(num_blocks):
            delta = np.frombuffer(f.read(2), dtype=np.float16)[0]
            min_val = np.frombuffer(f.read(2), dtype=np.float16)[0]
            qh = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            quants = np.frombuffer(f.read(16), dtype=np.uint8)

            j = np.arange(16)
            xh0 = ((qh >> j) & 1) << 4
            xh1 = ((qh >> (j + 16)) & 1) << 4
            values = np.empty(32, dtype=np.float32)
            values[0:16] = ((quants & 0x0F) | xh0).astype(np.float32)
            values[16:32] = ((quants >> 4) | xh1).astype(np.float32)

            result.extend(values * delta + min_val)

        return np.array(result[:np.prod(shape)], dtype=np.float32).reshape(shape)

    def _dequantize_q8_0(self, f, shape: tuple) -> np.ndarray:
        """Dequantize Q8_0 format (8-bit with delta)."""
        block_size = 32
        num_blocks = (np.prod(shape) + block_size - 1) // block_size

        result = []
        for _ in range(num_blocks):
            delta = np.frombuffer(f.read(2), dtype=np.float16)[0]
            quants = np.frombuffer(f.read(32), dtype=np.int8)
            result.extend(quants.astype(np.float32) * delta)

        return np.array(result[:np.prod(shape)], dtype=np.float32).reshape(shape)

    def _dequantize_k_quant(self, f, shape: tuple, quant_type: int) -> np.ndarray:
        """Dequantize K-quant formats (Q2_K through Q6_K).

        Delegates to the reference-accurate implementations in
        :mod:`aether.compiler.stage1_ingestion.gguf_loader`, which are faithful
        transcriptions of llama.cpp's ggml-quanta.c dequantizers. Never returns
        placeholder data: an unsupported or malformed K-quant tensor raises
        ``UnsupportedFormatError`` so compilation fails closed.
        """
        from aether.compiler.stage1_ingestion import gguf_loader as _gguf_impl

        dequant_fn = _gguf_impl._DEQUANT_FN.get(quant_type)
        if dequant_fn is None:
            raise UnsupportedFormatError(
                f"K-quant dequantization is not implemented for ggml type {quant_type}"
            )
        block_elems = _gguf_impl._BLOCK_ELEMS.get(quant_type, 256)
        block_bytes = _gguf_impl._BLOCK_SIZES.get(quant_type)
        num_elems = int(np.prod(shape))
        if block_bytes is None or num_elems % block_elems != 0:
            raise UnsupportedFormatError(
                f"Cannot dequantize ggml type {quant_type}: tensor with "
                f"{num_elems} elements does not divide into {block_elems}-element blocks"
            )
        raw = f.read(block_bytes * (num_elems // block_elems))
        if len(raw) < block_bytes * (num_elems // block_elems):
            raise IngestionError(
                f"Truncated K-quant tensor: expected "
                f"{block_bytes * (num_elems // block_elems)} bytes, got {len(raw)}"
            )
        return dequant_fn(raw, num_elems).reshape(shape)

    def get_architecture_info(self) -> dict[str, Any]:
        """Extract architecture information from metadata."""
        prefix = f"{self.architecture}."

        info = {
            "architecture": self.architecture,
            "layers": self.metadata.get(f"{prefix}block_count", 32),
            "hidden_size": self.metadata.get(f"{prefix}embedding_length", 4096),
            "num_attention_heads": self.metadata.get(f"{prefix}attention.head_count", 32),
            "num_kv_heads": self.metadata.get(f"{prefix}attention.head_count_kv", None),
            "context_length": self.metadata.get(f"{prefix}context_length", 2048),
            "vocab_size": self.metadata.get(f"{self.architecture}.vocab_size", 32000),
            "intermediate_size": self.metadata.get(f"{prefix}feed_forward_length", 11008),
        }

        # MoE information
        expert_count = self.metadata.get(f"{prefix}expert_count", 0)
        if expert_count > 0:
            info["is_moe"] = True
            info["num_experts"] = expert_count
            info["num_activated_experts"] = self.metadata.get(
                f"{prefix}expert_used_count", 2
            )

        return info

    def export_tokenizer_json(self, output_path: Path):
        """Export embedded tokenizer to tokenizer.json format compatible with HuggingFace."""
        if not self.tokenizer.get("vocab"):
            raise IngestionError("No tokenizer vocabulary found in GGUF")

        vocab = self.tokenizer["vocab"]

        # Create HuggingFace-compatible tokenizer.json
        tokenizer_json = {
            "version": "1.0",
            "model": {
                "type": "BPE" if self.tokenizer.get("merges") else "Unigram",
                "vocab": {token: idx for idx, token in enumerate(vocab["tokens"])},
            },
            "added_tokens": self.tokenizer.get("added_tokens", []),
            "special_tokens": self.tokenizer.get("special_tokens", {}),
        }

        if self.tokenizer.get("merges"):
            tokenizer_json["model"]["merges"] = self.tokenizer["merges"]

        output_path.write_text(json.dumps(tokenizer_json, indent=2))
        logger.info(f"Exported tokenizer to {output_path}")


class GGUFLoader:
    """High-level GGUF model loader."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.reader = GGUFReader(self.model_path)

    def load(self) -> dict[str, Any]:
        """Load GGUF model and return dictionary with all information."""
        return {
            "metadata": self.reader.metadata,
            "architecture": self.reader.get_architecture_info(),
            "tensors": self.reader.tensors,
            "tokenizer": self.reader.tokenizer,
            "path": str(self.model_path),
        }

    def get_tokenizer(self) -> dict[str, Any]:
        """Get embedded tokenizer information."""
        return self.reader.tokenizer

    def dequantize_all(self) -> dict[str, np.ndarray]:
        """Dequantize all tensors to FP32.

        Warning: This can be memory-intensive for large models.
        """
        dequantized = {}
        for tensor_name in self.reader.tensors.keys():
            try:
                dequantized[tensor_name] = self.reader.dequantize_tensor(tensor_name)
            except Exception as e:
                logger.warning(f"Failed to dequantize {tensor_name}: {e}")

        return dequantized

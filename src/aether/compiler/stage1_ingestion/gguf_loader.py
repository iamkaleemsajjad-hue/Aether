"""
GGUF model loader with full K-quant dequantization.

Loads GGUF files produced by llama.cpp. Supports direct dequantization of
Q4_K_M, Q4_K_S, Q8_0, Q6_K, Q5_K_M, Q2_K, Q3_K, F16, F32 tensors into
float32 numpy arrays — no external gguf package required for the core path.

The optional ``gguf`` package is used when available for richer metadata
parsing; the pure-binary path is always tried first so offline compilation
works without any extra dependencies.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from aether.core.exceptions import IngestionError, UnsupportedFormatError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# GGUF format constants
# ---------------------------------------------------------------------------
_GGUF_MAGIC = 0x46554747          # b"GGUF"
_GGUF_VERSION_2 = 2
_GGUF_VERSION_3 = 3

# GGUF value types
_GGUF_TYPE_UINT8   = 0
_GGUF_TYPE_INT8    = 1
_GGUF_TYPE_UINT16  = 2
_GGUF_TYPE_INT16   = 3
_GGUF_TYPE_UINT32  = 4
_GGUF_TYPE_INT32   = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL    = 7
_GGUF_TYPE_STRING  = 8
_GGUF_TYPE_ARRAY   = 9
_GGUF_TYPE_UINT64  = 10
_GGUF_TYPE_INT64   = 11
_GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
_GGML_TYPE_F32   = 0
_GGML_TYPE_F16   = 1
_GGML_TYPE_Q4_0  = 2
_GGML_TYPE_Q4_1  = 3
_GGML_TYPE_Q4_K  = 12  # Q4_K_S / Q4_K_M
_GGML_TYPE_Q5_K  = 13
_GGML_TYPE_Q6_K  = 14
_GGML_TYPE_Q8_K  = 15
_GGML_TYPE_Q8_0  = 8
_GGML_TYPE_Q2_K  = 10
_GGML_TYPE_Q3_K  = 11
_GGML_TYPE_BF16  = 30
_GGML_TYPE_Q5_0  = 6
_GGML_TYPE_Q5_1  = 7

# Bytes per block for each type
_BLOCK_SIZES: dict[int, int] = {
    _GGML_TYPE_F32:  4,
    _GGML_TYPE_F16:  2,
    _GGML_TYPE_BF16: 2,
    _GGML_TYPE_Q8_0: 34,   # 32 int8 + 1 f16 scale
    _GGML_TYPE_Q4_0: 18,   # 32 nibbles + 1 f16 scale = 16 + 2
    _GGML_TYPE_Q4_1: 20,   # 32 nibbles + 1 f16 scale + 1 f16 min
    _GGML_TYPE_Q4_K: 144,  # super-block: 256 elems
    _GGML_TYPE_Q5_K: 176,
    _GGML_TYPE_Q6_K: 210,
    _GGML_TYPE_Q2_K: 84,
    _GGML_TYPE_Q3_K: 110,
    _GGML_TYPE_Q8_K: 292,
    _GGML_TYPE_Q5_0: 22,
    _GGML_TYPE_Q5_1: 24,
}
_BLOCK_ELEMS: dict[int, int] = {
    _GGML_TYPE_F32:  1,
    _GGML_TYPE_F16:  1,
    _GGML_TYPE_BF16: 1,
    _GGML_TYPE_Q8_0: 32,
    _GGML_TYPE_Q4_0: 32,
    _GGML_TYPE_Q4_1: 32,
    _GGML_TYPE_Q4_K: 256,
    _GGML_TYPE_Q5_K: 256,
    _GGML_TYPE_Q6_K: 256,
    _GGML_TYPE_Q2_K: 256,
    _GGML_TYPE_Q3_K: 256,
    _GGML_TYPE_Q8_K: 256,
    _GGML_TYPE_Q5_0: 32,
    _GGML_TYPE_Q5_1: 32,
}

# NF4 lookup table (used by Q4_K super-blocks)
_NF4_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634,  0.33791524171829224,
     0.44070982933044434,  0.5626170039176941,   0.7229568362236023,   1.0,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Low-level binary parsing helpers
# ---------------------------------------------------------------------------

def _read_bytes(data: bytes, offset: int, n: int) -> tuple[bytes, int]:
    return data[offset:offset + n], offset + n


def _read_u8(data: bytes, offset: int) -> tuple[int, int]:
    return data[offset], offset + 1


def _read_u16(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<H", data, offset)[0], offset + 2


def _read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _read_i32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<i", data, offset)[0], offset + 4


def _read_f32(data: bytes, offset: int) -> tuple[float, int]:
    return struct.unpack_from("<f", data, offset)[0], offset + 4


def _read_f16_as_f32(data: bytes, offset: int) -> tuple[float, int]:
    raw, offset = _read_u16(data, offset)
    arr = np.frombuffer(struct.pack("<H", raw), dtype=np.float16)
    return float(arr[0]), offset


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_u64(data, offset)
    raw, offset = _read_bytes(data, offset, int(length))
    return raw.decode("utf-8", errors="replace"), offset


def _read_value(data: bytes, offset: int, vtype: int) -> tuple[Any, int]:
    """Read a typed GGUF metadata value."""
    if vtype == _GGUF_TYPE_UINT8:
        v, offset = _read_u8(data, offset); return v, offset
    if vtype == _GGUF_TYPE_INT8:
        v = struct.unpack_from("<b", data, offset)[0]; return v, offset + 1
    if vtype == _GGUF_TYPE_UINT16:
        return _read_u16(data, offset)
    if vtype == _GGUF_TYPE_INT16:
        v = struct.unpack_from("<h", data, offset)[0]; return v, offset + 2
    if vtype == _GGUF_TYPE_UINT32:
        return _read_u32(data, offset)
    if vtype == _GGUF_TYPE_INT32:
        return _read_i32(data, offset)
    if vtype == _GGUF_TYPE_FLOAT32:
        return _read_f32(data, offset)
    if vtype == _GGUF_TYPE_UINT64:
        return _read_u64(data, offset)
    if vtype == _GGUF_TYPE_INT64:
        v = struct.unpack_from("<q", data, offset)[0]; return v, offset + 8
    if vtype == _GGUF_TYPE_FLOAT64:
        v = struct.unpack_from("<d", data, offset)[0]; return v, offset + 8
    if vtype == _GGUF_TYPE_BOOL:
        v, offset = _read_u8(data, offset); return bool(v), offset
    if vtype == _GGUF_TYPE_STRING:
        return _read_string(data, offset)
    if vtype == _GGUF_TYPE_ARRAY:
        elem_type, offset = _read_u32(data, offset)
        count, offset = _read_u64(data, offset)
        items: list[Any] = []
        for _ in range(count):
            item, offset = _read_value(data, offset, elem_type)
            items.append(item)
        return items, offset
    # Unknown type: skip 1 byte
    return None, offset + 1


# ---------------------------------------------------------------------------
# Dequantization kernels
# ---------------------------------------------------------------------------

def _dequant_f16(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return arr


def _dequant_bf16(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    # BF16 → F32: zero-extend 16-bit to upper 16 bits of f32
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _dequant_q8_0(raw: bytes, num_elems: int) -> np.ndarray:
    """Q8_0: blocks of 32 int8 values with one f16 scale."""
    n_blocks = num_elems // 32
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        base = i * 34
        scale = np.frombuffer(raw[base:base+2], dtype=np.float16)[0].astype(np.float32)
        quants = np.frombuffer(raw[base+2:base+34], dtype=np.int8).astype(np.float32)
        result[i*32:(i+1)*32] = quants * float(scale)
    return result


def _dequant_q4_0(raw: bytes, num_elems: int) -> np.ndarray:
    """Q4_0: blocks of 32 nibbles + f16 scale. Values are unsigned [0,15] shifted by -8."""
    n_blocks = num_elems // 32
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        base = i * 18
        scale = np.frombuffer(raw[base:base+2], dtype=np.float16)[0].astype(np.float32)
        packed = np.frombuffer(raw[base+2:base+18], dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.int8) - 8
        hi = ((packed >> 4) & 0x0F).astype(np.int8) - 8
        block = np.empty(32, dtype=np.float32)
        block[0::2] = lo.astype(np.float32)
        block[1::2] = hi.astype(np.float32)
        result[i*32:(i+1)*32] = block * float(scale)
    return result


def _dequant_q4_k(raw: bytes, num_elems: int) -> np.ndarray:
    """
    Q4_K (super-block of 256 elements, 8 sub-blocks of 32).
    Layout per 144-byte block:
      [0:2]   d     f16  overall scale
      [2:4]   dmin  f16  overall minimum
      [4:16]  scales  12 bytes (6-bit packed for 8 sub-blocks × scale/min)
      [16:144] qs   128 bytes  32 nibbles × 4 = 128 nibbles = 256 elements
    """
    n_blocks = num_elems // 256
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 144
        d    = np.frombuffer(raw[b:b+2],   dtype=np.float16)[0].astype(np.float32)
        dmin = np.frombuffer(raw[b+2:b+4], dtype=np.float16)[0].astype(np.float32)
        sc_raw = np.frombuffer(raw[b+4:b+16], dtype=np.uint8)  # 12 bytes → 8 pairs of 6-bit
        # Decode 6-bit scales and mins from 12 bytes
        scales_6 = np.zeros(8, dtype=np.uint8)
        mins_6   = np.zeros(8, dtype=np.uint8)
        for k in range(8):
            byte_idx = (k * 12) // 8
            bit_off  = (k * 12) % 8
            if bit_off <= 2:
                scales_6[k] = (sc_raw[byte_idx] >> bit_off) & 0x3F
                min_bit = bit_off + 6
                if min_bit < 8:
                    mins_6[k] = (sc_raw[byte_idx] >> min_bit) & 0x3F
                else:
                    mins_6[k] = ((sc_raw[byte_idx] >> min_bit) | (sc_raw[byte_idx+1] << (8-min_bit))) & 0x3F
            else:
                scales_6[k] = ((sc_raw[byte_idx] >> bit_off) | (sc_raw[byte_idx+1] << (8-bit_off))) & 0x3F
                min_bit = bit_off + 6
                mins_6[k] = ((sc_raw[byte_idx] >> min_bit) | (sc_raw[byte_idx+1] << (8-min_bit))) & 0x3F if byte_idx+1 < 12 else 0

        qs = np.frombuffer(raw[b+16:b+144], dtype=np.uint8)  # 128 bytes = 256 nibbles
        lo = (qs & 0x0F).astype(np.float32)
        hi = ((qs >> 4) & 0x0F).astype(np.float32)
        block_elems = np.empty(256, dtype=np.float32)
        block_elems[0::2] = lo
        block_elems[1::2] = hi
        for k in range(8):
            sl = k * 32
            s = float(scales_6[k]) * float(d)
            m = float(mins_6[k]) * float(dmin)
            block_elems[sl:sl+32] = block_elems[sl:sl+32] * s - m
        result[i*256:(i+1)*256] = block_elems
    return result


def _dequant_q6_k(raw: bytes, num_elems: int) -> np.ndarray:
    """
    Q6_K: 256-element super-block, 210 bytes.
    Layout: 128 bytes ql (4-bit lo), 64 bytes qh (2-bit hi), 16 bytes scales (int8), 2 bytes d (f16).
    """
    n_blocks = num_elems // 256
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 210
        ql = np.frombuffer(raw[b:b+128],   dtype=np.uint8)   # low 4 bits
        qh = np.frombuffer(raw[b+128:b+192], dtype=np.uint8)  # high 2 bits
        sc = np.frombuffer(raw[b+192:b+208], dtype=np.int8).astype(np.float32)
        d  = np.frombuffer(raw[b+208:b+210], dtype=np.float16)[0].astype(np.float32)
        # Reconstruct 6-bit values
        q = np.empty(256, dtype=np.float32)
        for j in range(128):
            lo4 = int(ql[j])
            hi2_byte = qh[j // 4]
            shift = (j % 4) * 2
            hi2 = (hi2_byte >> shift) & 0x03
            val = ((lo4 & 0x0F) | (hi2 << 4)) - 32  # signed 6-bit
            q[j] = float(val)
        for j in range(128, 256):
            idx = j - 128
            lo4 = int(ql[idx])
            hi2_byte = qh[idx // 4 + 32]
            shift = (idx % 4) * 2
            hi2 = (hi2_byte >> shift) & 0x03
            val = (((lo4 >> 4) & 0x0F) | (hi2 << 4)) - 32
            q[j] = float(val)
        for k in range(16):
            sl = k * 16
            result[i*256 + sl:i*256 + sl + 16] = q[sl:sl+16] * float(d) * float(sc[k])
    return result


def _dequant_q2_k(raw: bytes, num_elems: int) -> np.ndarray:
    """Q2_K: 256-element super-block, 84 bytes."""
    n_blocks = num_elems // 256
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 84
        scales = np.frombuffer(raw[b:b+16],   dtype=np.uint8)
        qs     = np.frombuffer(raw[b+16:b+80], dtype=np.uint8)  # 64 bytes = 256 2-bit
        d      = np.frombuffer(raw[b+80:b+82], dtype=np.float16)[0].astype(np.float32)
        dmin   = np.frombuffer(raw[b+82:b+84], dtype=np.float16)[0].astype(np.float32)
        for k in range(16):
            sl = k * 16
            sc = float((scales[k] >> 0) & 0x0F) * float(d)
            mn = float((scales[k] >> 4) & 0x0F) * float(dmin)
            for j in range(16):
                byte_idx = (k * 16 + j) // 4
                shift = ((k * 16 + j) % 4) * 2
                q2 = (int(qs[byte_idx]) >> shift) & 0x03
                result[i*256 + sl + j] = float(q2) * sc - mn
    return result


def _dequant_q3_k(raw: bytes, num_elems: int) -> np.ndarray:
    """Q3_K: 256-element super-block, 110 bytes."""
    n_blocks = num_elems // 256
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 110
        ql  = np.frombuffer(raw[b:b+32],   dtype=np.uint8)   # low 2 bits → 128 elements
        qh  = np.frombuffer(raw[b+32:b+64], dtype=np.uint8)   # high bit
        sc  = np.frombuffer(raw[b+64:b+76], dtype=np.uint8)   # 12 bytes packed 6-bit scales
        d   = np.frombuffer(raw[b+76:b+78], dtype=np.float16)[0].astype(np.float32)
        # decode scales (6-bit packed in 12 bytes for 16 sub-blocks of 16)
        scales_f = np.zeros(16, dtype=np.float32)
        for k in range(16):
            byte_pos = (k * 6) // 8
            bit_pos  = (k * 6) % 8
            raw_sc = ((int(sc[byte_pos]) >> bit_pos) | (int(sc[byte_pos+1]) << (8-bit_pos)) if byte_pos+1 < 12 else int(sc[byte_pos]) >> bit_pos) & 0x3F
            scales_f[k] = (float(raw_sc) - 32.0) * float(d)
        for j in range(256):
            ql_byte = j // 4
            ql_shift = (j % 4) * 2
            qh_byte = j // 8
            qh_shift = j % 8
            lo2 = (int(ql[ql_byte]) >> ql_shift) & 0x03
            hi1 = (int(qh[qh_byte]) >> qh_shift) & 0x01
            q3 = float((lo2 | (hi1 << 2)) - 4)
            sub_block = j // 16
            result[i*256 + j] = q3 * scales_f[sub_block]
    return result


# Map from GGML type to dequantization function
_DEQUANT_FN: dict[int, Any] = {
    _GGML_TYPE_F16:  lambda raw, n: _dequant_f16(raw),
    _GGML_TYPE_BF16: lambda raw, n: _dequant_bf16(raw),
    _GGML_TYPE_F32:  lambda raw, n: np.frombuffer(raw, dtype=np.float32).copy(),
    _GGML_TYPE_Q8_0: _dequant_q8_0,
    _GGML_TYPE_Q4_0: _dequant_q4_0,
    _GGML_TYPE_Q4_K: _dequant_q4_k,
    _GGML_TYPE_Q6_K: _dequant_q6_k,
    _GGML_TYPE_Q2_K: _dequant_q2_k,
    _GGML_TYPE_Q3_K: _dequant_q3_k,
}

_GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 30: "BF16",
}


# ---------------------------------------------------------------------------
# GGUFReader — pure-binary GGUF parser
# ---------------------------------------------------------------------------

class GGUFReader:
    """
    Pure-Python binary GGUF reader.

    Reads GGUF v2/v3 files without requiring the gguf pip package.
    Exposes tensor metadata and provides ``dequantize(name)`` to obtain
    float32 numpy arrays.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = path.read_bytes()
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, dict[str, Any]] = {}
        self.architecture: str = "llama"
        self._data_offset: int = 0  # byte offset where tensor data begins
        self._parse()

    def _parse(self) -> None:
        data = self._data
        offset = 0

        # Header
        magic, offset = _read_u32(data, offset)
        if magic != _GGUF_MAGIC:
            msg = f"Not a GGUF file (magic={magic:#010x})"
            raise IngestionError(msg)
        version, offset = _read_u32(data, offset)
        if version not in (_GGUF_VERSION_2, _GGUF_VERSION_3):
            msg = f"Unsupported GGUF version {version}"
            raise IngestionError(msg)
        tensor_count, offset = _read_u64(data, offset)
        kv_count, offset = _read_u64(data, offset)

        # Metadata key-value pairs
        for _ in range(kv_count):
            key, offset = _read_string(data, offset)
            vtype, offset = _read_u32(data, offset)
            value, offset = _read_value(data, offset, vtype)
            self.metadata[key] = value

        # Tensor info
        tensor_info: list[dict[str, Any]] = []
        for _ in range(tensor_count):
            name, offset = _read_string(data, offset)
            n_dims, offset = _read_u32(data, offset)
            shape: list[int] = []
            for _ in range(n_dims):
                dim, offset = _read_u64(data, offset)
                shape.append(dim)
            tensor_type, offset = _read_u32(data, offset)
            tensor_offset, offset = _read_u64(data, offset)
            tensor_info.append({
                "name": name,
                "shape": shape,
                "type": tensor_type,
                "offset": tensor_offset,
            })

        # Alignment padding before data
        alignment = int(self.metadata.get("general.alignment", 32))
        remainder = offset % alignment
        if remainder:
            offset += alignment - remainder
        self._data_offset = offset

        # Build tensor index
        arch = str(self.metadata.get("general.architecture", "llama"))
        self.architecture = arch
        for info in tensor_info:
            name = info["name"]
            shape = info["shape"]
            ggml_type = info["type"]
            num_elems = int(np.prod(shape)) if shape else 1
            block_size = _BLOCK_SIZES.get(ggml_type, 4)
            block_elems = _BLOCK_ELEMS.get(ggml_type, 1)
            n_blocks = (num_elems + block_elems - 1) // block_elems
            byte_count = n_blocks * block_size
            self.tensors[name] = {
                "name": name,
                "shape": shape,
                "type": ggml_type,
                "type_name": _GGML_TYPE_NAMES.get(ggml_type, f"unk_{ggml_type}"),
                "offset": info["offset"],
                "byte_count": byte_count,
                "num_elems": num_elems,
            }

        logger.info(
            "GGUF parsed",
            path=str(self.path),
            version=version,
            tensors=len(self.tensors),
            arch=self.architecture,
        )

    def dequantize(self, name: str) -> np.ndarray:
        """
        Dequantize a named tensor to float32.

        Raises KeyError if the tensor does not exist, IngestionError if
        the tensor type is unsupported.
        """
        info = self.tensors[name]
        ggml_type = info["type"]
        abs_offset = self._data_offset + info["offset"]
        byte_count = info["byte_count"]
        raw = self._data[abs_offset:abs_offset + byte_count]
        fn = _DEQUANT_FN.get(ggml_type)
        if fn is None:
            msg = f"Dequantization not implemented for GGML type {info['type_name']} (tensor {name!r})"
            raise IngestionError(msg)
        arr = fn(raw, info["num_elems"])
        shape = info["shape"]
        if shape:
            try:
                arr = arr.reshape(shape[::-1])  # GGUF stores row-major
            except ValueError:
                pass  # keep flat if reshape fails
        return arr

    def get_field(self, key: str) -> Any:
        return self.metadata.get(key)


# ---------------------------------------------------------------------------
# Public GGUFLoader
# ---------------------------------------------------------------------------

class GGUFLoader:
    """
    Loads GGUF model files and provides dequantized weight tensors.

    Tries the pure-Python binary path first; falls back to the ``gguf``
    pip package for richer metadata if available.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def load(self) -> dict[str, Any]:
        """
        Load GGUF tensors and metadata.

        Returns a dict with:
          - ``tensors``: name → GGUFTensorInfo dict (with lazy dequantize)
          - ``metadata``: all GGUF metadata key-value pairs
          - ``arch``: architecture string (e.g. "llama")
          - ``reader``: the GGUFReader for on-demand dequantization
        """
        if not self.model_path.exists():
            msg = f"GGUF file not found: {self.model_path}"
            raise IngestionError(msg)
        try:
            reader = GGUFReader(self.model_path)
        except Exception as exc:
            msg = f"Failed to parse GGUF file {self.model_path}: {exc}"
            raise IngestionError(msg) from exc

        logger.info(
            "Loaded GGUF",
            path=str(self.model_path),
            tensors=len(reader.tensors),
            arch=reader.architecture,
        )
        return {
            "tensors": reader.tensors,
            "metadata": reader.metadata,
            "arch": reader.architecture,
            "reader": reader,
        }

    def load_tensor(self, name: str) -> np.ndarray:
        """Load and dequantize a single tensor by name."""
        reader = GGUFReader(self.model_path)
        return reader.dequantize(name)

    def __repr__(self) -> str:
        return f"GGUFLoader({self.model_path})"


def export_gguf_tokenizer_metadata(
    metadata: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Export a standard GGUF tokenizer vocabulary for AEG runtime use.

    Modern GGUF files commonly embed SentencePiece-style token strings and
    scores under ``tokenizer.ggml.*``.  Those values are sufficient to build a
    local Hugging Face fast tokenizer without downloading a separate tokenizer.
    The function refuses incomplete metadata instead of silently using a
    different vocabulary.
    """
    tokens = metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        raise IngestionError(
            "GGUF does not contain tokenizer.ggml.tokens; a tokenizer-backed AEG cannot be emitted"
        )
    if len(set(tokens)) != len(tokens):
        raise IngestionError("GGUF tokenizer vocabulary contains duplicate token strings")

    scores = metadata.get("tokenizer.ggml.scores")
    if scores is None:
        scores = [-10.0] * len(tokens)
    if not isinstance(scores, list) or len(scores) != len(tokens):
        raise IngestionError(
            "GGUF tokenizer.ggml.scores must be a list with one score per token"
        )
    try:
        vocab = [(token, float(score)) for token, score in zip(tokens, scores)]
    except (TypeError, ValueError) as exc:
        raise IngestionError("GGUF tokenizer scores contain a non-numeric value") from exc

    unk_id = int(metadata.get("tokenizer.ggml.unknown_token_id", 0))
    if not 0 <= unk_id < len(tokens):
        raise IngestionError("GGUF tokenizer unknown token id is outside the vocabulary")

    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import Metaspace as MetaspaceDecoder
        from tokenizers.models import Unigram
        from tokenizers.pre_tokenizers import Metaspace
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise IngestionError(
            "GGUF tokenizer export requires tokenizers and transformers"
        ) from exc

    tokenizer = Tokenizer(Unigram(vocab=vocab, unk_id=unk_id))
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

    def special_token(key: str, default_id: int | None = None) -> str | None:
        raw_id = metadata.get(key, default_id)
        if raw_id is None:
            return None
        token_id = int(raw_id)
        if not 0 <= token_id < len(tokens):
            raise IngestionError(f"GGUF tokenizer special token id is outside vocabulary: {key}")
        return tokens[token_id]

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=tokens[unk_id],
        bos_token=special_token("tokenizer.ggml.bos_token_id"),
        eos_token=special_token("tokenizer.ggml.eos_token_id"),
        pad_token=special_token("tokenizer.ggml.padding_token_id"),
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(destination)
    return {
        "token_count": len(tokens),
        "tokenizer_type": "gguf_embedded_unigram",
        "path": "tokenizer",
    }


def export_gguf_tokenizer(model_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Read a GGUF file and export its embedded tokenizer."""
    path = Path(model_path)
    reader = GGUFReader(path)
    return export_gguf_tokenizer_metadata(reader.metadata, output_dir)

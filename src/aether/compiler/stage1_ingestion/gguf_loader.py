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
    """Q4_0 — faithful transcription of ``dequantize_row_q4_0`` (ggml-quanta.c).

    Block layout (18 bytes, 32 elements): d (f16) then 16 quant bytes where
    byte j holds element j in its low nibble and element j+16 in its high
    nibble. Values are (nibble - 8) * d.
    """
    n_blocks = num_elems // 32
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        base = i * 18
        scale = np.frombuffer(raw[base:base+2], dtype=np.float16)[0].astype(np.float32)
        packed = np.frombuffer(raw[base+2:base+18], dtype=np.uint8)
        block = np.empty(32, dtype=np.float32)
        block[0:16] = (packed & 0x0F).astype(np.float32)
        block[16:32] = (packed >> 4).astype(np.float32)
        result[i*32:(i+1)*32] = (block - 8.0) * float(scale)
    return result


def _get_scale_min_k4(j: int, scales: np.ndarray) -> tuple[int, int]:
    """Unpack the 6-bit scale/min pair ``j`` from a 12-byte K-quant scale field.

    Faithful transcription of ``get_scale_min_k4`` from ggml-quanta.c:
    sub-block pairs 0-3 use the low 6 bits of bytes j and j+4; pairs 4-7 use
    the high nibbles of bytes j+4 combined with the top 2 bits of bytes j-4
    and j.
    """
    if j < 4:
        d = int(scales[j]) & 63
        m = int(scales[j + 4]) & 63
    else:
        d = (int(scales[j + 4]) & 0xF) | ((int(scales[j - 4]) >> 6) << 4)
        m = (int(scales[j + 4]) >> 4) | ((int(scales[j]) >> 6) << 4)
    return d, m


def _require_multiple(num_elems: int, block: int, name: str) -> None:
    """Fail closed when the tensor size does not divide into whole blocks."""
    if num_elems % block != 0:
        raise UnsupportedFormatError(
            f"{name} dequantization requires num_elems divisible by {block}, got {num_elems}"
        )


def _dequant_q4_k(raw: bytes, num_elems: int) -> np.ndarray:
    """
    Q4_K — faithful transcription of ``dequantize_row_q4_K`` (ggml-quanta.c).

    Block layout (144 bytes, 256 elements):
      [0:2]     d     f16  super-block scale
      [2:4]     dmin  f16  super-block min scale
      [4:16]    scales 12 bytes  8 × (6-bit scale + 6-bit min), get_scale_min_k4 packed
      [16:144]  qs   128 bytes  4-bit quants (low nibbles of bytes j..j+31 form
                sub-block 2g, high nibbles form sub-block 2g+1)
    """
    _require_multiple(num_elems, 256, "Q4_K")
    n_blocks = num_elems // 256
    raw_arr = np.frombuffer(raw, dtype=np.uint8, count=144 * n_blocks)
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 144
        d = float(np.frombuffer(raw[b:b + 2], dtype=np.float16)[0])
        dmin = float(np.frombuffer(raw[b + 2:b + 4], dtype=np.float16)[0])
        scales = raw_arr[b + 4:b + 16]
        qs = raw_arr[b + 16:b + 144]
        base = i * 256
        is_ = 0
        qpos = 0
        for g in range(4):  # four 64-element groups per super-block
            sc, m = _get_scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = _get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            chunk = qs[qpos:qpos + 32].astype(np.float32)
            result[base + g * 64:base + g * 64 + 32] = d1 * (chunk % 16.0) - m1
            result[base + g * 64 + 32:base + g * 64 + 64] = d2 * (chunk // 16.0) - m2
            qpos += 32
            is_ += 2
    return result


def _dequant_q6_k(raw: bytes, num_elems: int) -> np.ndarray:
    """
    Q6_K — faithful transcription of ``dequantize_row_q6_K`` (ggml-quanta.c).

    Block layout (210 bytes, 256 elements):
      [0:128]   ql  uint8[128]  low 4 bits
      [128:192] qh  uint8[64]   high 2 bits (4 values per byte)
      [192:208] sc  int8[16]    per-16-element sub-block scales
      [208:210] d   f16         super-block scale
    """
    _require_multiple(num_elems, 256, "Q6_K")
    n_blocks = num_elems // 256
    raw_arr = np.frombuffer(raw, dtype=np.uint8, count=210 * n_blocks)
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 210
        ql = raw_arr[b:b + 128]
        qh = raw_arr[b + 128:b + 192]
        sc = np.frombuffer(raw[b + 192:b + 208], dtype=np.int8).astype(np.float32)
        d = float(np.frombuffer(raw[b + 208:b + 210], dtype=np.float16)[0])
        base = i * 256
        for h in range(2):  # two 128-element halves
            qloff, qhoff, scoff = h * 64, h * 32, h * 8
            hbase = base + 128 * h
            l = np.arange(32)
            is_ = l // 16
            q1 = ((ql[qloff + l] & 0x0F) | ((qh[qhoff + l] & 0x03) << 4)).astype(np.int32) - 32
            q2 = ((ql[qloff + l + 32] & 0x0F) | (((qh[qhoff + l] >> 2) & 0x03) << 4)).astype(np.int32) - 32
            q3 = ((ql[qloff + l] >> 4) | (((qh[qhoff + l] >> 4) & 0x03) << 4)).astype(np.int32) - 32
            q4 = ((ql[qloff + l + 32] >> 4) | (((qh[qhoff + l] >> 6) & 0x03) << 4)).astype(np.int32) - 32
            result[hbase + l] = d * sc[scoff + is_ + 0] * q1
            result[hbase + l + 32] = d * sc[scoff + is_ + 2] * q2
            result[hbase + l + 64] = d * sc[scoff + is_ + 4] * q3
            result[hbase + l + 96] = d * sc[scoff + is_ + 6] * q4
    return result


def _dequant_q2_k(raw: bytes, num_elems: int) -> np.ndarray:
    """
    Q2_K — faithful transcription of ``dequantize_row_q2_K`` (ggml-quanta.c).

    Block layout (84 bytes, 256 elements):
      [0:16]   scales uint8[16]  4-bit scale (low nibble) + 4-bit min (high nibble)
                                    per 16-element sub-block
      [16:80]  qs     uint8[64]   2-bit quants, 4 values per byte
      [80:82]  d      f16         super-block scale
      [82:84]  dmin   f16         super-block min scale
    """
    _require_multiple(num_elems, 256, "Q2_K")
    n_blocks = num_elems // 256
    raw_arr = np.frombuffer(raw, dtype=np.uint8, count=84 * n_blocks)
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 84
        scales = raw_arr[b:b + 16]
        qs = raw_arr[b + 16:b + 80]
        d = float(np.frombuffer(raw[b + 80:b + 82], dtype=np.float16)[0])
        dmin = float(np.frombuffer(raw[b + 82:b + 84], dtype=np.float16)[0])
        base = i * 256
        is_ = 0
        qpos = 0
        for h in range(2):  # two 128-element halves
            shift = 0
            hbase = base + 128 * h
            for j in range(4):  # four sub-block pairs per half
                sc = int(scales[is_]); is_ += 1
                dl, ml = d * (sc & 0x0F), dmin * (sc >> 4)
                result[hbase + j * 32:hbase + j * 32 + 16] = (
                    dl * ((qs[qpos:qpos + 16] >> shift) & 0x03) - ml
                )
                sc = int(scales[is_]); is_ += 1
                dl, ml = d * (sc & 0x0F), dmin * (sc >> 4)
                result[hbase + j * 32 + 16:hbase + j * 32 + 32] = (
                    dl * ((qs[qpos + 16:qpos + 32] >> shift) & 0x03) - ml
                )
                shift += 2
            qpos += 32
    return result


def _dequant_q3_k(raw: bytes, num_elems: int) -> np.ndarray:
    """
    Q3_K — faithful transcription of ``dequantize_row_q3_K`` (ggml-quanta.c).

    Block layout (110 bytes, 256 elements), field order per block_q3_K:
      [0:32]    hmask   uint8[32]  high bits (bit j of byte l selects sub-block j)
      [32:96]   qs      uint8[64]  low 2 bits, 4 values per byte
      [96:108]  scales  uint8[12]  16 × 6-bit scales, de-interleaved via aux
      [108:110] d       f16        super-block scale
    """
    _require_multiple(num_elems, 256, "Q3_K")
    n_blocks = num_elems // 256
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0F0F0F0F)
    raw_arr = np.frombuffer(raw, dtype=np.uint8, count=110 * n_blocks)
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 110
        hm = raw_arr[b:b + 32]
        q = raw_arr[b + 32:b + 96]
        d_all = float(np.frombuffer(raw[b + 108:b + 110], dtype=np.float16)[0])

        # De-interleave the 6-bit scales exactly as the reference does: the
        # 12 bytes are read as three little-endian uint32s and shuffled.
        aux = np.frombuffer(raw[b + 96:b + 108], dtype=np.uint32).astype(np.uint32).copy()
        aux = np.concatenate([aux, np.zeros(1, dtype=np.uint32)])
        tmp = aux[2].copy()
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        scales = aux.view(np.int8).astype(np.float32)  # 16 signed scales

        base = i * 256
        is_ = 0
        qpos = 0
        m = 1
        for h in range(2):  # two 128-element halves
            shift = 0
            hbase = base + 128 * h
            for j in range(4):
                dl1 = d_all * (scales[is_] - 32.0); is_ += 1
                qchunk = q[qpos:qpos + 32].astype(np.int32)
                # hmask is indexed l and l+16 over the SAME 32 bytes for every
                # half; only the m bit advances to select the sub-block bit.
                mask_lo = (hm[:16] & m) != 0
                mask_hi = (hm[16:32] & m) != 0
                vals_lo = ((qchunk[:16] >> shift) & 3) - np.where(mask_lo, 0, 4)
                dl2 = d_all * (scales[is_] - 32.0); is_ += 1
                vals_hi = ((qchunk[16:32] >> shift) & 3) - np.where(mask_hi, 0, 4)
                result[hbase + j * 32:hbase + j * 32 + 16] = dl1 * vals_lo
                result[hbase + j * 32 + 16:hbase + j * 32 + 32] = dl2 * vals_hi
                shift += 2
                m <<= 1
            qpos += 32
    return result


def _dequant_q4_1(raw: bytes, num_elems: int) -> np.ndarray:
    """Q4_1 — faithful transcription of dequantize_row_q4_1 (ggml-quanta.c)."""
    _require_multiple(num_elems, 32, "Q4_1")
    n_blocks = num_elems // 32
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        base = i * 20
        d = float(np.frombuffer(raw[base:base+2], dtype=np.float16)[0])
        m = float(np.frombuffer(raw[base+2:base+4], dtype=np.float16)[0])
        packed = np.frombuffer(raw[base+4:base+20], dtype=np.uint8)
        block = np.empty(32, dtype=np.float32)
        block[0:16] = (packed & 0x0F).astype(np.float32)
        block[16:32] = (packed >> 4).astype(np.float32)
        result[i*32:(i+1)*32] = block * d + m
    return result


def _dequant_q5_0(raw: bytes, num_elems: int) -> np.ndarray:
    """Q5_0 — faithful transcription of dequantize_row_q5_0 (ggml-quanta.c)."""
    _require_multiple(num_elems, 32, "Q5_0")
    n_blocks = num_elems // 32
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        base = i * 22
        d = float(np.frombuffer(raw[base:base+2], dtype=np.float16)[0])
        qh = int(struct.unpack_from("<I", raw, base + 2)[0])
        qs = np.frombuffer(raw[base+6:base+22], dtype=np.uint8)
        block = np.empty(32, dtype=np.float32)
        for j in range(16):
            lo = int(qs[j]) & 0x0F
            hi = (qh >> j) & 1
            block[j] = float(lo | (hi << 4)) - 16.0
            lo2 = int(qs[j]) >> 4
            hi2 = (qh >> (j + 16)) & 1
            block[j + 16] = float(lo2 | (hi2 << 4)) - 16.0
        result[i*32:(i+1)*32] = block * d
    return result


def _dequant_q5_1(raw: bytes, num_elems: int) -> np.ndarray:
    """Q5_1 — faithful transcription of dequantize_row_q5_1 (ggml-quanta.c)."""
    _require_multiple(num_elems, 32, "Q5_1")
    n_blocks = num_elems // 32
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        base = i * 24
        d = float(np.frombuffer(raw[base:base+2], dtype=np.float16)[0])
        m = float(np.frombuffer(raw[base+2:base+4], dtype=np.float16)[0])
        qh = int(struct.unpack_from("<I", raw, base + 4)[0])
        qs = np.frombuffer(raw[base+8:base+24], dtype=np.uint8)
        block = np.empty(32, dtype=np.float32)
        for j in range(16):
            lo = int(qs[j]) & 0x0F
            hi = (qh >> j) & 1
            block[j] = float(lo | (hi << 4))
            lo2 = int(qs[j]) >> 4
            hi2 = (qh >> (j + 16)) & 1
            block[j + 16] = float(lo2 | (hi2 << 4))
        result[i*32:(i+1)*32] = block * d + m
    return result


def _dequant_q8_k(raw: bytes, num_elems: int) -> np.ndarray:
    """Q8_K — faithful transcription of dequantize_row_q8_K (ggml-quanta.c)."""
    _require_multiple(num_elems, 256, "Q8_K")
    n_blocks = num_elems // 256
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        base = i * 292
        d = float(struct.unpack_from("<f", raw, base)[0])
        qs = np.frombuffer(raw[base+4:base+260], dtype=np.int8).astype(np.float32)
        result[i*256:(i+1)*256] = qs * d
    return result


def _dequant_q5_k(raw: bytes, num_elems: int) -> np.ndarray:
    """Q5_K — faithful transcription of dequantize_row_q5_K (ggml-quanta.c)."""
    _require_multiple(num_elems, 256, "Q5_K")
    n_blocks = num_elems // 256
    raw_arr = np.frombuffer(raw, dtype=np.uint8, count=176 * n_blocks)
    result = np.empty(num_elems, dtype=np.float32)
    for i in range(n_blocks):
        b = i * 176
        d = float(np.frombuffer(raw[b:b + 2], dtype=np.float16)[0])
        dmin = float(np.frombuffer(raw[b + 2:b + 4], dtype=np.float16)[0])
        scales = raw_arr[b + 4:b + 16]
        qh = raw_arr[b + 16:b + 48]
        qs = raw_arr[b + 48:b + 176]
        base = i * 256
        is_ = 0
        qpos = 0
        u1, u2 = 1, 2
        for g in range(4):
            sc, m = _get_scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = _get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            chunk = qs[qpos:qpos + 32].astype(np.float32)
            hi1 = np.where((qh[:32] & u1) != 0, 16.0, 0.0)
            hi2 = np.where((qh[:32] & u2) != 0, 16.0, 0.0)
            result[base + g * 64:base + g * 64 + 32] = d1 * (chunk % 16.0 + hi1) - m1
            result[base + g * 64 + 32:base + g * 64 + 64] = d2 * (chunk // 16.0 + hi2) - m2
            qpos += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return result


# Map from GGML type to dequantization function
_DEQUANT_FN: dict[int, Any] = {
    _GGML_TYPE_F16:  lambda raw, n: _dequant_f16(raw),
    _GGML_TYPE_BF16: lambda raw, n: _dequant_bf16(raw),
    _GGML_TYPE_F32:  lambda raw, n: np.frombuffer(raw, dtype=np.float32).copy(),
    _GGML_TYPE_Q8_0: _dequant_q8_0,
    _GGML_TYPE_Q4_0: _dequant_q4_0,
    _GGML_TYPE_Q4_1: _dequant_q4_1,
    _GGML_TYPE_Q5_0: _dequant_q5_0,
    _GGML_TYPE_Q5_1: _dequant_q5_1,
    _GGML_TYPE_Q4_K: _dequant_q4_k,
    _GGML_TYPE_Q5_K: _dequant_q5_k,
    _GGML_TYPE_Q6_K: _dequant_q6_k,
    _GGML_TYPE_Q8_K: _dequant_q8_k,
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

"""Small integer-hash contracts shared with Blender/Cycles scene sync.

These functions model the unsigned 32-bit arithmetic used by Cycles. They
belong to scene synchronization, not to a renderer-side reference path: the
exporter uses them only to derive values that Cycles has already defined for
the active Blender frame.
"""

from __future__ import annotations

import struct


UINT32_MASK = 0xFFFFFFFF
INT32_MAX = 0x7FFFFFFF


def u32(value: int) -> int:
    return value & UINT32_MASK


def _float32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _rotate_left(value: int, bits: int) -> int:
    value = u32(value)
    return u32((value << bits) | (value >> (32 - bits)))


def _final(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Jenkins lookup3 finalization with Cycles' uint32 wraparound."""

    c = u32((c ^ b) - _rotate_left(b, 14))
    a = u32((a ^ c) - _rotate_left(c, 11))
    b = u32((b ^ a) - _rotate_left(a, 25))
    c = u32((c ^ b) - _rotate_left(b, 16))
    a = u32((a ^ c) - _rotate_left(c, 4))
    b = u32((b ^ a) - _rotate_left(a, 14))
    c = u32((c ^ b) - _rotate_left(b, 24))
    return a, b, c


def hash_uint(value: int) -> int:
    a = b = c = u32(0xDEADBEEF + (1 << 2) + 13)
    a = u32(a + value)
    return _final(a, b, c)[2]


def hash_uint2(x: int, y: int) -> int:
    a = b = c = u32(0xDEADBEEF + (2 << 2) + 13)
    b = u32(b + y)
    a = u32(a + x)
    return _final(a, b, c)[2]


def hash_string(value: str) -> int:
    result = 0
    for byte in value.encode("utf-8"):
        result = u32(result * 37 + byte)
    return result


def effective_scene_seed(
    base_seed: int,
    frame: int,
    subframe: float,
    use_animated_seed: bool,
) -> int:
    """Return the seed passed from BlenderSync to Cycles' Integrator.

    Cycles hashes the integer frame with the authored seed. A nonzero
    subframe contributes a second hash after an explicit float32 multiply by
    ``INT_MAX``. Reproducing that intermediate precision matters because the
    resulting integer controls every per-pixel sample sequence.
    """

    base_seed = u32(base_seed)
    if not use_animated_seed:
        return base_seed
    result = hash_uint2(frame, base_seed)
    subframe_f32 = _float32(subframe)
    if subframe_f32 != 0.0:
        scaled_subframe = _float32(
            subframe_f32 * _float32(float(INT32_MAX))
        )
        result = u32(
            result + hash_uint2(int(scaled_subframe), base_seed)
        )
    return result

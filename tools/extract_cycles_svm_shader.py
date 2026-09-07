#!/usr/bin/env python3
"""Relocate one shader from an external Cycles 5.2.1 PSYSVM52 dump.

Only the three NODE_SHADER_JUMP targets change. Node words and typed payloads
are copied verbatim; this tool neither compiles nor evaluates a shader.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys


def extract(path: Path, shader: int) -> tuple[str, list[int]]:
    data = memoryview(path.read_bytes())
    if len(data) < 24 or data[:8] != b"PSYSVM52":
        raise ValueError("not a PSYSVM52 dump")
    version, count, shader_count = struct.unpack_from("<IQI", data, 8)
    if version != 1 or not 0 <= shader < shader_count:
        raise ValueError("unsupported version or shader index")
    names = {}
    cursor = 24
    for _ in range(shader_count):
        index, length = struct.unpack_from("<II", data, cursor)
        cursor += 8
        names[index] = bytes(data[cursor:cursor + length]).decode("utf-8")
        cursor += length
    if len(data) - cursor != count * 4:
        raise ValueError("truncated or trailing word data")
    words = struct.unpack_from(f"<{count}I", data, cursor)
    if count < 4 * shader_count:
        raise ValueError("truncated global jump table")
    jump = words[shader * 4:shader * 4 + 4]
    start = jump[1]
    end = words[(shader + 1) * 4 + 1] if shader + 1 < shader_count else count
    if jump[0] != 1 or not 4 * shader_count <= start < end <= count:
        raise ValueError("invalid shader interval")
    if not all(start <= offset < end for offset in jump[1:]):
        raise ValueError("jump target outside shader interval")
    return names[shader], [jump[0], *(offset - start + 4 for offset in jump[1:]),
                           *words[start:end]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("shader", type=int)
    args = parser.parse_args()
    name, words = extract(args.dump, args.shader)
    print(f"shader {args.shader}: {name!r}, {len(words)} local words", file=sys.stderr)
    print(len(words))
    print(" ".join(f"{word:08x}" for word in words))


if __name__ == "__main__":
    main()

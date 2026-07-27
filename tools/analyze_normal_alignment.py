"""Diagnose image orientation and normal-basis mismatches in a differential.

This is a debugging aid only; it never changes or accepts a comparison.
Run with Blender's bundled Python:

    python analyze_normal_alignment.py cycles.exr psycles-normal.pfm
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy
import OpenImageIO as oiio


def _read(path: pathlib.Path) -> tuple[numpy.ndarray, list[str]]:
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(oiio.geterror())
    try:
        spec = source.spec()
        pixels = numpy.asarray(
            source.read_image(format=oiio.FLOAT),
            dtype=numpy.float32,
        ).reshape(spec.height, spec.width, spec.nchannels)
        return pixels, list(spec.channelnames)
    finally:
        source.close()


def _main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: analyze_normal_alignment.py cycles.exr "
            "psycles-normal.pfm"
        )
    cycles, names = _read(pathlib.Path(sys.argv[1]))
    actual, _ = _read(pathlib.Path(sys.argv[2]))
    indices = [
        next(
            index
            for index, name in enumerate(names)
            if name.endswith(f".Normal.{channel}")
        )
        for channel in "XYZ"
    ]
    reference = cycles[:, :, indices]
    actual = actual[:, :, :3]
    candidates = []
    for flip_name, oriented in {
        "identity": actual,
        "flip_y": actual[::-1, :, :],
        "flip_x": actual[:, ::-1, :],
        "flip_xy": actual[::-1, ::-1, :],
    }.items():
        for permutation in itertools.permutations(range(3)):
            permuted = oriented[:, :, permutation]
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                transformed = permuted * numpy.asarray(
                    signs, dtype=numpy.float32
                )
                difference = transformed - reference
                rmse = float(numpy.sqrt(numpy.mean(difference * difference)))
                candidates.append(
                    (rmse, flip_name, permutation, signs)
                )
    for rmse, flip_name, permutation, signs in sorted(candidates)[:12]:
        print(
            f"rmse={rmse:.9f} orientation={flip_name} "
            f"permutation={permutation} signs={signs}"
        )


if __name__ == "__main__":
    _main()

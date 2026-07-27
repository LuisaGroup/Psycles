"""Write a viewable PNG from a linear Cycles/Psycles EXR pass.

Run this with Blender's bundled Python, which ships OpenImageIO and NumPy:

    blender/4.5/python/bin/python3.11 tools/exr_preview.py \
        render.exr preview.png [channel-prefix] [exposure]

The default channel prefix is ``View Layer.Combined``. This conversion is
only for visual inspection; numerical comparisons always read the linear EXR.
"""

from __future__ import annotations

import pathlib
import sys

import numpy
import OpenImageIO as oiio


def _main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: exr_preview.py input.exr output.png "
            "[channel-prefix] [exposure]"
        )
    source = pathlib.Path(sys.argv[1])
    destination = pathlib.Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else "View Layer.Combined"
    exposure = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    image_input = oiio.ImageInput.open(str(source))
    if image_input is None:
        raise RuntimeError(oiio.geterror())
    try:
        specification = image_input.spec()
        pixels = image_input.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise RuntimeError(image_input.geterror())
        pixels = numpy.asarray(pixels, dtype=numpy.float32).reshape(
            specification.height,
            specification.width,
            specification.nchannels,
        )
        requested = [f"{prefix}.{channel}" for channel in "RGB"]
        indices = [specification.channelnames.index(name) for name in requested]
        linear = numpy.maximum(pixels[:, :, indices] * (2.0**exposure), 0.0)
        # Stable photographic preview only. The EXR remains the source of truth.
        display = linear / (1.0 + linear)
        display = numpy.where(
            display <= 0.0031308,
            display * 12.92,
            1.055 * numpy.power(display, 1.0 / 2.4) - 0.055,
        )
        encoded = numpy.ascontiguousarray(
            numpy.clip(display * 255.0 + 0.5, 0.0, 255.0),
            dtype=numpy.uint8,
        )
    finally:
        image_input.close()

    destination.parent.mkdir(parents=True, exist_ok=True)
    image_output = oiio.ImageOutput.create(str(destination))
    if image_output is None:
        raise RuntimeError(oiio.geterror())
    output_specification = oiio.ImageSpec(
        specification.width,
        specification.height,
        3,
        oiio.UINT8,
    )
    if not image_output.open(str(destination), output_specification):
        raise RuntimeError(image_output.geterror())
    try:
        if not image_output.write_image(encoded):
            raise RuntimeError(image_output.geterror())
    finally:
        image_output.close()


if __name__ == "__main__":
    _main()

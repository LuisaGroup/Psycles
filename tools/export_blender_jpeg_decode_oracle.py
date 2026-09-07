"""Generate a small JPEG/decoded-byte fixture with original Blender image I/O.

Run Blender 5.2.1 --background --python-exit-code 1 --python this_file -- OUTPUT.
This exercises Blender's ImBuf codec only; it does not implement a renderer.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import pathlib
import sys

import bpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if bpy.app.version[:3] != (5, 2, 1):
        raise RuntimeError("The image oracle must use Blender 5.2.1")
    args.output.mkdir(parents=True, exist_ok=True)
    width, height = 32, 24
    # Nonconstant chroma and block edges distinguish decoders and row order.
    source = bpy.data.images.new("JPEG codec witness", width, height, alpha=False)
    values = array.array("f")
    for y in range(height):
        for x in range(width):
            values.extend(((x * 17 + y * 29) % 256 / 255.0,
                           (x * 71 + y * 13) % 256 / 255.0,
                           (x * 31 + y * 53) % 256 / 255.0, 1.0))
    source.pixels.foreach_set(values)
    source.file_format = "JPEG"
    jpeg = args.output / "blender_jpeg_decode.jpg"
    source.filepath_raw = str(jpeg)
    source.save()
    decoded = bpy.data.images.load(str(jpeg), check_existing=False)
    assert not decoded.is_float and decoded.channels == 4
    assert tuple(decoded.size) == (width, height)
    pixels = array.array("f", [0.0]) * len(decoded.pixels)
    decoded.pixels.foreach_get(pixels)
    # rna_Image_pixels_get reads the original byte buffer divided by 255,
    # without color management. Canonical encoded-file rows are top-down.
    rgba = bytes(round(pixels[(y * width + x) * 4 + c] * 255.0)
                 for y in reversed(range(height))
                 for x in range(width) for c in range(4))
    output = args.output / "blender_jpeg_decode.rgba"
    output.write_bytes(rgba)
    record = {"blender": bpy.app.version_string,
              "build_hash": bpy.app.build_hash.decode(),
              "width": width, "height": height, "channels": 4,
              "row_order": "top-down", "storage": "unorm8",
              "source": "generated integer RGB pattern in this script",
              "jpeg_sha256": hashlib.sha256(jpeg.read_bytes()).hexdigest(),
              "rgba_sha256": hashlib.sha256(rgba).hexdigest()}
    (args.output / "blender_jpeg_decode.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()

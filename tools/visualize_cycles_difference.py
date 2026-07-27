"""Visualize a Cycles/Psycles linear-pass difference.

Run with Blender's bundled Python:

    blender/4.5/python/bin/python3.11 \
      tools/visualize_cycles_difference.py \
      cycles.exr Combined psycles-combined.pfm output.png

The output contains Cycles, Psycles, and an automatically normalized absolute
error heatmap from left to right. Numerical acceptance still uses
``compare_cycles.py``; this image is only a spatial diagnostic.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import OpenImageIO as oiio


_PASS_CHANNELS = {
    "Combined": ("R", "G", "B"),
    "Normal": ("X", "Y", "Z"),
    "DiffCol": ("R", "G", "B"),
}


def _read(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(f"could not open {path}: {oiio.geterror()}")
    try:
        spec = source.spec()
        pixels = source.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise RuntimeError(source.geterror())
        return (
            np.asarray(pixels, dtype=np.float32).reshape(
                spec.height, spec.width, spec.nchannels
            ),
            list(spec.channelnames),
        )
    finally:
        source.close()


def _cycles_pass(
    pixels: np.ndarray, channels: list[str], pass_name: str
) -> np.ndarray:
    indices = []
    for suffix in _PASS_CHANNELS[pass_name]:
        matches = [
            index
            for index, name in enumerate(channels)
            if name.endswith(f".{pass_name}.{suffix}")
            or name == f"{pass_name}.{suffix}"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"could not resolve {pass_name}.{suffix}: {channels}"
            )
        indices.append(matches[0])
    return pixels[:, :, indices]


def _srgb(linear: np.ndarray) -> np.ndarray:
    positive = np.maximum(linear, 0.0)
    return np.where(
        positive <= 0.0031308,
        positive * 12.92,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    )


def _display(values: np.ndarray, pass_name: str) -> np.ndarray:
    if pass_name == "Normal":
        return np.clip(values * 0.5 + 0.5, 0.0, 1.0)
    if pass_name == "Combined":
        values = np.maximum(values, 0.0)
        values = values / (1.0 + values)
    return np.clip(_srgb(values), 0.0, 1.0)


def _heatmap(error: np.ndarray) -> tuple[np.ndarray, float]:
    finite = error[np.isfinite(error)]
    scale = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    scale = max(scale, 1.0e-20)
    value = np.clip(error / scale, 0.0, 1.0)
    stops = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    position = value * float(len(stops) - 1)
    lower = np.minimum(position.astype(np.int32), len(stops) - 2)
    fraction = (position - lower)[..., None]
    return (
        stops[lower] * (1.0 - fraction) + stops[lower + 1] * fraction,
        scale,
    )


def _main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: visualize_cycles_difference.py "
            "cycles.exr Pass psycles.pfm output.png"
        )
    cycles_path = pathlib.Path(sys.argv[1])
    pass_name = sys.argv[2]
    psycles_path = pathlib.Path(sys.argv[3])
    output_path = pathlib.Path(sys.argv[4])
    if pass_name not in _PASS_CHANNELS:
        raise ValueError(f"unsupported pass: {pass_name}")

    cycles_pixels, cycles_channels = _read(cycles_path)
    psycles_pixels, _ = _read(psycles_path)
    reference = _cycles_pass(
        cycles_pixels, cycles_channels, pass_name
    )
    actual = psycles_pixels[:, :, : reference.shape[2]]
    if psycles_path.suffix.lower() == ".pfm":
        actual = actual[::-1, :, :]
    if actual.shape != reference.shape:
        raise RuntimeError(
            f"shape mismatch: {reference.shape} != {actual.shape}"
        )

    error = np.sqrt(np.mean((actual - reference) ** 2, axis=2))
    heatmap, scale = _heatmap(error)
    panels = np.concatenate(
        [
            _display(reference, pass_name),
            _display(actual, pass_name),
            heatmap,
        ],
        axis=1,
    )
    enlargement = max(1, 768 // panels.shape[1])
    panels = np.repeat(
        np.repeat(panels, enlargement, axis=0),
        enlargement,
        axis=1,
    )
    encoded = np.ascontiguousarray(
        np.clip(panels * 255.0 + 0.5, 0.0, 255.0),
        dtype=np.uint8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination = oiio.ImageOutput.create(str(output_path))
    if destination is None:
        raise RuntimeError(oiio.geterror())
    specification = oiio.ImageSpec(
        encoded.shape[1], encoded.shape[0], 3, oiio.UINT8
    )
    specification.attribute("psycles:error_p99_scale", scale)
    if not destination.open(str(output_path), specification):
        raise RuntimeError(destination.geterror())
    try:
        if not destination.write_image(encoded):
            raise RuntimeError(destination.geterror())
    finally:
        destination.close()
    print(f"{pass_name} heatmap P99 scale: {scale:.9g}")


if __name__ == "__main__":
    _main()

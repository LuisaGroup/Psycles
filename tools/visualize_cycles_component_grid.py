"""Visualize Cycles and Psycles light components on shared row scales.

Run with Blender's bundled Python:

    blender/4.5/python/bin/python3.11 \
      tools/visualize_cycles_component_grid.py \
      cycles.exr psycles-output-stem output.png

The Psycles stem resolves the linear PFM files emitted by
``psycles_render_blender_scene``. Each row uses one display scale shared by
the Cycles and Psycles panels. The third panel is an independently normalized
linear absolute-error heatmap; numerical acceptance still uses
``compare_cycles.py``.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import OpenImageIO as oiio


_PASS_CHANNELS = {
    "Combined": ("R", "G", "B"),
    "DiffCol": ("R", "G", "B"),
    "DiffDir": ("R", "G", "B"),
    "DiffInd": ("R", "G", "B"),
    "GlossCol": ("R", "G", "B"),
    "GlossDir": ("R", "G", "B"),
    "GlossInd": ("R", "G", "B"),
    "Emit": ("R", "G", "B"),
    "Env": ("R", "G", "B"),
}

_LUMINANCE = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class Component:
    label: str
    cycles: np.ndarray
    psycles: np.ndarray


def _read(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(f"could not open {path}: {oiio.geterror()}")
    try:
        specification = source.spec()
        pixels = source.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise RuntimeError(source.geterror())
        image = np.asarray(pixels, dtype=np.float32).reshape(
            specification.height,
            specification.width,
            specification.nchannels,
        )
        return image, list(specification.channelnames)
    finally:
        source.close()


def _cycles_pass(
    pixels: np.ndarray,
    channel_names: list[str],
    pass_name: str,
) -> np.ndarray:
    indices: list[int] = []
    for suffix in _PASS_CHANNELS[pass_name]:
        candidates = [
            index
            for index, name in enumerate(channel_names)
            if name.endswith(f".{pass_name}.{suffix}")
            or name == f"{pass_name}.{suffix}"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"could not resolve {pass_name}.{suffix}: {channel_names}"
            )
        indices.append(candidates[0])
    return np.ascontiguousarray(pixels[:, :, indices], dtype=np.float32)


def _psycles_pass(stem: pathlib.Path, suffix: str) -> np.ndarray:
    pixels, _ = _read(
        stem.parent / f"{stem.name}-{suffix}.pfm"
    )
    return np.ascontiguousarray(pixels[::-1, :, :3], dtype=np.float32)


def _srgb(linear: np.ndarray) -> np.ndarray:
    positive = np.maximum(linear, 0.0)
    return np.where(
        positive <= 0.0031308,
        positive * 12.92,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    )


def _display_pair(
    reference: np.ndarray,
    actual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    finite = np.concatenate(
        [
            reference[np.isfinite(reference)],
            actual[np.isfinite(actual)],
        ]
    )
    positive = finite[finite > 0.0]
    scale = (
        float(np.percentile(positive, 99.0))
        if positive.size
        else 1.0
    )
    scale = max(scale, 1.0e-20)
    return (
        np.clip(_srgb(reference / scale), 0.0, 1.0),
        np.clip(_srgb(actual / scale), 0.0, 1.0),
        scale,
    )


def _heatmap(
    reference: np.ndarray,
    actual: np.ndarray,
) -> tuple[np.ndarray, float]:
    error = np.sqrt(np.mean((actual - reference) ** 2, axis=2))
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
        stops[lower] * (1.0 - fraction) +
        stops[lower + 1] * fraction,
        scale,
    )


def _mean_ratio(reference: np.ndarray, actual: np.ndarray) -> float:
    reference_mean = float(np.mean(reference @ _LUMINANCE))
    actual_mean = float(np.mean(actual @ _LUMINANCE))
    return actual_mean / max(abs(reference_mean), 1.0e-20)


def _enlarge(image: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(
        np.repeat(image, factor, axis=0),
        factor,
        axis=1,
    )


def _write(
    path: pathlib.Path,
    pixels: np.ndarray,
    labels: list[tuple[int, int, str, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unlabeled_path = path.with_name(
        f"{path.stem}-unlabeled{path.suffix}"
    )
    encoded = np.ascontiguousarray(
        np.clip(pixels * 255.0 + 0.5, 0.0, 255.0),
        dtype=np.uint8,
    )
    destination = oiio.ImageOutput.create(str(unlabeled_path))
    if destination is None:
        raise RuntimeError(oiio.geterror())
    specification = oiio.ImageSpec(
        pixels.shape[1],
        pixels.shape[0],
        3,
        oiio.UINT8,
    )
    if not destination.open(str(unlabeled_path), specification):
        raise RuntimeError(destination.geterror())
    try:
        if not destination.write_image(encoded):
            raise RuntimeError(destination.geterror())
    finally:
        destination.close()

    command = [
        "/usr/bin/convert",
        str(unlabeled_path),
        "-font",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "-fill",
        "#f5f5f5",
        "-stroke",
        "none",
    ]
    for x, y, text, size in labels:
        command.extend(
            [
                "-pointsize",
                str(size),
                "-annotate",
                f"+{x}+{y + size}",
                text,
            ]
        )
    command.append(str(path))
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ImageMagick text rendering failed: {error.stderr}"
        ) from error
    finally:
        unlabeled_path.unlink(missing_ok=True)


def _main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: visualize_cycles_component_grid.py "
            "cycles.exr psycles-stem output.png"
        )
    cycles_path = pathlib.Path(sys.argv[1])
    psycles_stem = pathlib.Path(sys.argv[2])
    output_path = pathlib.Path(sys.argv[3])

    cycles_pixels, cycles_channels = _read(cycles_path)
    cycles = {
        name: _cycles_pass(cycles_pixels, cycles_channels, name)
        for name in _PASS_CHANNELS
    }
    psycles = {
        "Combined": _psycles_pass(psycles_stem, "combined"),
        "DiffCol": _psycles_pass(psycles_stem, "albedo"),
        "DiffDir": _psycles_pass(psycles_stem, "diffuse-direct"),
        "DiffInd": _psycles_pass(psycles_stem, "diffuse-indirect"),
        "GlossCol": _psycles_pass(psycles_stem, "glossy-color"),
        "GlossDir": _psycles_pass(psycles_stem, "glossy-direct"),
        "GlossInd": _psycles_pass(psycles_stem, "glossy-indirect"),
        "Emit": _psycles_pass(psycles_stem, "emission"),
        "Env": _psycles_pass(psycles_stem, "environment"),
    }
    components = [
        Component("Combined", cycles["Combined"], psycles["Combined"]),
        Component(
            "Diffuse direct",
            cycles["DiffCol"] * cycles["DiffDir"],
            psycles["DiffCol"] * psycles["DiffDir"],
        ),
        Component(
            "Diffuse indirect",
            cycles["DiffCol"] * cycles["DiffInd"],
            psycles["DiffCol"] * psycles["DiffInd"],
        ),
        Component(
            "Glossy direct",
            cycles["GlossCol"] * cycles["GlossDir"],
            psycles["GlossCol"] * psycles["GlossDir"],
        ),
        Component(
            "Glossy indirect",
            cycles["GlossCol"] * cycles["GlossInd"],
            psycles["GlossCol"] * psycles["GlossInd"],
        ),
        Component("Emission", cycles["Emit"], psycles["Emit"]),
        Component("Environment", cycles["Env"], psycles["Env"]),
    ]
    shape = components[0].cycles.shape
    if any(
        component.cycles.shape != shape or
        component.psycles.shape != shape
        for component in components
    ):
        raise RuntimeError("component image shapes do not match")

    enlargement = max(1, 192 // shape[0])
    panel_height = shape[0] * enlargement
    panel_width = shape[1] * enlargement
    header_height = 64
    label_width = 220
    gutter = 10
    width = label_width + 3 * panel_width + 4 * gutter
    height = (
        header_height +
        len(components) * panel_height +
        (len(components) + 1) * gutter
    )
    output = np.full((height, width, 3), 0.022, dtype=np.float32)
    labels: list[tuple[int, int, str, int]] = [
        (gutter, 10, "Linear light-component comparison", 22),
        (
            label_width + gutter,
            36,
            "Cycles",
            17,
        ),
        (
            label_width + 2 * gutter + panel_width,
            36,
            "Psycles",
            17,
        ),
        (
            label_width + 3 * gutter + 2 * panel_width,
            36,
            "Absolute error (P99 scale)",
            17,
        ),
    ]
    for row, component in enumerate(components):
        y = header_height + gutter + row * (panel_height + gutter)
        reference_display, actual_display, display_scale = _display_pair(
            component.cycles,
            component.psycles,
        )
        error_display, error_scale = _heatmap(
            component.cycles,
            component.psycles,
        )
        panels = [
            _enlarge(reference_display, enlargement),
            _enlarge(actual_display, enlargement),
            _enlarge(error_display, enlargement),
        ]
        for column, panel in enumerate(panels):
            x = label_width + (column + 1) * gutter + column * panel_width
            output[y : y + panel_height, x : x + panel_width] = panel
        ratio = _mean_ratio(component.cycles, component.psycles)
        labels.extend(
            [
                (
                    gutter,
                    y + 14,
                    component.label,
                    17,
                ),
                (
                    gutter,
                    y + 42,
                    f"mean ratio: {ratio * 100.0:.2f}%",
                    14,
                ),
                (
                    gutter,
                    y + 66,
                    f"display P99: {display_scale:.4g}",
                    13,
                ),
                (
                    gutter,
                    y + 88,
                    f"error P99: {error_scale:.4g}",
                    13,
                ),
            ]
        )
    _write(output_path, output, labels)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    _main()

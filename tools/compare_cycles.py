"""Compare Psycles linear passes against a Cycles multilayer EXR.

Run with Blender's Python so OpenImageIO and NumPy are available:

    blender --background --python compare_cycles.py -- \
      cycles.exr report.json --triptych-dir triptychs \
      Combined=psycles.exr \
      Normal=psycles.exr \
      DiffCol=psycles.exr
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import OpenImageIO as oiio
from PIL import Image, ImageDraw, ImageFont


_PASS_CHANNELS = {
    "Combined": ("R", "G", "B"),
    "Normal": ("X", "Y", "Z"),
    "DiffCol": ("R", "G", "B"),
    "GlossCol": ("R", "G", "B"),
    "TransCol": ("R", "G", "B"),
    "DiffDir": ("R", "G", "B"),
    "DiffInd": ("R", "G", "B"),
    "GlossDir": ("R", "G", "B"),
    "GlossInd": ("R", "G", "B"),
    "TransDir": ("R", "G", "B"),
    "TransInd": ("R", "G", "B"),
    "Emit": ("R", "G", "B"),
    "Env": ("R", "G", "B"),
    "Depth": ("Z",),
    "Debug Sample Count": ("X",),
}

_CYCLES_PASS_ALIASES = {
    "DiffCol": ("DiffCol", "Diffuse Color"),
    "GlossCol": ("GlossCol", "Glossy Color"),
    "TransCol": ("TransCol", "Transmission Color"),
    "DiffDir": ("DiffDir", "Diffuse Direct"),
    "DiffInd": ("DiffInd", "Diffuse Indirect"),
    "GlossDir": ("GlossDir", "Glossy Direct"),
    "GlossInd": ("GlossInd", "Glossy Indirect"),
    "TransDir": ("TransDir", "Transmission Direct"),
    "TransInd": ("TransInd", "Transmission Indirect"),
    "Emit": ("Emit", "Emission"),
    "Env": ("Env", "Environment"),
}


def _read_image(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(f"could not open image: {path}")
    try:
        arrays: list[np.ndarray] = []
        channel_names: list[str] = []
        extent: tuple[int, int] | None = None
        subimage = 0
        while source.seek_subimage(subimage, 0):
            spec = source.spec()
            current_extent = (spec.height, spec.width)
            if extent is None:
                extent = current_extent
            elif current_extent != extent:
                raise RuntimeError(
                    f"multipart image has inconsistent extents: {path}"
                )
            pixels = source.read_image(format=oiio.FLOAT)
            if pixels is None:
                raise RuntimeError(
                    f"could not read subimage {subimage}: {path}"
                )
            arrays.append(
                np.asarray(pixels, dtype=np.float32).reshape(
                    spec.height, spec.width, spec.nchannels
                )
            )
            channel_names.extend(spec.channelnames)
            subimage += 1
        if not arrays:
            raise RuntimeError(f"image has no readable subimages: {path}")
        return np.concatenate(arrays, axis=2), channel_names
    finally:
        source.close()


def _find_cycles_channels(
    channel_names: list[str], pass_name: str
) -> list[int]:
    suffixes = _PASS_CHANNELS[pass_name]
    aliases = _CYCLES_PASS_ALIASES.get(
        pass_name, (pass_name,)
    )
    indices: list[int] = []
    for suffix in suffixes:
        candidates = [
            index
            for index, name in enumerate(channel_names)
            if any(
                name.endswith(f".{alias}.{suffix}")
                or name == f"{alias}.{suffix}"
                for alias in aliases
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"could not uniquely resolve {pass_name}.{suffix}; "
                f"available channels: {channel_names}"
            )
        indices.append(candidates[0])
    return indices


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise RuntimeError(
            f"shape mismatch: Cycles {reference.shape}, "
            f"Psycles {actual.shape}"
        )
    finite = np.isfinite(reference) & np.isfinite(actual)
    valid_pixels = np.all(finite, axis=2)
    invalid_pixels = int(valid_pixels.size - np.count_nonzero(valid_pixels))
    if not np.any(valid_pixels):
        raise RuntimeError("no finite pixels to compare")
    difference = actual[valid_pixels] - reference[valid_pixels]
    absolute = np.abs(difference)
    reference_values = reference[valid_pixels]
    actual_values = actual[valid_pixels]
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(reference_values * reference_values)))
    per_pixel = np.sqrt(np.mean(difference * difference, axis=1))
    reference_channel_mean = np.mean(reference_values, axis=0)
    actual_channel_mean = np.mean(actual_values, axis=0)
    channel_mean_ratio = np.divide(
        actual_channel_mean,
        reference_channel_mean,
        out=np.zeros_like(actual_channel_mean),
        where=np.abs(reference_channel_mean) > 1.0e-20,
    )
    mean_metrics: dict[str, Any] = {
        "cycles_channel_mean": reference_channel_mean.tolist(),
        "psycles_channel_mean": actual_channel_mean.tolist(),
        "channel_mean_ratio": channel_mean_ratio.tolist(),
    }
    if reference_values.shape[1] == 3:
        luminance_weights = np.asarray(
            [0.2126, 0.7152, 0.0722], dtype=np.float32
        )
        reference_luminance_mean = float(
            np.mean(reference_values @ luminance_weights)
        )
        actual_luminance_mean = float(
            np.mean(actual_values @ luminance_weights)
        )
        mean_metrics.update(
            {
                "cycles_luminance_mean": reference_luminance_mean,
                "psycles_luminance_mean": actual_luminance_mean,
                "luminance_mean_ratio": (
                    actual_luminance_mean
                    / max(abs(reference_luminance_mean), 1.0e-20)
                ),
            }
        )
    orientation_rmse = {}
    for name, candidate in {
        "identity": actual,
        "flip_y": actual[::-1, :, :],
        "flip_x": actual[:, ::-1, :],
        "flip_xy": actual[::-1, ::-1, :],
    }.items():
        candidate_finite = np.isfinite(reference) & np.isfinite(candidate)
        candidate_valid = np.all(candidate_finite, axis=2)
        candidate_difference = (
            candidate[candidate_valid] - reference[candidate_valid]
        )
        orientation_rmse[name] = float(
            np.sqrt(np.mean(candidate_difference * candidate_difference))
        )
    return {
        "shape": list(reference.shape),
        "valid_pixels": int(np.count_nonzero(valid_pixels)),
        "invalid_pixels": invalid_pixels,
        "mean_absolute_error": float(np.mean(absolute)),
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, 1.0e-20),
        "p95_pixel_rmse": float(np.percentile(per_pixel, 95.0)),
        "p99_pixel_rmse": float(np.percentile(per_pixel, 99.0)),
        "maximum_absolute_error": float(np.max(absolute)),
        "cycles_rms": reference_rms,
        "psycles_rms": float(
            math.sqrt(float(np.mean(actual[valid_pixels] ** 2)))
        ),
        **mean_metrics,
        "orientation_rmse": orientation_rmse,
    }


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0.0)
    return np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )


def _write_triptych(
    reference: np.ndarray,
    actual: np.ndarray,
    pass_name: str,
    metrics: dict[str, Any],
    output_path: pathlib.Path,
) -> dict[str, Any]:
    finite_reference = np.nan_to_num(
        reference, nan=0.0, posinf=0.0, neginf=0.0
    )
    finite_actual = np.nan_to_num(
        actual, nan=0.0, posinf=0.0, neginf=0.0
    )
    if finite_reference.shape[2] == 1:
        finite_reference = np.repeat(finite_reference, 3, axis=2)
        finite_actual = np.repeat(finite_actual, 3, axis=2)

    if pass_name == "Normal":
        display_scale = 1.0
        display_reference = np.clip(
            finite_reference * 0.5 + 0.5, 0.0, 1.0
        )
        display_actual = np.clip(
            finite_actual * 0.5 + 0.5, 0.0, 1.0
        )
        mapping = "normal_xyz_to_rgb"
    else:
        positive_values = np.concatenate(
            (
                np.maximum(finite_reference, 0.0).reshape(-1),
                np.maximum(finite_actual, 0.0).reshape(-1),
            )
        )
        positive_peak = float(
            np.percentile(positive_values, 99.5)
        )
        display_scale = (
            0.9 / positive_peak
            if positive_peak > 1.0e-20
            else 1.0
        )
        display_reference = np.clip(
            _linear_to_srgb(
                finite_reference * display_scale
            ),
            0.0,
            1.0,
        )
        display_actual = np.clip(
            _linear_to_srgb(finite_actual * display_scale),
            0.0,
            1.0,
        )
        mapping = "shared_linear_srgb"

    absolute_difference = np.abs(
        finite_actual - finite_reference
    )
    difference_peak = float(
        np.percentile(absolute_difference, 99.5)
    )
    difference_scale = (
        0.9 / difference_peak
        if difference_peak > 1.0e-20
        else 1.0
    )
    display_difference = np.clip(
        _linear_to_srgb(
            absolute_difference * difference_scale
        ),
        0.0,
        1.0,
    )

    panels = [
        ("Cycles", display_reference),
        ("Psycles", display_actual),
        (f"|Difference| x {difference_scale:.3g}", display_difference),
    ]
    height, width, _ = display_reference.shape
    pixel_scale = max(
        1,
        min(
            8,
            max(1, 512 // max(width, 1)),
            max(1, 512 // max(height, 1)),
        ),
    )
    panel_width = width * pixel_scale
    panel_height = height * pixel_scale
    title_height = 28
    footer_height = 42
    gap = 8
    canvas = Image.new(
        "RGB",
        (
            panel_width * len(panels) +
            gap * (len(panels) - 1),
            title_height + panel_height + footer_height,
        ),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for panel_index, (title, pixels) in enumerate(panels):
        x = panel_index * (panel_width + gap)
        image = Image.fromarray(
            np.rint(pixels * 255.0)
            .clip(0.0, 255.0)
            .astype(np.uint8)
        )
        if pixel_scale != 1:
            image = image.resize(
                (panel_width, panel_height),
                resample=Image.Resampling.NEAREST,
            )
        canvas.paste(image, (x, title_height))
        draw.text(
            (x + 6, 8),
            title,
            fill=(235, 235, 235),
            font=font,
        )
    footer = (
        f"{pass_name}  RMSE={metrics['rmse']:.7g}  "
        f"MAE={metrics['mean_absolute_error']:.7g}  "
        f"max={metrics['maximum_absolute_error']:.7g}  "
        f"mapping={mapping} scale={display_scale:.7g}"
    )
    draw.text(
        (6, title_height + panel_height + 10),
        footer,
        fill=(220, 220, 220),
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return {
        "path": str(output_path),
        "mapping": mapping,
        "display_scale": display_scale,
        "difference_scale": difference_scale,
        "difference_percentile": 99.5,
        "pixel_scale": pixel_scale,
    }


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) < 3:
        raise SystemExit(
            "expected: cycles.exr report.json "
            "[--triptych-dir directory] "
            "Pass=psycles.exr [Pass=psycles.exr ...]"
        )
    cycles_path = pathlib.Path(args[0]).resolve()
    report_path = pathlib.Path(args[1]).resolve()
    triptych_dir: pathlib.Path | None = None
    bindings: list[str] = []
    argument_index = 2
    while argument_index < len(args):
        argument = args[argument_index]
        if argument == "--triptych-dir":
            if argument_index + 1 >= len(args):
                raise ValueError(
                    "--triptych-dir requires a directory"
                )
            triptych_dir = pathlib.Path(
                args[argument_index + 1]
            ).resolve()
            argument_index += 2
            continue
        bindings.append(argument)
        argument_index += 1
    if not bindings:
        raise ValueError("at least one pass binding is required")
    cycles, channel_names = _read_image(cycles_path)
    results: dict[str, Any] = {}
    for binding in bindings:
        if "=" not in binding:
            raise ValueError(f"invalid pass binding: {binding}")
        pass_name, path_text = binding.split("=", 1)
        if pass_name not in _PASS_CHANNELS:
            raise ValueError(f"unsupported pass name: {pass_name}")
        psycles_path = pathlib.Path(path_text).resolve()
        psycles, psycles_channel_names = _read_image(psycles_path)
        indices = _find_cycles_channels(channel_names, pass_name)
        reference = cycles[:, :, indices]
        if psycles_path.suffix.lower() == ".pfm":
            actual = psycles[:, :, : len(indices)]
        else:
            actual_indices = _find_cycles_channels(
                psycles_channel_names, pass_name
            )
            actual = psycles[:, :, actual_indices]
        # PFM stores scanlines bottom-to-top. OpenImageIO exposes that storage
        # order directly for these files, whereas EXR is returned top-to-bottom.
        # Canonicalize both to the renderer's top-left image origin before
        # measuring error; this is a format conversion, not a best-orientation
        # search.
        if psycles_path.suffix.lower() == ".pfm":
            actual = actual[::-1, :, :]
        metrics = _metrics(reference, actual)
        results[pass_name] = {
            "cycles": str(cycles_path),
            "psycles": str(psycles_path),
            **metrics,
        }
        if triptych_dir is not None:
            triptych = _write_triptych(
                reference,
                actual,
                pass_name,
                metrics,
                triptych_dir /
                f"{pass_name.lower().replace(' ', '-')}.png",
            )
            results[pass_name]["triptych"] = triptych

    report = {
        "schema": "psycles.cycles-differential.v1",
        "cycles_channels": channel_names,
        "passes": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["passes"], indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()

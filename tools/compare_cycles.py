"""Compare Psycles linear PFM passes against a Cycles multilayer EXR.

Run with Blender's Python so OpenImageIO and NumPy are available:

    blender --background --python compare_cycles.py -- \
      cycles.exr report.json \
      Combined=psycles-combined.pfm \
      Normal=psycles-normal.pfm \
      DiffCol=psycles-albedo.pfm
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import OpenImageIO as oiio


_PASS_CHANNELS = {
    "Combined": ("R", "G", "B"),
    "Normal": ("X", "Y", "Z"),
    "DiffCol": ("R", "G", "B"),
    "Depth": ("Z",),
    "Debug Sample Count": ("X",),
}


def _read_image(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(f"could not open image: {path}")
    try:
        spec = source.spec()
        pixels = source.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise RuntimeError(f"could not read pixels: {path}")
        array = np.asarray(pixels, dtype=np.float32).reshape(
            spec.height, spec.width, spec.nchannels
        )
        return array, list(spec.channelnames)
    finally:
        source.close()


def _find_cycles_channels(
    channel_names: list[str], pass_name: str
) -> list[int]:
    suffixes = _PASS_CHANNELS[pass_name]
    indices: list[int] = []
    for suffix in suffixes:
        candidates = [
            index
            for index, name in enumerate(channel_names)
            if name.endswith(f".{pass_name}.{suffix}")
            or name == f"{pass_name}.{suffix}"
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
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(reference_values * reference_values)))
    per_pixel = np.sqrt(np.mean(difference * difference, axis=1))
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
        "orientation_rmse": orientation_rmse,
    }


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) < 3:
        raise SystemExit(
            "expected: cycles.exr report.json "
            "Pass=psycles.pfm [Pass=psycles.pfm ...]"
        )
    cycles_path = pathlib.Path(args[0]).resolve()
    report_path = pathlib.Path(args[1]).resolve()
    cycles, channel_names = _read_image(cycles_path)
    results: dict[str, Any] = {}
    for binding in args[2:]:
        if "=" not in binding:
            raise ValueError(f"invalid pass binding: {binding}")
        pass_name, path_text = binding.split("=", 1)
        if pass_name not in _PASS_CHANNELS:
            raise ValueError(f"unsupported pass name: {pass_name}")
        psycles_path = pathlib.Path(path_text).resolve()
        psycles, _ = _read_image(psycles_path)
        indices = _find_cycles_channels(channel_names, pass_name)
        reference = cycles[:, :, indices]
        actual = psycles[:, :, : len(indices)]
        # PFM stores scanlines bottom-to-top. OpenImageIO exposes that storage
        # order directly for these files, whereas EXR is returned top-to-bottom.
        # Canonicalize both to the renderer's top-left image origin before
        # measuring error; this is a format conversion, not a best-orientation
        # search.
        if psycles_path.suffix.lower() == ".pfm":
            actual = actual[::-1, :, :]
        results[pass_name] = {
            "cycles": str(cycles_path),
            "psycles": str(psycles_path),
            **_metrics(reference, actual),
        }

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

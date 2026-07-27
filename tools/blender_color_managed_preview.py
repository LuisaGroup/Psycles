"""Preview linear Cycles/Psycles output with Blender's OCIO view settings.

Run with Blender's bundled Python, which includes OpenImageIO, NumPy, and
PyOpenColorIO:

    blender/4.5/python/bin/python3.11 \
      tools/blender_color_managed_preview.py \
      psycles-combined.pfm preview.png \
      --reference cycles.exr \
      --scene exported-scene/scene.json \
      --pass Combined

With ``--reference``, the output contains the reference, the source, and a
linear absolute-error heatmap from left to right. Numerical acceptance still
uses ``compare_cycles.py``; this tool only makes the displayed comparison use
one explicit Blender color-management pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import OpenImageIO as oiio
import PyOpenColorIO as ocio


_PASS_CHANNELS = {
    "Combined": ("R", "G", "B"),
    "DiffCol": ("R", "G", "B"),
    "Normal": ("X", "Y", "Z"),
}


@dataclass(frozen=True)
class ColorManagement:
    display_device: str = "sRGB"
    view_transform: str = "Standard"
    look: str = "None"
    exposure: float = 0.0
    gamma: float = 1.0
    use_curve_mapping: bool = False


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply Blender's OCIO display pipeline to a linear render pass."
        )
    )
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--reference",
        type=pathlib.Path,
        help=(
            "optional reference image; emit reference/source/error panels"
        ),
    )
    parser.add_argument(
        "--scene",
        type=pathlib.Path,
        help=(
            "exported scene.json or its containing directory; reads "
            "render.color_management"
        ),
    )
    parser.add_argument(
        "--pass",
        dest="pass_name",
        choices=tuple(_PASS_CHANNELS),
        default="Combined",
    )
    parser.add_argument("--display")
    parser.add_argument("--view")
    parser.add_argument("--look")
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument(
        "--ocio-config",
        type=pathlib.Path,
        help=(
            "Blender config.ocio; inferred from OCIO or Blender's bundled "
            "Python when omitted"
        ),
    )
    return parser.parse_args()


def _scene_settings(path: pathlib.Path | None) -> ColorManagement:
    if path is None:
        return ColorManagement()
    scene_path = path / "scene.json" if path.is_dir() else path
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    values = payload.get("render", {}).get("color_management", {})
    return ColorManagement(
        display_device=str(values.get("display_device", "sRGB")),
        view_transform=str(values.get("view_transform", "Standard")),
        look=str(values.get("look", "None")),
        exposure=float(values.get("exposure", 0.0)),
        gamma=float(values.get("gamma", 1.0)),
        use_curve_mapping=bool(values.get("use_curve_mapping", False)),
    )


def _ocio_config_path(explicit: pathlib.Path | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if value := os.environ.get("OCIO"):
        candidates.append(pathlib.Path(value))
    executable = pathlib.Path(sys.executable).resolve()
    for parent in executable.parents:
        candidates.append(
            parent / "datafiles" / "colormanagement" / "config.ocio"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "could not locate Blender config.ocio; pass --ocio-config or set OCIO"
    )


def _read(path: pathlib.Path, pass_name: str) -> np.ndarray:
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
        names = list(specification.channelnames)
    finally:
        source.close()

    suffixes = _PASS_CHANNELS[pass_name]
    indices: list[int] = []
    for suffix in suffixes:
        candidates = [
            index
            for index, name in enumerate(names)
            if name.endswith(f".{pass_name}.{suffix}")
            or name == f"{pass_name}.{suffix}"
        ]
        if len(candidates) == 1:
            indices.append(candidates[0])
        elif len(names) >= 3 and len(indices) == 0:
            # Single-pass PFM/EXR files normally expose generic R/G/B names.
            indices = [0, 1, 2]
            break
        else:
            raise RuntimeError(
                f"could not resolve {pass_name}.{suffix} in {path}: {names}"
            )
    image = np.ascontiguousarray(image[:, :, indices], dtype=np.float32)
    # PFM stores scanlines bottom-to-top; OIIO exposes that storage order.
    if path.suffix.lower() == ".pfm":
        image = np.ascontiguousarray(image[::-1, :, :])
    return image


def _processor(
    config: ocio.Config,
    settings: ColorManagement,
) -> ocio.CPUProcessor:
    scene_linear = config.getRoleColorSpace(ocio.ROLE_SCENE_LINEAR)
    if not scene_linear:
        raise RuntimeError("OCIO config has no scene_linear role")
    group = ocio.GroupTransform()
    if settings.look and settings.look.lower() != "none":
        look = ocio.LookTransform()
        look.setSrc(scene_linear)
        look.setDst(scene_linear)
        look.setLooks(settings.look)
        group.appendTransform(look)
    display = ocio.DisplayViewTransform()
    display.setSrc(scene_linear)
    display.setDisplay(settings.display_device)
    display.setView(settings.view_transform)
    group.appendTransform(display)
    return config.getProcessor(group).getDefaultCPUProcessor()


def _display(
    linear: np.ndarray,
    processor: ocio.CPUProcessor,
    settings: ColorManagement,
    pass_name: str,
) -> np.ndarray:
    if pass_name == "Normal":
        return np.clip(linear * 0.5 + 0.5, 0.0, 1.0)
    if pass_name != "Combined":
        positive = np.maximum(linear, 0.0)
        encoded = np.where(
            positive <= 0.0031308,
            positive * 12.92,
            1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
        )
        return np.clip(encoded, 0.0, 1.0)
    if settings.use_curve_mapping:
        raise RuntimeError(
            "Blender view curve mapping is enabled; its editable curve is "
            "not serialized in the current scene bundle"
        )
    if not math.isfinite(settings.gamma) or settings.gamma <= 0.0:
        raise ValueError(f"invalid Blender view gamma: {settings.gamma}")
    display = np.ascontiguousarray(
        linear * np.float32(2.0**settings.exposure),
        dtype=np.float32,
    )
    descriptor = ocio.PackedImageDesc(
        display,
        display.shape[1],
        display.shape[0],
        display.shape[2],
    )
    processor.apply(descriptor)
    if settings.gamma != 1.0:
        display = np.sign(display) * np.power(
            np.abs(display), 1.0 / settings.gamma
        )
    return np.clip(display, 0.0, 1.0)


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


def _write(path: pathlib.Path, image: np.ndarray, error_scale: float | None) -> None:
    encoded = np.ascontiguousarray(
        np.clip(image * 255.0 + 0.5, 0.0, 255.0),
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = oiio.ImageOutput.create(str(path))
    if destination is None:
        raise RuntimeError(oiio.geterror())
    specification = oiio.ImageSpec(
        encoded.shape[1], encoded.shape[0], 3, oiio.UINT8
    )
    if error_scale is not None:
        specification.attribute("psycles:error_p99_scale", error_scale)
    if not destination.open(str(path), specification):
        raise RuntimeError(destination.geterror())
    try:
        if not destination.write_image(encoded):
            raise RuntimeError(destination.geterror())
    finally:
        destination.close()


def _main() -> None:
    args = _arguments()
    settings = _scene_settings(args.scene)
    settings = ColorManagement(
        display_device=args.display or settings.display_device,
        view_transform=args.view or settings.view_transform,
        look=args.look if args.look is not None else settings.look,
        exposure=(
            args.exposure if args.exposure is not None else settings.exposure
        ),
        gamma=args.gamma if args.gamma is not None else settings.gamma,
        use_curve_mapping=settings.use_curve_mapping,
    )
    config_path = _ocio_config_path(args.ocio_config)
    config = ocio.Config.CreateFromFile(str(config_path))
    processor = _processor(config, settings)

    actual = _read(args.source, args.pass_name)
    actual_display = _display(
        actual, processor, settings, args.pass_name
    )
    error_scale: float | None = None
    output = actual_display
    if args.reference is not None:
        reference = _read(args.reference, args.pass_name)
        if reference.shape != actual.shape:
            raise RuntimeError(
                f"shape mismatch: {reference.shape} != {actual.shape}"
            )
        reference_display = _display(
            reference, processor, settings, args.pass_name
        )
        error = np.sqrt(np.mean((actual - reference) ** 2, axis=2))
        heatmap, error_scale = _heatmap(error)
        output = np.concatenate(
            [reference_display, actual_display, heatmap], axis=1
        )

    _write(args.output, output, error_scale)
    print(
        "Blender color preview: "
        f"display={settings.display_device!r}, "
        f"view={settings.view_transform!r}, look={settings.look!r}, "
        f"exposure={settings.exposure:g}, gamma={settings.gamma:g}"
    )
    if error_scale is not None:
        print(f"{args.pass_name} linear-error P99 scale: {error_scale:.9g}")


if __name__ == "__main__":
    _main()

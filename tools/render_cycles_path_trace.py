"""Render one authoritative Cycles path as a 32-bit multilayer EXR trace.

This script requires the diagnostic Cycles build described in
``docs/cycles-path-trace.md``. Coordinates use Cycles film convention:
``(0, 0)`` is the lower-left pixel of the uncropped image.

Usage:

    blender scene.blend --background --python render_cycles_path_trace.py -- \
        trace.exr --pixel-x 320 --pixel-y 240 --cycles-device CPU
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import bpy

_TOOLS = pathlib.Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from cycles_path_trace_schema import (  # noqa: E402
    AOV_COUNT,
    SCHEMA_VERSION,
    SLOTS,
    aov_name,
    schema_document,
)


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _arguments(scene: Any) -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Render one instrumented Cycles path to multilayer EXR"
    )
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--width",
        type=_positive_integer,
        default=scene.render.resolution_x,
    )
    parser.add_argument(
        "--height",
        type=_positive_integer,
        default=scene.render.resolution_y,
    )
    parser.add_argument("--pixel-x", type=_nonnegative_integer)
    parser.add_argument("--pixel-y", type=_nonnegative_integer)
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        default=scene.cycles.seed,
    )
    parser.add_argument(
        "--cycles-device",
        default="CPU",
        help="Cycles compute backend, currently CPU or HIP",
    )
    parser.add_argument(
        "--device-name",
        default="",
        help="case-insensitive substring required in the selected device",
    )
    parser.add_argument(
        "--sampling-pattern",
        default="TABULATED_SOBOL",
        help="Cycles sampling pattern used by the oracle",
    )
    parser.add_argument(
        "--scrambling-distance",
        type=float,
        default=1.0,
    )
    return parser.parse_args(argv)


def _device_record(device: Any) -> dict[str, Any]:
    return {
        "name": device.name,
        "type": device.type,
        "id": device.id,
    }


def _configure_cycles_device(
    scene: Any,
    requested_type: str,
    requested_name: str,
) -> list[dict[str, Any]]:
    compute_type = requested_type.strip().upper()
    name_filter = requested_name.strip().casefold()
    if compute_type not in {"CPU", "HIP"}:
        raise ValueError(
            "path-trace oracle currently supports --cycles-device CPU or HIP"
        )

    preferences = bpy.context.preferences.addons["cycles"].preferences
    if compute_type != "CPU":
        try:
            preferences.compute_device_type = compute_type
        except (TypeError, ValueError) as exception:
            raise RuntimeError(
                f"Cycles compute backend {compute_type!r} is unavailable"
            ) from exception
    preferences.get_devices()
    selected = [
        device
        for device in preferences.devices
        if device.type == compute_type
        and (
            not name_filter
            or name_filter in device.name.casefold()
        )
    ]
    if not selected:
        available = ", ".join(
            f"{device.type}:{device.name}"
            for device in preferences.devices
        )
        raise RuntimeError(
            f"Cycles exposed no {compute_type} device matching "
            f"{requested_name!r}; available devices: {available}"
        )
    selected_ids = {device.id for device in selected}
    for device in preferences.devices:
        device.use = device.id in selected_ids
    scene.cycles.device = "CPU" if compute_type == "CPU" else "GPU"
    return [_device_record(device) for device in selected]


def _configure_trace_aovs(view_layer: Any) -> None:
    while len(view_layer.aovs):
        view_layer.aovs.remove(view_layer.aovs[-1])
    for slot in SLOTS:
        if slot.aov != aov_name(slot.index):
            raise AssertionError("trace schema AOV names are inconsistent")
        aov = view_layer.aovs.add()
        aov.name = slot.aov
        aov.type = "COLOR"
    if len(view_layer.aovs) != AOV_COUNT:
        raise AssertionError(
            f"created {len(view_layer.aovs)} trace AOVs, expected {AOV_COUNT}"
        )


def _set_if_present(owner: Any, name: str, value: Any) -> None:
    if hasattr(owner, name):
        setattr(owner, name, value)


def _main() -> None:
    scene = bpy.context.scene
    arguments = _arguments(scene)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    width = arguments.width
    height = arguments.height
    pixel_x = arguments.pixel_x
    pixel_y = arguments.pixel_y
    if pixel_x is None:
        pixel_x = width // 2
    if pixel_y is None:
        pixel_y = height // 2
    if pixel_x >= width or pixel_y >= height:
        raise ValueError(
            f"trace pixel ({pixel_x}, {pixel_y}) is outside {width}x{height}"
        )

    selected_layer = bpy.context.view_layer
    for view_layer in scene.view_layers:
        view_layer.use = view_layer == selected_layer
    selected_layer.use_pass_combined = True
    _configure_trace_aovs(selected_layer)

    # Render passes are registered when the engine is initialized. Configure
    # the AOV collection first, then force a fresh Cycles engine instance so
    # every trace slot is present in the Render Result and multilayer EXR.
    render_engines = {
        item.identifier
        for item in scene.render.bl_rna.properties["engine"].enum_items
    }
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT"
        if "BLENDER_EEVEE_NEXT" in render_engines
        else "BLENDER_EEVEE"
    )
    scene.render.engine = "CYCLES"
    enabled_devices = _configure_cycles_device(
        scene,
        arguments.cycles_device,
        arguments.device_name,
    )

    scene.cycles.samples = 1
    scene.cycles.seed = arguments.seed
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    _set_if_present(scene.cycles, "sampling_pattern", arguments.sampling_pattern)
    _set_if_present(
        scene.cycles,
        "scrambling_distance",
        arguments.scrambling_distance,
    )
    _set_if_present(scene.cycles, "use_auto_scrambling_distance", False)

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.use_border = True
    scene.render.use_crop_to_border = True
    scene.render.border_min_x = pixel_x / width
    scene.render.border_max_x = (pixel_x + 1) / width
    scene.render.border_min_y = pixel_y / height
    scene.render.border_max_y = (pixel_y + 1) / height
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.filepath = str(output)

    image_settings = scene.render.image_settings
    if hasattr(image_settings, "media_type"):
        image_settings.media_type = "MULTI_LAYER_IMAGE"
    image_settings.file_format = "OPEN_EXR_MULTILAYER"
    image_settings.color_mode = "RGBA"
    image_settings.color_depth = "32"
    image_settings.exr_codec = "ZIP"

    begin = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - begin

    metadata = {
        "schema": "psycles.cycles-path-trace-render.v1",
        "trace_schema": schema_document(),
        "trace_schema_version": SCHEMA_VERSION,
        "source": bpy.data.filepath,
        "output": str(output),
        "blender": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode("ascii"),
        "cycles_compute_device_type": (
            arguments.cycles_device.strip().upper()
        ),
        "cycles_enabled_devices": enabled_devices,
        "frame": scene.frame_current,
        "camera": scene.camera.name if scene.camera else None,
        "width": width,
        "height": height,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "samples": scene.cycles.samples,
        "seed": scene.cycles.seed,
        "sampling_pattern": getattr(
            scene.cycles,
            "sampling_pattern",
            None,
        ),
        "scrambling_distance": getattr(
            scene.cycles,
            "scrambling_distance",
            None,
        ),
        "elapsed_seconds": elapsed,
    }
    output.with_suffix(".render.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Cycles path trace rendered pixel ({pixel_x}, {pixel_y}) of "
        f"{width}x{height} on "
        f"{arguments.cycles_device.strip().upper()} in {elapsed:.3f}s: "
        f"{output}"
    )


if __name__ == "__main__":
    _main()

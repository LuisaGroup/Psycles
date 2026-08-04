"""Render authoritative linear passes with Blender Cycles.

Usage:

    blender scene.blend --background --python render_cycles_golden.py -- \
        output.exr [width height samples seed] \
        [--cycles-device CPU|HIP] [--device-name substring] \
        [--sampling-pattern TABULATED_SOBOL] \
        [--scrambling-distance 1.0]

The output is a multilayer, 32-bit float EXR before display transforms and
without compositor modifications. It is the rendering oracle for Psycles.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import bpy


_GOLDEN_PASSES = (
    "Combined",
    "Normal",
    "DiffCol",
    "DiffDir",
    "DiffInd",
    "GlossCol",
    "GlossDir",
    "GlossInd",
    "TransCol",
    "TransDir",
    "TransInd",
    "Emit",
    "Env",
    "Volume Direct",
    "Volume Indirect",
    "Depth",
    "Debug Sample Count",
)


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _arguments(scene: Any) -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Render authoritative Cycles linear passes"
    )
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "width", nargs="?", type=_positive_integer, default=320
    )
    parser.add_argument(
        "height", nargs="?", type=_positive_integer, default=240
    )
    parser.add_argument(
        "samples", nargs="?", type=_positive_integer, default=128
    )
    parser.add_argument(
        "seed",
        nargs="?",
        type=_nonnegative_integer,
        default=scene.cycles.seed,
    )
    parser.add_argument(
        "--cycles-device",
        default="CPU",
        help="Cycles compute backend, for example CPU or HIP",
    )
    parser.add_argument(
        "--device-name",
        default="",
        help="case-insensitive substring required in the selected device",
    )
    parser.add_argument(
        "--sampling-pattern",
        default="TABULATED_SOBOL",
        help="Cycles sampling pattern used by the reference render",
    )
    parser.add_argument(
        "--scrambling-distance",
        type=float,
        default=1.0,
        help="Cycles sampler scrambling distance",
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
    if not compute_type:
        raise ValueError("the Cycles device type is empty")

    preferences = bpy.context.preferences.addons[
        "cycles"
    ].preferences
    if compute_type == "CPU":
        preferences.get_devices()
        selected = [
            device
            for device in preferences.devices
            if device.type == "CPU"
            and (
                not name_filter
                or name_filter in device.name.casefold()
            )
        ]
        selected_ids = {device.id for device in selected}
        for device in preferences.devices:
            device.use = device.id in selected_ids
        if not selected:
            raise RuntimeError(
                "Cycles did not expose a CPU device matching "
                f"{requested_name!r}"
            )
        scene.cycles.device = "CPU"
        return [_device_record(device) for device in selected]

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
    selected_ids = {device.id for device in selected}
    for device in preferences.devices:
        device.use = device.id in selected_ids
    if not selected:
        available = ", ".join(
            f"{device.type}:{device.name}"
            for device in preferences.devices
        )
        raise RuntimeError(
            f"Cycles exposed no {compute_type} device matching "
            f"{requested_name!r}; available devices: {available}"
        )
    scene.cycles.device = "GPU"
    return [_device_record(device) for device in selected]


def _configure_view_layer_passes(view_layer: Any) -> None:
    """Enable every linear pass used by the canonical differential report."""
    view_layer.use_pass_combined = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_diffuse_indirect = True
    view_layer.use_pass_glossy_color = True
    view_layer.use_pass_glossy_direct = True
    view_layer.use_pass_glossy_indirect = True
    view_layer.use_pass_transmission_color = True
    view_layer.use_pass_transmission_direct = True
    view_layer.use_pass_transmission_indirect = True
    view_layer.use_pass_emit = True
    view_layer.use_pass_environment = True
    view_layer.use_pass_z = True
    view_layer.cycles.use_pass_volume_direct = True
    view_layer.cycles.use_pass_volume_indirect = True
    if hasattr(
        view_layer.cycles,
        "use_pass_debug_sample_count",
    ):
        view_layer.cycles.use_pass_debug_sample_count = True
    elif hasattr(
        view_layer.cycles,
        "pass_debug_sample_count",
    ):
        view_layer.cycles.pass_debug_sample_count = True


def _set_if_present(owner: Any, name: str, value: Any) -> None:
    if hasattr(owner, name):
        setattr(owner, name, value)


def _configure_sampler(
    scene: Any,
    sampling_pattern: str,
    scrambling_distance: float,
) -> None:
    _set_if_present(
        scene.cycles,
        "sampling_pattern",
        sampling_pattern,
    )
    _set_if_present(
        scene.cycles,
        "scrambling_distance",
        scrambling_distance,
    )
    _set_if_present(
        scene.cycles,
        "use_auto_scrambling_distance",
        False,
    )


def _main() -> None:
    scene = bpy.context.scene
    arguments = _arguments(scene)
    output = arguments.output.resolve()
    width = arguments.width
    height = arguments.height
    samples = arguments.samples
    seed = arguments.seed
    output.parent.mkdir(parents=True, exist_ok=True)

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
    scene.cycles.samples = samples
    scene.cycles.seed = seed
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    _configure_sampler(
        scene,
        arguments.sampling_pattern,
        arguments.scrambling_distance,
    )
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
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

    for view_layer in scene.view_layers:
        view_layer.use = True
        _configure_view_layer_passes(view_layer)

    begin = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - begin

    metadata = {
        "schema": "psycles.cycles-golden.v1",
        "source": bpy.data.filepath,
        "output": str(output),
        "blender": bpy.app.version_string,
        "cycles_device": scene.cycles.device,
        "cycles_compute_device_type": (
            arguments.cycles_device.strip().upper()
        ),
        "cycles_enabled_devices": enabled_devices,
        "frame": scene.frame_current,
        "camera": scene.camera.name if scene.camera else None,
        "width": width,
        "height": height,
        "samples": samples,
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
        "adaptive_sampling": scene.cycles.use_adaptive_sampling,
        "denoising": scene.cycles.use_denoising,
        "transparent": scene.render.film_transparent,
        "elapsed_seconds": elapsed,
        "passes": list(_GOLDEN_PASSES),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Cycles golden rendered {width}x{height} at {samples} spp "
        f"on {arguments.cycles_device.strip().upper()} "
        f"in {elapsed:.3f}s: {output}"
    )


if __name__ == "__main__":
    _main()

"""Render authoritative linear passes with Blender Cycles.

Usage:

    blender scene.blend --background --python render_cycles_golden.py -- \
        output.exr [width height samples]

The output is a multilayer, 32-bit float EXR before display transforms and
without compositor modifications. It is the rendering oracle for Psycles.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import bpy


def _integer(args: list[str], index: int, fallback: int) -> int:
    if index >= len(args):
        return fallback
    value = int(args[index])
    if value <= 0:
        raise ValueError(f"argument {index + 1} must be positive")
    return value


def _nonnegative_integer(args: list[str], index: int, fallback: int) -> int:
    if index >= len(args):
        return fallback
    value = int(args[index])
    if value < 0:
        raise ValueError(f"argument {index + 1} must be non-negative")
    return value


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not args:
        raise SystemExit(
            "expected output path: blender scene.blend --background "
            "--python render_cycles_golden.py -- output.exr "
            "[width height samples seed]"
        )
    output = pathlib.Path(args[0]).resolve()
    width = _integer(args, 1, 320)
    height = _integer(args, 2, 240)
    samples = _integer(args, 3, 128)
    seed = _nonnegative_integer(args, 4, bpy.context.scene.cycles.seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
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
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.seed = seed
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
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

    begin = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - begin

    metadata = {
        "schema": "psycles.cycles-golden.v1",
        "source": bpy.data.filepath,
        "output": str(output),
        "blender": bpy.app.version_string,
        "cycles_device": scene.cycles.device,
        "frame": scene.frame_current,
        "camera": scene.camera.name if scene.camera else None,
        "width": width,
        "height": height,
        "samples": samples,
        "seed": scene.cycles.seed,
        "adaptive_sampling": scene.cycles.use_adaptive_sampling,
        "denoising": scene.cycles.use_denoising,
        "transparent": scene.render.film_transparent,
        "elapsed_seconds": elapsed,
        "passes": [
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
            "Depth",
            "Debug Sample Count",
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Cycles golden rendered {width}x{height} at {samples} spp "
        f"in {elapsed:.3f}s: {output}"
    )


if __name__ == "__main__":
    _main()

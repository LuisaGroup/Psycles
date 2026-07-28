"""Lock the Blender 4.5/5.2 multilayer OpenEXR configuration contract."""

from __future__ import annotations

import bpy


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

settings = bpy.context.scene.render.image_settings
if hasattr(settings, "media_type"):
    settings.media_type = "MULTI_LAYER_IMAGE"
settings.file_format = "OPEN_EXR_MULTILAYER"
settings.color_mode = "RGBA"
settings.color_depth = "32"

assert settings.file_format == "OPEN_EXR_MULTILAYER"
if hasattr(settings, "media_type"):
    assert settings.media_type == "MULTI_LAYER_IMAGE"

cycles_layer = scene.view_layers[0].cycles
if hasattr(cycles_layer, "use_pass_debug_sample_count"):
    cycles_layer.use_pass_debug_sample_count = True
else:
    cycles_layer.pass_debug_sample_count = True

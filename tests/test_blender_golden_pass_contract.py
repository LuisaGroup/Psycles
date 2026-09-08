"""Canonical golden passes must replace, not extend, authored output passes."""

from __future__ import annotations

import pathlib
import runpy
import sys

import bpy


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    golden = runpy.run_path(str(pathlib.Path(args[0]).resolve()))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    layer = scene.view_layers[0]
    # These are real extra workloads inherited by the original four-scene
    # benchmark (AO even launches additional shadow paths), not just EXR labels.
    for name in ("use_pass_ambient_occlusion", "use_pass_mist", "use_pass_z",
                 "use_pass_object_index", "use_pass_vector",
                 "use_pass_cryptomatte_object"):
        setattr(layer, name, True)
    layer.cycles.denoising_store_passes = True
    for name in ("pass_debug_sample_count", "use_pass_debug_sample_count", "pass_render_time"):
        if hasattr(layer.cycles, name):
            setattr(layer.cycles, name, True)
    layer.aovs.add().name = "Inherited AOV"
    layer.lightgroups.add().name = "Inherited light group"
    layer.use_pass_cryptomatte_accurate = False
    disabled = scene.view_layers.new("Authored disabled layer")
    disabled.use = False
    disabled.use_pass_ambient_occlusion = True
    names = golden["_configure_enabled_view_layer_passes"](scene)
    expected = {
        "use_pass_combined", "use_pass_normal", "use_pass_diffuse_color",
        "use_pass_diffuse_direct", "use_pass_diffuse_indirect",
        "use_pass_glossy_color", "use_pass_glossy_direct", "use_pass_glossy_indirect",
        "use_pass_transmission_color", "use_pass_transmission_direct",
        "use_pass_transmission_indirect", "use_pass_emit", "use_pass_environment",
        "use_pass_volume_direct", "use_pass_volume_indirect"}
    enabled = {
        prop.identifier
        for owner in (layer, layer.cycles)
        for prop in owner.bl_rna.properties
        if prop.type == "BOOLEAN" and "pass" in prop.identifier
        and getattr(owner, prop.identifier)
        and prop.identifier not in {"use_pass_cryptomatte_accurate",
                                    "denoising_pass_follow_reflections",
                                    "denoising_pass_use_albedo_roughness_weighting"}}
    assert enabled == expected, ("unexpected pass workload", enabled - expected,
                                 "missing", expected - enabled)
    assert not layer.aovs and not layer.lightgroups, "extra custom output passes"
    assert not layer.use_pass_cryptomatte_accurate, "changed a quality setting, not a pass toggle"
    assert not disabled.use and disabled.use_pass_ambient_occlusion
    assert names == [layer.name]
    assert len(golden["_GOLDEN_PASSES"]) == 15
    print("Golden pass contract: exactly 15 common passes; inherited extras cleared")


if __name__ == "__main__":
    main()

"""Blender-side regressions for effective Cycles render settings."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy
from mathutils import Vector


def _export(
    exporter: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, object]:
    old_argv = sys.argv
    try:
        sys.argv = [str(exporter), "--", str(output)]
        runpy.run_path(str(exporter), run_name="__main__")
    finally:
        sys.argv = old_argv
    return json.loads(
        (output / "scene.json").read_text(encoding="utf-8")
    )


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit(
            "expected exporter and shader-probe paths after '--'"
        )
    exporter, probe_creator = map(
        lambda value: pathlib.Path(value).resolve(),
        args,
    )
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-render-settings-"
    ) as temporary:
        root = pathlib.Path(temporary)
        scene.cycles.pixel_filter_type = "BOX"
        scene.cycles.filter_width = 0.01
        box_payload = _export(exporter, root / "box")
        box = box_payload["render"]
        build = box_payload.get("blender_build")
        if (
            not isinstance(build, dict)
            or build.get("version") != bpy.app.version_string
            or build.get("build_hash")
            != bpy.app.build_hash.decode("utf-8")
            or build.get("build_branch")
            != bpy.app.build_branch.decode("utf-8")
            or build.get("build_type")
            != bpy.app.build_type.decode("utf-8")
            or build.get("version_tuple") != list(bpy.app.version)
        ):
            raise AssertionError(
                "scene export did not preserve Blender build identity: "
                f"{build}"
            )
        if (
            box["pixel_filter_type"] != "BOX"
            or float(box["filter_width"]) != 1.0
        ):
            raise AssertionError(
                "BOX export did not use Cycles' effective one-pixel width: "
                f"{box}"
            )

        scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
        scene.cycles.filter_width = 1.5
        blackman_harris = _export(
            exporter, root / "blackman-harris"
        )["render"]
        if (
            blackman_harris["pixel_filter_type"]
            != "BLACKMAN_HARRIS"
            or float(blackman_harris["filter_width"]) != 1.5
        ):
            raise AssertionError(
                "Blackman-Harris export changed its configured width: "
                f"{blackman_harris}"
            )

        camera = scene.camera
        if camera is None:
            raise AssertionError("render-settings fixture has no camera")
        dof = camera.data.dof
        original_use_dof = dof.use_dof
        original_focus_object = dof.focus_object
        original_focus_subtarget = dof.focus_subtarget
        original_focus_distance = dof.focus_distance
        original_camera_scale = camera.scale.copy()
        focus = bpy.data.objects.new("Psycles DOF Focus", None)
        scene.collection.objects.link(focus)
        try:
            camera.scale = (0.5, 0.75, 1.25)
            camera_position = camera.matrix_world.translation
            camera_z = camera.matrix_world.col[2].xyz.normalized()
            camera_x = camera.matrix_world.col[0].xyz.normalized()
            focus.location = Vector(camera_position) - 7.0 * camera_z + (
                3.0 * camera_x
            )
            dof.use_dof = True
            dof.focus_object = focus
            dof.focus_subtarget = ""
            dof.focus_distance = 0.125
            focus_object_camera = _export(
                exporter, root / "focus-object-camera"
            )["camera"]
            exported_distance = float(
                focus_object_camera["dof"]["focus_distance"]
            )
            if abs(exported_distance - 7.0) > 1.0e-5:
                raise AssertionError(
                    "focus object did not override the authored DOF "
                    "distance with Cycles' camera-Z projection: "
                    f"{focus_object_camera}"
                )
            exported_transform = focus_object_camera["transform"]
            basis_lengths = tuple(
                sum(
                    float(exported_transform[column * 4 + row]) ** 2
                    for row in range(3)
                )
                ** 0.5
                for column in range(3)
            )
            if any(abs(length - 1.0) > 1.0e-6 for length in basis_lengths):
                raise AssertionError(
                    "camera object scale leaked into the Cycles camera "
                    f"transform: lengths={basis_lengths}"
                )
        finally:
            dof.use_dof = original_use_dof
            dof.focus_object = original_focus_object
            dof.focus_subtarget = original_focus_subtarget
            dof.focus_distance = original_focus_distance
            camera.scale = original_camera_scale
            bpy.data.objects.remove(focus, do_unlink=True)

        # Cycles does not pass the authored seed straight to the sampler when
        # Animated Seed is enabled. These constants are pinned from official
        # Cycles hash_uint2 scene sync, including its float32 subframe step.
        original_frame = scene.frame_current
        original_subframe = scene.frame_subframe
        original_seed = scene.cycles.seed
        original_animated_seed = scene.cycles.use_animated_seed
        try:
            scene.frame_set(7, subframe=0.25)
            scene.cycles.seed = 12345
            scene.cycles.use_animated_seed = True
            animated = _export(exporter, root / "animated-seed")[
                "render"
            ]["cycles"]
            if (
                animated["seed"] != 12345
                or animated["effective_seed"] != 4188122935
                or not animated["use_animated_seed"]
                or animated["seed_frame"] != 7
                or float(animated["seed_subframe"]) != 0.25
            ):
                raise AssertionError(
                    "Animated Seed did not match Cycles scene sync: "
                    f"{animated}"
                )

            # This is the exact configuration that exposed the production
            # Barbershop mismatch: frame one, authored seed zero.
            scene.frame_set(1, subframe=0.0)
            scene.cycles.seed = 0
            barbershop = _export(
                exporter, root / "barbershop-animated-seed"
            )["render"]["cycles"]
            if barbershop["effective_seed"] != 1267069554:
                raise AssertionError(
                    "Barbershop Animated Seed regression changed: "
                    f"{barbershop}"
                )

            scene.cycles.use_animated_seed = False
            scene.cycles.seed = 12345
            static = _export(exporter, root / "static-seed")["render"][
                "cycles"
            ]
            if static["effective_seed"] != 12345:
                raise AssertionError(
                    "disabled Animated Seed changed the authored seed: "
                    f"{static}"
                )
        finally:
            scene.frame_set(original_frame, subframe=original_subframe)
            scene.cycles.seed = original_seed
            scene.cycles.use_animated_seed = original_animated_seed

    probes = runpy.run_path(
        str(probe_creator),
        run_name="psycles_test_cycles_shader_probes",
    )
    runner = runpy.run_path(
        str(probe_creator.with_name("run_cycles_shader_probes.py")),
        run_name="psycles_test_cycles_shader_probe_runner",
    )
    registered_probes = tuple(sorted(probes["_CANONICAL_PROBES"]))
    expected_probes = tuple(sorted(runner["_ALL_PROBES"]))
    if registered_probes != expected_probes:
        raise AssertionError(
            "shader-probe registry differs from the canonical runner: "
            f"registered={registered_probes}, expected={expected_probes}"
        )
    oracle_probes = set(probes["_CYCLES_SVM_ORACLE_PROBES"])
    if oracle_probes & set(registered_probes):
        raise AssertionError(
            "Cycles SVM oracle probes overlap the canonical runner"
        )
    if set(probes["_PROBES"]) != set(registered_probes) | oracle_probes:
        raise AssertionError(
            "combined shader-probe registry is not an exact partition"
        )
    if oracle_probes != {
        "metallic_svm_oracle",
        "principled_burley_svm_oracle",
        "principled_coat_svm_oracle",
        "principled_metallic_svm_oracle",
        "principled_random_walk_skin_svm_oracle",
        "principled_sheen_svm_oracle",
        "principled_svm_oracle",
        "principled_thin_wall_svm_oracle",
        "principled_transmission_svm_oracle",
        "svm_tangent_dynamic",
        "standalone_sheen_ashikhmin_svm_oracle",
        "standalone_sheen_microfiber_svm_oracle",
        "standalone_toon_diffuse_svm_oracle",
        "standalone_toon_glossy_svm_oracle",
        "subsurface_burley_svm_oracle",
        "subsurface_random_walk_svm_oracle",
        "vector_to_scalar",
        "volume_scatter_svm",
    }:
        raise AssertionError(
            "unexpected Cycles SVM oracle-only probe set: "
            f"{sorted(oracle_probes)}"
        )
    golden = runpy.run_path(
        str(exporter.with_name("render_cycles_golden.py")),
        run_name="psycles_test_cycles_golden",
    )
    scene.cycles.sampling_pattern = "AUTOMATIC"
    scene.cycles.scrambling_distance = 0.25
    golden["_configure_sampler"](
        scene,
        "TABULATED_SOBOL",
        1.0,
    )
    if (
        scene.cycles.sampling_pattern != "TABULATED_SOBOL"
        or float(scene.cycles.scrambling_distance) != 1.0
        or (
            hasattr(
                scene.cycles,
                "use_auto_scrambling_distance",
            )
            and scene.cycles.use_auto_scrambling_distance
        )
    ):
        raise AssertionError(
            "Cycles golden did not pin the path-sampler contract"
        )

    view_layer = scene.view_layers[0]
    view_layer.cycles.use_pass_volume_direct = False
    view_layer.cycles.use_pass_volume_indirect = False
    golden["_configure_view_layer_passes"](view_layer)
    if (
        not view_layer.cycles.use_pass_volume_direct
        or not view_layer.cycles.use_pass_volume_indirect
        or "Volume Direct" not in golden["_GOLDEN_PASSES"]
        or "Volume Indirect" not in golden["_GOLDEN_PASSES"]
    ):
        raise AssertionError(
            "Cycles golden did not enable the canonical volume passes"
        )

    disabled_layer = scene.view_layers.new("Psycles Disabled Golden Layer")
    disabled_layer.use = False
    disabled_layer.use_pass_normal = False
    try:
        enabled_names = golden[
            "_configure_enabled_view_layer_passes"
        ](scene)
        if (
            disabled_layer.use
            or disabled_layer.use_pass_normal
            or disabled_layer.name in enabled_names
        ):
            raise AssertionError(
                "Cycles golden changed an authored-disabled view layer"
            )
    finally:
        scene.view_layers.remove(disabled_layer)

    scene.cycles.sampling_pattern = "AUTOMATIC"
    probes["_PROBES"]["camera_dof_disk"](scene)
    if (
        scene.cycles.sampling_pattern != "TABULATED_SOBOL"
        or scene.cycles.pixel_filter_type != "BOX"
        or float(scene.cycles.filter_width) != 1.0
    ):
        raise AssertionError(
            "DOF sample-correspondence probe did not pin the Cycles "
            "sampler/filter contract"
        )

    scene.cycles.sampling_pattern = "AUTOMATIC"
    probes["_PROBES"]["camera_blackman_harris_filter"](scene)
    if (
        scene.cycles.sampling_pattern != "TABULATED_SOBOL"
        or scene.cycles.pixel_filter_type != "BLACKMAN_HARRIS"
        or float(scene.cycles.filter_width) != 1.5
    ):
        raise AssertionError(
            "film-filter sample-correspondence probe did not pin the "
            "Cycles sampler/filter contract"
        )

    toon_oracles = (
        (
            "standalone_toon_diffuse_svm_oracle",
            "Standalone Diffuse Toon SVM Oracle",
            "DIFFUSE",
            (0.31, 0.73, 0.19, 1.0),
            0.37,
            0.21,
        ),
        (
            "standalone_toon_glossy_svm_oracle",
            "Standalone Glossy Toon SVM Oracle",
            "GLOSSY",
            (0.84, 0.22, 0.56, 1.0),
            0.63,
            0.14,
        ),
    )
    for probe, _name, _component, _color, _size, _smooth in toon_oracles:
        probes["_PROBES"][probe](scene)
    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-toon-oracle-"
    ) as temporary:
        payload = _export(exporter, pathlib.Path(temporary))
    materials = {
        material["name"]: material
        for material in payload["materials"]
    }
    for _probe, name, component, color, size, smooth in toon_oracles:
        material = materials.get(name)
        if material is None:
            raise AssertionError(f"Toon oracle material is absent: {name}")
        nodes = [
            node
            for node in material["node_tree"]["nodes"]
            if node["type"] == "BSDF_TOON"
        ]
        if len(nodes) != 1:
            raise AssertionError(
                f"Toon oracle did not export one raw BSDF_TOON: {name}"
            )
        node = nodes[0]
        inputs = {entry["identifier"]: entry for entry in node["inputs"]}
        if (
            node["properties"].get("component") != component
            or any(
                abs(float(actual) - expected) > 1.0e-6
                for actual, expected in zip(
                    inputs["Color"]["default"], color, strict=True
                )
            )
            or abs(float(inputs["Size"]["default"]) - size) > 1.0e-6
            or abs(float(inputs["Smooth"]["default"]) - smooth) > 1.0e-6
            or inputs["Normal"]["linked"]
            or tuple(float(value) for value in inputs["Normal"]["default"])
            != (0.0, 0.0, 0.0)
            or node["outputs"][0]["identifier"] != "BSDF"
        ):
            raise AssertionError(
                f"Toon oracle was altered or pre-baked during export: {node}"
            )

    print("Psycles Blender render-settings regression passed")


if __name__ == "__main__":
    _main()

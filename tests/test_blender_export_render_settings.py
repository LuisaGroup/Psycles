"""Blender-side regressions for effective Cycles render settings."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


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
        box = _export(exporter, root / "box")["render"]
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

    probes = runpy.run_path(
        str(probe_creator),
        run_name="psycles_test_cycles_shader_probes",
    )
    runner = runpy.run_path(
        str(probe_creator.with_name("run_cycles_shader_probes.py")),
        run_name="psycles_test_cycles_shader_probe_runner",
    )
    registered_probes = tuple(sorted(probes["_PROBES"]))
    expected_probes = tuple(sorted(runner["_ALL_PROBES"]))
    if registered_probes != expected_probes:
        raise AssertionError(
            "shader-probe registry differs from the canonical runner: "
            f"registered={registered_probes}, expected={expected_probes}"
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

    print("Psycles Blender render-settings regression passed")


if __name__ == "__main__":
    _main()

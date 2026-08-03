#!/usr/bin/env python3
"""Run canonical shader probes through Cycles and Psycles-Luisa.

The runner creates each .blend, renders the authoritative Cycles multilayer
EXR, exports the unchanged node graph, renders it with Psycles, and writes the
linear pass differential report.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


_ALL_PROBES = (
    "add_shader_emission",
    "area_light",
    "area_light_ellipse",
    "area_light_spread",
    "background_world",
    "blackbody_matrix",
    "bump_matrix",
    "bump_nested_matrix",
    "bump_surface",
    "brightness_contrast",
    "brick_texture",
    "brick_texture_constants",
    "camera_blackman_harris_filter",
    "camera_dof_disk",
    "checker_texture_matrix",
    "clamp",
    "color_ramp_alpha_modes",
    "color_ramp_modes",
    "color_ramp_rgb",
    "combine_color_modes",
    "diffuse_bsdf_matrix",
    "diffuse_surface",
    "emission_surface",
    "fresnel_matrix",
    "flat_light_distribution",
    "gamma_color",
    "glass_transport",
    "gradient_matrix",
    "gradient_spherical",
    "hosek_wilkie_diffuse_transport",
    "hue_saturation_value",
    "image_texture_srgb",
    "image_texture_sampling_modes",
    "image_texture_projection_modes",
    "indirect_diffuse",
    "indirect_principled",
    "integrator_clamp_direct",
    "invert_color_matrix",
    "legacy_separate_combine_matrix",
    "layer_weight_matrix",
    "map_range_matrix",
    "math_edge_cases",
    "math_operations",
    "mapping_modes",
    "mix_color_modes",
    "mix_color_edge_cases",
    "mix_data_types",
    "mix_rgb_legacy_modes",
    "mix_shader_emission",
    "negative_scale_surface",
    "nishita_diffuse_transport",
    "node_group_color",
    "noise_bump_object",
    "noise_color_3d",
    "noise_factor_2d",
    "noise_fbm_matrix",
    "noise_hetero_terrain_matrix",
    "noise_hybrid_multifractal_matrix",
    "noise_multifractal_matrix",
    "noise_ridged_multifractal_matrix",
    "normal_map_surface",
    "normal_map_matrix",
    "normal_map_named_uv_matrix",
    "particle_random_instances",
    "particle_random_nonparticle",
    "point_light",
    "point_light_light_path",
    "point_light_nodes",
    "point_light_shadow_light_path",
    "point_light_shadow_limit",
    "point_light_soft_disk",
    "point_light_soft_sphere",
    "principled_alpha_surface",
    "principled_bump_glossy",
    "principled_coat_surface",
    "principled_emission",
    "principled_emission_layers",
    "principled_sheen_surface",
    "principled_surface",
    "principled_transmission_surface",
    "rgb_emission",
    "rgb_curve_matrix",
    "rgb_to_bw",
    "separate_color_modes",
    "spot_light",
    "spot_light_soft",
    "sun_light",
    "sun_light_disk",
    "transparent_mix",
    "transparent_data_pass",
    "translucent_bsdf_matrix",
    "translucent_surface",
    "triangle_light_solid_angle",
    "value_emission",
    "vector_math_matrix",
    "wavelength_matrix",
    "white_noise_dimensions",
)

_REPORT_PASSES = (
    "Combined",
    "Normal",
    "DiffCol",
    "GlossCol",
    "TransCol",
    "DiffDir",
    "DiffInd",
    "GlossDir",
    "GlossInd",
    "TransDir",
    "TransInd",
    "Emit",
    "Env",
)

# These gates cover renderer defects that the canonical probe was introduced
# to catch. Energy gates are appropriate when multiple unbiased direction
# mappings are valid. A probe may additionally require a relative-RMSE gate
# when its purpose is to preserve Cycles' exact sample mapping.
_PROBE_RATIO_GATES = {
    "camera_blackman_harris_filter": {
        "Combined": (0.9995, 1.0005),
    },
    "camera_dof_disk": {
        "Combined": (0.9995, 1.0005),
    },
    "glass_transport": {
        "Combined": (0.9999, 1.0001),
        "GlossCol": (0.9999, 1.0001),
        "GlossDir": (0.9999, 1.0001),
        "TransCol": (0.9999, 1.0001),
        "TransDir": (0.9999, 1.0001),
    },
    "indirect_principled": {
        "DiffInd": (0.98, 1.02),
        "GlossInd": (0.98, 1.02),
    },
    "nishita_diffuse_transport": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "hosek_wilkie_diffuse_transport": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "point_light_light_path": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "point_light_shadow_light_path": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "point_light_shadow_limit": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "principled_alpha_surface": {
        "Combined": (0.999999, 1.000001),
        "Env": (0.999999, 1.000001),
    },
    "principled_coat_surface": {
        "Combined": (0.99999, 1.00001),
        "DiffDir": (0.99999, 1.00001),
        "GlossDir": (0.99999, 1.00001),
        "DiffCol": (0.99999, 1.00001),
        "GlossCol": (0.99999, 1.00001),
        "Normal": (0.99999, 1.00001),
    },
    "principled_transmission_surface": {
        "Combined": (0.99999, 1.00001),
        "DiffCol": (0.99999, 1.00001),
        "GlossCol": (0.99999, 1.00001),
        "TransCol": (0.99999, 1.00001),
        "TransDir": (0.99999, 1.00001),
        "Normal": (0.99999, 1.00001),
    },
    "principled_emission": {
        "Combined": (0.999999, 1.000001),
        "Emit": (0.999999, 1.000001),
    },
    "principled_emission_layers": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "principled_sheen_surface": {
        "Combined": (0.99999, 1.00001),
        "DiffDir": (0.99999, 1.00001),
        "DiffCol": (0.99999, 1.00001),
        "Normal": (0.99999, 1.00001),
    },
    "triangle_light_solid_angle": {
        "Combined": (0.995, 1.005),
        "DiffDir": (0.995, 1.005),
    },
}

_PROBE_RELATIVE_RMSE_GATES = {
    "camera_blackman_harris_filter": {
        "Combined": 0.0005,
    },
    "camera_dof_disk": {
        "Combined": 0.0005,
    },
    "glass_transport": {
        "Combined": 0.0002,
        "GlossCol": 0.0001,
        "GlossDir": 0.0001,
        "TransCol": 0.0001,
        "TransDir": 0.0002,
    },
    "nishita_diffuse_transport": {
        "Combined": 0.0005,
        "DiffDir": 0.0005,
    },
    "point_light_light_path": {
        "Combined": 0.000005,
        "DiffDir": 0.000005,
    },
    "point_light_shadow_light_path": {
        "Combined": 0.000005,
        "DiffDir": 0.000005,
    },
    "point_light_shadow_limit": {
        "Combined": 0.000005,
        "DiffDir": 0.000005,
    },
    "principled_alpha_surface": {
        "Combined": 0.000001,
        "Env": 0.000001,
    },
    "principled_coat_surface": {
        "Combined": 0.0001,
        "DiffDir": 0.0001,
        "GlossDir": 0.0001,
        "DiffCol": 0.0001,
        "GlossCol": 0.0001,
        "Normal": 0.0001,
    },
    "principled_transmission_surface": {
        "Combined": 0.000005,
        "DiffCol": 0.0000001,
        "GlossCol": 0.000001,
        "TransCol": 0.0000001,
        "TransDir": 0.000002,
        "Normal": 0.0000001,
    },
    "principled_emission": {
        "Combined": 0.000001,
        "Emit": 0.000001,
    },
    "principled_emission_layers": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "principled_sheen_surface": {
        "Combined": 0.000002,
        "DiffDir": 0.000002,
        "DiffCol": 0.0000001,
        "Normal": 0.000001,
    },
    "triangle_light_solid_angle": {
        "Combined": 0.005,
        "DiffDir": 0.005,
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=pathlib.Path, required=True)
    parser.add_argument(
        "--psycles-render", type=pathlib.Path, required=True
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--backend", default="fallback")
    parser.add_argument("--cycles-device", default="CPU")
    parser.add_argument("--cycles-device-name", default="")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "probes",
        nargs="*",
        choices=_ALL_PROBES,
        default=list(_ALL_PROBES),
    )
    result = parser.parse_args()
    if result.width <= 0 or result.height <= 0 or result.samples <= 0:
        parser.error("width, height, and samples must be positive")
    return result


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _cycles_golden_command(
    blender: str,
    blend: pathlib.Path,
    golden_script: pathlib.Path,
    cycles: pathlib.Path,
    *,
    width: int,
    height: int,
    samples: int,
    device: str,
    device_name: str,
) -> list[str]:
    # Do not pass a seed override. The canonical probe .blend and the
    # exported Psycles scene must consume the same scene-owned seed.
    return [
        blender,
        str(blend),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(golden_script),
        "--",
        str(cycles),
        str(width),
        str(height),
        str(samples),
        "--cycles-device",
        device,
        "--device-name",
        device_name,
    ]


def _comparison_command(
    python: str,
    compare_script: pathlib.Path,
    cycles: pathlib.Path,
    report: pathlib.Path,
    triptych_dir: pathlib.Path,
    psycles_exr: pathlib.Path,
) -> list[str]:
    # Comparison is a standalone Python/OIIO/Pillow tool. It must not inherit
    # Blender's private Python environment, which need not contain Pillow.
    return [
        python,
        str(compare_script),
        str(cycles),
        str(report),
        "--triptych-dir",
        str(triptych_dir),
        *(
            f"{pass_name}={psycles_exr}"
            for pass_name in _REPORT_PASSES
        ),
    ]


def _probe_gate_failures(
    probe: str,
    report_payload: dict[str, object],
) -> list[str]:
    ratio_gates = _PROBE_RATIO_GATES.get(probe, {})
    rmse_gates = _PROBE_RELATIVE_RMSE_GATES.get(probe, {})
    passes = report_payload.get("passes")
    if not isinstance(passes, dict):
        return [f"{probe}: report has no pass dictionary"]
    failures: list[str] = []
    for pass_name, (minimum, maximum) in ratio_gates.items():
        pass_report = passes.get(pass_name)
        if not isinstance(pass_report, dict):
            failures.append(f"{probe}: missing {pass_name}")
            continue
        ratio = pass_report.get("luminance_mean_ratio")
        if not isinstance(ratio, (float, int)):
            failures.append(
                f"{probe}: {pass_name} has no numeric energy ratio"
            )
        elif not minimum <= float(ratio) <= maximum:
            failures.append(
                f"{probe}: {pass_name} energy ratio {float(ratio):.6f} "
                f"is outside [{minimum:.6f}, {maximum:.6f}]"
            )
    for pass_name, maximum in rmse_gates.items():
        pass_report = passes.get(pass_name)
        if not isinstance(pass_report, dict):
            if pass_name not in ratio_gates:
                failures.append(f"{probe}: missing {pass_name}")
            continue
        relative_rmse = pass_report.get("relative_rmse")
        if not isinstance(relative_rmse, (float, int)):
            failures.append(
                f"{probe}: {pass_name} has no numeric relative RMSE"
            )
        elif float(relative_rmse) > maximum:
            failures.append(
                f"{probe}: {pass_name} relative RMSE "
                f"{float(relative_rmse):.6f} exceeds {maximum:.6f}"
            )
    return failures


def _main() -> int:
    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parent.parent
    create_script = root / "tools/create_cycles_shader_probe.py"
    golden_script = root / "tools/render_cycles_golden.py"
    export_script = root / "tools/export_psycles_scene.py"
    compare_script = root / "tools/compare_cycles.py"
    blender = str(arguments.blender.resolve())
    renderer = str(arguments.psycles_render.resolve())
    output_root = arguments.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for probe in arguments.probes:
        probe_root = output_root / probe
        probe_root.mkdir(parents=True, exist_ok=True)
        blend = probe_root / f"{probe}.blend"
        cycles = probe_root / f"{probe}-cycles.exr"
        bundle = probe_root / "export"
        preview = probe_root / f"{probe}-psycles.ppm"
        stem = preview.with_suffix("")
        psycles_exr = stem.with_suffix(".exr")
        report = probe_root / f"{probe}-diff.json"
        try:
            _run(
                [
                    blender,
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(create_script),
                    "--",
                    str(blend),
                    probe,
                ]
            )
            _run(
                _cycles_golden_command(
                    blender,
                    blend,
                    golden_script,
                    cycles,
                    width=arguments.width,
                    height=arguments.height,
                    samples=arguments.samples,
                    device=arguments.cycles_device,
                    device_name=arguments.cycles_device_name,
                )
            )
            _run(
                [
                    blender,
                    str(blend),
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(export_script),
                    "--",
                    str(bundle),
                ]
            )
            _run(
                [
                    renderer,
                    str(bundle),
                    str(preview),
                    arguments.backend,
                    str(arguments.width),
                    str(arguments.height),
                    str(arguments.samples),
                ]
            )
            _run(
                _comparison_command(
                    sys.executable,
                    compare_script,
                    cycles,
                    report,
                    probe_root / "triptychs",
                    psycles_exr,
                )
            )
            gate_failures = _probe_gate_failures(
                probe,
                json.loads(report.read_text(encoding="utf-8")),
            )
            if gate_failures:
                failures.append(probe)
                for failure in gate_failures:
                    print(f"probe gate failed: {failure}", file=sys.stderr)
        except subprocess.CalledProcessError:
            failures.append(probe)
            print(f"probe failed: {probe}", file=sys.stderr)

    if failures:
        print(
            "failed probes: " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    print(
        f"completed {len(arguments.probes)} shader probes: {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

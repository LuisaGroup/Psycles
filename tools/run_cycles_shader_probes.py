#!/usr/bin/env python3
"""Run canonical shader probes through Cycles and Psycles-Luisa.

The runner creates each .blend, renders the authoritative Cycles multilayer
EXR, exports the unchanged node graph, renders it with Psycles, and writes the
linear pass differential report.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


_ALL_PROBES = (
    "add_shader_emission",
    "background_world",
    "diffuse_surface",
    "emission_surface",
    "mix_shader_emission",
    "rgb_emission",
    "transparent_mix",
    "value_emission",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=pathlib.Path, required=True)
    parser.add_argument(
        "--psycles-render", type=pathlib.Path, required=True
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--backend", default="fallback")
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
        report = probe_root / f"{probe}-diff.json"
        try:
            _run(
                [
                    blender,
                    "--background",
                    "--python",
                    str(create_script),
                    "--",
                    str(blend),
                    probe,
                ]
            )
            _run(
                [
                    blender,
                    str(blend),
                    "--background",
                    "--python",
                    str(golden_script),
                    "--",
                    str(cycles),
                    str(arguments.width),
                    str(arguments.height),
                    str(arguments.samples),
                ]
            )
            _run(
                [
                    blender,
                    str(blend),
                    "--background",
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
                [
                    blender,
                    "--background",
                    "--python",
                    str(compare_script),
                    "--",
                    str(cycles),
                    str(report),
                    f"Combined={stem}-combined.pfm",
                    f"Normal={stem}-normal.pfm",
                    f"DiffCol={stem}-albedo.pfm",
                ]
            )
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

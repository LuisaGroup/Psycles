#!/usr/bin/env python3
"""Run the canonical five-way Cycles/Psycles scene benchmark.

The default matrix is intentionally fixed:

* Cycles CPU
* Cycles HIP
* Psycles/Luisa fallback
* Psycles/Luisa HIP
* Psycles/Luisa Vulkan

Every renderer receives the same scene-owned seed, resolution, and sample
count. The runner preserves command logs, structured timing metadata, EXR
hashes, and differential reports against both Cycles device variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any


_LUISA_BACKENDS = ("fallback", "hip", "vk")
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
_PSYCLES_TIMING_PATTERNS = {
    "scene_compile_seconds": re.compile(
        r"compiled .* in ([0-9.eE+-]+) s$",
        re.MULTILINE,
    ),
    "shader_jit_seconds": re.compile(
        r"Luisa shader JIT completed in ([0-9.eE+-]+) s$",
        re.MULTILINE,
    ),
    "render_seconds": re.compile(
        r"^Rendered .* in ([0-9.eE+-]+) s:",
        re.MULTILINE,
    ),
}


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one Blender scene on Cycles CPU/HIP and "
            "Psycles fallback/HIP/Vulkan"
        )
    )
    parser.add_argument("--blender", type=pathlib.Path, required=True)
    parser.add_argument(
        "--psycles-render",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument("--blend", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--bundle",
        type=pathlib.Path,
        help="scene export directory (default: OUTPUT_DIR/export)",
    )
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="reuse an existing bundle instead of exporting the scene",
    )
    parser.add_argument(
        "--cycles-hip-device-name",
        required=True,
        help="case-insensitive Cycles HIP device-name filter",
    )
    parser.add_argument(
        "--cycles-cpu-device-name",
        default="",
        help="optional Cycles CPU device-name filter",
    )
    parser.add_argument(
        "--width",
        type=_positive_integer,
        default=640,
    )
    parser.add_argument(
        "--height",
        type=_positive_integer,
        default=480,
    )
    parser.add_argument(
        "--samples",
        type=_positive_integer,
        default=64,
    )
    parser.add_argument(
        "--max-samples-per-dispatch",
        type=_positive_integer,
        default=8,
    )
    parser.add_argument(
        "--compiler-tmp",
        type=pathlib.Path,
        help="TMPDIR used by Luisa backend compilers",
    )
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="record timings and EXRs without differential reports",
    )
    return parser.parse_args()


def _cycles_command(
    blender: pathlib.Path,
    blend: pathlib.Path,
    script: pathlib.Path,
    output: pathlib.Path,
    *,
    width: int,
    height: int,
    samples: int,
    device: str,
    device_name: str,
) -> list[str]:
    return [
        str(blender),
        str(blend),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        str(output),
        str(width),
        str(height),
        str(samples),
        "--cycles-device",
        device,
        "--device-name",
        device_name,
    ]


def _export_command(
    blender: pathlib.Path,
    blend: pathlib.Path,
    script: pathlib.Path,
    bundle: pathlib.Path,
) -> list[str]:
    return [
        str(blender),
        str(blend),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        str(bundle),
    ]


def _psycles_command(
    renderer: pathlib.Path,
    bundle: pathlib.Path,
    preview: pathlib.Path,
    backend: str,
    *,
    width: int,
    height: int,
    samples: int,
    max_samples_per_dispatch: int,
) -> list[str]:
    return [
        str(renderer),
        str(bundle),
        str(preview),
        backend,
        str(width),
        str(height),
        str(samples),
        str(max_samples_per_dispatch),
    ]


def _comparison_command(
    python: pathlib.Path,
    script: pathlib.Path,
    reference: pathlib.Path,
    actual: pathlib.Path,
    report: pathlib.Path,
    triptychs: pathlib.Path,
    *,
    reference_label: str,
    actual_label: str,
) -> list[str]:
    return [
        str(python),
        str(script),
        str(reference),
        str(report),
        "--triptych-dir",
        str(triptychs),
        "--reference-label",
        reference_label,
        "--actual-label",
        actual_label,
        *(f"{name}={actual}" for name in _REPORT_PASSES),
    ]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _run_logged(
    command: list[str],
    log_path: pathlib.Path,
    *,
    environment: dict[str, str] | None = None,
    echo_output: bool = True,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    begin = time.perf_counter()
    output_parts: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        if process.stdout is None:
            raise RuntimeError("subprocess stdout pipe was not created")
        for line in process.stdout:
            output_parts.append(line)
            log.write(line)
            log.flush()
            if echo_output:
                print(line, end="", flush=True)
        returncode = process.wait()
    wall_seconds = time.perf_counter() - begin
    record = {
        "command": command,
        "log": str(log_path),
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "output": "".join(output_parts),
    }
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}: "
            + " ".join(command)
        )
    return record


def _public_process_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "output"
    }


def _parse_psycles_timings(output: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for name, pattern in _PSYCLES_TIMING_PATTERNS.items():
        match = pattern.search(output)
        if match is None:
            raise RuntimeError(
                f"Psycles output did not report {name}"
            )
        timings[name] = float(match.group(1))
    return timings


def _require_output(path: pathlib.Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"renderer did not produce output: {path}")


def _write_manifest(
    path: pathlib.Path,
    manifest: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative_performance(
    manifest: dict[str, Any],
) -> dict[str, dict[str, float]]:
    cycles = manifest["renderers"]["cycles"]
    psycles = manifest["renderers"]["psycles"]
    hip_seconds = float(cycles["hip"]["render_seconds"])
    cpu_seconds = float(cycles["cpu"]["render_seconds"])
    result: dict[str, dict[str, float]] = {}
    for backend, record in psycles.items():
        render_seconds = float(record["render_seconds"])
        result[backend] = {
            "speedup_over_cycles_hip": (
                hip_seconds / render_seconds
            ),
            "slowdown_vs_cycles_hip": (
                render_seconds / hip_seconds
            ),
            "speedup_over_cycles_cpu": (
                cpu_seconds / render_seconds
            ),
            "slowdown_vs_cycles_cpu": (
                render_seconds / cpu_seconds
            ),
        }
    result["cycles_cpu"] = {
        "speedup_over_cycles_hip": hip_seconds / cpu_seconds,
        "slowdown_vs_cycles_hip": cpu_seconds / hip_seconds,
    }
    return result


def _main() -> int:
    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parent.parent
    golden_script = root / "tools/render_cycles_golden.py"
    export_script = root / "tools/export_psycles_scene.py"
    compare_script = root / "tools/compare_cycles.py"
    blender = arguments.blender.resolve()
    renderer = arguments.psycles_render.resolve()
    blend = arguments.blend.resolve()
    output_root = arguments.output_dir.resolve()
    bundle = (
        arguments.bundle.resolve()
        if arguments.bundle is not None
        else output_root / "export"
    )
    for executable in (blender, renderer):
        if not executable.is_file():
            raise RuntimeError(f"executable does not exist: {executable}")
    if not blend.is_file():
        raise RuntimeError(f"scene does not exist: {blend}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "benchmark.json"
    manifest: dict[str, Any] = {
        "schema": "psycles.scene-benchmark.v1",
        "status": "running",
        "matrix": {
            "cycles": ["cpu", "hip"],
            "psycles": list(_LUISA_BACKENDS),
        },
        "scene": {
            "blend": str(blend),
            "sha256": _sha256(blend),
            "bundle": str(bundle),
        },
        "settings": {
            "width": arguments.width,
            "height": arguments.height,
            "samples": arguments.samples,
            "max_samples_per_dispatch": (
                arguments.max_samples_per_dispatch
            ),
        },
        "renderers": {
            "cycles": {},
            "psycles": {},
        },
        "commands": {},
        "comparisons": {},
    }
    _write_manifest(manifest_path, manifest)

    environment = os.environ.copy()
    if arguments.compiler_tmp is not None:
        compiler_tmp = arguments.compiler_tmp.resolve()
        compiler_tmp.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(compiler_tmp)
        manifest["compiler_tmp"] = str(compiler_tmp)

    try:
        cycles_outputs: dict[str, pathlib.Path] = {}
        for key, device, device_name in (
            ("cpu", "CPU", arguments.cycles_cpu_device_name),
            ("hip", "HIP", arguments.cycles_hip_device_name),
        ):
            output = output_root / "cycles" / f"{key}.exr"
            output.parent.mkdir(parents=True, exist_ok=True)
            record = _run_logged(
                _cycles_command(
                    blender,
                    blend,
                    golden_script,
                    output,
                    width=arguments.width,
                    height=arguments.height,
                    samples=arguments.samples,
                    device=device,
                    device_name=device_name,
                ),
                output_root / "logs" / f"cycles-{key}.log",
                environment=environment,
            )
            _require_output(output)
            metadata_path = output.with_suffix(".json")
            _require_output(metadata_path)
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            manifest["commands"][f"cycles_{key}"] = (
                _public_process_record(record)
            )
            manifest["renderers"]["cycles"][key] = {
                "output": str(output),
                "sha256": _sha256(output),
                "render_seconds": metadata["elapsed_seconds"],
                "device": metadata["cycles_enabled_devices"],
                "metadata": str(metadata_path),
                "process_wall_seconds": record["wall_seconds"],
            }
            cycles_outputs[key] = output
            _write_manifest(manifest_path, manifest)

        if arguments.reuse_export:
            _require_output(bundle / "scene.json")
            manifest["commands"]["export"] = {
                "reused": True,
            }
        else:
            export_record = _run_logged(
                _export_command(
                    blender,
                    blend,
                    export_script,
                    bundle,
                ),
                output_root / "logs/export.log",
                environment=environment,
            )
            _require_output(bundle / "scene.json")
            _require_output(bundle / "geometry.bin")
            manifest["commands"]["export"] = (
                _public_process_record(export_record)
            )
        manifest["scene"]["scene_json_sha256"] = _sha256(
            bundle / "scene.json"
        )
        manifest["scene"]["geometry_bin_sha256"] = _sha256(
            bundle / "geometry.bin"
        )
        _write_manifest(manifest_path, manifest)

        psycles_outputs: dict[str, pathlib.Path] = {}
        for backend in _LUISA_BACKENDS:
            preview = (
                output_root / "psycles" / f"{backend}.ppm"
            )
            preview.parent.mkdir(parents=True, exist_ok=True)
            record = _run_logged(
                _psycles_command(
                    renderer,
                    bundle,
                    preview,
                    backend,
                    width=arguments.width,
                    height=arguments.height,
                    samples=arguments.samples,
                    max_samples_per_dispatch=(
                        arguments.max_samples_per_dispatch
                    ),
                ),
                output_root / "logs" / f"psycles-{backend}.log",
                environment=environment,
            )
            exr = preview.with_suffix(".exr")
            _require_output(exr)
            timings = _parse_psycles_timings(record["output"])
            manifest["commands"][f"psycles_{backend}"] = (
                _public_process_record(record)
            )
            manifest["renderers"]["psycles"][backend] = {
                "output": str(exr),
                "sha256": _sha256(exr),
                **timings,
                "process_wall_seconds": record["wall_seconds"],
            }
            psycles_outputs[backend] = exr
            _write_manifest(manifest_path, manifest)

        manifest["relative_performance"] = (
            _relative_performance(manifest)
        )

        if not arguments.skip_comparisons:
            candidates = {
                "cycles-cpu": (
                    cycles_outputs["cpu"],
                    "Cycles CPU",
                ),
                **{
                    f"psycles-{backend}": (
                        output,
                        f"Psycles {backend}",
                    )
                    for backend, output in psycles_outputs.items()
                },
            }
            references = {
                "cycles-hip": (
                    cycles_outputs["hip"],
                    "Cycles HIP",
                ),
                "cycles-cpu": (
                    cycles_outputs["cpu"],
                    "Cycles CPU",
                ),
            }
            for reference_key, (
                reference,
                reference_label,
            ) in references.items():
                for candidate_key, (
                    actual,
                    actual_label,
                ) in candidates.items():
                    if actual == reference:
                        continue
                    comparison_key = (
                        f"{candidate_key}-vs-{reference_key}"
                    )
                    comparison_root = (
                        output_root
                        / "comparisons"
                        / comparison_key
                    )
                    report = comparison_root / "report.json"
                    record = _run_logged(
                        _comparison_command(
                            pathlib.Path(sys.executable),
                            compare_script,
                            reference,
                            actual,
                            report,
                            comparison_root / "triptychs",
                            reference_label=reference_label,
                            actual_label=actual_label,
                        ),
                        (
                            output_root
                            / "logs"
                            / f"{comparison_key}.log"
                        ),
                        environment=environment,
                        echo_output=False,
                    )
                    _require_output(report)
                    manifest["comparisons"][comparison_key] = {
                        "reference": str(reference),
                        "actual": str(actual),
                        "report": str(report),
                        "process_wall_seconds": (
                            record["wall_seconds"]
                        ),
                    }
                    _write_manifest(manifest_path, manifest)

        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
    except Exception as exception:
        manifest["status"] = "failed"
        manifest["error"] = str(exception)
        _write_manifest(manifest_path, manifest)
        raise

    print(
        "Completed Cycles CPU/HIP and Psycles "
        "fallback/HIP/Vulkan benchmark: "
        f"{manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

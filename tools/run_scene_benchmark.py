#!/usr/bin/env python3
"""Run a tracked Cycles/Psycles scene benchmark.

The default matrix remains the canonical AMD five-way run:

* Cycles CPU
* Cycles HIP
* Psycles/Luisa fallback
* Psycles/Luisa HIP
* Psycles/Luisa Vulkan

Every renderer receives the same scene-owned seed, resolution, and sample
count. The runner preserves command logs, structured timing metadata, EXR
hashes, and differential reports against both Cycles device variants.

The device and backend selectors also support native platform matrices, such
as Cycles Metal against Psycles fallback/Metal on Apple Silicon.
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
_KNOWN_LUISA_BACKENDS = {
    "cuda",
    "dx",
    "fallback",
    "hip",
    "metal",
    "remote",
    "vk",
}
_KNOWN_PSYCLES_SCHEDULERS = (
    "megakernel",
    "megakernel-per-sample",
    "wavefront",
    "wavefront-graph",
    "wavefront-staged",
    "persistent",
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
    "Volume Direct",
    "Volume Indirect",
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


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _wavefront_tail_threshold(value: str) -> int | str:
    if value == "auto":
        return value
    return _nonnegative_integer(value)


def _backend_list(value: str) -> tuple[str, ...]:
    result = tuple(
        backend.strip().lower()
        for backend in value.split(",")
        if backend.strip()
    )
    if not result:
        raise argparse.ArgumentTypeError(
            "at least one Psycles backend is required"
        )
    unknown = sorted(set(result) - _KNOWN_LUISA_BACKENDS)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown Psycles backend(s): " + ", ".join(unknown)
        )
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "Psycles backends must not be repeated"
        )
    return result


def _scheduler_list(value: str) -> tuple[str, ...]:
    result = tuple(
        scheduler.strip().lower()
        for scheduler in value.split(",")
        if scheduler.strip()
    )
    if not result:
        raise argparse.ArgumentTypeError(
            "at least one Psycles scheduler is required"
        )
    unknown = sorted(
        set(result) - set(_KNOWN_PSYCLES_SCHEDULERS)
    )
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown Psycles scheduler(s): " + ", ".join(unknown)
        )
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "Psycles schedulers must not be repeated"
        )
    return result


def _wavefront_block_size(value: str) -> int:
    result = _positive_integer(value)
    if result > 1024 or result % 32 != 0:
        raise argparse.ArgumentTypeError(
            "wavefront block size must be a multiple of 32 in [32, 1024]"
        )
    return result


def _psycles_run_matrix(
    backends: tuple[str, ...],
    schedulers: tuple[str, ...],
) -> tuple[tuple[str, str, str | None], ...]:
    if not schedulers:
        return tuple((backend, backend, None) for backend in backends)
    return tuple(
        (f"{backend}-{scheduler}", backend, scheduler)
        for backend in backends
        for scheduler in schedulers
    )


def _device_key(device: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", device.lower()).strip("-")
    if not result or result == "cpu":
        raise ValueError(
            "the selected Cycles GPU device must not be CPU"
        )
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one Blender scene on Cycles CPU/GPU and "
            "selected Psycles backends"
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
        "--resume",
        action="store_true",
        help=(
            "reuse completed renders from a compatible benchmark.json "
            "after validating commands and output hashes"
        ),
    )
    parser.add_argument(
        "--cycles-hip-device-name",
        default="",
        help=(
            "legacy alias for --cycles-gpu-device-name when the "
            "GPU device is HIP"
        ),
    )
    parser.add_argument(
        "--cycles-gpu-device",
        default="HIP",
        help="Cycles GPU compute backend, for example HIP or METAL",
    )
    parser.add_argument(
        "--cycles-gpu-device-name",
        default="",
        help="case-insensitive Cycles GPU device-name filter",
    )
    parser.add_argument(
        "--cycles-cpu-device-name",
        default="",
        help="optional Cycles CPU device-name filter",
    )
    parser.add_argument(
        "--skip-cycles-cpu",
        action="store_true",
        help="benchmark only the selected Cycles GPU device",
    )
    parser.add_argument(
        "--psycles-backends",
        type=_backend_list,
        default=_LUISA_BACKENDS,
        help=(
            "comma-separated Luisa backends "
            "(default: fallback,hip,vk)"
        ),
    )
    parser.add_argument(
        "--psycles-schedulers",
        type=_scheduler_list,
        default=(),
        help=(
            "optional comma-separated scheduler matrix; when omitted, "
            "the renderer's legacy default scheduler is used"
        ),
    )
    parser.add_argument(
        "--wavefront-execution-block-size",
        type=_wavefront_block_size,
        default=32,
        help="wavefront execution block size (default: 32)",
    )
    parser.add_argument(
        "--persistent-worker-count",
        type=_positive_integer,
        default=32768,
        help="persistent scheduler worker count (default: 32768)",
    )
    parser.add_argument(
        "--persistent-block-size",
        type=_wavefront_block_size,
        default=32,
        help="persistent scheduler block size (default: 32)",
    )
    parser.add_argument(
        "--persistent-fetch-size",
        type=_positive_integer,
        default=1,
        help="persistent scheduler fetch size (default: 1)",
    )
    parser.add_argument(
        "--staged-surface-sorting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable staged scheduler surface sorting",
    )
    parser.add_argument(
        "--staged-direct-light-queue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the staged scheduler direct-light queue",
    )
    parser.add_argument(
        "--wavefront-counter-readback-batch-size",
        type=_positive_integer,
        default=4,
        help="graph counter readback batch size (default: 4)",
    )
    parser.add_argument(
        "--wavefront-counter-readback-pipeline-depth",
        type=_positive_integer,
        default=2,
        help="graph counter readback pipeline depth (default: 2)",
    )
    parser.add_argument(
        "--wavefront-tail-megakernel-threshold",
        type=_wavefront_tail_threshold,
        default="auto",
        help=(
            "graph tail-megakernel threshold, or auto to derive it from "
            "active frame capacity (default: auto)"
        ),
    )
    parser.add_argument(
        "--wavefront-graph-worker-count",
        type=_nonnegative_integer,
        default=0,
        help="graph worker count, or zero for automatic (default: 0)",
    )
    parser.add_argument(
        "--wavefront-graph-selective",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable graph selective scheduling",
    )
    parser.add_argument(
        "--wavefront-graph-refill-threshold",
        type=_nonnegative_integer,
        default=0,
        help="graph selective refill threshold (default: 0)",
    )
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Psycles fast math",
    )
    parser.add_argument(
        "--wavefront-frame-capacity",
        type=_positive_integer,
        default=1048576,
        help="wavefront frame capacity (default: 1048576)",
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
        default=4,
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
    result = parser.parse_args()
    result.cycles_gpu_device = (
        result.cycles_gpu_device.strip().upper()
    )
    if not result.cycles_gpu_device:
        parser.error("--cycles-gpu-device must not be empty")
    if not result.cycles_gpu_device_name:
        if (
            result.cycles_gpu_device == "HIP"
            and result.cycles_hip_device_name
        ):
            result.cycles_gpu_device_name = (
                result.cycles_hip_device_name
            )
        else:
            parser.error(
                "--cycles-gpu-device-name is required "
                "(or use --cycles-hip-device-name for HIP)"
            )
    return result


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
    scheduler: str | None = None,
    wavefront_execution_block_size: int = 32,
    persistent_worker_count: int = 32768,
    persistent_block_size: int = 32,
    persistent_fetch_size: int = 1,
    staged_surface_sorting: bool = True,
    staged_direct_light_queue: bool = False,
    wavefront_counter_readback_batch_size: int = 4,
    wavefront_counter_readback_pipeline_depth: int = 2,
    wavefront_tail_megakernel_threshold: int | str = "auto",
    wavefront_graph_worker_count: int = 0,
    wavefront_graph_selective: bool = False,
    wavefront_graph_refill_threshold: int = 0,
    fast_math: bool = True,
    wavefront_frame_capacity: int = 1048576,
) -> list[str]:
    command = [
        str(renderer),
        str(bundle),
        str(preview),
        backend,
        str(width),
        str(height),
        str(samples),
        str(max_samples_per_dispatch),
    ]
    if scheduler is not None:
        command.extend([
            "-",
            str(width // 2),
            str(height // 2),
            "0",
            "0",
            str(samples),
            "-",
            "1",
            "0",
            scheduler,
            str(wavefront_execution_block_size),
            str(persistent_worker_count),
            str(persistent_block_size),
            str(persistent_fetch_size),
            str(int(staged_surface_sorting)),
            str(int(staged_direct_light_queue)),
            str(wavefront_counter_readback_batch_size),
            str(wavefront_counter_readback_pipeline_depth),
            str(wavefront_tail_megakernel_threshold),
            str(wavefront_graph_worker_count),
            str(int(wavefront_graph_selective)),
            str(wavefront_graph_refill_threshold),
            str(int(fast_math)),
            str(wavefront_frame_capacity),
        ])
    return command


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


_BLENDER_BUILD_IDENTITY_FIELDS = (
    "version",
    "version_cycle",
    "version_tuple",
    "build_hash",
    "build_branch",
    "build_type",
)


def _blender_build_identity(
    document: dict[str, Any],
    source: pathlib.Path,
) -> dict[str, Any]:
    identity = document.get("blender_build")
    if not isinstance(identity, dict):
        raise RuntimeError(
            f"{source} has no exact Blender build identity; regenerate it "
            "with the current Psycles exporter/golden script"
        )
    missing = [
        field
        for field in _BLENDER_BUILD_IDENTITY_FIELDS
        if field not in identity
    ]
    if missing:
        raise RuntimeError(
            f"{source} has an incomplete Blender build identity: "
            + ", ".join(missing)
        )
    if (
        not isinstance(identity["version_tuple"], list)
        or len(identity["version_tuple"]) != 3
        or not all(
            isinstance(value, int)
            for value in identity["version_tuple"]
        )
    ):
        raise RuntimeError(
            f"{source} has an invalid Blender version tuple"
        )
    text_fields = (
        field
        for field in _BLENDER_BUILD_IDENTITY_FIELDS
        if field != "version_tuple"
    )
    invalid_text_fields = [
        field
        for field in text_fields
        if not isinstance(identity[field], str)
        or not identity[field]
    ]
    if invalid_text_fields:
        raise RuntimeError(
            f"{source} has invalid Blender build identity fields: "
            + ", ".join(invalid_text_fields)
        )
    return {
        field: identity[field]
        for field in _BLENDER_BUILD_IDENTITY_FIELDS
    }


def _require_same_blender_build(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_source: pathlib.Path,
    candidate_source: pathlib.Path,
) -> None:
    if candidate != reference:
        raise RuntimeError(
            "Cycles golden and Psycles export use different Blender builds: "
            f"{reference_source} reports {reference}, while "
            f"{candidate_source} reports {candidate}. Re-export and render "
            "the golden with the same Blender executable."
        )


def _validate_resume_configuration(
    previous: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Require the render-defining benchmark identity to match exactly."""

    if previous.get("schema") != expected["schema"]:
        raise RuntimeError(
            "cannot resume benchmark with a different manifest schema"
        )
    for section in ("matrix", "settings"):
        if previous.get(section) != expected[section]:
            raise RuntimeError(
                f"cannot resume benchmark with different {section}"
            )
    previous_scene = previous.get("scene")
    if not isinstance(previous_scene, dict):
        raise RuntimeError("cannot resume benchmark without scene identity")
    for key in ("blend", "sha256", "bundle"):
        if previous_scene.get(key) != expected["scene"][key]:
            raise RuntimeError(
                "cannot resume benchmark with a different scene or bundle"
            )


def _has_matching_output(
    record: Any,
    expected_output: pathlib.Path,
) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        recorded_output = pathlib.Path(record["output"]).resolve()
        recorded_hash = record["sha256"]
    except (KeyError, TypeError):
        return False
    if recorded_output != expected_output.resolve():
        return False
    if not expected_output.is_file() or expected_output.stat().st_size == 0:
        return False
    if not isinstance(recorded_hash, str):
        return False
    return _sha256(expected_output) == recorded_hash


def _can_resume_render(
    manifest: dict[str, Any],
    *,
    command_key: str,
    renderer_group: str,
    renderer_key: str,
    expected_command: list[str],
    expected_output: pathlib.Path,
    required_timings: tuple[str, ...],
    metadata_path: pathlib.Path | None = None,
) -> bool:
    commands = manifest.get("commands")
    renderers = manifest.get("renderers")
    if not isinstance(commands, dict) or not isinstance(renderers, dict):
        return False
    command_record = commands.get(command_key)
    group = renderers.get(renderer_group)
    if not isinstance(command_record, dict) or not isinstance(group, dict):
        return False
    if command_record.get("returncode") != 0:
        return False
    if command_record.get("command") != expected_command:
        return False
    record = group.get(renderer_key)
    if not _has_matching_output(record, expected_output):
        return False
    assert isinstance(record, dict)
    for timing in required_timings:
        value = record.get(timing)
        if not isinstance(value, (int, float)) or value < 0.0:
            return False
    if metadata_path is not None:
        recorded_metadata = record.get("metadata")
        if not isinstance(recorded_metadata, str):
            return False
        if pathlib.Path(recorded_metadata).resolve() != metadata_path.resolve():
            return False
        if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
            return False
        metadata_hash = record.get("metadata_sha256")
        if metadata_hash is not None and (
            not isinstance(metadata_hash, str)
            or _sha256(metadata_path) != metadata_hash
        ):
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            elapsed_seconds = float(metadata["elapsed_seconds"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if elapsed_seconds != float(record["render_seconds"]):
            return False
    return True


def _bundle_matches_manifest(
    manifest: dict[str, Any],
    bundle: pathlib.Path,
) -> bool:
    scene = manifest.get("scene")
    if not isinstance(scene, dict):
        return False
    for key, path in (
        ("scene_json_sha256", bundle / "scene.json"),
        ("geometry_bin_sha256", bundle / "geometry.bin"),
    ):
        recorded_hash = scene.get(key)
        if (
            not isinstance(recorded_hash, str)
            or not path.is_file()
            or path.stat().st_size == 0
            or _sha256(path) != recorded_hash
        ):
            return False
    return True


def _can_resume_export(
    manifest: dict[str, Any],
    expected_command: list[str],
    bundle: pathlib.Path,
) -> bool:
    commands = manifest.get("commands")
    if not isinstance(commands, dict):
        return False
    record = commands.get("export")
    return (
        isinstance(record, dict)
        and record.get("returncode") == 0
        and record.get("command") == expected_command
        and _bundle_matches_manifest(manifest, bundle)
    )


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
    gpu_key: str | None = None,
) -> dict[str, dict[str, float]]:
    cycles = manifest["renderers"]["cycles"]
    psycles = manifest["renderers"]["psycles"]
    if gpu_key is None:
        gpu_keys = [key for key in cycles if key != "cpu"]
        if len(gpu_keys) != 1:
            raise RuntimeError(
                "benchmark manifest must contain exactly one Cycles GPU"
            )
        gpu_key = gpu_keys[0]
    gpu_seconds = float(cycles[gpu_key]["render_seconds"])
    cpu_seconds = (
        float(cycles["cpu"]["render_seconds"])
        if "cpu" in cycles
        else None
    )
    result: dict[str, dict[str, float]] = {}
    for backend, record in psycles.items():
        render_seconds = float(record["render_seconds"])
        result[backend] = {
            f"speedup_over_cycles_{gpu_key}": (
                gpu_seconds / render_seconds
            ),
            f"slowdown_vs_cycles_{gpu_key}": (
                render_seconds / gpu_seconds
            ),
        }
        if cpu_seconds is not None:
            result[backend].update({
                "speedup_over_cycles_cpu": (
                    cpu_seconds / render_seconds
                ),
                "slowdown_vs_cycles_cpu": (
                    render_seconds / cpu_seconds
                ),
            })
    if cpu_seconds is not None:
        result["cycles_cpu"] = {
            f"speedup_over_cycles_{gpu_key}": (
                gpu_seconds / cpu_seconds
            ),
            f"slowdown_vs_cycles_{gpu_key}": (
                cpu_seconds / gpu_seconds
            ),
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
    gpu_device = arguments.cycles_gpu_device
    gpu_key = _device_key(gpu_device)
    gpu_label = f"Cycles {gpu_device}"
    cycles_matrix = (
        ["cpu", gpu_key]
        if not arguments.skip_cycles_cpu
        else [gpu_key]
    )
    psycles_runs = _psycles_run_matrix(
        arguments.psycles_backends,
        arguments.psycles_schedulers,
    )
    psycles_matrix = [run_key for run_key, _, _ in psycles_runs]
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
    fresh_manifest: dict[str, Any] = {
        "schema": "psycles.scene-benchmark.v1",
        "status": "running",
        "matrix": {
            "cycles": cycles_matrix,
            "psycles": psycles_matrix,
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
            "cycles_gpu_device": gpu_device,
            "cycles_gpu_device_name": (
                arguments.cycles_gpu_device_name
            ),
            "psycles_environment": {
                name: os.environ.get(name)
                for name in (
                    "PSYCLES_COMPACT_SURFACE_VALUES",
                    "PSYCLES_POPULATE_SURFACE_ONCE",
                )
            },
        },
        "renderers": {
            "cycles": {},
            "psycles": {},
        },
        "commands": {},
        "comparisons": {},
    }
    if arguments.psycles_schedulers:
        fresh_manifest["settings"].update({
            "psycles_schedulers": list(
                arguments.psycles_schedulers
            ),
            "psycles_scheduler_config": {
                "wavefront_execution_block_size": (
                    arguments.wavefront_execution_block_size
                ),
                "persistent_worker_count": (
                    arguments.persistent_worker_count
                ),
                "persistent_block_size": arguments.persistent_block_size,
                "persistent_fetch_size": arguments.persistent_fetch_size,
                "staged_surface_sorting": arguments.staged_surface_sorting,
                "staged_direct_light_queue": (
                    arguments.staged_direct_light_queue
                ),
                "wavefront_counter_readback_batch_size": (
                    arguments.wavefront_counter_readback_batch_size
                ),
                "wavefront_counter_readback_pipeline_depth": (
                    arguments.wavefront_counter_readback_pipeline_depth
                ),
                "wavefront_tail_megakernel_threshold": (
                    arguments.wavefront_tail_megakernel_threshold
                ),
                "wavefront_graph_worker_count": (
                    arguments.wavefront_graph_worker_count
                ),
                "wavefront_graph_selective": (
                    arguments.wavefront_graph_selective
                ),
                "wavefront_graph_refill_threshold": (
                    arguments.wavefront_graph_refill_threshold
                ),
                "fast_math": arguments.fast_math,
                "wavefront_frame_capacity": (
                    arguments.wavefront_frame_capacity
                ),
            },
        })
    resumed_existing_manifest = arguments.resume and manifest_path.is_file()
    if resumed_existing_manifest:
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exception:
            raise RuntimeError(
                f"cannot resume invalid benchmark manifest: {manifest_path}"
            ) from exception
        if not isinstance(manifest, dict):
            raise RuntimeError(
                f"cannot resume invalid benchmark manifest: {manifest_path}"
            )
        _validate_resume_configuration(manifest, fresh_manifest)
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest.pop("relative_performance", None)
        manifest["comparisons"] = {}
    else:
        manifest = fresh_manifest
    manifest["resume"] = {
        "requested": arguments.resume,
        "existing_manifest": resumed_existing_manifest,
        "reused_stages": [],
    }
    _write_manifest(manifest_path, manifest)

    environment = os.environ.copy()
    if arguments.compiler_tmp is not None:
        compiler_tmp = arguments.compiler_tmp.resolve()
        compiler_tmp.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(compiler_tmp)
        manifest["compiler_tmp"] = str(compiler_tmp)
    if "vk" in arguments.psycles_backends:
        environment.setdefault(
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV",
            "1",
        )

    try:
        cycles_outputs: dict[str, pathlib.Path] = {}
        cycles_build: dict[str, Any] | None = None
        cycles_build_source: pathlib.Path | None = None
        cycles_runs = []
        if not arguments.skip_cycles_cpu:
            cycles_runs.append(
                ("cpu", "CPU", arguments.cycles_cpu_device_name)
            )
        cycles_runs.append((
            gpu_key,
            gpu_device,
            arguments.cycles_gpu_device_name,
        ))
        for key, device, device_name in cycles_runs:
            output = output_root / "cycles" / f"{key}.exr"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = _cycles_command(
                blender,
                blend,
                golden_script,
                output,
                width=arguments.width,
                height=arguments.height,
                samples=arguments.samples,
                device=device,
                device_name=device_name,
            )
            metadata_path = output.with_suffix(".json")
            stage = f"cycles_{key}"
            if arguments.resume and _can_resume_render(
                manifest,
                command_key=stage,
                renderer_group="cycles",
                renderer_key=key,
                expected_command=command,
                expected_output=output,
                required_timings=("render_seconds",),
                metadata_path=metadata_path,
            ):
                manifest["resume"]["reused_stages"].append(stage)
                print(f"Reusing validated benchmark stage: {stage}")
            else:
                manifest["commands"].pop(stage, None)
                manifest["renderers"]["cycles"].pop(key, None)
                _write_manifest(manifest_path, manifest)
                record = _run_logged(
                    command,
                    output_root / "logs" / f"cycles-{key}.log",
                    environment=environment,
                )
                _require_output(output)
                _require_output(metadata_path)
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                manifest["commands"][stage] = (
                    _public_process_record(record)
                )
                manifest["renderers"]["cycles"][key] = {
                    "output": str(output),
                    "sha256": _sha256(output),
                    "render_seconds": metadata["elapsed_seconds"],
                    "device": metadata["cycles_enabled_devices"],
                    "metadata": str(metadata_path),
                    "metadata_sha256": _sha256(metadata_path),
                    "process_wall_seconds": record["wall_seconds"],
                }
            cycles_outputs[key] = output
            metadata_document = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            current_cycles_build = _blender_build_identity(
                metadata_document, metadata_path
            )
            if cycles_build is None:
                cycles_build = current_cycles_build
                cycles_build_source = metadata_path
            else:
                assert cycles_build_source is not None
                _require_same_blender_build(
                    cycles_build,
                    current_cycles_build,
                    reference_source=cycles_build_source,
                    candidate_source=metadata_path,
                )
            _write_manifest(manifest_path, manifest)

        export_command = _export_command(
            blender,
            blend,
            export_script,
            bundle,
        )
        if arguments.reuse_export:
            _require_output(bundle / "scene.json")
            _require_output(bundle / "geometry.bin")
            if (
                resumed_existing_manifest
                and not _bundle_matches_manifest(manifest, bundle)
            ):
                raise RuntimeError(
                    "cannot resume because the reused export changed"
                )
            manifest["commands"]["export"] = {
                "reused": True,
            }
            if resumed_existing_manifest:
                manifest["resume"]["reused_stages"].append("export")
        elif arguments.resume and _can_resume_export(
            manifest,
            export_command,
            bundle,
        ):
            manifest["resume"]["reused_stages"].append("export")
            print("Reusing validated benchmark stage: export")
        else:
            manifest["commands"].pop("export", None)
            if resumed_existing_manifest:
                for run_key, _, _ in psycles_runs:
                    manifest["commands"].pop(
                        f"psycles_{run_key}", None
                    )
                    manifest["renderers"]["psycles"].pop(
                        run_key, None
                    )
            _write_manifest(manifest_path, manifest)
            export_record = _run_logged(
                export_command,
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
        bundle_scene_path = bundle / "scene.json"
        bundle_scene = json.loads(
            bundle_scene_path.read_text(encoding="utf-8")
        )
        if not isinstance(bundle_scene, dict):
            raise RuntimeError(
                f"scene bundle root is not an object: {bundle_scene_path}"
            )
        bundle_build = _blender_build_identity(
            bundle_scene, bundle_scene_path
        )
        if cycles_build is None or cycles_build_source is None:
            raise RuntimeError("benchmark has no Cycles build identity")
        _require_same_blender_build(
            cycles_build,
            bundle_build,
            reference_source=cycles_build_source,
            candidate_source=bundle_scene_path,
        )
        manifest["scene"]["blender_build"] = bundle_build
        _write_manifest(manifest_path, manifest)

        psycles_outputs: dict[str, pathlib.Path] = {}
        psycles_labels: dict[str, str] = {}
        for run_key, backend, scheduler in psycles_runs:
            preview = (
                output_root / "psycles" / f"{run_key}.ppm"
            )
            preview.parent.mkdir(parents=True, exist_ok=True)
            command = _psycles_command(
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
                scheduler=scheduler,
                wavefront_execution_block_size=(
                    arguments.wavefront_execution_block_size
                ),
                persistent_worker_count=arguments.persistent_worker_count,
                persistent_block_size=arguments.persistent_block_size,
                persistent_fetch_size=arguments.persistent_fetch_size,
                staged_surface_sorting=arguments.staged_surface_sorting,
                staged_direct_light_queue=(
                    arguments.staged_direct_light_queue
                ),
                wavefront_counter_readback_batch_size=(
                    arguments.wavefront_counter_readback_batch_size
                ),
                wavefront_counter_readback_pipeline_depth=(
                    arguments.wavefront_counter_readback_pipeline_depth
                ),
                wavefront_tail_megakernel_threshold=(
                    arguments.wavefront_tail_megakernel_threshold
                ),
                wavefront_graph_worker_count=(
                    arguments.wavefront_graph_worker_count
                ),
                wavefront_graph_selective=(
                    arguments.wavefront_graph_selective
                ),
                wavefront_graph_refill_threshold=(
                    arguments.wavefront_graph_refill_threshold
                ),
                fast_math=arguments.fast_math,
                wavefront_frame_capacity=arguments.wavefront_frame_capacity,
            )
            exr = preview.with_suffix(".exr")
            stage = f"psycles_{run_key}"
            if arguments.resume and _can_resume_render(
                manifest,
                command_key=stage,
                renderer_group="psycles",
                renderer_key=run_key,
                expected_command=command,
                expected_output=exr,
                required_timings=(
                    "scene_compile_seconds",
                    "shader_jit_seconds",
                    "render_seconds",
                ),
            ):
                manifest["resume"]["reused_stages"].append(stage)
                print(f"Reusing validated benchmark stage: {stage}")
            else:
                manifest["commands"].pop(stage, None)
                manifest["renderers"]["psycles"].pop(run_key, None)
                _write_manifest(manifest_path, manifest)
                record = _run_logged(
                    command,
                    output_root / "logs" / f"psycles-{run_key}.log",
                    environment=environment,
                )
                _require_output(exr)
                timings = _parse_psycles_timings(record["output"])
                manifest["commands"][stage] = (
                    _public_process_record(record)
                )
                renderer_record = {
                    "output": str(exr),
                    "sha256": _sha256(exr),
                    "backend": backend,
                    **timings,
                    "process_wall_seconds": record["wall_seconds"],
                }
                if scheduler is not None:
                    renderer_record["scheduler"] = scheduler
                manifest["renderers"]["psycles"][run_key] = (
                    renderer_record
                )
            psycles_outputs[run_key] = exr
            psycles_labels[run_key] = (
                f"Psycles {backend}"
                if scheduler is None
                else f"Psycles {backend} ({scheduler})"
            )
            _write_manifest(manifest_path, manifest)

        manifest["relative_performance"] = (
            _relative_performance(manifest, gpu_key)
        )

        if not arguments.skip_comparisons:
            candidates = {
                f"psycles-{run_key}": (
                    output,
                    psycles_labels[run_key],
                )
                for run_key, output in psycles_outputs.items()
            }
            if "cpu" in cycles_outputs:
                candidates["cycles-cpu"] = (
                    cycles_outputs["cpu"],
                    "Cycles CPU",
                )
            references = {
                f"cycles-{gpu_key}": (
                    cycles_outputs[gpu_key],
                    gpu_label,
                ),
            }
            if "cpu" in cycles_outputs:
                references["cycles-cpu"] = (
                    cycles_outputs["cpu"],
                    "Cycles CPU",
                )
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
        "Completed Cycles/Psycles benchmark "
        f"({', '.join(cycles_matrix)} vs "
        f"{', '.join(psycles_matrix)}): "
        f"{manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Run a command while measuring AMD VRAM usage through Linux sysfs.

Example:

    python3 tools/measure_amd_vram.py \
        --output /tmp/cycles-vram.json -- \
        blender scene.blend --background --python render.py

The report distinguishes the machine-wide VRAM baseline from the absolute
peak and its increase over that baseline.  The wrapped command keeps its
normal standard streams, and this process returns the same exit status.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any


_AMD_PCI_VENDOR = "0x1002"


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="measure AMD VRAM while running a command"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="JSON report path",
    )
    parser.add_argument(
        "--device",
        type=pathlib.Path,
        help=(
            "DRM card device directory; by default the unique AMD card "
            "with VRAM accounting is selected"
        ),
    )
    parser.add_argument(
        "--interval",
        type=_positive_float,
        default=0.05,
        help="sampling interval in seconds (default: 0.05)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command following --",
    )
    arguments = parser.parse_args()
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    if not arguments.command:
        parser.error("a command is required after --")
    return arguments


def _read_integer(path: pathlib.Path) -> int:
    return int(path.read_text(encoding="ascii").strip())


def _device_paths(device: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    resolved = device.resolve()
    used = resolved / "mem_info_vram_used"
    total = resolved / "mem_info_vram_total"
    if not used.is_file() or not total.is_file():
        raise RuntimeError(
            f"{resolved} does not expose AMD VRAM accounting"
        )
    return used, total


def _discover_device() -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    for card in sorted(pathlib.Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        vendor = device / "vendor"
        try:
            is_amd = (
                vendor.read_text(encoding="ascii").strip().casefold()
                == _AMD_PCI_VENDOR
            )
        except OSError:
            continue
        if is_amd:
            try:
                _device_paths(device)
            except RuntimeError:
                continue
            candidates.append(device.resolve())
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) != 1:
        rendered = ", ".join(map(str, unique_candidates)) or "none"
        raise RuntimeError(
            "expected exactly one AMD DRM device with VRAM accounting; "
            f"found {rendered}. Pass --device explicitly."
        )
    return unique_candidates[0]


def _write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _main() -> int:
    arguments = _arguments()
    device = (
        arguments.device.resolve()
        if arguments.device is not None
        else _discover_device()
    )
    used_path, total_path = _device_paths(device)
    total_bytes = _read_integer(total_path)
    baseline_bytes = _read_integer(used_path)
    peak_bytes = baseline_bytes
    peak_elapsed_seconds = 0.0
    sample_count = 1
    begin = time.monotonic()

    process = subprocess.Popen(
        arguments.command,
        start_new_session=True,
    )
    interrupted = False
    try:
        while True:
            used_bytes = _read_integer(used_path)
            elapsed_seconds = time.monotonic() - begin
            sample_count += 1
            if used_bytes > peak_bytes:
                peak_bytes = used_bytes
                peak_elapsed_seconds = elapsed_seconds
            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(arguments.interval)
    except KeyboardInterrupt:
        interrupted = True
        os.killpg(process.pid, signal.SIGINT)
        return_code = process.wait()

    final_bytes = _read_integer(used_path)
    elapsed_seconds = time.monotonic() - begin
    sample_count += 1
    if final_bytes > peak_bytes:
        peak_bytes = final_bytes
        peak_elapsed_seconds = elapsed_seconds
    report = {
        "schema": "psycles.amd-vram-measurement.v1",
        "command": arguments.command,
        "device": str(device),
        "total_bytes": total_bytes,
        "baseline_bytes": baseline_bytes,
        "peak_bytes": peak_bytes,
        "peak_increase_bytes": max(peak_bytes - baseline_bytes, 0),
        "peak_elapsed_seconds": peak_elapsed_seconds,
        "final_bytes": final_bytes,
        "elapsed_seconds": elapsed_seconds,
        "sampling_interval_seconds": arguments.interval,
        "sample_count": sample_count,
        "return_code": return_code,
        "interrupted": interrupted,
    }
    _write_report(arguments.output, report)
    return return_code


if __name__ == "__main__":
    sys.exit(_main())

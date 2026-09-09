"""Validate a rebuilt SDK with the original full-resolution Barbershop input.

This is one integration canary, not a paired benchmark or a parity verdict.
The reviewed equal-pass command and socket-metadata control are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import compare_cycles
import render_pass_contract
import run_scene_benchmark as benchmark

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline", type=Path)
parser.add_argument("controls", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.baseline = args.baseline.resolve()
args.controls = args.controls.resolve()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
assert not any(args.output.iterdir()), "use a fresh, empty output directory"
manifest_path = args.baseline / "barbershop/run-1/benchmark.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["schema"] == "psycles.scene-benchmark.v3"
assert manifest["status"] == "complete"
original_bundle = Path(manifest["scene"]["bundle"])
assert benchmark._bundle_matches_manifest(manifest, original_bundle)
assert benchmark._sha256(Path(manifest["scene"]["blend"])) == manifest["scene"]["sha256"]
control = json.loads((args.controls / "barbershop-control.json").read_text())
bundle = args.controls / "barbershop"
assert control["schema"] == "psycles.socket-metadata-control.v1"
assert Path(control["old_bundle"]).resolve() == original_bundle.resolve()
assert Path(control["control_bundle"]).resolve() == bundle.resolve()
assert benchmark._sha256(original_bundle / "scene.json") == control["old_scene"]["sha256"]
assert benchmark._sha256(bundle / "scene.json") == control["control_scene"]["sha256"]
assert benchmark._sha256(bundle / "geometry.bin") == control["old_geometry"]["sha256"]
files = ["build/bin/psycles_render_blender_scene", "build/libpsycles_core.so",
         "build/libpsycles_luisa_runtime.so", "build/bin/libluisa-coro.so",
         "build/bin/libluisa-xir.so", "build/bin/libluisa-backend-hip.so"]
identities = {name: benchmark._sha256(ROOT / name) for name in files}
revisions = {name: subprocess.check_output(
    ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    for name, path in (("root", ROOT), ("child", ROOT / "third_party/LuisaCompute"))}
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(("LUISA_DUMP_", "LUISA_HIP_DUMP_"))}
environment["PSYCLES_DISABLE_SHADER_CACHE"] = "1"
environment.pop("LUISA_CORO_WAVEFRONT_STATS", None)
command = list(manifest["commands"]["psycles_hip-wavefront-staged"]["command"])
command[1] = str(bundle)
command[2] = str(args.output / "barbershop.ppm")
assert command[4:7] == ["2048", "858", "256"]
assert command[13] == "256"
os.chdir(args.output)
render = benchmark._run_logged(command, args.output / "render.log",
                               environment=environment, echo_output=False)
frame = re.search(r"path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)",
                  render["output"])
assert frame is not None
actual = args.output / "barbershop.exr"
pixels, channels = compare_cycles._read_image(actual)
assert len(channels) == 46
invalid_lanes = int(np.count_nonzero(~np.isfinite(pixels)))
for name, expected in identities.items():
    assert benchmark._sha256(ROOT / name) == expected, name
print(json.dumps({
    "schema": "psycles.surface-audit-sdk-canary.v1",
    "scope": "single full-input SDK integration and finiteness gate; not a paired benchmark or parity proof",
    "revisions": revisions, "implementation_sha256": identities,
    "main_shader_cache": "disabled", "cwd": str(args.output),
    "manifest": {"path": str(manifest_path), "sha256": benchmark._sha256(manifest_path)},
    "socket_metadata_control": control,
    "render": benchmark._public_process_record(render),
    "timings": benchmark._parse_psycles_timings(render["output"]),
    "frame": dict(zip(("stages", "fields", "bytes"), map(int, frame.groups()))),
    "output_sha256": benchmark._sha256(actual),
    "pass_contract": render_pass_contract.inspect_image(actual),
    "channels": channels, "invalid_lanes": invalid_lanes,
    "actual_all_finite": invalid_lanes == 0}, indent=2))
raise SystemExit(0 if invalid_lanes == 0 else 2)

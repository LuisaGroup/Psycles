"""Capture full portal continuation films with the original Cycles HIP renderer.

Inputs are authored by create_cycles_ray_portal_scenes.py. This runner only
exports/renders those immutable .blend files and records process identities.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--blender", type=Path, required=True)
parser.add_argument("--case", action="append")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
capture = args.capture.resolve()
cases = ("portal-depth-world", "portal-depth-surface", "portal-depth-chain",
         "portal-signed-weight", "portal-transparent-mix", "portal-limit-zero",
         "portal-depth-nee", "portal-depth-shadow", "portal-default-position")
selected = args.case or cases
assert all(case in cases for case in selected), selected
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(("LUISA_DUMP_", "LUISA_HIP_DUMP_"))}


def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


processes, records = [], []
blender_before = identity(args.blender)
for name in selected:
    blend = capture / (name + ".blend")
    before = identity(blend)
    bundle = capture / (name + "-export")
    original = capture / (name + "-original.exr")
    assert not bundle.exists() and not original.exists(), name
    base = [args.blender, "--background", "--threads", "32", "--python-exit-code", "1", blend]
    commands = (
        ("export", base + ["--python", root / "tools/export_psycles_scene.py", "--", bundle]),
        ("original", base + ["--python", root / "tools/render_cycles_golden.py", "--", original,
                             "16", "16", "1", "0", "--cycles-device", "HIP",
                             "--device-name", "Radeon RX 9070 XT"]))
    for label, command in commands:
        command = list(map(str, command))
        log = capture / (name + "-" + label + ".log")
        with log.open("x") as stream:
            status = subprocess.run(command, cwd=capture, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT).returncode
        processes.append({"command": command, "cwd": str(capture),
                          "status": status, "log": identity(log)})
        if status:
            raise RuntimeError(f"{name}/{label} exited {status}: {log}")
    assert before == identity(blend)
    records.append({"name": name, "blend": before, "bundle": str(bundle),
                    "original": identity(original), "metadata": identity(original.with_suffix(".json"))})
assert blender_before == identity(args.blender)
print(json.dumps({"schema": "psycles.cycles-ray-portal-capture.v1", "records": records,
                  "blender": blender_before, "processes": processes}, indent=2))

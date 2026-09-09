"""Run original HIP and unchanged production HIP on the boundary witnesses.

No CPU evaluator or expected-pixel calculation. This is a correctness
counterexample, not a timing benchmark; all process outputs are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--blender", type=Path, required=True)
parser.add_argument("--surface-integration", action="store_true",
                    help="run object-holdout and portal admission/continuation controls")
args = parser.parse_args()
args.destination.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.root / "tools"))
import compare_cycles

def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}

records = []
environment = os.environ.copy()
for key in list(environment):
    if key.startswith(("LUISA_DUMP_", "LUISA_HIP_DUMP_")):
        del environment[key]
environment["PSYCLES_DISABLE_SHADER_CACHE"] = "1"

def run(command, label):
    log = args.destination / (label + ".log")
    with log.open("x") as stream:
        status = subprocess.run([str(c) for c in command], cwd=args.destination,
                                env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
    records.append({"command": [str(c) for c in command], "cwd": str(args.destination),
                    "status": status, "log": identity(log)})
    print(label, status, flush=True)
    assert status == 0, label

base = [args.blender, "--background", "--threads", "32", "--python-exit-code", "1"]
generator = "create_surface_scenes.py" if args.surface_integration else "create_boundary_scenes.py"
run(base + ["--python", Path(__file__).with_name(generator),
            "--", args.destination], "create")
renderer = args.root / "build/bin/psycles_render_blender_scene"
binary_before = identity(renderer)
observations = []
cases = ("ordinary-emission", "object-holdout", "transparent-control", "ray-portal") if args.surface_integration else ("boundary-1", "boundary-2", "boundary-2-budget-control")
for count, case in enumerate(cases, 1):
    blend = args.destination / f"{case}.blend"
    bundle = args.destination / f"{case}-export"
    before = identity(blend)
    run(base + [blend, "--python", args.root / "tools/export_psycles_scene.py", "--", bundle],
        f"export-{count}")
    original = args.destination / f"original-{count}.exr"
    actual = args.destination / f"actual-{count}.ppm"
    run(base + [blend, "--python", args.root / "tools/render_cycles_golden.py", "--",
                original, "16", "16", "1", "--cycles-device", "HIP",
                "--device-name", "Radeon RX 9070 XT"], f"original-{count}")
    # Same staged queue mode as the full-scene campaign; no path trace or
    # profiler, one sample, and no override of the authored bounce limits.
    run([renderer, bundle, actual, "hip", "16", "16", "1", "1", "-", "8", "8",
         "0", "0", "1", "-", "1", "0", "wavefront-staged", "32", "32768", "32",
         "1", "1", "1", "4", "2", "0", "0", "0", "0", "1", "1048576"],
        f"actual-{count}")
    observations.append({"scene": case, "blend": before, "outputs": {}})
    observed_rgb = {}
    for name, path in (("original", original), ("actual", actual.with_suffix(".exr"))):
        pixels, names = compare_cycles._read_image(path)
        rgb_names = [n for n in names if ".Combined." in n and n.rsplit(".", 1)[-1] in ("R", "G", "B")]
        # Actual EXR uses Combined.R while Blender prefixes the view layer.
        if not rgb_names:
            rgb_names = [n for n in names if n.startswith("Combined.") and n[-1] in "RGB"]
        assert len(rgb_names) == 3, names
        values = pixels[:, :, [names.index(n) for n in rgb_names]]
        observed_rgb[name] = values
        observations[-1]["outputs"][name] = {
            **identity(path), "channels": rgb_names, "all_finite": bool(np.isfinite(values).all()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": values.mean(axis=(0, 1)).tolist(), "nonzero_lanes": int(np.count_nonzero(values))}
    reference = observed_rgb["original"]
    evaluated = observed_rgb["actual"]
    observations[-1]["combined_rgb_max_absolute_error"] = float(np.max(np.abs(evaluated - reference)))
    observations[-1]["combined_rgb_parity"] = bool(np.isfinite(reference).all() and
        np.isfinite(evaluated).all() and np.allclose(evaluated, reference, atol=2e-6, rtol=2e-6))
    assert before == identity(blend)
assert binary_before == identity(renderer)
print(json.dumps({"schema": "psycles.surface-integration-witness.v1", "scope": "unchanged production vs original HIP; not a speed benchmark",
                  "binary": binary_before, "processes": records, "observations": observations}, indent=2))
raise SystemExit(0 if all(row["combined_rgb_parity"] for row in observations) else 2)

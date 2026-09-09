"""Extract exact original scene/geometry/film bytes for portal integration.

No transport or expected-pixel arithmetic is implemented here. Output maps
fixture filenames to text, for review and apply_patch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import compare_cycles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--additional-capture", type=Path, action="append", default=[])
args = parser.parse_args()

def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}

files, records = {}, []
captures = [(name, args.capture / (name + "-export"),
             args.capture / f"original-{index}.exr", args.capture / (name + ".blend"))
            for index, name in ((3, "transparent-control"), (4, "ray-portal"))]
for additional in args.additional_capture:
    extra = json.loads((additional / "capture.json").read_text())
    assert extra["schema"] == "psycles.cycles-ray-portal-capture.v1"
    assert len(extra["processes"]) == 2 * len(extra["records"])
    assert all(p["status"] == 0 for p in extra["processes"])
    for row in extra["records"]:
        original = Path(row["original"]["path"])
        blend = Path(row["blend"]["path"])
        assert identity(original) == row["original"] and identity(blend) == row["blend"]
        captures.append((row["name"], Path(row["bundle"]), original, blend))
for name, bundle, original, blend in captures:
    metadata = json.loads(original.with_suffix(".json").read_text())
    assert metadata["cycles_compute_device_type"] == "HIP" and metadata["cycles_device"] == "GPU"
    assert metadata["blender_build"]["build_hash"] == "9e2066aef7ef"
    assert metadata["width"] == metadata["height"] == 16 and metadata["samples"] == 1
    pixels, channels = compare_cycles._read_image(original)
    rgb = [channels.index("ViewLayer.Combined." + lane) for lane in "RGB"]
    assert pixels.shape[:2] == (16, 16) and np.isfinite(pixels[:, :, rgb]).all()
    files[name + "-scene.json"] = (bundle / "scene.json").read_text()
    geometry = (bundle / "geometry.bin").read_bytes()
    files[name + "-geometry.txt"] = "\n".join(
        " ".join(f"{value:02x}" for value in geometry[offset:offset + 16])
        for offset in range(0, len(geometry), 16)) + "\n"
    files[name + "-film.txt"] = "\n".join(
        f"{x} {y} " + " ".join(format(float(pixels[y, x, lane]), ".9g") for lane in rgb)
        for y in range(16) for x in range(16)) + "\n"
    records.append({"name": name, "scene": identity(bundle / "scene.json"),
                    "geometry": identity(bundle / "geometry.bin"), "geometry_bytes": len(geometry),
                    "original_image": identity(original), "metadata": identity(original.with_suffix(".json")),
                    "blend": identity(blend),
                    "original_settings": {key: metadata[key] for key in (
                        "blender_build", "cycles_device", "cycles_compute_device_type",
                        "samples", "width", "height", "effective_seed")}})
files["manifest.json"] = json.dumps({"schema": "psycles.cycles-ray-portal-render-fixture.v1",
    "scope": "exact exported inputs and complete original HIP film; no local renderer oracle",
    "records": records, "fixture_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in files.items()}, "tolerance": {"absolute": 2e-6, "relative": 2e-6}}, indent=2) + "\n"
print(json.dumps(files))

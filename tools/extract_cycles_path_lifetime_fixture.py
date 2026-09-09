"""Emit reviewable fixtures from the recorded original Cycles HIP renders.

This only copies exported inputs and reads original EXR channels. It never
calculates an expected transport, shader, sampler or pixel value. The JSON
output maps repository-relative fixture names to text for apply_patch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import compare_cycles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
args = parser.parse_args()

def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

files = {}
records = []
for index, name in enumerate(("boundary-1", "boundary-2", "boundary-2-budget-control"), 1):
    bundle = args.capture / (name + "-export")
    original = args.capture / f"original-{index}.exr"
    metadata_path = original.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["cycles_compute_device_type"] == "HIP"
    assert metadata["cycles_device"] == "GPU"
    assert metadata["width"] == metadata["height"] == 16
    assert metadata["samples"] == 1
    pixels, channels = compare_cycles._read_image(original)
    rgb = [channels.index("ViewLayer.Combined." + lane) for lane in "RGB"]
    assert pixels.shape[:2] == (16, 16) and np.isfinite(pixels[:, :, rgb]).all()
    files[name + "-scene.json"] = (bundle / "scene.json").read_text()
    geometry = (bundle / "geometry.bin").read_bytes()
    files[name + "-geometry.txt"] = "\n".join(
        " ".join(f"{byte:02x}" for byte in geometry[offset:offset + 16])
        for offset in range(0, len(geometry), 16)) + "\n"
    files[name + "-film.txt"] = "\n".join(
        f"{x} {y} " + " ".join(format(float(pixels[y, x, lane]), ".9g") for lane in rgb)
        for y in range(16) for x in range(16)) + "\n"
    records.append({"name": name, "scene": identity(bundle / "scene.json"),
                    "geometry": identity(bundle / "geometry.bin"), "geometry_bytes": len(geometry),
                    "original_image": identity(original), "metadata": identity(metadata_path),
                    "blend": identity(args.capture / (name + ".blend")),
                    "original_settings": {key: metadata[key] for key in (
                        "blender_build", "cycles_device", "cycles_compute_device_type",
                        "cycles_enabled_devices", "samples", "width", "height", "effective_seed")}})
manifest = {"schema": "psycles.cycles-path-lifetime-fixture.v1",
            "scope": "exact exported scenes/geometry; complete original HIP Combined RGB, no local oracle evaluator",
            "tolerance": {"absolute": 2e-6, "relative": 2e-6}, "records": records,
            "fixture_sha256": {name: hashlib.sha256(content.encode()).hexdigest()
                               for name, content in files.items()}}
files["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
print(json.dumps(files))

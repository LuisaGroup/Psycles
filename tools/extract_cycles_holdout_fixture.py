"""Extract original inputs, all 46 film channels, and unchanged local SVM words."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import compare_cycles
from extract_cycles_svm_shader import extract
from render_pass_contract import PASSES, PASS_ALIASES, validate_channels

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--word-source", type=Path, required=True,
                    help="Exact observer-enabled Cycles 5.2.1 source checkout")
args = parser.parse_args()
capture = json.loads((args.capture / "capture.json").read_text())
assert capture["schema"] == "psycles.cycles-holdout-capture.v1"
assert len(capture["records"]) == 12 and len(capture["processes"]) == 36
assert all(p["status"] == 0 for p in capture["processes"])


def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


source = args.word_source.resolve()
revision = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert revision == "cb168525138fecc792cc393f94afc39582b0103c"
source_files = (
    "intern/cycles/scene/svm.cpp", "intern/cycles/scene/shader_graph.cpp",
    "intern/cycles/scene/shader_nodes.cpp", "intern/cycles/scene/shader_nodes.h",
    "intern/cycles/kernel/svm/svm.h", "intern/cycles/kernel/svm/closure.h",
    "intern/cycles/kernel/integrator/surface_shader.h",
    "intern/cycles/kernel/integrator/shade_surface.h",
    "intern/cycles/scene/light.cpp",
)
word_source = {
    "path": str(source), "revision": revision,
    "identity_scope": "Source files observed during fixture extraction; dirty observer files retained.",
    "source_sha256": {name: identity(source / name)["sha256"] for name in source_files},
    "dirty_status": subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True).splitlines(),
}
for producer in capture["producers"]:
    assert identity(Path(producer["path"])) == producer
for process in capture["processes"]:
    assert identity(Path(process["log"]["path"])) == process["log"]


files, records = {}, []
for row in capture["records"]:
    name = row["name"]
    bundle = Path(row["bundle"])
    for key in ("blend", "original", "metadata", "words"):
        assert identity(Path(row[key]["path"])) == row[key]
    metadata = json.loads(Path(row["metadata"]["path"]).read_text())
    assert metadata["cycles_device"] == "GPU" and metadata["cycles_compute_device_type"] == "HIP"
    assert metadata["blender_build"]["build_hash"] == "9e2066aef7ef"
    assert (metadata["width"], metadata["height"], metadata["samples"]) == (16, 16, 1)
    observer_path = args.capture / (name + "-word-observer.json")
    observer = json.loads(observer_path.read_text())
    assert observer["blender_build"]["build_hash"] == revision[:12]
    assert observer["cycles_device"] == "GPU" and observer["cycles_compute_device_type"] == "HIP"
    assert (observer["width"], observer["height"], observer["samples"],
            observer["effective_seed"]) == (16, 16, 1, 0)
    pixels, channels = compare_cycles._read_image(Path(row["original"]["path"]))
    validate_channels(channels)
    ordered = ["ViewLayer." + p + "." + lane for p in PASSES
               for lane in ("RGBA" if p == "Combined" else "XYZ" if p == "Normal" else "RGB")]
    aliases = {alias: p for p, names in PASS_ALIASES.items() for alias in names}
    canonical = []
    for channel in channels:
        layer, p, lane = channel.rsplit(".", 2)
        canonical.append(layer + "." + aliases.get(p, p) + "." + lane)
    indices = [canonical.index(channel) for channel in ordered]
    assert pixels.shape == (16, 16, 46) and np.isfinite(pixels).all()
    scene_text = (bundle / "scene.json").read_text()
    scene = json.loads(scene_text)
    material, = [m for m in scene["materials"] if m["name"] == name]
    shader_index = material["cycles_sync"]["shader_index"]
    original_name, words = extract(Path(row["words"]["path"]), shader_index)
    assert original_name == name
    geometry = (bundle / "geometry.bin").read_bytes()
    files[name + "-scene.json"] = scene_text
    files[name + "-geometry.txt"] = str(len(geometry)) + "\n" + "\n".join(
        " ".join(f"{v:02x}" for v in geometry[start:start + 16])
        for start in range(0, len(geometry), 16)) + "\n"
    files[name + "-film.txt"] = "\n".join(
        f"{x} {y} " + " ".join(format(float(pixels[y, x, i]), ".9g") for i in indices)
        for y in range(16) for x in range(16)) + "\n"
    files[name + "-words.txt"] = str(len(words)) + "\n" + " ".join(f"{v:08x}" for v in words) + "\n"
    records.append({**row, "scene": identity(bundle / "scene.json"),
                    "geometry": identity(bundle / "geometry.bin"), "geometry_bytes": len(geometry),
                    "word_observer_metadata": identity(observer_path),
                    "word_observer_build": observer["blender_build"],
                    "shader_index": shader_index, "word_count": len(words),
                    "original_settings": {k: metadata[k] for k in (
                        "blender_build", "cycles_device", "cycles_compute_device_type",
                        "width", "height", "samples", "effective_seed")}})
files["manifest.json"] = json.dumps({
    "schema": "psycles.cycles-holdout-fixture.v1", "records": records,
    "capture": {"record": identity(args.capture / "capture.json"),
                "producers": capture["producers"], "processes": capture["processes"]},
    "word_source": word_source,
    "channels": ordered, "passes": list(PASSES),
    "word_scope": "Only ShaderJump relocation; all typed payload words copied from original Cycles.",
    "fixture_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()},
    "tolerance": {"absolute": 2e-6, "relative": 2e-6},
}, indent=2) + "\n"
print(json.dumps(files))

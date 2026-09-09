"""Capture original Holdout volume words and exported input; no shader evaluation on the host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from extract_cycles_svm_shader import extract

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--blender", type=Path, required=True)
parser.add_argument("--word-observer", type=Path, required=True)
parser.add_argument("--word-source", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
capture = args.capture.resolve()
capture.mkdir(parents=True, exist_ok=True)
assert not any(capture.iterdir()), "capture directory must be empty"
source = args.word_source.resolve()
revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert revision == "cb168525138fecc792cc393f94afc39582b0103c"


def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


producers = [identity(args.blender), identity(args.word_observer)]
blend = capture / "holdout-volume.blend"
bundle = capture / "export"
words = capture / "holdout-volume.svm52"
film = capture / "word-observer.exr"
base = ["--background", "--threads", "32", "--python-exit-code", "1"]
commands = (
    ("author", [args.blender, *base, "--python", root / "tools/create_cycles_holdout_volume_scene.py",
                "--", blend]),
    ("export", [args.blender, *base, blend, "--python", root / "tools/export_psycles_scene.py",
                "--", bundle]),
    ("word-observer", [args.word_observer, *base, blend, "--python", root / "tools/render_cycles_golden.py",
                       "--", film, "16", "16", "1", "0", "--cycles-device", "HIP",
                       "--device-name", "Radeon RX 9070 XT"]),
)
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(("LUISA_DUMP_", "LUISA_HIP_DUMP_", "PSYCLES_CYCLES_"))}
processes = []
blend_identity = None
for label, command in commands:
    command = list(map(str, command))
    run_environment = dict(environment)
    if label == "word-observer":
        run_environment["PSYCLES_CYCLES_SVM_DUMP"] = str(words)
    log = capture / (label + ".log")
    with log.open("x") as stream:
        status = subprocess.run(command, cwd=capture, env=run_environment,
                                stdout=stream, stderr=subprocess.STDOUT).returncode
    processes.append({"label": label, "command": command, "status": status,
                      "cwd": str(capture), "log": identity(log)})
    if status:
        raise RuntimeError(f"{label}: exit {status}: {log}")
    if blend_identity is None:
        blend_identity = identity(blend)
    assert blend_identity == identity(blend)
    print("Captured " + label, flush=True, file=sys.stderr)
assert producers == [identity(args.blender), identity(args.word_observer)]
metadata = json.loads(film.with_suffix(".json").read_text())
assert metadata["blender_build"]["build_hash"] == revision[:12]
assert metadata["cycles_device"] == "GPU" and metadata["cycles_compute_device_type"] == "HIP"
scene_text = (bundle / "scene.json").read_text()
scene = json.loads(scene_text)
geometry = (bundle / "geometry.bin").read_bytes()
names = ("holdout-only-volume", "holdout-emission-volume", "holdout-shared-domains")
records, images = [], [str(len(names))]
for name in names:
    material, = [material for material in scene["materials"] if material["name"] == name]
    shader = material["cycles_sync"]["shader_index"]
    original_name, image = extract(words, shader)
    assert original_name == name
    records.append({"name": name, "shader_index": shader, "word_count": len(image)})
    images.append(json.dumps(name) + f" {shader} {len(image)}")
    images.append(" ".join(f"{word:08x}" for word in image))
files = {
    "scene.json": scene_text,
    "geometry.txt": str(len(geometry)) + "\n" + "\n".join(
        " ".join(f"{byte:02x}" for byte in geometry[start:start + 16])
        for start in range(0, len(geometry), 16)) + "\n",
    "words.txt": "\n".join(images) + "\n",
}
source_files = ("intern/cycles/scene/svm.cpp", "intern/cycles/scene/shader_graph.cpp",
                "intern/cycles/scene/shader_nodes.cpp", "intern/cycles/scene/shader_nodes.h",
                "intern/cycles/kernel/svm/svm.h", "intern/cycles/kernel/svm/closure.h")
files["manifest.json"] = json.dumps({
    "schema": "psycles.cycles-holdout-volume-fixture.v1", "records": records,
    "producers": producers, "processes": processes,
    "source": {"path": str(source), "revision": revision,
               "file_sha256": {name: identity(source / name)["sha256"] for name in source_files},
               "dirty_status": subprocess.check_output(
                   ["git", "-C", str(source), "status", "--porcelain"], text=True).splitlines()},
    "blend": blend_identity, "scene": identity(bundle / "scene.json"),
    "geometry": identity(bundle / "geometry.bin"), "words": identity(words),
    "word_observer_metadata": identity(film.with_suffix(".json")),
    "word_observer_build": metadata["blender_build"],
    "scope": "Compiler word oracle only: three ShaderJump targets relocated; every other word unchanged.",
    "fixture_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()},
}, indent=2) + "\n"
print(json.dumps(files))

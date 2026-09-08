"""Retain raw exported graphs and exact original-Cycles words for host tests."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from extract_cycles_svm_shader import extract


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("export", type=Path)
parser.add_argument("dump", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--stem", default="cycles_hidden_socket")
parser.add_argument("--count", type=int, default=15)
parser.add_argument("--replace", action="store_true", help="regenerate these two owned fixture files")
args = parser.parse_args()
scene = json.loads((args.export / "scene.json").read_text())
# Shader compilation needs no geometry, images, or reference shader evaluator.
# The material/group payloads (including hidden defaults) are copied unchanged.
for key in ("geometries", "curve_geometries", "instances", "lights"):
    scene[key] = []
scene["world"] = scene["world_environment"] = None
assert len(scene["materials"]) == args.count and not scene["images"]
records = []
for material in scene["materials"]:
    index = material["cycles_sync"]["shader_index"]
    name, words = extract(args.dump, index)
    assert name == material["name"]
    records.append((index, name, words))
args.destination.mkdir(parents=True, exist_ok=True)
mode = "w" if args.replace else "x"
with (args.destination / f"{args.stem}_scene.json").open(mode) as output:
    json.dump(scene, output, separators=(",", ":"), sort_keys=True)
    output.write("\n")
with (args.destination / f"{args.stem}_words.txt").open(mode) as output:
    output.write(f"{len(records)}\n")
    for index, name, words in records:
        output.write(f'{index} {json.dumps(name)} {len(words)}\n')
        output.write(" ".join(f"{word:08x}" for word in words) + "\n")

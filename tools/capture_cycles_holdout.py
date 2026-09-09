"""Capture immutable input, original HIP films and original SVM word images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--blender", type=Path, required=True)
parser.add_argument("--word-observer", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
capture = args.capture.resolve()
blends = sorted(capture.glob("*.blend"))
assert len(blends) == 12
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(("LUISA_DUMP_", "LUISA_HIP_DUMP_", "PSYCLES_CYCLES_"))}


def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


producers = [identity(args.blender), identity(args.word_observer)]
records, processes = [], []
for blend in blends:
    name = blend.stem
    before = identity(blend)
    bundle = capture / (name + "-export")
    original = capture / (name + "-original.exr")
    words = capture / (name + ".svm52")
    base = ["--background", "--threads", "32", "--python-exit-code", "1", blend]
    golden = ["--python", root / "tools/render_cycles_golden.py", "--", original,
              "16", "16", "1", "0", "--cycles-device", "HIP",
              "--device-name", "Radeon RX 9070 XT"]
    observer_golden = list(golden)
    observer_golden[3] = capture / (name + "-word-observer.exr")
    commands = (
        ("export", [args.blender, *base, "--python", root / "tools/export_psycles_scene.py", "--", bundle]),
        ("original", [args.blender, *base, *golden]),
        ("word-observer", [args.word_observer, *base, *observer_golden]),
    )
    for label, command in commands:
        command = list(map(str, command))
        run_environment = dict(environment)
        if label == "word-observer":
            run_environment["PSYCLES_CYCLES_SVM_DUMP"] = str(words)
        log = capture / (name + "-" + label + ".log")
        with log.open("x") as stream:
            status = subprocess.run(command, cwd=capture, env=run_environment,
                                    stdout=stream, stderr=subprocess.STDOUT).returncode
        processes.append({"command": command, "status": status, "cwd": str(capture),
                          "svm_dump": str(words) if label == "word-observer" else None,
                          "log": identity(log)})
        if status:
            raise RuntimeError(f"{name}/{label}: exit {status}: {log}")
    assert before == identity(blend)
    records.append({"name": name, "blend": before, "bundle": str(bundle),
                    "original": identity(original), "metadata": identity(original.with_suffix(".json")),
                    "words": identity(words)})
    print("Captured " + name, flush=True, file=sys.stderr)
assert producers == [identity(args.blender), identity(args.word_observer)]
print(json.dumps({"schema": "psycles.cycles-holdout-capture.v1", "records": records,
                  "producers": producers, "processes": processes}, indent=2))

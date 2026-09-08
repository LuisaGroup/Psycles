"""Greedily reduce original material links while preserving a typed witness.

Every candidate is compiled by original Cycles HIP and the production Psycles
scene compiler. No expected stream or shader evaluation is synthesized here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
REPORTS = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("material")
    parser.add_argument("output", type=Path)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--dump-scene", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--nested", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    history = []
    current, material = args.source, args.material
    previous_measure = None
    for step in range(args.rounds):
        directory = args.output / f"round-{step:02d}"
        directory.mkdir()

        def run(label, command, env=None, binary_output=None):
            with (directory / (label + ".log")).open("w") as log:
                print("+", *map(str, command), file=log, flush=True)
                if binary_output is not None:
                    with binary_output.open("wb") as output:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=log, check=True)
                else:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        blend = directory / "candidates.blend"
        run("create", [args.blender, "--background", "--python-exit-code", "1", "--python",
            ROOT / "tools/create_cycles_link_reduction_batch.py", "--", current, material, blend,
            *(["--nested"] if args.nested else [])])
        original = directory / "cycles.svm52"
        environment = os.environ.copy()
        environment["PSYCLES_CYCLES_SVM_DUMP"] = str(original)
        run("cycles", [args.blender, blend, "--background", "--python-exit-code", "1", "--python",
            ROOT / "tools/render_cycles_golden.py", "--", directory / "cycles.exr",
            "16", "16", "1", "0", "--cycles-device", "HIP"], env=environment)
        run("export", [args.blender, blend, "--background", "--python-exit-code", "1", "--python",
            ROOT / "tools/export_psycles_scene.py", "--", directory / "export"])
        actual = directory / "psycles.svm52"
        run("psycles", [args.dump_scene, directory / "export"], binary_output=actual)
        raw = directory / "words.json"
        run("words", [sys.executable, REPORTS / "lamp-routing-and-surface/audit_scene_words.py",
            original, actual], binary_output=raw)
        typed = directory / "typed.json"
        run("typed", [sys.executable, REPORTS / "svm-math-expansion/audit_typed_words.py",
            args.layouts, raw, original, actual], binary_output=typed)
        words = json.loads(raw.read_text())
        classified = json.loads(typed.read_text())
        candidates = json.loads(blend.with_suffix(".json").read_text())["candidates"]
        witnesses = []
        for row in classified["shaders"]:
            difference = row["first_node_layout_difference"]
            if not difference or difference["cycles"][1] != "NODE_MAPPING":
                continue
            if difference["psycles"][1] not in ("NODE_TEX_IMAGE", "NODE_TEX_CHECKER"):
                continue
            raw_row = next(r for r in words["shaders"] if r["name"] == row["name"])
            if raw_row["size_difference"] != 0 or row["node_counts"]["cycles"] != row["node_counts"]["psycles"]:
                continue
            candidate = next(c for c in candidates if c["name"] == row["name"])
            witnesses.append({**candidate, "words": raw_row["cycles_words"], "witness": difference})
        assert any(row["name"] == "candidate-000" for row in witnesses), "baseline witness disappeared"
        selected = min(witnesses, key=lambda row: (row["nodes"], row["links"], row["words"]))
        measure = selected["nodes"], selected["links"]
        record = {"round": step, "selected": selected, "witnesses": witnesses}
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(step, selected["name"], "nodes/links", measure, "words", selected["words"], flush=True)
        if selected["name"] == "candidate-000":
            print("No single link deletion preserves the witness in the selected scope; not a global minimality proof.", flush=True)
            break
        assert previous_measure is None or measure < previous_measure
        previous_measure = measure
        current, material = blend, selected["name"]


if __name__ == "__main__":
    main()

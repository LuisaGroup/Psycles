"""Isolate exported socket flags while retaining byte-identical geometry/images.

Blender's repeated evaluated geometry export is not bit deterministic. Verify
that fresh metadata differs ONLY by hide_value and exporter identity, then
combine that fresh JSON with the previous immutable geometry/texture snapshot.
This does not evaluate a shader, edit defaults, or invent flags from a profile.
"""
import argparse
import json
from pathlib import Path
import shutil

from audit_surface import source


def without_new_metadata(value):
    if isinstance(value, dict):
        return {k: without_new_metadata(v) for k, v in value.items()
                if k not in {"hide_value", "exporter"}}
    if isinstance(value, list):
        return [without_new_metadata(v) for v in value]
    return value


def prepare(old, fresh, output):
    original = json.loads((old / "scene.json").read_text())
    updated = json.loads((fresh / "scene.json").read_text())
    assert without_new_metadata(original) == without_new_metadata(updated)
    assert output.resolve() not in {old.resolve(), fresh.resolve()}
    output.mkdir(parents=True, exist_ok=False)
    # Copy, not mutable hard links: future consumers cannot corrupt the
    # immutable earlier reference/control input through this new bundle.
    for path in old.iterdir():
        if path.name == "scene.json":
            shutil.copyfile(fresh / path.name, output / path.name)
        elif path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copyfile(path, output / path.name)
    record = {
        "schema": "psycles.socket-metadata-control.v1",
        "scope": "only new source socket flags and exporter identity; old geometry and image bytes retained",
        "old_bundle": str(old.resolve()), "fresh_export": str(fresh.resolve()),
        "control_bundle": str(output.resolve()),
        "old_scene": source(old / "scene.json"),
        "fresh_scene": source(fresh / "scene.json"),
        "control_scene": source(output / "scene.json"),
        "old_geometry": source(old / "geometry.bin"),
        "fresh_geometry": source(fresh / "geometry.bin"),
        "control_geometry": source(output / "geometry.bin"),
    }
    assert record["old_geometry"]["sha256"] == record["control_geometry"]["sha256"]
    assert record["fresh_scene"]["sha256"] == record["control_scene"]["sha256"]
    (output.parent / (output.name + "-control.json")).write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.old, args.fresh, args.output), indent=2))

"""Inventory source boundaries and serialized programs, not a shader oracle.

No shader evaluation, path simulation, word normalization, or timing inference.
The opcode map is mechanical presence evidence; the accompanying review owns
semantic judgments and must not turn a nonempty handler into a parity claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys


def provenance(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def catalog(path: Path) -> list[str]:
    result = []
    for kind, name in re.findall(
            r"^SHADER_NODE_TYPE(_DERIVATIVE)?\((NODE_\w+)\)",
            path.read_text(), re.MULTILINE):
        result.append(name)
        if kind:
            result.append(name + "_DERIVATIVE")
    assert len(result) == len(set(result))
    return result


def dispatches(path: Path, macro: str) -> dict:
    source = path.read_text()
    entries = list(re.finditer(r"\b" + macro + r"\((NODE_\w+)\)", source))
    result = {}
    for i, match in enumerate(entries):
        end = entries[i + 1].start() if i + 1 < len(entries) else len(source)
        result[match[1]] = {
            "path": str(path), "line": source.count("\n", 0, match.start()) + 1,
            "calls": sorted(set(re.findall(
                r"\b(?:svm_)?node_\w+(?=\s*(?:<|\())", source[match.end():end])))
        }
    assert len(result) == len(entries)
    return result


def source_mentions(path: Path, token: str) -> list[dict]:
    return [{"path": str(path), "line": i, "text": line.strip()}
            for i, line in enumerate(path.read_text().splitlines(), 1)
            if re.search(r"\b" + re.escape(token) + r"\b", line)]


def opcode_inventory(root: Path, cycles: Path) -> dict:
    actual_catalog = root / "include/psycles/compiler/cycles_svm_node_types_template.h"
    original_catalog = cycles / "kernel/svm/node_types_template.h"
    names = catalog(actual_catalog)
    assert names == catalog(original_catalog), "native opcode enumeration drift"
    actual = dispatches(root / "src/luisa/cycles_svm.cpp", "PSYCLES_SVM_CASE")
    original = dispatches(cycles / "kernel/svm/svm.h", "SVM_CASE")
    sentinels = {"NODE_NONE", "NODE_PAD1"}
    assert not (set(actual) - set(names))
    actual_sources = sorted((root / "src/luisa").glob("cycles_svm*.cpp"))
    original_sources = sorted((cycles / "kernel/svm").glob("*.h"))
    # These are only navigation links to names mentioned by the dispatch.
    # Calls and definitions are deliberately not conflated with test coverage.
    def locations(calls, paths):
        return {call: [hit for path in paths
                       for hit in source_mentions(path, call)] for call in calls}
    rows = []
    for tag, name in enumerate(names):
        row = {"tag": tag, "name": name,
               "presence": "sentinel" if name in sentinels else
                           "implemented" if name in actual else "missing",
               "actual_dispatch": actual.get(name),
               "original_dispatch": original.get(name)}
        if name in actual:
            row["actual_handler_mentions"] = locations(actual[name]["calls"], actual_sources)
        if name in original:
            row["original_handler_mentions"] = locations(original[name]["calls"], original_sources)
        rows.append(row)
    return {"scope": "catalog/case presence and navigation only; not handler semantics or test coverage",
            "summary": dict(Counter(row["presence"] for row in rows)),
            "sources": [provenance(p) for p in (actual_catalog, original_catalog,
                                                root / "src/luisa/cycles_svm.cpp",
                                                cycles / "kernel/svm/svm.h")],
            "rows": rows}


def source_projections(root: Path, cycles: Path) -> dict:
    surface = root / "src/luisa/path_tracer_cycles_svm_surface.cpp"
    closure = root / "include/psycles/luisa/cycles_closure.h"
    body = surface.read_text().split("UInt events_from_label(", 1)[1].split(
        "UInt bssrdf_method(", 1)[0]
    mappings = re.findall(r"append\(closure::(label_\w+),\s*contract::(event_\w+)\)", body)
    native_labels = set(re.findall(r"\blabel_\w+", closure.read_text()))
    mapped = {label for label, _ in mappings}
    symbols = [
        (root / "src/luisa/path_tracer_cycles_svm_surface.cpp", "sample_nonempty_impl"),
        (root / "src/luisa/path_kernel_surface_scatter.cpp", "label_from_events"),
        (root / "include/psycles/luisa/cycles_svm.h", "transparent_roughness_squared_threshold"),
        (root / "src/luisa/path_tracer_cycles_svm_kernel_globals.h", "transparent_roughness_squared_threshold"),
        (root / "src/luisa/path_kernel_pipeline.cpp", "max_path_steps"),
        (root / "src/luisa/cycles_integrator_limits.h", "cycles_path_step_limit"),
        (cycles / "kernel/integrator/path_state.h", "VOLUME_BOUNDS_MAX"),
        (cycles / "kernel/integrator/shade_surface.h", "integrate_surface_ray_portal"),
        (cycles / "kernel/integrator/shade_surface.h", "integrate_surface_holdout"),
        (root / "src/luisa/path_tracer_cycles_svm_object.cpp", "use_holdout"),
        (root / "src/luisa/path_tracer_cycles_svm_object.cpp", "cycles_uses_light_linking"),
    ]
    return {
        "event_mapping": mappings,
        "declared_but_unmapped_labels": sorted(native_labels - mapped - {"label_from_events"}),
        "citations": [{"searched_file": provenance(path), "token": token,
                       "mentions": source_mentions(path, token)} for path, token in symbols]}


def scene_metadata(path: Path) -> dict:
    scene = json.loads(path.read_text())
    instances, lights = scene["instances"], scene["lights"]
    fields = ["use_holdout", "is_shadow_catcher", "is_caustics_caster", "is_caustics_receiver"]
    named_flags = {field: {kind: [item["name"] for item in items if item.get(field, False)]
                          for kind, items in [("instances", instances), ("lights", lights)]}
                   for field in fields}
    # Exported tree census is conservative source presence, not reachability.
    source_nodes = Counter()
    def walk(value):
        if isinstance(value, dict):
            if "bl_idname" in value:
                source_nodes[value["bl_idname"]] += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for collection in (scene["materials"], scene["node_groups"], scene["world"]):
        walk(collection)
    return {"source": provenance(path), "cycles_sync": scene["cycles_sync"],
            "render_cycles": scene["render"]["cycles"],
            "film_transparent": scene["render"]["transparent"],
            "pass_alpha_threshold": scene["render"].get("pass_alpha_threshold"),
            "object_flags": named_flags,
            "images_by_source": dict(Counter(i["source"] for i in scene["images"])),
            "exported_source_node_presence_not_reachability": dict(sorted(source_nodes.items()))}


def program_census(root: Path, layouts: Path, dump: Path) -> dict:
    reports = root / "docs/validation/2026-09-08"
    sys.path.insert(0, str(reports / "svm-math-expansion"))
    from audit_typed_words import decode, read_dump
    metadata = json.loads(layouts.read_text())
    names, images = read_dump(dump)
    domains = {name: Counter() for name in ("surface", "volume", "displacement")}
    closure_types = Counter()
    texture_modes = Counter()
    branch_counts = Counter()
    rows = []
    for index, words in images.items():
        nodes, _ = decode(words, metadata)
        local_domains = {name: Counter() for name in domains}
        for pc, node, size in nodes:
            if pc == 0:
                continue
            domain = "surface" if pc < words[2] else "volume" if pc < words[3] else "displacement"
            domains[domain][node] += 1
            local_domains[domain][node] += 1
            if domain != "surface":
                continue
            if node == "NODE_CLOSURE_BSDF":
                kind = metadata["closures"][str(words[pc + 1])]
                closure_types[kind] += 1
            elif node == "NODE_TEX_NOISE":
                texture_modes[f"noise:dimensions={words[pc + 1]},type={words[pc + 2]}"] += 1
            elif node == "NODE_TEX_VORONOI":
                texture_modes[f"voronoi:dimensions={words[pc + 1]},feature={words[pc + 2]},metric={words[pc + 3]}"] += 1
            if node in ("NODE_JUMP_IF_ZERO", "NODE_JUMP_IF_ONE", "NODE_ENTER_BUMP_EVAL",
                        "NODE_LEAVE_BUMP_EVAL", "NODE_CLOSURE_SET_NORMAL"):
                branch_counts[node] += 1
        rows.append({"index": index, "name": names[index], "words": len(words),
                     "jump": words[:4], "nodes_by_domain": local_domains})
    return {"scope": "serialized typed nodes by ShaderJump interval, including END-only dense holes; counts are not dynamic execution frequencies",
            "source": provenance(dump), "layout_source": provenance(layouts),
            "decoder_source": provenance(reports / "svm-math-expansion/audit_typed_words.py"),
            "domain_totals": domains, "surface_closure_payloads": closure_types,
            "surface_procedural_modes": texture_modes,
            "surface_control_nodes": branch_counts, "shaders": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("cycles", type=Path, help="original intern/cycles directory")
    parser.add_argument("controls", type=Path)
    parser.add_argument("layouts", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    result = {"schema": "psycles.surface-structural-inventory.v1",
              "scope": "source/serialized-program audit, no new rendering or speedup claim",
              "root_revision": subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
              "opcodes": opcode_inventory(args.root, args.cycles),
              "projections": source_projections(args.root, args.cycles), "scenes": {}}
    for name in ("barbershop", "monk", "monster", "classroom"):
        result["scenes"][name] = {
            "metadata": scene_metadata(args.controls / name / "scene.json"),
            "program": program_census(args.root, args.layouts, args.evidence / (name + ".svm52")),
            "compilation_log": (args.evidence / (name + "-compile.log")).read_text()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

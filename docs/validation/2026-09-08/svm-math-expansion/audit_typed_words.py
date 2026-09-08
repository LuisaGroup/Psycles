"""Classify raw SVM differences by declared payload; never normalize words."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys

REPORTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORTS / "lamp-routing-and-surface"))
from audit_scene_words import read_dump
from audit_surface import source

RESOURCE_FIELDS = {
    "SVMNodeTexImage.id", "SVMNodeTexImageBox.id", "SVMNodeTexEnvironment.id",
    "SVMNodeTexSkyNishitaData.texture_id", "SVMNodeAttr.attr",
    "SVMNodeNormalMap.attr", "SVMNodeNormalMap.attr_sign", "SVMNodeTangent.attr",
}


def decode(words, metadata):
    assert len(words) >= 5
    assert metadata["nodes"].get(str(words[0]), {}).get("name") == "NODE_SHADER_JUMP"
    nodes, annotations, targets = [], {}, []
    pc = 0
    layouts = metadata["structs"]

    def payload(name, begin):
        layout = layouts[name]
        end = begin + layout["words"]
        assert end <= len(words), (name, begin, end, len(words))
        for field, (offset, size) in layout["fields"].items():
            assert 0 <= offset < offset + size <= layout["words"] * 4
            for byte in range(size):
                annotations[begin * 4 + offset + byte] = name + "." + field
        return end

    def integer(name, field, begin):
        offset, size = layouts[name]["fields"][field]
        assert offset % 4 == 0 and size == 4
        return words[begin + offset // 4]

    while pc < len(words):
        node = metadata["nodes"].get(str(words[pc]))
        assert node is not None, ("unknown opcode", pc, words[pc])
        name, typed = node["name"], node["payload"]
        for byte in range(4):
            annotations[pc * 4 + byte] = "opcode"
        end = payload(typed, pc + 1) if typed else pc + 1
        if name == "NODE_CLOSURE_BSDF":
            closure = metadata["closures"].get(str(integer(typed, "closure_type", pc + 1)))
            assert closure is not None, ("unknown closure", pc)
            end = payload(closure, end)
        elif name in ("NODE_RGB_RAMP", "NODE_CURVES", "NODE_FLOAT_CURVE"):
            count = integer(typed, "table_size", pc + 1)
            size = count * (1 if name == "NODE_FLOAT_CURVE" else 4)
            assert 0 < count and end + size <= len(words)
            for byte in range(end * 4, (end + size) * 4):
                annotations[byte] = typed + ".table_float"
            end += size
        elif name in ("NODE_TEX_COORD", "NODE_TEX_COORD_DERIVATIVE"):
            offset, size = layouts[typed]["fields"]["texco_type"]
            packed = struct.pack("<" + "I" * (end - pc - 1), *words[pc + 1:end])
            kind = int.from_bytes(packed[offset:offset + size], "little")
            if kind == metadata["constants"]["NODE_TEXCO_OBJECT_WITH_TRANSFORM"]:
                size = metadata["transform_words"]
                for byte in range(end * 4, (end + size) * 4):
                    annotations[byte] = "PackedTransform"
                end += size
        elif name == "NODE_TEX_SKY":
            sky = integer(typed, "sky_type", pc + 1)
            constants = metadata["constants"]
            assert sky in {constants[k] for k in constants if k.startswith("NODE_SKY_")}
            extra = "SVMNodeTexSkyPreethamData" if sky in (
                constants["NODE_SKY_PREETHAM"], constants["NODE_SKY_HOSEK"]) else "SVMNodeTexSkyNishitaData"
            end = payload(extra, end)
        elif name == "NODE_RAYCAST":
            raise AssertionError("Raycast variable attribute data not audited here")
        if name in ("NODE_JUMP_IF_ZERO", "NODE_JUMP_IF_ONE"):
            delta = integer(typed, "jump_offset", pc + 1)
            delta = delta if delta < 2 ** 31 else delta - 2 ** 32
            targets.append(end + delta)
        elif name == "NODE_SHADER_JUMP":
            assert pc == 0
            targets.extend(words[1:4])
        assert pc < end <= len(words)
        nodes.append((pc, name, end - pc))
        pc = end
    boundaries = {node[0] for node in nodes}
    assert all(target in boundaries for target in targets), ("invalid target", targets)
    for target in words[1:4]:
        assert target in boundaries
    return nodes, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("layouts", "word_audit", "cycles", "psycles"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.layouts.read_text())
    audit = json.loads(args.word_audit.read_text())
    _, originals = read_dump(args.cycles)
    _, actuals = read_dump(args.psycles)
    rows, classes = [], Counter()
    for shader in audit["shaders"]:
        original, actual = originals[shader["cycles_index"]], actuals[shader["index"]]
        c_nodes, c_fields = decode(original, metadata)
        p_nodes, p_fields = decode(actual, metadata)
        same_layout = c_nodes == p_nodes and c_fields == p_fields
        differences = []
        fields = Counter()
        for offset, (c, p) in enumerate(zip(original, actual)):
            if c == p:
                continue
            changed_fields = sorted({c_fields.get(offset * 4 + byte, "implicit padding")
                for byte in range(4) if ((c ^ p) >> (byte * 8)) & 255})
            fields.update(changed_fields)
            if len(differences) < 12:
                differences.append({"word": offset, "cycles": f"{c:08x}",
                                    "psycles": f"{p:08x}", "original_fields": changed_fields})
        if original == actual:
            classification = "raw_equal"
        elif same_layout and set(fields) <= RESOURCE_FIELDS:
            classification = "resource_fields_only_unresolved_bindings"
        elif same_layout:
            classification = "other_typed_payload"
        else:
            classification = "different_node_layout"
        classes[classification] += 1
        first_layout = next(({"cycles": c, "psycles": p} for c, p in zip(c_nodes, p_nodes) if c != p), None)
        if first_layout is None and len(c_nodes) != len(p_nodes):
            index = min(len(c_nodes), len(p_nodes))
            first_layout = {"cycles": c_nodes[index] if index < len(c_nodes) else None,
                            "psycles": p_nodes[index] if index < len(p_nodes) else None}
        rows.append({"name": shader["name"], "index": shader["index"],
            "classification": classification, "same_node_layout": same_layout,
            "node_counts": {"cycles": len(c_nodes), "psycles": len(p_nodes)},
            "first_node_layout_difference": first_layout,
            "differing_common_words": shader["differing_common_words"],
            "original_field_counts": dict(fields), "first_differences": differences})
    print(json.dumps({"schema": "psycles.typed-svm-word-audit.v1",
        "scope": "sequential typed decoding of all original and actual words, including variable tables and closure payloads; validate jump destinations; no resource-ID normalization or presumed binding equivalence; field annotations are comparative only for same-layout images",
        "summary": dict(classes), "shaders": rows,
        "sources": [source(path) for path in vars(args).values()]}, indent=2))


if __name__ == "__main__":
    main()

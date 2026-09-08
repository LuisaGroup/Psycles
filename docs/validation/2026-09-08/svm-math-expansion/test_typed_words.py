"""Check the diagnostic parser against original input and corruptions only.

This does not evaluate a shader or supply Psycles' expected compiler words.
"""
import argparse
import copy
import json
from pathlib import Path

from audit_typed_words import decode, read_dump
from make_word_layouts import checked_structs


def rejected(operation):
    try:
        operation()
    except AssertionError:
        return
    raise AssertionError("corrupt input was admitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layouts", type=Path)
    parser.add_argument("original", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.layouts.read_text())
    _, images = read_dump(args.original)
    decoded = {index: decode(words, metadata) for index, words in images.items()}
    tags = {entry["name"]: int(tag) for tag, entry in metadata["nodes"].items()}
    checks = ["all original shader images decode"]

    def original_with(opcode):
        for index, (nodes, _) in decoded.items():
            for pc, name, _ in nodes:
                if name == opcode:
                    return list(images[index]), pc
        raise AssertionError("original input lacks " + opcode)

    def field_word(opcode, field, pc):
        typed = metadata["nodes"][str(tags[opcode])]["payload"]
        offset, size = metadata["structs"][typed]["fields"][field]
        assert offset % 4 == 0 and size == 4
        return pc + 1 + offset // 4

    original, pc = original_with("NODE_CLOSURE_BSDF")
    corrupt = original.copy()
    corrupt[0] = tags["NODE_END"]
    rejected(lambda: decode(corrupt, metadata))
    checks.append("missing ShaderJump prefix rejected")
    corrupt = original.copy()
    corrupt[1] = pc + 1
    rejected(lambda: decode(corrupt, metadata))
    checks.append("jump into typed payload rejected")
    corrupt = original.copy()
    corrupt[pc] = 0xffffffff
    rejected(lambda: decode(corrupt, metadata))
    checks.append("unknown opcode rejected")
    corrupt = original.copy()
    corrupt[field_word("NODE_CLOSURE_BSDF", "closure_type", pc)] = 0xffffffff
    rejected(lambda: decode(corrupt, metadata))
    checks.append("unknown closure rejected")
    rejected(lambda: decode(original[:pc + 2], metadata))
    checks.append("truncated typed payload rejected")

    ramp, pc = original_with("NODE_RGB_RAMP")
    corrupt = ramp.copy()
    corrupt[field_word("NODE_RGB_RAMP", "table_size", pc)] = 0xffffffff
    rejected(lambda: decode(corrupt, metadata))
    checks.append("truncated variable table rejected")
    # A numeric value equal to an unknown opcode is still just table data.
    typed = metadata["nodes"][str(tags["NODE_RGB_RAMP"])]["payload"]
    ramp_nodes = decode(ramp, metadata)[0]
    ramp[pc + 1 + metadata["structs"][typed]["words"]] = 0xffffffff
    assert decode(ramp, metadata)[0] == ramp_nodes
    checks.append("table data is not dispatched as an opcode")

    # Synthetic malformed diagnostic input, not an expected shader stream.
    typed = metadata["nodes"][str(tags["NODE_RAYCAST"])]["payload"]
    raycast = [tags["NODE_SHADER_JUMP"], 4, 4, 4, tags["NODE_RAYCAST"]]
    raycast += [0] * metadata["structs"][typed]["words"] + [tags["NODE_END"]]
    rejected(lambda: decode(raycast, metadata))
    checks.append("unaudited variable Raycast payload fails closed")
    bad_layout = copy.deepcopy(metadata)
    bad_layout["structs"]["SVMNodeClosureBsdf"]["fields"]["closure_type"] = [100000, 4]
    rejected(lambda: decode(original, bad_layout))
    checks.append("out-of-bounds field declaration rejected")
    declaration = "struct SVMNodeTest { unsigned value; };"
    assert checked_structs(declaration, declaration)
    rejected(lambda: checked_structs(declaration, declaration.replace("unsigned", "float")))
    checks.append("different mirrored typed declaration rejected")
    print(json.dumps({"original_images": len(images), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

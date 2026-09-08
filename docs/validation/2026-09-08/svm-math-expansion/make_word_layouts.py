"""Generate a sizeof/offsetof diagnostic from the original typed declarations.

No shader evaluation: this emits layout metadata for decoding original SVM
word images. Refuse a mirrored payload declaration different from Cycles.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]


def structs(text):
    return dict(re.findall(r"struct (SVMNode\w+)\s*\{(.*?)\};", text, re.S))


def checked_structs(original, mirrored):
    declarations = structs(original)
    assert declarations and declarations == structs(mirrored), "typed declarations differ from original Cycles"
    return declarations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cycles", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    original = (args.cycles / "intern/cycles/kernel/svm/node_types.h").read_text()
    mirrored = (ROOT / "include/psycles/compiler/cycles_svm_node_types.h").read_text()
    declarations = checked_structs(original, mirrored)
    bindings = {}
    for comment, payload in re.findall(r"/\*((?:(?!\*/).)*)\*/\s*struct (SVMNode\w+)", original, re.S):
        for name in re.findall(r"\bNODE_[A-Z_]+\b", comment):
            assert name not in bindings, name
            bindings[name] = payload
    # Original svm.h cases without a typed payload.
    for name in ("NODE_END", "NODE_AOV_START"):
        assert name not in bindings
        bindings[name] = ""
    closure = (args.cycles / "intern/cycles/kernel/svm/closure.h").read_text()
    skip = closure.split("int svm_node_closure_bsdf_skip(", 1)[1].split("return offset;", 1)[0]
    closures = {}
    for cases, payload in re.findall(r"((?:\s*case \w+:)+)\s*offset \+= sizeof\((\w+)\)", skip):
        for name in re.findall(r"case (\w+):", cases):
            closures[name] = payload
    assert len(closures) > 20
    # The full native closure dispatch also admits this legacy BSSRDF tag.
    assert "case CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID:" in closure
    closures["CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID"] = "SVMNodeBssrdfData"
    lines = ["// Generated declaration-layout inspection, never a shader evaluator.",
             "#include <psycles/compiler/cycles_svm_node_types.h>",
             "#include <cstddef>", "#include <iostream>",
             "using namespace psycles::compiler::cycles_svm;", "int main() {"]

    def literal(text):
        lines.append("std::cout << " + json.dumps(text) + ";")

    literal('{"structs":{')
    for index, (name, body) in enumerate(declarations.items()):
        literal(("," if index else "") + json.dumps(name) + ':{"words":')
        lines.append(f"std::cout << sizeof({name}) / 4;")
        literal(',"fields":{')
        clean = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        fields = re.findall(r"\b(\w+)\s+(\w+)(?:\[\d+\])?\s*;", clean)
        assert len(fields) == clean.count(";"), (name, clean)
        for field_index, (_, field) in enumerate(fields):
            literal(("," if field_index else "") + json.dumps(field) + ":[")
            lines.append(f"std::cout << offsetof({name}, {field}) << ',' << sizeof((({name} *)nullptr)->{field});")
            literal("]")
        literal("}}")
    literal('},"nodes":{')
    for index, (name, payload) in enumerate(bindings.items()):
        literal(("," if index else "") + '"')
        lines.append(f"std::cout << unsigned({name});")
        literal('":' + json.dumps({"name": name, "payload": payload}, separators=(",", ":")))
    literal('},"closures":{')
    for index, (name, payload) in enumerate(closures.items()):
        literal(("," if index else "") + '"')
        lines.append(f"std::cout << unsigned({name});")
        literal('":' + json.dumps(payload))
    literal('},"constants":{')
    constants = ("NODE_TEXCO_OBJECT_WITH_TRANSFORM", "NODE_SKY_SINGLE_SCATTERING",
                 "NODE_SKY_MULTIPLE_SCATTERING", "NODE_SKY_PREETHAM", "NODE_SKY_HOSEK")
    for index, name in enumerate(constants):
        literal(("," if index else "") + json.dumps(name) + ":")
        lines.append(f"std::cout << unsigned({name});")
    literal('},"transform_words":')
    lines.append("std::cout << sizeof(PackedTransform) / 4;")
    literal("}\n")
    lines.append("}")
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "layouts.cpp").write_text("\n".join(lines) + "\n")
    (args.destination / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\nproject(svm_layouts LANGUAGES CXX)\n"
        "add_executable(layouts layouts.cpp)\ntarget_compile_features(layouts PRIVATE cxx_std_20)\n"
        f"target_include_directories(layouts PRIVATE {ROOT}/include)\n")


if __name__ == "__main__":
    main()

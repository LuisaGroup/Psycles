"""Freeze native Math expansion regressions and complete measured controls."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
REPORTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORTS / "lamp-routing-and-surface"))
from archive_validation import campaign, gate
from audit_surface import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = args.evidence
    red, green = [(path / name).read_text() for name in ("red.log", "green.log")]
    assert red.count("first difference=") == 15
    assert green.strip() == "18 Math expansion/scheduling graphs match original Cycles"
    raw = json.loads((path / "scene-words.json").read_text())
    typed = json.loads((path / "typed-words.json").read_text())
    assert len(raw["shaders"]) == len(typed["shaders"]) == 279
    assert all(row["size_difference"] == 0 for row in raw["shaders"])
    assert typed["summary"] == {"raw_equal": 115,
        "resource_fields_only_unresolved_bindings": 158, "different_node_layout": 6}
    previous = json.loads((REPORTS / "svm-group-forwarding/scene-words.json").read_text())
    repaired = []
    for name in ("curtain_fabric", "door_inside", "door_inside.001", "Towel", "globe_main", "clock_backface"):
        before = next(row for row in previous["shaders"] if row["name"] == name)
        after = next(row for row in raw["shaders"] if row["name"] == name)
        classification = next(row for row in typed["shaders"] if row["name"] == name)
        assert classification["classification"] == "resource_fields_only_unresolved_bindings"
        repaired.append({"name": name, "before_differing_words": before["differing_common_words"],
            "after_differing_words": after["differing_common_words"]})
    native = (path / "native-vk.log").read_text()
    assert all(flag in native for flag in ("LUISA_VULKAN_USE_XIR=1",
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1", "LUISA_VULKAN_DISABLE_DXC=1"))
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native, re.I)
    assert native.count("SPIR-V compilation successful") == 31
    profile = json.loads((path / "profile-audit-unscoped.json").read_text())
    profile["scope"] = (
        "full Barbershop 2048x858/64 spp/seed 0; identical bundle and device code, "
        "native host Math expansion scheduling repair; sequential profiler follow-up, "
        "not fresh paired Cycles wall timings or an isolated speedup estimate")
    before_text, after_text = path / "surface-before.text", path / "surface-after.text"
    assert before_text.read_bytes() == after_text.read_bytes()
    profile["machine_code_control"] = {"identical_text_section": True,
        "before": source(before_text), "after": source(after_text),
        "scope": "llvm-objcopy .text of stage-map/LLVM/ELF-verified shade_surface; whole ELF hashes differ"}
    qa = json.loads((path / "typed-decoder-tests.json").read_text())
    assert qa["original_images"] == 541 and len(qa["checks"]) == 11
    payload = {
        "schema": "psycles.svm-math-expansion-validation.v1",
        "implementation": {"parent": "cbb73185", "child": "85e5300f1d85c66cfad7c2962dfe5028d9358980",
            "no_device_arithmetic_or_compiler_policy_change": True},
        "original_probes": {
            "scope": "18 original graphs; 15 exact-word failures before repair; all 18 raw-exact after; no tolerance or altered expected stream",
            "sources": [source(path / name) for name in (
                "probe-v2.blend", "cycles-v2.svm52", "cycles-v2.exr", "export-v2/scene.json",
                "create-v2.log", "cycles-v2.log", "export-v2.log")],
            "red": {"source": source(path / "red.log"), "output": red},
            "green": {"source": source(path / "green.log"), "output": green}},
        "fixtures": [source(ROOT / "tests/data" / name) for name in (
            "cycles_math_expand_scene.json", "cycles_math_expand_words.txt")],
        "phase_history": {name: source(path / name) for name in (
            "formal-analysis.md", "test-build-red.log", "test-build-fix.log", "first-fix.log", "full-build.log")},
        "word_audit": {"raw_summary": raw["summary"], "typed_summary": typed["summary"],
            "repaired_scheduling": repaired,
            "remaining_layouts": [row for row in typed["shaders"] if not row["same_node_layout"]],
            "sources": [source(path / name) for name in (
                "barbershop-fixed.svm52", "barbershop-fixed.log", "scene-words.json", "typed-words.json", "layouts.json")],
            "decoder_qa": {"source": source(path / "typed-decoder-tests.json"), "data": qa}},
        "profile_followup": profile,
        "retained_cycles_profile": source(REPORTS / "matched-pass-hip/barbershop-kernel-profile.json"),
        "gates": {"host": gate(path / "full-host.log", 166),
            "hip": gate(path / "full-hip.log", 182), "fallback": gate(path / "full-fallback.log", 184),
            "native_vulkan": {**gate(path / "native-vk.log", 2), "native_modules": 31, "dxc_dxil_loaded": False},
            "child": {"scope": "unchanged 85e5300f1; retained prior 155/155, not rerun here",
                      "source": source(REPORTS / "lamp-routing-and-surface/validation-results.json")}},
        "canaries": campaign(path / "canaries/canaries.json", True),
        "visual_qa": {"inspected": [str(path / "canaries" / (scene + "-fixed-1-triptychs/combined.png"))
            for scene in ("barbershop", "monk", "monster", "classroom")],
            "scope": "four first-run Combined triptychs at resized viewing resolution, not pixel parity"},
        "report_contract": {
            "surface": "existing repository Markdown requested by user; no additional web artifact",
            "audience": "technical; formal cause, original counterexamples, measured full-module and scene validation",
            "tables": "exact lookups separate raw words, layouts, resource IDs, GPU totals, wall time and storage",
            "limitation": "retained Cycles references and six follow-ups, not fresh timing pairs; no causal speedup claim"},
    }
    for name, data in (("scene-words.json", raw), ("typed-words.json", typed)):
        args.output.with_name(name).write_text(json.dumps(data, indent=2) + "\n")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

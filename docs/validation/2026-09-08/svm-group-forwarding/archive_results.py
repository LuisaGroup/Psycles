"""Freeze Blender forwarding counterexamples and measured full-scene controls."""
import argparse
from collections import Counter
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
    red = (path / "red.log").read_text()
    green = (path / "controls-green.log").read_text()
    assert red.count("first difference=") == 21
    assert green.strip() == "34 group-forwarding graphs match Cycles (typed literal roundoff only)"
    probe_words = json.loads((path / "probe-words-green.json").read_text())
    assert probe_words["summary"] == {"equal": 34, "different_payload": 1}
    # The full production dump also includes the world; it is not a 35th probe.
    differing = [row for row in probe_words["shaders"] if not row["raw_equal"]]
    assert len(differing) == 1 and differing[0]["name"] == "rgb-node-actual-float"
    assert differing[0]["size_difference"] == 0
    assert differing[0]["differing_common_words"] == 6
    before_path = REPORTS / "svm-socket-declarations/scene-words.json"
    before = json.loads(before_path.read_text())
    after = json.loads((path / "scene-words.json").read_text())
    assert len(before["shaders"]) == len(after["shaders"]) == 279
    assert after["summary"] == {"equal": 115, "different_payload": 164}
    words = {}
    for label, data in [("before", before), ("after", after)]:
        words[label] = {
            "raw_images": data["summary"],
            "surface_with_bump_word_deltas": dict(Counter(
                row["domains"]["surface_with_bump"]["size_difference"]
                for row in data["shaders"])),
        }
    native = (path / "native-vk.log").read_text()
    assert all(flag in native for flag in (
        "LUISA_VULKAN_USE_XIR=1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1",
        "LUISA_VULKAN_DISABLE_DXC=1"))
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native, re.I)
    assert native.count("SPIR-V compilation successful") == 31
    profile = json.loads((path / "profile-followup-unscoped.json").read_text())
    profile["scope"] = (
        "full Barbershop 2048x858/64 spp/seed 0; identical exported bundle, "
        "Blender linked/primitive forwarding and primitive luma/Gamma fixes; "
        "profiler follow-up, not fresh paired Cycles wall timings")
    before_text, after_text = path / "surface-before.text", path / "surface-after.text"
    profile["machine_code_control"] = {
        "source": "llvm-objcopy --dump-section .text on verified stage code objects",
        "before": source(before_text), "after": source(after_text),
        "identical_text_section": before_text.read_bytes() == after_text.read_bytes(),
    }
    assert profile["machine_code_control"]["identical_text_section"]
    assert profile["records"]["before"]["work"]["shade_surface"] == 332307890
    assert profile["records"]["after"]["work"]["shade_surface"] == 332307893
    payload = {
        "schema": "psycles.svm-group-forwarding-validation.v1",
        "implementation": {
            "parent": "6f743862", "child": "85e5300f1d85c66cfad7c2962dfe5028d9358980",
            "no_device_arithmetic_or_compiler_policy_change": True,
        },
        "original_probes": {
            "v1": {"scope": "28 original graphs; the old primitive mode is actually an authored RGB LinkedSocketValue",
                "sources": [source(path / name) for name in [
                    "probe.blend", "cycles.svm52", "cycles.exr", "export/scene.json"]],
                "red": {"source": source(path / "red.log"), "output": red}},
            "v2": {"scope": "34 original graphs including six staged Gamma controls; 33 bit-exact, one with three-ULP typed literal differences",
                "sources": [source(path / name) for name in [
                    "probe-v2.blend", "cycles-v2.svm52", "cycles-v2.exr", "export-v2/scene.json",
                    "psycles-green-v2.svm52", "probe-words-green.json"]],
                "green": {"source": source(path / "controls-green.log"), "output": green},
                "typed_roundoff_difference": differing[0]},
        },
        "fixtures": {name: source(ROOT / "tests/data" / name) for name in [
            "cycles_group_forward_scene.json", "cycles_group_forward_words.txt"]},
        "words": words,
        "word_sources": [source(before_path), source(path / "scene-words.json"),
                         source(path / "barbershop-green.log"), source(path / "barbershop-green.svm52")],
        "phase_history": {name: source(path / name) for name in [
            "formal-cause.md", "first-fix.log", "first-fix-detail.log", "full-host.log",
            "full-build.log", "verified-build.log", "complete-build.log", "test-controls-build.log"]},
        "profile_followup": profile,
        "retained_cycles_profile": source(REPORTS / "matched-pass-hip/barbershop-kernel-profile.json"),
        "gates": {
            "host": gate(path / "complete-host.log", 165),
            "hip": gate(path / "full-hip.log", 182),
            "fallback": gate(path / "full-fallback.log", 184),
            "native_vulkan": {**gate(path / "native-vk.log", 2),
                "native_modules": 31, "dxc_dxil_loaded": False},
            "child": {"scope": "unchanged published 85e5300f1; retained prior 155/155, not rerun here",
                "source": source(REPORTS / "lamp-routing-and-surface/validation-results.json")},
        },
        "canaries": campaign(path / "canaries/canaries.json", True),
        "visual_qa": {
            "inspected": [str(path / "canaries" / (scene + "-fixed-1-triptychs/combined.png"))
                          for scene in ["barbershop", "monk", "monster", "classroom"]],
            "scope": "four first-run Combined triptychs inspected at resized viewing resolution; not pixel parity",
        },
        "report_contract": {
            "surface": "user-requested existing repository Markdown documentation; no additional web report",
            "audience": "technical",
            "structure": ["technical summary", "formal source-category invariants and original probes",
                          "full-scene word audit", "runtime and performance controls",
                          "source and metric definitions", "limitations and remaining questions"],
            "tables": "exact per-run and per-gate lookup; separate wall, GPU sum, storage and word counts; no causal gain inferred from temporal follow-ups",
            "baseline": "retained equal-pass Cycles references and preceding Psycles checkpoint; six follow-ups, not fresh timing pairs",
        },
    }
    args.output.with_name("scene-words.json").write_text(json.dumps(after, indent=2) + "\n")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

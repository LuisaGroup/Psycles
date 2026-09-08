"""Archive original socket/declaration counterexamples and full-scene controls."""
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


def probes(path, blend, dump, image, red, green, failures, success):
    red_text, green_text = (path / red).read_text(), (path / green).read_text()
    assert red_text.count("first difference=") == failures
    assert green_text.strip() == success
    return {
        "blend": source(path / blend), "original_cycles_words": source(path / dump),
        "original_cycles_image": source(path / image),
        "red": {"source": source(path / red), "output": red_text},
        "green": {"source": source(path / green), "output": green_text},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = args.evidence
    before_path = REPORTS / "svm-graph-boundaries/scene-words.json"
    before = json.loads(before_path.read_text())
    after = json.loads((path / "scene-words.json").read_text())
    assert len(before["shaders"]) == len(after["shaders"]) == 279
    word_counts = {}
    for label, data in [("before", before), ("after", after)]:
        word_counts[label] = {
            "raw_images": data["summary"],
            "surface_with_bump_word_deltas": dict(Counter(
                row["domains"]["surface_with_bump"]["size_difference"]
                for row in data["shaders"])),
        }
    originals = {
        "texture_socket_types": probes(path, "probe.blend", "cycles.svm52", "cycles.exr",
            "red.log", "green.log", 19, "27 typed texture-input graphs match Cycles"),
        "closure_input_order": probes(path, "closure-inputs.blend", "closure-cycles.svm52",
            "closure-cycles.exr", "closure-red.log", "closure-green.log", 9,
            "26 closure-input graphs match Cycles"),
    }
    identity_red = (path / "identity-red.log").read_text()
    assert identity_red.strip() == "ConvertNode identity must include both Cycles socket types"
    originals["conversion_identity"] = {
        "red": {"source": source(path / "identity-red.log"), "output": identity_red},
        "green": "same permanent host invariant in the complete 164/164 host suite",
    }
    native = (path / "native-vk.log").read_text()
    assert all(flag in native for flag in (
        "LUISA_VULKAN_USE_XIR=1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1",
        "LUISA_VULKAN_DISABLE_DXC=1"))
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native, re.I)
    assert native.count("SPIR-V compilation successful") > 0
    fixtures = {}
    for stem in ["cycles_socket_types", "cycles_closure_inputs"]:
        for suffix in ["_scene.json", "_words.txt"]:
            name = stem + suffix
            fixtures[name] = source(ROOT / "tests/data" / name)
    profile = json.loads((path / "profile-followup-unscoped.json").read_text())
    profile["scope"] = (
        "full Barbershop 2048x858/64 spp/seed 0; identical exported input bundle, "
        "native texture socket types/conversion identity, closure declaration order "
        "and unavailable Voronoi default repairs; profiler follow-up, "
        "not fresh paired Cycles wall timings")
    before_text, after_text = path / "surface-before.text", path / "surface-after.text"
    profile["machine_code_control"] = {
        "source": "llvm-objcopy --dump-section .text on the verified stage code objects",
        "before": source(before_text), "after": source(after_text),
        "identical_text_section": before_text.read_bytes() == after_text.read_bytes(),
    }
    assert profile["machine_code_control"]["identical_text_section"]
    assert profile["records"]["before"]["work"]["shade_surface"] == 332307894
    assert profile["records"]["after"]["work"]["shade_surface"] == 332307890
    payload = {
        "schema": "psycles.svm-socket-declaration-validation.v1",
        "implementation": {
            "parent": "33c7b335995e3e39abc21f4010c50b5f01a06eb1",
            "test_refactor": "322a3855",
            "child": "85e5300f1d85c66cfad7c2962dfe5028d9358980",
            "no_device_arithmetic_or_compiler_policy_change": True,
        },
        "original_probes": originals,
        "fixtures": fixtures,
        "words": word_counts,
        "word_sources": [source(before_path), source(path / "scene-words.json"),
                         source(path / "barbershop-compile.log"),
                         source(path / "barbershop-typed.svm52")],
        "phase_history": {name: source(path / name) for name in [
            "formal-cause.md", "typed-fix.log", "full-host.log", "phase-host.log",
            "verified-host.log", "complete-build.log", "post-split-build.log"]},
        "profile_followup": profile,
        "retained_cycles_profile": source(
            REPORTS / "matched-pass-hip/barbershop-kernel-profile.json"),
        "gates": {
            "host": gate(path / "complete-host.log", 164),
            "hip": gate(path / "full-hip.log", 182),
            "fallback": gate(path / "full-fallback.log", 184),
            "native_vulkan": {**gate(path / "native-vk.log", 2),
                "native_modules": native.count("SPIR-V compilation successful"),
                "dxc_dxil_loaded": False},
        },
        "canaries": campaign(path / "canaries/canaries.json", True),
        "visual_qa": {
            "inspected": [str(path / "canaries" / (scene + "-fixed-1-triptychs/combined.png"))
                          for scene in ["barbershop", "monk", "monster", "classroom"]],
            "scope": "all four first-run Combined triptychs inspected at resized viewing resolution; no obvious new layout/material regression; not pixel parity",
        },
        "report_contract": {
            "surface": "existing repository Markdown documentation requested by the user; no additional web report",
            "audience": "technical",
            "structure": ["technical summary", "formal graph invariants and original probes",
                          "full-scene word audit", "runtime and performance controls",
                          "source/metric definitions", "limitations and remaining questions"],
            "tables": "exact per-run and per-gate lookup; separate wall time, kernel sum, storage and word counts; no causal gain inferred from one temporal follow-up",
            "baseline": "retained Cycles equal-pass references and previous Psycles binaries; six current follow-ups, not fresh Cycles timing pairs",
        },
    }
    args.output.with_name("scene-words.json").write_text(json.dumps(after, indent=2) + "\n")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

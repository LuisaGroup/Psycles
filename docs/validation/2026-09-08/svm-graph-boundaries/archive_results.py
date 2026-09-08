"""Archive original probes, complete gates and measured full-scene controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
PRIOR = Path(__file__).resolve().parent.parent / "lamp-routing-and-surface"
sys.path.insert(0, str(PRIOR))
from archive_validation import campaign, gate
from audit_surface import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = args.evidence
    before = json.loads((PRIOR / "scene-words-fixed.json").read_text())
    after = json.loads((path / "scene-words-textures-fixed.json").read_text())
    assert len(before["shaders"]) == len(after["shaders"]) == 279
    word_counts = {}
    for label, data in [("before", before), ("after", after)]:
        word_counts[label] = {
            "raw_images": data["summary"],
            "surface_with_bump_word_deltas": dict(Counter(
                row["domains"]["surface_with_bump"]["size_difference"]
                for row in data["shaders"])),
        }
    originals = {}
    for stem, red, green, failures in [
        ("bump-alias", "probe-red.log", "probe-green.log", 7),
        ("texture-outputs", "texture-red.log", "texture-green.log", 8),
    ]:
        red_text = (path / red).read_text()
        green_text = (path / green).read_text()
        assert red_text.count("first difference=") == failures
        assert "12 " in green_text and "match Cycles" in green_text
        originals[stem] = {
            "blend": source(path / (stem + ".blend")),
            "original_cycles_words": source(path / (stem + "-cycles.svm52")),
            "original_cycles_image": source(path / (stem + "-cycles.exr")),
            "red": {"source": source(path / red), "output": red_text},
            "green": {"source": source(path / green), "output": green_text},
        }
    native = (path / "native-vk.log").read_text()
    assert all(flag in native for flag in (
        "LUISA_VULKAN_USE_XIR=1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1",
        "LUISA_VULKAN_DISABLE_DXC=1"))
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native, re.I)
    assert native.count("SPIR-V compilation successful") > 0
    fixtures = {}
    for stem in ["cycles_bump_alias", "cycles_texture_outputs"]:
        for suffix in ["_scene.json", "_words.txt"]:
            name = stem + suffix
            fixtures[name] = source(ROOT / "tests/data" / name)
    profile = json.loads((path / "profile-followup-unscoped.json").read_text())
    profile["scope"] = (
        "full Barbershop 2048x858/64 spp/seed 0; identical exported input bundle, "
        "host bump edge/domain and procedural output repairs; profiler follow-up, "
        "not fresh paired Cycles wall timings")
    before_text, after_text = (path / "surface-before.text", path / "surface-after.text")
    profile["machine_code_control"] = {
        "source": "llvm-objcopy --dump-section .text on the verified stage code objects",
        "before": source(before_text), "after": source(after_text),
        "identical_text_section": before_text.read_bytes() == after_text.read_bytes(),
    }
    assert profile["machine_code_control"]["identical_text_section"]
    assert profile["records"]["before"]["work"]["shade_surface"] == 332307894
    assert profile["records"]["after"]["work"]["shade_surface"] == 332307894
    payload = {
        "schema": "psycles.svm-graph-boundary-validation.v1",
        "implementation": {"parent": "5096a41f3cd60bb72bca2cc565bb01c19d0c042e",
                           "child": "85e5300f1d85c66cfad7c2962dfe5028d9358980",
                           "no_device_arithmetic_or_compiler_policy_change": True},
        "original_probes": originals,
        "fixtures": fixtures,
        "words": word_counts,
        "word_sources": [source(PRIOR / "scene-words-fixed.json"),
                         source(path / "scene-words-textures-fixed.json"),
                         source(path / "barbershop-textures-compile.log")],
        "profile_followup": profile,
        "retained_cycles_profile": source(
            PRIOR.parent / "matched-pass-hip/barbershop-kernel-profile.json"),
        "gates": {"host": gate(path / "verified-host.log", 161, failed=1),
                  "hip": gate(path / "full-hip.log", 182),
                  "fallback": gate(path / "full-fallback.log", 184),
                  "native_vulkan": {**gate(path / "native-vk.log", 2),
                                     "native_modules": native.count("SPIR-V compilation successful"),
                                     "dxc_dxil_loaded": False}},
        "canaries": campaign(path / "canaries/canaries.json", True),
        "visual_qa": {
            "inspected": [str(path / "canaries" / (scene + "-fixed-1-triptychs/combined.png"))
                          for scene in ["barbershop", "monk", "monster", "classroom"]],
            "scope": "all four first-run Combined triptychs inspected at resized viewing resolution; no obvious new layout/material regression; not pixel parity",
        },
        "report_contract": {
            "surface": "existing repository Markdown documentation requested by the user; no additional web report",
            "audience": "technical",
            "structure": ["technical summary", "findings and word-count audit", "source/metric definitions",
                          "formal graph invariants and original probes", "runtime checks and performance controls",
                          "limitations and remaining questions"],
            "tables": "exact per-run and per-gate lookup; separate wall time, kernel sum, storage and word counts; no invented trend or causal attribution",
            "baseline": "retained Cycles equal-pass references and previous Psycles binaries; six current follow-ups, not fresh Cycles timing pairs",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

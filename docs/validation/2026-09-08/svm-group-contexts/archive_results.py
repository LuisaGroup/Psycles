"""Freeze original group-context regressions, whole-scene audits and timings."""
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
    parser.add_argument("--implementation", required=True)
    args = parser.parse_args()
    path = args.evidence
    red = (path / "context-red-v2.log").read_text()
    green = (path / "context-green.log").read_text()
    assert red.count("first difference=") == 12
    assert "minimal-chain-bump0: Psycles=22 Cycles=25 words" in red
    assert green.strip() == "30 lazy group input/context graphs match original Cycles"
    native = (path / "native-vk.log").read_text()
    assert all(flag in native for flag in ("LUISA_VULKAN_USE_XIR=1",
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1", "LUISA_VULKAN_DISABLE_DXC=1"))
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native, re.I)
    assert native.count("SPIR-V compilation successful") == 31
    attribution_qa = (path / "profile-artifacts-qa.log").read_text()
    assert "Ran 4 tests" in attribution_qa and attribution_qa.rstrip().endswith("OK")
    raw = json.loads((path / "scene-words.json").read_text())
    typed = json.loads((path / "typed-words.json").read_text())
    assert len(raw["shaders"]) == len(typed["shaders"]) == 279
    assert all(row["size_difference"] == 0 for row in raw["shaders"])
    assert typed["summary"] == {"raw_equal": 115,
        "resource_fields_only_unresolved_bindings": 164}
    previous = json.loads((REPORTS / "svm-mapping-declarations/results.json").read_text())
    repaired = []
    for before in previous["typed_word_audit"]["shaders"]:
        if before["classification"] != "different_node_layout":
            continue
        after = next(row for row in typed["shaders"] if row["name"] == before["name"])
        assert after["classification"] == "resource_fields_only_unresolved_bindings"
        repaired.append({"name": after["name"],
            "before_differing_words": before["differing_common_words"],
            "after_differing_words": after["differing_common_words"]})
    assert len(repaired) == 6
    profile = json.loads((path / "profile-unscoped.json").read_text())
    profile["scope"] = ("complete Barbershop 2048x858/64 spp/seed 0 before/after host group-context "
        "repair; identical bundle, sequential profiler runs, no concurrent heavy work; "
        "not fresh paired Cycles wall timings or an isolated causal speedup estimate")
    before_text = Path("/var/tmp/psycles-mapping-declarations-2mxqRk/surface-after.text")
    after_text = path / "surface-after.text"
    profile["machine_code_control"] = {"identical_text_section": before_text.read_bytes() == after_text.read_bytes(),
        "before": source(before_text), "after": source(after_text),
        "scope": "stage-map, final LLVM and ELF verified shade_surface .text; not whole ELF identity"}
    sources = ("formal-cause.md", "intervention.patch", "probe-v2.blend", "cycles-v2.svm52",
        "cycles-v2.exr", "export-v2/scene.json", "cycles-v2.log", "red.log", "first-fix.log",
        "context.blend", "context.svm52", "context.exr", "context-export/scene.json",
        "context-cycles.log", "context-red-v2.log", "context-green.log", "context-red.svm52",
        "red-build.log", "fix-build.log", "context-red-build.log", "context-red-build-v2.log",
        "context-fix-build.log", "full-build.log", "barbershop-fixed.svm52", "barbershop-fixed.log")
    reduction = Path("/var/tmp/psycles-default-coordinates-FTAxTb")
    payload = {"schema": "psycles.svm-group-contexts-validation.v1",
        "implementation": {"parent": args.implementation, "child": "85e5300f1",
            "change": "host lazy group input and persistent instance context only",
            "no_device_arithmetic_or_compiler_policy_change": True},
        "original_regressions": {"images": 30, "before_failed": 12, "after_raw_exact": 30,
            "red": red, "green": green, "sources": [source(path / name) for name in sources],
            "scope": "initial sixteen-case red precedes implementation; fourteen extra controls replay unchanged pre-fix source; fixtures are original-only and unchanged after their red runs"},
        "fixtures": [source(ROOT / "tests/data" / (stem + suffix))
            for stem in ("cycles_group_liveness", "cycles_group_context")
            for suffix in ("_scene.json", "_words.txt")],
        "reduction_history": {"sources": [source(reduction / name) for name in (
            "formal-analysis.md", "words-red.json", "material-typed-v2.json",
            "link-reduction/history.json", "nested-reduction-v2/history.json",
            "nested-reduction-v2/round-05/candidates.blend", "nested-reduction-v2/round-05/cycles.svm52")],
            "scope": "32 exact default-coordinate controls refute the initial simple phase hypothesis; original graph reduction exposes lazy group inputs; final greedy witness is one-link-minimal only in the selected scope"},
        "word_audit": {"raw_summary": raw["summary"], "typed_summary": typed["summary"],
            "repaired_layouts": repaired, "resource_binding_equivalence": "unresolved",
            "sources": [source(path / name) for name in ("scene-words.json", "typed-words.json")]},
        "profile_followup": profile,
        "profile_attribution_qa": {"source": source(path / "profile-artifacts-qa.log"),
            "tests": 4, "scope": "independent dump counters, missing entry, ambiguous LLVM and ELF matches"},
        "gates": {"host": gate(path / "full-host.log", 168), "hip": gate(path / "full-hip.log", 182),
            "fallback": gate(path / "full-fallback.log", 184),
            "native_vulkan": {**gate(path / "native-vk.log", 2), "native_modules": 31, "dxc_dxil_loaded": False},
            "child": {"scope": "unchanged 85e5300f1, prior 155/155 retained, not rerun"}},
        "canaries": campaign(path / "canaries/canaries.json", True),
        "visual_qa": {"inspected": [str(path / "canaries" / (scene + "-fixed-1-triptychs/combined.png"))
            for scene in ("barbershop", "monk", "monster", "classroom")],
            "scope": "all four first-run Combined triptychs viewed; not pixel-parity proof"},
        "open": "resource binding identity, surface per-invocation cost, residual DiffInd/shadow work, remaining unsupported node families and legacy displacement removal"}
    for name, value in (("scene-words.json", raw), ("typed-words.json", typed)):
        args.output.with_name(name).write_text(json.dumps(value, indent=2) + "\n")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

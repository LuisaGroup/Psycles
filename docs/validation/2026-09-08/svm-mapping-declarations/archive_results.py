"""Archive native Mapping declaration counterexamples and full-scene control."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
REPORTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORTS / "lamp-routing-and-surface"))
from archive_validation import gate
from audit_surface import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    path = args.evidence
    red = (path / "red-v2.log").read_text()
    green = (path / "first-fix.log").read_text()
    assert red.count("first difference=") == 5
    assert "minimal-mapping-shared: Psycles=47 Cycles=44 words" in red
    assert green.strip() == "18 native Mapping declaration graphs match original Cycles"
    native = (path / "native-vk.log").read_text()
    assert all(flag in native for flag in ("LUISA_VULKAN_USE_XIR=1",
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1", "LUISA_VULKAN_DISABLE_DXC=1"))
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native, re.I)
    assert native.count("SPIR-V compilation successful") == 31
    words = json.loads((path / "typed-words.json").read_text())
    assert words["summary"] == {"raw_equal": 115,
        "resource_fields_only_unresolved_bindings": 158, "different_node_layout": 6}
    profile = json.loads((path / "profile-unscoped.json").read_text())
    profile["scope"] = ("complete Barbershop 2048x858/64 spp/seed 0 before/after native Mapping "
        "POINT restoration; identical bundle; no concurrent heavy work; not new paired Cycles timings")
    text = source(path / "surface-after.text")
    assert text["sha256"] == "2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb"
    profile["machine_code_control"] = {"identical_to_previous_surface_text": True, "after": text,
        "before": source(Path("/var/tmp/psycles-svm-schedule-LaccgH/surface-after.text"))}
    payload = {"schema": "psycles.svm-mapping-declarations-validation.v1",
        "implementation": {"parent": "61443f70", "child": "85e5300f1",
            "change": "host native Mapping declarations only; no device arithmetic or inlining change"},
        "oracle": {"red": red, "green": green,
            "sources": [source(path / name) for name in (
                "formal-analysis.md", "probe-v2.blend", "cycles-v2.svm52", "cycles-v2.exr",
                "export-v2/scene.json", "cycles-v2.log", "red-v2.log", "first-fix.log",
                "red-v2-build.log", "fix-build.log", "full-build.log")]},
        "fixtures": [source(ROOT / "tests/data" / name) for name in (
            "cycles_mapping_declarations_scene.json", "cycles_mapping_declarations_words.txt")],
        "typed_word_audit": words, "raw_word_audit": json.loads((path / "scene-words.json").read_text()),
        "profile_followup": profile,
        "gates": {"host": gate(path / "first-host.log", 167), "hip": gate(path / "full-hip.log", 182),
            "fallback": gate(path / "full-fallback.log", 184),
            "native_vulkan": {**gate(path / "native-vk.log", 2), "native_modules": 31, "dxc_dxil_loaded": False}},
        "benchmark_scope": "latest 256-spp four-scene campaign remains cbb73185; not rerun for this isolated follow-up",
        "open": "six remaining layouts, resource binding equivalence, DiffInd and surface per-invocation cost"}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

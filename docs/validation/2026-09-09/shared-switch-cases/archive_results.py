"""Freeze shared-case regression, final-code and four-scene evidence."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/validation/2026-09-08/lamp-routing-and-surface"))
from archive_validation import campaign, gate
from audit_surface import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    args = parser.parse_args()
    evidence = args.evidence
    profiles = json.loads((evidence / "profiles.json").read_text())
    assert profiles["schema"] == "psycles.shared-switch-case-profiles.v1"
    for name in ("runtime-hip-final.log", "runtime-fallback-final.log", "runtime-vk-green.log"):
        assert "211 asserts" in (evidence / name).read_text()
    assert "552 asserts" in (evidence / "cfg-green.log").read_text()
    assert "17/35" in (evidence / "bsdf-structure-red.log").read_text()
    assert "35/35" in (evidence / "bsdf-structure-green.log").read_text()
    native = (evidence / "native-vk-final.log").read_text()
    assert all(flag in native for flag in (
        "LUISA_VULKAN_USE_XIR=1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1",
        "LUISA_VULKAN_DISABLE_DXC=1"))
    for log in (native, (evidence / "runtime-vk-green.log").read_text()):
        assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil|dxc)", log, re.I)
    modules = native.count("SPIR-V compilation successful")
    assert modules > 0
    result = {
        "schema": "psycles.shared-switch-cases.v1",
        "implementation": {"parent": args.parent, "child": args.child,
            "before_parent": "e6085640", "before_child": "d51a33d48",
            "groups": 7, "old_bodies": 32, "new_bodies": 7,
            "no_inlining_policy_math_or_register_limit_change": True},
        "original_structural_reference": {
            "revision": "cb168525138fecc792cc393f94afc39582b0103c",
            "source": source(Path("/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles/kernel/closure/bsdf.h")),
            "scope": "sample/eval, roughness/eta, label and blur switch grouping; no changed SVM words or CPU shader evaluator"},
        "regressions": {"psycles_structural_shapes": 35,
            "psycles_before_failed_shapes": 18, "xir_shared_cfg_shapes": 48,
            "xir_shared_cfg_assertions": 552, "each_backend_runtime_assertions": 211,
            "sources": [source(evidence / name) for name in (
                "formal-analysis.md", "roundtrip-red.log", "cfg-shared-target-analysis.md",
                "cfg-red.log", "cfg-green.log", "spirv-literal-analysis.md",
                "runtime-vk-narrow-red.log", "runtime-hip-final.log",
                "runtime-fallback-final.log", "runtime-vk-green.log",
                "bsdf-structure-red.log", "bsdf-structure-green.log")]},
        "profiles": {"source": source(evidence / "profiles.json"), "data": profiles},
        "gates": {"host": gate(evidence / "full-host.log", 169),
            "hip": gate(evidence / "full-hip.log", 182),
            "fallback": gate(evidence / "full-fallback.log", 184),
            "child_hip_configuration": gate(evidence / "child-unit-complete-expanded.log", 133),
            "child_exact_unit_label": gate(evidence / "child-unit-complete.log", 132),
            "native_vulkan": {**gate(evidence / "native-vk-final.log", 3),
                "modules": modules, "dxc_dxil_loaded": False},
            "builds": [source(evidence / name) for name in (
                "root-full-build-final.log", "child-full-build-final.log")]},
        "canaries": campaign(evidence / "canaries/canaries.json", True),
        "visual_qa": {"inspected": [str(evidence / "canaries" / (scene + "-fixed-1-triptychs/combined.png"))
            for scene in ("barbershop", "monk", "monster", "classroom")],
            "scope": "all four first-run Combined triptychs viewed; quantitative pass differences remain unresolved"},
        "open": ["Barbershop per-surface-invocation cost and residual DiffInd/shadow work",
            "closure setup dispatch and placement of parameter evaluation relative to native guards",
            "164 used-shader resource binding identities; layouts already matched separately",
            "unimplemented SVM nodes and remaining legacy displacement path",
            "separate existing native Vulkan f16/f64 remainder failures",
            "remaining general restructure_cfg proof obligations from the child loop-epoch audit"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

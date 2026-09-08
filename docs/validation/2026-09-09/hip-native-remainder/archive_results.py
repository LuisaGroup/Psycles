"""Freeze native-remainder semantics, bounded ISA counts and full-scene controls."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/validation/2026-09-08/lamp-routing-and-surface"))
from archive_validation import campaign, gate
from audit_surface import source


def output_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        if re.match(r"^\d+ \d+ (?:[0-9a-f]{8} ){4}", line):
            columns = line.split()
            rows[tuple(map(int, columns[:2]))] = columns[2:6]
    assert len(rows) == 32, path
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--child", required=True)
    parser.add_argument("--parent", required=True)
    args = parser.parse_args()
    evidence = args.evidence
    profiles = []
    controls = {}
    for label, filename in (("native_first", "profile-unscoped.json"),
                            ("native_repeat", "profile-repeat-unscoped.json"),
                            ("restored", "profile-restored-unscoped.json")):
        report = json.loads((evidence / filename).read_text())
        if not profiles:
            profiles.append({"label": "before", **report["records"]["before"]})
        row = {"label": label, **report["records"]["after"]}
        log = Path(row["directory"], "render.log").read_text()
        row["session_init_seconds"] = float(re.search(r"Luisa shader JIT completed in ([\d.]+) s", log)[1])
        profiles.append(row)
        controls[label] = report["image_control"]
    first_log = Path(profiles[0]["directory"], "render.log").read_text()
    profiles[0]["session_init_seconds"] = float(re.search(r"Luisa shader JIT completed in ([\d.]+) s", first_log)[1])
    before_text = Path(profiles[0]["directory"], "surface.text")
    restored_text = Path(profiles[-1]["directory"], "surface.text")
    assert before_text.read_bytes() == restored_text.read_bytes()
    for row in profiles:
        assert row["frame"] == {"stages": 6, "fields": 93, "bytes": 416}
        row["total_function_static_instructions"] = sum(
            function["static_instructions"] for function in row["functions"].values())
        row["total_function_bytes"] = sum(function["bytes"] for function in row["functions"].values())

    oracle = output_rows(evidence / "cycles-fast.txt")
    probe_results = []
    for label, filename in (("old_lowering", "luisa-fast-2U0xap/run.log"),
                            ("early_external_diagnostic", "early-ocml-fast-UpTVIw/run.log"),
                            ("native_lowering", "luisa-native-fast-Zb0yZ0/run.log")):
        actual = output_rows(evidence / filename)
        remainder_lanes = noise_lanes = 0
        differences = []
        for key, expected in oracle.items():
            for lane, reference in enumerate(expected):
                result = actual[key][lane]
                if key[1] == 2:
                    noise_lanes += 1
                    if reference != result:
                        differences.append({"input": key[0], "lane": lane,
                            "original_bits": reference, "actual_bits": result,
                            "unsigned_bit_distance": abs(int(reference, 16) - int(result, 16))})
                else:
                    remainder_lanes += 1
                    assert result == reference, (label, key, lane)
        assert remainder_lanes == 96 and noise_lanes == 32
        assert len(differences) == 1 and differences[0]["unsigned_bit_distance"] == 1
        probe_results.append({"label": label, "exact_remainder_lanes": remainder_lanes,
            "noise_lanes": noise_lanes, "ignored_one_ulp_noise_difference": differences,
            "source": source(evidence / filename)})

    red = (evidence / "unit-red.log").read_text()
    green = (evidence / "unit-green.log").read_text()
    assert "24 failed" in red and "396 asserts" in green
    assert "5232 asserts" in (evidence / "runtime-special-hip.log").read_text()
    assert "5232 asserts" in (evidence / "runtime-root-fallback.log").read_text()
    assert "1744 asserts" in (evidence / "runtime-root-vk-float.log").read_text()
    assert "1076 failed" in (evidence / "runtime-root-vk.log").read_text()
    native = (evidence / "native-vk.log").read_text()
    assert all(flag in native for flag in ("LUISA_VULKAN_USE_XIR=1",
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1", "LUISA_VULKAN_DISABLE_DXC=1"))
    assert native.count("SPIR-V compilation successful") == 31
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native, re.I)
    payload = {"schema": "psycles.hip-native-remainder.v1",
        "implementation": {"parent": args.parent, "child": args.child,
            "before_child": "85e5300f1", "shader_cache_codegen_revision": 85,
            "change": "XIR floating BINARY_MOD to installed native OCML before IPO; integer remainder and inlining policy unchanged"},
        "original_gpu_probe": {"records": probe_results,
            "cycles_source_revision": "cb168525138fecc792cc393f94afc39582b0103c",
            "original_headers": [source(Path("/home/mike/Projects/blender-cycles-trace-5.2/intern/cycles") / name)
                for name in ("kernel/svm/noise.h", "util/math_float3.h", "util/math_float4.h")],
            "scope": "original Cycles HIP functions with dynamic operands; no CPU shader oracle",
            "sources": [source(evidence / name) for name in (
                "cycles-fast.txt", "cycles-precise.txt", "build/cycles-fast.ll",
                "cycles-fast-unbundled.co", "CMakeLists.txt", "formal-analysis.md")]},
        "regressions": {"ir_abi_shapes": 12, "before_failed_checks": 24,
            "after_ir_assertions": 396, "hip_arithmetic_assertions": 5232,
            "scope": "old CreateFRem expression extracted unchanged at test seam; lowering-quality red, not old numerical failure",
            "sources": [source(evidence / name) for name in (
                "unit-red.log", "unit-green.log", "runtime-baseline-hip.log",
                "runtime-special-hip.log", "arithmetic-fixed-hip.log", "noise-fixed-hip.log",
                "profile-artifacts-qa.log", "runtime-final-hip-v2.log",
                "arithmetic-final-hip.log", "cache-final-hip.log",
                "runtime-root-fallback.log", "runtime-root-vk-float.log",
                "runtime-root-vk.log", "vulkan-followup.md")]},
        "full_scene_profiles": {"records": profiles, "image_controls": controls,
            "scope": "sequential full Barbershop 2048x858/64 spp/seed 0 A/B/B/A, no overlapping heavy work; not fresh Cycles timing pairs or an isolated speedup estimate",
            "restored_text_identical": True, "restoration_sources": [source(before_text), source(restored_text)],
            "static_counts": "ELF STT_FUNC extents only; excludes decoded alignment padding; not dynamic instruction or spill traffic",
            "jit_scope": "session initialization includes setup/baking; first native HIPRTC compile is cold relative to later downstream cache hits; not compiler-only or a matched cold-JIT comparison"},
        "gates": {"host": gate(evidence / "full-host.log", 168),
            "hip": gate(evidence / "full-hip.log", 182),
            "fallback": gate(evidence / "full-fallback.log", 184),
            "child_hip_configuration": gate(evidence / "child-unit.log", 131),
            "native_vulkan": {**gate(evidence / "native-vk.log", 2), "modules": 31, "dxc_dxil_loaded": False,
                "extra_float32_remainder_assertions": 1744,
                "extra_all_type_remainder_failed_assertions": 1076,
                "open_failure_scope": "existing f16/f64 OpFRem lowering; no Vulkan source changes in this HIP intervention"}},
        "canaries": campaign(evidence / "canaries/canaries.json", True),
        "visual_qa": {"inspected": [str(evidence / "canaries" / (scene + "-fixed-1-triptychs/combined.png"))
            for scene in ("barbershop", "monk", "monster", "classroom")],
            "scope": "all four first-run Combined triptychs viewed; quantitative pass differences remain unresolved"},
        "open": "Barbershop surface per-invocation gap, shared closure-case emission, resource binding identities, residual DiffInd/shadow work, unsupported nodes and legacy displacement removal"}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

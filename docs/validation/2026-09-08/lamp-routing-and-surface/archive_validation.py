"""Freeze complete, validated canary records and original-probe provenance."""
import argparse
import json
from pathlib import Path
import re

from audit_surface import source

ROOT = Path(__file__).resolve().parents[4]
ORDER = [("barbershop", 1), ("monk", 1), ("monster", 1),
         ("classroom", 1), ("barbershop", 2), ("barbershop", 3)]


def campaign(path, controlled):
    data = json.loads(path.read_text())
    assert data["schema"] == "psycles.light-endpoint-canaries.v1"
    assert data["main_shader_cache"] == "disabled"
    assert [(r["scene"], r["repeat"]) for r in data["records"]] == ORDER
    assert len(data["implementation_sha256"]) == 6
    for row in data["records"]:
        assert row["actual_all_finite"]
        assert len(row["pass_contract"]["channels"]) == 46
        assert len(row["passes"]) == 15
        assert all(p["actual_invalid_pixels"] == 0 for p in row["passes"].values())
        assert row["render"]["returncode"] == row["comparison"]["returncode"] == 0
        if controlled:
            record = row["socket_metadata_control"]
            assert record["old_geometry"]["sha256"] == record["control_geometry"]["sha256"]
            assert record["fresh_scene"]["sha256"] == record["control_scene"]["sha256"]
    return {"source": source(path), "data": data}


def gate(path, total, failed=0):
    text = path.read_text()
    match = re.search(r"\d+% tests passed(?:, (\d+) tests failed)? out of (\d+)", text)
    assert match, path
    assert (int(match[2]), int(match[1] or 0)) == (total, failed), path
    elapsed = re.search(r"Total Test time \(real\) =\s*([\d.]+) sec", text)
    return {"source": source(path), "tests": total, "failed": failed,
            "elapsed_seconds": float(elapsed[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("routing_evidence", type=Path)
    parser.add_argument("alignment_evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before, after = args.routing_evidence, args.alignment_evidence
    native_log = (after / "queue-fix-native-vk.log").read_text()
    assert all(flag in native_log for flag in (
        "LUISA_VULKAN_USE_XIR=1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1",
        "LUISA_VULKAN_DISABLE_DXC=1"))
    assert not re.search(r"(?:find library=|calling init:).*lib(?:dxcompiler|dxil)", native_log, re.I)
    modules = native_log.count("SPIR-V compilation successful")
    assert modules > 0
    probe_paths = ["cycles.svm52", "vector-stage-domains-cycles.svm52",
                   "probe.blend", "vector-stage-domains.blend"]
    sources = {name: source(after / name) for name in probe_paths}
    sources["barbershop_original"] = source(Path(
        "/var/tmp/psycles-native-volume-svm-06XnDX/barbershop-native-oracle.svm52"))
    sources["barbershop_aligned"] = source(after / "barbershop-psycles-verified.svm52")
    fixture_names = ["cycles_hidden_socket_scene.json", "cycles_hidden_socket_words.txt",
                     "cycles_vector_fold_scene.json", "cycles_vector_fold_words.txt"]
    payload = {
        "schema": "psycles.lamp-routing-surface-validation.v1",
        "implementation_scope": {
            "post_routing_parent": "773f1aca", "post_routing_child": "4284e8cb9",
            "alignment_parent": "cf59ab5b",
            "alignment_child": "85e5300f1",
            "change": "host socket provenance and Vector Math folding; fallback queue wait protocol",
            "no_device_arithmetic_or_inlining_policy_change": True,
            "timing_scope": "six follow-ups per checkpoint, retained Cycles references; not fresh pairs",
        },
        "original_sources": sources,
        "fixtures": {name: source(ROOT / "tests/data" / name) for name in fixture_names},
        "gates": {
            "hip": gate(after / "queue-fix-hip.log", 182),
            "fallback": gate(after / "queue-fix-fallback.log", 184),
            "host": gate(after / "verified-host.log", 159, failed=1),
            "child": gate(after / "queue-fix-child-unit.log", 155),
            "queue_repeat": gate(after / "queue-fix-repeat.log", 2),
            "camera_repeat": gate(after / "queue-fix-camera-repeat.log", 1),
            "native_vulkan": {**gate(after / "queue-fix-native-vk.log", 2),
                              "native_compilations": modules, "dxc_dxil_loaded": False},
        },
        "post_routing": campaign(before / "full-resolution-canaries-nEW0oN/canaries.json", False),
        "aligned": campaign(after / "canaries/canaries.json", True),
    }
    assert (after / "queue-fix-repeat.log").read_text().count("Passed") == 200
    assert (after / "queue-fix-camera-repeat.log").read_text().count("Passed") == 100
    for key, filename in [("shutdown_red", "fallback-queue-gated-red.log"),
                          ("completion_red", "fallback-queue-completion-red.log")]:
        text = (after / filename).read_text()
        assert "1 tests failed out of 1" in text
        payload["gates"][key] = {"source": source(after / filename), "output": text}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

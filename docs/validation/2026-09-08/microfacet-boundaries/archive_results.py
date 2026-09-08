"""Scope the retained full-scene microfacet A/B/A control accurately."""
import argparse
import json
from pathlib import Path
import sys

REPORTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORTS / "lamp-routing-and-surface"))
from archive_validation import gate
from audit_surface import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads((args.evidence / "audit-unscoped.json").read_text())
    data["schema"] = "psycles.surface-microfacet-boundary-experiment.v1"
    data["intervention"] = (
        "B adds ordinary outline_with_name boundaries to the existing anonymous "
        "microfacet_eval and microfacet_sample host helpers before pool.microfacet; "
        "no noinline/inline override, arithmetic, register limit or compiler policy changes")
    data["decision"] = "reject and fully restore B; no production implementation change retained"
    data["runs"]["ordinary_microfacet_callables"] = data["runs"].pop("keep_boundaries")
    baseline, restored = (data["runs"][key] for key in ("baseline", "restored"))
    assert baseline["code_object_sha256"] == restored["code_object_sha256"]
    changed = data["runs"]["ordinary_microfacet_callables"]
    assert changed["profile"]["shade_surface"]["kernel"] == "kernel_b0154ec50a2a9db0"
    for distribution in ("ggx", "beckmann"):
        for operation in ("eval", "sample"):
            assert f"bsdf_microfacet_{distribution}_{operation}" in changed["functions"]
    assert all(row["frame"] == baseline["frame"] for row in data["runs"].values())
    data["implementation"] = {
        "parent": "6f743862", "documentation_checkpoint": "0086bcae",
        "child": "85e5300f1d85c66cfad7c2962dfe5028d9358980",
        "baseline_and_restored_runtime_sha256": "0a98cba57b82cff4b4466ef91bae5aae62bcba8d4fdeda793d5b65b309fdbef9",
        "intervention_runtime_sha256": "aa4ece4531f9d683c376ed6dc893b446887257b669d3adb497a82905b98d8408",
        "renderer_executable_sha256": "d22d47d63758ca20dacba56b12222613b990f8e851cedf9163fcf887155e61d0",
    }
    data["sources"] = [source(args.evidence / name) for name in (
        "formal-analysis.md", "intervention.patch", "intervention-build.log",
        "restored-build.log", "audit-unscoped.json")]
    data["intervention_gates"] = {
        "host": gate(args.evidence / "intervention-host.log", 165),
        "focused_hip": gate(args.evidence / "intervention-focused-hip.log", 10),
        "scope": "10 focused HIP tests and host gate before B; no full backend gate claimed for the rejected experiment",
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

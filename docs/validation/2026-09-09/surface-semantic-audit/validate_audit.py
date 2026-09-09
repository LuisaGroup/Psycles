"""Check audit integrity and reproduce its image-comparison classifications.

This validates a report containing known renderer failures. A successful
validation is not a successful rendering-conformance gate. --live requires
the captured local sources/images and checks their bytes and observed RGB.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--live", action="store_true")
parser.add_argument("--captured-binary-root", type=Path,
                    help="verify archived binaries after the working build has advanced")
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[3]
report = json.loads((here / "results.json").read_text())
sdk = json.loads((here / "sdk-integration.json").read_text())
assert report["schema"] == "psycles.surface-semantic-audit.v1"
assert sdk["schema"] == "psycles.surface-audit-sdk-integration.v1"
assert sdk["child"] == sdk["canary"]["revisions"]["child"]
assert report["snapshot"]["child"].startswith(sdk["captured_witness_child"])
assert sdk["build"]["exit_status"] == 0
assert [row["total"] for row in sdk["tests"]] == [176, 8, 8, 8]
assert all(row["passed"] == row["total"] and row["exit_status"] == 0 for row in sdk["tests"])
assert sdk["canary"]["actual_all_finite"] and sdk["canary"]["invalid_lanes"] == 0
assert len(sdk["canary"]["channels"]) == 46
assert sdk["canary"]["render"]["command"][4:7] == ["2048", "858", "256"]
rows = report["opcodes"]["rows"]
assert [row["tag"] for row in rows] == list(range(110))
assert len({row["name"] for row in rows}) == 110
assert dict(Counter(row["presence"] for row in rows)) == {
    "implemented": 99, "missing": 9, "sentinel": 2}
assert set(report["scenes"]) == {"barbershop", "monk", "monster", "classroom"}
for name, scene in report["scenes"].items():
    program = scene["program"]
    assert sum(program["domain_totals"]["surface"].values()) == program["surface_serialized_node_count"]
    assert len(program["domain_totals"]["surface"]) == program["surface_opcode_count"]
    assert scene["original_settings"]["settings"]["film_transparent_glass"] is False
    for flag in ("use_holdout", "is_shadow_catcher", "is_caustics_caster", "is_caustics_receiver"):
        assert scene["metadata"]["object_flags"][flag]["instances"] == []

expected = {"boundary-1": True, "boundary-2": False,
            "boundary-2-budget-control": True, "ordinary-emission": True,
            "object-holdout": False, "transparent-control": True, "ray-portal": False}
observed = {}
for witness in report["witnesses"].values():
    assert witness["expected_gate_exit_status"] == 2
    assert all(process["status"] == 0 for process in witness["processes"])
    for row in witness["observations"]:
        assert row["scene"] not in observed
        observed[row["scene"]] = row["combined_rgb_parity"]
        metadata = row["original_render_metadata"]
        assert metadata["cycles_compute_device_type"] == "HIP"
        assert metadata["cycles_device"] == "GPU"
        assert metadata["samples"] == 1
        assert metadata["width"] == metadata["height"] == 16
        assert all(output["all_finite"] for output in row["outputs"].values())
assert observed == expected

for path in here.glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
for link in re.findall(r"\]\(([^)]+)\)", (here / "README.md").read_text()):
    if "://" not in link and not link.startswith("#"):
        assert (here / link.split("#", 1)[0]).resolve().exists(), link

if args.live:
    import numpy as np
    sys.path.insert(0, str(root / "tools"))
    import compare_cycles
    identities = []
    def collect(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                identities.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(report)
    for entry in identities:
        path = Path(entry["path"])
        if args.captured_binary_root and path.is_relative_to(root / "build"):
            path = args.captured_binary_root / path.name
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], path
    native_controls = {}
    for witness in report["witnesses"].values():
        for row in witness["observations"]:
            rgb = {}
            for name, output in row["outputs"].items():
                pixels, channels = compare_cycles._read_image(Path(output["path"]))
                rgb[name] = pixels[:, :, [channels.index(c) for c in output["channels"]]]
            parity = bool(np.isfinite(rgb["original"]).all() and np.isfinite(rgb["actual"]).all()
                          and np.allclose(rgb["original"], rgb["actual"], atol=2e-6, rtol=2e-6))
            assert parity == row["combined_rgb_parity"]
            assert float(np.max(np.abs(rgb["actual"] - rgb["original"]))) == row["combined_rgb_max_absolute_error"]
            native_controls[row["scene"]] = rgb["original"]
    assert np.allclose(native_controls["boundary-2"], native_controls["boundary-2-budget-control"],
                       atol=2e-6, rtol=2e-6)
    print(f"Verified {len(identities)} captured file identities and seven original/actual image pairs.")
    sdk_logs = [sdk["build"]["log"], sdk["canary_runner_log"]]
    sdk_logs.extend(row["log"] for row in sdk["tests"])
    for entry in sdk_logs:
        with Path(entry["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
    for row in sdk["tests"]:
        log = Path(row["log"]["path"]).read_text()
        assert f"100% tests passed out of {row['total']}" in log
    vk_log = Path(sdk["tests"][-1]["log"]["path"]).read_text()
    assert vk_log.count("SPIR-V compilation successful") == sdk["native_vulkan"]["spirv_compilation_success_lines"]
    assert not re.search(r"(?:calling init:|file=|trying file=).*?(?:dxcompiler|dxil)", vk_log, re.I)
    canary = sdk["canary"]
    canary_path = Path(canary["render"]["command"][2]).with_suffix(".exr")
    with canary_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == canary["output_sha256"]
    pixels, channels = compare_cycles._read_image(canary_path)
    assert channels == canary["channels"] and np.isfinite(pixels).all()
    print("Verified six SDK gate logs, strict native Vulkan loader evidence and the finite 46-channel canary.")
print("Audit integrity passed; rendering parity remains four controls agreeing, THREE FAILURES.")

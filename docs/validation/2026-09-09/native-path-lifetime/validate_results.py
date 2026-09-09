"""Validate this captured checkpoint, without treating known scene failures as green.

--live also verifies retained local evidence and captured implementation bytes.
It is intentionally not a gate for a later, different build or source revision.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--live", action="store_true")
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
d = json.loads((HERE / "results.json").read_text())
assert d["schema"] == "psycles.native-path-lifetime-repair.v1"
assert d["snapshot"]["child"] == "03a0f5158b53768abefea555f480f87ee5bc5a1e"
assert d["regression_red"]["exit_status"] == 8
assert [int(r[2]) for r in d["regression_red"]["observations"]] == [0, 0, 768, 768, 0, 0]
assert len(d["regression_green"]["observations"]) == 6
assert all(int(r[2]) == 0 for r in d["regression_green"]["observations"])
assert [r["total"] for r in d["tests"]] == [177, 7, 188, 190, 7]
assert all(r["passed"] == r["total"] and r["exit_status"] == 0 for r in d["tests"])
assert d["native_vulkan"]["spirv_compilation_success_lines"] == 129
assert d["native_vulkan"]["dxc_dxil_loader_matches"] == 0
c = d["canaries"]
assert c["runner_exit_status"] == 2 and c["main_shader_cache"] == "disabled"
assert [(r["scene"], r["repeat"]) for r in c["records"]] == [
    ("barbershop", 1), ("monk", 1), ("monster", 1), ("classroom", 1),
    ("barbershop", 2), ("barbershop", 3)]
assert sum(r["actual_all_finite"] for r in c["records"]) == 5
assert [r["frame"]["bytes"] for r in c["records"]] == [416, 216, 276, 256, 416, 416]
for r in c["records"]:
    assert len(r["passes"]) == 15 and len(r["pass_contract"]["channels"]) == 46
    assert r["render"]["command"][6] == "256"
    assert r["render"]["returncode"] == r["comparison"]["returncode"] == 0
classroom = c["records"][3]
assert {name: r["actual_invalid_pixels"] for name, r in classroom["passes"].items()
        if r["actual_invalid_pixels"]} == {"DiffDir": 7, "GlossDir": 7, "DiffInd": 2, "GlossInd": 2}
interventions = d["image_interventions"]
old_classroom = interventions["intervention"]["records"][3]
assert old_classroom["invalid_lanes_before"] == old_classroom["invalid_lanes_after"] == 18
assert old_classroom["invalid_pixels_after"] == 8
assert old_classroom["nonfinite_classification_unchanged"]
assert d["identical_binary_repeat"]["implementation_sha256"] == d["snapshot"]["implementation_sha256"]
same = d["same_sdk_control"]
assert same["loader_init_paths_verified"] and same["exit_status"] == 0
current = {Path(name).name: digest for name, digest in d["snapshot"]["implementation_sha256"].items()}
old = {Path(r["path"]).name: r["sha256"] for r in same["loaded_implementation"]}
assert set(current) == set(old)
assert [name for name in current if current[name] != old[name]] == ["libpsycles_luisa_runtime.so"]
repeat = interventions["repeat-intervention"]["records"][0]
assert repeat["invalid_lanes_after"] == 0
assert repeat["passes"]["ViewLayer.Normal"]["rmse"] > 0
assert d["qa"]["repeatability_cause"].startswith("unresolved")
for path in HERE.glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
for link in re.findall(r"\]\(([^)]+)\)", (HERE / "README.md").read_text()):
    if "://" not in link and not link.startswith("#"):
        assert (HERE / link.split("#", 1)[0]).resolve().exists(), link

if args.live:
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
    collect(d)
    for name, digest in d["snapshot"]["implementation_sha256"].items():
        identities.append({"path": str(ROOT / name), "sha256": digest})
    for r in c["records"]:
        for operation in ("render", "comparison"):
            identities.append({"path": r[operation]["log"], "sha256": r[operation]["log_sha256"]})
        identities.append({"path": r["source_manifest"], "sha256": r["source_manifest_sha256"]})
        identities.append({"path": str(Path(r["render"]["command"][2]).with_suffix(".exr")),
                           "sha256": r["output_sha256"]})
        reference_path = r["comparison"]["command"][2]
        assert reference_path.endswith(".exr"), reference_path
        identities.append({"path": reference_path, "sha256": r["reference_sha256"]})
    unique = {(i["path"], i["sha256"]) for i in identities}
    for name, expected in unique:
        with Path(name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
    log = Path(d["tests"][-1]["log"]["path"]).read_text()
    assert log.count("SPIR-V compilation successful") == 129
    assert not re.search(r"(?:calling init:|find library=|trying file=).*?(?:dxcompiler|dxil)", log, re.I)
    for key, value in d["native_vulkan"]["environment"].items():
        if key != "LD_DEBUG":
            assert f"{key}={value}" in log
    log = Path(same["log"]["path"]).read_text()
    for provider in same["loaded_implementation"]:
        if provider["path"].endswith(".so"):
            assert re.search(r"calling init: " + re.escape(provider["path"]) + r"(?: \[0\])?\s*$", log, re.M)
    print(f"Verified {len(unique)} retained file identities and strict loader evidence.")
print("Checkpoint integrity passed: native lifetime regression green; full-scene finite gate 5/6, parity/performance/repeatability still open.")

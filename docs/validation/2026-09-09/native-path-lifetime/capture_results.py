"""Archive existing test/render evidence; does not render or compute an oracle.

All captured outputs remain in the evidence directory. JSON is emitted to
stdout for review and apply_patch, not silently installed in the repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("evidence", type=Path)
parser.add_argument("canaries", type=Path)
parser.add_argument("same_sdk_control", type=Path)
parser.add_argument("repeat_control", type=Path)
args = parser.parse_args()


def identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def read(path):
    return json.loads(path.read_text())


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


e = args.evidence
canaries = read(args.canaries / "canaries.json")
same_sdk = read(args.same_sdk_control / "control.json")
repeat = read(args.repeat_control / "control.json")
assert len(canaries["records"]) == 6
assert len({(r["scene"], r["repeat"]) for r in canaries["records"]}) == 6
assert sum(r["actual_all_finite"] for r in canaries["records"]) == 5
for name, digest in canaries["implementation_sha256"].items():
    assert identity(ROOT / name)["sha256"] == digest, name
assert repeat["implementation_sha256"] == canaries["implementation_sha256"]
assert same_sdk["loader_init_paths_verified"]

selection = r"^psycles\.luisa_cycles_(path_lifetime|volume_boundary|volume_emission_film|lamp_routing|zero_bsdf|film_routing|svm_subsurface_exit)_"
tests = []
for name, total, jobs, expression, filename in (
    ("host", 177, 32, ["-E", "_(hip|fallback|vk|cuda|metal|dx)$"], "green-host.log"),
    ("focused-hip", 7, 1, ["-R", selection + "hip$"], "green-focused-hip.log"),
    ("full-hip", 188, 1, ["-R", "_hip$"], "full-hip.log"),
    ("full-fallback", 190, 1, ["-R", "_fallback$"], "full-fallback.log"),
    ("strict-native-vulkan", 7, 1, ["-R", selection + "vk$"], "strict-native-vk.log"),
):
    log = (e / filename).read_text()
    assert f"100% tests passed out of {total}" in log
    elapsed = float(re.search(r"Total Test time \(real\) =\s*([\d.]+) sec", log)[1])
    tests.append({"name": name, "passed": total, "total": total, "exit_status": 0,
                  "wall_seconds": elapsed, "log": identity(e / filename),
                  "replay_command": ["ctest", "--test-dir", str(ROOT / "build"),
                                     "--parallel", str(jobs), "--output-on-failure"] + expression})
vk = (e / "strict-native-vk.log").read_text()
assert not re.search(r"(?:calling init:|find library=|trying file=).*?(?:dxcompiler|dxil)", vk, re.I)
guards = {"LUISA_VULKAN_USE_XIR": "1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV": "1",
          "LUISA_VULKAN_DISABLE_DXC": "1"}
assert all(f"{key}={value}" in vk for key, value in guards.items())
assert vk.count("SPIR-V compilation successful") == 129
red = (e / "red-hip.log").read_text()
assert "0% tests passed, 1 tests failed out of 1" in red
red_rows = re.findall(r"(boundary-\S+) scheduler=(\S+) original RGB lanes=768 mismatches=(\d+)", red)
assert len(red_rows) == 6 and [int(r[2]) for r in red_rows] == [0, 0, 768, 768, 0, 0]
green = (e / "green-focused-hip.log").read_text()
green_rows = re.findall(r"(boundary-\S+) scheduler=(\S+) original RGB lanes=768 mismatches=(\d+)", green)
assert len(green_rows) == 6 and all(int(r[2]) == 0 for r in green_rows)

records = []
jit_links = []
metric_keys = ("rmse", "relative_rmse", "maximum_absolute_error", "actual_invalid_pixels",
               "reference_invalid_pixels", "invalid_pixels", "valid_pixels", "shape")
for r in canaries["records"]:
    assert r["render"]["returncode"] == r["comparison"]["returncode"] == 0
    assert len(r["passes"]) == 15 and len(r["pass_contract"]["channels"]) == 46
    output = Path(r["render"]["command"][2]).with_suffix(".exr")
    assert identity(output)["sha256"] == r["output_sha256"]
    records.append({key: r[key] for key in (
        "scene", "repeat", "socket_metadata_control", "timings", "render", "comparison",
        "source_manifest", "source_manifest_sha256", "output_sha256", "reference_sha256",
        "frame", "pass_contract", "actual_all_finite")})
    records[-1]["passes"] = {name: {key: values[key] for key in metric_keys}
                             for name, values in r["passes"].items()}
    if r["scene"] == "barbershop":
        log = Path(r["render"]["log"]).read_text()
        links = re.findall(r"Linked HIP LLVM bitcode to AMDGPU code object \(861120 bytes\) in ([\d.]+) ms", log)
        assert len(links) == 1
        jit_links.append({"repeat": r["repeat"], "code_object_bytes": 861120,
                          "hiprtc_link_milliseconds": float(links[0])})

interventions = {}
for name in ("intervention", "same-sdk-intervention", "sdk-only-intervention", "repeat-intervention"):
    data = read(e / (name + ".json"))
    assert all(row["nonfinite_classification_unchanged"] for row in data["records"])
    interventions[name] = {"capture": identity(e / (name + ".json")), **data}

source_names = [
    "src/luisa/cycles_integrator_limits.h", "src/luisa/path_kernel_pipeline.cpp",
    "src/luisa/path_tracer_kernel.cpp", "src/luisa/path_tracer_types.h",
    "tests/test_cycles_integrator_limits.cpp", "tests/test_luisa_cycles_lamp_routing.cpp",
    "tests/test_luisa_cycles_path_lifetime.cpp", "tests/test_cycles_path_lifetime_fixture.py",
    "tools/extract_cycles_path_lifetime_fixture.py", "cmake/PsyclesCoreTests.cmake",
    "cmake/PsyclesSurfaceBackendTests.cmake", "tests/data/cycles_path_lifetime/manifest.json",
]
original = Path("/home/mike/Projects/blender-cycles-trace-5.2")
print(json.dumps({
    "schema": "psycles.native-path-lifetime-repair.v1",
    "status": "native lifetime regression repaired; full-scene finiteness and overall parity/performance remain open",
    "snapshot": {"root_base": revision(ROOT), "child": revision(ROOT / "third_party/LuisaCompute"),
                 "source_files": [identity(ROOT / name) for name in source_names],
                 "implementation_sha256": canaries["implementation_sha256"],
                 "original_source": revision(original),
                 "original_inherited_dirty_files": [identity(original / "intern/cycles/scene" / name)
                                                     for name in ("light.cpp", "svm.cpp")]},
    "builds": [{"command": ["cmake", "--build", str(ROOT / "build"), "--parallel", "32"],
                "exit_status": 0, "log": identity(e / name)}
               for name in ("green-build.log", "green-final-build.log")],
    "regression_red": {"exit_status": 8, "log": identity(e / "red-hip.log"),
                       "implementation": "9e3ba165", "sdk": same_sdk["current_sdk"],
                       "observations": red_rows},
    "regression_green": {"observations": green_rows}, "tests": tests,
    "native_vulkan": {"environment": {**guards, "LD_DEBUG": "libs"},
                      "spirv_compilation_success_lines": 129, "dxc_dxil_loader_matches": 0},
    "canaries": {"manifest": identity(args.canaries / "canaries.json"),
                 "runner_log": identity(e / "canaries-run.log"), "runner_exit_status": 2,
                 "reference_scope": canaries["reference_scope"],
                 "main_shader_cache": canaries["main_shader_cache"], "records": records},
    "same_sdk_control": same_sdk, "identical_binary_repeat": repeat,
    "image_interventions": interventions,
    "jit_link_observations": {"scope": "same-sized main code object; first-seen input versus repeats, not a controlled cold-cache benchmark",
                              "records": jit_links,
                              "downstream_cache_mechanism": "not independently verified"},
    "qa": {"unit_green_is_not_full_scene_green": True,
           "sdk_and_runtime_changes_not_confounded_as_one_cause": True,
           "identical_binary_repeat_not_pixel_identical": True,
           "repeatability_cause": "unresolved; not attributed to truncation, RNG, floating-point error or SDK",
           "no_fresh_paired_speedup_claim": True,
           "classroom_invalid_lanes": 18, "classroom_invalid_pixels": 8,
           "classroom_nonfinite_classes_unchanged": True}
}, indent=2))

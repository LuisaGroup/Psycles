"""Archive original-GPU regressions and full-scene observations, not an oracle.

Existing immutable captures are read and checked. Output goes to stdout for
review and apply_patch; this script neither renders nor edits source files.
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
args = parser.parse_args()
e = args.evidence.resolve()


def identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def ctest(name, filename, total, jobs, expression):
    path = e / filename
    log = path.read_text()
    assert f"100% tests passed out of {total}" in log, name
    elapsed = float(re.search(r"Total Test time \(real\) =\s*([\d.]+) sec", log)[1])
    return {"name": name, "passed": total, "total": total, "exit_status": 0,
            "wall_seconds": elapsed, "log": identity(path),
            "command": ["ctest", "--test-dir", str(ROOT / "build"), "--parallel", str(jobs),
                        "--output-on-failure"] + expression}


fixture = read(ROOT / "tests/data/cycles_ray_portal_render/manifest.json")
names = tuple(row["name"] for row in fixture["records"])
assert len(names) == len(set(names)) == 11
expected = {(name, scheduler) for name in names for scheduler in ("megakernel", "wavefront-staged")}
pattern = r"([\w-]+) scheduler=([\w-]+) original RGB lanes=768 mismatches=(\d+)"


def observations(path, keys):
    rows = {}
    for name, scheduler, mismatches in re.findall(pattern, path.read_text()):
        key = (name, scheduler)
        if key in keys:
            # CTest -V repeats a failed test's output. Identical repetitions
            # are one observation, never extra cases or contradictory results.
            value = int(mismatches)
            assert key not in rows or rows[key] == value, key
            rows[key] = value
    assert set(rows) == keys, (path, keys - set(rows))
    return [{"case": key[0], "scheduler": key[1], "rgb_lanes": 768, "mismatches": value}
            for key, value in sorted(rows.items())]


red_keys = {key for key in expected if key[0] in ("transparent-control", "ray-portal")}
red = observations(e / "render-red-hip.log", red_keys)
assert all(r["mismatches"] == (768 if r["case"] == "ray-portal" else 0) for r in red)
assert "1 tests failed out of 1" in (e / "render-red-hip.log").read_text()
green = {}
for backend, filename in (("hip", "full-hip.log"), ("fallback", "full-fallback.log"),
                          ("vk", "strict-native-vk.log")):
    path = e / filename
    rows = observations(path, expected)
    assert all(r["mismatches"] == 0 for r in rows)
    assert "original Cycles portal/path state: 26 cases, 0 mismatches" in path.read_text()
    green[backend] = {"log": identity(path), "state_cases": 26, "render_observations": rows}

selection = r"^psycles\.luisa_cycles_(path_lifetime|volume_boundary|volume_emission_film|lamp_routing|zero_bsdf|film_routing|svm_subsurface_exit|ray_portal_state|ray_portal_render)_vk$"
tests = [ctest("host", "final-host.log", 178, 32, ["-E", "_(hip|fallback|vk|cuda|metal|dx)$"]),
         ctest("hip", "full-hip.log", 190, 1, ["-V", "-R", "_hip$"]),
         ctest("fallback", "full-fallback.log", 192, 1, ["-V", "-R", "_fallback$"]),
         ctest("strict-native-vulkan", "strict-native-vk.log", 9, 1, ["-V", "-R", selection])]
vk = (e / "strict-native-vk.log").read_text()
guards = {"LUISA_VULKAN_USE_XIR": "1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV": "1",
          "LUISA_VULKAN_DISABLE_DXC": "1"}
assert all(f"{key}={value}" in vk for key, value in guards.items())
assert not re.search(r"(?:calling init:|find library=|trying file=).*?(?:dxcompiler|dxil)", vk, re.I)
spirv_count = vk.count("SPIR-V compilation successful")
assert spirv_count > 0
packed = (e / "luisa-packed-hip.log").read_text()
assert "all tests passed" in packed and "14 asserts in 1 tests" in packed
assert "100% tests passed out of 2" in (e / "luisa-upstream-host.log").read_text()

canaries = read(args.canaries / "canaries.json")
assert len(canaries["records"]) == 6
assert {(r["scene"], r["repeat"]) for r in canaries["records"]} == {
    ("barbershop", 1), ("barbershop", 2), ("barbershop", 3),
    ("monk", 1), ("monster", 1), ("classroom", 1)}
for name, digest in canaries["implementation_sha256"].items():
    assert identity(ROOT / name)["sha256"] == digest, name
records = []
for r in canaries["records"]:
    assert r["render"]["returncode"] == r["comparison"]["returncode"] == 0
    assert len(r["passes"]) == 15 and len(r["pass_contract"]["channels"]) == 46
    assert identity(r["source_manifest"])["sha256"] == r["source_manifest_sha256"]
    output = Path(r["render"]["command"][2]).with_suffix(".exr")
    assert identity(output)["sha256"] == r["output_sha256"]
    assert r["actual_all_finite"] == all(p["actual_invalid_pixels"] == 0 for p in r["passes"].values())
    row = {key: r[key] for key in ("scene", "repeat", "timings", "render", "comparison",
           "source_manifest", "source_manifest_sha256", "output_sha256", "reference_sha256",
           "frame", "actual_all_finite")}
    metrics = ("rmse", "relative_rmse", "maximum_absolute_error", "actual_invalid_pixels",
               "reference_invalid_pixels", "invalid_pixels", "valid_pixels", "shape")
    row["passes"] = {name: {key: values[key] for key in metrics} for name, values in r["passes"].items()}
    log = Path(r["render"]["log"]).read_text()
    row["hip_link_observations"] = [{"code_object_bytes": int(size), "milliseconds": float(ms)}
        for size, ms in re.findall(r"Linked HIP LLVM bitcode to AMDGPU code object \((\d+) bytes\) in ([\d.]+) ms", log)]
    records.append(row)

intervention = read(e / "image-intervention.json")
assert {r["scene"] for r in intervention["records"]} == {"barbershop", "monk", "monster", "classroom"}
assert intervention["after_manifest"] == identity(args.canaries / "canaries.json")
source_names = """
cmake/PsyclesCoreTests.cmake
cmake/PsyclesLuisaSources.cmake
cmake/PsyclesSurfaceBackendTests.cmake
include/psycles/luisa/cycles_path_state.h
src/luisa/cycles_svm_bsdf.h
src/luisa/path_kernel_builder.h
src/luisa/path_kernel_direct_light_task.cpp
src/luisa/path_kernel_direct_light_task.h
src/luisa/path_kernel_direct_light_transport.cpp
src/luisa/path_kernel_environment_light.cpp
src/luisa/path_kernel_setup.cpp
src/luisa/path_kernel_subsurface.cpp
src/luisa/path_kernel_surface_geometry.cpp
src/luisa/path_kernel_surface_scatter.cpp
src/luisa/path_kernel_surface_shading.cpp
src/luisa/path_kernel_volume_direct_light.cpp
src/luisa/path_kernel_volume_segment.cpp
src/luisa/path_kernel_volume_shadow.cpp
src/luisa/path_tracer_common.cpp
src/luisa/path_tracer_cycles_svm_light.cpp
src/luisa/path_tracer_cycles_svm_shadow.cpp
src/luisa/path_tracer_cycles_svm_surface.cpp
src/luisa/path_tracer_surfaces.h
src/luisa/path_tracer_types.h
tests/test_luisa_cycles_shadow_queue.cpp
tests/test_luisa_cycles_svm_bsdf_dispatch.cpp
tests/test_luisa_cycles_svm_native_shadow.cpp
tests/test_luisa_direct_lighting_plan.cpp
""".split()
source_names += [
    "src/luisa/cycles_svm_surface_integrator.h", "src/luisa/cycles_svm_surface_integrator.cpp",
    "src/luisa/path_kernel_cycles_svm_surface_scatter.h", "src/luisa/path_kernel_cycles_svm_surface_scatter.cpp",
    "tests/cycles_ray_portal_fixture.h", "tests/test_cycles_ray_portal_fixture.py",
    "tests/test_luisa_cycles_ray_portal_state.cpp", "tests/test_luisa_cycles_ray_portal_render.cpp",
    "tools/cycles_ray_portal_oracle.hip", "tools/create_cycles_ray_portal_scenes.py",
    "tools/capture_cycles_ray_portal.py", "tools/extract_cycles_ray_portal_fixture.py",
    "tests/data/cycles_ray_portal_state.json", "tests/data/cycles_ray_portal_state.txt",
    "tests/data/cycles_ray_portal_render/manifest.json"]
original = Path("/home/mike/Projects/blender-cycles-trace-5.2")
edge_traces = {}
for name in ("original", "hip", "fallback"):
    decoded = read(e / f"trace-{name}.decoded.json")
    edge_traces[name] = {
        "capture": identity(e / f"trace-{name}.decoded.json"),
        "log": identity(e / f"trace-{name}.log"),
        "events": [{"index": event["index"], "state": {key: event["slots"][key]
                     for key in ("state_lobes", "ray_p", "ray_d", "isect_coord", "isect_id", "surface_p")}}
                   for event in decoded["events"] if event["written"]]}
assert len(edge_traces["original"]["events"]) == len(edge_traces["hip"]["events"]) == 1
assert len(edge_traces["fallback"]["events"]) == 4
print(json.dumps({
    "schema": "psycles.native-surface-continuation.v1",
    "status": "portal and native-label continuation repaired; remaining surface obligations and overall performance goal are open",
    "snapshot": {"root_base": revision(ROOT), "child": revision(ROOT / "third_party/LuisaCompute"),
                 "source_files": [identity(ROOT / name) for name in sorted(set(source_names))],
                 "implementation_sha256": canaries["implementation_sha256"],
                 "original_source": revision(original),
                 "original_inherited_dirty_files": [identity(original / "intern/cycles/scene" / name)
                                                     for name in ("light.cpp", "svm.cpp")]},
    "builds": [{"command": ["cmake", "--build", str(ROOT / "build"), "--parallel", "32"],
                "exit_status": 0, "log": identity(e / name)}
               for name in ("upstream-build.log", "final-fixture-build.log")],
    "regression_red": {"log": identity(e / "render-red-hip.log"), "exit_status": 8,
                       "root": "4d1a3b19077627797f5d6adec4afc11a1476891b",
                       "observations": red},
    "regression_green": green, "tests": tests,
    "fixture_correction": {
        "reason": "Position requires a link; former relocation fixtures actually retained sd.P and exposed shared-edge traversal ties",
        "old_full_fallback_log": identity(e / "red-full-fallback.log"),
        "old_full_fallback_result": {"passed": 191, "total": 192, "exit_status": 8},
        "old_manifest": identity(e / "unlinked-position-manifest.json"),
        "pixel_diagnostics": identity(e / "diagnostic-fallback.log"),
        "original_source_observer": "cb168525138fecc792cc393f94afc39582b0103c",
        "edge_traces": edge_traces,
        "original_gpu_boundary_capture": identity(e / "original-portal-edges.txt"),
        "boundary_gpu_test_logs": [identity(e / f"edge-state-{backend}.log") for backend in ("hip", "fallback")],
        "new_original_capture": identity(e / "linked-position-PtuwQX/capture.json"),
        "new_original_capture_processes": 18,
        "runtime_arithmetic_unchanged": True, "tolerance_unchanged": True,
        "general_shared_edge_parity_certified": False},
    "native_vulkan": {"environment": {**guards, "LD_DEBUG": "libs"},
                      "spirv_compilation_success_lines": spirv_count, "dxc_dxil_loader_matches": 0},
    "upstream_coro_regression": {"host_log": identity(e / "luisa-upstream-host.log"),
                                 "host_tests": 2, "hip_log": identity(e / "luisa-packed-hip.log"),
                                 "hip_assertions": 14, "build_log": identity(e / "luisa-upstream-tests-build.log")},
    "canaries": {"manifest": identity(args.canaries / "canaries.json"),
                 "runner_log": identity(e / "canaries-run.log"),
                 "runner_exit_status": 0 if all(r["actual_all_finite"] for r in records) else 2,
                 "reference_scope": canaries["reference_scope"],
                 "main_shader_cache": canaries["main_shader_cache"], "records": records},
    "image_intervention": {"capture": identity(e / "image-intervention.json"), **intervention},
    "qa": {"no_fresh_paired_speedup_claim": True,
           "renderer_and_sdk_changed_since_S1": True,
           "no_exclusive_renderer_causal_timing_claim": True,
           "portal_plus_volume_nee_not_certified": True,
           "shadow_task_aos_bytes": 224, "shadow_task_soa_growth_bytes_per_path": 4,
           "unchanged_opcode_streams_and_static_sizing_algorithm": True,
           "holdout_and_legacy_displacement_still_open": True}
}, indent=2))

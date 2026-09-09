"""Archive completed native Holdout evidence; never render or evaluate shaders.

Output is JSON on stdout for review/apply_patch. Missing evidence is an error,
not a successful default. Suite totals come from logs and CTest selections.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
CHILD = ROOT / "third_party/LuisaCompute"
ORIGINAL = Path("/home/mike/Projects/blender-cycles-trace-5.2")
ORIGINAL_REVISION = "cb168525138fecc792cc393f94afc39582b0103c"
CHILD_REVISION = "6e58928d84604bf0d976f0d6b00f44006b21ceb5"
BACKENDS = r"_(hip|fallback|vk|cuda|metal|dx)$"
VK_SELECTION = (r"^psycles\.luisa_cycles_(path_lifetime|volume_boundary|volume_emission_film|"
                r"lamp_routing|zero_bsdf|film_routing|svm_subsurface_exit|ray_portal_state|"
                r"ray_portal_render|holdout_state|holdout_render)_vk$")
SELECTIONS = {"host": ["-E", BACKENDS], "hip": ["-R", "_hip$"],
              "fallback": ["-R", "_fallback$"], "vk": ["-R", VK_SELECTION]}
LOGS = {"host": "final-host.log", "hip": "full-hip.log",
        "fallback": "full-fallback.log", "vk": "strict-native-vk.log"}
PASSES = ("Combined", "Normal", "DiffCol", "GlossCol", "TransCol", "DiffDir", "DiffInd",
          "GlossDir", "GlossInd", "TransDir", "TransInd", "Emit", "Env",
          "Volume Direct", "Volume Indirect")
CANARY_KEYS = {("barbershop", 1), ("barbershop", 2), ("barbershop", 3),
               ("monk", 1), ("monster", 1), ("classroom", 1)}
BINARIES = ("build/bin/psycles_render_blender_scene", "build/libpsycles_core.so",
            "build/libpsycles_luisa_runtime.so", "build/bin/libluisa-coro.so",
            "build/bin/libluisa-xir.so", "build/bin/libluisa-backend-hip.so")
SOURCES = """
cmake/PsyclesCoreTests.cmake
cmake/PsyclesSurfaceBackendTests.cmake
include/psycles/compiler/core_nodes.h
src/adapter/blender_graph_lower_closures.cpp
src/adapter/cycles_shader_graph.cpp
src/compiler/core_nodes.cpp
src/compiler/cycles_svm_closure_inputs.h
src/compiler/cycles_svm_closure_nodes.cpp
src/compiler/cycles_svm_graph.cpp
src/luisa/cycles_svm.cpp
src/luisa/cycles_svm_closure.cpp
src/luisa/cycles_svm_internal.h
src/luisa/cycles_svm_surface_shader.cpp
src/luisa/cycles_svm_surface_shader.h
src/luisa/path_kernel_surface_shading.cpp
src/luisa/path_tracer_cycles_svm_surface.cpp
tests/cycles_holdout_fixture.h
tests/cycles_holdout_state_fixture.h
tests/test_cycles_holdout_fixture.py
tests/test_cycles_holdout_state_fixture.py
tests/test_cycles_holdout_volume_fixture.py
tests/test_cycles_svm_holdout.cpp
tests/test_cycles_svm_holdout_volume.cpp
tests/test_luisa_cycles_holdout_state.cpp
tests/test_luisa_cycles_holdout_render.cpp
tests/test_luisa_cycles_svm_static_pruning.cpp
tools/capture_cycles_holdout.py
tools/capture_cycles_holdout_volume.py
tools/create_cycles_holdout_scenes.py
tools/create_cycles_holdout_volume_scene.py
tools/cycles_holdout_oracle.hip
tools/extract_cycles_holdout_fixture.py
""".split()


def require(condition, message):
    if not condition:
        raise ValueError(str(message))


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def git(path, *arguments):
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).rstrip("\n")


def deduplicate(rows, key_fields):
    unique = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        require(key not in unique or unique[key] == row, ("conflicting repeated observation", key))
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def words(log, names):
    rows = [{"case": name, "words": int(actual), "original_words": int(original),
             "mismatches": int(mismatch)}
            for name, actual, original, mismatch in re.findall(
                r"([\w-]+) words=(\d+) original=(\d+) mismatches=(\d+)", log) if name in names]
    rows = deduplicate(rows, ("case",))
    require({row["case"] for row in rows} == set(names), "missing compiler observations")
    return rows


def films(log, names):
    rows = [{"case": name, "scheduler": scheduler, "film_lanes": int(lanes),
             "mismatches": int(mismatch)} for name, scheduler, lanes, mismatch in re.findall(
                 r"([\w-]+) scheduler=([\w-]+) original film lanes=(\d+) mismatches=(\d+)", log)
            if name in names]
    rows = deduplicate(rows, ("case", "scheduler"))
    expected = {(name, scheduler) for name in names for scheduler in ("megakernel", "wavefront-staged")}
    require({(r["case"], r["scheduler"]) for r in rows} == expected, "missing film observations")
    require(all(r["film_lanes"] == 16 * 16 * 46 for r in rows), "film lane contract changed")
    return rows


def state(log):
    observed = set(re.findall(r"original Cycles holdout state: (\d+) cases, (\d+) snapshots, (\d+) mismatches", log))
    require(len(observed) == 1, "missing/conflicting holdout state observation")
    cases, snapshots, mismatches = map(int, observed.pop())
    require((cases, snapshots) == (36, 3), "state fixture shape changed")
    return {"cases": cases, "snapshots": snapshots, "mismatches": mismatches}


def ctest_log(path):
    log = Path(path).read_text()
    summaries = re.findall(r"(\d+)% tests passed(?:, (\d+) tests failed)? out of (\d+)", log)
    require(len(summaries) == 1, (path, "missing/multiple CTest terminal summaries"))
    percentage, failures, total = summaries[0]
    failures, total = int(failures or 0), int(total)
    require(total > 0 and 0 <= failures <= total, "invalid CTest counts")
    rows = [{"number": int(number), "name": name, "result": result, "seconds": float(seconds)}
            for number, name, result, seconds in re.findall(
                r"^\s*\d+/\d+\s+Test\s+#(\d+):\s+(\S+)\s+\.{2,}\s*(\S+)\s+([\d.]+)\s+sec", log, re.M)]
    require(len(rows) == len({r["name"] for r in rows}) == total, (path, "missing/duplicate completed tests"))
    require(sum(r["result"] == "Passed" for r in rows) == total - failures, "CTest summary/results disagree")
    require(int(percentage) == int(100 * (total - failures) / total), "CTest percentage disagrees")
    elapsed = re.findall(r"Total Test time \(real\) =\s*([\d.]+) sec", log)
    require(len(elapsed) == 1, "missing/multiple CTest elapsed times")
    return {"passed": total - failures, "failed": failures, "total": total,
            "wall_seconds": float(elapsed[0]), "completed": sorted(rows, key=lambda r: r["name"]),
            "log": identity(path)}


def embedded_identities(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield {"path": value["path"], "sha256": value["sha256"]}
        for child in value.values():
            yield from embedded_identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from embedded_identities(child)


def checked_hashes(mapping, base):
    result = []
    for path, expected in mapping.items():
        actual = identity(Path(base) / path)
        require(actual["sha256"] == expected, (path, "hash differs from original capture"))
        result.append(actual)
    return result


def fixture_provenance():
    surface_path = ROOT / "tests/data/cycles_holdout/manifest.json"
    volume_path = ROOT / "tests/data/cycles_holdout_volume/manifest.json"
    state_path = ROOT / "tests/data/cycles_holdout_state.json"
    surface, volume, boundary = map(read, (surface_path, volume_path, state_path))
    require(surface["word_source"]["revision"] == volume["source"]["revision"] ==
            boundary["source_revision"] == ORIGINAL_REVISION, "original revision mismatch")
    require(len(surface["records"]) == 12 and len(volume["records"]) == 3, "incomplete original scenes")
    require(len(surface["capture"]["processes"]) == 36, "incomplete original process capture")
    require(all(p["status"] == 0 for p in surface["capture"]["processes"] + volume["processes"]),
            "original producer failed")
    require((boundary["rows"], boundary["float_lanes"], boundary["integer_lanes"]) == (36, 132, 24)
            and boundary["preinitialized_slot_sentinels"] and boundary["native_fast_math"], "invalid state provenance")
    ids = [identity(p) for p in (surface_path, volume_path, state_path)]
    for manifest, base in ((surface, surface_path.parent), (volume, volume_path.parent), (boundary, ROOT)):
        ids += checked_hashes(manifest["fixture_sha256"], base)
        ids += list(embedded_identities(manifest))
    ids += checked_hashes(surface["word_source"]["source_sha256"], ORIGINAL)
    ids += checked_hashes(volume["source"]["file_sha256"], ORIGINAL)
    ids += checked_hashes(boundary["original_sha256"], ROOT)
    ids = deduplicate(ids, ("path",))
    for record in ids:
        require(identity(record["path"]) == record, (record["path"], "retained evidence changed"))
    return surface, volume, {"identities": ids, "surface_scenes": 12, "volume_word_images": 3,
                             "state_cases": 36, "state_snapshots": 3}


def canary_observations(path):
    manifest = read(path)
    records = manifest["records"]
    require(len(records) == len(CANARY_KEYS) and
            {(r["scene"], r["repeat"]) for r in records} == CANARY_KEYS, "incomplete/duplicate six-render campaign")
    require(set(manifest["implementation_sha256"]) == set(BINARIES), "binary freeze set changed")
    require(manifest["main_shader_cache"] == "disabled" and
            manifest["reference_scope"] == "retained equal-pass Cycles run-1 images; not fresh timing pairs",
            "cache/reference scope changed")
    output = []
    for row in records:
        require(row["render"]["returncode"] == row["comparison"]["returncode"] == 0, "incomplete render/comparison")
        require(set(row["passes"]) == set(row["pass_contract"]["passes"]) == set(PASSES) and
                len(row["pass_contract"]["channels"]) == 46, "pass contract changed")
        render, comparison = row["render"], row["comparison"]
        require(render["command"][3] == "hip" and render["command"][6] == render["command"][13] == "256",
                "canary backend/sample contract changed")
        expected_finite = all(p["actual_invalid_pixels"] == 0 for p in row["passes"].values())
        require(row["actual_all_finite"] == expected_finite, "finite flag/pass masks disagree")
        ids = checked_hashes({row["source_manifest"]: row["source_manifest_sha256"],
                              str(Path(render["command"][2]).with_suffix(".exr")): row["output_sha256"],
                              comparison["command"][2]: row["reference_sha256"],
                              render["log"]: render["log_sha256"],
                              comparison["log"]: comparison["log_sha256"]}, ROOT)
        report_path = Path(comparison["command"][3])
        require(read(report_path)["passes"] == row["passes"], "comparison/manifest metrics differ")
        ids.append(identity(report_path))
        log = Path(render["log"]).read_text()
        patterns = {"scene_compile_seconds": r"Luisa/[^\n]+ compiled [^\n]+ in ([\d.]+) s",
                    "shader_jit_seconds": r"Luisa shader JIT completed in ([\d.]+) s",
                    "render_seconds": r"Rendered [^\n]+ in ([\d.]+) s:"}
        require({key: float(re.findall(pattern, log)[-1]) for key, pattern in patterns.items()} == row["timings"],
                "canary timing/log mismatch")
        frame = re.search(r"path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+)", log)
        require(frame is not None and dict(zip(("stages", "fields", "bytes"), map(int, frame.groups()))) == row["frame"],
                "canary frame/log mismatch")
        output.append({**row, "identities": ids, "hip_link_observations": [
            {"code_object_bytes": int(size), "milliseconds": float(ms)} for size, ms in re.findall(
                r"Linked HIP LLVM bitcode to AMDGPU code object \((\d+) bytes\) in ([\d.]+) ms", log)]})
    finite = all(row["actual_all_finite"] for row in output)
    require(finite or manifest["nonfinite_policy"] == "record and exit nonzero", "finite failure was not retained")
    return {"manifest": identity(path), "main_shader_cache": manifest["main_shader_cache"],
            "reference_scope": manifest["reference_scope"], "nonfinite_policy": manifest["nonfinite_policy"],
            "finite_gate_passed": finite, "policy_required_runner_exit_status": 0 if finite else 2,
            "records": output}


def capture(evidence, canaries):
    e, c = evidence.resolve(), canaries.resolve()
    surface, volume, provenance = fixture_provenance()
    names = {row["name"] for row in surface["records"]}
    volume_names = {row["name"] for row in volume["records"]}
    red = ctest_log(e / "red-hip-2.log")
    red["words"] = words((e / "red-hip-2.log").read_text(), names)
    red["films"] = films((e / "red-hip-2.log").read_text(), names)
    require(red["failed"] == 2 and sum(r["mismatches"] > 0 for r in red["words"]) == 4 and
            sum(r["mismatches"] > 0 for r in red["films"]) == 20, "pre-fix witness changed")
    allocator_red = ctest_log(e / "state-red-hip.log")
    allocator_red["state"] = state((e / "state-red-hip.log").read_text())
    require(allocator_red["failed"] == 1 and allocator_red["state"]["mismatches"] == 20, "sentinel red changed")
    volume_red = ctest_log(e / "volume-red.log")
    volume_red["words"] = words((e / "volume-red.log").read_text(), volume_names)
    require(volume_red["failed"] == 1 and {r["case"]: r["mismatches"] for r in volume_red["words"]} ==
            {"holdout-only-volume": 0, "holdout-emission-volume": 8, "holdout-shared-domains": 20}, "volume red changed")
    tests = {}
    for backend, filename in LOGS.items():
        test = ctest_log(e / filename)
        require(test["failed"] == 0, (backend, "full validation gate failed"))
        command = ["ctest", "--test-dir", str(ROOT / "build"), "--show-only=json-v1", *SELECTIONS[backend]]
        catalog = json.loads(subprocess.check_output(command, text=True))
        selected = sorted(row["name"] for row in catalog["tests"])
        require(selected == [row["name"] for row in test["completed"]], (backend, "log/catalog selection mismatch"))
        test["selection"] = {"arguments": SELECTIONS[backend], "catalog_command": command, "names": selected}
        log = (e / filename).read_text()
        if backend == "host":
            test["words"] = words(log, names | volume_names)
            require(all(r["mismatches"] == 0 and r["words"] == r["original_words"] for r in test["words"]), "word gate failed")
            pruning = set(re.findall(r"([\w-]+) words=\d+ original=\d+ mismatches=0 surface_holdout=([01]) inert_entries_pruned=2", log))
            require(pruning == {(name, str(int(name.startswith("node-")))) for name in names}, "entry-pruning proof absent")
            test["surface_entry_pruning"] = sorted(pruning)
        else:
            test["films"], test["state"] = films(log, names), state(log)
            require(test["state"]["mismatches"] == 0 and all(r["mismatches"] == 0 for r in test["films"]), "runtime gate failed")
        tests[backend] = test
    vk = (e / LOGS["vk"]).read_text()
    guards = {"LUISA_VULKAN_USE_XIR": "1", "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV": "1", "LUISA_VULKAN_DISABLE_DXC": "1"}
    require(all(f"{key}={value}" in vk for key, value in guards.items()), "native Vulkan guards absent")
    require(re.search(r"(?:calling init:|find library=|trying file=)", vk), "loader audit absent")
    require(not re.search(r"(?:calling init:|find library=|trying file=).*?(?:dxcompiler|dxil)", vk, re.I), "DXC/DXIL loaded")
    spirv = vk.count("SPIR-V compilation successful")
    require(spirv > 0, "native SPIR-V compilation evidence absent")
    build_status = e / "full-build.exit-status"
    require(build_status.read_text().strip() == "0", "all-target build did not finish successfully")
    campaign = canary_observations(c / "canaries.json")
    runner_log, runner_status = e / "canaries-run.log", e / "canaries-run.exit-status"
    runner = {"log": identity(runner_log) if runner_log.exists() else None,
              "exit_status": None, "exit_status_capture": None}
    if runner_status.exists():
        runner["exit_status"] = int(runner_status.read_text().strip())
        runner["exit_status_capture"] = identity(runner_status)
        require(runner["exit_status"] == campaign["policy_required_runner_exit_status"], "runner/policy exit mismatch")
    binaries = checked_hashes(read(c / "canaries.json")["implementation_sha256"], ROOT)
    require(git(CHILD, "rev-parse", "HEAD") == CHILD_REVISION and not git(CHILD, "status", "--porcelain"), "child pin/cleanliness changed")
    require(git(ORIGINAL, "rev-parse", "HEAD") == ORIGINAL_REVISION, "original source revision changed")
    dirty = git(ORIGINAL, "status", "--porcelain").splitlines()
    require(dirty == [" M intern/cycles/scene/light.cpp", " M intern/cycles/scene/svm.cpp"], "original observer dirty set changed")
    intervention_path = e / "image-intervention.json"
    intervention = None
    if intervention_path.exists():
        intervention = read(intervention_path)
        require(intervention["after_manifest"] == identity(c / "canaries.json") and
                {r["scene"] for r in intervention["records"]} == {key[0] for key in CANARY_KEYS}, "intervention campaign mismatch")
        intervention = {"capture": identity(intervention_path), **intervention}
    return {"schema": "psycles.native-holdout.v1", "evidence_directory": str(e), "canaries_directory": str(c),
            "status": "native Holdout validation checkpoint; whole-renderer parity and performance goal remain open",
            "snapshot": {"root_base": git(ROOT, "rev-parse", "HEAD"), "child": CHILD_REVISION,
                         "child_clean": True, "original_source": ORIGINAL_REVISION, "original_dirty_status": dirty,
                         "source_files": [identity(ROOT / name) for name in SOURCES], "binaries": binaries},
            "build": {"command": ["cmake", "--build", str(ROOT / "build"), "--parallel", "32"],
                      "log": identity(e / "full-build.log"), "exit_status": 0, "exit_status_capture": identity(build_status)},
            "original_provenance": provenance, "regression_red": {"surface": red, "allocator": allocator_red, "volume": volume_red},
            "tests": tests, "native_vulkan": {"environment": {**guards, "LD_DEBUG": "libs"},
                "spirv_compilation_success_lines": spirv, "dxc_dxil_loader_matches": 0},
            "canaries": campaign, "canary_runner": runner, "image_intervention": intervention,
            "qa": {"no_fresh_paired_speedup_claim": True, "no_exclusive_causal_timing_claim": True,
                   "no_shader_math_or_rendering_in_capture": True, "ctest_repetitions_deduplicated": True,
                   "complete_46_channel_finite_gate_separate_from_rgb_xyz_pass_metrics": True,
                   "whole_renderer_goal_complete": False}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("canaries", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(capture(args.evidence, args.canaries), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, IndexError) as error:
        sys.exit(f"Cannot archive incomplete or inconsistent Holdout evidence: {error}")

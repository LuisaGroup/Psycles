"""Publish compact, source-backed audit evidence; never evaluate a shader.

The full mechanical inventory remains separately hashed. Per-shader detail
and repeated source mentions are omitted here, not normalized or interpreted
as equivalent words. Semantic findings live in the reviewed Markdown report.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("cycles", type=Path, help="original intern/cycles")
parser.add_argument("evidence", type=Path)
args = parser.parse_args()

def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}

def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

inventory_path = args.evidence / "inventory.json"
inventory = json.loads(inventory_path.read_text())
result = {"schema": "psycles.surface-semantic-audit.v1",
          "scope": "source/word inventory and original-HIP integration witnesses; no production fix or new performance claim",
          "snapshot": {"root": revision(args.root),
                       "child": revision(args.root / "third_party/LuisaCompute"),
                       "original": revision(args.cycles)},
          "inventory": identity(inventory_path),
          "opcodes": {key: inventory["opcodes"][key] for key in ("scope", "summary", "sources")},
          "projections": inventory["projections"], "scenes": {}, "witnesses": {}, "sources": []}
result["opcodes"]["rows"] = [{key: row[key] for key in
    ("tag", "name", "presence", "actual_dispatch", "original_dispatch")}
    for row in inventory["opcodes"]["rows"]]
for name, scene in inventory["scenes"].items():
    program = scene["program"]
    result["scenes"][name] = {"metadata": scene["metadata"],
        "compilation_log": scene["compilation_log"],
        "program": {key: value for key, value in program.items() if key != "shaders"}}
    result["scenes"][name]["program"]["shader_count"] = len(program["shaders"])
    result["scenes"][name]["program"]["surface_opcode_count"] = len(program["domain_totals"]["surface"])
    result["scenes"][name]["program"]["surface_serialized_node_count"] = sum(program["domain_totals"]["surface"].values())
    result["scenes"][name]["bindings"] = identity(args.evidence / (name + ".bindings"))
settings_log = args.evidence / "original-scene-settings.log"
result["original_settings_source"] = identity(settings_log)
settings_count = 0
for line in settings_log.read_text().splitlines():
    if line.startswith("SURFACE_AUDIT_SETTINGS "):
        row = json.loads(line.split(" ", 1)[1])
        result["scenes"][row["scene"]]["original_settings"] = row
        settings_count += 1
assert settings_count == 4

# Each final runner prints process progress followed by one complete JSON
# object. Exit 2 means the RGB parity gate is red, not an execution failure.
for label in ("boundary-gate-expanded", "integration-gate"):
    log = args.evidence / (label + "-run.log")
    text = log.read_text()
    data = json.loads(text[text.index("{\n"):])
    assert all(p["status"] == 0 for p in data["processes"])
    data["parity_summary"] = dict(Counter(row["combined_rgb_parity"] for row in data["observations"]))
    data["runner_log"] = identity(log)
    data["expected_gate_exit_status"] = 0 if all(row["combined_rgb_parity"] for row in data["observations"]) else 2
    for row in data["observations"]:
        original = Path(row["outputs"]["original"]["path"]).with_suffix(".json")
        metadata = json.loads(original.read_text())
        assert metadata["cycles_compute_device_type"] == "HIP"
        assert metadata["cycles_device"] == "GPU"
        row["original_render_metadata"] = {
            "source": identity(original),
            **{key: metadata[key] for key in (
                "blender_build", "cycles_device", "cycles_compute_device_type",
                "cycles_enabled_devices", "samples", "width", "height",
                "adaptive_sampling", "denoising", "sampling_pattern", "effective_seed")}}
    result["witnesses"][label] = data

# The source list bounds the reviewed consumers beyond interpreter handlers.
actual_paths = [
    "src/luisa/cycles_svm.cpp", "src/luisa/cycles_svm_stack.cpp",
    "src/luisa/cycles_svm_closure.cpp", "src/luisa/cycles_svm_bsdf.cpp",
    "src/luisa/cycles_svm_surface_shader.cpp", "src/luisa/cycles_svm_ray_portal.cpp",
    "src/luisa/path_tracer_cycles_svm_surface.cpp", "src/luisa/path_tracer_cycles_svm_shader_data.cpp",
    "src/luisa/path_tracer_cycles_svm_kernel_globals.h", "src/luisa/path_tracer_cycles_svm_kernel_globals.cpp",
    "src/luisa/path_kernel_cycles_svm_surface_geometry.cpp", "src/luisa/path_kernel_surface_geometry.cpp",
    "src/luisa/path_kernel_surface_shading.cpp", "src/luisa/path_kernel_surface_scatter.cpp",
    "src/luisa/path_kernel_surface_emission.cpp", "src/luisa/path_kernel_direct_light_transport.cpp",
    "src/luisa/path_kernel_bounce_random.cpp", "src/luisa/path_kernel_pipeline.cpp",
    "src/luisa/path_kernel_executor.cpp", "src/luisa/cycles_integrator_limits.h",
    "src/luisa/path_tracer_scene.cpp", "src/luisa/path_tracer_surfaces.cpp",
    "src/luisa/path_tracer_displacement_scene.cpp", "src/luisa/path_tracer_cycles_svm_object.cpp",
    "src/compiler/cycles_svm_object_scene.cpp", "include/psycles/luisa/cycles_svm.h",
    "include/psycles/luisa/cycles_closure.h", "include/psycles/luisa/cycles_path_state.h",
    "include/psycles/luisa/cycles_volume_boundary.h", "include/psycles/luisa/surface.h",
    "tools/export_psycles_scene.py", "tools/blender_cycles_object.py", "tools/blender_image_export.py",
    "tests/test_luisa_cycles_svm_standalone_ray_portal.cpp", "tests/test_cycles_integrator_limits.cpp",
    "third_party/LuisaCompute/src/backends/hip/llvm_codegen/hip_codegen_llvm_impl_resource.cpp"]
original_paths = [
    "kernel/svm/svm.h", "kernel/svm/closure.h", "kernel/svm/bump.h",
    "kernel/svm/value.h", "kernel/closure/alloc.h", "kernel/integrator/shade_surface.h",
    "kernel/integrator/surface_shader.h", "kernel/integrator/path_state.h",
    "kernel/film/data_passes.h", "kernel/film/light_passes.h", "kernel/geom/object.h",
    "kernel/device/gpu/image.h", "scene/background.cpp", "scene/integrator.cpp",
    "scene/svm.cpp", "scene/light.cpp", "scene/shader_graph.cpp", "scene/shader_nodes.cpp"]
for base, paths in ((args.root, actual_paths), (args.cycles, original_paths)):
    result["sources"].extend(identity(base / path) for path in paths)
result["scripts"] = [identity(args.evidence / name) for name in
    ("audit_sources.py", "read_blend_settings.py", "create_boundary_scenes.py",
     "create_surface_scenes.py", "run_boundary_probe.py", "archive_audit.py")]
result["implementation_files_at_archive"] = [identity(args.root / name) for name in (
    "build/bin/psycles_render_blender_scene", "build/libpsycles_core.so",
    "build/libpsycles_luisa_runtime.so", "build/bin/libluisa-coro.so",
    "build/bin/libluisa-xir.so", "build/bin/libluisa-backend-hip.so")]
result["remaining_claim_limits"] = [
    "99 case bodies present is not 99 semantically proven handlers",
    "Serialized node counts and static machine instruction sites are not executed work",
    "No new four-scene timing or speedup is measured by this audit",
    "Scene admission is explicitly tested only for the six integration witnesses",
    "No proof of global RNG or every residual indirect path is established",
    "Unsupported opcode, disabled scene feature, and missing production consumer are distinct"]
print(json.dumps(result, indent=2))

"""Audit descriptor experiments separately from original-scene profiling."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import render_pass_contract
import compare_cycles
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("evidence", type=Path)
parser.add_argument("--summary-only", action="store_true")
args = parser.parse_args()
audit = args.evidence
evidence = audit.parent

def source(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

names = ["indirect_compact", "dense_image_plane", "combined_64_byte",
         "native_tex2d", "device_prefix_tex2d", "native_prefix_raw_sample",
         "explicit_index_waterfall"]
experiments = []
for prefix, layouts in [("layout-global", 6), ("layout-global-repeat", 6),
                         ("layout-waterfall", 7)]:
    csv_path = audit / f"{prefix}-timings.csv"
    validation_path = audit / f"{prefix}-validation.log"
    raw = list(csv.DictReader(csv_path.open()))
    assert len(raw) == 3 * layouts * 9
    seen = set()
    groups = defaultdict(list)
    for row in raw:
        pattern, round_, order, layout = [int(row[k]) for k in
                                         ("pattern", "round", "order", "layout")]
        value = float(row["milliseconds"])
        assert 0 <= pattern < 3 and 0 <= round_ < 9 and 0 <= order < layouts
        assert layout == (round_ + order) % layouts and math.isfinite(value) and value > 0
        assert (pattern, round_, layout) not in seen
        seen.add((pattern, round_, layout))
        groups[pattern, layout].append(value)
    validations = re.findall(r"validated pattern=(\d+) layout=(\d+) rgba_lanes=(\d+) max_abs=(\S+)",
                             validation_path.read_text())
    assert len(validations) == 3 * (layouts - 1)
    assert {(int(p), int(l)) for p, l, _, _ in validations} == {
        (p, l) for p in range(3) for l in range(layouts) if l != 3}
    assert all(int(n) == 4 * 1048576 and float(error) == 0 for _, _, n, error in validations)
    summary = []
    for (pattern, layout), values in sorted(groups.items()):
        assert len(values) == 9
        median = statistics.median(values)
        baseline = statistics.median(groups[pattern, 0])
        summary.append({"pattern": pattern, "layout": names[layout], "count": len(values),
                        "median_ms": median, "min_ms": min(values), "max_ms": max(values),
                        "time_ratio_to_indirect": median / baseline})
    experiments.append({"name": prefix, "summary": summary, "raw_rows": raw,
                        "validated_rgba_components": 3 * (layouts - 1) * 4 * 1048576,
                        "sources": [source(csv_path), source(validation_path)]})

profile = evidence / "barbershop-bound-sampler-kernel-profile"
log_path = profile.with_suffix(".log")
log = log_path.read_text()
mapping = dict(re.findall(r"stage='[^/]+/([^']+)' structural_hash=([0-9a-f]+)", log))
stats_path = profile / "psycles_kernel_stats.csv"
trace_path = profile / "psycles_kernel_trace.csv"
stats = {row["Name"]: row for row in csv.DictReader(stats_path.open())}
trace = list(csv.DictReader(trace_path.open()))
baseline_path = ROOT / "docs/validation/2026-09-08/matched-pass-hip/barbershop-kernel-profile.json"
baseline = json.loads(baseline_path.read_text())
stages = {}
for stage in ("shade_surface", "shade_volume", "intersect_closest"):
    kernel = "kernel_" + mapping[stage]
    rows = [r for r in trace if r["Kernel_Name"] == kernel]
    duration = sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in rows)
    assert duration == int(stats[kernel]["TotalDurationNs"])
    assert len(rows) == int(stats[kernel]["Calls"])
    stages[stage] = {"kernel": kernel, "calls": len(rows), "duration_ns": duration,
                     "duration_seconds": duration / 1e9,
                     "vgpr_counts": sorted({int(r["VGPR_Count"]) for r in rows}),
                     "scratch_bytes": sorted({int(r["Scratch_Size"]) for r in rows}),
                     "retained_cycles": baseline["engines"]["cycles"]["stages"][stage],
                     "prior_psycles": baseline["engines"]["psycles"]["stages"][stage]}
canary_path = evidence / "bound-sampler-canaries.json"
canary = json.loads(canary_path.read_text())
for relative, sha in canary["implementation_sha256"].items():
    assert source(ROOT / relative)["sha256"] == sha
command = list(canary["records"][0]["render"]["command"])
assert command[6] == command[13] == "256"
command[2] = str(profile.with_suffix(".ppm"))
command[6] = command[13] = "64"
render_seconds = float(re.search(r"Rendered 2048x858 at 64 spp .* in ([\d.]+) s:", log)[1])
comparison_path = evidence / "barbershop-bound-sampler-profile-comparison.json"
comparison = json.loads(comparison_path.read_text())
assert len(comparison["passes"]) == 15
assert all(p["actual_invalid_pixels"] == p["reference_invalid_pixels"] == 0
           for p in comparison["passes"].values())
pixels, channels = compare_cycles._read_image(profile.with_suffix(".exr"))
assert len(channels) == 46 and np.isfinite(pixels).all()
del pixels
result = {
    "schema": "psycles.hip-texture-descriptor-audit.v1",
    "production_renderer_revision": "e2d38fc0",
    "test_revision": "251fc767",
    "luisa_revision": "9ea3b720f",
    "experiment_scope": "Synthetic HIP descriptor-layout kernels, not renderer speedups or a CPU oracle",
    "parameters": {"lanes": 1048576, "dependent_samples_per_lane": 16,
                   "descriptor_count": 4096, "actual_images": 256,
                   "image_extent": [64, 64], "filter": "linear", "address_modes": 4,
                   "pattern_names": ["wave_coherent", "neighboring_per_lane", "hashed_per_lane"],
                   "fast_math": True, "warmups_per_layout": 2, "timed_repeats": 9},
    "experiments": experiments,
    "allocation_report": (audit / "allocation-result.txt").read_text(),
    "production_profile": {"extent": [2048, 858], "samples": 64,
        "renderer_command": command,
        "profiler_prefix": ["env", "PSYCLES_DISABLE_SHADER_CACHE=1", "LUISA_CORO_SHADER_MAP=1",
            "/opt/rocm/bin/rocprofv3", "--kernel-trace", "--stats", "--output-format", "csv",
            "--output-directory", str(profile), "--output-file", "psycles", "--"],
        "render_wall_seconds_instrumented": render_seconds,
        "stages": stages,
        "pass_contract": render_pass_contract.inspect_image(profile.with_suffix(".exr")),
        "all_actual_channels_finite": True,
        "comparison": comparison,
        "scope": "Single instrumented follow-up against retained equal-pass original Cycles profile; calls are not active paths",
        "implementation_sha256": canary["implementation_sha256"],
        "sources": [source(p) for p in [stats_path, trace_path, log_path,
                                        profile.with_suffix(".exr"), baseline_path, canary_path,
                                        comparison_path,
                                        evidence / "barbershop-matched-kernel-profile-cycles.json",
                                        Path(canary["records"][0]["render"]["command"][1]) / "scene.json"]]},
    "sources": [source(p) for p in [
        ROOT / "tools/hip_texture_descriptor_probe.hip",
        ROOT / "tools/hip_texture_allocation_probe.hip",
        ROOT / "tools/cycles_svm_image_binding_oracle.hip",
        audit / "texture-layout-probe-global.hip",
        audit / "texture-layout-probe-global",
        audit / "texture-layout-waterfall-probe",
        audit / "texture-allocation-probe",
        audit / "allocation-result.txt",
        audit / "hip_kernel_final_0.ll",
        audit / "hip_isa_0.co",
        audit / "cycles-image-oracle.ll",
        audit / "cycles-image-oracle.s",
        audit / "layout-probe-global.s",
        audit / "layout-probe-waterfall.s"]]}
if args.summary_only:
    for experiment in result["experiments"]:
        del experiment["raw_rows"]  # Lossless rows are separately committed CSVs.
print(json.dumps(result, indent=2))

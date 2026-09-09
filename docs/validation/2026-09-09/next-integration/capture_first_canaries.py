"""Capture two completed image canaries; never compile or invoke a renderer.

The input plan contains recorded renderer/comparison commands. Comparisons must
already have completed and have separately recorded exit-status files. All 46
image channels, including Combined alpha, are checked independently of the
RGB/XYZ-only comparison metrics. Historical captures are read, never rewritten.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import compare_cycles as compare
import render_pass_contract as contract
import run_scene_benchmark as benchmark


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(token):
        raise ValueError(f"non-standard JSON number {token} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def finite_profile(pixels, channels):
    finite = np.isfinite(pixels)
    invalid_channels = np.count_nonzero(~finite, axis=(0, 1))
    invalid_pixels = np.any(~finite, axis=2)
    return {
        "shape": list(pixels.shape), "scalar_count": int(pixels.size),
        "pixel_count": int(invalid_pixels.size),
        "nonfinite_scalars": int(np.count_nonzero(~finite)),
        "nonfinite_pixels": int(np.count_nonzero(invalid_pixels)),
        "nonfinite_by_channel": dict(zip(channels, map(int, invalid_channels))),
        "finite_lane_mask_sha256": hashlib.sha256(np.packbits(finite).tobytes()).hexdigest(),
        "invalid_pixel_mask_sha256": hashlib.sha256(np.packbits(invalid_pixels).tobytes()).hexdigest(),
        "mask_encoding": "numpy.packbits, C order, big bit order; shape and channel order recorded",
    }


def pass_invalid(pixels):
    return int(np.count_nonzero(~np.all(np.isfinite(pixels), axis=2)))


def one_match(pattern, text, name):
    matches = re.findall(pattern, text, re.MULTILINE)
    require(len(matches) == 1, f"expected one {name} marker, found {len(matches)}")
    return matches[0]


def capture_scene(integration, planned, historical):
    name = planned["scene"]
    render_log = Path(planned["render_log"])
    text = render_log.read_text()
    timings = benchmark._parse_psycles_timings(text)
    for key, pattern in benchmark._PSYCLES_TIMING_PATTERNS.items():
        require(len(pattern.findall(text)) == 1, f"ambiguous {key} in {name}")
    require(all(math.isfinite(x) and x >= 0 for x in timings.values()), "invalid render timings")
    extent = one_match(r"^Rendered (\d+)x(\d+) at (\d+) spp from absolute sample range "
                       r"\[(\d+), (\d+)\) of (\d+) in ", text, "sample interval")
    require(tuple(map(int, extent)) == (planned["width"], planned["height"], 256, 0, 256, 256),
            "rendered extent/sample interval differs from the matched plan")
    frame = one_match(r"path coroutine: subroutines=(\d+) frame_fields=(\d+) frame_bytes=(\d+) "
                      r"capacity=(\d+)", text, "coroutine frame")
    report_path = Path(planned["comparison_command"][3])
    comparison_exit_path = integration / f"{name}-next-1-compare.exit-status"
    comparison_exit = int(comparison_exit_path.read_text().strip())
    require(comparison_exit == 0, f"comparison failed for {name}")
    report = read(report_path)
    require(report["schema"] == "psycles.cycles-differential.v2", "unexpected comparison schema")
    require(report["build_identity"]["verified"], "comparison lacks matching Blender build metadata")
    require(set(report["passes"]) == set(contract.PASSES), "comparison pass subset or extra pass")

    fresh_path = Path(planned["render_command"][2]).with_suffix(".exr")
    old_path = Path(historical["render"]["command"][2]).with_suffix(".exr")
    reference_path = Path(planned["reference_exr"])
    images = {}
    records = {}
    for key, path in (("fresh", fresh_path), ("old_s3", old_path), ("cycles", reference_path)):
        workload = contract.inspect_image(path)
        require((workload["width"], workload["height"]) == (planned["width"], planned["height"]),
                f"{key} extent mismatch")
        pixels, channels = compare._read_image(path)
        require(pixels.shape == (planned["height"], planned["width"], 46), f"{key} image shape mismatch")
        records[key] = {"file": identity(path), "pass_contract": workload,
                        "finite": finite_profile(pixels, channels)}
        images[key] = (pixels, channels)

    fresh, fresh_channels = images["fresh"]
    old, old_channels = images["old_s3"]
    reference, reference_channels = images["cycles"]
    require(fresh_channels == old_channels, "old/new channel order changed; explicit remapping required")
    old_vs_new = {}
    independent_checks = {}
    for pass_name in contract.PASSES:
        actual = fresh[:, :, compare._find_cycles_channels(fresh_channels, pass_name)]
        previous = old[:, :, compare._find_cycles_channels(old_channels, pass_name)]
        original = reference[:, :, compare._find_cycles_channels(reference_channels, pass_name)]
        metrics = report["passes"][pass_name]
        valid = np.all(np.isfinite(actual), axis=2) & np.all(np.isfinite(original), axis=2)
        require(metrics["actual_invalid_pixels"] == pass_invalid(actual), "actual finite count differs")
        require(metrics["reference_invalid_pixels"] == pass_invalid(original), "reference finite count differs")
        require(metrics["valid_pixels"] == int(np.count_nonzero(valid)), "comparison denominator differs")
        require(metrics["invalid_pixels"] + metrics["valid_pixels"] == planned["width"] * planned["height"],
                "comparison does not partition every pixel")
        old_vs_new[pass_name] = compare._metrics(previous, actual)
        if pass_name in ("Combined", "DiffInd", "GlossInd", "Normal"):
            # Independent aggregate check, not renderer arithmetic or a new
            # image oracle. Float64 accumulation avoids an analysis overflow.
            delta = actual[valid].astype(np.float64) - original[valid].astype(np.float64)
            independent_checks[pass_name] = {
                "reported_float32_rmse": metrics["rmse"],
                "independent_float64_rmse": float(np.sqrt(np.mean(delta * delta))),
                "valid_pixels": int(np.count_nonzero(valid)),
            }
    alpha = [i for i, channel in enumerate(fresh_channels) if channel.endswith(".Combined.A")]
    require(len(alpha) == 1, "alpha is missing or ambiguous")
    old_vs_new["Combined Alpha"] = compare._metrics(old[:, :, alpha], fresh[:, :, alpha])
    fresh_finite = records["fresh"]["finite"]["nonfinite_scalars"] == 0
    require(fresh_finite == all(x["actual_invalid_pixels"] == 0 for x in report["passes"].values())
            or not fresh_finite, "finite gate disagreement")
    return {
        "scene": name, "repeat": 1, "samples": 256,
        "render_command": planned["render_command"], "render_log": identity(render_log),
        "render_exit_code": 0,
        "render_exit_provenance": "Parent agent confirmed both process handles terminated with exit 0 before CPU release",
        "comparison_command": planned["comparison_command"],
        "comparison_exit_code": comparison_exit, "comparison_exit_file": identity(comparison_exit_path),
        "comparison_report": identity(report_path), "comparison_log": identity(planned["comparison_log"]),
        "timings": timings,
        "frame": dict(zip(("stages", "fields", "bytes", "capacity"), map(int, frame))),
        "hip_shader_cache_load_markers": text.count("Loaded HIP shader '"),
        "performance_scope": "single new observation; original retained Cycles and S3 results are not freshly paired",
        "images": records, "all_46_channels_finite": fresh_finite,
        "old_new_finite_lane_masks_equal": bool(np.array_equal(np.isfinite(old), np.isfinite(fresh))),
        "old_vs_new_pass_metrics": old_vs_new,
        "cycles_pass_metrics": report["passes"],
        "independent_rmse_checks": independent_checks,
        "build_identity": report["build_identity"],
        "historical_s3_timings": historical["timings"],
        "historical_s3_frame": historical["frame"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("integration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an existing capture")
    integration = args.integration.resolve()
    plan_path = integration / "validation-plan.json"
    plan = read(plan_path)
    old_path = Path(plan["historical_canaries"]["manifest"])
    old = read(old_path)
    binaries = {}
    for line in (integration / "binary-hashes.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        binaries[name] = identity(ROOT / name)
        require(binaries[name]["sha256"] == digest, f"binary changed after build: {name}")
    sdk = subprocess.check_output(["git", "-C", str(ROOT / "third_party/LuisaCompute"),
                                   "rev-parse", "HEAD"], text=True).strip()
    require(sdk == plan["identity"]["sdk_head"], "SDK revision changed")
    records = []
    for planned in plan["proposed_first_canaries"]:
        historical = next(x for x in old["records"] if x["scene"] == planned["scene"] and x["repeat"] == 1)
        records.append(capture_scene(integration, planned, historical))
        print(planned["scene"], records[-1]["timings"],
              "nonfinite scalars", records[-1]["images"]["fresh"]["finite"]["nonfinite_scalars"], flush=True)
    result = {
        "schema": "psycles.next-integration.first-canaries.v1",
        "scope": "two completed HIP 256-spp canaries, not a full multi-scene or all-backend checkpoint",
        "sdk_revision": sdk, "plan": identity(plan_path), "historical_manifest": identity(old_path),
        "capture_source": identity(__file__), "binary_hashes_file": identity(integration / "binary-hashes.txt"),
        "binaries": binaries, "records": records,
        "first_canary_finite_gate_passed": all(x["all_46_channels_finite"] for x in records),
        "comparison_completion_is_not_correctness_acceptance": True,
        "full_checkpoint_complete": False,
        "no_renderer_or_backend_invoked_by_capture": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

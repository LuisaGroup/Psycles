"""Audit shared-case full-scene controls using final ELF function extents."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/validation/2026-09-08/lamp-routing-and-surface"))
from audit_inline_experiment import inspect
from audit_surface import source

sys.path.insert(0, str(ROOT / "tools"))
import compare_cycles as compare
import render_pass_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("before", "translator_only", "grouped_first", "grouped_repeat", "restored"):
        parser.add_argument(label, type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    directories = {name: path for name, path in vars(args).items() if name != "output"}
    rows = {name: inspect(path) for name, path in directories.items()}
    baseline, channels = compare._read_image(args.before / "psycles.exr")
    render_pass_contract.validate_channels(channels)
    assert baseline.shape == (858, 2048, 46) and np.isfinite(baseline).all()
    for name, path in directories.items():
        row = rows[name]
        row["surface_text"] = source(path / "surface.text")
        row["session_init_seconds"] = float(re.search(
            r"Luisa shader JIT completed in ([\d.]+) s", (path / "render.log").read_text())[1])
        row["total_function_bytes"] = sum(f["bytes"] for f in row["functions"].values())
        row["total_static_instructions"] = sum(f["static_instructions"] for f in row["functions"].values())
        assert row["frame"] == {"stages": 6, "fields": 93, "bytes": 416}
        pixels, actual_channels = ((baseline, channels) if name == "before"
                                   else compare._read_image(path / "psycles.exr"))
        render_pass_contract.validate_channels(actual_channels)
        assert pixels.shape == baseline.shape and np.isfinite(pixels).all()
        passes = {}
        for render_pass in render_pass_contract.PASSES:
            expected = baseline[:, :, compare._find_cycles_channels(channels, render_pass)]
            actual = pixels[:, :, compare._find_cycles_channels(actual_channels, render_pass)]
            delta = actual.astype(np.float64) - expected
            rms = np.sqrt(np.mean(expected.astype(np.float64) ** 2))
            passes[render_pass] = {
                "relative_rmse": float(np.sqrt(np.mean(delta * delta)) / max(rms, 1e-20)),
                "maximum_absolute_error": float(np.max(np.abs(delta))),
                "different_pixels": int(np.count_nonzero(np.any(delta != 0, axis=2))),
            }
        row["image_control"] = {"shape": list(pixels.shape), "all_channels_finite": True,
                                "passes_vs_before": passes}
        if name != "before":
            del pixels
    assert rows["before"]["surface_text"]["sha256"] == rows["translator_only"]["surface_text"]["sha256"]
    assert rows["before"]["surface_text"]["sha256"] == rows["restored"]["surface_text"]["sha256"]
    assert rows["grouped_first"]["surface_text"]["sha256"] == rows["grouped_repeat"]["surface_text"]["sha256"]
    before = [rows[n] for n in ("before", "restored")]
    grouped = [rows[n] for n in ("grouped_first", "grouped_repeat")]
    def medians(records):
        return {"render_seconds": statistics.median(r["render_wall_seconds"] for r in records),
                "surface_seconds": statistics.median(r["profile"]["shade_surface"]["gpu_seconds"] for r in records)}
    a, b = medians(before), medians(grouped)
    result = {
        "schema": "psycles.shared-switch-case-profiles.v1",
        "scope": "sequential full Barbershop 2048x858/64 spp/seed 0; A/control/B/B/A; no overlapping heavy work; not fresh Cycles pairs",
        "intervention": "preserve several switch labels around one body through Luisa AST/XIR and emit the original Cycles BSDF groups; no inlining policy, arithmetic or register-limit changes",
        "isa_scope": "ELF STT_FUNC extents only, excluding alignment padding; static sites, not dynamically executed instruction or spill counts",
        "image_scope": "all 15 passes/46 finite channels against A as intervention controls; these are not a CPU shader oracle or a substitute for original Cycles comparisons",
        "jit_scope": "session initialization includes setup/baking; main shader cache disabled but downstream caches are not cleared; not matched cold-JIT measurements",
        "runs": rows,
        "medians": {"before": a, "grouped": b,
                    "grouped_over_before": {key: b[key] / a[key] for key in a}},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

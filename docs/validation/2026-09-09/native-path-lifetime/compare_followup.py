"""Compare captured before/after images without synthesizing expected pixels.

This is an intervention check, not a replacement for the original Cycles
comparisons or the full-scene finite-value gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import compare_cycles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
parser.add_argument("--scenes", nargs="+", default=["barbershop", "monk", "monster", "classroom"])
parser.add_argument("--before-manifest", type=Path)
parser.add_argument("--after-manifest", type=Path)
args = parser.parse_args()

def identity(path):
    with path.open("rb") as stream:
        return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}

records = []
for scene in args.scenes:
    before_path = args.before / (scene + "-fixed-1.exr")
    after_path = args.after / (scene + "-fixed-1.exr")
    before, channels = compare_cycles._read_image(before_path)
    after, actual_channels = compare_cycles._read_image(after_path)
    assert channels == actual_channels and len(channels) == 46
    assert before.shape == after.shape
    invalid = ~np.isfinite(after)
    row = {"scene": scene, "before": identity(before_path), "after": identity(after_path),
           "shape": list(after.shape), "channels": channels,
           "invalid_lanes_before": int(np.count_nonzero(~np.isfinite(before))),
           "invalid_lanes_after": int(np.count_nonzero(invalid)),
           "invalid_pixels_after": int(np.count_nonzero(np.any(invalid, axis=2))),
           "nonfinite_classification_unchanged": all(np.array_equal(operation(before), operation(after))
               for operation in (np.isnan, np.isposinf, np.isneginf)), "passes": {}}
    for prefix in sorted({name.rsplit(".", 1)[0] for name in channels}):
        indices = [i for i, name in enumerate(channels)
                   if name.rsplit(".", 1)[0] == prefix and name.rsplit(".", 1)[1] != "A"]
        # Color passes use RGB, while native Normal uses XYZ. Both are
        # complete three-component passes; Combined alpha is excluded here.
        assert len(indices) == 3, (prefix, indices)
        a, b = after[:, :, indices], before[:, :, indices]
        valid = np.isfinite(a) & np.isfinite(b)
        delta = a[valid].astype(np.float64) - b[valid].astype(np.float64)
        reference = b[valid].astype(np.float64)
        rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else None
        scale = float(np.sqrt(np.mean(reference * reference))) if reference.size else None
        error = np.zeros(a.shape, dtype=np.float64)
        error[valid] = np.abs(delta)
        pixel_error = np.max(error, axis=2)
        # Diagnostic counts only, not acceptance thresholds or tolerance changes.
        changed = pixel_error > 1e-3
        ys, xs = np.where(changed)
        row["passes"][prefix] = {"finite_lanes_compared": int(np.count_nonzero(valid)),
            "invalid_lanes_before": int(np.count_nonzero(~np.isfinite(b))),
            "invalid_lanes_after": int(np.count_nonzero(~np.isfinite(a))),
            "rmse": rmse, "before_rms": scale,
            "relative_rmse": rmse / scale if scale else (0.0 if rmse == 0.0 else None),
            "maximum_absolute_error": float(np.max(np.abs(delta))) if delta.size else None,
            "pixels_above_absolute_error": {str(t): int(np.count_nonzero(pixel_error > t))
                for t in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)},
            "above_1e-3_bbox_xy_inclusive": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if xs.size else None}
    records.append(row)
print(json.dumps({"schema": "psycles.native-path-lifetime-image-intervention.v1",
    "scope": "captured before/after image differences; not original-renderer parity or a speedup experiment",
    "before_manifest": identity(args.before_manifest or args.before / "canaries.json"),
    "after_manifest": identity(args.after_manifest or args.after / "canaries.json"),
    "records": records}, indent=2))

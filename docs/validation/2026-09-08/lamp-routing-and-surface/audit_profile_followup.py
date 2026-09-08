"""Compare two measured full-scene profiles after the imported-graph repair."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_inline_experiment import inspect, compare, render_pass_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    records = {label: inspect(path) for label, path in vars(args).items()}
    reference, reference_channels = compare._read_image(args.before / "psycles.exr")
    actual, actual_channels = compare._read_image(args.after / "psycles.exr")
    for image, channels in [(reference, reference_channels), (actual, actual_channels)]:
        render_pass_contract.validate_channels(channels)
        assert len(channels) == 46 and np.isfinite(image).all()
    assert reference.shape == actual.shape
    passes = {}
    for name in render_pass_contract.PASSES:
        original = reference[:, :, compare._find_cycles_channels(reference_channels, name)]
        result = actual[:, :, compare._find_cycles_channels(actual_channels, name)]
        delta = result.astype(np.float64) - original
        rms = np.sqrt(np.mean(original.astype(np.float64) ** 2))
        passes[name] = {"relative_rmse": float(np.sqrt(np.mean(delta * delta)) / max(rms, 1e-20)),
                        "maximum_absolute_error": float(np.max(np.abs(delta)))}
    print(json.dumps({
        "schema": "psycles.surface-profile-followup.v1",
        "scope": "full Barbershop 2048x858/64 spp/seed 0; fixed native geometry, new socket metadata and host Vector Math fold; profiler runs, not paired Cycles wall timings",
        "records": records,
        "image_control": {"all_46_channels_finite": True, "passes_vs_before": passes},
    }, indent=2))


if __name__ == "__main__":
    main()

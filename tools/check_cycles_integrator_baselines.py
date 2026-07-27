#!/usr/bin/env python3
"""Validate versioned Cycles/Psycles integrator probe baselines."""

from __future__ import annotations

import argparse
import json
import math
import pathlib


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path)
    return parser.parse_args()


def _main() -> None:
    arguments = _arguments()
    with arguments.baseline.open(encoding="utf-8") as source:
        baseline = json.load(source)
    if baseline.get("schema") != (
        "psycles.cycles-integrator-probe-baselines.v1"
    ):
        raise ValueError("unsupported integrator-probe baseline schema")
    verified = baseline.get("verified")
    if not isinstance(verified, dict) or not verified:
        raise ValueError("integrator baseline has no verified probes")
    for contract, probe in verified.items():
        max_rmse = probe.get("max_rmse")
        if (
            not isinstance(max_rmse, (int, float))
            or not math.isfinite(max_rmse)
            or max_rmse < 0.0
        ):
            raise ValueError(f"{contract} has invalid max_rmse")
        passes = probe.get("passes")
        if not isinstance(passes, dict) or not passes:
            raise ValueError(f"{contract} has no pass metrics")
        for pass_name, metrics in passes.items():
            for metric_name in ("rmse", "maximum_absolute_error"):
                value = metrics.get(metric_name)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value > max_rmse
                ):
                    raise ValueError(
                        f"{contract}/{pass_name} has invalid "
                        f"{metric_name}: {value}"
                    )
    print(
        f"Cycles integrator baselines: {len(verified)} verified "
        f"contract(s) for {baseline.get('blender_version')}"
    )


if __name__ == "__main__":
    _main()

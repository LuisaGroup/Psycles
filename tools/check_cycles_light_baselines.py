#!/usr/bin/env python3
"""Validate versioned Cycles/Psycles analytic-light probe baselines."""

from __future__ import annotations

import argparse
import json
import math
import pathlib


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path)
    return parser.parse_args()


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _main() -> None:
    arguments = _arguments()
    with arguments.baseline.open(encoding="utf-8") as source:
        baseline = json.load(source)
    if baseline.get("schema") != "psycles.cycles-light-probe-baselines.v1":
        raise ValueError("unsupported analytic-light baseline schema")
    if baseline.get("backend") != "Luisa/fallback":
        raise ValueError(
            "analytic-light baselines must identify the Luisa fallback "
            "backend, not a separate CPU renderer"
        )

    verified = baseline.get("verified")
    if not isinstance(verified, dict) or not verified:
        raise ValueError("analytic-light baseline has no verified probes")

    required_families = {"point", "spot", "area", "sun"}
    recorded_families: set[str] = set()
    for contract, probe in verified.items():
        family = probe.get("family")
        if family not in required_families:
            raise ValueError(f"{contract} has invalid light family")
        recorded_families.add(family)
        if not isinstance(probe.get("probe"), str):
            raise ValueError(f"{contract} has no probe name")
        samples = probe.get("samples")
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError(f"{contract} has invalid sample count")

        limits = probe.get("limits")
        metrics = probe.get("combined")
        if not isinstance(limits, dict) or not isinstance(metrics, dict):
            raise ValueError(f"{contract} has no Combined metrics or limits")
        for metric_name in (
            "relative_rmse",
            "maximum_absolute_error",
            "energy_relative_error",
        ):
            value = metrics.get(metric_name)
            limit = limits.get(f"max_{metric_name}")
            if not _finite_nonnegative(value):
                raise ValueError(f"{contract} has invalid {metric_name}")
            if not _finite_nonnegative(limit) or value > limit:
                raise ValueError(
                    f"{contract} has unverified {metric_name}: "
                    f"{value} > {limit}"
                )
        if metrics.get("invalid_pixels") != 0:
            raise ValueError(f"{contract} contains invalid pixels")

    missing = required_families - recorded_families
    if missing:
        raise ValueError(
            "analytic-light baseline is missing families: "
            + ", ".join(sorted(missing))
        )
    print(
        f"Cycles analytic-light baselines: {len(verified)} verified "
        f"contract(s) for {baseline.get('blender_version')} on "
        f"{baseline.get('backend')}"
    )


if __name__ == "__main__":
    _main()

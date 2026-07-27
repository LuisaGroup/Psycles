"""Compare a Luisa Nishita probe against an official Cycles probe."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cycles", type=pathlib.Path)
    parser.add_argument("luisa", type=pathlib.Path)
    args = parser.parse_args()

    cycles: dict[str, Any] = json.loads(
        args.cycles.read_text(encoding="utf-8")
    )
    luisa: dict[str, Any] = json.loads(
        args.luisa.read_text(encoding="utf-8")
    )
    rows: dict[str, Any] = {}
    global_max_absolute = 0.0
    global_max_relative = 0.0
    for name, luisa_rgb in luisa["probes"].items():
        cycles_rgb = cycles["probes"][name]["radiance"]
        absolute = [
            abs(float(a) - float(b))
            for a, b in zip(luisa_rgb, cycles_rgb, strict=True)
        ]
        relative = [
            error / max(abs(float(reference)), 1.0e-20)
            for error, reference in zip(
                absolute, cycles_rgb, strict=True
            )
        ]
        rmse = math.sqrt(
            sum(error * error for error in absolute) /
            len(absolute)
        )
        rows[name] = {
            "rmse": rmse,
            "max_absolute": max(absolute),
            "max_relative": max(relative),
        }
        global_max_absolute = max(
            global_max_absolute, max(absolute)
        )
        global_max_relative = max(
            global_max_relative, max(relative)
        )

    print(
        json.dumps(
            {
                "cycles": str(args.cycles),
                "luisa": str(args.luisa),
                "probes": rows,
                "max_absolute": global_max_absolute,
                "max_relative": global_max_relative,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()

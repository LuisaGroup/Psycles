#!/usr/bin/env python3
"""Reject last-bit emulation policies in production Luisa shaders."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


STRICT_FAST_MATH_PATTERN = re.compile(
    r"\.enable_fast_math\s*=\s*false\b"
)

# Strict arithmetic is admissible only when a proof obligation depends on a
# directed bound, not when a regression merely prefers one floating-point bit
# pattern. The volume maximum is consumed as a delta-tracking majorant, so an
# underestimate biases the estimator. Keep this allowlist intentionally exact:
# a second strict kernel, even in the same file, requires an explicit review.
ALLOWED_STRICT_FAST_MATH = {
    "src/luisa/path_tracer_volume_majorant_scene.cpp": 1,
}

# This old component re-ran a software Pluecker triangle test after native
# traversal to reproduce one accelerator's coincident-hit bits. Native ray
# query is the production traversal contract; the component was dead after the
# Blender 5.2 revalidation and must not be wired back under another include.
FORBIDDEN_LAST_BIT_EMULATION_TOKENS = (
    "CyclesTriangleIntersectionComponent",
    "make_cycles_triangle_intersection_component",
    "pluecker_cross",
    "clamp_bvh_direction",
    "normalize_exact",
)

# Compare whitespace-normalized source so line wrapping cannot hide the two
# scalar spellings that previously froze reciprocal-square-root rounding.
# Length-producing algorithms remain legal; only a normalization/inverse-sqrt
# operation expressed as scalar division is rejected.
FORBIDDEN_SCALAR_NORMALIZATION_TOKENS = (
    "1.0f / sqrt(",
    "value / sqrt(",
    "direction / sqrt(",
    "input / sqrt(",
    "offset / sqrt(",
)

NATIVE_NORMALIZE_CONTRACT = {
    "normalize_unchecked": "rsqrt(",
    "safe_normalize_nonzero": "normalize_in_domain(",
    "safe_normalize_nonzero_or": "normalize_in_domain(",
    "normalize_above_or": "normalize_in_domain(",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> int:
    root = parse_args().root.resolve()
    source_roots = (
        root / "src" / "luisa",
        root / "include" / "psycles" / "luisa",
    )
    for source_root in source_roots:
        if not source_root.is_dir():
            print(
                f"error: missing Luisa source tree: {source_root}",
                file=sys.stderr,
            )
            return 2

    violations: list[str] = []
    strict_counts: dict[str, int] = {}
    source_paths = sorted(
        path
        for source_root in source_roots
        for path in source_root.rglob("*")
    )
    for path in source_paths:
        if not path.is_file() or path.suffix not in {".cpp", ".h"}:
            continue
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        normalized_source = " ".join(source.split())
        strict_count = len(STRICT_FAST_MATH_PATTERN.findall(source))
        if strict_count:
            strict_counts[relative] = strict_count
        for token in FORBIDDEN_LAST_BIT_EMULATION_TOKENS:
            if token in source:
                violations.append(
                    f"{relative}: forbidden last-bit emulation token {token!r}"
                )
        for token in FORBIDDEN_SCALAR_NORMALIZATION_TOKENS:
            if token in normalized_source:
                violations.append(
                    f"{relative}: scalar normalization bypasses native rsqrt "
                    f"with {token!r}"
                )

    native_normalize_path = (
        root / "include" / "psycles" / "luisa" / "native_vector_math.h"
    )
    native_normalize_source = native_normalize_path.read_text(encoding="utf-8")
    for function, required_operation in NATIVE_NORMALIZE_CONTRACT.items():
        if (
            function not in native_normalize_source
            or required_operation not in native_normalize_source
        ):
            violations.append(
                "include/psycles/luisa/native_vector_math.h: missing native "
                f"normalization contract {function!r} -> {required_operation!r}"
            )

    if strict_counts != ALLOWED_STRICT_FAST_MATH:
        violations.append(
            "strict fast-math allowlist mismatch: "
            f"observed={strict_counts}, expected={ALLOWED_STRICT_FAST_MATH}"
        )

    cmake_sources = (root / "cmake" / "PsyclesLuisaSources.cmake").read_text(
        encoding="utf-8"
    )
    if "cycles_triangle_intersection_component" in cmake_sources:
        violations.append(
            "cmake/PsyclesLuisaSources.cmake: dead software re-intersection "
            "component is still linked"
        )

    if violations:
        for violation in violations:
            print(f"error: {violation}", file=sys.stderr)
        return 1
    print(
        "production Luisa shader policy passed: fast math enabled except for "
        "the proven volume-majorant bound; native reciprocal-square-root "
        "normalization; no last-bit emulation paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

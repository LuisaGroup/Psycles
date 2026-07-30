"""Compare two decoded Cycles/Psycles path traces by schema semantics.

Discrete state and sampled random values are exact gates. Continuous values
use an explicit float32 error bound. Triangle ID and barycentric differences
on shared edges are accepted only when object, primitive type, shader, surface
position, and all geometric/shading normals describe the same surface event.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

from cycles_path_trace_schema import (
    COMPARE_EXACT,
    COMPARE_FLOAT32,
    COMPARE_RANDOM_EXACT,
    COMPARE_RESERVED,
    COMPARE_TOPOLOGY,
    MAX_EVENTS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SLOTS,
    TraceSlot,
    comparison_policies,
)


DEFAULT_ABSOLUTE_TOLERANCE = 1.0e-6
DEFAULT_RELATIVE_TOLERANCE = 2.0e-6


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Cycles/Psycles per-path traces"
    )
    parser.add_argument("reference", type=pathlib.Path)
    parser.add_argument("actual", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path, nargs="?")
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=DEFAULT_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_TOLERANCE,
    )
    return parser.parse_args()


def _load(path: pathlib.Path) -> dict[str, Any]:
    if path.suffix.casefold() == ".exr":
        from decode_cycles_path_trace import decode_trace

        return decode_trace(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") == "psycles.cycles-path-trace-raw":
        from decode_cycles_path_trace import decode_raw_trace

        return decode_raw_trace(path)
    return document


def _record(trace: dict[str, Any], slot: TraceSlot) -> dict[str, Any]:
    if slot.scope == "global":
        return trace["global"][slot.name]
    assert slot.event is not None
    event = trace["events"][slot.event]
    if slot.scope == "event":
        return event["slots"][slot.name]
    assert slot.closure is not None
    closure = next(
        (
            item
            for item in event["closures"]
            if item["index"] == slot.closure
        ),
        None,
    )
    if closure is None:
        return {
            "written": False,
            **{component: 0.0 for component in slot.components},
        }
    return closure[slot.name]


def _within_float32_bound(
    reference: float,
    actual: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[bool, float, float]:
    if not math.isfinite(reference) or not math.isfinite(actual):
        return reference == actual, math.inf, 0.0
    error = abs(reference - actual)
    bound = absolute_tolerance + relative_tolerance * max(
        abs(reference),
        abs(actual),
    )
    return error <= bound, error, bound


def _event_surface_equivalent(
    reference: dict[str, Any],
    actual: dict[str, Any],
    event: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    reference_event = reference["events"][event]["slots"]
    actual_event = actual["events"][event]["slots"]
    exact_invariants = (
        ("isect_id", "object"),
        ("isect_id", "primitive_type"),
        ("surface_meta", "shader_low16"),
        ("surface_meta", "shader_high16"),
    )
    for slot, component in exact_invariants:
        if reference_event[slot][component] != actual_event[slot][component]:
            return False
    for slot in ("surface_p", "surface_ng", "surface_n"):
        for component in ("x", "y", "z"):
            equivalent, _, _ = _within_float32_bound(
                float(reference_event[slot][component]),
                float(actual_event[slot][component]),
                absolute_tolerance,
                relative_tolerance,
            )
            if not equivalent:
                return False
    return True


def compare_traces(
    reference: dict[str, Any],
    actual: dict[str, Any],
    *,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    for label, trace in (("reference", reference), ("actual", actual)):
        if trace.get("schema") != SCHEMA_NAME:
            raise ValueError(f"{label} trace has an unknown schema")
        if trace.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"{label} trace schema version is {trace.get('version')}, "
                f"expected {SCHEMA_VERSION}"
            )
        if len(trace.get("events", [])) != MAX_EVENTS:
            raise ValueError(f"{label} trace has the wrong event count")

    failures: list[dict[str, Any]] = []
    topology_equivalent: list[dict[str, Any]] = []
    checked = {
        COMPARE_EXACT: 0,
        COMPARE_RANDOM_EXACT: 0,
        COMPARE_FLOAT32: 0,
        COMPARE_TOPOLOGY: 0,
    }
    max_absolute_error = 0.0
    max_relative_error = 0.0
    max_error_field: str | None = None

    for slot in SLOTS:
        reference_record = _record(reference, slot)
        actual_record = _record(actual, slot)
        field_prefix = (
            f"global.{slot.name}"
            if slot.scope == "global"
            else f"events[{slot.event}].{slot.scope}.{slot.closure}."
            f"{slot.name}"
            if slot.scope == "closure"
            else f"events[{slot.event}].{slot.name}"
        )
        if reference_record["written"] != actual_record["written"]:
            failures.append(
                {
                    "field": f"{field_prefix}.written",
                    "policy": COMPARE_EXACT,
                    "reference": reference_record["written"],
                    "actual": actual_record["written"],
                }
            )
            continue
        if not reference_record["written"]:
            continue

        for component, policy in zip(
            slot.components,
            comparison_policies(slot),
        ):
            if policy == COMPARE_RESERVED:
                continue
            checked[policy] += 1
            field = f"{field_prefix}.{component}"
            reference_value = reference_record[component]
            actual_value = actual_record[component]
            if policy in {COMPARE_EXACT, COMPARE_RANDOM_EXACT}:
                if reference_value != actual_value:
                    failures.append(
                        {
                            "field": field,
                            "policy": policy,
                            "reference": reference_value,
                            "actual": actual_value,
                        }
                    )
                continue

            equivalent, error, bound = _within_float32_bound(
                float(reference_value),
                float(actual_value),
                absolute_tolerance,
                relative_tolerance,
            )
            denominator = max(
                abs(float(reference_value)),
                abs(float(actual_value)),
            )
            relative_error = error / denominator if denominator else error
            if policy == COMPARE_FLOAT32:
                if error > max_absolute_error:
                    max_absolute_error = error
                    max_error_field = field
                max_relative_error = max(max_relative_error, relative_error)
            if equivalent:
                continue
            mismatch = {
                "field": field,
                "policy": policy,
                "reference": reference_value,
                "actual": actual_value,
                "absolute_error": error,
                "bound": bound,
            }
            if (
                policy == COMPARE_TOPOLOGY
                and slot.event is not None
                and _event_surface_equivalent(
                    reference,
                    actual,
                    slot.event,
                    absolute_tolerance,
                    relative_tolerance,
                )
            ):
                topology_equivalent.append(mismatch)
            else:
                failures.append(mismatch)

    return {
        "schema": "psycles.cycles-path-trace-comparison.v1",
        "reference": reference.get("source"),
        "actual": actual.get("source"),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "checked": checked,
        "topology_equivalent": topology_equivalent,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "max_error_field": max_error_field,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def _main() -> None:
    arguments = _arguments()
    report = compare_traces(
        _load(arguments.reference),
        _load(arguments.actual),
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
        print(
            f"Path trace comparison {'passed' if report['passed'] else 'failed'}: "
            f"{arguments.output}"
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()

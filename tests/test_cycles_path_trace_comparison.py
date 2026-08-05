"""Regressions for formal per-field path trace comparison."""

from __future__ import annotations

import copy
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compare_cycles_path_traces as comparison
import cycles_path_trace_schema as schema


def _trace() -> dict[str, object]:
    globals_: dict[str, object] = {}
    events: list[dict[str, object]] = [
        {"index": event, "written": True, "slots": {}, "closures": []}
        for event in range(schema.MAX_EVENTS)
    ]
    for slot in schema.SLOTS:
        record = {
            "written": True,
            **{component: 0.0 for component in slot.components},
        }
        if slot.scope == "global":
            globals_[slot.name] = record
        elif slot.scope == "event":
            assert slot.event is not None
            events[slot.event]["slots"][slot.name] = record
        else:
            assert slot.event is not None and slot.closure is not None
            closures = events[slot.event]["closures"]
            closure = next(
                (
                    item
                    for item in closures
                    if item["index"] == slot.closure
                ),
                None,
            )
            if closure is None:
                closure = {"index": slot.closure}
                closures.append(closure)
            closure[slot.name] = record
    globals_["header"]["schema_version"] = schema.SCHEMA_VERSION
    return {
        "schema": schema.SCHEMA_NAME,
        "version": schema.SCHEMA_VERSION,
        "source": "synthetic",
        "global": globals_,
        "events": events,
    }


class CyclesPathTraceComparisonTests(unittest.TestCase):
    def test_float32_rounding_is_bounded(self) -> None:
        reference = _trace()
        actual = copy.deepcopy(reference)
        actual["events"][0]["slots"]["surface_p"]["x"] = 5.0e-7
        report = comparison.compare_traces(reference, actual)
        self.assertTrue(report["passed"])
        self.assertEqual(report["failure_count"], 0)

    def test_random_dimensions_are_exact_gates(self) -> None:
        reference = _trace()
        actual = copy.deepcopy(reference)
        actual["events"][0]["slots"]["random_bsdf"]["u"] = 1.0e-8
        report = comparison.compare_traces(reference, actual)
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["failures"][0]["policy"],
            schema.COMPARE_RANDOM_EXACT,
        )

    def test_shared_edge_uses_surface_event_invariants(self) -> None:
        reference = _trace()
        actual = copy.deepcopy(reference)
        actual["events"][0]["slots"]["isect_id"]["primitive"] = 1.0
        actual["events"][0]["slots"]["isect_coord"]["u"] = 0.5
        actual["events"][0]["slots"]["isect_coord"]["v"] = 0.5
        report = comparison.compare_traces(reference, actual)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["topology_equivalent"]), 3)

        actual["events"][0]["slots"]["surface_p"]["x"] = 0.01
        report = comparison.compare_traces(reference, actual)
        self.assertFalse(report["passed"])
        self.assertGreaterEqual(report["failure_count"], 4)

    def test_backend_shadow_hit_diagnostics_are_not_equality_gates(self) -> None:
        reference = _trace()
        actual = copy.deepcopy(reference)
        reference["events"][0]["slots"]["shadow_hit_id"]["written"] = False
        actual["events"][0]["slots"]["shadow_hit_id"]["object"] = 17.0
        actual["events"][0]["slots"]["shadow_hit_id"]["primitive"] = 23.0
        report = comparison.compare_traces(reference, actual)
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()

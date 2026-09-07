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
    def test_subsurface_rng_gap_is_not_an_extra_surface_event(self) -> None:
        reference = _trace()
        for index, event in enumerate(reference["events"]):
            event["slots"]["state_depth"].update(event=index, rng_offset=16 * (index + 1))
        # Offset 32 belongs to the subsurface transport, not an exit shader.
        for record in reference["events"][1]["slots"].values():
            record["written"] = False
        for closure in reference["events"][1]["closures"]:
            for name, record in closure.items():
                if name != "index":
                    record["written"] = False
        actual = copy.deepcopy(reference)
        actual["events"] = [actual["events"][i] for i in (0, 2, 3, 1)]
        for index, event in enumerate(actual["events"]):
            event["slots"]["state_depth"]["event"] = index
        report = comparison.compare_traces(reference, actual, align_by_rng_offset=True)
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual([row["actual_event"] for row in report["event_alignment"]["matched"]], [0, 1, 2])
        actual["events"][1]["slots"]["state_depth"]["bounce"] += 1
        report = comparison.compare_traces(reference, actual, align_by_rng_offset=True)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failures"][0]["field"], "events[2].state_depth.bounce")

    def test_rng_alignment_rejects_ambiguous_or_missing_events(self) -> None:
        reference = _trace()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            comparison.compare_traces(reference, copy.deepcopy(reference), align_by_rng_offset=True)
        for i, event in enumerate(reference["events"]):
            event["slots"]["state_depth"]["rng_offset"] = 16 * (i + 1)
        actual = copy.deepcopy(reference)
        actual["events"][2]["slots"]["state_depth"]["rng_offset"] = 96
        report = comparison.compare_traces(reference, actual, align_by_rng_offset=True)
        self.assertFalse(report["passed"])
        self.assertEqual(report["event_alignment"]["unmatched_reference"], [48])
        self.assertEqual(report["event_alignment"]["unmatched_actual"], [96])

    def test_rescaled_selection_is_derived_not_a_new_random_draw(self) -> None:
        reference = _trace()
        actual = copy.deepcopy(reference)
        actual["events"][0]["slots"]["closure_random"]["selection_rescaled"] = 5e-7
        self.assertTrue(comparison.compare_traces(reference, actual)["passed"])
        actual["events"][0]["slots"]["random_bsdf"]["selection"] = 5e-7
        self.assertFalse(comparison.compare_traces(reference, actual)["passed"])

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

    def test_forward_emission_measure_is_a_complete_gate(self) -> None:
        reference = _trace()
        actual = copy.deepcopy(reference)
        fields = (
            ("forward_emission", "r", 0.25),
            ("forward_policy", "emission_sampling", 4.0),
            ("forward_policy", "selection_pdf", 0.125),
            ("forward_policy", "pdf_valid", 1.0),
            ("forward_mis", "bsdf_pdf", 0.25),
            ("forward_mis", "light_pdf", 0.125),
            ("forward_mis", "mis_weight", 0.8),
            ("forward_contribution", "b", 0.5),
        )
        for slot, component, value in fields:
            candidate = copy.deepcopy(actual)
            candidate["events"][1]["slots"][slot][component] = value
            report = comparison.compare_traces(reference, candidate)
            self.assertFalse(report["passed"], (slot, component))
            self.assertEqual(report["failure_count"], 1)
            self.assertEqual(
                report["failures"][0]["field"],
                f"events[1].{slot}.{component}",
            )


if __name__ == "__main__":
    unittest.main()

"""Regressions for the cross-backend path trace schema."""

from __future__ import annotations

import pathlib
import sys
import unittest
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import cycles_path_trace_schema as trace_schema


class CyclesPathTraceSchemaTests(unittest.TestCase):
    def test_generated_cpp_header_is_current(self) -> None:
        header = (
            ROOT
            / "include"
            / "psycles"
            / "luisa"
            / "path_trace_schema.h"
        )
        self.assertEqual(
            header.read_text(encoding="utf-8"),
            trace_schema.cpp_header(),
        )

    def test_trace_slots_are_fixed_and_contiguous(self) -> None:
        schema: Any = trace_schema
        self.assertEqual(schema.SCHEMA_VERSION, 3)
        self.assertEqual(schema.GLOBAL_SLOT_COUNT, 8)
        self.assertEqual(schema.EVENT_SLOT_COUNT, 72)
        self.assertEqual(schema.SHADOW_EVENT_SLOT_COUNT, 8)
        self.assertEqual(schema.FORWARD_EVENT_SLOT_COUNT, 4)
        self.assertEqual(schema.SHADOW_EVENT_BASE, 296)
        self.assertEqual(schema.FORWARD_EVENT_BASE, 328)
        self.assertEqual(schema.MAX_EVENTS, 4)
        self.assertEqual(schema.MAX_CLOSURES, 8)
        self.assertEqual(schema.AOV_COUNT, 344)
        self.assertEqual(len(schema.SLOTS), schema.AOV_COUNT)
        self.assertEqual(
            [slot.index for slot in schema.SLOTS],
            list(range(schema.AOV_COUNT)),
        )
        self.assertEqual(schema.SLOTS[0].aov, "PsyTrace000")
        self.assertEqual(schema.SLOTS[-1].aov, "PsyTrace343")

    def test_each_event_has_identical_layout(self) -> None:
        schema: Any = trace_schema
        for event in range(schema.MAX_EVENTS):
            event_slots = [
                slot for slot in schema.SLOTS if slot.event == event
            ]
            self.assertEqual(
                len(event_slots),
                schema.EVENT_SLOT_COUNT
                + schema.SHADOW_EVENT_SLOT_COUNT
                + schema.FORWARD_EVENT_SLOT_COUNT,
            )
            self.assertEqual(
                sum(slot.scope == "closure" for slot in event_slots),
                24,
            )
            self.assertEqual(
                {
                    slot.closure
                    for slot in event_slots
                    if slot.scope == "closure"
                },
                set(range(schema.MAX_CLOSURES)),
            )

    def test_cycles_random_component_order_is_explicit(self) -> None:
        schema: Any = trace_schema
        by_scope_and_name = {
            (slot.scope, slot.name): slot
            for slot in schema.SLOTS
            if slot.event in {None, 0}
        }
        self.assertEqual(
            by_scope_and_name[
                ("global", "lens_time")
            ].components,
            ("time", "lens_u", "lens_v"),
        )
        self.assertEqual(
            by_scope_and_name[
                ("event", "random_light")
            ].components,
            ("u", "v", "selection"),
        )
        self.assertEqual(
            by_scope_and_name[
                ("event", "random_bsdf")
            ].components,
            ("u", "v", "selection"),
        )
        self.assertEqual(
            by_scope_and_name[
                ("event", "closure_random")
            ].components,
            ("u", "v", "selection_rescaled"),
        )

    def test_nee_phase_slots_are_typed_float32_gates(self) -> None:
        schema: Any = trace_schema
        event_slots = {
            slot.name: slot
            for slot in schema.SLOTS
            if slot.scope == "event" and slot.event == 0
        }
        names = (
            "nee_bsdf",
            "nee_diffuse",
            "nee_glossy",
            "nee_weighted_bsdf",
            "nee_light_shader",
            "nee_unshadowed",
            "nee_contribution",
        )
        self.assertEqual(
            [event_slots[name].index for name in names],
            list(range(49, 56)),
        )
        for name in names:
            self.assertEqual(
                trace_schema.comparison_policies(event_slots[name]),
                (trace_schema.COMPARE_FLOAT32,) * 3,
            )

    def test_shadow_trace_is_an_append_only_event_tail(self) -> None:
        schema: Any = trace_schema
        shadow_names = (
            "shadow_ray_p",
            "shadow_ray_d",
            "shadow_ray_range",
            "shadow_source",
            "shadow_light",
            "shadow_hit_id",
            "shadow_hit_coord",
            "shadow_transmittance",
        )
        for event in range(schema.MAX_EVENTS):
            event_slots = {
                slot.name: slot
                for slot in schema.SLOTS
                if slot.scope == "event" and slot.event == event
            }
            self.assertEqual(
                [event_slots[name].index for name in shadow_names],
                list(
                    range(
                        schema.SHADOW_EVENT_BASE
                        + event * schema.SHADOW_EVENT_SLOT_COUNT,
                        schema.SHADOW_EVENT_BASE
                        + (event + 1) * schema.SHADOW_EVENT_SLOT_COUNT,
                    )
                ),
            )

    def test_forward_trace_is_an_append_only_event_tail(self) -> None:
        schema: Any = trace_schema
        names = (
            "forward_emission",
            "forward_policy",
            "forward_mis",
            "forward_contribution",
        )
        for event in range(schema.MAX_EVENTS):
            event_slots = {
                slot.name: slot
                for slot in schema.SLOTS
                if slot.scope == "event" and slot.event == event
            }
            self.assertEqual(
                [event_slots[name].index for name in names],
                list(
                    range(
                        schema.FORWARD_EVENT_BASE
                        + event * schema.FORWARD_EVENT_SLOT_COUNT,
                        schema.FORWARD_EVENT_BASE
                        + (event + 1) * schema.FORWARD_EVENT_SLOT_COUNT,
                    )
                ),
            )

    def test_schema_document_is_json_shaped(self) -> None:
        schema: Any = trace_schema
        document = schema.schema_document()
        self.assertEqual(document["name"], "psycles.cycles-path-trace")
        self.assertEqual(document["version"], schema.SCHEMA_VERSION)
        self.assertEqual(document["aov_count"], schema.AOV_COUNT)
        self.assertEqual(len(document["slots"]), schema.AOV_COUNT)
        self.assertEqual(document["slots"][24]["name"], "random_bsdf")
        self.assertEqual(
            document["slots"][24]["comparison"],
            [schema.COMPARE_RANDOM_EXACT] * 3,
        )


if __name__ == "__main__":
    unittest.main()

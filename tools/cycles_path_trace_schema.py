"""Stable schema for Cycles/Psycles per-path differential traces.

The diagnostic Cycles kernel writes one RGB value per color AOV.  This module
is deliberately Blender-independent so the render harness, EXR decoder, tests,
and the Luisa implementation all share one indexed contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SCHEMA_NAME = "psycles.cycles-path-trace"
SCHEMA_VERSION = 1
AOV_PREFIX = "PsyTrace"
GLOBAL_SLOT_COUNT = 8
EVENT_SLOT_COUNT = 72
MAX_EVENTS = 4
MAX_CLOSURES = 8
AOV_COUNT = GLOBAL_SLOT_COUNT + EVENT_SLOT_COUNT * MAX_EVENTS

COMPARE_EXACT = "exact"
COMPARE_RANDOM_EXACT = "random_exact"
COMPARE_FLOAT32 = "float32"
COMPARE_TOPOLOGY = "topology"
COMPARE_RESERVED = "reserved"


@dataclass(frozen=True)
class TraceSlot:
    index: int
    scope: str
    name: str
    components: tuple[str, str, str]
    event: int | None = None
    closure: int | None = None

    @property
    def aov(self) -> str:
        return aov_name(self.index)


_GLOBAL_LAYOUT = (
    ("header", ("schema_version", "pixel_x", "pixel_y")),
    ("rng", ("sample", "rng_pixel_low16", "rng_pixel_high16")),
    ("filter", ("filter_x", "filter_y", "reserved")),
    # Cycles stores PRNG_LENS_TIME as (time, lens_u, lens_v).
    ("lens_time", ("time", "lens_u", "lens_v")),
    ("ray_p", ("x", "y", "z")),
    ("ray_d", ("x", "y", "z")),
    ("ray_range", ("tmin", "tmax", "time")),
    ("reserved", ("x", "y", "z")),
)

_EVENT_LAYOUT = (
    ("state_depth", ("event", "rng_offset", "bounce")),
    (
        "state_lobes",
        ("transparent_bounce", "diffuse_bounce", "glossy_bounce"),
    ),
    (
        "state_visibility",
        ("transmission_bounce", "visibility_low16", "visibility_high16"),
    ),
    (
        "state_flags",
        ("path_flag_low16", "path_flag_high16", "continuation_probability"),
    ),
    ("throughput", ("r", "g", "b")),
    ("ray_p", ("x", "y", "z")),
    ("ray_d", ("x", "y", "z")),
    ("ray_range", ("tmin", "tmax", "time")),
    ("isect_coord", ("distance", "u", "v")),
    ("isect_id", ("object", "primitive", "primitive_type")),
    (
        "surface_meta",
        ("shader_low16", "shader_high16", "closure_count"),
    ),
    ("surface_p", ("x", "y", "z")),
    ("surface_ng", ("x", "y", "z")),
    ("surface_n", ("x", "y", "z")),
    ("random_scalars", ("terminate", "light_terminate", "reserved")),
    # Cycles uses the first two Sobol components for the directional/shape
    # sample and the third for discrete selection.
    ("random_light", ("u", "v", "selection")),
    ("random_bsdf", ("u", "v", "selection")),
    ("light_meta", ("type", "emitter", "primitive")),
    ("light_id", ("object", "group", "reserved")),
    ("light_pdf", ("pdf", "selection_pdf", "eval_factor")),
    ("light_d", ("x", "y", "z")),
    ("light_p", ("x", "y", "z")),
    ("light_ng", ("x", "y", "z")),
    ("light_eval", ("distance", "bsdf_pdf", "mis_weight")),
    ("closure_pick", ("index", "type", "sample_weight")),
    ("closure_random", ("u", "v", "selection_rescaled")),
    ("closure_weight", ("r", "g", "b")),
    ("closure_n", ("x", "y", "z")),
    ("bsdf_meta", ("pdf", "unguided_pdf", "label")),
    ("bsdf_wo", ("x", "y", "z")),
    ("bsdf_eval", ("r", "g", "b")),
    ("bsdf_roughness_eta", ("roughness_u", "roughness_v", "eta")),
    ("post_depth", ("bounce", "transparent_bounce", "rng_offset")),
    ("post_throughput", ("r", "g", "b")),
    ("post_ray_p", ("x", "y", "z")),
    ("post_ray_d", ("x", "y", "z")),
    (
        "post_flags",
        ("path_flag_low16", "path_flag_high16", "reserved"),
    ),
    (
        "post_mis",
        ("mis_ray_pdf", "min_ray_pdf", "continuation_probability"),
    ),
    (
        "surface_flags",
        ("runtime_flag_low16", "runtime_flag_high16", "reserved"),
    ),
    (
        "post_visibility",
        ("visibility_low16", "visibility_high16", "reserved"),
    ),
    (
        "light_shader",
        ("shader_low16", "shader_high16", "reserved"),
    ),
    ("reserved_41", ("x", "y", "z")),
    ("reserved_42", ("x", "y", "z")),
    ("reserved_43", ("x", "y", "z")),
    ("reserved_44", ("x", "y", "z")),
    ("reserved_45", ("x", "y", "z")),
    ("reserved_46", ("x", "y", "z")),
    ("reserved_47", ("x", "y", "z")),
)


def aov_name(index: int) -> str:
    if not 0 <= index < AOV_COUNT:
        raise ValueError(f"trace AOV index {index} is out of range")
    return f"{AOV_PREFIX}{index:03d}"


_GLOBAL_COMPARISON = {
    "header": (COMPARE_EXACT,) * 3,
    "rng": (COMPARE_EXACT,) * 3,
    "filter": (
        COMPARE_RANDOM_EXACT,
        COMPARE_RANDOM_EXACT,
        COMPARE_RESERVED,
    ),
    "lens_time": (COMPARE_RANDOM_EXACT,) * 3,
    "ray_p": (COMPARE_FLOAT32,) * 3,
    "ray_d": (COMPARE_FLOAT32,) * 3,
    "ray_range": (COMPARE_FLOAT32,) * 3,
    "reserved": (COMPARE_RESERVED,) * 3,
}

_EVENT_COMPARISON = {
    "state_depth": (COMPARE_EXACT,) * 3,
    "state_lobes": (COMPARE_EXACT,) * 3,
    "state_visibility": (COMPARE_EXACT,) * 3,
    "state_flags": (
        COMPARE_EXACT,
        COMPARE_EXACT,
        COMPARE_FLOAT32,
    ),
    "throughput": (COMPARE_FLOAT32,) * 3,
    "ray_p": (COMPARE_FLOAT32,) * 3,
    "ray_d": (COMPARE_FLOAT32,) * 3,
    "ray_range": (COMPARE_FLOAT32,) * 3,
    "isect_coord": (
        COMPARE_FLOAT32,
        COMPARE_TOPOLOGY,
        COMPARE_TOPOLOGY,
    ),
    "isect_id": (
        COMPARE_EXACT,
        COMPARE_TOPOLOGY,
        COMPARE_EXACT,
    ),
    "surface_meta": (COMPARE_EXACT,) * 3,
    "surface_p": (COMPARE_FLOAT32,) * 3,
    "surface_ng": (COMPARE_FLOAT32,) * 3,
    "surface_n": (COMPARE_FLOAT32,) * 3,
    "random_scalars": (
        COMPARE_RANDOM_EXACT,
        COMPARE_RANDOM_EXACT,
        COMPARE_RESERVED,
    ),
    "random_light": (COMPARE_RANDOM_EXACT,) * 3,
    "random_bsdf": (COMPARE_RANDOM_EXACT,) * 3,
    "light_meta": (COMPARE_EXACT,) * 3,
    "light_id": (
        COMPARE_EXACT,
        COMPARE_EXACT,
        COMPARE_RESERVED,
    ),
    "light_pdf": (COMPARE_FLOAT32,) * 3,
    "light_d": (COMPARE_FLOAT32,) * 3,
    "light_p": (COMPARE_FLOAT32,) * 3,
    "light_ng": (COMPARE_FLOAT32,) * 3,
    "light_eval": (COMPARE_FLOAT32,) * 3,
    "closure_pick": (
        COMPARE_EXACT,
        COMPARE_EXACT,
        COMPARE_FLOAT32,
    ),
    "closure_random": (COMPARE_RANDOM_EXACT,) * 3,
    "closure_weight": (COMPARE_FLOAT32,) * 3,
    "closure_n": (COMPARE_FLOAT32,) * 3,
    "bsdf_meta": (
        COMPARE_FLOAT32,
        COMPARE_FLOAT32,
        COMPARE_EXACT,
    ),
    "bsdf_wo": (COMPARE_FLOAT32,) * 3,
    "bsdf_eval": (COMPARE_FLOAT32,) * 3,
    "bsdf_roughness_eta": (COMPARE_FLOAT32,) * 3,
    "post_depth": (COMPARE_EXACT,) * 3,
    "post_throughput": (COMPARE_FLOAT32,) * 3,
    "post_ray_p": (COMPARE_FLOAT32,) * 3,
    "post_ray_d": (COMPARE_FLOAT32,) * 3,
    "post_flags": (
        COMPARE_EXACT,
        COMPARE_EXACT,
        COMPARE_RESERVED,
    ),
    "post_mis": (COMPARE_FLOAT32,) * 3,
    "surface_flags": (
        COMPARE_EXACT,
        COMPARE_EXACT,
        COMPARE_RESERVED,
    ),
    "post_visibility": (
        COMPARE_EXACT,
        COMPARE_EXACT,
        COMPARE_RESERVED,
    ),
    "light_shader": (
        COMPARE_EXACT,
        COMPARE_EXACT,
        COMPARE_RESERVED,
    ),
    "reserved_41": (COMPARE_RESERVED,) * 3,
    "reserved_42": (COMPARE_RESERVED,) * 3,
    "reserved_43": (COMPARE_RESERVED,) * 3,
    "reserved_44": (COMPARE_RESERVED,) * 3,
    "reserved_45": (COMPARE_RESERVED,) * 3,
    "reserved_46": (COMPARE_RESERVED,) * 3,
    "reserved_47": (COMPARE_RESERVED,) * 3,
}


def comparison_policies(
    slot: TraceSlot,
) -> tuple[str, str, str]:
    if slot.scope == "global":
        return _GLOBAL_COMPARISON[slot.name]
    if slot.scope == "event":
        return _EVENT_COMPARISON[slot.name]
    if slot.name == "meta":
        return (
            COMPARE_EXACT,
            COMPARE_EXACT,
            COMPARE_FLOAT32,
        )
    if slot.name in {"weight", "normal"}:
        return (COMPARE_FLOAT32,) * 3
    raise AssertionError(f"unknown trace slot comparison policy: {slot}")


def _closure_slots(event: int, event_base: int) -> Iterable[TraceSlot]:
    closure_base = len(_EVENT_LAYOUT)
    for closure in range(MAX_CLOSURES):
        slot_base = event_base + closure_base + closure * 3
        yield TraceSlot(
            slot_base,
            "closure",
            "meta",
            ("index", "type", "sample_weight"),
            event=event,
            closure=closure,
        )
        yield TraceSlot(
            slot_base + 1,
            "closure",
            "weight",
            ("r", "g", "b"),
            event=event,
            closure=closure,
        )
        yield TraceSlot(
            slot_base + 2,
            "closure",
            "normal",
            ("x", "y", "z"),
            event=event,
            closure=closure,
        )


def slots() -> tuple[TraceSlot, ...]:
    result = [
        TraceSlot(index, "global", name, components)
        for index, (name, components) in enumerate(_GLOBAL_LAYOUT)
    ]
    for event in range(MAX_EVENTS):
        event_base = GLOBAL_SLOT_COUNT + event * EVENT_SLOT_COUNT
        result.extend(
            TraceSlot(
                event_base + relative,
                "event",
                name,
                components,
                event=event,
            )
            for relative, (name, components) in enumerate(_EVENT_LAYOUT)
        )
        result.extend(_closure_slots(event, event_base))
    if len(result) != AOV_COUNT:
        raise AssertionError(
            f"trace schema has {len(result)} slots, expected {AOV_COUNT}"
        )
    if any(slot.index != index for index, slot in enumerate(result)):
        raise AssertionError("trace schema slots are not contiguous")
    return tuple(result)


SLOTS = slots()


def cpp_header() -> str:
    """Generate the C++ half of the indexed trace contract."""

    def enum_members(
        layout: tuple[tuple[str, tuple[str, str, str]], ...],
    ) -> list[str]:
        return [
            f"    {name} = {index}u,"
            for index, (name, _components) in enumerate(layout)
        ]

    lines = [
        "#pragma once",
        "",
        "// Generated by tools/cycles_path_trace_schema.py. Do not edit.",
        "",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "namespace psycles::luisa_backend::path_trace_schema {",
        "",
        f'inline constexpr std::string_view name{{"{SCHEMA_NAME}"}};',
        f"inline constexpr std::uint32_t version = {SCHEMA_VERSION}u;",
        (
            "inline constexpr std::uint32_t global_slot_count = "
            f"{GLOBAL_SLOT_COUNT}u;"
        ),
        (
            "inline constexpr std::uint32_t event_slot_count = "
            f"{EVENT_SLOT_COUNT}u;"
        ),
        f"inline constexpr std::uint32_t max_events = {MAX_EVENTS}u;",
        f"inline constexpr std::uint32_t max_closures = {MAX_CLOSURES}u;",
        f"inline constexpr std::uint32_t slot_count = {AOV_COUNT}u;",
        "",
        "enum class GlobalSlot : std::uint32_t {",
        *enum_members(_GLOBAL_LAYOUT),
        "};",
        "",
        "enum class EventSlot : std::uint32_t {",
        *enum_members(_EVENT_LAYOUT),
        f"    closure_base = {len(_EVENT_LAYOUT)}u,",
        "};",
        "",
        "[[nodiscard]] constexpr std::uint32_t index(",
        "    GlobalSlot slot) noexcept {",
        "    return static_cast<std::uint32_t>(slot);",
        "}",
        "",
        "[[nodiscard]] constexpr std::uint32_t index(",
        "    std::uint32_t event,",
        "    EventSlot slot) noexcept {",
        "    return global_slot_count + event * event_slot_count +",
        "           static_cast<std::uint32_t>(slot);",
        "}",
        "",
        "[[nodiscard]] constexpr std::uint32_t closure_index(",
        "    std::uint32_t event,",
        "    std::uint32_t closure,",
        "    std::uint32_t field) noexcept {",
        "    return index(event, EventSlot::closure_base) + closure * 3u +",
        "           field;",
        "}",
        "",
        "}// namespace psycles::luisa_backend::path_trace_schema",
        "",
    ]
    return "\n".join(lines)


def schema_document() -> dict[str, object]:
    return {
        "name": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "aov_prefix": AOV_PREFIX,
        "aov_count": AOV_COUNT,
        "global_slot_count": GLOBAL_SLOT_COUNT,
        "event_slot_count": EVENT_SLOT_COUNT,
        "max_events": MAX_EVENTS,
        "max_closures": MAX_CLOSURES,
        "slots": [
            {
                "index": slot.index,
                "aov": slot.aov,
                "scope": slot.scope,
                "event": slot.event,
                "closure": slot.closure,
                "name": slot.name,
                "components": list(slot.components),
                "comparison": list(comparison_policies(slot)),
            }
            for slot in SLOTS
        ],
    }

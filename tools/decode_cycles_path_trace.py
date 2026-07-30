"""Decode a one-pixel Cycles/Psycles trace EXR to structured JSON."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import numpy
import OpenImageIO as oiio

from cycles_path_trace_schema import (
    AOV_COUNT,
    MAX_EVENTS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SLOTS,
    TraceSlot,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode an instrumented one-pixel path trace EXR"
    )
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path, nargs="?")
    return parser.parse_args()


def _channel_value(
    channel_values: dict[str, float],
    aov: str,
    component: str,
) -> float:
    suffix = f".{aov}.{component}"
    matches = [
        value
        for name, value in channel_values.items()
        if name == f"{aov}.{component}" or name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one EXR channel for {aov}.{component}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _read_trace_pixel(path: pathlib.Path) -> dict[int, tuple[float, ...]]:
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise RuntimeError(f"OpenImageIO could not open {path}")
    try:
        channel_values: dict[str, float] = {}
        subimage = 0
        while image.seek_subimage(subimage, 0):
            spec = image.spec()
            if spec.width != 1 or spec.height != 1 or spec.depth != 1:
                raise RuntimeError(
                    "path trace EXR parts must contain exactly one pixel, got "
                    f"{spec.width}x{spec.height}x{spec.depth} in part "
                    f"{subimage}"
                )
            pixels = numpy.asarray(
                image.read_image(oiio.FLOAT),
                dtype=numpy.float32,
            )
            pixels = pixels.reshape(spec.height, spec.width, spec.nchannels)
            pixel = pixels[0, 0]
            for index, name in enumerate(spec.channelnames):
                if name in channel_values:
                    raise RuntimeError(
                        f"duplicate EXR channel {name!r} in part {subimage}"
                    )
                channel_values[name] = float(pixel[index])
            subimage += 1
        if not channel_values:
            raise RuntimeError(f"EXR {path} has no readable channels")

        result: dict[int, tuple[float, ...]] = {}
        for slot in SLOTS:
            values = [
                _channel_value(channel_values, slot.aov, component)
                for component in ("R", "G", "B", "A")
            ]
            result[slot.index] = tuple(values)
        if len(result) != AOV_COUNT:
            raise AssertionError("decoded trace AOV count does not match schema")
        return result
    finally:
        image.close()


def _slot_value(
    slot: TraceSlot,
    rgba: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "slot": slot.index,
        "aov": slot.aov,
        "written": rgba[3] == 1.0,
        **{
            component: value
            for component, value in zip(slot.components, rgba[:3])
        },
    }


def _combine_u32(
    record: dict[str, Any],
    low_name: str,
    high_name: str,
) -> int | None:
    low = record.get(low_name)
    high = record.get(high_name)
    if (
        not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
        or not math.isfinite(low)
        or not math.isfinite(high)
    ):
        return None
    return int(low) | (int(high) << 16)


def _decode_values(
    values: dict[int, tuple[float, ...]],
    source: pathlib.Path,
) -> dict[str, Any]:
    globals_: dict[str, Any] = {}
    events: list[dict[str, Any]] = [
        {"index": event, "slots": {}, "closures": []}
        for event in range(MAX_EVENTS)
    ]
    closure_records: dict[tuple[int, int], dict[str, Any]] = {}

    for slot in SLOTS:
        record = _slot_value(slot, values[slot.index])
        if slot.scope == "global":
            globals_[slot.name] = record
        elif slot.scope == "event":
            assert slot.event is not None
            events[slot.event]["slots"][slot.name] = record
        else:
            assert slot.event is not None and slot.closure is not None
            closure = closure_records.setdefault(
                (slot.event, slot.closure),
                {"index": slot.closure},
            )
            closure[slot.name] = record

    for (event, _), closure in closure_records.items():
        if any(
            part.get("written", False)
            for name, part in closure.items()
            if name != "index"
        ):
            events[event]["closures"].append(closure)

    rng = globals_["rng"]
    rng["rng_pixel"] = _combine_u32(
        rng,
        "rng_pixel_low16",
        "rng_pixel_high16",
    )
    for event in events:
        flags = event["slots"]["state_flags"]
        flags["path_flag"] = _combine_u32(
            flags,
            "path_flag_low16",
            "path_flag_high16",
        )
        post_flags = event["slots"]["post_flags"]
        post_flags["path_flag"] = _combine_u32(
            post_flags,
            "path_flag_low16",
            "path_flag_high16",
        )
        visibility = event["slots"]["state_visibility"]
        visibility["visibility"] = _combine_u32(
            visibility,
            "visibility_low16",
            "visibility_high16",
        )
        surface_meta = event["slots"]["surface_meta"]
        surface_meta["shader"] = _combine_u32(
            surface_meta,
            "shader_low16",
            "shader_high16",
        )
        surface_flags = event["slots"]["surface_flags"]
        surface_flags["runtime_flag"] = _combine_u32(
            surface_flags,
            "runtime_flag_low16",
            "runtime_flag_high16",
        )
        post_visibility = event["slots"]["post_visibility"]
        post_visibility["visibility"] = _combine_u32(
            post_visibility,
            "visibility_low16",
            "visibility_high16",
        )
        light_shader = event["slots"]["light_shader"]
        light_shader["shader"] = _combine_u32(
            light_shader,
            "shader_low16",
            "shader_high16",
        )
        event["written"] = event["slots"]["state_depth"]["written"]

    header = globals_["header"]
    if not header["written"]:
        raise RuntimeError(
            "trace header was not written; use the instrumented Cycles build"
        )
    actual_version = int(header["schema_version"])
    if actual_version != SCHEMA_VERSION:
        raise RuntimeError(
            f"trace schema version is {actual_version}, decoder expects "
            f"{SCHEMA_VERSION}"
        )

    return {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "source": str(source.resolve()),
        "global": globals_,
        "events": events,
    }


def decode_trace(path: pathlib.Path) -> dict[str, Any]:
    return _decode_values(_read_trace_pixel(path), path)


def decode_raw_trace(path: pathlib.Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "psycles.cycles-path-trace-raw":
        raise RuntimeError(f"{path} is not a raw Psycles path trace")
    if document.get("trace_schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "raw path trace schema version is "
            f"{document.get('trace_schema_version')}, expected "
            f"{SCHEMA_VERSION}"
        )
    raw_slots = document.get("slots")
    if not isinstance(raw_slots, list) or len(raw_slots) != AOV_COUNT:
        raise RuntimeError(
            f"raw path trace must contain exactly {AOV_COUNT} slots"
        )
    values: dict[int, tuple[float, ...]] = {}
    for index, raw_slot in enumerate(raw_slots):
        if not isinstance(raw_slot, list) or len(raw_slot) != 4:
            raise RuntimeError(
                f"raw path trace slot {index} is not RGBA"
            )
        values[index] = tuple(float(value) for value in raw_slot)
    return _decode_values(values, path)


def _main() -> None:
    arguments = _arguments()
    result = (
        decode_trace(arguments.input)
        if arguments.input.suffix.casefold() == ".exr"
        else decode_raw_trace(arguments.input)
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
        print(f"Decoded path trace: {arguments.output}")


if __name__ == "__main__":
    _main()

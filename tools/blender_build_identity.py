"""Stable Blender build identity shared by export and validation tooling."""

from __future__ import annotations

import pathlib
from typing import Any


FIELDS = (
    "version",
    "version_cycle",
    "version_tuple",
    "build_hash",
    "build_branch",
    "build_type",
)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def current(app: Any) -> dict[str, Any]:
    """Return the semantic build identity used by scene evaluation.

    Build date and time are deliberately excluded: rebuilding the same source
    revision with the same configuration must not invalidate a benchmark.
    """

    return {
        "version": _text(app.version_string),
        "version_cycle": _text(app.version_cycle),
        "version_tuple": [int(value) for value in app.version],
        "build_hash": _text(app.build_hash),
        "build_branch": _text(app.build_branch),
        "build_type": _text(app.build_type),
    }


def from_document(
    document: dict[str, Any],
    source: pathlib.Path,
) -> dict[str, Any]:
    """Parse and validate an exact Blender identity from a JSON document."""

    identity = document.get("blender_build")
    if not isinstance(identity, dict):
        raise RuntimeError(
            f"{source} has no exact Blender build identity; regenerate it "
            "with the current Psycles exporter/golden script"
        )
    missing = [field for field in FIELDS if field not in identity]
    if missing:
        raise RuntimeError(
            f"{source} has an incomplete Blender build identity: "
            + ", ".join(missing)
        )
    version_tuple = identity["version_tuple"]
    if (
        not isinstance(version_tuple, list)
        or len(version_tuple) != 3
        or not all(type(value) is int and value >= 0 for value in version_tuple)
    ):
        raise RuntimeError(
            f"{source} has an invalid Blender version tuple"
        )
    invalid_text_fields = [
        field
        for field in FIELDS
        if field != "version_tuple"
        and (
            not isinstance(identity[field], str)
            or not identity[field]
        )
    ]
    if invalid_text_fields:
        raise RuntimeError(
            f"{source} has invalid Blender build identity fields: "
            + ", ".join(invalid_text_fields)
        )
    return {field: identity[field] for field in FIELDS}


def require_same(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_source: pathlib.Path,
    candidate_source: pathlib.Path,
) -> None:
    """Require two artifacts to have the same scene-evaluation oracle."""

    if candidate != reference:
        raise RuntimeError(
            "comparison inputs use different Blender builds: "
            f"{reference_source} reports {reference}, while "
            f"{candidate_source} reports {candidate}. Re-export and render "
            "the golden with the same Blender executable."
        )

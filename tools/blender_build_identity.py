"""Stable Blender build identity shared by export and golden tooling."""

from __future__ import annotations

from typing import Any


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

"""Content identity for the Blender-to-Psycles scene contract.

The exported JSON is not solely a function of ``export_psycles_scene.py``:
the exporter delegates graph serialization, Blender build identity, and Cycles
hashing to sibling modules.  Reusing a bundle is sound only when all of these
semantic inputs are byte-identical to the implementation that produced it.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any


_SCHEMA = "psycles.exporter-identity.v1"
_SEMANTIC_SOURCES = (
    "export_psycles_scene.py",
    "blender_scene_manifest.py",
    "blender_build_identity.py",
    "blender_particle_hair.py",
    "cycles_hash.py",
    "exporter_identity.py",
)


def current(export_script: pathlib.Path) -> dict[str, Any]:
    """Return a deterministic identity for every local semantic source."""

    export_script = export_script.resolve()
    source_root = export_script.parent
    sources: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for name in _SEMANTIC_SOURCES:
        path = (
            export_script
            if name == "export_psycles_scene.py"
            else source_root / name
        )
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        sources[name] = digest

        encoded_name = name.encode("utf-8")
        aggregate.update(len(encoded_name).to_bytes(4, "little"))
        aggregate.update(encoded_name)
        aggregate.update(len(payload).to_bytes(8, "little"))
        aggregate.update(payload)
    return {
        "schema": _SCHEMA,
        "sha256": aggregate.hexdigest(),
        "sources": sources,
    }


def from_document(
    document: dict[str, Any], source: pathlib.Path
) -> dict[str, Any]:
    """Read and validate an exporter identity from a scene bundle."""

    identity = document.get("exporter")
    if not isinstance(identity, dict):
        raise RuntimeError(
            f"scene bundle has no exact exporter identity: {source}"
        )
    expected_keys = {"schema", "sha256", "sources"}
    if set(identity) != expected_keys:
        raise RuntimeError(
            f"scene bundle has an invalid exporter identity: {source}"
        )
    if identity["schema"] != _SCHEMA:
        raise RuntimeError(
            f"scene bundle has an unsupported exporter identity: {source}"
        )
    if (
        not isinstance(identity["sha256"], str)
        or len(identity["sha256"]) != 64
        or not isinstance(identity["sources"], dict)
        or set(identity["sources"]) != set(_SEMANTIC_SOURCES)
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in identity["sources"].values()
        )
    ):
        raise RuntimeError(
            f"scene bundle has invalid exporter identity fields: {source}"
        )
    return identity


def require_current(
    document: dict[str, Any],
    source: pathlib.Path,
    export_script: pathlib.Path,
) -> dict[str, Any]:
    """Fail closed unless a bundle came from the current exporter closure."""

    recorded = from_document(document, source)
    expected = current(export_script)
    if recorded != expected:
        raise RuntimeError(
            "scene bundle was produced by a different exporter implementation: "
            f"{source}"
        )
    return recorded

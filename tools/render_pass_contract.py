"""Exact linear-pass workload shared by the two benchmark renderers."""

from __future__ import annotations

from collections import Counter
import pathlib


SCHEMA = "psycles.linear-pass-contract.v1"
PASSES = (
    "Combined", "Normal", "DiffCol", "GlossCol", "TransCol", "DiffDir",
    "DiffInd", "GlossDir", "GlossInd", "TransDir", "TransInd", "Emit", "Env",
    "Volume Direct", "Volume Indirect",
)
PASS_ALIASES = {
    "DiffCol": ("DiffCol", "Diffuse Color"),
    "GlossCol": ("GlossCol", "Glossy Color"),
    "TransCol": ("TransCol", "Transmission Color"),
    "DiffDir": ("DiffDir", "Diffuse Direct"),
    "DiffInd": ("DiffInd", "Diffuse Indirect"),
    "GlossDir": ("GlossDir", "Glossy Direct"),
    "GlossInd": ("GlossInd", "Glossy Indirect"),
    "TransDir": ("TransDir", "Transmission Direct"),
    "TransInd": ("TransInd", "Transmission Indirect"),
    "Emit": ("Emit", "Emission"),
    "Env": ("Env", "Environment"),
    "Volume Direct": ("Volume Direct", "VolumeDir"),
    "Volume Indirect": ("Volume Indirect", "VolumeInd"),
}
_CANONICAL_NAMES = {alias: name for name, aliases in PASS_ALIASES.items() for alias in aliases}


def validate_channels(channels: list[str]) -> None:
    """Require one layer and exactly 46 channels, not merely a common subset."""
    expected = Counter(
        f"{name}.{component}"
        for name in PASSES
        for component in ("RGBA" if name == "Combined" else "XYZ" if name == "Normal" else "RGB")
    )
    actual: Counter[str] = Counter()
    layers: set[str] = set()
    for channel in channels:
        parts = channel.rsplit(".", 2)
        if len(parts) not in (2, 3):
            raise RuntimeError(f"unrecognized benchmark channel: {channel!r}")
        layers.add(parts[0] if len(parts) == 3 else "")
        name = _CANONICAL_NAMES.get(parts[-2], parts[-2])
        actual[f"{name}.{parts[-1]}"] += 1
    if len(layers) != 1 or actual != expected:
        raise RuntimeError(
            "unequal benchmark pass workload: "
            f"layers={sorted(layers)}, extra={list((actual - expected).elements())}, "
            f"missing={list((expected - actual).elements())}"
        )


def inspect_image(path: pathlib.Path) -> dict:
    """Read EXR headers only; this is outside either renderer's timed interval."""
    import OpenImageIO as oiio

    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(f"could not inspect benchmark image: {path}")
    channels: list[str] = []
    extent = None
    try:
        subimage = 0
        while source.seek_subimage(subimage, 0):
            spec = source.spec()
            current = (spec.width, spec.height)
            if extent is not None and current != extent:
                raise RuntimeError(f"inconsistent benchmark subimage extent: {path}")
            extent = current
            channels.extend(spec.channelnames)
            subimage += 1
    finally:
        source.close()
    validate_channels(channels)
    assert extent is not None
    return {"schema": SCHEMA, "passes": list(PASSES), "channels": channels,
            "width": extent[0], "height": extent[1]}

"""Psycles Render Engine Blender extension package."""

from __future__ import annotations

import atexit

from . import bridge, engine, properties, ui


def register() -> None:
    properties.register()
    ui.register()
    engine.register()
    atexit.unregister(bridge.shutdown_all)
    atexit.register(bridge.shutdown_all)


def unregister() -> None:
    engine.unregister()
    ui.unregister()
    properties.unregister()
    bridge.shutdown_all()

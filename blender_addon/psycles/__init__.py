"""Psycles Render Engine Blender extension package."""

from __future__ import annotations

import atexit

import bpy

from . import bridge, engine, properties, scene_cache, ui


@bpy.app.handlers.persistent
def _on_load_pre(_dummy: object = None) -> None:
    engine.cancel_all_viewports(wait=False)
    scene_cache.clear_all()


@bpy.app.handlers.persistent
def _on_depsgraph_update_post(
    _scene: object = None, depsgraph: object = None
) -> None:
    if depsgraph is not None:
        scene_cache.get_mesh_cache().invalidate_from_depsgraph(depsgraph)


def register() -> None:
    properties.register()
    ui.register()
    engine.register()
    if _on_load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_on_load_pre)
    if _on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)
    atexit.unregister(bridge.shutdown_all)
    atexit.register(bridge.shutdown_all)


def unregister() -> None:
    if _on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update_post)
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    engine.unregister()
    ui.unregister()
    properties.unregister()
    bridge.shutdown_all()

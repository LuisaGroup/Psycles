"""Render properties panels and native Cycles UI panel compatibility."""

from __future__ import annotations

from typing import Any

import bpy

from . import bridge
from .properties import get_preferences

_MISSING = object()
_PATCHED_COMPAT: list[tuple[type, Any]] = []
_PATCHED_POLL: list[tuple[type, Any]] = []

_EXCLUDED_PANEL_PREFIXES = (
    "CYCLES_RENDER_PT_sampling",
    "CYCLES_RENDER_PT_light_paths",
    "CYCLES_RENDER_PT_film",
    "CYCLES_RENDER_PT_performance",
    "CYCLES_RENDER_PT_bake",
    "CYCLES_RENDER_PT_debug",
    "CYCLES_RENDER_PT_motion_blur",
    "CYCLES_RENDER_PT_simplify",
    "CYCLES_PT_",
    "CYCLES_VIEW3D_PT_",
)

_EXCLUDED_PANEL_EXACT = {
    "CYCLES_RENDER_PT_subdivision",
    "CYCLES_RENDER_PT_volumes",
    "CYCLES_OBJECT_PT_motion_blur",
}

_RENDER_COMPAT_BLACKLIST = {
    "RENDER_PT_eevee_",
    "RENDER_PT_gpencil",
    "RENDER_PT_grease_pencil",
}


class _PsyclesPanelBase:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"PSYCLES"}

    @classmethod
    def poll(cls, context: Any) -> bool:
        return context.engine in cls.COMPAT_ENGINES


class PSYCLES_RENDER_PT_device(_PsyclesPanelBase, bpy.types.Panel):
    bl_label = "Device & Backend"
    bl_order = 1

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        ps = context.scene.psycles
        layout.prop(ps, "backend")
        layout.prop(ps, "scheduler")
        if ps.backend in {"AUTO", "hip"}:
            layout.prop(ps, "hip_visible_device")

        col = layout.column(heading="Compiler & Execution")
        col.prop(ps, "enable_fast_math")
        col.prop(ps, "staged_surface_sorting")
        col.prop(ps, "staged_direct_light_queue")

        prefs = get_preferences(context)
        prefs_dir = prefs.runtime_library_dir if prefs is not None else ""
        available = bridge.probe_backends(prefs_dir)
        info_text = (
            "Available: " + ", ".join(available)
            if available
            else "Runtime libraries not detected (check Add-on Preferences)"
        )
        row = layout.row()
        row.enabled = False
        row.label(text=info_text)


class PSYCLES_RENDER_PT_sampling(_PsyclesPanelBase, bpy.types.Panel):
    bl_label = "Sampling"
    bl_order = 2

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles
        ps = scene.psycles

        col = layout.column(align=True)
        col.prop(cscene, "samples", text="Render Samples")
        col.prop(cscene, "preview_samples", text="Viewport Samples")

        col = layout.column(align=True)
        col.prop(cscene, "seed")
        col.prop(cscene, "use_animated_seed")

        col = layout.column(align=True)
        col.prop(ps, "progressive_chunk_samples")
        col.prop(ps, "viewport_start_samples")


class PSYCLES_RENDER_PT_light_paths(_PsyclesPanelBase, bpy.types.Panel):
    bl_label = "Light Paths"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        cscene = context.scene.cycles
        col = layout.column(heading="Max Bounces", align=True)
        col.prop(cscene, "max_bounces", text="Total")
        col.prop(cscene, "diffuse_bounces", text="Diffuse")
        col.prop(cscene, "glossy_bounces", text="Glossy")
        col.prop(cscene, "transmission_bounces", text="Transmission")
        col.prop(cscene, "volume_bounces", text="Volume")
        col.prop(cscene, "transparent_max_bounces", text="Transparent")

        col = layout.column(heading="Clamping", align=True)
        col.prop(cscene, "sample_clamp_direct", text="Direct Light")
        col.prop(cscene, "sample_clamp_indirect", text="Indirect Light")

        col = layout.column(heading="Caustics & Sampling")
        col.prop(cscene, "caustics_reflective")
        col.prop(cscene, "caustics_refractive")
        col.prop(cscene, "blur_glossy", text="Filter Glossy")
        col.prop(cscene, "use_light_tree")


class PSYCLES_RENDER_PT_film(_PsyclesPanelBase, bpy.types.Panel):
    bl_label = "Film"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        scene = context.scene
        cscene = scene.cycles
        layout.prop(scene.render, "film_transparent")
        layout.prop(cscene, "film_exposure", text="Exposure")
        layout.prop(cscene, "pixel_filter_type", text="Filter Type")
        if cscene.pixel_filter_type != "BOX":
            layout.prop(cscene, "filter_width", text="Filter Width")


_CUSTOM_CLASSES = (
    PSYCLES_RENDER_PT_device,
    PSYCLES_RENDER_PT_sampling,
    PSYCLES_RENDER_PT_light_paths,
    PSYCLES_RENDER_PT_film,
)


def _all_panel_subclasses() -> list[type]:
    found: list[type] = []
    seen: set[type] = set()
    stack = [bpy.types.Panel]
    while stack:
        parent = stack.pop()
        for child in parent.__subclasses__():
            if child not in seen:
                seen.add(child)
                found.append(child)
                stack.append(child)
    return found


def _is_excluded_panel(name: str) -> bool:
    if name in _EXCLUDED_PANEL_EXACT:
        return True
    if name == "CYCLES_PT_post_processing":
        return False
    if name == "CYCLES_PT_context_material":
        return False
    if any(name.startswith(prefix) for prefix in _EXCLUDED_PANEL_PREFIXES):
        return True
    if any(name.startswith(prefix) for prefix in _RENDER_COMPAT_BLACKLIST):
        return True
    return False


def _save_and_set_compat(cls: type, engines: set[str]) -> None:
    if any(entry[0] is cls for entry in _PATCHED_COMPAT):
        return
    original = cls.__dict__.get("COMPAT_ENGINES", _MISSING)
    _PATCHED_COMPAT.append((cls, set(original) if original is not _MISSING else _MISSING))
    cls.COMPAT_ENGINES = engines


def _wrap_excluded_poll(cls: type) -> None:
    if "poll" not in cls.__dict__ or any(entry[0] is cls for entry in _PATCHED_POLL):
        return
    orig_descriptor = cls.__dict__["poll"]
    _PATCHED_POLL.append((cls, orig_descriptor))
    orig_func = cls.poll.__func__

    @classmethod  # type: ignore[misc]
    def _guarded_poll(inner_cls: type, context: Any) -> bool:
        if context.engine not in inner_cls.COMPAT_ENGINES:
            return False
        return bool(orig_func(inner_cls, context))

    cls.poll = _guarded_poll


def register() -> None:
    for cls in _CUSTOM_CLASSES:
        bpy.utils.register_class(cls)

    panels = _all_panel_subclasses()
    cycles_bases: set[type] = set()
    for panel in panels:
        if "CYCLES" not in getattr(panel, "COMPAT_ENGINES", ()):
            continue
        for base in panel.__mro__:
            if base.__name__ in {"CyclesButtonsPanel", "CyclesDebugButtonsPanel", "CyclesShadingButtonsPanel"}:
                cycles_bases.add(base)

    for base in cycles_bases:
        if "COMPAT_ENGINES" in base.__dict__ and "CYCLES" in base.COMPAT_ENGINES:
            new_set = set(base.COMPAT_ENGINES)
            new_set.add("PSYCLES")
            _save_and_set_compat(base, new_set)

    for panel in panels:
        name = panel.__name__
        if name.startswith("PSYCLES_"):
            continue
        compat = getattr(panel, "COMPAT_ENGINES", None)
        if compat is None or "CYCLES" not in compat:
            continue
        bare_name = name[5:] if name.startswith("NODE_") else name
        if _is_excluded_panel(name) or _is_excluded_panel(bare_name):
            _save_and_set_compat(panel, {"CYCLES"})
            _wrap_excluded_poll(panel)
            continue
        if "COMPAT_ENGINES" in panel.__dict__:
            new_set = set(panel.__dict__["COMPAT_ENGINES"])
            new_set.add("PSYCLES")
            _save_and_set_compat(panel, new_set)


def unregister() -> None:
    for cls, orig_poll in reversed(_PATCHED_POLL):
        if orig_poll is _MISSING:
            if "poll" in cls.__dict__:
                delattr(cls, "poll")
        else:
            setattr(cls, "poll", orig_poll)
    _PATCHED_POLL.clear()

    for cls, orig_compat in reversed(_PATCHED_COMPAT):
        if orig_compat is _MISSING:
            if "COMPAT_ENGINES" in cls.__dict__:
                delattr(cls, "COMPAT_ENGINES")
        else:
            cls.COMPAT_ENGINES = orig_compat
    _PATCHED_COMPAT.clear()

    for cls in reversed(_CUSTOM_CLASSES):
        bpy.utils.unregister_class(cls)

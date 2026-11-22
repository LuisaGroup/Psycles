"""Addon preferences and scene render properties for the Psycles Blender plugin."""

from __future__ import annotations

from typing import Any

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import bridge


def get_preferences(context: Any | None = None) -> "PsyclesAddonPreferences | None":
    ctx = context if context is not None else getattr(bpy, "context", None)
    prefs = getattr(ctx, "preferences", None)
    if prefs is None:
        return None
    addons = getattr(prefs, "addons", None)
    if addons is None:
        return None
    addon_entry = addons.get(__package__)
    if addon_entry is None:
        return None
    return addon_entry.preferences


class PsyclesAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    runtime_library_dir: StringProperty(
        name="Runtime Library Directory",
        description="Optional override path to directory containing psycles_blender and Luisa backends",
        subtype="DIR_PATH",
        default="",
    )
    cache_dir: StringProperty(
        name="Shader Cache Directory",
        description="Directory for compiled Luisa/LLVM/SPIR-V/HIP shader caches (blank = Blender user config)",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.prop(self, "runtime_library_dir")
        layout.prop(self, "cache_dir")

        box = layout.box()
        try:
            rt = bridge.resolve_runtime_dir(self.runtime_library_dir)
            backends = bridge.probe_backends(str(rt))
            box.label(text=f"Runtime Directory: {rt}", icon="CHECKMARK")
            box.label(
                text="Detected Backends: " + (", ".join(backends) if backends else "none")
            )
        except Exception as err:
            box.label(text=f"Runtime not found: {err}", icon="ERROR")
        box.label(text=f"Active Cache Directory: {bridge.default_cache_dir(self.cache_dir)}")


class PsyclesSceneSettings(bpy.types.PropertyGroup):
    backend: EnumProperty(
        name="Backend",
        description="LuisaCompute execution backend",
        items=[
            ("AUTO", "Automatic", "Select fastest available GPU or CPU backend (vk > dx > hip > metal > simd > fallback)"),
            ("vk", "Vulkan", "Vulkan SPIR-V GPU backend"),
            ("dx", "DirectX 12", "DirectX 12 SM 6.6+ GPU backend"),
            ("hip", "ROCm HIP", "AMD ROCm HIP / HIPRT GPU backend"),
            ("metal", "Metal", "Apple Metal GPU backend"),
            ("metal4", "Metal 4", "Apple Metal 4 AIR GPU backend"),
            ("simd", "CPU SIMD", "LLVM/Embree packet-vectorized CPU SIMD backend"),
            ("fallback", "CPU Fallback", "LLVM/Embree scalar CPU backend"),
        ],
        default="AUTO",
    )
    scheduler: EnumProperty(
        name="Scheduler",
        description="Path tracing kernel scheduling mode",
        items=[
            ("AUTO", "Automatic", "wavefront_staged on GPU backends, megakernel on CPU"),
            ("WAVEFRONT_STAGED", "Wavefront Staged", "Coroutine-split sorted GPU shader stages"),
            ("WAVEFRONT", "Wavefront", "Queue-based GPU wavefront scheduler"),
            ("MEGAKERNEL", "Megakernel", "Single persistent loop over bounces"),
            ("PERSISTENT", "Persistent", "Work-stealing persistent GPU megakernel"),
        ],
        default="AUTO",
    )
    progressive_chunk_samples: IntProperty(
        name="Progressive Update Samples",
        description="Samples rendered per progressive final-render UI refresh",
        default=16,
        min=1,
        max=4096,
    )
    viewport_start_samples: IntProperty(
        name="Viewport Start Samples",
        description="Samples dispatched on the very first interactive viewport update",
        default=1,
        min=1,
        max=16,
    )
    hip_visible_device: IntProperty(
        name="HIP Device Index",
        description="GPU device index for ROCm HIP (-1 uses HIP_VISIBLE_DEVICES / default discrete GPU)",
        default=-1,
        min=-1,
        max=16,
    )
    enable_fast_math: BoolProperty(
        name="Fast Math",
        description="Allow backend fast-math optimizations during shader compilation",
        default=True,
    )
    staged_surface_sorting: BoolProperty(
        name="Staged Surface Sorting",
        description="Sort rays by Cycles shader ID inside wavefront-staged evaluation",
        default=True,
    )
    staged_direct_light_queue: BoolProperty(
        name="Staged Direct Light Queue",
        description="Evaluate shadow/NEE rays in a dedicated staged queue",
        default=True,
    )


_CLASSES = (
    PsyclesAddonPreferences,
    PsyclesSceneSettings,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.psycles = PointerProperty(type=PsyclesSceneSettings)


def unregister() -> None:
    if hasattr(bpy.types.Scene, "psycles"):
        del bpy.types.Scene.psycles
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

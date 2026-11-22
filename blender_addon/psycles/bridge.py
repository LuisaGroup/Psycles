"""Native C ABI bindings to libpsycles_blender / psycles_blender.dll."""

from __future__ import annotations

import atexit
import ctypes
from dataclasses import dataclass
import os
import pathlib
import sys
import tempfile
import threading
from typing import Any

import numpy as np

PSYCLES_CAPI_VERSION = 1

PSYCLES_SCHEDULER_AUTO = 0xFFFFFFFF
PSYCLES_SCHEDULER_MEGAKERNEL = 0
PSYCLES_SCHEDULER_MEGAKERNEL_PER_SAMPLE = 1
PSYCLES_SCHEDULER_WAVEFRONT = 2
PSYCLES_SCHEDULER_WAVEFRONT_GRAPH = 3
PSYCLES_SCHEDULER_WAVEFRONT_STAGED = 4
PSYCLES_SCHEDULER_PERSISTENT = 5

PSYCLES_PROJECTION_PERSPECTIVE = 0
PSYCLES_PROJECTION_ORTHOGRAPHIC = 1
PSYCLES_PROJECTION_PANORAMA = 2

PSYCLES_SENSOR_FIT_AUTO = 0
PSYCLES_SENSOR_FIT_HORIZONTAL = 1
PSYCLES_SENSOR_FIT_VERTICAL = 2

PSYCLES_PIXEL_FILTER_BOX = 0
PSYCLES_PIXEL_FILTER_GAUSSIAN = 1
PSYCLES_PIXEL_FILTER_BLACKMAN_HARRIS = 2
PSYCLES_PIXEL_FILTER_FROM_SCENE = 255

# Matches psycles::contract::PassKind in include/psycles/contract/render.h.
PASS_KIND_COMBINED = 0
PASS_KIND_DEPTH = 1
PASS_KIND_NORMAL = 2
PASS_KIND_POSITION = 3
PASS_KIND_ALBEDO = 4
PASS_KIND_EMISSION = 5
PASS_KIND_DIFFUSE_DIRECT = 6
PASS_KIND_DIFFUSE_INDIRECT = 7
PASS_KIND_GLOSSY_DIRECT = 8
PASS_KIND_GLOSSY_INDIRECT = 9
PASS_KIND_TRANSMISSION_DIRECT = 10
PASS_KIND_TRANSMISSION_INDIRECT = 11
PASS_KIND_VOLUME_DIRECT = 12
PASS_KIND_VOLUME_INDIRECT = 13
PASS_KIND_MIST = 14
PASS_KIND_SHADOW_CATCHER = 15
PASS_KIND_CRYPTOMATTE_OBJECT = 16
PASS_KIND_CRYPTOMATTE_MATERIAL = 17
PASS_KIND_DENOISING_NORMAL = 18
PASS_KIND_DENOISING_ALBEDO = 19
PASS_KIND_SAMPLE_COUNT = 20
PASS_KIND_AOV_COLOR = 21
PASS_KIND_AOV_VALUE = 22
PASS_KIND_GLOSSY_COLOR = 23
PASS_KIND_TRANSMISSION_COLOR = 24
PASS_KIND_ENVIRONMENT = 25

SCHEDULER_MAP = {
    "AUTO": PSYCLES_SCHEDULER_AUTO,
    "WAVEFRONT_STAGED": PSYCLES_SCHEDULER_WAVEFRONT_STAGED,
    "WAVEFRONT": PSYCLES_SCHEDULER_WAVEFRONT,
    "MEGAKERNEL": PSYCLES_SCHEDULER_MEGAKERNEL,
    "PERSISTENT": PSYCLES_SCHEDULER_PERSISTENT,
}

PIXEL_FILTER_MAP = {
    "BOX": PSYCLES_PIXEL_FILTER_BOX,
    "GAUSSIAN": PSYCLES_PIXEL_FILTER_GAUSSIAN,
    "BLACKMAN_HARRIS": PSYCLES_PIXEL_FILTER_BLACKMAN_HARRIS,
}


class PsyclesEngineConfig(ctypes.Structure):
    _fields_ = [
        ("cache_dir", ctypes.c_char_p),
        ("scheduler", ctypes.c_uint32),
        ("wavefront_frame_capacity", ctypes.c_uint32),
        ("wavefront_execution_block_size", ctypes.c_uint32),
        ("max_samples_per_dispatch", ctypes.c_uint32),
        ("enable_fast_math", ctypes.c_uint8),
        ("staged_surface_sorting", ctypes.c_uint8),
        ("staged_direct_light_queue", ctypes.c_uint8),
        ("vulkan_require_native_xir", ctypes.c_uint8),
        ("hip_visible_device", ctypes.c_int32),
    ]


class PsyclesSceneInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("samples", ctypes.c_uint32),
        ("seed", ctypes.c_uint32),
        ("geometry_count", ctypes.c_uint32),
        ("instance_count", ctypes.c_uint32),
        ("material_count", ctypes.c_uint32),
        ("light_count", ctypes.c_uint32),
        ("transparent_background", ctypes.c_uint8),
    ]


class PsyclesPassDesc(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16),
        ("channels", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("light_group", ctypes.c_char_p),
    ]


class PsyclesCameraDesc(ctypes.Structure):
    _fields_ = [
        ("projection", ctypes.c_uint32),
        ("sensor_fit", ctypes.c_uint32),
        ("transform", ctypes.c_float * 16),
        ("field_of_view", ctypes.c_float),
        ("horizontal_field_of_view", ctypes.c_float),
        ("orthographic_scale", ctypes.c_float),
        ("lens_shift_x", ctypes.c_float),
        ("lens_shift_y", ctypes.c_float),
        ("near_clip", ctypes.c_float),
        ("far_clip", ctypes.c_float),
        ("aperture_radius", ctypes.c_float),
        ("focal_distance", ctypes.c_float),
        ("aperture_blades", ctypes.c_uint32),
        ("aperture_rotation", ctypes.c_float),
        ("aperture_ratio", ctypes.c_float),
        ("has_sensor", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("sensor_width_mm", ctypes.c_float),
        ("sensor_height_mm", ctypes.c_float),
        ("sensor_lens_mm", ctypes.c_float),
        ("pixel_aspect_x", ctypes.c_float),
        ("pixel_aspect_y", ctypes.c_float),
    ]


class PsyclesRenderParams(ctypes.Structure):
    _fields_ = [
        ("full_width", ctypes.c_uint32),
        ("full_height", ctypes.c_uint32),
        ("window_x", ctypes.c_uint32),
        ("window_y", ctypes.c_uint32),
        ("window_width", ctypes.c_uint32),
        ("window_height", ctypes.c_uint32),
        ("seed", ctypes.c_uint32),
        ("override_seed", ctypes.c_uint8),
        ("transparent_background", ctypes.c_uint8),
        ("override_transparent_background", ctypes.c_uint8),
        ("pixel_filter", ctypes.c_uint8),
        ("filter_width", ctypes.c_float),
        ("pass_alpha_threshold", ctypes.c_float),
        ("override_pass_alpha_threshold", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


def _shared_lib_filename() -> str:
    if sys.platform == "win32":
        return "psycles_blender.dll"
    if sys.platform == "darwin":
        return "libpsycles_blender.dylib"
    return "libpsycles_blender.so"


def resolve_runtime_dir(prefs_dir: str = "") -> pathlib.Path:
    lib_name = _shared_lib_filename()
    addon_dir = pathlib.Path(__file__).resolve().parent
    candidates: list[pathlib.Path] = []
    if prefs_dir:
        candidates.append(pathlib.Path(prefs_dir))
    env_dir = os.environ.get("PSYCLES_BLENDER_LIB_DIR", "").strip()
    if env_dir:
        candidates.append(pathlib.Path(env_dir))
    candidates.extend([
        addon_dir / "bin",
        addon_dir.parents[1] / "build_main" / "bin",
        addon_dir.parents[1] / "build" / "bin",
        addon_dir.parents[2] / "build_main" / "bin",
    ])
    for candidate in candidates:
        resolved = candidate.expanduser()
        if (resolved / lib_name).is_file():
            return resolved.resolve()
        if (resolved / "bin" / lib_name).is_file():
            return (resolved / "bin").resolve()
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find {lib_name}. Searched: {searched}. "
        "Set Psycles Add-on Preferences -> Runtime Directory or PSYCLES_BLENDER_LIB_DIR."
    )


def default_cache_dir(prefs_cache_dir: str = "") -> pathlib.Path:
    if prefs_cache_dir:
        p = pathlib.Path(prefs_cache_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    try:
        import bpy

        user_cfg = bpy.utils.user_resource("CONFIG", path="psycles_cache", create=True)
        if user_cfg:
            return pathlib.Path(user_cfg).resolve()
    except Exception:
        pass
    fallback = pathlib.Path(tempfile.gettempdir()) / "psycles_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


_LIB_LOCK = threading.RLock()
_LOADED_LIBS: dict[pathlib.Path, ctypes.CDLL] = {}
_DLL_DIR_HANDLES: dict[pathlib.Path, Any] = {}
_ENGINE_CACHE: dict[tuple[Any, ...], "PsyclesEngine"] = {}
_AUTO_BACKEND_ORDER = ("vk", "dx", "hip", "metal4", "metal", "simd", "fallback")


def _load_library(runtime_dir: pathlib.Path) -> ctypes.CDLL:
    runtime_dir = runtime_dir.resolve()
    with _LIB_LOCK:
        cached = _LOADED_LIBS.get(runtime_dir)
        if cached is not None:
            return cached
        lib_path = runtime_dir / _shared_lib_filename()
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            for directory in (runtime_dir, runtime_dir.parent):
                if directory.is_dir() and directory not in _DLL_DIR_HANDLES:
                    _DLL_DIR_HANDLES[directory] = os.add_dll_directory(str(directory))
        lib = ctypes.CDLL(str(lib_path))
        lib.psycles_capi_version.argtypes = []
        lib.psycles_capi_version.restype = ctypes.c_uint32

        lib.psycles_probe_backends.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.psycles_probe_backends.restype = ctypes.c_uint32

        lib.psycles_engine_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(PsyclesEngineConfig),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.psycles_engine_create.restype = ctypes.c_void_p

        lib.psycles_engine_destroy.argtypes = [ctypes.c_void_p]
        lib.psycles_engine_destroy.restype = None

        lib.psycles_scene_compile_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.POINTER(PsyclesSceneInfo),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.psycles_scene_compile_memory.restype = ctypes.c_void_p

        lib.psycles_scene_destroy.argtypes = [ctypes.c_void_p]
        lib.psycles_scene_destroy.restype = None

        lib.psycles_session_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(PsyclesRenderParams),
            ctypes.POINTER(PsyclesPassDesc),
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.psycles_session_create.restype = ctypes.c_void_p

        lib.psycles_session_update_camera.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PsyclesCameraDesc),
            ctypes.POINTER(PsyclesRenderParams),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.psycles_session_update_camera.restype = ctypes.c_int

        lib.psycles_session_render_samples.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint8,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.psycles_session_render_samples.restype = ctypes.c_int

        lib.psycles_session_get_pass_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.psycles_session_get_pass_buffer.restype = ctypes.c_int

        lib.psycles_session_get_last_timings.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.psycles_session_get_last_timings.restype = ctypes.c_int

        lib.psycles_session_cancel.argtypes = [ctypes.c_void_p]
        lib.psycles_session_cancel.restype = None

        lib.psycles_session_destroy.argtypes = [ctypes.c_void_p]
        lib.psycles_session_destroy.restype = None

        ver = int(lib.psycles_capi_version())
        if ver != PSYCLES_CAPI_VERSION:
            raise RuntimeError(
                f"psycles_blender C API version mismatch: got {ver}, expected {PSYCLES_CAPI_VERSION}"
            )
        _LOADED_LIBS[runtime_dir] = lib
        return lib


def probe_backends(prefs_dir: str = "") -> list[str]:
    try:
        runtime_dir = resolve_runtime_dir(prefs_dir)
        lib = _load_library(runtime_dir)
    except Exception:
        return []
    buf = ctypes.create_string_buffer(1024)
    lib.psycles_probe_backends(str(runtime_dir).encode("utf-8"), buf, len(buf))
    raw = buf.value.decode("utf-8", errors="replace").strip()
    return [item for item in raw.split(",") if item]


def resolve_backend(requested: str, prefs_dir: str = "") -> str:
    available = probe_backends(prefs_dir)
    if requested != "AUTO":
        if available and requested not in available:
            raise RuntimeError(
                f"Requested backend {requested!r} is unavailable. Available: {available}"
            )
        return requested
    for candidate in _AUTO_BACKEND_ORDER:
        if candidate in available:
            return candidate
    if available:
        return available[0]
    return "fallback"


@dataclass(frozen=True)
class RenderTimings:
    integration_s: float
    pass_output_s: float
    render_call_s: float


class PsyclesEngine:
    def __init__(
        self,
        runtime_dir: pathlib.Path,
        backend_name: str,
        config: PsyclesEngineConfig,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.backend_name = backend_name
        self._lib = _load_library(runtime_dir)
        err = ctypes.create_string_buffer(2048)
        ptr = self._lib.psycles_engine_create(
            str(runtime_dir).encode("utf-8"),
            backend_name.encode("utf-8"),
            ctypes.byref(config),
            err,
            len(err),
        )
        if not ptr:
            raise RuntimeError(
                err.value.decode("utf-8", errors="replace")
                or f"Failed to initialize Psycles backend {backend_name!r}"
            )
        self._handle = ctypes.c_void_p(ptr)

    def compile_scene_memory(
        self,
        scene_json: bytes,
        geometry_bin: bytes | bytearray | memoryview,
        asset_base_dir: str = "",
    ) -> tuple["PsyclesScene", PsyclesSceneInfo]:
        if not self._handle:
            raise RuntimeError("PsyclesEngine has been destroyed")
        if isinstance(geometry_bin, (bytearray, memoryview)):
            mv = memoryview(geometry_bin)
            if mv.readonly:
                geom_arg: Any = ctypes.cast(
                    ctypes.c_char_p(bytes(mv)), ctypes.c_void_p
                )
            else:
                geom_arr = (ctypes.c_ubyte * len(mv)).from_buffer(mv)
                geom_arg = ctypes.cast(geom_arr, ctypes.c_void_p)
            geom_len = len(mv)
        else:
            geom_arg = ctypes.cast(
                ctypes.c_char_p(geometry_bin), ctypes.c_void_p
            )
            geom_len = len(geometry_bin)
        info = PsyclesSceneInfo()
        err = ctypes.create_string_buffer(2048)
        ptr = self._lib.psycles_scene_compile_memory(
            self._handle,
            scene_json,
            len(scene_json),
            geom_arg,
            geom_len,
            asset_base_dir.encode("utf-8"),
            ctypes.byref(info),
            err,
            len(err),
        )
        if not ptr:
            raise RuntimeError(
                err.value.decode("utf-8", errors="replace")
                or "Scene compilation failed"
            )
        return PsyclesScene(self, ctypes.c_void_p(ptr)), info

    def destroy(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.psycles_engine_destroy(handle)
            self._handle = None

    def __del__(self) -> None:
        self.destroy()


class PsyclesScene:
    def __init__(self, engine: PsyclesEngine, handle: ctypes.c_void_p) -> None:
        self._engine = engine
        self._lib = engine._lib
        self._handle = handle

    def create_session(
        self,
        params: PsyclesRenderParams,
        passes: list[tuple[int, int, str, str]],
    ) -> "PsyclesSession":
        if not self._handle or not self._engine._handle:
            raise RuntimeError("PsyclesScene or PsyclesEngine has been destroyed")
        keepalive: list[bytes] = []
        pass_arr = (PsyclesPassDesc * len(passes))()
        for idx, (kind, channels, name, light_group) in enumerate(passes):
            name_b = name.encode("utf-8")
            keepalive.append(name_b)
            lg_b = light_group.encode("utf-8") if light_group else b""
            if lg_b:
                keepalive.append(lg_b)
            pass_arr[idx].kind = kind
            pass_arr[idx].reserved = 0
            pass_arr[idx].channels = channels
            pass_arr[idx].name = name_b
            pass_arr[idx].light_group = lg_b if lg_b else None
        err = ctypes.create_string_buffer(2048)
        ptr = self._lib.psycles_session_create(
            self._engine._handle,
            self._handle,
            ctypes.byref(params),
            pass_arr if passes else None,
            len(passes),
            err,
            len(err),
        )
        if not ptr:
            raise RuntimeError(
                err.value.decode("utf-8", errors="replace")
                or "Session creation failed"
            )
        return PsyclesSession(self, ctypes.c_void_p(ptr), len(passes))

    def destroy(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.psycles_scene_destroy(handle)
            self._handle = None

    def __del__(self) -> None:
        self.destroy()


class PsyclesSession:
    def __init__(
        self,
        scene: PsyclesScene,
        handle: ctypes.c_void_p,
        pass_count: int,
    ) -> None:
        self._scene = scene
        self._lib = scene._lib
        self._handle = handle
        self._pass_count = pass_count

    def update_camera(
        self,
        camera: PsyclesCameraDesc,
        params: PsyclesRenderParams,
    ) -> None:
        if not self._handle:
            raise RuntimeError("PsyclesSession has been destroyed")
        err = ctypes.create_string_buffer(2048)
        rc = self._lib.psycles_session_update_camera(
            self._handle,
            ctypes.byref(camera),
            ctypes.byref(params),
            err,
            len(err),
        )
        if rc != 1:
            raise RuntimeError(
                err.value.decode("utf-8", errors="replace")
                or "Session update_camera failed"
            )

    def render_samples(
        self,
        sample_first: int,
        sample_count: int,
        total_samples: int,
        *,
        flip_y: bool = True,
    ) -> bool:
        if not self._handle:
            return False
        err = ctypes.create_string_buffer(2048)
        rc = self._lib.psycles_session_render_samples(
            self._handle,
            sample_first,
            sample_count,
            total_samples,
            1 if flip_y else 0,
            err,
            len(err),
        )
        if rc == 1:
            return True
        if rc == 0:
            return False
        raise RuntimeError(
            err.value.decode("utf-8", errors="replace")
            or "Render samples failed"
        )

    def get_pass_array(self, pass_index: int) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not self._handle or pass_index < 0 or pass_index >= self._pass_count:
            raise IndexError(f"Invalid pass index {pass_index}")
        ptr = ctypes.POINTER(ctypes.c_float)()
        w = ctypes.c_uint32()
        h = ctypes.c_uint32()
        ch = ctypes.c_uint32()
        ok = self._lib.psycles_session_get_pass_buffer(
            self._handle,
            pass_index,
            ctypes.byref(ptr),
            ctypes.byref(w),
            ctypes.byref(h),
            ctypes.byref(ch),
        )
        if not ok or not ptr:
            raise RuntimeError(f"Pass buffer {pass_index} unavailable")
        width = int(w.value)
        height = int(h.value)
        channels = int(ch.value)
        return np.ctypeslib.as_array(ptr, shape=(height, width, channels))

    def last_timings(self) -> RenderTimings | None:
        if not self._handle:
            return None
        integ = ctypes.c_double()
        pass_out = ctypes.c_double()
        call_s = ctypes.c_double()
        ok = self._lib.psycles_session_get_last_timings(
            self._handle,
            ctypes.byref(integ),
            ctypes.byref(pass_out),
            ctypes.byref(call_s),
        )
        if not ok:
            return None
        return RenderTimings(
            integration_s=float(integ.value),
            pass_output_s=float(pass_out.value),
            render_call_s=float(call_s.value),
        )

    def cancel(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.psycles_session_cancel(handle)

    def destroy(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.psycles_session_destroy(handle)
            self._handle = None

    def __del__(self) -> None:
        self.destroy()


def get_or_create_engine(
    *,
    prefs_dir: str = "",
    prefs_cache_dir: str = "",
    backend: str = "AUTO",
    scheduler: str = "AUTO",
    enable_fast_math: bool = True,
    staged_surface_sorting: bool = True,
    staged_direct_light_queue: bool = True,
    hip_visible_device: int = -1,
) -> PsyclesEngine:
    runtime_dir = resolve_runtime_dir(prefs_dir)
    resolved_backend = resolve_backend(backend, str(runtime_dir))
    cache_dir = default_cache_dir(prefs_cache_dir)
    sched_id = SCHEDULER_MAP.get(scheduler, PSYCLES_SCHEDULER_AUTO)
    key = (
        str(runtime_dir),
        resolved_backend,
        sched_id,
        str(cache_dir),
        bool(enable_fast_math),
        bool(staged_surface_sorting),
        bool(staged_direct_light_queue),
        int(hip_visible_device),
    )
    with _LIB_LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is not None and engine._handle:
            return engine
        cfg = PsyclesEngineConfig(
            cache_dir=str(cache_dir).encode("utf-8"),
            scheduler=sched_id,
            wavefront_frame_capacity=0,
            wavefront_execution_block_size=0,
            max_samples_per_dispatch=0,
            enable_fast_math=1 if enable_fast_math else 0,
            staged_surface_sorting=1 if staged_surface_sorting else 0,
            staged_direct_light_queue=1 if staged_direct_light_queue else 0,
            vulkan_require_native_xir=1,
            hip_visible_device=int(hip_visible_device),
        )
        engine = PsyclesEngine(runtime_dir, resolved_backend, cfg)
        _ENGINE_CACHE[key] = engine
        return engine


def shutdown_all() -> None:
    with _LIB_LOCK:
        for engine in list(_ENGINE_CACHE.values()):
            engine.destroy()
        _ENGINE_CACHE.clear()


atexit.register(shutdown_all)

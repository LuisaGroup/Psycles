"""Psycles RenderEngine implementation for final renders, previews, and 3D viewport."""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np

from . import bridge, scene_sync
from .properties import get_preferences


def _collect_passes(
    view_layer: Any,
    *,
    is_preview: bool = False,
) -> list[tuple[int, int, str, str, str]]:
    """Return (PassKind, channels, blender_name, chan_ids, chan_type) tuples."""
    passes: list[tuple[int, int, str, str, str]] = [
        (bridge.PASS_KIND_COMBINED, 4, "Combined", "RGBA", "COLOR"),
    ]
    if is_preview or view_layer is None:
        return passes

    cview = getattr(view_layer, "cycles", None)
    if getattr(view_layer, "use_pass_normal", False):
        passes.append((bridge.PASS_KIND_NORMAL, 3, "Normal", "XYZ", "VECTOR"))
    if getattr(view_layer, "use_pass_diffuse_color", False):
        passes.append((bridge.PASS_KIND_ALBEDO, 3, "Diffuse Color", "RGB", "COLOR"))
    if getattr(view_layer, "use_pass_glossy_color", False):
        passes.append((bridge.PASS_KIND_GLOSSY_COLOR, 3, "Glossy Color", "RGB", "COLOR"))
    if getattr(view_layer, "use_pass_transmission_color", False):
        passes.append(
            (bridge.PASS_KIND_TRANSMISSION_COLOR, 3, "Transmission Color", "RGB", "COLOR")
        )
    if getattr(view_layer, "use_pass_diffuse_direct", False):
        passes.append((bridge.PASS_KIND_DIFFUSE_DIRECT, 3, "Diffuse Direct", "RGB", "COLOR"))
    if getattr(view_layer, "use_pass_diffuse_indirect", False):
        passes.append(
            (bridge.PASS_KIND_DIFFUSE_INDIRECT, 3, "Diffuse Indirect", "RGB", "COLOR")
        )
    if getattr(view_layer, "use_pass_glossy_direct", False):
        passes.append((bridge.PASS_KIND_GLOSSY_DIRECT, 3, "Glossy Direct", "RGB", "COLOR"))
    if getattr(view_layer, "use_pass_glossy_indirect", False):
        passes.append((bridge.PASS_KIND_GLOSSY_INDIRECT, 3, "Glossy Indirect", "RGB", "COLOR"))
    if getattr(view_layer, "use_pass_transmission_direct", False):
        passes.append(
            (
                bridge.PASS_KIND_TRANSMISSION_DIRECT,
                3,
                "Transmission Direct",
                "RGB",
                "COLOR",
            )
        )
    if getattr(view_layer, "use_pass_transmission_indirect", False):
        passes.append(
            (
                bridge.PASS_KIND_TRANSMISSION_INDIRECT,
                3,
                "Transmission Indirect",
                "RGB",
                "COLOR",
            )
        )
    if getattr(cview, "use_pass_volume_direct", False):
        passes.append((bridge.PASS_KIND_VOLUME_DIRECT, 3, "Volume Direct", "RGB", "COLOR"))
    if getattr(cview, "use_pass_volume_indirect", False):
        passes.append(
            (bridge.PASS_KIND_VOLUME_INDIRECT, 3, "Volume Indirect", "RGB", "COLOR")
        )
    if getattr(view_layer, "use_pass_emit", False):
        passes.append((bridge.PASS_KIND_EMISSION, 3, "Emission", "RGB", "COLOR"))
    if getattr(view_layer, "use_pass_environment", False):
        passes.append((bridge.PASS_KIND_ENVIRONMENT, 3, "Environment", "RGB", "COLOR"))
    if getattr(cview, "denoising_store_passes", False):
        passes.append(
            (bridge.PASS_KIND_DENOISING_NORMAL, 3, "Denoising Normal", "XYZ", "VECTOR")
        )
        passes.append(
            (bridge.PASS_KIND_DENOISING_ALBEDO, 3, "Denoising Albedo", "RGB", "COLOR")
        )
    if getattr(cview, "pass_debug_sample_count", False):
        passes.append(
            (bridge.PASS_KIND_SAMPLE_COUNT, 1, "Debug Sample Count", "X", "VALUE")
        )
    return passes


def _make_render_params(
    *,
    full_width: int,
    full_height: int,
    window_x: int = 0,
    window_y: int = 0,
    window_width: int = 0,
    window_height: int = 0,
    scene: Any | None = None,
) -> bridge.PsyclesRenderParams:
    pixel_filter = bridge.PSYCLES_PIXEL_FILTER_FROM_SCENE
    filter_width = 0.0
    if scene is not None and hasattr(scene, "cycles"):
        pft = str(scene.cycles.pixel_filter_type)
        pixel_filter = bridge.PIXEL_FILTER_MAP.get(pft, bridge.PSYCLES_PIXEL_FILTER_FROM_SCENE)
        filter_width = 1.0 if pft == "BOX" else float(scene.cycles.filter_width)
    return bridge.PsyclesRenderParams(
        full_width=max(int(full_width), 1),
        full_height=max(int(full_height), 1),
        window_x=max(int(window_x), 0),
        window_y=max(int(window_y), 0),
        window_width=max(int(window_width), 0),
        window_height=max(int(window_height), 0),
        seed=0,
        override_seed=0,
        transparent_background=0,
        override_transparent_background=0,
        pixel_filter=pixel_filter,
        filter_width=filter_width,
        pass_alpha_threshold=0.5,
        override_pass_alpha_threshold=0,
        reserved=(ctypes.c_uint8 * 3)(0, 0, 0),
    )


def _engine_settings_from_scene(scene: Any) -> dict[str, Any]:
    ps = getattr(scene, "psycles", None)
    prefs = get_preferences()
    return {
        "prefs_dir": prefs.runtime_library_dir if prefs is not None else "",
        "prefs_cache_dir": prefs.cache_dir if prefs is not None else "",
        "backend": str(ps.backend) if ps is not None else "AUTO",
        "scheduler": str(ps.scheduler) if ps is not None else "AUTO",
        "enable_fast_math": bool(ps.enable_fast_math) if ps is not None else True,
        "staged_surface_sorting": (
            bool(ps.staged_surface_sorting) if ps is not None else True
        ),
        "staged_direct_light_queue": (
            bool(ps.staged_direct_light_queue) if ps is not None else True
        ),
        "hip_visible_device": int(ps.hip_visible_device) if ps is not None else -1,
    }


class PsyclesRenderEngine(bpy.types.RenderEngine):
    bl_idname = "PSYCLES"
    bl_label = "Psycles"
    bl_use_preview = True
    bl_use_eevee_viewport = True
    bl_use_shading_nodes_custom = False
    bl_use_postprocess = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vp_lock = threading.Lock()
        self._vp_thread: threading.Thread | None = None
        self._vp_stop = threading.Event()
        self._vp_generation = 0
        self._vp_engine: bridge.PsyclesEngine | None = None
        self._vp_scene: bridge.PsyclesScene | None = None
        self._vp_session: bridge.PsyclesSession | None = None
        self._vp_camera_sig: tuple[Any, ...] | None = None
        self._vp_engine_key: tuple[Any, ...] | None = None
        self._vp_rgba: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._vp_rgba_size: tuple[int, int] = (0, 0)
        self._vp_gpu_tex: Any | None = None
        self._vp_gpu_tex_size: tuple[int, int] = (0, 0)
        self._vp_rendered_samples = 0
        self._vp_target_samples = 64

    def __del__(self) -> None:
        try:
            self._stop_viewport_worker()
            with self._vp_lock:
                if self._vp_session is not None:
                    self._vp_session.destroy()
                    self._vp_session = None
                if self._vp_scene is not None:
                    self._vp_scene.destroy()
                    self._vp_scene = None
                self._vp_gpu_tex = None
        except Exception:
            pass

    def update_render_passes(self, scene: Any = None, renderlayer: Any = None) -> None:
        if scene is None or renderlayer is None:
            return
        for _kind, channels, name, chan_ids, chan_type in _collect_passes(
            renderlayer, is_preview=bool(self.is_preview)
        ):
            self.register_pass(
                scene, renderlayer, name, channels, chan_ids, chan_type
            )

    def render(self, depsgraph: Any) -> None:
        scene = depsgraph.scene
        view_layer = depsgraph.view_layer
        ps = getattr(scene, "psycles", None)

        full_w = max(int(self.resolution_x), 1)
        full_h = max(int(self.resolution_y), 1)
        use_border = bool(scene.render.use_border) and not bool(self.is_preview)
        if use_border:
            bx = max(0, min(full_w - 1, int(round(scene.render.border_min_x * full_w))))
            by = max(0, min(full_h - 1, int(round(scene.render.border_min_y * full_h))))
            bx2 = max(bx + 1, min(full_w, int(round(scene.render.border_max_x * full_w))))
            by2 = max(by + 1, min(full_h, int(round(scene.render.border_max_y * full_h))))
            bw = bx2 - bx
            bh = by2 - by
        else:
            bx, by, bw, bh = 0, 0, full_w, full_h

        total_samples = max(int(getattr(scene.cycles, "samples", 64)), 1)
        if self.is_preview:
            preview_spp = int(getattr(scene.cycles, "preview_samples", 16))
            total_samples = max(1, min(preview_spp if preview_spp > 0 else 16, 16))

        chunk_cfg = int(getattr(ps, "progressive_chunk_samples", 16)) if ps is not None else 16
        if bpy.app.background or self.is_preview:
            chunk_size = total_samples
        else:
            chunk_size = max(1, chunk_cfg)

        passes_meta = _collect_passes(view_layer, is_preview=bool(self.is_preview))
        pass_descs = [(kind, ch, name, "") for kind, ch, name, _cid, _ct in passes_meta]

        self.update_stats("", "Psycles: Syncing scene")
        engine_cfg = _engine_settings_from_scene(scene)
        compiled_scene: bridge.PsyclesScene | None = None
        session: bridge.PsyclesSession | None = None
        result = None

        try:
            engine = bridge.get_or_create_engine(**engine_cfg)
            json_bytes, geom_bytes, base_dir = scene_sync.export_depsgraph_memory(
                depsgraph,
                scene=scene,
                view_layer=view_layer,
                render_width=full_w,
                render_height=full_h,
            )
            self.update_stats("", "Psycles: Compiling scene shaders & BVH")
            compiled_scene, _info = engine.compile_scene_memory(
                json_bytes, geom_bytes, base_dir
            )

            result = self.begin_result(0, 0, bw, bh, layer=view_layer.name)
            actual_bw = max(int(result.resolution_x), 1)
            actual_bh = max(int(result.resolution_y), 1)
            if actual_bw != bw or actual_bh != bh:
                bw, bh = actual_bw, actual_bh
            win_y = max(0, full_h - (by + bh))

            params = _make_render_params(
                full_width=full_w,
                full_height=full_h,
                window_x=bx if use_border else 0,
                window_y=win_y if use_border else 0,
                window_width=bw if use_border else full_w,
                window_height=bh if use_border else full_h,
                scene=scene,
            )
            session = compiled_scene.create_session(params, pass_descs)

            render_layer = result.layers[0]
            rendered = 0
            cancelled = False
            while rendered < total_samples:
                if self.test_break():
                    session.cancel()
                    cancelled = True
                    break
                batch = min(chunk_size, total_samples - rendered)
                self.update_stats(
                    "",
                    f"Psycles ({engine.backend_name}): Sample {rendered}/{total_samples}",
                )
                ok = session.render_samples(rendered, batch, total_samples, flip_y=True)
                if not ok:
                    cancelled = True
                    break
                rendered += batch

                for pass_idx, (_kind, ch, name, _cid, _ct) in enumerate(passes_meta):
                    rpass = render_layer.passes.get(name)
                    if rpass is None:
                        continue
                    arr = session.get_pass_array(pass_idx)
                    flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
                    expected = bw * bh * ch
                    if flat.size == expected:
                        rpass.rect.foreach_set(flat)

                self.update_result(result)
                self.update_progress(float(rendered) / float(total_samples))
                if self.test_break():
                    session.cancel()
                    cancelled = True
                    break

            if not cancelled:
                timings = session.last_timings()
                if timings is not None:
                    self.update_stats(
                        "",
                        f"Psycles ({engine.backend_name}): Done {total_samples} spp "
                        f"(integ {timings.integration_s:.2f}s)",
                    )
        except Exception as err:
            self.report({"ERROR"}, f"Psycles render failed: {err}")
            raise
        finally:
            if result is not None:
                self.end_result(result)
            if session is not None:
                session.destroy()
            if compiled_scene is not None:
                compiled_scene.destroy()

    def _stop_viewport_worker(self) -> None:
        self._vp_generation += 1
        self._vp_stop.set()
        session = self._vp_session
        if session is not None:
            session.cancel()
        thread = self._vp_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._vp_thread = None

    def _start_viewport_worker(
        self,
        session: bridge.PsyclesSession,
        target_samples: int,
        start_samples: int,
        width: int,
        height: int,
        backend_name: str,
    ) -> None:
        self._vp_stop.clear()
        generation = self._vp_generation

        def _worker() -> None:
            rendered = 0
            step = max(1, min(start_samples, target_samples))
            while rendered < target_samples and not self._vp_stop.is_set():
                count = min(step, target_samples - rendered)
                try:
                    ok = session.render_samples(rendered, count, target_samples, flip_y=True)
                except Exception as err:
                    self.report({"WARNING"}, f"Psycles viewport error: {err}")
                    break
                if not ok or self._vp_stop.is_set():
                    break
                with self._vp_lock:
                    if generation != self._vp_generation or self._vp_session is not session:
                        break
                    arr = session.get_pass_array(0)
                    self._vp_rgba = np.ascontiguousarray(arr, dtype=np.float32).copy()
                    self._vp_rgba_size = (width, height)
                    rendered += count
                    self._vp_rendered_samples = rendered
                try:
                    self.tag_redraw()
                except Exception:
                    break
                step = min(max(step * 2, 1), 16)

        self._vp_thread = threading.Thread(
            target=_worker, name="PsyclesViewportWorker", daemon=True
        )
        self._vp_thread.start()

    def _launch_viewport_camera_reset(
        self,
        capi_cam: bridge.PsyclesCameraDesc,
        params: bridge.PsyclesRenderParams,
        width: int,
        height: int,
        target_samples: int,
        start_samples: int,
    ) -> None:
        self._stop_viewport_worker()
        with self._vp_lock:
            session = self._vp_session
            engine = self._vp_engine
        if session is None or engine is None:
            return
        session.update_camera(capi_cam, params)
        with self._vp_lock:
            self._vp_rendered_samples = 0
            self._vp_target_samples = target_samples
        self._start_viewport_worker(
            session, target_samples, start_samples, width, height, engine.backend_name
        )

    def view_update(self, context: Any, depsgraph: Any) -> None:
        region = context.region
        v3d = context.space_data
        rv3d = context.region_data
        if region is None or v3d is None or rv3d is None:
            return
        scene = depsgraph.scene
        view_layer = depsgraph.view_layer
        ps = getattr(scene, "psycles", None)
        width = max(int(region.width), 1)
        height = max(int(region.height), 1)
        target_samples = max(int(getattr(scene.cycles, "preview_samples", 64)) or 64, 1)
        start_samples = (
            max(1, int(ps.viewport_start_samples)) if ps is not None else 1
        )

        json_cam, capi_cam, cam_sig = scene_sync.extract_viewport_camera(
            region, v3d, rv3d, scene, depsgraph
        )
        engine_cfg = _engine_settings_from_scene(scene)
        engine_key = tuple(sorted(engine_cfg.items()))
        params = _make_render_params(full_width=width, full_height=height, scene=scene)

        has_updates = bool(depsgraph.updates)
        can_reuse_scene = (
            not has_updates
            and self._vp_session is not None
            and self._vp_scene is not None
            and self._vp_engine_key == engine_key
        )
        if can_reuse_scene:
            if cam_sig != self._vp_camera_sig or self._vp_target_samples != target_samples:
                self._vp_camera_sig = cam_sig
                self._launch_viewport_camera_reset(
                    capi_cam, params, width, height, target_samples, start_samples
                )
            return

        self._stop_viewport_worker()
        with self._vp_lock:
            if self._vp_session is not None:
                self._vp_session.destroy()
                self._vp_session = None
            if self._vp_scene is not None:
                self._vp_scene.destroy()
                self._vp_scene = None

        engine = bridge.get_or_create_engine(**engine_cfg)
        json_bytes, geom_bytes, base_dir = scene_sync.export_depsgraph_memory(
            depsgraph,
            scene=scene,
            view_layer=view_layer,
            override_camera=json_cam,
            render_width=width,
            render_height=height,
        )
        compiled_scene, _info = engine.compile_scene_memory(
            json_bytes, geom_bytes, base_dir
        )
        session = compiled_scene.create_session(
            params, [(bridge.PASS_KIND_COMBINED, 4, "Combined", "")]
        )
        with self._vp_lock:
            self._vp_engine = engine
            self._vp_scene = compiled_scene
            self._vp_session = session
            self._vp_camera_sig = cam_sig
            self._vp_engine_key = engine_key
            self._vp_rendered_samples = 0
            self._vp_target_samples = target_samples
        self._start_viewport_worker(
            session, target_samples, start_samples, width, height, engine.backend_name
        )

    def view_draw(self, context: Any, depsgraph: Any) -> None:
        region = context.region
        v3d = context.space_data
        rv3d = context.region_data
        if region is None or v3d is None or rv3d is None:
            return
        scene = depsgraph.scene
        ps = getattr(scene, "psycles", None)
        width = max(int(region.width), 1)
        height = max(int(region.height), 1)
        target_samples = max(int(getattr(scene.cycles, "preview_samples", 64)) or 64, 1)
        start_samples = (
            max(1, int(ps.viewport_start_samples)) if ps is not None else 1
        )

        _json_cam, capi_cam, cam_sig = scene_sync.extract_viewport_camera(
            region, v3d, rv3d, scene, depsgraph
        )
        if self._vp_session is not None and cam_sig != self._vp_camera_sig:
            self._vp_camera_sig = cam_sig
            params = _make_render_params(full_width=width, full_height=height, scene=scene)
            self._launch_viewport_camera_reset(
                capi_cam, params, width, height, target_samples, start_samples
            )

        with self._vp_lock:
            rgba = self._vp_rgba
            tex_w, tex_h = self._vp_rgba_size
            rendered = self._vp_rendered_samples
            total = self._vp_target_samples
            backend_name = self._vp_engine.backend_name if self._vp_engine else "none"

        if total > 0:
            self.update_stats(
                "", f"Psycles ({backend_name}): Viewport {rendered}/{total} spp"
            )

        if rgba is None or tex_w <= 0 or tex_h <= 0:
            return

        if (
            self._vp_gpu_tex is None
            or self._vp_gpu_tex_size != (tex_w, tex_h)
        ):
            buf = gpu.types.Buffer("FLOAT", tex_w * tex_h * 4, rgba.ravel())
            self._vp_gpu_tex = gpu.types.GPUTexture(
                (tex_w, tex_h), format="RGBA32F", data=buf
            )
            self._vp_gpu_tex_size = (tex_w, tex_h)
        else:
            # Recreate buffer/texture cleanly across Blender 4.2-5.x GPU backends.
            buf = gpu.types.Buffer("FLOAT", tex_w * tex_h * 4, rgba.ravel())
            self._vp_gpu_tex = gpu.types.GPUTexture(
                (tex_w, tex_h), format="RGBA32F", data=buf
            )

        self.bind_display_space_shader(scene)
        try:
            shader = gpu.shader.from_builtin("IMAGE")
            batch = batch_for_shader(
                shader,
                "TRI_FAN",
                {
                    "pos": ((0, 0), (width, 0), (width, height), (0, height)),
                    "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
                },
            )
            shader.bind()
            shader.uniform_sampler("image", self._vp_gpu_tex)
            batch.draw(shader)
        finally:
            self.unbind_display_space_shader()


def register() -> None:
    bpy.utils.register_class(PsyclesRenderEngine)


def unregister() -> None:
    bpy.utils.unregister_class(PsyclesRenderEngine)

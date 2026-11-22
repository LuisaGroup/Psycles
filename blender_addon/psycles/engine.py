"""Psycles RenderEngine implementation for final renders, previews, and 3D viewport."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import threading
from typing import Any
import weakref

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np

from . import bridge, scene_cache, scene_sync
from ._export import cycles_hash
from .properties import get_preferences

_VP_COORD_LOCK = threading.Lock()
_ACTIVE_VIEWPORT_STATES: set["_ViewportState"] = set()
_RETIRING_TASKS: list["_RetiringTask"] = []
_VP_TIMER_REGISTERED = False


@dataclass
class _RetiringTask:
    task: bridge.PsyclesViewportTask
    mode: str
    engine: bridge.PsyclesEngine | None
    engine_key: tuple[Any, ...] | None
    content_hash: bytes | None
    clear_gen: int
    cache_entry: scene_cache.CacheEntry | None
    leased_session: bool
    owned_temp_session: bridge.PsyclesSession | None
    superseded: bool


def _finalize_finished_retiring(item: _RetiringTask) -> None:
    cache = scene_cache.get_scene_cache()
    try:
        if (
            item.mode == "compile"
            and not item.superseded
            and item.engine is not None
            and item.engine_key is not None
            and item.content_hash is not None
        ):
            st = item.task.poll()
            if st.state != bridge.PSYCLES_VIEWPORT_STATE_FAILED:
                scn = item.task.acquire_scene()
                sess = item.task.acquire_session(scn) if scn is not None else None
                if scn is not None and sess is not None:
                    stored = cache.store_idle_if_fresh(
                        "main",
                        item.clear_gen,
                        item.engine_key,
                        item.content_hash,
                        item.engine,
                        scn,
                        sess,
                    )
                    if not stored:
                        sess.destroy()
                        scn.destroy()
                elif scn is not None:
                    scn.destroy()
    finally:
        item.task.destroy()
        if item.owned_temp_session is not None:
            try:
                item.owned_temp_session.destroy()
            except Exception:
                pass
        if item.cache_entry is not None:
            cache.release(item.cache_entry, item.leased_session)


def _reap_retiring_tasks(*, wait: bool = False) -> bool:
    """Reap finished retiring viewport tasks. Returns True if any remain in flight."""
    with _VP_COORD_LOCK:
        items = list(_RETIRING_TASKS)
        _RETIRING_TASKS.clear()
    still_running: list[_RetiringTask] = []
    for item in items:
        if wait:
            item.task.wait()
            _finalize_finished_retiring(item)
        else:
            st = item.task.poll()
            if st.is_finished:
                _finalize_finished_retiring(item)
            else:
                still_running.append(item)
    if still_running:
        with _VP_COORD_LOCK:
            _RETIRING_TASKS.extend(still_running)
        return True
    return False


def _viewport_timer_tick() -> float | None:
    """Main-thread Blender timer polling native viewport tasks and reaping retiring ones."""
    global _VP_TIMER_REGISTERED
    any_busy = _reap_retiring_tasks(wait=False)
    with _VP_COORD_LOCK:
        states = list(_ACTIVE_VIEWPORT_STATES)
    for vp_state in states:
        try:
            if vp_state.poll_on_main_thread(allow_tag_redraw=True):
                any_busy = True
        except Exception:
            pass

    if any_busy or _reap_retiring_tasks(wait=False):
        return 0.016
    with _VP_COORD_LOCK:
        _VP_TIMER_REGISTERED = False
    return None


def _ensure_viewport_timer() -> None:
    """Schedule the main-thread viewport poller if called on Blender's main thread."""
    global _VP_TIMER_REGISTERED
    if threading.current_thread() is not threading.main_thread():
        return
    with _VP_COORD_LOCK:
        if _VP_TIMER_REGISTERED:
            try:
                if bpy.app.timers.is_registered(_viewport_timer_tick):
                    return
            except Exception:
                pass
        try:
            bpy.app.timers.register(
                _viewport_timer_tick, first_interval=0.016, persistent=True
            )
            _VP_TIMER_REGISTERED = True
        except Exception:
            _VP_TIMER_REGISTERED = False


def cancel_all_viewports(*, wait: bool = False) -> None:
    """Cancel active & retiring viewport workers (blocking only when wait=True)."""
    with _VP_COORD_LOCK:
        states = list(_ACTIVE_VIEWPORT_STATES)
        _ACTIVE_VIEWPORT_STATES.clear()
        retiring = list(_RETIRING_TASKS)
    for item in retiring:
        try:
            item.superseded = True
            item.task.cancel()
        except Exception:
            pass
    for vp_state in states:
        try:
            vp_state.retire(superseded=True)
            vp_state.gpu_tex = None
        except Exception:
            pass
    if wait:
        _reap_retiring_tasks(wait=True)
        try:
            if bpy.app.timers.is_registered(_viewport_timer_tick):
                bpy.app.timers.unregister(_viewport_timer_tick)
        except Exception:
            pass
        with _VP_COORD_LOCK:
            global _VP_TIMER_REGISTERED
            _VP_TIMER_REGISTERED = False
    else:
        _ensure_viewport_timer()


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
    view_layer: Any | None = None,
) -> bridge.PsyclesRenderParams:
    pixel_filter = bridge.PSYCLES_PIXEL_FILTER_FROM_SCENE
    filter_width = 0.0
    seed = 0
    override_seed = 0
    transparent_bg = 0
    override_transparent_bg = 0
    alpha_thresh = 0.5
    override_alpha_thresh = 0

    if scene is not None:
        transparent_bg = 1 if bool(getattr(scene.render, "film_transparent", False)) else 0
        override_transparent_bg = 1
        cycles = getattr(scene, "cycles", None)
        if cycles is not None:
            pft = str(cycles.pixel_filter_type)
            pixel_filter = bridge.PIXEL_FILTER_MAP.get(
                pft, bridge.PSYCLES_PIXEL_FILTER_FROM_SCENE
            )
            filter_width = 1.0 if pft == "BOX" else float(cycles.filter_width)
            seed = cycles_hash.effective_scene_seed(
                int(cycles.seed),
                int(scene.frame_current),
                float(getattr(scene, "frame_subframe", 0.0)),
                bool(getattr(cycles, "use_animated_seed", False)),
            )
            override_seed = 1
    if view_layer is not None:
        alpha_thresh = max(
            0.0, min(1.0, float(getattr(view_layer, "pass_alpha_threshold", 0.5)))
        )
        override_alpha_thresh = 1

    return bridge.PsyclesRenderParams(
        full_width=max(int(full_width), 1),
        full_height=max(int(full_height), 1),
        window_x=max(int(window_x), 0),
        window_y=max(int(window_y), 0),
        window_width=max(int(window_width), 0),
        window_height=max(int(window_height), 0),
        seed=seed,
        override_seed=override_seed,
        transparent_background=transparent_bg,
        override_transparent_background=override_transparent_bg,
        pixel_filter=pixel_filter,
        filter_width=filter_width,
        pass_alpha_threshold=alpha_thresh,
        override_pass_alpha_threshold=override_alpha_thresh,
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


def _updates_are_camera_only(depsgraph: Any) -> bool:
    updates = getattr(depsgraph, "updates", None)
    if not updates:
        return True
    for u in updates:
        uid = getattr(u, "id", None)
        if isinstance(uid, bpy.types.Camera):
            continue
        if isinstance(uid, bpy.types.Object) and getattr(uid, "type", "") == "CAMERA":
            continue
        return False
    return True


class _ViewportState:
    """Plain Python state for one viewport engine (never holds a strong RNA ref)."""

    def __init__(self, engine_ref: weakref.ReferenceType[Any] | None) -> None:
        self.engine_ref = engine_ref
        self.task: bridge.PsyclesViewportTask | None = None
        self.task_mode: str = ""
        self.task_engine: bridge.PsyclesEngine | None = None
        self.task_engine_key: tuple[Any, ...] | None = None
        self.task_content_hash: bytes | None = None
        self.task_clear_gen: int = 0
        self.cache_entry: scene_cache.CacheEntry | None = None
        self.leased_session: bool = False
        self.owned_temp_session: bridge.PsyclesSession | None = None
        self.engine: bridge.PsyclesEngine | None = None
        self.engine_key: tuple[Any, ...] | None = None
        self.camera_sig: tuple[Any, ...] | None = None
        self.rgba: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self.rgba_size: tuple[int, int] = (0, 0)
        self.frame_gen: int = 0
        self.gpu_tex: Any | None = None
        self.gpu_tex_size: tuple[int, int] = (0, 0)
        self.gpu_tex_gen: int = 0
        self.rendered_samples = 0
        self.target_samples = 64
        self.start_samples = 1
        self.last_size: tuple[int, int] = (0, 0)
        self.last_state = -1
        self.status = ""

    def live_engine(self) -> Any | None:
        eng = self.engine_ref() if self.engine_ref is not None else None
        if eng is None:
            return None
        try:
            getattr(eng, "is_preview", False)
        except ReferenceError:
            return None
        return eng

    def _drop_dead_engine(self) -> None:
        with _VP_COORD_LOCK:
            _ACTIVE_VIEWPORT_STATES.discard(self)
        self.retire(superseded=False)
        self.gpu_tex = None

    def release_cache_leases(self) -> None:
        if self.owned_temp_session is not None:
            try:
                self.owned_temp_session.destroy()
            except Exception:
                pass
            self.owned_temp_session = None
        if self.cache_entry is not None:
            scene_cache.get_scene_cache().release(
                self.cache_entry, self.leased_session
            )
            self.cache_entry = None
            self.leased_session = False

    def harvest_finished(self, *, superseded: bool) -> None:
        task = self.task
        if task is None:
            return
        st = task.poll()
        if not st.is_finished:
            return
        cache = scene_cache.get_scene_cache()
        try:
            if (
                self.task_mode == "compile"
                and not superseded
                and self.cache_entry is None
                and st.state != bridge.PSYCLES_VIEWPORT_STATE_FAILED
                and self.task_engine is not None
                and self.task_engine_key is not None
                and self.task_content_hash is not None
            ):
                scn = task.acquire_scene()
                sess = task.acquire_session(scn) if scn is not None else None
                if scn is not None and sess is not None:
                    self.cache_entry = cache.store_and_lease(
                        "main",
                        self.task_engine_key,
                        self.task_content_hash,
                        self.task_engine,
                        scn,
                        sess,
                        lease_session=True,
                        expected_clear_gen=self.task_clear_gen,
                    )
                    self.leased_session = True
                elif scn is not None:
                    scn.destroy()
            elif (
                self.task_mode == "scene"
                and not superseded
                and self.owned_temp_session is None
                and st.state != bridge.PSYCLES_VIEWPORT_STATE_FAILED
                and self.cache_entry is not None
                and self.cache_entry.scene is not None
            ):
                sess = task.acquire_session(self.cache_entry.scene)
                if sess is not None:
                    self.owned_temp_session = sess
        finally:
            task.destroy()
            self.task = None

    def retire(self, *, superseded: bool) -> None:
        task = self.task
        if task is None:
            self.release_cache_leases()
            return
        task.cancel()
        st = task.poll()
        if st.is_finished:
            self.harvest_finished(superseded=superseded)
            self.release_cache_leases()
            return

        item = _RetiringTask(
            task=task,
            mode=self.task_mode,
            engine=self.task_engine,
            engine_key=self.task_engine_key,
            content_hash=self.task_content_hash,
            clear_gen=self.task_clear_gen,
            cache_entry=self.cache_entry,
            leased_session=self.leased_session,
            owned_temp_session=self.owned_temp_session,
            superseded=superseded,
        )
        self.task = None
        self.cache_entry = None
        self.leased_session = False
        self.owned_temp_session = None
        with _VP_COORD_LOCK:
            _RETIRING_TASKS.append(item)
        _ensure_viewport_timer()

    def poll_on_main_thread(self, *, allow_tag_redraw: bool) -> bool:
        """Poll active native viewport task on Blender's main thread."""
        eng = self.live_engine()
        if eng is None:
            self._drop_dead_engine()
            return False
        task = self.task
        if task is None:
            return False

        st = task.poll()
        needs_redraw = False
        backend = self.engine.backend_name if self.engine else "none"

        if st.frame_generation != self.frame_gen and st.frame_generation > 0:
            copied = task.copy_frame(self.rgba)
            if copied is not None:
                buf, w, h, gen = copied
                self.rgba = buf
                self.rgba_size = (w, h)
                self.frame_gen = gen
                self.rendered_samples = st.rendered_samples
                self.target_samples = max(st.target_samples, 1)
                self.status = (
                    f"Psycles ({backend}): Viewport "
                    f"{self.rendered_samples}/{self.target_samples} spp"
                )
                needs_redraw = True

        if st.state != self.last_state:
            self.last_state = st.state
            if st.state == bridge.PSYCLES_VIEWPORT_STATE_COMPILING:
                self.status = f"Psycles ({backend}): Compiling scene & shaders..."
                needs_redraw = True
            elif st.state == bridge.PSYCLES_VIEWPORT_STATE_FAILED:
                msg = st.error_message or "viewport render failed"
                self.status = f"Psycles ({backend}): Error"
                try:
                    eng.report({"WARNING"}, f"Psycles viewport error: {msg}")
                except ReferenceError:
                    self._drop_dead_engine()
                    return False
                needs_redraw = True
            elif st.state == bridge.PSYCLES_VIEWPORT_STATE_DONE:
                self.rendered_samples = st.rendered_samples
                self.status = (
                    f"Psycles ({backend}): Viewport "
                    f"{st.rendered_samples}/{st.target_samples} spp"
                )
                needs_redraw = True

        if st.is_finished:
            self.harvest_finished(superseded=False)

        if needs_redraw and allow_tag_redraw:
            try:
                eng.tag_redraw()
            except ReferenceError:
                self._drop_dead_engine()
                return False
        return not st.is_finished

    def apply_camera_change(
        self,
        capi_cam: bridge.PsyclesCameraDesc,
        params: bridge.PsyclesRenderParams,
        target_samples: int,
        start_samples: int,
    ) -> bool:
        self.rendered_samples = 0
        self.target_samples = target_samples
        if self.task is not None:
            if self.task.request_camera(
                capi_cam, params, target_samples, start_samples
            ):
                _ensure_viewport_timer()
                return True
            st = self.task.poll()
            if st.is_finished:
                self.harvest_finished(superseded=False)
            else:
                self.retire(superseded=False)

        idle_sess: bridge.PsyclesSession | None = None
        if (
            self.cache_entry is not None
            and self.leased_session
            and self.cache_entry.session is not None
        ):
            idle_sess = self.cache_entry.session
        elif self.owned_temp_session is not None:
            idle_sess = self.owned_temp_session

        if idle_sess is not None and idle_sess._handle:
            self.task = bridge.PsyclesViewportTask.start_session(
                idle_sess,
                capi_cam,
                params,
                target_samples,
                start_samples,
                flip_y=True,
            )
            self.task_mode = "session"
            self.last_state = bridge.PSYCLES_VIEWPORT_STATE_RENDERING
            _ensure_viewport_timer()
            return True
        return False


class PsyclesRenderEngine(bpy.types.RenderEngine):
    bl_idname = "PSYCLES"
    bl_label = "Psycles"
    bl_use_preview = True
    bl_use_eevee_viewport = True
    bl_use_shading_nodes_custom = False
    bl_use_postprocess = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vp_state = _ViewportState(weakref.ref(self))

    def __del__(self) -> None:
        try:
            state = object.__getattribute__(self, "__dict__").get("_vp_state")
            if state is not None:
                with _VP_COORD_LOCK:
                    _ACTIVE_VIEWPORT_STATES.discard(state)
                state.retire(superseded=False)
                state.gpu_tex = None
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
        engine_key = tuple(sorted(engine_cfg.items()))
        slot_name = "preview" if self.is_preview else "main"
        cache = scene_cache.get_scene_cache()
        mesh_cache = scene_cache.get_mesh_cache()

        entry: scene_cache.CacheEntry | None = None
        leased_session = False
        temp_session: bridge.PsyclesSession | None = None
        result = None

        try:
            engine = bridge.get_or_create_engine(**engine_cfg)
            capi_cam = scene_sync.extract_scene_camera_capi(scene, depsgraph)
            json_bytes, geom_bytes, base_dir, content_hash = (
                scene_sync.export_depsgraph_with_hash(
                    depsgraph,
                    scene=scene,
                    view_layer=view_layer,
                    render_width=full_w,
                    render_height=full_h,
                    mesh_cache=mesh_cache,
                )
            )

            entry, leased_session = cache.acquire(slot_name, engine_key, content_hash)
            compiled_scene: bridge.PsyclesScene | None = None
            if entry is None:
                cache.evict_idle(slot_name)
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
                view_layer=view_layer,
            )

            if entry is not None and leased_session and entry.session is not None:
                self.update_stats(
                    "", f"Psycles ({engine.backend_name}): Reusing cached scene"
                )
                session = entry.session
                session.reconfigure(capi_cam, params, pass_descs)
            elif entry is not None and entry.scene is not None:
                self.update_stats(
                    "",
                    f"Psycles ({engine.backend_name}): Creating session (cached scene)",
                )
                temp_session = entry.scene.create_session(params, pass_descs)
                session = temp_session
            else:
                assert compiled_scene is not None
                session = compiled_scene.create_session(params, pass_descs)
                entry = cache.store_and_lease(
                    slot_name,
                    engine_key,
                    content_hash,
                    engine,
                    compiled_scene,
                    session,
                    lease_session=True,
                )
                leased_session = True

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
            if temp_session is not None:
                temp_session.destroy()
            if entry is not None:
                cache.release(entry, leased_session)

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
        params = _make_render_params(
            full_width=width, full_height=height, scene=scene, view_layer=view_layer
        )

        st = self._vp_state
        with _VP_COORD_LOCK:
            _ACTIVE_VIEWPORT_STATES.add(st)

        has_live_viewport = (
            st.engine_key == engine_key
            and (st.cache_entry is None or not st.cache_entry.evicted)
            and (
                st.task is not None
                or (st.cache_entry is not None and st.cache_entry.scene is not None)
            )
        )
        if has_live_viewport and _updates_are_camera_only(depsgraph):
            if (
                cam_sig != st.camera_sig
                or st.target_samples != target_samples
                or st.start_samples != start_samples
                or st.last_size != (width, height)
            ):
                st.camera_sig = cam_sig
                st.last_size = (width, height)
                st.start_samples = start_samples
                st.apply_camera_change(capi_cam, params, target_samples, start_samples)
            return

        cache = scene_cache.get_scene_cache()
        mesh_cache = scene_cache.get_mesh_cache()
        engine = bridge.get_or_create_engine(**engine_cfg)
        json_bytes, geom_bytes, base_dir, content_hash = (
            scene_sync.export_depsgraph_with_hash(
                depsgraph,
                scene=scene,
                view_layer=view_layer,
                override_camera=json_cam,
                render_width=width,
                render_height=height,
                mesh_cache=mesh_cache,
            )
        )

        # If content hash & engine key match what this viewport already displays,
        # avoid full scene recompile (and keep progressing if camera/extent also match).
        if has_live_viewport and st.task_content_hash == content_hash:
            if (
                cam_sig == st.camera_sig
                and st.target_samples == target_samples
                and st.start_samples == start_samples
                and st.last_size == (width, height)
                and st.last_state != bridge.PSYCLES_VIEWPORT_STATE_FAILED
            ):
                return
            st.camera_sig = cam_sig
            st.last_size = (width, height)
            st.start_samples = start_samples
            if st.apply_camera_change(capi_cam, params, target_samples, start_samples):
                return

        st.retire(superseded=True)

        entry, leased_session = cache.acquire("main", engine_key, content_hash)
        st.engine = engine
        st.engine_key = engine_key
        st.camera_sig = cam_sig
        st.last_size = (width, height)
        st.rendered_samples = 0
        st.target_samples = target_samples
        st.start_samples = start_samples
        st.task_engine = engine
        st.task_engine_key = engine_key
        st.task_content_hash = content_hash

        if entry is not None and leased_session and entry.session is not None:
            st.cache_entry = entry
            st.leased_session = True
            st.task_mode = "session"
            st.last_state = bridge.PSYCLES_VIEWPORT_STATE_RENDERING
            st.status = (
                f"Psycles ({engine.backend_name}): Viewport 0/{target_samples} spp"
            )
            st.task = bridge.PsyclesViewportTask.start_session(
                entry.session,
                capi_cam,
                params,
                target_samples,
                start_samples,
                flip_y=True,
            )
        elif entry is not None and entry.scene is not None:
            st.cache_entry = entry
            st.leased_session = False
            st.task_mode = "scene"
            st.last_state = bridge.PSYCLES_VIEWPORT_STATE_COMPILING
            st.status = (
                f"Psycles ({engine.backend_name}): Initializing viewport..."
            )
            st.task = bridge.PsyclesViewportTask.start_scene(
                entry.scene,
                capi_cam,
                params,
                target_samples,
                start_samples,
                flip_y=True,
            )
        else:
            clear_gen = cache.current_clear_gen()
            cache.evict_idle("main")
            st.task_clear_gen = clear_gen
            st.task_mode = "compile"
            st.last_state = bridge.PSYCLES_VIEWPORT_STATE_COMPILING
            st.status = (
                f"Psycles ({engine.backend_name}): Compiling scene & shaders..."
            )
            st.task = bridge.PsyclesViewportTask.start_compile(
                engine,
                json_bytes,
                geom_bytes,
                base_dir,
                capi_cam,
                params,
                target_samples,
                start_samples,
                flip_y=True,
            )

        _ensure_viewport_timer()

    def view_draw(self, context: Any, depsgraph: Any) -> None:
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

        st = self._vp_state
        st.poll_on_main_thread(allow_tag_redraw=False)

        _json_cam, capi_cam, cam_sig = scene_sync.extract_viewport_camera(
            region, v3d, rv3d, scene, depsgraph
        )
        if (
            cam_sig != st.camera_sig or st.last_size != (width, height)
        ) and (st.task is not None or st.cache_entry is not None):
            st.camera_sig = cam_sig
            st.last_size = (width, height)
            params = _make_render_params(
                full_width=width, full_height=height, scene=scene, view_layer=view_layer
            )
            st.apply_camera_change(capi_cam, params, target_samples, start_samples)

        if st.status:
            self.update_stats("", st.status)

        rgba = st.rgba
        tex_w, tex_h = st.rgba_size
        if rgba is None or tex_w <= 0 or tex_h <= 0:
            return

        if (
            st.gpu_tex is None
            or st.gpu_tex_size != (tex_w, tex_h)
            or st.gpu_tex_gen != st.frame_gen
        ):
            buf = gpu.types.Buffer("FLOAT", tex_w * tex_h * 4, rgba.ravel())
            st.gpu_tex = gpu.types.GPUTexture(
                (tex_w, tex_h), format="RGBA32F", data=buf
            )
            st.gpu_tex_size = (tex_w, tex_h)
            st.gpu_tex_gen = st.frame_gen

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
            shader.uniform_sampler("image", st.gpu_tex)
            batch.draw(shader)
        finally:
            self.unbind_display_space_shader()


def register() -> None:
    bpy.utils.register_class(PsyclesRenderEngine)


def unregister() -> None:
    cancel_all_viewports(wait=True)
    bpy.utils.unregister_class(PsyclesRenderEngine)

"""Process-wide mesh extraction and compiled scene/session caches for Psycles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Any

import bpy

from . import bridge


class MeshExtractionCache:
    """Scan-resistant bounded LRU cache for Blender mesh geometry buffers."""

    def __init__(self, max_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        self._max_bytes = max(int(max_bytes), 1024 * 1024)
        self._lock = threading.Lock()
        self._entries: OrderedDict[
            tuple[Any, ...], tuple[dict[str, Any], bytes, int]
        ] = OrderedDict()
        self._total_bytes = 0
        self._pass_gen = 0
        self._last_frame: tuple[int, float] | None = None
        self._last_filepath: str = ""
        self._hits = 0
        self._misses = 0

    def prepare_for_scene(self, scene: Any) -> None:
        frame_sig = (
            int(getattr(scene, "frame_current", 0)),
            float(getattr(scene, "frame_subframe", 0.0)),
        )
        filepath = str(getattr(bpy.data, "filepath", ""))
        with self._lock:
            self._pass_gen += 1
            if (
                self._last_frame is not None and self._last_frame != frame_sig
            ) or (self._last_filepath != filepath):
                self._entries.clear()
                self._total_bytes = 0
            self._last_frame = frame_sig
            self._last_filepath = filepath

    def invalidate_from_depsgraph(self, depsgraph: Any) -> None:
        if depsgraph is None or str(getattr(depsgraph, "mode", "")) == "RENDER":
            return
        updates = getattr(depsgraph, "updates", None)
        if not updates:
            return
        dirty_mesh_ptrs: set[int] = set()
        clear_modified = False
        clear_all_meshes = False
        for u in updates:
            uid = getattr(u, "id", None)
            if getattr(u, "is_updated_geometry", False):
                clear_modified = True
                if isinstance(uid, bpy.types.Mesh):
                    orig = getattr(uid, "original", uid)
                    dirty_mesh_ptrs.add(int(orig.as_pointer()))
                elif isinstance(uid, bpy.types.Object) and isinstance(
                    getattr(uid, "data", None), bpy.types.Mesh
                ):
                    orig = getattr(uid.data, "original", uid.data)
                    dirty_mesh_ptrs.add(int(orig.as_pointer()))
                elif isinstance(uid, (bpy.types.Scene, bpy.types.Collection)):
                    clear_all_meshes = True
                    break
            elif getattr(u, "is_updated_transform", False):
                if isinstance(uid, bpy.types.Object) and getattr(
                    uid, "type", ""
                ) not in {"CAMERA", "LIGHT", "LIGHT_PROBE"}:
                    clear_modified = True

        with self._lock:
            if clear_all_meshes:
                self._entries.clear()
                self._total_bytes = 0
                return
            if not dirty_mesh_ptrs and not clear_modified:
                return
            keys_to_drop = [
                k
                for k in self._entries
                if (k[0] == "object" and (clear_modified or k[2] in dirty_mesh_ptrs))
                or (k[0] == "mesh" and k[1] in dirty_mesh_ptrs)
            ]
            for key in keys_to_drop:
                _tpl, raw, _pg = self._entries.pop(key)
                self._total_bytes -= len(raw)

    def get(self, key: tuple[Any, ...]) -> tuple[dict[str, Any], bytes] | None:
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._entries[key] = (item[0], item[1], self._pass_gen)
            self._hits += 1
            return item[0], item[1]

    def put(
        self,
        key: tuple[Any, ...],
        template_dict: dict[str, Any],
        raw_bytes: bytes,
    ) -> None:
        size = len(raw_bytes)
        if size > self._max_bytes:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._total_bytes -= len(old[1])
            if self._total_bytes + size > self._max_bytes:
                stale_keys = [
                    k
                    for k, (_tpl, _raw, pg) in self._entries.items()
                    if pg < self._pass_gen
                ]
                for stale_key in stale_keys:
                    if self._total_bytes + size <= self._max_bytes:
                        break
                    _tpl, evicted_bytes, _pg = self._entries.pop(stale_key)
                    self._total_bytes -= len(evicted_bytes)
                if self._total_bytes + size > self._max_bytes:
                    return
            self._entries[key] = (template_dict, raw_bytes, self._pass_gen)
            self._total_bytes += size

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    def stats(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                self._hits,
                self._misses,
                len(self._entries),
                self._total_bytes,
            )

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._misses = 0


@dataclass
class CacheEntry:
    slot_name: str
    engine_key: tuple[Any, ...]
    content_hash: bytes
    engine: bridge.PsyclesEngine
    scene: bridge.PsyclesScene | None
    session: bridge.PsyclesSession | None
    session_busy: bool = False
    active_refs: int = 0
    evicted: bool = False
    clear_gen: int = 0


def _destroy_pair(
    session: bridge.PsyclesSession | None,
    scene: bridge.PsyclesScene | None,
) -> None:
    if session is not None:
        try:
            session.destroy()
        except Exception:
            pass
    if scene is not None:
        try:
            scene.destroy()
        except Exception:
            pass


class SceneSessionCache:
    """Two-slot ('main' and 'preview') compiled scene + session cache with lease tracking."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._slots: dict[str, CacheEntry] = {}
        self._clear_gen = 0
        self._hits = 0
        self._scene_only_hits = 0
        self._misses = 0

    def current_clear_gen(self) -> int:
        with self._lock:
            return self._clear_gen

    def acquire(
        self,
        slot_name: str,
        engine_key: tuple[Any, ...],
        content_hash: bytes,
    ) -> tuple[CacheEntry | None, bool]:
        with self._lock:
            entry = self._slots.get(slot_name)
            if (
                entry is not None
                and not entry.evicted
                and entry.engine_key == engine_key
                and entry.content_hash == content_hash
                and entry.scene is not None
                and entry.scene._handle
            ):
                entry.active_refs += 1
                if (
                    entry.session is not None
                    and entry.session._handle
                    and not entry.session_busy
                ):
                    entry.session_busy = True
                    self._hits += 1
                    return entry, True
                self._scene_only_hits += 1
                return entry, False
            self._misses += 1
            return None, False

    def evict_idle(self, slot_name: str) -> None:
        to_destroy: list[tuple[bridge.PsyclesSession | None, bridge.PsyclesScene | None]] = []
        with self._lock:
            old = self._slots.get(slot_name)
            if old is not None and old.active_refs == 0:
                del self._slots[slot_name]
                old.evicted = True
                to_destroy.append((old.session, old.scene))
                old.session = None
                old.scene = None
        for sess, scn in to_destroy:
            _destroy_pair(sess, scn)

    def store_and_lease(
        self,
        slot_name: str,
        engine_key: tuple[Any, ...],
        content_hash: bytes,
        engine: bridge.PsyclesEngine,
        scene: bridge.PsyclesScene,
        session: bridge.PsyclesSession | None,
        *,
        lease_session: bool = True,
        expected_clear_gen: int | None = None,
    ) -> CacheEntry:
        to_destroy: list[tuple[bridge.PsyclesSession | None, bridge.PsyclesScene | None]] = []
        with self._lock:
            stale = (
                expected_clear_gen is not None
                and expected_clear_gen != self._clear_gen
            )
            entry = CacheEntry(
                slot_name=slot_name,
                engine_key=engine_key,
                content_hash=content_hash,
                engine=engine,
                scene=scene,
                session=session,
                session_busy=lease_session and session is not None,
                active_refs=1,
                evicted=stale,
                clear_gen=self._clear_gen,
            )
            if not stale:
                old = self._slots.get(slot_name)
                if old is not None:
                    old.evicted = True
                    if old.active_refs == 0:
                        to_destroy.append((old.session, old.scene))
                        old.session = None
                        old.scene = None
                self._slots[slot_name] = entry
        for sess, scn in to_destroy:
            _destroy_pair(sess, scn)
        return entry

    def store_idle_if_fresh(
        self,
        slot_name: str,
        expected_clear_gen: int,
        engine_key: tuple[Any, ...],
        content_hash: bytes,
        engine: bridge.PsyclesEngine,
        scene: bridge.PsyclesScene,
        session: bridge.PsyclesSession | None,
    ) -> bool:
        to_destroy: list[tuple[bridge.PsyclesSession | None, bridge.PsyclesScene | None]] = []
        stored = False
        with self._lock:
            existing = self._slots.get(slot_name)
            can_install = (
                expected_clear_gen == self._clear_gen
                and scene._handle is not None
                and (existing is None or existing.evicted)
            )
            if can_install:
                if existing is not None:
                    existing.evicted = True
                    if existing.active_refs == 0:
                        to_destroy.append((existing.session, existing.scene))
                        existing.session = None
                        existing.scene = None
                self._slots[slot_name] = CacheEntry(
                    slot_name=slot_name,
                    engine_key=engine_key,
                    content_hash=content_hash,
                    engine=engine,
                    scene=scene,
                    session=session,
                    session_busy=False,
                    active_refs=0,
                    evicted=False,
                    clear_gen=self._clear_gen,
                )
                stored = True
        for sess, scn in to_destroy:
            _destroy_pair(sess, scn)
        return stored

    def release(
        self,
        entry: CacheEntry | None,
        owned_session_lease: bool,
    ) -> None:
        if entry is None:
            return
        sess: bridge.PsyclesSession | None = None
        scn: bridge.PsyclesScene | None = None
        with self._lock:
            if owned_session_lease:
                entry.session_busy = False
            if entry.active_refs > 0:
                entry.active_refs -= 1
            if entry.evicted and entry.active_refs == 0:
                sess = entry.session
                scn = entry.scene
                entry.session = None
                entry.scene = None
        _destroy_pair(sess, scn)

    def clear(self) -> None:
        to_destroy: list[tuple[bridge.PsyclesSession | None, bridge.PsyclesScene | None]] = []
        with self._lock:
            self._clear_gen += 1
            old_entries = list(self._slots.values())
            self._slots.clear()
            for entry in old_entries:
                entry.evicted = True
                if entry.active_refs == 0:
                    to_destroy.append((entry.session, entry.scene))
                    entry.session = None
                    entry.scene = None
        for sess, scn in to_destroy:
            _destroy_pair(sess, scn)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "scene_only_hits": self._scene_only_hits,
                "misses": self._misses,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = 0
            self._scene_only_hits = 0
            self._misses = 0


_SCENE_CACHE = SceneSessionCache()
_MESH_CACHE = MeshExtractionCache()


def get_scene_cache() -> SceneSessionCache:
    return _SCENE_CACHE


def get_mesh_cache() -> MeshExtractionCache:
    return _MESH_CACHE


def stats() -> dict[str, int]:
    mh, mm, mcount, mbytes = _MESH_CACHE.stats()
    return {
        **_SCENE_CACHE.stats(),
        "mesh_hits": mh,
        "mesh_misses": mm,
        "mesh_entries": mcount,
        "mesh_bytes": mbytes,
    }


def reset_stats() -> None:
    _SCENE_CACHE.reset_stats()
    _MESH_CACHE.reset_stats()


def clear_all() -> None:
    _MESH_CACHE.clear()
    _SCENE_CACHE.clear()

"""Cycles 5.2 Blender object and particle-source projection."""

from __future__ import annotations

from typing import Any

import cycles_hash


def object_random_id(name: str) -> int:
    return cycles_hash.hash_uint2(cycles_hash.hash_string(name), 0)


def uses_light_linking(object_instance: Any) -> bool:
    """Conservatively detect source state requiring Cycles link-set fields."""

    objects = [object_instance.object]
    if object_instance.is_instance:
        parent = object_instance.parent
        if parent is not None and parent != object_instance.object:
            objects.append(parent)
    for obj in objects:
        linking = getattr(obj, "light_linking", None)
        if linking is None:
            continue
        if (
            getattr(linking, "receiver_collection", None) is not None
            or getattr(linking, "blocker_collection", None) is not None
        ):
            return True
    return False


def light_group(
    object_instance: Any,
    light_groups: dict[str, int],
) -> int:
    obj = object_instance.object
    name = str(getattr(obj, "lightgroup", ""))
    if not name and object_instance.is_instance:
        parent = object_instance.parent
        if parent is not None and parent != obj:
            name = str(getattr(parent, "lightgroup", ""))
    return light_groups.get(name, -1)


def shadow_catcher(object_instance: Any) -> bool:
    obj = object_instance.object
    result = bool(getattr(obj, "is_shadow_catcher", False))
    if object_instance.is_instance:
        parent = object_instance.parent
        if parent is not None and parent != obj:
            result = result or bool(
                getattr(parent, "is_shadow_catcher", False)
            )
    return result


def object_ao_distance(object_instance: Any) -> float:
    """Match BlenderSync's child-then-instancer AO-distance lookup."""

    obj = object_instance.object
    cycles = getattr(obj, "cycles", None)
    result = float(getattr(cycles, "ao_distance", 0.0))
    if result == 0.0 and object_instance.is_instance:
        parent = object_instance.parent
        if parent is not None and parent != obj:
            parent_cycles = getattr(parent, "cycles", None)
            result = float(
                getattr(parent_cycles, "ao_distance", 0.0)
            )
    return max(result, 0.0)


def particle_parent_index(object_instance: Any) -> int:
    """Return the legacy parent index, before Cycles device-table packing."""

    source = particle_source(object_instance, None, None)
    return 0 if source is None else int(source["source_index"])


def _shadow_terminator_value(
    obj: Any,
    name: str,
    fallback: float,
) -> float:
    if hasattr(obj, name):
        return float(getattr(obj, name))
    return float(getattr(getattr(obj, "cycles", None), name, fallback))


def _asset_name(obj: Any) -> str:
    """Copy BlenderSync's topmost object-parent Cryptomatte identity."""

    parent = obj
    while parent.parent is not None:
        parent = parent.parent
    return str(parent.name)


def _use_holdout(object_instance: Any, view_layer: Any) -> bool:
    """Copy BlenderSync's view-layer holdout from the instancer base."""

    obj = object_instance.object
    base = (
        object_instance.parent
        if object_instance.is_instance and object_instance.parent is not None
        else obj
    )
    # DependencyGraph.object_instances exposes evaluated object copies, while
    # BlenderSync queries BKE_view_layer_base_find() with the original object.
    # The evaluated copy has no Base and therefore reports false even when its
    # original belongs to a holdout LayerCollection.
    original = base.original if base.original is not None else base
    return bool(original.holdout_get(view_layer=view_layer))


def object_properties(object_instance: Any, view_layer: Any) -> dict[str, Any]:
    """Return raw Object fields copied by BlenderSync::sync_object."""

    obj = object_instance.object
    if object_instance.is_instance:
        orco = object_instance.orco
        dupli_generated = [
            0.5 * float(orco[axis]) - 0.5 for axis in range(3)
        ]
        uv = object_instance.uv
        dupli_uv = [float(uv[0]), float(uv[1])]
    else:
        dupli_generated = [0.0, 0.0, 0.0]
        dupli_uv = [0.0, 0.0]
    object_color = tuple(float(component) for component in obj.color)
    cycles = getattr(obj, "cycles", None)
    return {
        "asset_name": _asset_name(obj),
        "object_color": list(object_color[:3]),
        "object_alpha": object_color[3],
        "object_pass_id": int(obj.pass_index),
        "dupli_generated": dupli_generated,
        "dupli_uv": dupli_uv,
        "shadow_terminator_shading_offset": _shadow_terminator_value(
            obj, "shadow_terminator_shading_offset", 0.0
        ),
        "shadow_terminator_geometry_offset": _shadow_terminator_value(
            obj, "shadow_terminator_geometry_offset", 0.1
        ),
        "use_holdout": _use_holdout(object_instance, view_layer),
        "is_caustics_caster": bool(
            getattr(cycles, "is_caustics_caster", False)
        ),
        "is_caustics_receiver": bool(
            getattr(cycles, "is_caustics_receiver", False)
        ),
    }


def particle_source(
    object_instance: Any,
    scene: Any | None,
    system_id: int | None,
) -> dict[str, Any] | None:
    """Capture one parent-particle record before global table packing.

    Child particle persistent IDs are outside ParticleSystem.totpart and are
    rejected by BlenderSync::sync_dupli_particle. ``scene`` and ``system_id``
    may be omitted only by the legacy parent-index compatibility helper.
    """

    particle_system = object_instance.particle_system
    if particle_system is None:
        return None
    persistent_index = int(object_instance.persistent_id[0])
    parent_count = len(particle_system.particles)
    if persistent_index < 0 or persistent_index >= parent_count:
        return None
    if scene is None or system_id is None:
        return {"source_index": persistent_index}

    particle = particle_system.particles[persistent_index]
    return {
        "system": system_id,
        "source_index": persistent_index,
        "age": float(scene.frame_current_final) - float(particle.birth_time),
        "lifetime": float(particle.lifetime),
        "location": [float(value) for value in particle.location],
        "rotation": [float(value) for value in particle.rotation],
        "size": float(particle.size),
        "velocity": [float(value) for value in particle.velocity],
        "angular_velocity": [
            float(value) for value in particle.angular_velocity
        ],
    }


class ParticleSourceRegistry:
    """Assign equality-only IDs to Cycles ParticleSystemKey groups.

    Cycles compares the evaluated dupli parent and persistent-id tail while
    deliberately ignoring element zero. The assigned integer is not a device
    offset: the runtime will order only qualifying groups by first use and add
    the dummy-entry/global-prefix mapping.
    """

    def __init__(self, scene: Any) -> None:
        self._scene = scene
        self._systems: dict[tuple[int, tuple[int, ...]], int] = {}

    def capture(self, object_instance: Any) -> dict[str, Any] | None:
        particle_system = object_instance.particle_system
        if particle_system is None:
            return None
        persistent = tuple(int(value) for value in object_instance.persistent_id)
        if persistent[0] < 0 or persistent[0] >= len(
            particle_system.particles
        ):
            return None
        parent = object_instance.parent
        if parent is None:
            raise RuntimeError(
                "particle dupli has no parent for Cycles ParticleSystemKey"
            )
        key = (int(parent.as_pointer()), persistent[1:])
        system_id = self._systems.setdefault(key, len(self._systems))
        return particle_source(object_instance, self._scene, system_id)

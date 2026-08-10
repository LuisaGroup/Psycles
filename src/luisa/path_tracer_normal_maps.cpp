#include "path_tracer_normal_maps.h"

#include "surface_normal_map.h"

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceNormalMapInput tangent_displaced_input(
    Float3 mapped,
    Float strength,
    Float3 object_tangent,
    Float tangent_sign,
    Bool tangent_attribute_found,
    Float3 object_shading_normal,
    Float3 normal_to_world_x,
    Float3 normal_to_world_y,
    Float3 normal_to_world_z,
    Float3 shading_normal,
    Bool back_facing,
    UInt geometry_index,
    Bool is_curve) noexcept {
    return {
        .mapped = mapped,
        .strength = strength,
        .object_tangent = object_tangent,
        .tangent_sign = tangent_sign,
        .tangent_attribute_found = tangent_attribute_found,
        .object_shading_normal = object_shading_normal,
        .undisplaced_object_shading_normal = make_float3(0.0f),
        .triangle_smooth = false,
        .normal_to_world_x = normal_to_world_x,
        .normal_to_world_y = normal_to_world_y,
        .normal_to_world_z = normal_to_world_z,
        .shading_normal = shading_normal,
        .back_facing = back_facing,
        .geometry_index = geometry_index,
        .is_curve = is_curve};
}

[[nodiscard]] SurfaceNormalMapInput tangent_original_input(
    Float3 mapped,
    Float strength,
    Float3 object_tangent,
    Float tangent_sign,
    Bool tangent_attribute_found,
    Float3 object_shading_normal,
    Float3 undisplaced_object_shading_normal,
    Bool triangle_smooth,
    Float3 normal_to_world_x,
    Float3 normal_to_world_y,
    Float3 normal_to_world_z,
    Float3 shading_normal,
    Bool back_facing,
    UInt geometry_index,
    Bool is_curve) noexcept {
    return {
        .mapped = mapped,
        .strength = strength,
        .object_tangent = object_tangent,
        .tangent_sign = tangent_sign,
        .tangent_attribute_found = tangent_attribute_found,
        .object_shading_normal = object_shading_normal,
        .undisplaced_object_shading_normal =
            undisplaced_object_shading_normal,
        .triangle_smooth = triangle_smooth,
        .normal_to_world_x = normal_to_world_x,
        .normal_to_world_y = normal_to_world_y,
        .normal_to_world_z = normal_to_world_z,
        .shading_normal = shading_normal,
        .back_facing = back_facing,
        .geometry_index = geometry_index,
        .is_curve = is_curve};
}

[[nodiscard]] SurfaceNormalMapInput object_input(
    Float3 mapped,
    Float strength,
    Float3 normal_to_world_x,
    Float3 normal_to_world_y,
    Float3 normal_to_world_z,
    Float3 shading_normal,
    Bool back_facing) noexcept {
    return {
        .mapped = mapped,
        .strength = strength,
        .object_tangent = make_float3(0.0f),
        .tangent_sign = 0.0f,
        .tangent_attribute_found = false,
        .object_shading_normal = make_float3(0.0f),
        .undisplaced_object_shading_normal = make_float3(0.0f),
        .triangle_smooth = false,
        .normal_to_world_x = normal_to_world_x,
        .normal_to_world_y = normal_to_world_y,
        .normal_to_world_z = normal_to_world_z,
        .shading_normal = shading_normal,
        .back_facing = back_facing,
        .geometry_index = ~0u,
        .is_curve = false};
}

[[nodiscard]] SurfaceNormalMapInput world_input(
    Float3 mapped,
    Float strength,
    Float3 shading_normal,
    Bool back_facing) noexcept {
    return {
        .mapped = mapped,
        .strength = strength,
        .object_tangent = make_float3(0.0f),
        .tangent_sign = 0.0f,
        .tangent_attribute_found = false,
        .object_shading_normal = make_float3(0.0f),
        .undisplaced_object_shading_normal = make_float3(0.0f),
        .triangle_smooth = false,
        .normal_to_world_x = make_float3(0.0f),
        .normal_to_world_y = make_float3(0.0f),
        .normal_to_world_z = make_float3(0.0f),
        .shading_normal = shading_normal,
        .back_facing = back_facing,
        .geometry_index = ~0u,
        .is_curve = false};
}

[[nodiscard]] TangentDisplacedNormalMapCallable
make_tangent_displaced_callable() noexcept {
    TangentDisplacedNormalMapCallable callable =
        [](Float3 mapped,
           Float strength,
           Float3 object_tangent,
           Float tangent_sign,
           Bool tangent_attribute_found,
           Float3 object_shading_normal,
           Float3 normal_to_world_x,
           Float3 normal_to_world_y,
           Float3 normal_to_world_z,
           Float3 shading_normal,
           Bool back_facing,
           UInt geometry_index,
           Bool is_curve) noexcept {
        return normal_map_tangent_displaced_inline(
            tangent_displaced_input(
                mapped,
                strength,
                object_tangent,
                tangent_sign,
                tangent_attribute_found,
                object_shading_normal,
                normal_to_world_x,
                normal_to_world_y,
                normal_to_world_z,
                shading_normal,
                back_facing,
                geometry_index,
                is_curve));
    };
    callable.set_name("surface_normal_map_tangent_displaced");
    return callable;
}

[[nodiscard]] TangentOriginalNormalMapCallable
make_tangent_original_callable() noexcept {
    TangentOriginalNormalMapCallable callable =
        [](Float3 mapped,
           Float strength,
           Float3 object_tangent,
           Float tangent_sign,
           Bool tangent_attribute_found,
           Float3 object_shading_normal,
           Float3 undisplaced_object_shading_normal,
           Bool triangle_smooth,
           Float3 normal_to_world_x,
           Float3 normal_to_world_y,
           Float3 normal_to_world_z,
           Float3 shading_normal,
           Bool back_facing,
           UInt geometry_index,
           Bool is_curve) noexcept {
        return normal_map_tangent_original_inline(
            tangent_original_input(
                mapped,
                strength,
                object_tangent,
                tangent_sign,
                tangent_attribute_found,
                object_shading_normal,
                undisplaced_object_shading_normal,
                triangle_smooth,
                normal_to_world_x,
                normal_to_world_y,
                normal_to_world_z,
                shading_normal,
                back_facing,
                geometry_index,
                is_curve));
    };
    callable.set_name("surface_normal_map_tangent_original");
    return callable;
}

[[nodiscard]] ObjectNormalMapCallable
make_object_callable() noexcept {
    ObjectNormalMapCallable callable =
        [](Float3 mapped,
           Float strength,
           Float3 normal_to_world_x,
           Float3 normal_to_world_y,
           Float3 normal_to_world_z,
           Float3 shading_normal,
           Bool back_facing) noexcept {
        return normal_map_object_inline(
            object_input(
                mapped,
                strength,
                normal_to_world_x,
                normal_to_world_y,
                normal_to_world_z,
                shading_normal,
                back_facing));
    };
    callable.set_name("surface_normal_map_object");
    return callable;
}

[[nodiscard]] WorldNormalMapCallable
make_world_callable() noexcept {
    WorldNormalMapCallable callable =
        [](Float3 mapped,
           Float strength,
           Float3 shading_normal,
           Bool back_facing) noexcept {
        return normal_map_world_inline(
            world_input(
                mapped,
                strength,
                shading_normal,
                back_facing));
    };
    callable.set_name("surface_normal_map_world");
    return callable;
}

}// namespace

Float3 CallableSurfaceNormalMapProvider::
    evaluate_tangent_displaced(
        const SurfaceNormalMapInput &input) const noexcept {
    if (!_tangent_displaced) {
        _tangent_displaced.emplace(
            make_tangent_displaced_callable());
    }
    return (*_tangent_displaced)(
        input.mapped,
        input.strength,
        input.object_tangent,
        input.tangent_sign,
        input.tangent_attribute_found,
        input.object_shading_normal,
        input.normal_to_world_x,
        input.normal_to_world_y,
        input.normal_to_world_z,
        input.shading_normal,
        input.back_facing,
        input.geometry_index,
        input.is_curve);
}

Float3 CallableSurfaceNormalMapProvider::
    evaluate_tangent_original(
        const SurfaceNormalMapInput &input) const noexcept {
    if (!_tangent_original) {
        _tangent_original.emplace(
            make_tangent_original_callable());
    }
    return (*_tangent_original)(
        input.mapped,
        input.strength,
        input.object_tangent,
        input.tangent_sign,
        input.tangent_attribute_found,
        input.object_shading_normal,
        input.undisplaced_object_shading_normal,
        input.triangle_smooth,
        input.normal_to_world_x,
        input.normal_to_world_y,
        input.normal_to_world_z,
        input.shading_normal,
        input.back_facing,
        input.geometry_index,
        input.is_curve);
}

Float3 CallableSurfaceNormalMapProvider::evaluate_object(
    const SurfaceNormalMapInput &input) const noexcept {
    if (!_object) {
        _object.emplace(make_object_callable());
    }
    return (*_object)(
        input.mapped,
        input.strength,
        input.normal_to_world_x,
        input.normal_to_world_y,
        input.normal_to_world_z,
        input.shading_normal,
        input.back_facing);
}

Float3 CallableSurfaceNormalMapProvider::evaluate_world(
    const SurfaceNormalMapInput &input) const noexcept {
    if (!_world) {
        _world.emplace(make_world_callable());
    }
    return (*_world)(
        input.mapped,
        input.strength,
        input.shading_normal,
        input.back_facing);
}

}// namespace psycles::luisa_backend::detail

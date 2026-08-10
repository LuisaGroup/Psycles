#include "path_tracer_bump.h"

#include "surface_bump.h"

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceBumpInput world_input(
    Float3 normal,
    Float filter_width,
    Float3 dPdx,
    Float3 dPdy,
    Float height_center,
    Float height_x,
    Float height_y,
    Float distance,
    Float strength) noexcept {
    return {
        .normal = normal,
        .filter_width = filter_width,
        .dPdx = dPdx,
        .dPdy = dPdy,
        .height_center = height_center,
        .height_x = height_x,
        .height_y = height_y,
        .distance = distance,
        .strength = strength,
        .normal_to_world_x = make_float3(0.0f),
        .normal_to_world_y = make_float3(0.0f),
        .normal_to_world_z = make_float3(0.0f),
        .object_shading_normal = make_float3(0.0f),
        .shading_normal = make_float3(0.0f)};
}

[[nodiscard]] SurfaceBumpInput object_input(
    Float3 normal,
    Float filter_width,
    Float3 dPdx,
    Float3 dPdy,
    Float height_center,
    Float height_x,
    Float height_y,
    Float distance,
    Float strength,
    Float3 normal_to_world_x,
    Float3 normal_to_world_y,
    Float3 normal_to_world_z,
    Float3 object_shading_normal,
    Float3 shading_normal) noexcept {
    return {
        .normal = normal,
        .filter_width = filter_width,
        .dPdx = dPdx,
        .dPdy = dPdy,
        .height_center = height_center,
        .height_x = height_x,
        .height_y = height_y,
        .distance = distance,
        .strength = strength,
        .normal_to_world_x = normal_to_world_x,
        .normal_to_world_y = normal_to_world_y,
        .normal_to_world_z = normal_to_world_z,
        .object_shading_normal = object_shading_normal,
        .shading_normal = shading_normal};
}

[[nodiscard]] WorldBumpCallable
make_world_callable() noexcept {
    WorldBumpCallable callable =
        [](Float3 normal,
           Float filter_width,
           Float3 dPdx,
           Float3 dPdy,
           Float height_center,
           Float height_x,
           Float height_y,
           Float distance,
           Float strength) noexcept {
        return bump_world_inline(
            world_input(
                normal,
                filter_width,
                dPdx,
                dPdy,
                height_center,
                height_x,
                height_y,
                distance,
                strength));
    };
    callable.set_name("surface_bump_world");
    return callable;
}

[[nodiscard]] ObjectBumpCallable
make_object_callable() noexcept {
    ObjectBumpCallable callable =
        [](Float3 normal,
           Float filter_width,
           Float3 dPdx,
           Float3 dPdy,
           Float height_center,
           Float height_x,
           Float height_y,
           Float distance,
           Float strength,
           Float3 normal_to_world_x,
           Float3 normal_to_world_y,
           Float3 normal_to_world_z,
           Float3 object_shading_normal,
           Float3 shading_normal) noexcept {
        return bump_object_inline(
            object_input(
                normal,
                filter_width,
                dPdx,
                dPdy,
                height_center,
                height_x,
                height_y,
                distance,
                strength,
                normal_to_world_x,
                normal_to_world_y,
                normal_to_world_z,
                object_shading_normal,
                shading_normal));
    };
    callable.set_name("surface_bump_object");
    return callable;
}

}// namespace

Float3 CallableSurfaceBumpProvider::evaluate_world(
    const SurfaceBumpInput &input) const noexcept {
    if (!_world) {
        _world.emplace(make_world_callable());
    }
    return (*_world)(
        input.normal,
        input.filter_width,
        input.dPdx,
        input.dPdy,
        input.height_center,
        input.height_x,
        input.height_y,
        input.distance,
        input.strength);
}

Float3 CallableSurfaceBumpProvider::evaluate_object(
    const SurfaceBumpInput &input) const noexcept {
    if (!_object) {
        _object.emplace(make_object_callable());
    }
    return (*_object)(
        input.normal,
        input.filter_width,
        input.dPdx,
        input.dPdy,
        input.height_center,
        input.height_x,
        input.height_y,
        input.distance,
        input.strength,
        input.normal_to_world_x,
        input.normal_to_world_y,
        input.normal_to_world_z,
        input.object_shading_normal,
        input.shading_normal);
}

}// namespace psycles::luisa_backend::detail

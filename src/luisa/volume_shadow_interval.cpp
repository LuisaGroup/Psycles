#include <psycles/luisa/volume_shadow_interval.h>

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend {

VolumeShadowIntervalCursor::
    VolumeShadowIntervalCursor(
        luisa::compute::Float minimum) noexcept
    : _minimum{std::move(minimum)} {}

luisa::compute::Float
VolumeShadowIntervalCursor::minimum()
    const noexcept {
    return _minimum;
}

luisa::compute::Float
VolumeShadowIntervalCursor::shader_ray_length()
    const noexcept {
    return 0.0f;
}

luisa::compute::Float
VolumeShadowIntervalCursor::advance(
    luisa::compute::Float committed_t) noexcept {
    _minimum = committed_t;
    return surface_ray::
        intersection_t_offset(
            std::move(committed_t));
}

}// namespace psycles::luisa_backend

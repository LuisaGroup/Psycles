#include "path_kernel_area_light.h"

#include <utility>

namespace psycles::luisa_backend::detail {

using namespace luisa::compute;

AreaLightSampleInput
area_light_sample_input(
    Var<LightGpu> light,
    Float3 reference,
    Float2 random) noexcept {
    return {
        .reference = std::move(reference),
        .center = light.position,
        .axis_u = light.axis_x,
        .axis_v = light.axis_y,
        .axis_z = light.axis_z,
        .length_u = light.size_u,
        .length_v = light.size_v,
        .spread = light.spread,
        .ellipse =
            (light.flags &
             light_flag_ellipse) != 0u,
        .full_spread =
            (light.flags &
             light_flag_full_spread) != 0u,
        .random = std::move(random),
        .normalize_power =
            (light.flags &
             light_flag_normalize) != 0u};
}

}// namespace psycles::luisa_backend::detail

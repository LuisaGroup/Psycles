#pragma once

#include "cycles_shader_identity.h"

#include <psycles/luisa/cycles_path_state.h>

namespace psycles::luisa_backend::detail {

// Cycles kernel/light/common.h::is_light_shader_visible_to_path. Shader
// visibility is not BVH visibility: diffuse transmission still observes
// EXCLUDE_DIFFUSE; EXCLUDE_GLOSSY applies to reflection only.
[[nodiscard]] inline luisa::compute::Bool
cycles_light_shader_visible(luisa::compute::UInt shader,
                            luisa::compute::UInt visibility,
                            luisa::compute::UInt flag) noexcept {
  namespace s = cycles_shader_identity;
  namespace p = cycles_path_state;
  return !((((shader & s::exclude_diffuse) != 0u) &
            ((visibility & p::visibility_diffuse) != 0u)) |
           (((shader & s::exclude_glossy) != 0u) &
            ((visibility & p::visibility_glossy) != 0u) &
            ((flag & p::flag_reflect) != 0u)) |
           (((shader & s::exclude_transmit) != 0u) &
            ((visibility & p::visibility_transmit) != 0u)) |
           (((shader & s::exclude_camera) != 0u) &
            ((visibility & p::visibility_camera) != 0u)) |
           (((shader & s::exclude_scatter) != 0u) &
            ((visibility & p::visibility_volume_scatter) != 0u)));
}

} // namespace psycles::luisa_backend::detail

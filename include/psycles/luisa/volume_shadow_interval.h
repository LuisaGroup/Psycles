#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_shadow_interval.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

// Stateful host-stage representation of Cycles' transparent-shadow interval
// contract. A committed surface t has two deliberately distinct consumers:
// volume transport starts at the raw t, while the next surface query starts
// one representable float later. shader_setup_from_volume() also resets
// Light Path.Ray Length for every interval.
class VolumeShadowIntervalCursor {

  private:
    luisa::compute::Float _minimum;

  public:
    explicit VolumeShadowIntervalCursor(
        luisa::compute::Float minimum) noexcept;

    [[nodiscard]] luisa::compute::Float
    minimum() const noexcept;
    [[nodiscard]] luisa::compute::Float
    shader_ray_length() const noexcept;

    // Commits the raw medium boundary and returns the independently offset
    // tmin to use for the next ordered surface query.
    [[nodiscard]] luisa::compute::Float
    advance(
        luisa::compute::Float committed_t) noexcept;
};

}// namespace psycles::luisa_backend

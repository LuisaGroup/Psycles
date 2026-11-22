#pragma once

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] luisa::compute::Float fresnel_dielectric_cos(
    luisa::compute::Float cosine, luisa::compute::Float eta) noexcept;

[[nodiscard]] luisa::compute::Float f0_from_ior(
    luisa::compute::Float ior) noexcept;

[[nodiscard]] luisa::compute::Float3 fresnel_conductor(
    luisa::compute::Float cosine,
    luisa::compute::Float3 ior,
    luisa::compute::Float3 extinction) noexcept;

} // namespace psycles::luisa_backend::detail

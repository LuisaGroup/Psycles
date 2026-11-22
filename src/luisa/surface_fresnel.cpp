#include "surface_fresnel.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

using luisa::compute::abs;
using luisa::compute::clamp;
using luisa::compute::Float;
using luisa::compute::Float3;
using luisa::compute::make_float3;
using luisa::compute::max;
using luisa::compute::select;
using luisa::compute::sqrt;

Float fresnel_dielectric_cos(Float cosine, Float eta) noexcept {
    const auto c = abs(cosine);
    const auto g_squared = eta * eta - 1.0f + c * c;
    const auto g = sqrt(max(g_squared, 0.0f));
    const auto denom_a = g + c;
    const auto a = (g - c) / select(denom_a, 1.0f, denom_a == 0.0f);
    const auto denom_b = c * (g - c) + 1.0f;
    const auto b =
        (c * (g + c) - 1.0f) / select(denom_b, 1.0f, denom_b == 0.0f);
    return select(1.0f, 0.5f * a * a * (1.0f + b * b), g_squared > 0.0f);
}

Float f0_from_ior(Float ior) noexcept {
    const auto ratio = (ior - 1.0f) / max(ior + 1.0f, 1.0e-20f);
    return ratio * ratio;
}

Float3 fresnel_conductor(Float cosine, Float3 ior, Float3 extinction) noexcept {
    // Exact unpolarized Fresnel at an air/conductor interface. This is the
    // vector form of the two polarized rational equations used by Cycles;
    // n and k are already canonical non-negative physical parameters.
    const auto c = clamp(cosine, 0.0f, 1.0f);
    const auto n2 = ior * ior;
    const auto k2 = extinction * extinction;
    const auto two_nk = 2.0f * ior * extinction;
    const auto t1 = n2 - k2 - (1.0f - c * c);
    const auto t2 =
        sqrt(max(t1 * t1 + two_nk * two_nk, make_float3(0.0f)));
    const auto u2 = max(0.5f * (t2 + t1), make_float3(0.0f));
    const auto v2 = max(0.5f * (t2 - t1), make_float3(0.0f));
    const auto u = sqrt(u2);
    const auto v = sqrt(v2);
    const auto rs_numerator = (c - u) * (c - u) + v2;
    const auto rs_denominator = (c + u) * (c + u) + v2;
    const auto t3 = (n2 - k2) * c;
    const auto t4 = two_nk * c;
    const auto rp_numerator = (t3 - u) * (t3 - u) + (t4 - v) * (t4 - v);
    const auto rp_denominator = (t3 + u) * (t3 + u) + (t4 + v) * (t4 + v);
    const auto rs =
        rs_numerator / max(rs_denominator, make_float3(1.0e-20f));
    const auto rp =
        rp_numerator / max(rp_denominator, make_float3(1.0e-20f));
    return clamp(0.5f * (rs + rp), make_float3(0.0f), make_float3(1.0f));
}

} // namespace psycles::luisa_backend::detail

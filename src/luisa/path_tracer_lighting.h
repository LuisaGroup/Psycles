#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

using SafeNormalizeCallable =
    Callable<luisa::float3(luisa::float3, luisa::float3)>;
using ForwardLightWeightCallable = Callable<float(float, float, bool, bool)>;
using NeeLightWeightCallable = Callable<float(float, float)>;
using ClampLightContributionCallable =
    Callable<luisa::float3(luisa::float3, luisa::uint, float, float)>;
using LightSampleRouletteCallable =
    Callable<float(luisa::float3, float, float)>;
using LightComponentRatioCallable =
    Callable<luisa::float3(luisa::float3, luisa::float3)>;
using SplitScatteredLightCallable = Callable<LightPassContributionCall(
    luisa::float3, luisa::float3, luisa::float3, bool)>;
struct LightTransportCallables {
    SafeNormalizeCallable safe_normalize;
    ForwardLightWeightCallable forward_light_weight;
    NeeLightWeightCallable nee_light_weight;
    ClampLightContributionCallable clamp_light_contribution;
    LightSampleRouletteCallable light_sample_roulette_weight;
    LightComponentRatioCallable light_component_ratio;
    SplitScatteredLightCallable split_scattered_light;
};

[[nodiscard]] LightTransportCallables
make_light_transport_callables(contract::DirectLightSampling mode) noexcept;

}// namespace psycles::luisa_backend::detail

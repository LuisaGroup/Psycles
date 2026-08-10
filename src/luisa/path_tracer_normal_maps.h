#pragma once

#include "path_tracer_internal.h"

#include <cstdint>
#include <optional>

namespace psycles::luisa_backend::detail {

using TangentDisplacedNormalMapCallable = Callable<luisa::float3(
    luisa::float3,
    float,
    luisa::float3,
    float,
    bool,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    bool,
    std::uint32_t,
    bool)>;
using TangentOriginalNormalMapCallable = Callable<luisa::float3(
    luisa::float3,
    float,
    luisa::float3,
    float,
    bool,
    luisa::float3,
    luisa::float3,
    bool,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    bool,
    std::uint32_t,
    bool)>;
using ObjectNormalMapCallable = Callable<luisa::float3(
    luisa::float3,
    float,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    bool)>;
using WorldNormalMapCallable = Callable<luisa::float3(
    luisa::float3,
    float,
    luisa::float3,
    bool)>;

class CallableSurfaceNormalMapProvider final
    : public SurfaceNormalMapProvider {

private:
    mutable std::optional<TangentDisplacedNormalMapCallable>
        _tangent_displaced;
    mutable std::optional<TangentOriginalNormalMapCallable>
        _tangent_original;
    mutable std::optional<ObjectNormalMapCallable> _object;
    mutable std::optional<WorldNormalMapCallable> _world;

public:
    [[nodiscard]] Float3 evaluate_tangent_displaced(
        const SurfaceNormalMapInput &input) const noexcept override;

    [[nodiscard]] Float3 evaluate_tangent_original(
        const SurfaceNormalMapInput &input) const noexcept override;

    [[nodiscard]] Float3 evaluate_object(
        const SurfaceNormalMapInput &input) const noexcept override;

    [[nodiscard]] Float3 evaluate_world(
        const SurfaceNormalMapInput &input) const noexcept override;
};

}// namespace psycles::luisa_backend::detail

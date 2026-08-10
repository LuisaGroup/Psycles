#pragma once

#include "path_tracer_internal.h"

#include <optional>

namespace psycles::luisa_backend::detail {

using WorldBumpCallable = Callable<luisa::float3(
    luisa::float3,
    float,
    luisa::float3,
    luisa::float3,
    float,
    float,
    float,
    float,
    float)>;
using ObjectBumpCallable = Callable<luisa::float3(
    luisa::float3,
    float,
    luisa::float3,
    luisa::float3,
    float,
    float,
    float,
    float,
    float,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    luisa::float3,
    luisa::float3)>;

class CallableSurfaceBumpProvider final
    : public SurfaceBumpProvider {

private:
    mutable std::optional<WorldBumpCallable> _world;
    mutable std::optional<ObjectBumpCallable> _object;

public:
    [[nodiscard]] Float3 evaluate_world(
        const SurfaceBumpInput &input) const noexcept override;

    [[nodiscard]] Float3 evaluate_object(
        const SurfaceBumpInput &input) const noexcept override;
};

}// namespace psycles::luisa_backend::detail

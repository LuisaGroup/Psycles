#pragma once

#include "path_tracer_internal.h"

#include <optional>

namespace psycles::luisa_backend::detail {

using ShaderLocatedVectorMappingCallable =
    Callable<luisa::float3(
        luisa::float3,
        luisa::float3,
        luisa::float3,
        luisa::float3)>;
using ShaderLinearVectorMappingCallable =
    Callable<luisa::float3(
        luisa::float3,
        luisa::float3,
        luisa::float3)>;

class CallableSurfaceVectorMappingProvider final
    : public SurfaceVectorMappingProvider {

private:
    // Mapping mode is immutable graph metadata. Laziness keeps the callable
    // module bounded by the subset of the four semantic modes actually used
    // by reachable graph nodes.
    mutable std::optional<ShaderLocatedVectorMappingCallable> _point;
    mutable std::optional<ShaderLocatedVectorMappingCallable> _texture;
    mutable std::optional<ShaderLinearVectorMappingCallable> _vector;
    mutable std::optional<ShaderLinearVectorMappingCallable> _normal;

public:
    CallableSurfaceVectorMappingProvider() noexcept = default;

    [[nodiscard]] Float3 map_point(
        Float3 input,
        Float3 location,
        Float3 rotation,
        Float3 scale) const noexcept override;

    [[nodiscard]] Float3 map_texture(
        Float3 input,
        Float3 location,
        Float3 rotation,
        Float3 scale) const noexcept override;

    [[nodiscard]] Float3 map_vector(
        Float3 input,
        Float3 rotation,
        Float3 scale) const noexcept override;

    [[nodiscard]] Float3 map_normal(
        Float3 input,
        Float3 rotation,
        Float3 scale) const noexcept override;
};

}// namespace psycles::luisa_backend::detail

#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

// Minimal typed contract for the one semantic operation shared by every
// attribute-valued graph node. The query deliberately carries only values
// used by the lookup; passing SurfacePoint would enlarge every call site and
// obscure the dependency boundary.
struct SurfaceAttributeLookupQueryCall {
    luisa::ulong attribute_id{};
    luisa::float2 barycentric{};
    luisa::uint geometry_index{};
    luisa::uint primitive_id{};
};

struct SurfaceAttributeLookupResultCall {
    luisa::float4 value{};
    luisa::uint found{};
};

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceAttributeLookupQueryCall,
    attribute_id,
    barycentric,
    geometry_index,
    primitive_id) {};

LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceAttributeLookupResultCall,
    value,
    found) {};

namespace psycles::luisa_backend::detail {

using SurfaceAttributeLookupCallable = Callable<
    SurfaceAttributeLookupResultCall(
        BindlessArray,
        SurfaceAttributeLookupQueryCall)>;

// Canonical direct implementation. Standalone/test services retain this
// route, while production graph operations call the exact same body through
// the shared typed callable below.
[[nodiscard]] ShaderAttribute resolve_surface_attribute(
    const BindlessVar &geometry_heap,
    std::uint32_t attribute_binding_slot,
    std::uint32_t attribute_range_slot,
    Expr<luisa::ulong> attribute_id,
    Expr<std::uint32_t> geometry_index,
    Expr<std::uint32_t> primitive_id,
    Expr<luisa::float2> barycentric) noexcept;

[[nodiscard]] ShaderAttribute resolve_surface_attribute(
    const BindlessArray &geometry_heap,
    std::uint32_t attribute_binding_slot,
    std::uint32_t attribute_range_slot,
    Expr<luisa::ulong> attribute_id,
    Expr<std::uint32_t> geometry_index,
    Expr<std::uint32_t> primitive_id,
    Expr<luisa::float2> barycentric) noexcept;

[[nodiscard]] SurfaceAttributeLookupCallable
make_surface_attribute_lookup_callable(
    std::uint32_t attribute_binding_slot,
    std::uint32_t attribute_range_slot) noexcept;

class CallableSurfaceAttributeLookupProvider final {

private:
    Expr<BindlessArray> _geometry_heap;
    const SurfaceAttributeLookupCallable &_callable;

public:
    CallableSurfaceAttributeLookupProvider(
        Expr<BindlessArray> geometry_heap,
        const SurfaceAttributeLookupCallable &callable) noexcept;

    [[nodiscard]] ShaderAttribute lookup(
        Expr<luisa::ulong> attribute_id,
        const SurfacePoint &point) const noexcept;
};

} // namespace psycles::luisa_backend::detail

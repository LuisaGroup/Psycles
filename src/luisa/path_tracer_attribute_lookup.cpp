#include "path_tracer_attribute_lookup.h"

#include <utility>

namespace psycles::luisa_backend::detail {

namespace {

template<typename GeometryHeap>
[[nodiscard]] ShaderAttribute resolve_surface_attribute_impl(
    const GeometryHeap &geometry_heap,
    std::uint32_t attribute_binding_slot,
    std::uint32_t attribute_range_slot,
    Expr<luisa::ulong> attribute_id,
    Expr<std::uint32_t> geometry_index,
    Expr<std::uint32_t> primitive_id,
    Expr<luisa::float2> barycentric) noexcept {
    auto result = ShaderAttribute::missing();
    // Attribute cardinality is scene data, not shader structure. Look up only
    // the current geometry's compact range so AST/XIR size remains constant as
    // scenes add meshes, UV maps, or color attributes.
    $if (geometry_index != ~0u) {
        Var<AttributeRangeGpu> range =
            geometry_heap
                ->template buffer<AttributeRangeGpu>(attribute_range_slot)
                .read(geometry_index);
        UInt local_index = 0u;
        Bool found = false;
        $while ((local_index < range.count) & !found) {
            Var<AttributeBindingGpu> binding =
                geometry_heap
                    ->template buffer<AttributeBindingGpu>(
                        attribute_binding_slot)
                    .read(range.offset + local_index);
            $if (attribute_id == binding.id) {
                Var<Triangle> triangle =
                    geometry_heap
                        ->template buffer<Triangle>(range.triangle_slot)
                        .read(primitive_id);
                UInt i0 = triangle.i0;
                UInt i1 = triangle.i1;
                UInt i2 = triangle.i2;
                $if (binding.domain == attribute_domain_corner) {
                    const auto corner = primitive_id * 3u;
                    i0 = corner;
                    i1 = corner + 1u;
                    i2 = corner + 2u;
                };
                $if (binding.domain == attribute_domain_face) {
                    i0 = primitive_id;
                    i1 = primitive_id;
                    i2 = primitive_id;
                };
                const auto values =
                    geometry_heap
                        ->template buffer<luisa::float4>(binding.value_slot);
                const auto v0 = values.read(i0);
                const auto v1 = values.read(i1);
                const auto v2 = values.read(i2);
                result.value = triangle_interpolate(
                    barycentric, v0, v1, v2);
                found = true;
            };
            local_index += 1u;
        };
        result.found = found;
    };
    return result;
}

} // namespace

ShaderAttribute resolve_surface_attribute(
    const BindlessVar &geometry_heap,
    std::uint32_t attribute_binding_slot,
    std::uint32_t attribute_range_slot,
    Expr<luisa::ulong> attribute_id,
    Expr<std::uint32_t> geometry_index,
    Expr<std::uint32_t> primitive_id,
    Expr<luisa::float2> barycentric) noexcept {
    return resolve_surface_attribute_impl(
        geometry_heap,
        attribute_binding_slot,
        attribute_range_slot,
        attribute_id,
        geometry_index,
        primitive_id,
        barycentric);
}

ShaderAttribute resolve_surface_attribute(
    const BindlessArray &geometry_heap,
    std::uint32_t attribute_binding_slot,
    std::uint32_t attribute_range_slot,
    Expr<luisa::ulong> attribute_id,
    Expr<std::uint32_t> geometry_index,
    Expr<std::uint32_t> primitive_id,
    Expr<luisa::float2> barycentric) noexcept {
    return resolve_surface_attribute_impl(
        geometry_heap,
        attribute_binding_slot,
        attribute_range_slot,
        attribute_id,
        geometry_index,
        primitive_id,
        barycentric);
}

SurfaceAttributeLookupCallable
make_surface_attribute_lookup_callable(
    std::uint32_t attribute_binding_slot,
    std::uint32_t attribute_range_slot) noexcept {
    SurfaceAttributeLookupCallable callable =
        [attribute_binding_slot,
         attribute_range_slot](
            BindlessVar geometry_heap,
            Var<SurfaceAttributeLookupQueryCall> query) noexcept {
            const auto result = resolve_surface_attribute(
                geometry_heap,
                attribute_binding_slot,
                attribute_range_slot,
                query.attribute_id,
                query.geometry_index,
                query.primitive_id,
                query.barycentric);
            Var<SurfaceAttributeLookupResultCall> packed;
            packed.value = result.value;
            packed.found = select(0u, 1u, result.found);
            return packed;
        };
    callable.set_name("surface_attribute_lookup");
    return callable;
}

CallableSurfaceAttributeLookupProvider::
    CallableSurfaceAttributeLookupProvider(
        Expr<BindlessArray> geometry_heap,
        const SurfaceAttributeLookupCallable &callable) noexcept
    : _geometry_heap{std::move(geometry_heap)},
      _callable{callable} {}

ShaderAttribute CallableSurfaceAttributeLookupProvider::lookup(
    Expr<luisa::ulong> attribute_id,
    const SurfacePoint &point) const noexcept {
    Var<SurfaceAttributeLookupQueryCall> query;
    query.attribute_id = attribute_id;
    query.barycentric = point.barycentric;
    query.geometry_index = point.geometry_index;
    query.primitive_id = point.primitive_id;
    const auto result = _callable(_geometry_heap, query);
    return {
        .value = result.value,
        .found = result.found != 0u};
}

} // namespace psycles::luisa_backend::detail

#include "../src/luisa/path_tracer_attribute_lookup.h"

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::luisa_backend::SurfacePoint;
using psycles::test_support::approximately_equal;
using psycles::test_support::xir_instruction_count;

inline constexpr auto binding_slot = std::uint32_t{0u};
inline constexpr auto range_slot = std::uint32_t{1u};
inline constexpr auto triangle_slot = std::uint32_t{2u};
inline constexpr auto point_value_slot = std::uint32_t{3u};
inline constexpr auto corner_value_slot = std::uint32_t{4u};
inline constexpr auto face_value_slot = std::uint32_t{5u};
inline constexpr auto curve_segment_slot = std::uint32_t{6u};
inline constexpr auto curve_uv_slot = std::uint32_t{7u};
inline constexpr auto curve_color_slot = std::uint32_t{8u};
inline constexpr auto point_attribute = luisa::ulong{0x1020304050607080ull};
inline constexpr auto corner_attribute = luisa::ulong{0x2131415161718191ull};
inline constexpr auto face_attribute = luisa::ulong{0x32425262728292a2ull};
inline constexpr auto curve_uv_attribute =
    luisa::ulong{0x435363738393a3b3ull};
inline constexpr auto curve_color_attribute =
    luisa::ulong{0x5464748494a4b4c4ull};
inline constexpr auto missing_attribute = luisa::ulong{0xfedcba9876543210ull};

[[nodiscard]] Kernel1D<BindlessArray, Buffer<luisa::float4>>
make_repeated_lookup_kernel(bool shared) {
    constexpr auto repetition_count = std::uint32_t{8u};
    const auto callable = make_surface_attribute_lookup_callable(
        binding_slot, range_slot);
    return [callable, shared](
               BindlessVar geometry_heap,
               BufferFloat4 output) noexcept {
        CallableSurfaceAttributeLookupProvider provider{
            geometry_heap,
            callable};
        SurfacePoint point{};
        point.geometry_index = 0u;
        point.primitive_id = 0u;
        point.barycentric = make_float2(0.2f, 0.3f);
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const auto attribute_id =
                point_attribute + static_cast<luisa::ulong>(index & 1u);
            const auto result = shared
                                    ? provider.lookup(attribute_id, point)
                                    : resolve_surface_attribute(
                                          geometry_heap,
                                          binding_slot,
                                          range_slot,
                                          attribute_id,
                                          point.geometry_index,
                                          point.primitive_id,
                                          point.barycentric);
            output.write(index, result.value);
        }
    };
}

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape = make_repeated_lookup_kernel(false);
    const auto shared_shape = make_repeated_lookup_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout << "attribute lookup 8x XIR: inline="
                  << inline_instructions << ", shared="
                  << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions ||
        shared_shape.function()->function()
                .custom_callables()
                .size() != 1u) {
        std::cerr << "Shared attribute lookup did not bound repeated XIR: "
                  << "inline=" << inline_instructions
                  << ", shared=" << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    // Equal independently constructed callables must merge, while every
    // captured bindless-slot constant is part of the semantic hash.
    const auto first = make_surface_attribute_lookup_callable(
        binding_slot, range_slot);
    const auto second = make_surface_attribute_lookup_callable(
        binding_slot, range_slot);
    const auto different = make_surface_attribute_lookup_callable(
        range_slot, binding_slot);
    Kernel1D hash_shape = [first, second, different](
                              BindlessVar geometry_heap,
                              BufferFloat4 output) noexcept {
        SurfacePoint point{};
        point.geometry_index = 0u;
        point.barycentric = make_float2(0.25f);
        point.primitive_id = 0u;
        CallableSurfaceAttributeLookupProvider a{geometry_heap, first};
        CallableSurfaceAttributeLookupProvider b{geometry_heap, second};
        CallableSurfaceAttributeLookupProvider c{geometry_heap, different};
        output.write(0u, a.lookup(point_attribute, point).value);
        output.write(1u, b.lookup(point_attribute, point).value);
        output.write(2u, c.lookup(point_attribute, point).value);
    };
    if (hash_shape.function()->function()
            .custom_callables()
            .size() != 2u) {
        std::cerr << "Attribute callable hash omitted or failed to merge "
                     "captured slot metadata\n";
        return EXIT_FAILURE;
    }

    constexpr auto query_count = std::uint32_t{9u};
    const auto callable = make_surface_attribute_lookup_callable(
        binding_slot, range_slot);
    Kernel1D compare = [callable](
                           BindlessVar geometry_heap,
                           BufferULong attribute_ids,
                           BufferUInt2 primitive_identity,
                           BufferFloat2 barycentrics,
                           BufferFloat4 values,
                           BufferUInt found) noexcept {
        const auto query_index = dispatch_x();
        const auto identity = primitive_identity.read(query_index);
        SurfacePoint point{};
        point.geometry_index = identity.x;
        point.primitive_id = identity.y;
        point.barycentric = barycentrics.read(query_index);
        const auto attribute_id = attribute_ids.read(query_index);
        const auto direct = resolve_surface_attribute(
            geometry_heap,
            binding_slot,
            range_slot,
            attribute_id,
            point.geometry_index,
            point.primitive_id,
            point.barycentric);
        CallableSurfaceAttributeLookupProvider provider{
            geometry_heap,
            callable};
        const auto shared = provider.lookup(attribute_id, point);
        const auto output_base = query_index * 2u;
        values.write(output_base, direct.value);
        values.write(output_base + 1u, shared.value);
        found.write(output_base, select(0u, 1u, direct.found));
        found.write(output_base + 1u, select(0u, 1u, shared.found));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();

    constexpr std::array bindings{
        AttributeBindingGpu{
            .id = point_attribute,
            .value_slot = point_value_slot,
            .domain = attribute_domain_point},
        AttributeBindingGpu{
            .id = corner_attribute,
            .value_slot = corner_value_slot,
            .domain = attribute_domain_corner},
        AttributeBindingGpu{
            .id = face_attribute,
            .value_slot = face_value_slot,
            .domain = attribute_domain_face},
        AttributeBindingGpu{
            .id = curve_uv_attribute,
            .value_slot = curve_uv_slot,
            .domain = pack_attribute_layout(
                attribute_domain_curve, attribute_format_float2)},
        AttributeBindingGpu{
            .id = curve_color_attribute,
            .value_slot = curve_color_slot,
            .domain = pack_attribute_layout(
                attribute_domain_curve, attribute_format_float4)}};
    constexpr std::array ranges{
        AttributeRangeGpu{
            .offset = 0u,
            .count = 3u,
            .primitive_slot = triangle_slot},
        AttributeRangeGpu{
            .offset = 3u,
            .count = 2u,
            .primitive_slot = curve_segment_slot}};
    constexpr std::array triangles{Triangle{0u, 1u, 2u}};
    constexpr std::array point_values{
        luisa::float4{1.0f, 2.0f, 3.0f, 4.0f},
        luisa::float4{5.0f, 6.0f, 7.0f, 8.0f},
        luisa::float4{9.0f, 10.0f, 11.0f, 12.0f}};
    constexpr std::array corner_values{
        luisa::float4{0.1f, 0.2f, 0.3f, 0.4f},
        luisa::float4{0.5f, 0.6f, 0.7f, 0.8f},
        luisa::float4{0.9f, 1.0f, 1.1f, 1.2f}};
    constexpr std::array face_values{
        luisa::float4{13.0f, 14.0f, 15.0f, 16.0f}};
    constexpr std::array curve_segments{
        CurveSegmentGpu{.curve_index = 1u},
        CurveSegmentGpu{.curve_index = 0u}};
    constexpr std::array curve_uv_values{
        luisa::float2{0.11f, 0.22f},
        luisa::float2{0.77f, 0.88f}};
    constexpr std::array curve_color_values{
        luisa::float4{0.12f, 0.23f, 0.34f, 1.0f},
        luisa::float4{0.67f, 0.78f, 0.89f, 1.0f}};
    constexpr std::array attribute_ids{
        point_attribute,
        corner_attribute,
        face_attribute,
        missing_attribute,
        point_attribute,
        curve_uv_attribute,
        curve_uv_attribute,
        curve_color_attribute,
        curve_color_attribute};
    constexpr std::array identities{
        luisa::uint2{0u, 0u},
        luisa::uint2{0u, 0u},
        luisa::uint2{0u, 0u},
        luisa::uint2{0u, 0u},
        luisa::uint2{~luisa::uint{0u}, 0u},
        luisa::uint2{1u, 0u},
        luisa::uint2{1u, 1u},
        luisa::uint2{1u, 0u},
        luisa::uint2{1u, 1u}};
    constexpr std::array barycentrics{
        luisa::float2{0.2f, 0.3f},
        luisa::float2{0.6f, 0.1f},
        luisa::float2{0.4f, 0.4f},
        luisa::float2{0.2f, 0.2f},
        luisa::float2{0.2f, 0.3f},
        luisa::float2{0.9f, 0.9f},
        luisa::float2{0.1f, 0.8f},
        luisa::float2{0.7f, 0.1f},
        luisa::float2{0.3f, 0.6f}};
    // These values independently encode the semantic contract rather than
    // only comparing two implementations of it. Triangle interpolation uses
    // weights (1 - u - v, u, v); face values are constant by construction.
    constexpr std::array expected_values{
        luisa::float4{4.2f, 5.2f, 6.2f, 7.2f},
        luisa::float4{0.42f, 0.52f, 0.62f, 0.72f},
        luisa::float4{13.0f, 14.0f, 15.0f, 16.0f},
        luisa::float4{0.0f},
        luisa::float4{0.0f},
        luisa::float4{0.77f, 0.88f, 0.0f, 0.0f},
        luisa::float4{0.11f, 0.22f, 0.0f, 0.0f},
        luisa::float4{0.67f, 0.78f, 0.89f, 1.0f},
        luisa::float4{0.12f, 0.23f, 0.34f, 1.0f}};
    constexpr std::array expected_found{
        1u, 1u, 1u, 0u, 0u, 1u, 1u, 1u, 1u};

    auto binding_buffer =
        device.create_buffer<AttributeBindingGpu>(bindings.size());
    auto range_buffer =
        device.create_buffer<AttributeRangeGpu>(ranges.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto point_value_buffer =
        device.create_buffer<luisa::float4>(point_values.size());
    auto corner_value_buffer =
        device.create_buffer<luisa::float4>(corner_values.size());
    auto face_value_buffer =
        device.create_buffer<luisa::float4>(face_values.size());
    auto curve_segment_buffer =
        device.create_buffer<CurveSegmentGpu>(curve_segments.size());
    auto curve_uv_buffer =
        device.create_buffer<luisa::float2>(curve_uv_values.size());
    auto curve_color_buffer =
        device.create_buffer<luisa::float4>(curve_color_values.size());
    auto geometry_heap = device.create_bindless_array(9u);
    geometry_heap.emplace_on_update(binding_slot, binding_buffer);
    geometry_heap.emplace_on_update(range_slot, range_buffer);
    geometry_heap.emplace_on_update(triangle_slot, triangle_buffer);
    geometry_heap.emplace_on_update(point_value_slot, point_value_buffer);
    geometry_heap.emplace_on_update(corner_value_slot, corner_value_buffer);
    geometry_heap.emplace_on_update(face_value_slot, face_value_buffer);
    geometry_heap.emplace_on_update(curve_segment_slot, curve_segment_buffer);
    geometry_heap.emplace_on_update(curve_uv_slot, curve_uv_buffer);
    geometry_heap.emplace_on_update(curve_color_slot, curve_color_buffer);

    auto attribute_id_buffer =
        device.create_buffer<luisa::ulong>(query_count);
    auto identity_buffer =
        device.create_buffer<luisa::uint2>(query_count);
    auto barycentric_buffer =
        device.create_buffer<luisa::float2>(query_count);
    auto value_output =
        device.create_buffer<luisa::float4>(query_count * 2u);
    auto found_output =
        device.create_buffer<luisa::uint>(query_count * 2u);
    auto shader = device.compile(compare);
    std::vector<luisa::float4> actual_values(query_count * 2u);
    std::vector<luisa::uint> actual_found(query_count * 2u);
    stream << binding_buffer.copy_from(luisa::span{bindings})
           << range_buffer.copy_from(luisa::span{ranges})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << point_value_buffer.copy_from(luisa::span{point_values})
           << corner_value_buffer.copy_from(luisa::span{corner_values})
           << face_value_buffer.copy_from(luisa::span{face_values})
           << curve_segment_buffer.copy_from(luisa::span{curve_segments})
           << curve_uv_buffer.copy_from(luisa::span{curve_uv_values})
           << curve_color_buffer.copy_from(luisa::span{curve_color_values})
           << attribute_id_buffer.copy_from(luisa::span{attribute_ids})
           << identity_buffer.copy_from(luisa::span{identities})
           << barycentric_buffer.copy_from(luisa::span{barycentrics})
           << geometry_heap.update()
           << shader(
                  geometry_heap,
                  attribute_id_buffer,
                  identity_buffer,
                  barycentric_buffer,
                  value_output,
                  found_output)
                  .dispatch(query_count)
           << value_output.copy_to(luisa::span{actual_values})
           << found_output.copy_to(luisa::span{actual_found})
           << synchronize();

    for (auto query = std::uint32_t{0u}; query < query_count; ++query) {
        const auto base = query * 2u;
        if (!approximately_equal(
                actual_values[base], actual_values[base + 1u], 1.0e-7f) ||
            actual_found[base] != actual_found[base + 1u] ||
            !approximately_equal(
                actual_values[base + 1u], expected_values[query], 1.0e-6f) ||
            actual_found[base + 1u] != expected_found[query]) {
            std::cerr << "Attribute lookup contract mismatch on " << backend
                      << " at query " << query << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

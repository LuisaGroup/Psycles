#include "../src/luisa/path_tracer_internal.h"
#include "../src/luisa/path_tracer_tangent_space.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using psycles::luisa_backend::detail::AttributeUpload;
using psycles::luisa_backend::detail::GeometryUpload;
using psycles::luisa_backend::detail::UvTangentLayerUpload;
using psycles::luisa_backend::detail::attribute_domain_corner;
using psycles::luisa_backend::detail::geometry_uv_corner;
using psycles::luisa_backend::detail::initialize_cycles_undisplaced_tangent_space;
using psycles::luisa_backend::detail::recompute_cycles_tangent_space;

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

[[nodiscard]] bool near(float lhs, float rhs) noexcept {
    return std::abs(lhs - rhs) <= 2.0e-5f;
}

void require_unit_sign(float value, std::string_view message) {
    require(std::isfinite(value) && near(std::abs(value), 1.0f), message);
}

[[nodiscard]] GeometryUpload make_triangle() {
    GeometryUpload upload;
    upload.attribute_domains = geometry_uv_corner;
    upload.default_uv_available = true;
    upload.positions = {
        luisa::make_float3(0.0f, 0.0f, 0.0f),
        luisa::make_float3(1.0f, 0.0f, 0.0f),
        luisa::make_float3(0.0f, 1.0f, 0.0f)};
    upload.normals = {
        luisa::make_float3(0.0f, 0.0f, 1.0f),
        luisa::make_float3(0.0f, 0.0f, 1.0f),
        luisa::make_float3(0.0f, 0.0f, 1.0f)};
    upload.uv = {
        luisa::make_float2(0.0f, 0.0f),
        luisa::make_float2(1.0f, 0.0f),
        luisa::make_float2(0.0f, 1.0f)};
    upload.triangles.emplace_back(
        luisa::compute::Triangle{0u, 1u, 2u});
    upload.triangle_smooth = {0u};

    AttributeUpload named_uv;
    named_uv.id = psycles::contract::uv_attribute_id("MappedUV");
    named_uv.domain = attribute_domain_corner;
    named_uv.values = {
        luisa::make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        luisa::make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        luisa::make_float4(0.0f, 1.0f, 0.0f, 0.0f)};
    upload.attributes.emplace_back(std::move(named_uv));

    AttributeUpload current;
    current.id = psycles::contract::uv_tangent_attribute_id("MappedUV");
    current.domain = attribute_domain_corner;
    upload.attributes.emplace_back(std::move(current));

    AttributeUpload original;
    original.id =
        psycles::contract::uv_undisplaced_tangent_attribute_id("MappedUV");
    original.domain = attribute_domain_corner;
    upload.attributes.emplace_back(std::move(original));
    upload.uv_tangent_layers.emplace_back(UvTangentLayerUpload{
        .uv_attribute_index = 0u,
        .tangent_attribute_index = 1u,
        .undisplaced_tangent_attribute_index = 2u});
    return upload;
}

void test_current_and_original_frames_are_distinct() {
    auto upload = make_triangle();
    recompute_cycles_tangent_space(upload);
    initialize_cycles_undisplaced_tangent_space(upload);
    upload.undisplaced_uv_tangents = upload.uv_tangents;

    require(upload.uv_tangents.size() == 3u,
            "standard MikkTSpace output is not corner-domain");
    require(upload.attributes[1u].values.size() == 3u &&
                upload.attributes[2u].values.size() == 3u,
            "named current/original tangent attributes were not generated");
    for (std::size_t corner = 0u; corner < 3u; ++corner) {
        const auto standard = upload.uv_tangents[corner];
        const auto named = upload.attributes[1u].values[corner];
        const auto original = upload.attributes[2u].values[corner];
        require(near(standard.x, 1.0f) && near(standard.y, 0.0f) &&
                    near(standard.z, 0.0f),
                "planar standard tangent differs from Cycles MikkTSpace");
        require(near(named.x, standard.x) && near(named.y, standard.y) &&
                    near(named.z, standard.z) && near(named.w, standard.w),
                "named UV tangent did not use the standard Mikk algorithm");
        require(near(original.x, named.x) && near(original.y, named.y) &&
                    near(original.z, named.z) && near(original.w, named.w),
                "ORIGINAL tangent was not initialized from the initial frame");
        require_unit_sign(standard.w, "standard tangent sign is invalid");
    }

    // A real vertex displacement changes dP/du. Cycles regenerates the
    // current Mikk frame from the deformed mesh while ORIGINAL remains an
    // immutable attribute of the pre-displacement mesh.
    upload.positions[1u].z = 1.0f;
    recompute_cycles_tangent_space(upload);

    for (std::size_t corner = 0u; corner < 3u; ++corner) {
        const auto current = upload.uv_tangents[corner];
        const auto standard_original = upload.undisplaced_uv_tangents[corner];
        const auto named_current = upload.attributes[1u].values[corner];
        const auto named_original = upload.attributes[2u].values[corner];
        require(current.z > 0.6f && current.x > 0.6f,
                "post-displacement standard tangent was not regenerated");
        require(near(standard_original.x, 1.0f) &&
                    near(standard_original.z, 0.0f),
                "standard ORIGINAL tangent mutated with displaced geometry");
        require(named_current.z > 0.6f && named_current.x > 0.6f,
                "named DISPLACED tangent was not regenerated");
        require(near(named_original.x, 1.0f) &&
                    near(named_original.z, 0.0f),
                "named ORIGINAL tangent mutated with displaced geometry");
        require_unit_sign(current.w, "displaced tangent sign is invalid");
        require_unit_sign(named_current.w,
                          "named displaced tangent sign is invalid");
    }
}

} // namespace

int main() {
    test_current_and_original_frames_are_distinct();
    return EXIT_SUCCESS;
}

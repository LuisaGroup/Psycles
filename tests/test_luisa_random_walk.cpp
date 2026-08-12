#include "luisa_surface_test_support.h"
#include "subsurface_random_walk_component.h"

#include <psycles/luisa/cycles_sample_mapping.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;

inline constexpr std::uint32_t record_count = 22u;

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

[[nodiscard]] bool close(luisa::float4 actual,
                         luisa::float4 expected,
                         float tolerance = 4.0e-5f) noexcept {
    return approximately_equal(actual, expected, tolerance);
}

void dump(std::string_view backend,
          const std::array<luisa::float4, record_count> &values) {
    std::cerr << "Random Walk regression failed on " << backend << '\n';
    for (std::uint32_t index = 0u; index < record_count; ++index) {
        const auto value = values[index];
        std::cerr << "  " << index << ": {" << value.x << ", "
                  << value.y << ", " << value.z << ", "
                  << value.w << "}\n";
    }
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    Kernel1D probe = [](BufferFloat4 output) noexcept {
        const SubsurfaceRandomWalkComponent random_walk;
        const auto albedo = make_float3(0.6f, 0.4f, 0.2f);
        const auto radius = make_float3(0.3f, 0.5f, 0.8f);
        const auto throughput = make_float3(0.3f, 0.24f, 0.14f);
        const auto standard = random_walk.coefficients(
            static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk),
            albedo,
            radius,
            0.35f,
            throughput);
        const auto legacy = random_walk.coefficients(
            static_cast<std::uint32_t>(
                SurfaceBssrdfMethod::random_walk_legacy),
            albedo,
            radius,
            0.35f,
            throughput);
        const auto negative_legacy = random_walk.coefficients(
            static_cast<std::uint32_t>(
                SurfaceBssrdfMethod::random_walk_legacy),
            albedo,
            radius,
            -0.35f,
            throughput);
        const auto skin = random_walk.coefficients(
            static_cast<std::uint32_t>(
                SurfaceBssrdfMethod::random_walk_skin),
            albedo,
            radius,
            0.2f,
            throughput);
        const auto low_albedo = random_walk.coefficients(
            static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk),
            make_float3(0.01f, 0.025f, 0.05f),
            make_float3(1.0f),
            0.0f,
            make_float3(0.01f, 0.025f, 0.05f));

        output.write(0u, make_float4(
            standard.sigma_t, select(0.0f, 1.0f, standard.valid)));
        output.write(1u, make_float4(standard.alpha, 0.0f));
        output.write(2u, make_float4(standard.sigma_s, 0.0f));
        output.write(3u, make_float4(standard.throughput, 0.0f));
        output.write(4u, make_float4(legacy.sigma_t, 0.0f));
        output.write(5u, make_float4(legacy.alpha, 0.0f));
        output.write(6u, make_float4(legacy.sigma_s, 0.0f));
        output.write(7u, make_float4(negative_legacy.sigma_t, 0.0f));
        output.write(8u, make_float4(negative_legacy.alpha, 0.0f));
        output.write(9u, make_float4(skin.sigma_t, 0.0f));
        output.write(10u, make_float4(skin.alpha, 0.0f));
        output.write(11u, make_float4(low_albedo.alpha, 0.0f));
        output.write(12u, make_float4(low_albedo.throughput, 0.0f));

        auto point = make_surface_point();
        point.incoming = normalize(make_float3(0.2f, -0.1f, 0.974679434f));
        auto closure = SurfaceSample::zero();
        closure.bssrdf_normal = make_float3(0.0f, 0.0f, 1.0f);
        closure.bssrdf_ior = 1.4f;
        closure.bssrdf_roughness = 0.36f;

        closure.bssrdf_method = static_cast<std::uint32_t>(
            SurfaceBssrdfMethod::random_walk);
        const auto regular_random = make_float2(0.73f, 0.19f);
        const auto regular_entry = random_walk.sample_entry(
            point, closure, regular_random);
        const auto regular_half =
            cycles_sample_mapping::sample_ggx_visible_normal(
                closure.bssrdf_normal,
                point.incoming,
                closure.bssrdf_roughness,
                regular_random);
        const auto regular_half_cosine = dot(regular_half, point.incoming);
        const auto inverse_ior = 1.0f / closure.bssrdf_ior;
        const auto regular_transmitted_cosine = max(
            sqrt(max(1.0f - inverse_ior * inverse_ior *
                                (1.0f - regular_half_cosine *
                                            regular_half_cosine),
                     0.0f)),
            1.0e-7f);
        const auto regular_reference =
            -inverse_ior * point.incoming +
            (inverse_ior * regular_half_cosine -
             regular_transmitted_cosine) *
                regular_half;
        output.write(13u, make_float4(
            regular_entry.direction,
            select(0.0f, 1.0f, regular_entry.valid)));
        output.write(14u, make_float4(regular_reference, 1.0f));

        closure.bssrdf_method = static_cast<std::uint32_t>(
            SurfaceBssrdfMethod::random_walk_skin);
        closure.bssrdf_roughness = 1.0f;
        const auto diffuse_entry = random_walk.sample_entry(
            point, closure, make_float2(0.2f, 0.7f));
        const auto diffuse_reference =
            cycles_sample_mapping::sample_cosine_hemisphere(
                -closure.bssrdf_normal, make_float2(0.4f, 0.7f))
                .direction;
        output.write(15u, make_float4(
            diffuse_entry.direction,
            select(0.0f, 1.0f, diffuse_entry.valid)));
        output.write(16u, make_float4(diffuse_reference, 1.0f));

        const auto skin_refractive_entry = random_walk.sample_entry(
            point, closure, make_float2(0.75f, 0.19f));
        const auto skin_half =
            cycles_sample_mapping::sample_ggx_visible_normal(
                closure.bssrdf_normal,
                point.incoming,
                closure.bssrdf_roughness,
                make_float2(0.5f, 0.19f));
        const auto skin_half_cosine = dot(skin_half, point.incoming);
        const auto skin_transmitted_cosine = max(
            sqrt(max(1.0f - inverse_ior * inverse_ior *
                                (1.0f - skin_half_cosine *
                                            skin_half_cosine),
                     0.0f)),
            1.0e-7f);
        const auto skin_reference =
            -inverse_ior * point.incoming +
            (inverse_ior * skin_half_cosine - skin_transmitted_cosine) *
                skin_half;
        output.write(17u, make_float4(
            skin_refractive_entry.direction,
            select(0.0f, 1.0f, skin_refractive_entry.valid)));
        output.write(18u, make_float4(skin_reference, 1.0f));

        point.geometric_normal = make_float3(0.0f, 0.0f, -1.0f);
        const auto rejected = random_walk.sample_entry(
            point, closure, make_float2(0.2f, 0.7f));
        output.write(19u, make_float4(
            rejected.direction,
            select(0.0f, 1.0f, rejected.valid)));

        Var<luisa::compute::CommittedHit> source_hit;
        source_hit.inst = 17u;
        source_hit.prim = 23u;
        source_hit.bary = make_float2(0.25f, 0.625f);
        source_hit.hit_type = static_cast<std::uint32_t>(
            luisa::compute::HitType::Surface);
        source_hit.committed_ray_t = 4.75f;
        PendingSubsurfaceHit pending_hit{
            .instance = 0u,
            .primitive = 0u,
            .barycentric = make_float2(0.0f),
            .committed_ray_t = 0.0f};
        pending_hit.store_surface(source_hit);
        const auto restored_hit = pending_hit.materialize_surface();
        output.write(20u,
                     make_float4(cast<float>(restored_hit->inst),
                                 cast<float>(restored_hit->prim),
                                 restored_hit->bary));
        output.write(21u,
                     make_float4(restored_hit->committed_ray_t,
                                 cast<float>(restored_hit->hit_type),
                                 0.0f,
                                 0.0f));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(record_count);
    auto kernel = device.compile(probe);
    std::array<luisa::float4, record_count> actual{};
    stream << kernel(output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (const auto value : actual) {
        if (!finite(value)) {
            dump(backend, actual);
            return EXIT_FAILURE;
        }
    }

    // Constants below are direct evaluations of the current Cycles
    // subsurface_random_walk_remap equations. They are fixture values, not a
    // second renderer or an independently invented material model.
    const auto coefficients_ok =
        close(actual[0u], {3.33333333f, 2.0f, 1.25f, 1.0f}) &&
        close(actual[1u], {0.96771446f, 0.89767403f, 0.70935001f, 0.0f}) &&
        close(actual[2u], {3.22571487f, 1.79534806f, 0.88668751f, 0.0f}) &&
        close(actual[3u], {0.5f, 0.6f, 0.7f, 0.0f}) &&
        close(actual[4u], {5.12820513f, 3.07692308f, 1.92307692f, 0.0f}) &&
        close(actual[5u], {0.96967342f, 0.90115015f, 0.71457835f, 0.0f}) &&
        close(actual[6u], {4.97268423f, 2.77276970f, 1.37418914f, 0.0f}) &&
        close(actual[7u], {3.33333333f, 2.0f, 1.25f, 0.0f}) &&
        close(actual[8u], {0.93519850f, 0.80857169f, 0.54024818f, 0.0f}) &&
        close(actual[9u], {4.16666667f, 2.5f, 1.5625f, 0.0f}) &&
        close(actual[10u], {0.96036394f, 0.87731575f, 0.66491312f, 0.0f}) &&
        close(actual[11u], {0.2f, 0.2f, 0.21848890f, 0.0f}) &&
        close(actual[12u], {0.24463722f, 0.58549397f, 1.0f, 0.0f});
    const auto entries_ok =
        close(actual[13u], actual[14u]) &&
        close(actual[15u], actual[16u]) &&
        close(actual[17u], actual[18u]) &&
        approximately_equal(actual[19u].w, 0.0f);
    const auto pending_hit_ok =
        close(actual[20u], {17.0f, 23.0f, 0.25f, 0.625f}) &&
        close(actual[21u],
              {4.75f,
               static_cast<float>(luisa::compute::HitType::Surface),
               0.0f,
               0.0f});
    if (!coefficients_ok || !entries_ok || !pending_hit_ok) {
        dump(backend, actual);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

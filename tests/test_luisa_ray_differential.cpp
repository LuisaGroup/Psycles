#include <psycles/luisa/cycles_ray_differential.h>
#include <psycles/luisa/surface_closure_evaluation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

[[nodiscard]] bool approximately_equal(
    float actual, float expected) noexcept {
    return std::abs(actual - expected) <=
           2.0e-6f *
               (1.0f + std::max(
                           std::abs(actual),
                           std::abs(expected)));
}

} // namespace

int main(int argc, char **argv) {
    using namespace luisa::compute;
    using psycles::luisa_backend::cycles_ray_differential::
        after_surface_bounce;
    using psycles::luisa_backend::cycles_ray_differential::
        for_surface_shadow;

    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(4u);
    Kernel1D kernel = [&](BufferFloat4 result) noexcept {
        const auto diffuse = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 1.0f, false);
        const auto narrower = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 0.04f, false);
        const auto rough = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 0.36f, false);
        const auto transparent = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 1.0f, true);
        const auto shadow = for_surface_shadow(
            0.25f, 0.5f, 0.36f);
        psycles::luisa_backend::SurfaceClosureEvaluationAccumulator
            mixture;
        Var<psycles::luisa_backend::
                SurfaceClosureEvaluationContributionCall>
            diffuse_contribution;
        diffuse_contribution->f = make_float3(0.0f);
        diffuse_contribution->diffuse_f = make_float3(0.0f);
        diffuse_contribution->glossy_f = make_float3(0.0f);
        diffuse_contribution->total_sample_weight = 3.0f;
        diffuse_contribution->weighted_pdf = 2.0f;
        diffuse_contribution->weighted_roughness_squared = 2.0f;
        diffuse_contribution->events =
            static_cast<std::uint32_t>(
                psycles::contract::event_diffuse);
        mixture.add(diffuse_contribution);
        Var<psycles::luisa_backend::
                SurfaceClosureEvaluationContributionCall>
            rough_contribution;
        rough_contribution->f = make_float3(0.0f);
        rough_contribution->diffuse_f = make_float3(0.0f);
        rough_contribution->glossy_f = make_float3(0.0f);
        rough_contribution->total_sample_weight = 5.0f;
        rough_contribution->weighted_pdf = 6.0f;
        rough_contribution->weighted_roughness_squared = 0.24f;
        rough_contribution->events =
            static_cast<std::uint32_t>(
                psycles::contract::event_glossy);
        mixture.add(rough_contribution);
        Var<psycles::luisa_backend::
                SurfaceClosureEvaluationContributionCall>
            singular_contribution;
        singular_contribution->f = make_float3(0.0f);
        singular_contribution->diffuse_f = make_float3(0.0f);
        singular_contribution->glossy_f = make_float3(0.0f);
        singular_contribution->total_sample_weight = 2.0f;
        singular_contribution->weighted_pdf = 2.0f;
        singular_contribution->weighted_roughness_squared = 0.0f;
        singular_contribution->events =
            static_cast<std::uint32_t>(
                psycles::contract::event_singular);
        mixture.add(singular_contribution);
        const auto mixed = mixture.finish(true);
        const auto filtered = mixture.finish(false);
        result.write(0u,
            make_float4(
                diffuse.position,
                diffuse.direction,
                narrower.position,
                narrower.direction));
        result.write(1u,
            make_float4(
                rough.position,
                rough.direction,
                transparent.position,
                transparent.direction));
        result.write(2u,
            make_float4(
                shadow.position,
                shadow.direction,
                0.0f,
                0.0f));
        result.write(3u,
            make_float4(
                mixed.average_roughness_squared,
                mixed.pdf,
                filtered.average_roughness_squared,
                filtered.pdf));
    };
    auto compiled = device.compile(kernel);
    std::array<luisa::float4, 4u> actual{};
    stream << compiled(output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    constexpr std::array expected{
        luisa::float4{0.5f, 1.0f, 0.5f, 0.25f},
        luisa::float4{0.5f, 0.6f, 0.125f, 0.25f},
        luisa::float4{0.5f, 0.6f, 0.0f, 0.0f},
        luisa::float4{0.224f, 1.0f, 0.224f, 0.0f}};
    for (std::size_t record = 0u;
         record < actual.size();
         ++record) {
        const auto got = actual[record];
        const auto want = expected[record];
        if (!approximately_equal(got.x, want.x) ||
            !approximately_equal(got.y, want.y) ||
            !approximately_equal(got.z, want.z) ||
            !approximately_equal(got.w, want.w)) {
            std::cerr
                << "Cycles surface ray-differential regression on "
                << backend << " at record " << record
                << ": got {" << got.x << ", " << got.y
                << ", " << got.z << ", " << got.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

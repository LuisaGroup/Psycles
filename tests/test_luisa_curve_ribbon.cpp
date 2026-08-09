#include "curve_ribbon_component.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

void expect_near(
    float actual,
    float expected,
    float tolerance,
    const std::string &message) {
    expect(
        std::abs(actual - expected) <= tolerance,
        message + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
}

void expect_runtime_subdivision_loop(
    const std::shared_ptr<const CurveRibbonComponent> &ribbon) {
    Kernel1D shape = [ribbon](BufferUInt subdivision,
                              BufferFloat4 output) noexcept {
        const auto ray = make_ray(
            make_float3(0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            0.0f,
            10.0f);
        const CurveControlPoints curve{
            .before = make_float4(-2.0f, 0.0f, 2.0f, 0.5f),
            .begin = make_float4(-1.0f, 0.0f, 2.0f, 0.5f),
            .end = make_float4(1.0f, 0.0f, 2.0f, 0.5f),
            .after = make_float4(2.0f, 0.0f, 2.0f, 0.5f)};
        const auto intersection = ribbon->intersect(
            ray, curve, subdivision.read(0u));
        output.write(
            0u,
            make_float4(
                intersection.distance,
                intersection.u,
                intersection.v,
                select(0.0f, 1.0f, intersection.valid)));
    };
    auto module = luisa::compute::xir::ast_to_xir_translate(
        shape.function()->function(), {});
    std::size_t loops = 0u;
    for (auto *function : module->function_list()) {
        if (auto *definition = function->definition()) {
            definition->traverse_instructions(
                [&](const luisa::compute::xir::Instruction
                        *instruction) noexcept {
                    loops +=
                        instruction->isa<
                            luisa::compute::xir::LoopInst>() ||
                                instruction->isa<
                                    luisa::compute::xir::SimpleLoopInst>()
                            ? 1u
                            : 0u;
                });
        }
    }
    expect(
        loops == 1u,
        "runtime curve subdivision did not lower to exactly one device loop");
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    try {
        const auto ribbon = make_curve_ribbon_component();
        expect_runtime_subdivision_loop(ribbon);

        Context context{argv[0]};
        auto device = context.create_device(backend);
        auto stream = device.create_stream();

        constexpr std::array identity_bounds{
            AABB{
                .packed_min = {-2.1f, -0.6f, 1.9f},
                .packed_max = {2.1f, 0.6f, 2.1f}},
            AABB{
                .packed_min = {1.9f, -0.6f, 0.65f},
                .packed_max = {6.1f, 0.6f, 0.85f}},
            AABB{
                .packed_min = {5.9f, -0.6f, 0.9f},
                .packed_max = {10.1f, 0.6f, 1.1f}}};
        constexpr std::array transformed_bounds{
            AABB{
                .packed_min = {-2.1f, -0.6f, 1.9f},
                .packed_max = {2.1f, 0.6f, 2.1f}}};
        auto identity_bounds_buffer =
            device.create_buffer<AABB>(identity_bounds.size());
        auto transformed_bounds_buffer =
            device.create_buffer<AABB>(transformed_bounds.size());
        auto identity_curves = device.create_procedural_primitive(
            identity_bounds_buffer);
        auto transformed_curves = device.create_procedural_primitive(
            transformed_bounds_buffer);
        auto accel = device.create_accel();
        accel.emplace_back(identity_curves);
        const auto transformed_object_to_world = make_float4x4(
            make_float4(1.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 1.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 2.0f, 0.0f),
            make_float4(12.0f, 0.0f, 0.0f, 1.0f));
        accel.emplace_back(
            transformed_curves,
            transformed_object_to_world);

        auto output = device.create_buffer<luisa::float4>(5u);
        Kernel1D trace = [ribbon](
                             BufferFloat4 records,
                             AccelVar scene_accel) noexcept {
            const UInt test = dispatch_x();
            $if(test == 0u) {
                const CurveControlPoints polynomial{
                    .before = make_float4(0.0f),
                    .begin = make_float4(1.0f),
                    .end = make_float4(4.0f),
                    .after = make_float4(9.0f)};
                records.write(
                    4u,
                    make_float4(
                        ribbon->evaluate(polynomial, 0.25f).x,
                        ribbon->derivative(polynomial, 0.25f).x,
                        0.0f,
                        0.0f));
            };
            const Float ray_x = select(
                cast<float>(test) * 4.0f,
                12.0f,
                test == 3u);
            const Float ray_y = select(0.0f, 0.25f, test == 0u);
            const Float3 ray_direction = select(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, 2.0f),
                test == 3u);
            const auto world_ray = make_ray(
                make_float3(ray_x, ray_y, 0.0f),
                ray_direction,
                0.0f,
                10.0f);

            const auto control_points =
                [](UInt instance, UInt primitive) noexcept {
                    auto center_x = cast<float>(primitive) * 4.0f;
                    auto depth = select(
                        select(2.0f, 0.75f, primitive == 1u),
                        1.0f,
                        primitive == 2u);
                    center_x = select(
                        center_x, 0.0f, instance == 1u);
                    depth = select(depth, 2.0f, instance == 1u);
                    return CurveControlPoints{
                        .before = make_float4(
                            center_x - 2.0f, 0.0f, depth, 0.5f),
                        .begin = make_float4(
                            center_x - 1.0f, 0.0f, depth, 0.5f),
                        .end = make_float4(
                            center_x + 1.0f, 0.0f, depth, 0.5f),
                        .after = make_float4(
                            center_x + 2.0f, 0.0f, depth, 0.5f)};
                };
            const auto object_ray =
                [&](UInt instance,
                    const Var<Ray> &ray) noexcept {
                    const auto world_to_object = inverse(
                        scene_accel->instance_transform(instance));
                    return make_ray(
                        (world_to_object *
                         make_float4(ray->origin(), 1.0f))
                            .xyz(),
                        (world_to_object *
                         make_float4(ray->direction(), 0.0f))
                            .xyz(),
                        ray->t_min(),
                        ray->t_max());
                };

            const auto committed =
                scene_accel
                    .traverse(world_ray, {})
                    .on_surface_candidate(
                        [](SurfaceCandidate &) noexcept {})
                    .on_procedural_candidate(
                        [&](ProceduralCandidate &candidate) noexcept {
                            const auto candidate_hit = candidate.hit();
                            const auto intersection = ribbon->intersect(
                                object_ray(
                                    candidate_hit->inst,
                                    candidate.ray()),
                                control_points(
                                    candidate_hit->inst,
                                    candidate_hit->prim),
                                2u);
                            $if(intersection.valid) {
                                candidate.commit(intersection.distance);
                            };
                        })
                    .trace();

            Float distance = -1.0f;
            Float u = 0.0f;
            Float v = 0.0f;
            Float procedural = 0.0f;
            $if(committed->is_procedural()) {
                const auto intersection = ribbon->intersect(
                    object_ray(committed->inst, world_ray),
                    control_points(committed->inst, committed->prim),
                    2u);
                distance = committed->committed_ray_t;
                u = intersection.u;
                v = intersection.v;
                procedural = 1.0f;
            };
            records.write(
                test,
                make_float4(distance, u, v, procedural));
        };
        auto shader = device.compile(trace);
        std::array<luisa::float4, 5u> results{};
        stream
            << identity_bounds_buffer.copy_from(
                   luisa::span{identity_bounds})
            << transformed_bounds_buffer.copy_from(
                   luisa::span{transformed_bounds})
            << identity_curves.build()
            << transformed_curves.build()
            << accel.build()
            << shader(output, accel).dispatch(4u)
            << output.copy_to(luisa::span{results})
            << synchronize();

        expect_near(results[0u].x, 2.0f, 2.0e-6f, "front ribbon distance");
        expect_near(results[0u].y, 0.5f, 2.0e-6f, "front ribbon u");
        expect_near(results[0u].z, 0.5f, 2.0e-6f, "front ribbon v");
        expect(results[0u].w == 1.0f, "front ribbon did not commit");

        expect(results[1u].x == -1.0f,
               "ribbon inside the strict 2r exclusion committed");
        expect(results[2u].x == -1.0f,
               "ribbon on the strict 2r boundary committed");

        expect_near(
            results[3u].x, 2.0f, 2.0e-6f,
            "transformed ribbon did not preserve the ray parameter");
        expect_near(results[3u].y, 0.5f, 2.0e-6f, "transformed ribbon u");
        expect_near(results[3u].z, 0.0f, 2.0e-6f, "transformed ribbon v");
        expect(results[3u].w == 1.0f,
               "transformed ribbon did not commit");
        expect_near(
            results[4u].x, 1.5625f, 1.0e-7f,
            "Catmull-Rom polynomial evaluation");
        expect_near(
            results[4u].y, 2.5f, 1.0e-7f,
            "Catmull-Rom polynomial derivative");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "curve ribbon test (" << backend
                  << ") failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

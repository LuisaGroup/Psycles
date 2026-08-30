#include "curve_ribbon_component.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/coro/schedulers/state_machine.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/passes/dom_tree.h>
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

void expect_runtime_subdivision_control_flow(
    const std::shared_ptr<const CurveRibbonComponent> &ribbon) {
  Kernel1D shape =
      [ribbon](BufferUInt subdivision, BufferFloat4 control_points,
               BufferFloat4 output) noexcept {
        const auto ray = make_ray(
            make_float3(0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            0.0f,
            10.0f);
        const CurveControlPoints curve{.before = control_points.read(0u),
                                       .begin = control_points.read(1u),
                                       .end = control_points.read(2u),
                                       .after = control_points.read(3u)};
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
  std::size_t breaks = 0u;
  std::size_t in_loop_scalar_divisions = 0u;
  std::size_t guarded_in_loop_scalar_divisions = 0u;
  for (auto *function : module->function_list()) {
    if (auto *definition = function->definition()) {
      std::vector<luisa::compute::xir::LoopInst *> loop_instructions;
      std::vector<luisa::compute::xir::IfInst *> if_instructions;
      std::vector<luisa::compute::xir::ArithmeticInst *> scalar_divisions;
      definition->traverse_instructions([&](luisa::compute::xir::Instruction
                                                *instruction) noexcept {
        if (instruction->isa<luisa::compute::xir::LoopInst>()) {
          loop_instructions.emplace_back(
              static_cast<luisa::compute::xir::LoopInst *>(instruction));
        } else if (instruction->isa<luisa::compute::xir::SimpleLoopInst>()) {
          ++loops;
        } else if (instruction->isa<luisa::compute::xir::IfInst>()) {
          if_instructions.emplace_back(
              static_cast<luisa::compute::xir::IfInst *>(instruction));
        } else if (instruction->isa<luisa::compute::xir::ArithmeticInst>()) {
          auto *arithmetic =
              static_cast<luisa::compute::xir::ArithmeticInst *>(instruction);
          if (arithmetic->op() ==
                  luisa::compute::xir::ArithmeticOp::BINARY_DIV &&
              arithmetic->type() == Type::of<float>()) {
            scalar_divisions.emplace_back(arithmetic);
          }
        }
        breaks += instruction->isa<luisa::compute::xir::BreakInst>() ? 1u : 0u;
      });
      loops += loop_instructions.size();
      const auto dominance = luisa::compute::xir::compute_dom_tree(function);
      for (auto *division : scalar_divisions) {
        for (auto *loop : loop_instructions) {
          if (!dominance.dominates(loop->body_block(),
                                   division->parent_block())) {
            continue;
          }
          ++in_loop_scalar_divisions;
          auto guarded = false;
          for (auto *if_instruction : if_instructions) {
            const auto conditional_is_in_loop = dominance.dominates(
                loop->body_block(), if_instruction->parent_block());
            const auto true_path_dominates_division = dominance.dominates(
                if_instruction->true_block(), division->parent_block());
            if (conditional_is_in_loop && true_path_dominates_division) {
              guarded = true;
              break;
            }
          }
          guarded_in_loop_scalar_divisions += guarded ? 1u : 0u;
        }
      }
    }
  }
  expect(loops == 1u,
         "runtime curve subdivision did not lower to exactly one device loop");
  expect(breaks == 1u,
         "a committed ribbon hit does not terminate subdivision immediately");
  expect(in_loop_scalar_divisions == 1u,
         "unexpected exact ribbon division shape inside subdivision loop");
  expect(
      guarded_in_loop_scalar_divisions == 1u,
      "exact ribbon intersection is not control-dependent on coarse culling");
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    try {
        const auto ribbon = make_curve_ribbon_component();
        expect_runtime_subdivision_control_flow(ribbon);

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
        constexpr std::array boundary_control_points{
            luisa::float4{0.2665650546550751f, -4.7415385246276855f,
                          0.8878118991851807f, 8.992105722427368e-05f},
            luisa::float4{0.2706359326839447f, -4.740843772888184f,
                          0.8848966360092163f, 8.417015487793833e-05f},
            luisa::float4{0.2736737430095673f, -4.7385821342468262f,
                          0.8816856741905212f, 7.852141425246373e-05f},
            luisa::float4{0.2769835293292999f, -4.7353634834289551f,
                          0.8777113556861877f, 7.159107917686924e-05f}};
        auto identity_bounds_buffer =
            device.create_buffer<AABB>(identity_bounds.size());
        auto transformed_bounds_buffer =
            device.create_buffer<AABB>(transformed_bounds.size());
        auto boundary_control_point_buffer =
            device.create_buffer<luisa::float4>(boundary_control_points.size());
        auto identity_curves = device.create_procedural_primitive(
            identity_bounds_buffer);
        auto transformed_curves = device.create_procedural_primitive(
            transformed_bounds_buffer);
        auto subdivision = device.create_buffer<std::uint32_t>(1u);
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

        auto output = device.create_buffer<luisa::float4>(6u);
        Kernel1D trace = [ribbon](
                             BufferFloat4 records,
                             BufferUInt runtime_subdivision,
                             BufferFloat4 runtime_control_points,
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

                // Reduced from a unique-valid-interval ribbon hit in the
                // Blender 5.2 benchmark scene. Cycles selects subdivision 5
                // of 16. Returning a later loop-carried value changes u/v
                // structurally even though the distance is nearly unchanged.
                const CurveControlPoints interval_boundary{
                    .before = runtime_control_points.read(0u),
                    .begin = runtime_control_points.read(1u),
                    .end = runtime_control_points.read(2u),
                    .after = runtime_control_points.read(3u)};
                const auto boundary_hit = ribbon->intersect(
                    make_ray(
                        make_float3(
                            3.0629506111145020f, -4.7604250907897949f,
                            0.1933582425117493f),
                        make_float3(
                            -0.9707216024398804f, 0.0070224693045020f,
                            0.2401046752929688f),
                        0.0f, 101.48377227783203f),
                    interval_boundary, runtime_subdivision.read(0u));
                records.write(
                    5u,
                    make_float4(
                        boundary_hit.distance, boundary_hit.u, boundary_hit.v,
                        select(0.0f, 1.0f, boundary_hit.valid)));
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
        std::array<luisa::float4, 6u> results{};
        constexpr std::array subdivision_value{4u};
        stream
            << subdivision.copy_from(luisa::span{subdivision_value})
            << boundary_control_point_buffer.copy_from(
                   luisa::span{boundary_control_points})
            << identity_bounds_buffer.copy_from(
                   luisa::span{identity_bounds})
            << transformed_bounds_buffer.copy_from(
                   luisa::span{transformed_bounds})
            << identity_curves.build()
            << transformed_curves.build()
            << accel.build()
            << shader(output, subdivision, boundary_control_point_buffer,
                      accel)
                   .dispatch(4u)
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
        expect(results[5u].w == 1.0f,
               "unique-valid-interval ribbon did not commit");
        expect_near(results[5u].x, 2.8753504753112793f, 2.0e-6f,
                    "unique-valid-interval ribbon distance");
        expect_near(results[5u].y, 0.3611741960048676f, 5.0e-5f,
                    "unique-valid-interval ribbon u");
        expect_near(results[5u].z, 0.8586872816085815f, 5.0e-4f,
                    "unique-valid-interval ribbon v");

        auto coroutine_output =
            device.create_buffer<luisa::float4>(1u);
        auto coroutine = Coroutine<void(
            Buffer<luisa::float4>, Buffer<std::uint32_t>,
            Buffer<luisa::float4>)>{
            [ribbon](BufferFloat4 records,
                     BufferUInt runtime_subdivision,
                     BufferFloat4 runtime_control_points) noexcept {
                $suspend("shade_surface");
                const CurveControlPoints curve{
                    .before = runtime_control_points.read(0u),
                    .begin = runtime_control_points.read(1u),
                    .end = runtime_control_points.read(2u),
                    .after = runtime_control_points.read(3u)};
                const auto hit = ribbon->intersect(
                    make_ray(
                        make_float3(
                            3.0629506111145020f, -4.7604250907897949f,
                            0.1933582425117493f),
                        make_float3(
                            -0.9707216024398804f, 0.0070224693045020f,
                            0.2401046752929688f),
                        0.0f, 101.48377227783203f),
                    curve, runtime_subdivision.read(0u));
                records.write(
                    0u,
                    make_float4(
                        hit.distance, hit.u, hit.v,
                        select(0.0f, 1.0f, hit.valid)));
            }};
        luisa::compute::coro::StateMachineCoroScheduler<
            Buffer<luisa::float4>, Buffer<std::uint32_t>,
            Buffer<luisa::float4>> coroutine_scheduler{device, coroutine};
        coroutine_scheduler(
            coroutine_output, subdivision, boundary_control_point_buffer)
            .dispatch(1u)(stream);
        std::array<luisa::float4, 1u> coroutine_result{};
        stream << coroutine_output.copy_to(luisa::span{coroutine_result})
               << synchronize();
        expect(coroutine_result[0u].w == 1.0f,
               "coroutine unique-valid-interval ribbon did not commit");
        expect_near(coroutine_result[0u].x, 2.8753504753112793f,
                    5.0e-6f, "coroutine ribbon distance");
        expect_near(coroutine_result[0u].y, 0.3611741960048676f,
                    5.0e-5f, "coroutine ribbon u");
        expect_near(coroutine_result[0u].z, 0.8586872816085815f,
                    5.0e-4f, "coroutine ribbon v");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "curve ribbon test (" << backend
                  << ") failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

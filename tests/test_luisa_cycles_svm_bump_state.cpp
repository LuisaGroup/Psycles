#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
using psycles::test_support::DefaultCyclesSvmKernelGlobals;

constexpr auto scenario_count = 8u;
constexpr auto scenario_output_count = 9u;
constexpr auto displacement_scenario_count = 4u;

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 4.0e-5f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool require_float4(luisa::float4 actual,
                                  luisa::float4 expected,
                                  std::string_view label,
                                  float tolerance = 4.0e-5f) {
  if (near(actual.x, expected.x, tolerance) &&
      near(actual.y, expected.y, tolerance) &&
      near(actual.z, expected.z, tolerance) &&
      near(actual.w, expected.w, tolerance)) {
    return true;
  }
  std::cerr << label << " mismatch: (" << actual.x << ", " << actual.y
            << ", " << actual.z << ", " << actual.w << ") != ("
            << expected.x << ", " << expected.y << ", " << expected.z
            << ", " << expected.w << ")\n";
  return false;
}

class BumpStateKernelGlobals final : public DefaultCyclesSvmKernelGlobals {
public:
  [[nodiscard]] Var<AttributeMap>
  attribute_map(Expr<std::uint32_t> offset) const noexcept override {
    Var<AttributeMap> entry;
    entry.id = static_cast<luisa::ulong>(ATTR_STD_NONE);
    entry.offset = 0;
    entry.element = static_cast<std::uint16_t>(0u);
    entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT3);
    entry.pad = static_cast<std::uint8_t>(0u);
    $if(offset == 0u) {
      entry.id = static_cast<luisa::ulong>(ATTR_STD_POSITION_UNDISPLACED);
      entry.offset = 0;
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
    }
    $elif(offset == static_cast<std::uint32_t>(ATTR_PRIM_TYPES)) {
      entry.id = static_cast<luisa::ulong>(ATTR_STD_NORMAL_UNDISPLACED);
      entry.offset = 3;
      entry.element = static_cast<std::uint16_t>(ATTR_ELEMENT_VERTEX);
    };
    return entry;
  }

  [[nodiscard]] Var<packed_float3>
  attribute_float3(Expr<std::int32_t> offset) const noexcept override {
    Float3 source = make_float3(0.0f);
    $if(offset == 0) { source = make_float3(1.0f, 2.0f, 3.0f); }
    $elif(offset == 1) { source = make_float3(3.0f, 2.0f, 3.0f); }
    $elif(offset == 2) { source = make_float3(1.0f, 5.0f, 3.0f); }
    $elif(offset == 3) { source = make_float3(0.0f, 0.0f, 2.0f); }
    $elif(offset == 4) { source = make_float3(0.0f, 0.0f, 4.0f); }
    $elif(offset == 5) { source = make_float3(0.0f, 0.0f, 6.0f); };
    Var<packed_float3> result;
    result.x = source.x;
    result.y = source.y;
    result.z = source.z;
    return result;
  }
};

[[nodiscard]] Float4x4 static_object_to_world() noexcept {
  return make_float4x4(make_float4(2.0f, 0.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 3.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 0.0f, 4.0f, 0.0f),
                       make_float4(10.0f, 20.0f, 30.0f, 1.0f));
}

[[nodiscard]] Float4x4 static_world_to_object() noexcept {
  return make_float4x4(make_float4(0.5f, 0.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 1.0f / 3.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 0.0f, 0.25f, 0.0f),
                       make_float4(-5.0f, -20.0f / 3.0f, -7.5f, 1.0f));
}

[[nodiscard]] Float4x4 motion_object_to_world() noexcept {
  return make_float4x4(make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                       make_float4(100.0f, 200.0f, 300.0f, 1.0f));
}

[[nodiscard]] Float4x4 motion_world_to_object() noexcept {
  return make_float4x4(make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 1.0f, 0.0f, 0.0f),
                       make_float4(0.0f, 0.0f, 1.0f, 0.0f),
                       make_float4(-100.0f, -200.0f, -300.0f, 1.0f));
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(UInt shader, UInt flags, UInt object_flags) noexcept {
  return device_svm::ShaderData{
      make_float3(9.0f, -4.0f, 2.0f),
      make_float3(0.25f, 0.5f, 0.75f),
      make_float3(0.0f, 0.0f, 1.0f),
      make_float3(0.0f, 0.0f, -1.0f),
      device_svm::primitive_triangle,
      shader,
      flags,
      object_flags,
      0u,
      0.25f,
      0.5f,
      0u,
      0.5f,
      1.0f,
      0.75f,
      0.0f,
      0.1f,
      -0.2f,
      0.3f,
      0.4f,
      make_float3(2.0f, 0.0f, 0.0f),
      make_float3(0.0f, 3.0f, 0.0f),
      motion_object_to_world(),
      motion_world_to_object()};
}

[[nodiscard]] bool test_derivative_constants(Device &device, Stream &stream) {
  static constexpr std::array words{
      std::bit_cast<std::uint32_t>(1.25f), 0u, 10u,
      std::bit_cast<std::uint32_t>(0.6f),
      std::bit_cast<std::uint32_t>(-0.8f),
      std::bit_cast<std::uint32_t>(1.5f)};
  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
          [](BufferUInt payload, BufferFloat4 output) noexcept {
            device_svm::detail::Stack stack;
            UInt scalar_offset = 0u;
            device_svm::detail::Cursor scalar_cursor{payload, scalar_offset};
            device_svm::detail::node_value_f(scalar_cursor, stack, true);
            UInt vector_offset = 2u;
            device_svm::detail::Cursor vector_cursor{payload, vector_offset};
            device_svm::detail::node_value_v(vector_cursor, stack, true);
            output.write(
                0u, make_float4(
                        device_svm::detail::stack_load_float(stack, 0u),
                        device_svm::detail::stack_load_float(stack, 1u),
                        device_svm::detail::stack_load_float(stack, 2u), 0.0f));
            output.write(
                1u, make_float4(device_svm::detail::stack_load_float3(
                                     stack, 10u),
                                 0.0f));
            output.write(
                2u, make_float4(device_svm::detail::stack_load_float3(
                                     stack, 13u),
                                 0.0f));
            output.write(
                3u, make_float4(device_svm::detail::stack_load_float3(
                                     stack, 16u),
                                 0.0f));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto payload = device.create_buffer<std::uint32_t>(words.size());
  auto output = device.create_buffer<luisa::float4>(4u);
  std::array<luisa::float4, 4u> actual{};
  stream << payload.copy_from(luisa::span{words})
         << shader(payload, output).dispatch(1u)
         << output.copy_to(luisa::span{actual}) << synchronize();
  return require_float4(actual[0u], {1.25f, 0.0f, 0.0f, 0.0f},
                        "scalar derivative constant") &&
         require_float4(actual[1u], {0.6f, -0.8f, 1.5f, 0.0f},
                        "vector derivative constant value") &&
         require_float4(actual[2u], {0.0f, 0.0f, 0.0f, 0.0f},
                        "vector derivative constant dx") &&
         require_float4(actual[3u], {0.0f, 0.0f, 0.0f, 0.0f},
                        "vector derivative constant dy");
}

[[nodiscard]] bool test_displacement_handlers(Device &device,
                                              Stream &stream) {
  static constexpr std::array words{
      static_cast<std::uint32_t>(NODE_NORMAL_MAP_OBJECT),
      std::bit_cast<std::uint32_t>(2.0f),
      std::bit_cast<std::uint32_t>(0.5f),
      std::bit_cast<std::uint32_t>(0.25f), 10u | (20u << 8u), 20u,
      static_cast<std::uint32_t>(NODE_NORMAL_MAP_WORLD),
      std::bit_cast<std::uint32_t>(2.0f),
      std::bit_cast<std::uint32_t>(0.5f),
      std::bit_cast<std::uint32_t>(0.25f), 10u | (20u << 8u), 20u};
  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
          [](BufferUInt payload, BufferFloat4 output) noexcept {
            const UInt scenario = dispatch_x();
            const Bool world = scenario == 1u;
            const Bool motion = scenario == 3u;
            UInt object_flags = 0u;
            $if(motion) {
              object_flags = device_svm::shader_data_object_motion;
            };
            auto shader_data = make_shader_data(0u, 0u, object_flags);
            const auto identity = make_float4x4(1.0f);
            const device_svm::TransformState transforms{
                identity, identity, static_object_to_world(),
                static_world_to_object()};
            device_svm::detail::Stack stack;
            device_svm::detail::stack_store_float3(
                stack, 10u,
                select(make_float3(1.0f, 0.0f, 0.0f),
                       make_float3(0.0f, 1.0f, 0.0f), world));
            UInt offset = select(0u, 6u, world);
            device_svm::detail::Cursor cursor{payload, offset};
            $if(scenario == 2u) {
              device_svm::detail::node_displacement(
                  cursor, stack, transforms, shader_data, false, true);
              device_svm::detail::node_set_displacement(
                  cursor, stack, shader_data, false);
            }
            $else {
              device_svm::detail::node_displacement(
                  cursor, stack, transforms, shader_data, true, true);
              device_svm::detail::node_set_displacement(
                  cursor, stack, shader_data, true);
            };
            const Float3 displacement =
                device_svm::detail::stack_load_float3(stack, 20u);
            const UInt base = scenario * 2u;
            output.write(base, make_float4(displacement, 0.0f));
            output.write(base + 1u, make_float4(shader_data.P, 0.0f));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto payload = device.create_buffer<std::uint32_t>(words.size());
  auto output =
      device.create_buffer<luisa::float4>(displacement_scenario_count * 2u);
  std::array<luisa::float4, displacement_scenario_count * 2u> actual{};
  stream << payload.copy_from(luisa::span{words})
         << shader(payload, output).dispatch(displacement_scenario_count)
         << output.copy_to(luisa::span{actual}) << synchronize();

  static constexpr std::array expected{
      luisa::float4{0.75f, 0.0f, 0.0f, 0.0f},
      luisa::float4{9.75f, -4.0f, 2.0f, 0.0f},
      luisa::float4{0.0f, 0.375f, 0.0f, 0.0f},
      luisa::float4{9.0f, -3.625f, 2.0f, 0.0f},
      luisa::float4{0.0f, 0.0f, 0.0f, 0.0f},
      luisa::float4{9.0f, -4.0f, 2.0f, 0.0f},
      luisa::float4{0.375f, 0.0f, 0.0f, 0.0f},
      luisa::float4{9.375f, -4.0f, 2.0f, 0.0f}};
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    if (!require_float4(actual[index], expected[index],
                        "scalar displacement handler")) {
      std::cerr << "displacement result " << index << " failed\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_bump_state_handlers(Device &device, Stream &stream) {
  static constexpr std::array words{0u, 20u | (23u << 8u), 0u};
  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
          [](BufferUInt payload, BufferFloat4 output) noexcept {
            const UInt scenario = dispatch_x();
            UInt shader = 0u;
            UInt flags = 0u;
            UInt object_flags = 0u;
            $if(scenario == 1u) { shader = device_svm::shader_smooth_normal; };
            $if(scenario == 2u) {
              object_flags = static_cast<std::uint32_t>(
                  SD_OBJECT_NEGATIVE_SCALE | SD_OBJECT_TRANSFORM_APPLIED);
            };
            $if(scenario == 3u) {
              object_flags =
                  static_cast<std::uint32_t>(SD_OBJECT_NEGATIVE_SCALE);
            };
            $if(scenario == 4u) {
              flags = device_svm::shader_data_backfacing;
            };
            $if(scenario == 5u) {
              flags = device_svm::shader_data_backfacing;
              object_flags = static_cast<std::uint32_t>(
                  SD_OBJECT_NEGATIVE_SCALE | SD_OBJECT_TRANSFORM_APPLIED);
            };
            $if(scenario == 6u) {
              shader = device_svm::shader_smooth_normal;
              flags = device_svm::shader_data_backfacing;
            };
            $if(scenario == 7u) {
              object_flags = device_svm::shader_data_object_motion;
            };

            const auto identity = make_float4x4(1.0f);
            const device_svm::TransformState transforms{
                identity, identity, static_object_to_world(),
                static_world_to_object()};
            const BumpStateKernelGlobals kernel_globals;
            auto shader_data = make_shader_data(shader, flags, object_flags);
            device_svm::detail::Stack stack;
            UInt offset = 0u;
            device_svm::detail::Cursor cursor{payload, offset};
            device_svm::detail::node_enter_bump_eval(
                cursor, stack, kernel_globals, transforms, shader_data, true);

            const Float3 entered_position = shader_data.P;
            const Float entered_differential = shader_data.dP;
            const Float3 entered_normal = shader_data.N;
            const Float3 saved_position =
                device_svm::detail::stack_load_float3(stack, 0u);
            const Float saved_differential =
                device_svm::detail::stack_load_float(stack, 3u);
            const Float3 position_dx =
                device_svm::detail::stack_load_float3(stack, 4u);
            const Float3 position_dy =
                device_svm::detail::stack_load_float3(stack, 7u);

            device_svm::detail::stack_store_float3(
                stack, 20u, make_float3(0.6f, 0.8f, 0.0f));
            device_svm::detail::node_set_normal(cursor, stack, shader_data);
            const Float3 set_normal = shader_data.N;
            const Float3 normal_output =
                device_svm::detail::stack_load_float3(stack, 23u);
            device_svm::detail::node_leave_bump_eval(cursor, stack,
                                                     shader_data);

            const UInt base = scenario * scenario_output_count;
            output.write(base + 0u,
                         make_float4(entered_position, entered_differential));
            output.write(base + 1u, make_float4(entered_normal, 0.0f));
            output.write(base + 2u,
                         make_float4(saved_position, saved_differential));
            output.write(base + 3u, make_float4(position_dx, 0.0f));
            output.write(base + 4u, make_float4(position_dy, 0.0f));
            output.write(base + 5u, make_float4(set_normal, 0.0f));
            output.write(base + 6u, make_float4(normal_output, 0.0f));
            output.write(base + 7u,
                         make_float4(shader_data.P, shader_data.dP));
            output.write(base + 8u, make_float4(shader_data.N, 0.0f));
          }};

  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto payload = device.create_buffer<std::uint32_t>(words.size());
  auto output = device.create_buffer<luisa::float4>(
      scenario_count * scenario_output_count);
  std::array<luisa::float4, scenario_count * scenario_output_count> actual{};
  stream << payload.copy_from(luisa::span{words})
         << shader(payload, output).dispatch(scenario_count)
         << output.copy_to(luisa::span{actual}) << synchronize();

  const auto static_differential =
      0.5f * (std::sqrt(0.4f * 0.4f + 2.7f * 2.7f) +
              std::sqrt(0.8f * 0.8f + 3.6f * 3.6f));
  const auto motion_differential =
      0.5f * (std::sqrt(0.2f * 0.2f + 0.9f * 0.9f) +
              std::sqrt(0.4f * 0.4f + 1.2f * 1.2f));
  static constexpr std::array normal_sign{1.0f, 1.0f, -1.0f, 1.0f,
                                           -1.0f, 1.0f, -1.0f, 1.0f};
  for (auto scenario = std::size_t{}; scenario < scenario_count; ++scenario) {
    const auto base = scenario * scenario_output_count;
    const auto motion = scenario == 7u;
    const auto entered_position = motion ? luisa::float4{101.5f, 203.5f, 303.0f,
                                                         motion_differential}
                                         : luisa::float4{13.0f, 30.5f, 42.0f,
                                                         static_differential};
    const auto position_dx = motion ? luisa::float4{0.2f, 0.9f, 0.0f, 0.0f}
                                    : luisa::float4{0.4f, 2.7f, 0.0f, 0.0f};
    const auto position_dy = motion ? luisa::float4{-0.4f, 1.2f, 0.0f, 0.0f}
                                    : luisa::float4{-0.8f, 3.6f, 0.0f, 0.0f};
    if (!require_float4(actual[base + 0u], entered_position,
                        "ENTER undisplaced position") ||
        !require_float4(actual[base + 1u],
                        {0.0f, 0.0f, normal_sign[scenario], 0.0f},
                        "ENTER undisplaced normal") ||
        !require_float4(actual[base + 2u], {9.0f, -4.0f, 2.0f, 0.75f},
                        "ENTER saved state") ||
        !require_float4(actual[base + 3u], position_dx,
                        "ENTER position dx") ||
        !require_float4(actual[base + 4u], position_dy,
                        "ENTER position dy") ||
        !require_float4(actual[base + 5u], {0.6f, 0.8f, 0.0f, 0.0f},
                        "SET_NORMAL ShaderData N") ||
        !require_float4(actual[base + 6u], {0.6f, 0.8f, 0.0f, 0.0f},
                        "SET_NORMAL stack output") ||
        !require_float4(actual[base + 7u], {9.0f, -4.0f, 2.0f, 0.75f},
                        "LEAVE restored state") ||
        !require_float4(actual[base + 8u], {0.6f, 0.8f, 0.0f, 0.0f},
                        "LEAVE retained normal")) {
      std::cerr << "scenario " << scenario << " failed\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::array<bool, NODE_NUM> bump_state_node_types() {
  std::array<bool, NODE_NUM> result{};
  result[NODE_END] = true;
  result[NODE_SHADER_JUMP] = true;
  result[NODE_VALUE_F_DERIVATIVE] = true;
  result[NODE_VALUE_V_DERIVATIVE] = true;
  result[NODE_ENTER_BUMP_EVAL] = true;
  result[NODE_CLOSURE_SET_NORMAL] = true;
  result[NODE_LEAVE_BUMP_EVAL] = true;
  return result;
}

[[nodiscard]] bool test_eval_nodes_control_flow(Device &device,
                                                Stream &stream) {
  static constexpr std::array words{
      static_cast<std::uint32_t>(NODE_SHADER_JUMP), 4u, 18u, 18u,
      static_cast<std::uint32_t>(NODE_VALUE_F_DERIVATIVE),
      std::bit_cast<std::uint32_t>(1.25f), 30u,
      static_cast<std::uint32_t>(NODE_VALUE_V_DERIVATIVE), 20u,
      std::bit_cast<std::uint32_t>(0.6f),
      std::bit_cast<std::uint32_t>(0.8f),
      std::bit_cast<std::uint32_t>(0.0f),
      static_cast<std::uint32_t>(NODE_ENTER_BUMP_EVAL), 0u,
      static_cast<std::uint32_t>(NODE_CLOSURE_SET_NORMAL), 20u | (40u << 8u),
      static_cast<std::uint32_t>(NODE_LEAVE_BUMP_EVAL), 0u,
      static_cast<std::uint32_t>(NODE_END)};
  const auto node_types = bump_state_node_types();
  const auto kernel =
      Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
               Buffer<luisa::uint4>>{
          [node_types](BufferUInt bytecode, BufferFloat4 floating,
                       BufferUInt4 integer) noexcept {
            const auto identity = make_float4x4(1.0f);
            const device_svm::TransformState transforms{
                identity, identity, static_object_to_world(),
                static_world_to_object()};
            const BumpStateKernelGlobals kernel_globals;
            auto shader_data = make_shader_data(0u, 0u, 0u);
            const device_svm::PathState path_state{
                device_svm::path_ray_visibility_camera, 0u};
            device_svm::EvaluationResult result;
            device_svm::eval_nodes(
                kernel_globals, bytecode, SHADER_TYPE_SURFACE,
                device_svm::kernel_feature_object_motion,
                device_svm::kernel_feature_node_bump |
                    device_svm::kernel_feature_node_bump_state,
                node_types, transforms, shader_data, path_state, result);
            floating.write(0u, make_float4(shader_data.P, shader_data.dP));
            floating.write(1u, make_float4(shader_data.N, 0.0f));
            integer.write(0u, make_uint4(result.status, result.final_offset,
                                         shader_data.flag, 0u));
          }};
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto bytecode = device.create_buffer<std::uint32_t>(words.size());
  auto floating = device.create_buffer<luisa::float4>(2u);
  auto integer = device.create_buffer<luisa::uint4>(1u);
  std::array<luisa::float4, 2u> actual_floating{};
  std::array<luisa::uint4, 1u> actual_integer{};
  stream << bytecode.copy_from(luisa::span{words})
         << shader(bytecode, floating, integer).dispatch(1u)
         << floating.copy_to(luisa::span{actual_floating})
         << integer.copy_to(luisa::span{actual_integer}) << synchronize();
  return require_float4(actual_floating[0u], {9.0f, -4.0f, 2.0f, 0.75f},
                        "eval_nodes restored state") &&
         require_float4(actual_floating[1u], {0.6f, 0.8f, 0.0f, 0.0f},
                        "eval_nodes retained normal") &&
         actual_integer[0u].x ==
             static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
         actual_integer[0u].y == words.size();
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  if (!test_derivative_constants(device, stream) ||
      !test_displacement_handlers(device, stream) ||
      !test_bump_state_handlers(device, stream) ||
      !test_eval_nodes_control_flow(device, stream)) {
    return EXIT_FAILURE;
  }
  std::cout << "Luisa Cycles SVM bump-state tests passed on " << backend
            << '\n';
  return EXIT_SUCCESS;
}

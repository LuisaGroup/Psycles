#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto fresnel_case_count = std::uint32_t{5u};
constexpr auto layer_case_count = std::uint32_t{8u};
constexpr auto normal_offset = std::uint32_t{0u};
constexpr auto input_offset = std::uint32_t{3u};
constexpr auto output_offset = std::uint32_t{4u};

[[nodiscard]] constexpr std::uint32_t
pack_normal_output(std::uint32_t normal, std::uint32_t output) noexcept {
  return normal | (output << 8u);
}

[[nodiscard]] constexpr std::uint32_t float_bits(float value) noexcept {
  return std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] auto fresnel_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < fresnel_case_count) {
          svm_detail::Stack stack;
          svm_detail::stack_store_float(stack, input_offset, 1.45f);
          Float3 explicit_normal = make_float3(0.0f, 0.0f, 1.0f);
          $switch(index) {
            $case(2u) {
              explicit_normal = make_float3(0.9682458f, 0.0f, 0.25f);
            };
            $case(3u) {
              explicit_normal =
                  make_float3(0.07807466f, 0.10947394f, 0.8824053f);
            };
            $case(4u) { explicit_normal = make_float3(0.0f); };
            $default{};
          };
          svm_detail::stack_store_float3(stack, normal_offset, explicit_normal);

          UInt flags = 0u;
          $if(index == 2u) { flags = device_svm::shader_data_backfacing; };
          const auto identity = make_float4x4(1.0f);
          device_svm::ShaderData shader_data{make_float3(0.0f),
                                             make_float3(0.0f, 0.0f, 1.0f),
                                             make_float3(0.0f, 0.0f, 1.0f),
                                             make_float3(0.0f, 0.0f, 1.0f),
                                             device_svm::primitive_triangle,
                                             0u,
                                             flags,
                                             0u,
                                             0u,
                                             0.0f,
                                             0.0f,
                                             0u,
                                             0.0f,
                                             1.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             make_float3(1.0f, 0.0f, 0.0f),
                                             make_float3(0.0f, 1.0f, 0.0f),
                                             identity,
                                             identity};

          UInt cursor_offset = index * 2u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_fresnel(cursor, stack, shader_data);
          output.write(index,
                       svm_detail::stack_load_float(stack, output_offset));
          cursors.write(index, cursor_offset - begin);
        };
      }};
}

[[nodiscard]] auto layer_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < layer_case_count) {
          svm_detail::Stack stack;
          svm_detail::stack_store_float(stack, input_offset, 0.75f);
          Float3 explicit_normal = make_float3(0.0f, 0.0f, 1.0f);
          $switch(index) {
            $case(2u) {
              explicit_normal = make_float3(0.4358899f, 0.0f, 0.9f);
            };
            $case(3u) {
              explicit_normal = make_float3(0.4358899f, 0.0f, 0.9f);
            };
            $case(4u) {
              explicit_normal = make_float3(0.9682458f, 0.0f, 0.25f);
            };
            $case(5u) {
              explicit_normal = make_float3(0.4358899f, 0.0f, 0.9f);
            };
            $case(6u) {
              explicit_normal = make_float3(0.4358899f, 0.0f, 0.9f);
            };
            $case(7u) {
              explicit_normal = make_float3(0.9999995f, 0.0f, 0.001f);
            };
            $default{};
          };
          svm_detail::stack_store_float3(stack, normal_offset, explicit_normal);

          UInt flags = 0u;
          $if((index == 5u) | (index == 6u)) {
            flags = device_svm::shader_data_backfacing;
          };
          const auto identity = make_float4x4(1.0f);
          device_svm::ShaderData shader_data{make_float3(0.0f),
                                             make_float3(0.0f, 0.0f, 1.0f),
                                             make_float3(0.0f, 0.0f, 1.0f),
                                             make_float3(0.0f, 0.0f, 1.0f),
                                             device_svm::primitive_triangle,
                                             0u,
                                             flags,
                                             0u,
                                             0u,
                                             0.0f,
                                             0.0f,
                                             0u,
                                             0.0f,
                                             1.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             make_float3(1.0f, 0.0f, 0.0f),
                                             make_float3(0.0f, 1.0f, 0.0f),
                                             identity,
                                             identity};

          UInt cursor_offset = fresnel_case_count * 2u + index * 3u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_layer_weight(cursor, stack, shader_data);
          output.write(index,
                       svm_detail::stack_load_float(stack, output_offset));
          cursors.write(index, cursor_offset - begin);
        };
      }};
}

struct ModuleShape {
  std::size_t instructions{};
  std::size_t loops{};
  std::size_t callable_definitions{};
};

template <typename... Args>
[[nodiscard]] ModuleShape module_shape(const Kernel1D<Args...> &kernel) {
  auto module = luisa::compute::xir::ast_to_xir_translate(
      kernel.function()->function(), {});
  ModuleShape result;
  for (auto *function : module->function_list()) {
    result.callable_definitions +=
        function->derived_function_tag() ==
                luisa::compute::xir::DerivedFunctionTag::CALLABLE
            ? 1u
            : 0u;
    if (auto *definition = function->definition()) {
      definition->traverse_instructions(
          [&](const luisa::compute::xir::Instruction *instruction) noexcept {
            ++result.instructions;
            result.loops +=
                instruction->isa<luisa::compute::xir::LoopInst>() ? 1u : 0u;
          });
    }
  }
  return result;
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 5.0e-6f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool test_direct_handlers(Device &device, Stream &stream,
                                        std::string_view backend) {
  static constexpr std::array words{
      float_bits(1.5f),
      pack_normal_output(SVM_STACK_INVALID, output_offset),
      float_bits(0.5f),
      pack_normal_output(normal_offset, output_offset),
      float_bits(-2.0f),
      pack_normal_output(normal_offset, output_offset),
      SVM_INPUT_STACK_OFFSET_MASK | input_offset,
      pack_normal_output(normal_offset, output_offset),
      float_bits(1.33f),
      pack_normal_output(normal_offset, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FRESNEL),
      float_bits(-2.0f),
      pack_normal_output(SVM_STACK_INVALID, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FACING),
      float_bits(-2.0f),
      pack_normal_output(SVM_STACK_INVALID, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FRESNEL),
      float_bits(0.25f),
      pack_normal_output(normal_offset, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FACING),
      float_bits(0.25f),
      pack_normal_output(normal_offset, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FACING),
      float_bits(0.5f),
      pack_normal_output(normal_offset, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FRESNEL),
      float_bits(0.25f),
      pack_normal_output(normal_offset, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FACING),
      float_bits(0.25f),
      pack_normal_output(normal_offset, output_offset),
      static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FACING),
      SVM_INPUT_STACK_OFFSET_MASK | input_offset,
      pack_normal_output(normal_offset, output_offset)};
  static constexpr std::array fresnel_expected{
      0.0400000028f, 0.111111119f, 0.999915004f, 0.0347833373f, 1.0f};
  static constexpr std::array layer_expected{
      0.25f, 0.0f,          0.0209565088f, 0.0513167381f,
      0.75f, 0.0227367077f, 0.0513167381f, 0.999998987f};

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto fresnel_output = device.create_buffer<float>(fresnel_case_count);
  auto layer_output = device.create_buffer<float>(layer_case_count);
  auto fresnel_cursors =
      device.create_buffer<std::uint32_t>(fresnel_case_count);
  auto layer_cursors = device.create_buffer<std::uint32_t>(layer_case_count);
  auto fresnel_shader =
      device.compile(fresnel_kernel(), ShaderOption{.enable_cache = false,
                                                    .enable_fast_math = false});
  auto layer_shader =
      device.compile(layer_kernel(), ShaderOption{.enable_cache = false,
                                                  .enable_fast_math = false});
  std::array<float, fresnel_case_count> actual_fresnel{};
  std::array<float, layer_case_count> actual_layer{};
  std::array<std::uint32_t, fresnel_case_count> actual_fresnel_cursors{};
  std::array<std::uint32_t, layer_case_count> actual_layer_cursors{};
  stream << word_buffer.copy_from(words.data())
         << fresnel_shader(word_buffer, fresnel_output, fresnel_cursors)
                .dispatch(fresnel_case_count)
         << layer_shader(word_buffer, layer_output, layer_cursors)
                .dispatch(layer_case_count)
         << fresnel_output.copy_to(actual_fresnel.data())
         << layer_output.copy_to(actual_layer.data())
         << fresnel_cursors.copy_to(actual_fresnel_cursors.data())
         << layer_cursors.copy_to(actual_layer_cursors.data()) << synchronize();

  const auto fresnel_valid =
      std::equal(actual_fresnel.begin(), actual_fresnel.end(),
                 fresnel_expected.begin(),
                 [](float actual, float expected) noexcept {
                   return near(actual, expected);
                 }) &&
      std::all_of(actual_fresnel_cursors.begin(), actual_fresnel_cursors.end(),
                  [](auto cursor) noexcept { return cursor == 2u; });
  const auto layer_valid =
      std::equal(actual_layer.begin(), actual_layer.end(),
                 layer_expected.begin(),
                 [](float actual, float expected) noexcept {
                   return near(actual, expected);
                 }) &&
      std::all_of(actual_layer_cursors.begin(), actual_layer_cursors.end(),
                  [](auto cursor) noexcept { return cursor == 3u; });
  if (!fresnel_valid || !layer_valid) {
    std::cerr << "Cycles Fresnel-family direct handler mismatch on " << backend
              << '\n';
    for (auto index = std::size_t{}; index < actual_fresnel.size(); ++index) {
      std::cerr << "fresnel " << index << ": " << actual_fresnel[index]
                << " expected " << fresnel_expected[index] << '\n';
    }
    for (auto index = std::size_t{}; index < actual_layer.size(); ++index) {
      std::cerr << "layer " << index << ": " << actual_layer[index]
                << " expected " << layer_expected[index] << '\n';
    }
    return false;
  }

  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_FRESNEL_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Fresnel capture path " << capture_path
                << '\n';
      return false;
    }
    capture << "family\tcase\tvalue\tcursor\n" << std::setprecision(9);
    for (auto index = std::size_t{}; index < actual_fresnel.size(); ++index) {
      capture << "fresnel\t" << index << '\t' << actual_fresnel[index] << '\t'
              << actual_fresnel_cursors[index] << '\n';
    }
    for (auto index = std::size_t{}; index < actual_layer.size(); ++index) {
      capture << "layer_weight\t" << index << '\t' << actual_layer[index]
              << '\t' << actual_layer_cursors[index] << '\n';
    }
  }
  return true;
}

[[nodiscard]] auto
interpreter_kernel(std::array<bool, NODE_NUM> node_types_used) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[node_types_used](
                                             BufferUInt words,
                                             BufferFloat4 output,
                                             BufferUInt status) noexcept {
    const auto identity = make_float4x4(1.0f);
    const device_svm::TransformState transforms{identity, identity, identity,
                                                identity};
    device_svm::ShaderData shader_data{make_float3(0.0f),
                                       make_float3(0.0f, 0.0f, 1.0f),
                                       make_float3(0.0f, 0.0f, 1.0f),
                                       make_float3(0.0f, 0.0f, 1.0f),
                                       device_svm::primitive_triangle,
                                       0u,
                                       0u,
                                       0u,
                                       0u,
                                       0.0f,
                                       0.0f,
                                       0u,
                                       0.0f,
                                       1.0f,
                                       0.0f,
                                       0.0f,
                                       0.0f,
                                       0.0f,
                                       0.0f,
                                       0.0f,
                                       make_float3(1.0f, 0.0f, 0.0f),
                                       make_float3(0.0f, 1.0f, 0.0f),
                                       identity,
                                       identity};
    const device_svm::PathState path_state{
        device_svm::path_ray_visibility_camera, 0u};
    const psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
    device_svm::EvaluationResult result;
    device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                           device_svm::kernel_feature_node_emission,
                           node_types_used, transforms, shader_data, path_state,
                           result);
    output.write(0u, make_float4(shader_data.closure_emission_background,
                                 result.closure_weight.x));
    status.write(0u, result.status);
  }};
}

[[nodiscard]] bool test_interpreter_dispatch(Device &device, Stream &stream,
                                             std::string_view backend) {
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0u,
                                          .offset_volume = 0u,
                                          .offset_displacement = 0u});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(
      NODE_FRESNEL, SVMNodeFresnel{.ior = input_float(1.5f),
                                   .normal_offset = SVM_STACK_INVALID,
                                   .out_offset = 0u,
                                   ._pad = {0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_LAYER_WEIGHT,
      SVMNodeLayerWeight{.weight_type = NODE_LAYER_WEIGHT_FACING,
                         .blend = input_float(0.5f),
                         .normal_offset = SVM_STACK_INVALID,
                         .out_offset = 1u,
                         ._pad = {0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_VALUE_F,
      SVMNodeValueF{.value = 0.25f, .out_offset = 2u, ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color =
                                input_float3(static_cast<SVMStackOffset>(0u)),
                            .strength = input_float(1.0f)}));
  static_cast<void>(builder.add_node(
      NODE_CLOSURE_EMISSION,
      SVMNodeClosureEmission{.mix_weight_offset = SVM_STACK_INVALID,
                             ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(NODE_END));
  const auto volume = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  const auto displacement = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  builder.set_word(jump + 1u, surface);
  builder.set_word(jump + 2u, volume);
  builder.set_word(jump + 3u, displacement);

  std::array<bool, NODE_NUM> node_types{};
  for (const auto type :
       {NODE_END, NODE_SHADER_JUMP, NODE_FRESNEL, NODE_LAYER_WEIGHT,
        NODE_VALUE_F, NODE_EMISSION_WEIGHT, NODE_CLOSURE_EMISSION}) {
    node_types[type] = true;
  }
  const auto words = builder.words();
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(1u);
  auto status_buffer = device.create_buffer<std::uint32_t>(1u);
  auto shader = device.compile(
      interpreter_kernel(node_types),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  luisa::float4 actual{};
  std::uint32_t status{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer, status_buffer).dispatch(1u)
         << output_buffer.copy_to(&actual) << status_buffer.copy_to(&status)
         << synchronize();
  if (!near(actual.x, 0.0400000028f) || !near(actual.y, 0.0f) ||
      !near(actual.z, 0.25f) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Fresnel-family interpreter mismatch on " << backend
              << ": status=" << status << ", value=(" << actual.x << ", "
              << actual.y << ", " << actual.z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto fresnel_shape = module_shape(fresnel_kernel());
  const auto layer_shape = module_shape(layer_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Fresnel XIR: instructions="
              << fresnel_shape.instructions << ", loops=" << fresnel_shape.loops
              << ", callables=" << fresnel_shape.callable_definitions << '\n'
              << "Cycles SVM Layer Weight XIR: instructions="
              << layer_shape.instructions << ", loops=" << layer_shape.loops
              << ", callables=" << layer_shape.callable_definitions << '\n';
  }
  const auto bounded = [](const ModuleShape &shape) noexcept {
    return shape.instructions <= 1800u && shape.loops == 0u &&
           shape.callable_definitions == 0u;
  };
  if (!bounded(fresnel_shape) || !bounded(layer_shape)) {
    std::cerr << "Cycles Fresnel-family XIR shape regression\n";
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_direct_handlers(device, stream, backend) &&
                 test_interpreter_dispatch(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

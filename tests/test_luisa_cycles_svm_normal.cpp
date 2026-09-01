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

constexpr auto normal_case_count = std::uint32_t{10u};
constexpr auto payload_word_count = std::uint32_t{7u};
constexpr auto stack_input_offset = std::uint32_t{8u};
constexpr auto untouched = -91.0f;

[[nodiscard]] constexpr std::uint32_t float_bits(float value) noexcept {
  return std::bit_cast<std::uint32_t>(value);
}

constexpr auto payload_words =
    std::array<std::uint32_t, normal_case_count * payload_word_count>{
        0x00000000u, 0x00000000u,
        0x40800000u, 0x000000ffu,
        0x00000000u, 0x00000000u,
        0x40000000u, 0xc0400000u,
        0x00000000u, 0x00000000u,
        0x000000ffu, 0x40000000u,
        0x00000000u, 0x00000000u,
        0x40800000u, 0xc0000000u,
        0x3f800000u, 0x000000ffu,
        0x3f800000u, 0x40000000u,
        0x40400000u, 0x3f800000u,
        0x40000000u, 0x40400000u,
        0x000000ffu, 0x00000000u,
        0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u,
        0x00000000u, 0x000000ffu,
        0x3f800000u, 0x40000000u,
        0x40400000u, 0x00000000u,
        0x40400000u, 0x00000000u,
        0x0000ff00u, 0x40000000u,
        0x00000000u, 0x00000000u,
        0x41000000u, 0x3f800000u,
        0xc0000000u, 0x0000ff00u,
        0x00000000u, 0xc0400000u,
        0x40800000u, 0x3f800000u,
        0x40000000u, 0x40400000u,
        0x0000ff00u, 0x00000000u,
        0x00000000u, 0x00000000u,
        0xc0800000u, 0x40a00000u,
        0x3f800000u, 0x00000300u,
        0x3f800000u, 0xc0000000u,
        0x40400000u, SVM_INPUT_STACK_OFFSET_MASK | stack_input_offset,
        0x00000000u, 0x00000000u,
        0x00000300u, 0x40800000u,
        0x00000000u, 0x00000000u};

[[nodiscard]] auto direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < normal_case_count) {
          svm_detail::Stack stack;
          svm_detail::stack_store_float3(stack, 0u, make_float3(untouched));
          svm_detail::stack_store_float(stack, 3u, untouched);
          svm_detail::stack_store_float3(stack, stack_input_offset,
                                         make_float3(2.0f, 0.0f, 0.0f));
          UInt cursor_offset = index * payload_word_count;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_normal(cursor, stack);
          output.write(index,
                       make_float4(svm_detail::stack_load_float(stack, 0u),
                                   svm_detail::stack_load_float(stack, 1u),
                                   svm_detail::stack_load_float(stack, 2u),
                                   svm_detail::stack_load_float(stack, 3u)));
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
                        float tolerance = 1.0e-5f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool finite_equal(float actual, float expected) noexcept {
  return std::isfinite(actual) && near(actual, expected);
}

[[nodiscard]] bool test_direct_handler(Device &device, Stream &stream,
                                       std::string_view backend) {
  auto word_buffer = device.create_buffer<std::uint32_t>(payload_words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(normal_case_count);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(normal_case_count);
  auto shader =
      device.compile(direct_kernel(), ShaderOption{.enable_cache = false,
                                                   .enable_fast_math = false});
  std::array<luisa::float4, normal_case_count> actual{};
  std::array<std::uint32_t, normal_case_count> cursors{};
  stream << word_buffer.copy_from(payload_words.data())
         << shader(word_buffer, output_buffer, cursor_buffer)
                .dispatch(normal_case_count)
         << output_buffer.copy_to(actual.data())
         << cursor_buffer.copy_to(cursors.data()) << synchronize();

  const auto sentinel = [](float value) noexcept { return value == untouched; };
  auto valid = true;
  valid &= finite_equal(actual[0u].x, 1.0f) && sentinel(actual[0u].y) &&
           sentinel(actual[0u].z) && sentinel(actual[0u].w);
  valid &= finite_equal(actual[1u].x, -1.0f) && sentinel(actual[1u].y) &&
           sentinel(actual[1u].z) && sentinel(actual[1u].w);
  valid &= finite_equal(actual[2u].x, 3.0f / std::sqrt(294.0f)) &&
           sentinel(actual[2u].y) && sentinel(actual[2u].z) &&
           sentinel(actual[2u].w);
  valid &= !std::isfinite(actual[3u].x) && sentinel(actual[3u].y) &&
           sentinel(actual[3u].z) && sentinel(actual[3u].w);
  valid &= !std::isfinite(actual[4u].x) && sentinel(actual[4u].y) &&
           sentinel(actual[4u].z) && sentinel(actual[4u].w);
  valid &= finite_equal(actual[5u].x, 1.0f) &&
           finite_equal(actual[5u].y, 0.0f) &&
           finite_equal(actual[5u].z, 0.0f) && sentinel(actual[5u].w);
  valid &= finite_equal(actual[6u].x, 0.0f) &&
           finite_equal(actual[6u].y, -0.6f) &&
           finite_equal(actual[6u].z, 0.8f) && sentinel(actual[6u].w);
  valid &= !std::isfinite(actual[7u].x) && !std::isfinite(actual[7u].y) &&
           !std::isfinite(actual[7u].z) && sentinel(actual[7u].w);
  const auto inverse_sqrt_14 = 1.0f / std::sqrt(14.0f);
  valid &= finite_equal(actual[8u].x, inverse_sqrt_14) &&
           finite_equal(actual[8u].y, -2.0f * inverse_sqrt_14) &&
           finite_equal(actual[8u].z, 3.0f * inverse_sqrt_14) &&
           finite_equal(actual[8u].w, -11.0f / std::sqrt(588.0f));
  valid &= finite_equal(actual[9u].x, 1.0f) &&
           finite_equal(actual[9u].y, 0.0f) &&
           finite_equal(actual[9u].z, 0.0f) && finite_equal(actual[9u].w, 1.0f);
  valid &= std::ranges::all_of(cursors, [](auto cursor) noexcept {
    return cursor == payload_word_count;
  });

  if (!valid) {
    std::cerr << "Cycles Normal direct handler mismatch on " << backend << '\n'
              << std::setprecision(9);
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << index << ": (" << actual[index].x << ", " << actual[index].y
                << ", " << actual[index].z << ", " << actual[index].w
                << "), cursor=" << cursors[index] << '\n';
    }
    return false;
  }

  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_NORMAL_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Normal capture path " << capture_path
                << '\n';
      return false;
    }
    capture << "case\tx\ty\tz\tdot\tcursor\n" << std::hex << std::setfill('0');
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      capture << std::dec << index << std::hex << '\t' << std::setw(8)
              << float_bits(actual[index].x) << '\t' << std::setw(8)
              << float_bits(actual[index].y) << '\t' << std::setw(8)
              << float_bits(actual[index].z) << '\t' << std::setw(8)
              << float_bits(actual[index].w) << std::dec << '\t'
              << cursors[index] << '\n';
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
      NODE_NORMAL, SVMNodeNormal{.in_normal = input_float3(0.0f, 0.0f, 4.0f),
                                 .out_normal_offset = 0u,
                                 .out_dot_offset = 3u,
                                 ._pad = {0u, 0u},
                                 .direction_x = 0.0f,
                                 .direction_y = 0.0f,
                                 .direction_z = 2.0f}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color = input_float3(SVMStackOffset{0u}),
                            .strength = input_float(SVMStackOffset{3u})}));
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

  const auto words = builder.words();
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(1u);
  auto status_buffer = device.create_buffer<std::uint32_t>(1u);
  auto shader = device.compile(
      interpreter_kernel(builder.node_types_used()),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  luisa::float4 actual{};
  std::uint32_t status{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer, status_buffer).dispatch(1u)
         << output_buffer.copy_to(&actual) << status_buffer.copy_to(&status)
         << synchronize();
  if (!near(actual.x, 0.0f) || !near(actual.y, 0.0f) || !near(actual.z, 1.0f) ||
      !near(actual.w, 0.0f) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Normal interpreter mismatch on " << backend
              << ": status=" << status << ", value=(" << actual.x << ", "
              << actual.y << ", " << actual.z << ", " << actual.w << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(direct_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Normal XIR: instructions=" << shape.instructions
              << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 1800u || shape.loops != 0u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles Normal XIR shape regression\n";
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_direct_handler(device, stream, backend) &&
                 test_interpreter_dispatch(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

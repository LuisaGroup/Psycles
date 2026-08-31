#include <psycles/luisa/cycles_svm.h>
#include <psycles/compiler/cycles_svm_bytecode.h>

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
#include <span>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto vector_offset = std::uint32_t{0u};
constexpr auto w_offset = std::uint32_t{4u};
constexpr auto value_offset = std::uint32_t{8u};
constexpr auto color_offset = std::uint32_t{12u};
constexpr auto white_noise_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeTexWhiteNoise) / sizeof(std::uint32_t));
static_assert(white_noise_word_count == 6u);
constexpr auto oracle_case_count = std::uint32_t{11u};

template<typename T>
void append_payload(std::vector<std::uint32_t> &words,
                    const T &payload) {
  const auto encoded =
      std::bit_cast<std::array<std::uint32_t,
                               sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

[[nodiscard]] constexpr luisa::float3 color(
    std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
  return {f32(x), f32(y), f32(z)};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-7f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 2.0e-7f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

// Cycles 5.2.1 CPU Combined pixels sampled at the centers of the 4x2
// White Noise matrix. Each tile is one emission material, so these are direct
// node outputs from the external renderer rather than a host reference model.
constexpr std::array color_oracle{
    color(0x3f61eb51u, 0x3c430e8bu, 0x3e5761e7u),
    color(0x3f0468dfu, 0x3eb4e1cdu, 0x3e7fefa8u),
    color(0x3f693b67u, 0x3f4dfe01u, 0x3e56b382u),
    color(0x3ea4e93cu, 0x3dea2fcbu, 0x3db5df56u)};

[[nodiscard]] SVMNodeTexWhiteNoise payload(
    std::uint32_t dimensions, SVMStackOffset value_out,
    SVMStackOffset color_out, bool stack_inputs = false) noexcept {
  return {.dimensions = dimensions,
          .vector = stack_inputs
                        ? input_float3(
                              static_cast<SVMStackOffset>(vector_offset))
                        : input_float3(0.173f, -0.625f, 1.375f),
          .w = stack_inputs
                   ? input_float(static_cast<SVMStackOffset>(w_offset))
                   : input_float(-0.437f),
          .value_offset = value_out,
          .color_offset = color_out,
          ._pad = {0u, 0u}};
}

[[nodiscard]] auto white_noise_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(
            stack, vector_offset,
            make_float3(0.173f, -0.625f, 1.375f));
        svm_detail::stack_store_float(stack, w_offset, -0.437f);
        svm_detail::stack_store_float(stack, value_offset, -91.0f);
        svm_detail::stack_store_float3(
            stack, color_offset, make_float3(-92.0f, -93.0f, -94.0f));
        UInt cursor_offset = index * white_noise_word_count;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::node_tex_white_noise(cursor, stack);
        output.write(
            index,
            make_float4(
                svm_detail::stack_load_float3(stack, color_offset),
                svm_detail::stack_load_float(stack, value_offset)));
        cursors.write(index,
                      cursor_offset - index * white_noise_word_count);
      }};
}

struct ModuleShape {
  std::size_t instructions{};
  std::size_t callable_definitions{};
};

template<typename... Args>
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
          [&](const luisa::compute::xir::Instruction *) noexcept {
            ++result.instructions;
          });
    }
  }
  return result;
}

[[nodiscard]] bool test_oracle(Device &device, Stream &stream,
                               std::string_view backend) {
  std::vector<std::uint32_t> words;
  words.reserve(oracle_case_count * white_noise_word_count);
  for (auto dimensions = std::uint32_t{1u}; dimensions <= 4u;
       ++dimensions) {
    append_payload(
        words,
        payload(dimensions, static_cast<SVMStackOffset>(value_offset),
                SVM_STACK_INVALID));
  }
  for (auto dimensions = std::uint32_t{1u}; dimensions <= 4u;
       ++dimensions) {
    append_payload(
        words,
        payload(dimensions, SVM_STACK_INVALID,
                static_cast<SVMStackOffset>(color_offset)));
  }
  append_payload(
      words,
      payload(4u, static_cast<SVMStackOffset>(value_offset),
              static_cast<SVMStackOffset>(color_offset), true));
  for (const auto dimensions : {0u, 5u}) {
    append_payload(
        words,
        payload(dimensions, static_cast<SVMStackOffset>(value_offset),
                static_cast<SVMStackOffset>(color_offset)));
  }

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer =
      device.create_buffer<luisa::float4>(oracle_case_count);
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(oracle_case_count);
  auto shader = device.compile(
      white_noise_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, oracle_case_count> actual{};
  std::array<std::uint32_t, oracle_case_count> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer)
                .dispatch(oracle_case_count)
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  for (auto index = std::uint32_t{}; index < 4u; ++index) {
    if (!near(actual[index].w, color_oracle[index].x) ||
        actual[index].x != -92.0f || actual[index].y != -93.0f ||
        actual[index].z != -94.0f ||
        cursors[index] != white_noise_word_count) {
      std::cerr << "Cycles White Noise Value oracle mismatch on "
                << backend << ", dimensions=" << index + 1u << '\n';
      return false;
    }
  }
  for (auto index = std::uint32_t{}; index < 4u; ++index) {
    const auto case_index = index + 4u;
    if (!near(luisa::float3{actual[case_index].x,
                            actual[case_index].y,
                            actual[case_index].z},
              color_oracle[index]) ||
        actual[case_index].w != -91.0f ||
        cursors[case_index] != white_noise_word_count) {
      std::cerr << "Cycles White Noise Color oracle mismatch on "
                << backend << ", dimensions=" << index + 1u << '\n';
      return false;
    }
  }
  const auto stack_case = actual[8u];
  if (!near(luisa::float3{stack_case.x, stack_case.y, stack_case.z},
            color_oracle[3u]) ||
      !near(stack_case.w, color_oracle[3u].x) ||
      cursors[8u] != white_noise_word_count) {
    std::cerr << "Cycles White Noise stack-input oracle mismatch on "
              << backend << '\n';
    return false;
  }
  for (auto index = std::uint32_t{9u}; index < oracle_case_count; ++index) {
    if (actual[index].x != 1.0f || actual[index].y != 0.0f ||
        actual[index].z != 1.0f || actual[index].w != 0.0f ||
        cursors[index] != white_noise_word_count) {
      std::cerr << "Cycles White Noise invalid-dimension default mismatch on "
                << backend << '\n';
      return false;
    }
  }

  if (const auto *capture_path =
          std::getenv("PSYCLES_CAPTURE_CYCLES_SVM_WHITE_NOISE");
      capture_path != nullptr && capture_path[0] != '\0') {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Cycles White Noise capture path "
                << capture_path << '\n';
      return false;
    }
    capture << "output\tdimensions\tr\tg\tb\tvalue\n"
            << std::setprecision(9);
    for (auto index = std::uint32_t{}; index < 4u; ++index) {
      capture << "Value\t" << index + 1u << '\t' << actual[index].x
              << '\t' << actual[index].y << '\t' << actual[index].z
              << '\t' << actual[index].w << '\n';
    }
    for (auto index = std::uint32_t{}; index < 4u; ++index) {
      const auto value = actual[index + 4u];
      capture << "Color\t" << index + 1u << '\t' << value.x << '\t'
              << value.y << '\t' << value.z << '\t' << value.w << '\n';
    }
  }
  return true;
}

[[nodiscard]] auto interpreter_kernel(
    std::array<bool, NODE_NUM> node_types_used) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [node_types_used](BufferUInt words, BufferFloat4 output,
                        BufferUInt status) noexcept {
        const auto identity = make_float4x4(1.0f);
        const device_svm::TransformState transforms{
            identity, identity, identity, identity};
        const psycles::test_support::DefaultCyclesSvmKernelGlobals
            kernel_globals;
        device_svm::ShaderData shader_data{
            make_float3(0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            device_svm::primitive_triangle,
            0u,
            0u,
            0u,
            0u,
            0.2f,
            0.3f,
            0u,
            0.0f,
            4.0f,
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
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
            device_svm::kernel_feature_node_emission, node_types_used,
            transforms, shader_data, path_state, result);
        output.write(0u,
                     make_float4(shader_data.closure_emission_background,
                                 result.closure_weight.x));
        status.write(0u, result.status);
      }};
}

[[nodiscard]] bool test_interpreter_dispatch(
    Device &device, Stream &stream, std::string_view backend) {
  // Cycles 5.2.1 matrix shader `White Noise 4D Color`, rebased to shader
  // zero. This proves reachability through the real PC loop and opcode switch.
  static constexpr std::array words{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u,
      0x00000047u, 0x00000004u, 0x3e3126e9u, 0xbf200000u,
      0x3fb00000u, 0xbedfbe77u, 0x000000ffu, 0x00000007u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  std::array<bool, NODE_NUM> node_types{};
  node_types[NODE_END] = true;
  node_types[NODE_SHADER_JUMP] = true;
  node_types[NODE_TEX_WHITE_NOISE] = true;
  node_types[NODE_EMISSION_WEIGHT] = true;
  node_types[NODE_CLOSURE_EMISSION] = true;
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
         << output_buffer.copy_to(&actual)
         << status_buffer.copy_to(&status) << synchronize();
  if (!near(luisa::float3{actual.x, actual.y, actual.z},
            color_oracle[3u]) ||
      status != static_cast<std::uint32_t>(
                    device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles White Noise interpreter dispatch mismatch on "
              << backend << ": status=" << status << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(white_noise_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM White Noise XIR: instructions="
              << shape.instructions
              << ", callables=" << shape.callable_definitions << '\n';
  }
  // The runtime SVM payload requires Cycles's two independent output-validity
  // branches and two four-way dimension switches. Their color/value arms
  // contain sixteen expanded Jenkins-hash paths in total. The current direct
  // handler is 6694 XIR instructions; leave only a small structural-growth
  // margin and forbid the accidental creation of callable families.
  if (shape.instructions > 7000u || shape.callable_definitions != 0u) {
    std::cerr << "Cycles SVM White Noise shape regression: callables="
              << shape.callable_definitions
              << ", instructions=" << shape.instructions << '\n';
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_oracle(device, stream, backend) &&
                 test_interpreter_dispatch(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

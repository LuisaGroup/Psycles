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
constexpr auto factor_offset = std::uint32_t{4u};
constexpr auto color_offset = std::uint32_t{8u};
constexpr auto gradient_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeTexGradient) / sizeof(std::uint32_t));
static_assert(gradient_word_count == 2u);

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

struct OracleCase {
  NodeGradientType type;
  luisa::float3 point;
  float result;
};

// Cycles 5.2.1 CPU node results from the external 4x4 gradient_matrix EXR and
// final SVM stream. The table covers every mode and every saturation branch.
constexpr std::array oracle_cases{
    OracleCase{NODE_BLEND_LINEAR, {-2.0f, 0.0f, 0.0f}, 0.0f},
    OracleCase{NODE_BLEND_LINEAR, {0.37f, 2.0f, -1.0f}, f32(0x3ebd70a4u)},
    OracleCase{NODE_BLEND_LINEAR, {2.0f, 0.0f, 0.0f}, 1.0f},
    OracleCase{NODE_BLEND_QUADRATIC, {-1.0f, 0.0f, 0.0f}, 0.0f},
    OracleCase{NODE_BLEND_QUADRATIC, {0.5f, 0.0f, 0.0f}, 0.25f},
    OracleCase{NODE_BLEND_EASING, {-0.5f, 0.0f, 0.0f}, 0.0f},
    OracleCase{NODE_BLEND_EASING, {0.3f, 0.0f, 0.0f}, f32(0x3e5d2f1bu)},
    OracleCase{NODE_BLEND_EASING, {1.5f, 0.0f, 0.0f}, 1.0f},
    OracleCase{NODE_BLEND_DIAGONAL, {-1.0f, 0.2f, 0.0f}, 0.0f},
    OracleCase{NODE_BLEND_DIAGONAL, {1.4f, 0.8f, 0.0f}, 1.0f},
    OracleCase{NODE_BLEND_RADIAL, {1.0f, 0.0f, 0.0f}, 0.5f},
    OracleCase{NODE_BLEND_RADIAL, {0.0f, -1.0f, 0.0f}, 0.25f},
    OracleCase{NODE_BLEND_SPHERICAL, {0.0f, 0.0f, 0.0f}, f32(0x3f7fffefu)},
    OracleCase{NODE_BLEND_SPHERICAL, {1.0f, 0.0f, 0.0f}, 0.0f},
    OracleCase{
        NODE_BLEND_QUADRATIC_SPHERE, {0.0f, 0.0f, 0.0f}, f32(0x3f7fffdeu)},
    OracleCase{
        NODE_BLEND_QUADRATIC_SPHERE, {0.5f, 0.5f, 0.5f}, f32(0x3c9309a0u)}};
constexpr auto direct_case_count =
    static_cast<std::uint32_t>(oracle_cases.size() * 2u + 2u);

template <typename T>
void append_payload(std::vector<std::uint32_t> &words, const T &payload) {
  const auto encoded = std::bit_cast<
      std::array<std::uint32_t, sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] SVMNodeTexGradient payload(NodeGradientType type,
                                         SVMStackOffset factor_out,
                                         SVMStackOffset color_out) noexcept {
  return {.gradient_type = type,
          .co = static_cast<SVMStackOffset>(vector_offset),
          .fac_offset = factor_out,
          .color_offset = color_out,
          ._pad = {0u}};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-7f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] auto gradient_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 points, BufferFloat4 output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        const auto point = points.read(index);
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(stack, vector_offset, point.xyz());
        svm_detail::stack_store_float(stack, factor_offset, -91.0f);
        svm_detail::stack_store_float3(stack, color_offset,
                                       make_float3(-92.0f, -93.0f, -94.0f));
        UInt cursor_offset = index * gradient_word_count;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::node_tex_gradient(cursor, stack);
        output.write(
            index,
            make_float4(svm_detail::stack_load_float3(stack, color_offset),
                        svm_detail::stack_load_float(stack, factor_offset)));
        cursors.write(index, cursor_offset - index * gradient_word_count);
      }};
}

struct ModuleShape {
  std::size_t instructions{};
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
          [&](const luisa::compute::xir::Instruction *) noexcept {
            ++result.instructions;
          });
    }
  }
  return result;
}

[[nodiscard]] bool test_direct_handler(Device &device, Stream &stream,
                                       std::string_view backend) {
  std::vector<std::uint32_t> words;
  std::vector<luisa::float4> points;
  words.reserve(direct_case_count * gradient_word_count);
  points.reserve(direct_case_count);
  for (const auto &test : oracle_cases) {
    append_payload(words, payload(test.type,
                                  static_cast<SVMStackOffset>(factor_offset),
                                  SVM_STACK_INVALID));
    points.emplace_back(test.point.x, test.point.y, test.point.z, 0.0f);
  }
  for (const auto &test : oracle_cases) {
    append_payload(words, payload(test.type, SVM_STACK_INVALID,
                                  static_cast<SVMStackOffset>(color_offset)));
    points.emplace_back(test.point.x, test.point.y, test.point.z, 0.0f);
  }
  for (auto invalid : {7u, 0xffffffffu}) {
    append_payload(words, payload(static_cast<NodeGradientType>(invalid),
                                  static_cast<SVMStackOffset>(factor_offset),
                                  static_cast<SVMStackOffset>(color_offset)));
    points.emplace_back(0.3f, -0.7f, 0.2f, 0.0f);
  }

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto point_buffer = device.create_buffer<luisa::float4>(points.size());
  auto output_buffer = device.create_buffer<luisa::float4>(direct_case_count);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(direct_case_count);
  auto shader = device.compile(
      gradient_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, direct_case_count> actual{};
  std::array<std::uint32_t, direct_case_count> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << point_buffer.copy_from(luisa::span{points})
         << shader(word_buffer, point_buffer, output_buffer, cursor_buffer)
                .dispatch(direct_case_count)
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  for (auto index = std::size_t{}; index < oracle_cases.size(); ++index) {
    if (!near(actual[index].w, oracle_cases[index].result) ||
        actual[index].x != -92.0f || actual[index].y != -93.0f ||
        actual[index].z != -94.0f || cursors[index] != gradient_word_count) {
      std::cerr << "Cycles Gradient Factor oracle mismatch on " << backend
                << ", case=" << index << '\n';
      return false;
    }
  }
  for (auto index = std::size_t{}; index < oracle_cases.size(); ++index) {
    const auto output_index = index + oracle_cases.size();
    const auto value = oracle_cases[index].result;
    if (!near(actual[output_index].x, value) ||
        !near(actual[output_index].y, value) ||
        !near(actual[output_index].z, value) ||
        actual[output_index].w != -91.0f ||
        cursors[output_index] != gradient_word_count) {
      std::cerr << "Cycles Gradient Color oracle mismatch on " << backend
                << ", case=" << index << '\n';
      return false;
    }
  }
  for (auto index = oracle_cases.size(); index < oracle_cases.size() + 2u;
       ++index) {
    const auto output_index = index + oracle_cases.size();
    if (actual[output_index].x != 0.0f || actual[output_index].y != 0.0f ||
        actual[output_index].z != 0.0f || actual[output_index].w != 0.0f ||
        cursors[output_index] != gradient_word_count) {
      std::cerr << "Cycles Gradient invalid-type sentinel mismatch on "
                << backend << '\n';
      return false;
    }
  }

  if (const auto *capture_path =
          std::getenv("PSYCLES_CAPTURE_CYCLES_SVM_GRADIENT");
      capture_path != nullptr && capture_path[0] != '\0') {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Cycles Gradient capture path "
                << capture_path << '\n';
      return false;
    }
    capture << "output\tcase\tr\tg\tb\tfactor\n" << std::setprecision(9);
    for (auto index = std::size_t{}; index < oracle_cases.size(); ++index) {
      capture << "Factor\t" << index << '\t' << actual[index].x << '\t'
              << actual[index].y << '\t' << actual[index].z << '\t'
              << actual[index].w << '\n';
      const auto color = actual[index + oracle_cases.size()];
      capture << "Color\t" << index << '\t' << color.x << '\t' << color.y
              << '\t' << color.z << '\t' << color.w << '\n';
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
    const psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
    device_svm::ShaderData shader_data{make_float3(0.0f),
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
  // Complete shader-local surface/volume/displacement stream from the
  // external Cycles 5.2.1 gradient_spherical probe, rebased to shader zero.
  static constexpr std::array words{
      0x00000001u, 0x00000004u, 0x00000024u, 0x00000025u, 0x0000000fu,
      0x00000001u, 0x00000000u, 0x00000039u, 0x00000000u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x3f19999au, 0x3f19999au,
      0x3f19999au, 0x00000003u, 0x00000040u, 0x00000006u, 0x00ff0003u,
      0x0000000du, 0x00000000u, 0x00000100u, 0x00000007u, 0x7fc00001u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  std::array<bool, NODE_NUM> node_types{};
  node_types[NODE_END] = true;
  node_types[NODE_SHADER_JUMP] = true;
  node_types[NODE_TEX_COORD] = true;
  node_types[NODE_MAPPING] = true;
  node_types[NODE_TEX_GRADIENT] = true;
  node_types[NODE_CONVERT] = true;
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
         << output_buffer.copy_to(&actual) << status_buffer.copy_to(&status)
         << synchronize();
  const auto expected = f32(0x3f7fffefu);
  if (!near(actual.x, expected) || !near(actual.y, expected) ||
      !near(actual.z, expected) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Gradient interpreter dispatch mismatch on " << backend
              << ": status=" << status << ", value=" << actual.x << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(gradient_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Gradient XIR: instructions=" << shape.instructions
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 1000u || shape.callable_definitions != 0u) {
    std::cerr << "Cycles SVM Gradient shape regression: callables="
              << shape.callable_definitions
              << ", instructions=" << shape.instructions << '\n';
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

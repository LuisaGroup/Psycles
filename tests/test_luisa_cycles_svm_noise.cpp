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
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto vector_offset = std::uint32_t{0u};
constexpr auto value_offset = std::uint32_t{16u};
constexpr auto color_offset = std::uint32_t{20u};
constexpr auto noise_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeTexNoise) / sizeof(std::uint32_t));
constexpr auto noise_type_count = std::uint32_t{5u};
constexpr auto dimension_count = std::uint32_t{4u};
constexpr auto normalize_count = std::uint32_t{2u};
constexpr auto oracle_case_count =
    noise_type_count * dimension_count * normalize_count;

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

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 1.0e-4f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 1.0e-4f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] auto noise_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(
            stack, vector_offset,
            make_float3(0.173f, 0.0f, 1.375f));
        UInt cursor_offset = index * noise_word_count;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::node_tex_noise(cursor, stack);
        const auto value =
            svm_detail::stack_load_float(stack, value_offset);
        const auto color =
            svm_detail::stack_load_float3(stack, color_offset);
        output.write(index, make_float4(color, value));
        cursors.write(index,
                      cursor_offset - index * noise_word_count);
      }};
}

struct ModuleShape {
  std::size_t instructions{};
  std::size_t callable_definitions{};
  std::size_t loops{};
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
          [&](const luisa::compute::xir::Instruction *instruction) noexcept {
            ++result.instructions;
            result.loops +=
                instruction->isa<luisa::compute::xir::LoopInst>() ||
                        instruction->isa<luisa::compute::xir::SimpleLoopInst>()
                    ? 1u
                    : 0u;
          });
    }
  }
  return result;
}

[[nodiscard]] SVMNodeTexNoise payload(std::uint32_t dimensions,
                                      NodeNoiseType type,
                                      bool normalize,
                                      SVMStackOffset value_out,
                                      SVMStackOffset color_out,
                                      float detail = 2.375f,
                                      float roughness = 0.63f) noexcept {
  return {.dimensions = dimensions,
          .noise_type = type,
          .normalize = static_cast<std::uint32_t>(normalize),
          .w = input_float(-0.437f),
          .scale = input_float(2.35f),
          .detail = input_float(detail),
          .roughness = input_float(roughness),
          .lacunarity = input_float(2.17f),
          .offset = input_float(0.37f),
          .gain = input_float(1.11f),
          .distortion = input_float(0.42f),
          .vector = static_cast<SVMStackOffset>(vector_offset),
          .value_offset = value_out,
          .color_offset = color_out,
          ._pad = {0u}};
}

[[nodiscard]] constexpr luisa::float3 color(std::uint32_t x,
                                             std::uint32_t y,
                                             std::uint32_t z) noexcept {
  return {f32(x), f32(y), f32(z)};
}

// Cycles 5.2.1 CPU Combined pixels sampled at the centers of the five
// 4x4 Noise matrix probes. The scenes use one emission material per cell,
// so these are direct node values rather than a Psycles reference model.
constexpr std::array raw_oracle{
    std::array{color(0x3f924518u, 0x3f93496cu, 0x3f8383feu),
               color(0x3f6af9afu, 0x3fab031bu, 0x3f0278f0u),
               color(0x3f81c9abu, 0x3f84e287u, 0x3f6a8baau),
               color(0x3f420f11u, 0x3f5ad0a1u, 0x3fa8fd02u)},
    std::array{color(0x3e0cf9e3u, 0x3e1b708bu, 0x3d5fd271u),
               color(0xbcd279e3u, 0x3ea3b7b0u, 0xbe96003fu),
               color(0x3d2a8132u, 0x3d5e20d8u, 0xbd8e7b7cu),
               color(0xbe7df81au, 0xbe15b8a3u, 0x3ea2e51au)},
    std::array{color(0x3f0725dau, 0x3f3000c0u, 0x3f05deafu),
               color(0x3e99681bu, 0x3f4a9e25u, 0xbe80f072u),
               color(0x3f2b5a43u, 0x3ed66d3cu, 0x3ec801e1u),
               color(0x3e4f08cfu, 0x3ea01ca2u, 0x3f634c24u)},
    std::array{color(0x3e0fb33au, 0x3d49cf5cu, 0x3d9b6275u),
               color(0x3d180fc0u, 0x3ccee1c6u, 0x3d82086au),
               color(0x3cd645e8u, 0x3da231ffu, 0x3d9fff71u),
               color(0x3cdd5225u, 0x3d7cf556u, 0x3cc67ba5u)},
    std::array{color(0x3f1cb264u, 0x3f44acc8u, 0x3f2699e5u),
               color(0x3e9db352u, 0x3f6e0138u, 0xbef27c4du),
               color(0x3f3be35cu, 0x3eea0004u, 0x3ed02b9bu),
               color(0x3e643da8u, 0x3eb0404cu, 0x3f690610u)}};

constexpr std::array normalized_fbm_oracle{
    color(0x3f0855e4u, 0x3f093b7au, 0x3f0379d0u),
    color(0x3efca349u, 0x3f13878du, 0x3edc7536u),
    color(0x3f0283c9u, 0x3f033585u, 0x3ef77e96u),
    color(0x3ee1bd8fu, 0x3eee2f79u, 0x3f135d58u)};

[[nodiscard]] bool test_oracle(Device &device, Stream &stream,
                               std::string_view backend) {
  std::vector<std::uint32_t> words;
  words.reserve(oracle_case_count * noise_word_count);
  for (auto type = std::uint32_t{}; type < noise_type_count; ++type) {
    for (auto normalize = std::uint32_t{}; normalize < normalize_count;
         ++normalize) {
      for (auto dimension = std::uint32_t{1u};
           dimension <= dimension_count; ++dimension) {
        append_payload(
            words,
            payload(dimension, static_cast<NodeNoiseType>(type),
                    normalize != 0u,
                    static_cast<SVMStackOffset>(value_offset),
                    static_cast<SVMStackOffset>(color_offset)));
      }
    }
  }

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer =
      device.create_buffer<luisa::float4>(oracle_case_count);
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(oracle_case_count);
  auto shader = device.compile(
      noise_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, oracle_case_count> actual{};
  std::array<std::uint32_t, oracle_case_count> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer)
                .dispatch(oracle_case_count)
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  auto index = std::size_t{};
  auto maximum_absolute_error = 0.0f;
  for (auto type = std::uint32_t{}; type < noise_type_count; ++type) {
    for (auto normalize = std::uint32_t{}; normalize < normalize_count;
         ++normalize) {
      for (auto dimension = std::uint32_t{};
           dimension < dimension_count; ++dimension, ++index) {
        const auto expected_color =
            type == static_cast<std::uint32_t>(NODE_NOISE_FBM) &&
                    normalize != 0u
                ? normalized_fbm_oracle[dimension]
                : raw_oracle[type][dimension];
        const auto expected = luisa::float4{
            expected_color.x, expected_color.y, expected_color.z,
            expected_color.x};
        maximum_absolute_error = std::max(
            maximum_absolute_error,
            std::max({std::abs(actual[index].x - expected.x),
                      std::abs(actual[index].y - expected.y),
                      std::abs(actual[index].z - expected.z),
                      std::abs(actual[index].w - expected.w)}));
        if (!near(actual[index], expected) ||
            cursors[index] != noise_word_count) {
          std::cerr << "Cycles Noise oracle mismatch on " << backend
                    << ", type=" << type
                    << ", dimensions=" << dimension + 1u
                    << ", normalize=" << normalize << ": got {"
                    << actual[index].x << ", " << actual[index].y << ", "
                    << actual[index].z << ", " << actual[index].w
                    << "}, expected {" << expected.x << ", " << expected.y
                    << ", " << expected.z << ", " << expected.w
                    << "}, cursor=" << cursors[index] << '\n';
          return false;
        }
      }
    }
  }
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Noise " << backend
              << " maximum Cycles-CPU absolute error: "
              << maximum_absolute_error << '\n';
  }
  if (const auto *capture_path =
          std::getenv("PSYCLES_CAPTURE_CYCLES_SVM_NOISE");
      capture_path != nullptr && capture_path[0] != '\0') {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Cycles SVM Noise capture path "
                << capture_path << '\n';
      return false;
    }
    capture << "type\tnormalize\tdimensions\tr\tg\tb\tvalue"
               "\texpected_r\texpected_g\texpected_b\texpected_value\n"
            << std::setprecision(9);
    index = 0u;
    for (auto type = std::uint32_t{}; type < noise_type_count; ++type) {
      for (auto normalize = std::uint32_t{}; normalize < normalize_count;
           ++normalize) {
        for (auto dimension = std::uint32_t{1u};
             dimension <= dimension_count; ++dimension, ++index) {
          const auto value = actual[index];
          const auto expected_color =
              type == static_cast<std::uint32_t>(NODE_NOISE_FBM) &&
                      normalize != 0u
                  ? normalized_fbm_oracle[dimension - 1u]
                  : raw_oracle[type][dimension - 1u];
          capture << type << '\t' << normalize << '\t' << dimension << '\t'
                  << value.x << '\t' << value.y << '\t' << value.z << '\t'
                  << value.w << '\t' << expected_color.x << '\t'
                  << expected_color.y << '\t' << expected_color.z << '\t'
                  << expected_color.x << '\n';
        }
      }
    }
    if (!capture) {
      std::cerr << "Could not write Cycles SVM Noise capture path "
                << capture_path << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_clamp_and_output_validity(
    Device &device, Stream &stream, std::string_view backend) {
  constexpr auto case_count = std::uint32_t{8u};
  std::vector<std::uint32_t> words;
  words.reserve(case_count * noise_word_count);
  // Clamp equivalence pairs are a direct consequence of Cycles'
  // svm_node_tex_noise preconditions and avoid inventing a host evaluator.
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                static_cast<SVMStackOffset>(value_offset),
                                static_cast<SVMStackOffset>(color_offset),
                                20.0f, -0.3f));
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                static_cast<SVMStackOffset>(value_offset),
                                static_cast<SVMStackOffset>(color_offset),
                                15.0f, 0.0f));
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                static_cast<SVMStackOffset>(value_offset),
                                static_cast<SVMStackOffset>(color_offset),
                                -2.0f, 0.63f));
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                static_cast<SVMStackOffset>(value_offset),
                                static_cast<SVMStackOffset>(color_offset),
                                0.0f, 0.63f));
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                static_cast<SVMStackOffset>(value_offset),
                                SVM_STACK_INVALID));
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                SVM_STACK_INVALID,
                                static_cast<SVMStackOffset>(color_offset)));
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                SVM_STACK_INVALID, SVM_STACK_INVALID));
  append_payload(words, payload(3u, NODE_NOISE_FBM, true,
                                static_cast<SVMStackOffset>(value_offset),
                                static_cast<SVMStackOffset>(color_offset)));

  Kernel1D kernel = [](BufferUInt payloads, BufferFloat4 output) noexcept {
    const UInt index = dispatch_x();
    svm_detail::Stack stack;
    svm_detail::stack_store_float3(
        stack, vector_offset, make_float3(0.173f, 0.0f, 1.375f));
    svm_detail::stack_store_float(stack, value_offset, -91.0f);
    svm_detail::stack_store_float3(
        stack, color_offset, make_float3(-92.0f, -93.0f, -94.0f));
    UInt cursor_offset = index * noise_word_count;
    svm_detail::Cursor cursor{payloads, cursor_offset};
    svm_detail::node_tex_noise(cursor, stack);
    output.write(
        index,
        make_float4(
            svm_detail::stack_load_float3(stack, color_offset),
            svm_detail::stack_load_float(stack, value_offset)));
  };
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(case_count);
  auto shader = device.compile(
      kernel,
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, case_count> actual{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer).dispatch(case_count)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();
  if (!near(actual[0u], actual[1u]) || !near(actual[2u], actual[3u]) ||
      !near(actual[4u].w, actual[7u].w) || actual[4u].x != -92.0f ||
      actual[4u].y != -93.0f || actual[4u].z != -94.0f ||
      !near(actual[5u].x, actual[7u].x) ||
      !near(actual[5u].y, actual[7u].y) ||
      !near(actual[5u].z, actual[7u].z) || actual[5u].w != -91.0f ||
      actual[6u].x != -92.0f || actual[6u].y != -93.0f ||
      actual[6u].z != -94.0f || actual[6u].w != -91.0f) {
    std::cerr << "Cycles Noise clamp/output-validity mismatch on "
              << backend << '\n';
    return false;
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
  // Cycles 5.2.1 matrix shader `Noise FBM 3D Normalized Color`, rebased to
  // shader zero. This proves NODE_TEX_NOISE is reachable through the real
  // PC loop and opcode switch, not only through the isolated handler.
  static constexpr std::array words{
      0x00000001u, 0x00000004u, 0x0000001eu, 0x0000001fu,
      0x00000013u, 0x00000000u, 0x3e3126e9u, 0x00000000u,
      0x3fb00000u, 0x00000020u, 0x00000003u, 0x00000001u,
      0x00000001u, 0x00000000u, 0x40166666u, 0x40180000u,
      0x3f2147aeu, 0x400ae148u, 0x00000000u, 0x3f800000u,
      0x3ed70a3du, 0x0003ff00u, 0x00000007u, 0x7fc00003u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u,
      0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};
  std::array<bool, NODE_NUM> node_types{};
  node_types[NODE_END] = true;
  node_types[NODE_SHADER_JUMP] = true;
  node_types[NODE_VALUE_V] = true;
  node_types[NODE_TEX_NOISE] = true;
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
  const auto expected_color = normalized_fbm_oracle[2u];
  if (!near(actual.x, expected_color.x) ||
      !near(actual.y, expected_color.y) ||
      !near(actual.z, expected_color.z) ||
      status != static_cast<std::uint32_t>(
                    device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Noise interpreter dispatch mismatch on " << backend
              << ": got {" << actual.x << ", " << actual.y << ", "
              << actual.z << "}, status=" << status << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(noise_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Noise XIR: instructions=" << shape.instructions
              << ", callables=" << shape.callable_definitions
              << ", loops=" << shape.loops << '\n';
  }
  if (shape.instructions > 36000u || shape.callable_definitions != 12u ||
      shape.loops != 20u) {
    std::cerr << "Cycles SVM Noise specialization regression: callables="
              << shape.callable_definitions << ", loops=" << shape.loops
              << ", instructions=" << shape.instructions << '\n';
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_oracle(device, stream, backend) &&
                 test_clamp_and_output_validity(device, stream, backend) &&
                 test_interpreter_dispatch(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

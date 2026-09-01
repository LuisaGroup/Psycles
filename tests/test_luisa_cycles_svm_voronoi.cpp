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
constexpr auto w_input_offset = std::uint32_t{4u};
constexpr auto scale_input_offset = std::uint32_t{5u};
constexpr auto detail_input_offset = std::uint32_t{6u};
constexpr auto roughness_input_offset = std::uint32_t{7u};
constexpr auto lacunarity_input_offset = std::uint32_t{8u};
constexpr auto smoothness_input_offset = std::uint32_t{9u};
constexpr auto exponent_input_offset = std::uint32_t{10u};
constexpr auto randomness_input_offset = std::uint32_t{11u};
constexpr auto distance_offset = std::uint32_t{16u};
constexpr auto color_offset = std::uint32_t{20u};
constexpr auto position_offset = std::uint32_t{24u};
constexpr auto w_output_offset = std::uint32_t{28u};
constexpr auto radius_offset = std::uint32_t{29u};
constexpr auto voronoi_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeTexVoronoi) / sizeof(std::uint32_t));
static_assert(voronoi_word_count == 13u);

constexpr auto distance_sentinel = -91.0f;
constexpr auto color_sentinel = luisa::float3{-92.0f, -93.0f, -94.0f};
constexpr auto position_sentinel = luisa::float3{-95.0f, -96.0f, -97.0f};
constexpr auto w_sentinel = -98.0f;
constexpr auto radius_sentinel = -99.0f;

enum class OutputKind : std::uint8_t {
  distance,
  color,
  position,
  w,
  radius,
};

struct OracleCase {
  std::uint32_t dimensions;
  NodeVoronoiFeature feature;
  NodeVoronoiDistanceMetric metric;
  bool normalize;
  OutputKind output;
  luisa::float3 expected;
};

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

[[nodiscard]] constexpr luisa::float3 color(std::uint32_t x,
                                             std::uint32_t y,
                                             std::uint32_t z) noexcept {
  return {f32(x), f32(y), f32(z)};
}

// Cycles 5.2.1 CPU Combined pixels from svm_voronoi_matrix. Every cell feeds
// Geometry Normal through vector/scalar nodes, so all twenty values execute
// NODE_TEX_VORONOI in the final external SVM stream; none is host-folded.
constexpr std::array oracle_cases{
    OracleCase{1u, NODE_VORONOI_F1, NODE_VORONOI_EUCLIDEAN, false,
               OutputKind::distance,
               color(0x3dba5e40u, 0x3dba5e40u, 0x3dba5e40u)},
    OracleCase{1u, NODE_VORONOI_F2, NODE_VORONOI_CHEBYCHEV, true,
               OutputKind::w,
               color(0x3e2d03cbu, 0x3e2d03cbu, 0x3e2d03cbu)},
    OracleCase{1u, NODE_VORONOI_SMOOTH_F1, NODE_VORONOI_MINKOWSKI, false,
               OutputKind::color,
               color(0x3eb1ea2eu, 0x3f2b29adu, 0x3ffbca3bu)},
    OracleCase{1u, NODE_VORONOI_DISTANCE_TO_EDGE, NODE_VORONOI_EUCLIDEAN,
               true, OutputKind::distance,
               color(0x3d1b850cu, 0x3d1b850cu, 0x3d1b850cu)},
    OracleCase{1u, NODE_VORONOI_N_SPHERE_RADIUS, NODE_VORONOI_EUCLIDEAN,
               false, OutputKind::radius,
               color(0x3f000000u, 0x3f000000u, 0x3f000000u)},
    OracleCase{2u, NODE_VORONOI_F1, NODE_VORONOI_MANHATTAN, false,
               OutputKind::color,
               color(0x3d362d7au, 0x3ec48871u, 0x3ddb3abdu)},
    OracleCase{2u, NODE_VORONOI_F2, NODE_VORONOI_MINKOWSKI, false,
               OutputKind::position,
               color(0x3d6efa31u, 0x3e796ae6u, 0x00000000u)},
    OracleCase{2u, NODE_VORONOI_SMOOTH_F1, NODE_VORONOI_CHEBYCHEV, true,
               OutputKind::distance,
               color(0x3e7bc25du, 0x3e7bc25du, 0x3e7bc25du)},
    OracleCase{2u, NODE_VORONOI_DISTANCE_TO_EDGE, NODE_VORONOI_EUCLIDEAN,
               true, OutputKind::distance,
               color(0x3f2425b0u, 0x3f2425b0u, 0x3f2425b0u)},
    OracleCase{2u, NODE_VORONOI_N_SPHERE_RADIUS, NODE_VORONOI_EUCLIDEAN,
               false, OutputKind::radius,
               color(0x3ede0ea0u, 0x3ede0ea0u, 0x3ede0ea0u)},
    OracleCase{3u, NODE_VORONOI_F1, NODE_VORONOI_EUCLIDEAN, false,
               OutputKind::position,
               color(0x3da36004u, 0xbee3d43eu, 0x3f18a67fu)},
    OracleCase{3u, NODE_VORONOI_F2, NODE_VORONOI_MANHATTAN, true,
               OutputKind::distance,
               color(0x3e246549u, 0x3e246549u, 0x3e246549u)},
    OracleCase{3u, NODE_VORONOI_SMOOTH_F1, NODE_VORONOI_MINKOWSKI, false,
               OutputKind::color,
               color(0x3ef87935u, 0x3ecc8ba6u, 0x3ee078e5u)},
    OracleCase{3u, NODE_VORONOI_DISTANCE_TO_EDGE, NODE_VORONOI_EUCLIDEAN,
               false, OutputKind::distance,
               color(0x3d96b1eeu, 0x3d96b1eeu, 0x3d96b1eeu)},
    OracleCase{3u, NODE_VORONOI_N_SPHERE_RADIUS, NODE_VORONOI_EUCLIDEAN,
               true, OutputKind::radius,
               color(0x3e92582cu, 0x3e92582cu, 0x3e92582cu)},
    OracleCase{4u, NODE_VORONOI_F1, NODE_VORONOI_CHEBYCHEV, false,
               OutputKind::w,
               color(0x3e6fb724u, 0x3e6fb724u, 0x3e6fb724u)},
    OracleCase{4u, NODE_VORONOI_F2, NODE_VORONOI_EUCLIDEAN, true,
               OutputKind::color,
               color(0x3d54ab14u, 0x3f1dc54eu, 0x3ddec760u)},
    OracleCase{4u, NODE_VORONOI_SMOOTH_F1, NODE_VORONOI_MANHATTAN, true,
               OutputKind::position,
               color(0x3fb8c09au, 0xbf8c81ffu, 0x3fb73fb8u)},
    OracleCase{4u, NODE_VORONOI_DISTANCE_TO_EDGE, NODE_VORONOI_EUCLIDEAN,
               false, OutputKind::distance,
               color(0x3db284c0u, 0x3db284c0u, 0x3db284c0u)},
    OracleCase{4u, NODE_VORONOI_N_SPHERE_RADIUS, NODE_VORONOI_EUCLIDEAN,
               true, OutputKind::radius,
               color(0x3e8c0958u, 0x3e8c0958u, 0x3e8c0958u)},
};

template<typename T>
void append_payload(std::vector<std::uint32_t> &words, const T &payload) {
  const auto encoded =
      std::bit_cast<std::array<std::uint32_t,
                               sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] SVMStackOffset output_offset(OutputKind selected,
                                           OutputKind candidate,
                                           std::uint32_t offset) noexcept {
  return selected == candidate ? static_cast<SVMStackOffset>(offset)
                               : SVM_STACK_INVALID;
}

[[nodiscard]] SVMNodeTexVoronoi payload(const OracleCase &test) noexcept {
  return {
      .dimensions = test.dimensions,
      .feature = test.feature,
      .metric = test.metric,
      .w = input_float(static_cast<SVMStackOffset>(w_input_offset)),
      .scale = input_float(static_cast<SVMStackOffset>(scale_input_offset)),
      .detail = input_float(static_cast<SVMStackOffset>(detail_input_offset)),
      .roughness =
          input_float(static_cast<SVMStackOffset>(roughness_input_offset)),
      .lacunarity =
          input_float(static_cast<SVMStackOffset>(lacunarity_input_offset)),
      .smoothness =
          input_float(static_cast<SVMStackOffset>(smoothness_input_offset)),
      .exponent =
          input_float(static_cast<SVMStackOffset>(exponent_input_offset)),
      .randomness =
          input_float(static_cast<SVMStackOffset>(randomness_input_offset)),
      .normalize = static_cast<std::uint8_t>(test.normalize),
      .coord = static_cast<SVMStackOffset>(vector_offset),
      .distance_offset =
          output_offset(test.output, OutputKind::distance, distance_offset),
      .color_offset =
          output_offset(test.output, OutputKind::color, color_offset),
      .position_offset =
          output_offset(test.output, OutputKind::position, position_offset),
      .w_out_offset =
          output_offset(test.output, OutputKind::w, w_output_offset),
      .radius_offset =
          output_offset(test.output, OutputKind::radius, radius_offset),
      ._pad = {0u}};
}

[[nodiscard]] constexpr float select_parameter(
    std::size_t index, std::array<float, 4u> values) noexcept {
  return values[index % values.size()];
}

[[nodiscard]] std::array<luisa::float4, oracle_cases.size() * 3u>
oracle_inputs() noexcept {
  std::array<luisa::float4, oracle_cases.size() * 3u> result{};
  for (auto index = std::size_t{}; index < oracle_cases.size(); ++index) {
    // The external graph computes Vector = Geometry.Normal + this offset and
    // W = Geometry.Normal.z * authored_w. The matrix normal is exactly +Z.
    result[index * 3u + 0u] = {
        -0.73f + 0.097f * static_cast<float>(index),
        0.41f - 0.061f * static_cast<float>(index),
        -0.17f + 0.083f * static_cast<float>(index),
        -0.83f + 0.071f * static_cast<float>(index)};
    result[index * 3u + 1u] = {
        select_parameter(index, {-2.3f, 0.75f, 1.9f, 4.1f}),
        select_parameter(index, {0.0f, 0.375f, 2.625f, 5.25f}),
        select_parameter(index, {0.0f, 0.37f, 0.68f, 1.0f}),
        select_parameter(index, {2.0f, -1.25f, 0.73f, 2.4f})};
    result[index * 3u + 2u] = {
        select_parameter(index, {0.0f, 0.2f, 0.83f, 3.0f}),
        select_parameter(index, {0.5f, 1.3f, 2.0f, 3.7f}),
        select_parameter(index, {0.0f, 0.29f, 0.71f, 1.0f}),
        0.0f};
  }
  return result;
}

[[nodiscard]] auto voronoi_kernel(bool voronoi_extra_enabled) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::float4>, Buffer<luisa::float4>,
                  Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [voronoi_extra_enabled](
          BufferUInt words, BufferFloat4 inputs, BufferFloat4 color_distance,
          BufferFloat4 position_w, BufferFloat4 radius_sentinel_output,
          BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        const auto first = inputs.read(index * 3u);
        const auto second = inputs.read(index * 3u + 1u);
        const auto third = inputs.read(index * 3u + 2u);
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(stack, vector_offset, first.xyz());
        svm_detail::stack_store_float(stack, w_input_offset, first.w);
        svm_detail::stack_store_float(stack, scale_input_offset, second.x);
        svm_detail::stack_store_float(stack, detail_input_offset, second.y);
        svm_detail::stack_store_float(stack, roughness_input_offset, second.z);
        svm_detail::stack_store_float(stack, lacunarity_input_offset, second.w);
        svm_detail::stack_store_float(stack, smoothness_input_offset, third.x);
        svm_detail::stack_store_float(stack, exponent_input_offset, third.y);
        svm_detail::stack_store_float(stack, randomness_input_offset, third.z);
        svm_detail::stack_store_float(stack, distance_offset, distance_sentinel);
        svm_detail::stack_store_float3(stack, color_offset, color_sentinel);
        svm_detail::stack_store_float3(stack, position_offset,
                                       position_sentinel);
        svm_detail::stack_store_float(stack, w_output_offset, w_sentinel);
        svm_detail::stack_store_float(stack, radius_offset, radius_sentinel);

        UInt cursor_offset = index * voronoi_word_count;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::node_tex_voronoi(cursor, stack, voronoi_extra_enabled);
        color_distance.write(
            index,
            make_float4(svm_detail::stack_load_float3(stack, color_offset),
                        svm_detail::stack_load_float(stack, distance_offset)));
        position_w.write(
            index,
            make_float4(svm_detail::stack_load_float3(stack, position_offset),
                        svm_detail::stack_load_float(stack, w_output_offset)));
        radius_sentinel_output.write(
            index,
            make_float4(svm_detail::stack_load_float(stack, radius_offset),
                        0.0f, 0.0f, 0.0f));
        cursors.write(index, cursor_offset - index * voronoi_word_count);
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

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-4f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 2.0e-4f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool equal(luisa::float3 actual,
                         luisa::float3 expected) noexcept {
  return actual.x == expected.x && actual.y == expected.y &&
         actual.z == expected.z;
}

[[nodiscard]] bool test_direct_oracle(Device &device, Stream &stream,
                                      std::string_view backend) {
  std::vector<std::uint32_t> words;
  words.reserve(oracle_cases.size() * voronoi_word_count);
  for (const auto &test : oracle_cases) {
    append_payload(words, payload(test));
  }
  const auto inputs = oracle_inputs();
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto input_buffer = device.create_buffer<luisa::float4>(inputs.size());
  auto color_distance_buffer =
      device.create_buffer<luisa::float4>(oracle_cases.size());
  auto position_w_buffer =
      device.create_buffer<luisa::float4>(oracle_cases.size());
  auto radius_buffer =
      device.create_buffer<luisa::float4>(oracle_cases.size());
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(oracle_cases.size());
  auto shader = device.compile(
      voronoi_kernel(true),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, oracle_cases.size()> color_distance{};
  std::array<luisa::float4, oracle_cases.size()> position_w{};
  std::array<luisa::float4, oracle_cases.size()> radius{};
  std::array<std::uint32_t, oracle_cases.size()> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << input_buffer.copy_from(luisa::span{inputs})
         << shader(word_buffer, input_buffer, color_distance_buffer,
                   position_w_buffer, radius_buffer, cursor_buffer)
                .dispatch(oracle_cases.size())
         << color_distance_buffer.copy_to(luisa::span{color_distance})
         << position_w_buffer.copy_to(luisa::span{position_w})
         << radius_buffer.copy_to(luisa::span{radius})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  for (auto index = std::size_t{}; index < oracle_cases.size(); ++index) {
    const auto &test = oracle_cases[index];
    auto actual = luisa::float3{};
    switch (test.output) {
      case OutputKind::distance:
        actual = luisa::float3{color_distance[index].w};
        break;
      case OutputKind::color:
        actual = color_distance[index].xyz();
        break;
      case OutputKind::position:
        actual = position_w[index].xyz();
        break;
      case OutputKind::w:
        actual = luisa::float3{position_w[index].w};
        break;
      case OutputKind::radius:
        actual = luisa::float3{radius[index].x};
        break;
    }
    const auto untouched =
        (test.output == OutputKind::distance ||
         color_distance[index].w == distance_sentinel) &&
        (test.output == OutputKind::color ||
         equal(color_distance[index].xyz(), color_sentinel)) &&
        (test.output == OutputKind::position ||
         equal(position_w[index].xyz(), position_sentinel)) &&
        (test.output == OutputKind::w ||
         position_w[index].w == w_sentinel) &&
        (test.output == OutputKind::radius ||
         radius[index].x == radius_sentinel);
    if (!near(actual, test.expected) || !untouched ||
        cursors[index] != voronoi_word_count) {
      std::cerr << "Cycles Voronoi oracle mismatch on " << backend
                << ", case=" << index << ": got {" << actual.x << ", "
                << actual.y << ", " << actual.z << "}, expected {"
                << test.expected.x << ", " << test.expected.y << ", "
                << test.expected.z << "}, cursor=" << cursors[index]
                << ", untouched=" << untouched << '\n';
      return false;
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
        output.write(0u, make_float4(shader_data.closure_emission_background,
                                     result.final_offset.cast<float>()));
        status.write(0u, result.status);
      }};
}

[[nodiscard]] bool test_interpreter_dispatch(Device &device, Stream &stream,
                                             std::string_view backend) {
  // Complete Cycles 5.2.1 surface stream for `SVM Voronoi 00 1D F1`, with
  // only its leading global shader jump rebased to shader zero. Geometry
  // Normal supplies the runtime W path, proving real PC/opcode dispatch.
  static constexpr std::array words{
      0x00000001u, 0x00000004u, 0x00000038u, 0x00000038u,
      0x0000000bu, 0x00000001u, 0x00000000u, 0x00000054u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x0000ff00u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x0000ff01u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x00000302u, 0x0000002cu, 0x00000002u,
      0x7fc00003u, 0xbf547ae1u, 0x00000000u, 0x00000000u,
      0x00000015u, 0x0000000bu, 0x00000001u, 0x00000000u,
      0x00000041u, 0x00000001u, 0x00000000u, 0x00000000u,
      0x7fc00000u, 0xc0133333u, 0x00000000u, 0x00000000u,
      0x40000000u, 0x40a00000u, 0x3f000000u, 0x00000000u,
      0xff040100u, 0x00ffffffu, 0x0000000du, 0x00000000u,
      0x00000004u, 0x00000007u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  // The captured global SVM allocation retains three words of tail padding.
  // Cycles consumes NODE_END at word 56, so the semantic PC is 57 rather than
  // the allocation size (60).
  static constexpr auto cycles_stream_end_offset = 57u;
  std::array<bool, NODE_NUM> node_types{};
  node_types[NODE_END] = true;
  node_types[NODE_SHADER_JUMP] = true;
  node_types[NODE_GEOMETRY] = true;
  node_types[NODE_SEPARATE_VECTOR] = true;
  node_types[NODE_MATH] = true;
  node_types[NODE_ATTR] = true;
  node_types[NODE_TEX_VORONOI] = true;
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
  const auto expected = f32(0x3dba5e40u);
  if (!near(actual.x, expected) || !near(actual.y, expected) ||
      !near(actual.z, expected) ||
      actual.w != static_cast<float>(cycles_stream_end_offset) ||
      status != static_cast<std::uint32_t>(
                    device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Voronoi interpreter mismatch on " << backend
              << ": got {" << actual.x << ", " << actual.y << ", "
              << actual.z << "}, status=" << status
              << ", final_offset=" << actual.w << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto full_shape = module_shape(voronoi_kernel(true));
  const auto base_shape = module_shape(voronoi_kernel(false));
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Voronoi full XIR: instructions="
              << full_shape.instructions
              << ", callables=" << full_shape.callable_definitions
              << ", loops=" << full_shape.loops << '\n'
              << "Cycles SVM Voronoi base XIR: instructions="
              << base_shape.instructions
              << ", callables=" << base_shape.callable_definitions
              << ", loops=" << base_shape.loops << '\n';
  }
  // Four coordinate-domain callables preserve Cycles' runtime payload without
  // multiplying the AST by feature/metric/Normalize combinations. Dynamic
  // Detail must remain represented by loops, and removing VORONOI_EXTRA must
  // strictly shrink the module by deleting multidimensional X-FX.
  if (full_shape.callable_definitions != 4u ||
      base_shape.callable_definitions != 4u || full_shape.loops == 0u ||
      base_shape.loops == 0u ||
      base_shape.instructions >= full_shape.instructions) {
    std::cerr << "Cycles SVM Voronoi shape regression: full={callables="
              << full_shape.callable_definitions
              << ", loops=" << full_shape.loops
              << ", instructions=" << full_shape.instructions
              << "}, base={callables=" << base_shape.callable_definitions
              << ", loops=" << base_shape.loops
              << ", instructions=" << base_shape.instructions << "}\n";
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_direct_oracle(device, stream, backend) &&
                 test_interpreter_dispatch(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_checker.h>
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
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto vector_offset = std::uint32_t{0u};
constexpr auto color_offset = std::uint32_t{3u};
constexpr auto factor_offset = std::uint32_t{6u};
constexpr auto record_stride = std::uint32_t{19u};
constexpr auto direct_case_count = std::uint32_t{7u};

constexpr auto wave_word_count =
    static_cast<std::uint32_t>(sizeof(SVMNodeTexWave) / sizeof(std::uint32_t));
constexpr auto magic_word_count =
    static_cast<std::uint32_t>(sizeof(SVMNodeTexMagic) / sizeof(std::uint32_t));
constexpr auto checker_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeTexChecker) / sizeof(std::uint32_t));
constexpr auto brick_word_count =
    static_cast<std::uint32_t>(sizeof(SVMNodeTexBrick) / sizeof(std::uint32_t));
static_assert(wave_word_count == 11u);
static_assert(magic_word_count == 3u);
static_assert(checker_word_count == 8u);
static_assert(brick_word_count == record_stride);

template <typename T>
void append_record(std::vector<std::uint32_t> &words, const T &payload) {
  const auto encoded = std::bit_cast<
      std::array<std::uint32_t, sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
  words.resize(words.size() + record_stride - encoded.size(), 0u);
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 3.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 3.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] SVMNodeTexWave simple_wave_payload() noexcept {
  return {.wave_type = NODE_WAVE_BANDS,
          .bands_direction = NODE_WAVE_BANDS_DIRECTION_X,
          .rings_direction = NODE_WAVE_RINGS_DIRECTION_X,
          .profile = NODE_WAVE_PROFILE_SIN,
          .scale = input_float(-1.7f),
          .distortion = input_float(0.0f),
          .detail = input_float(2.0f),
          .dscale = input_float(1.0f),
          .droughness = input_float(0.5f),
          .phase = input_float(-0.83f),
          .co = static_cast<SVMStackOffset>(vector_offset),
          .color_offset = static_cast<SVMStackOffset>(color_offset),
          .fac_offset = static_cast<SVMStackOffset>(factor_offset),
          ._pad = {0u}};
}

[[nodiscard]] SVMNodeTexWave distorted_wave_payload() noexcept {
  return {.wave_type = NODE_WAVE_RINGS,
          .bands_direction = NODE_WAVE_BANDS_DIRECTION_X,
          .rings_direction = NODE_WAVE_RINGS_DIRECTION_SPHERICAL,
          .profile = NODE_WAVE_PROFILE_SIN,
          .scale = input_float(-7.5f),
          .distortion = input_float(1.5f),
          .detail = input_float(2.375f),
          .dscale = input_float(2.17f),
          .droughness = input_float(0.63f),
          .phase = input_float(1.1f),
          .co = static_cast<SVMStackOffset>(vector_offset),
          .color_offset = static_cast<SVMStackOffset>(color_offset),
          .fac_offset = static_cast<SVMStackOffset>(factor_offset),
          ._pad = {0u}};
}

[[nodiscard]] SVMNodeTexMagic magic_payload() noexcept {
  return {.scale = input_float(0.001f),
          .distortion = input_float(1.0f),
          .depth = 2u,
          .co = static_cast<SVMStackOffset>(vector_offset),
          .color_offset = static_cast<SVMStackOffset>(color_offset),
          .fac_offset = static_cast<SVMStackOffset>(factor_offset)};
}

[[nodiscard]] SVMNodeTexChecker checker_payload() noexcept {
  return {.color1 = input_float3(0.13f, 0.37f, 0.79f),
          .color2 = input_float3(0.83f, 0.61f, 0.17f),
          .scale = input_float(1.0f),
          .co = static_cast<SVMStackOffset>(vector_offset),
          .color_offset = static_cast<SVMStackOffset>(color_offset),
          .fac_offset = static_cast<SVMStackOffset>(factor_offset),
          ._pad = {0u}};
}

[[nodiscard]] SVMNodeTexBrick
brick_payload(SVMStackOffset color_out, SVMStackOffset factor_out) noexcept {
  return {.color1 = input_float3(0.78f, 0.08f, 0.03f),
          .color2 = input_float3(0.12f, 0.43f, 0.91f),
          .mortar = input_float3(0.07f, 0.21f, 0.13f),
          .scale = input_float(5.3f),
          .mortar_size = input_float(0.036f),
          .bias = input_float(-0.14f),
          .brick_width = input_float(0.53f),
          .row_height = input_float(0.19f),
          .mortar_smooth = input_float(0.017f),
          .offset_amount = 0.37f,
          .squash_amount = 0.72f,
          .offset_frequency = 3u,
          .squash_frequency = 2u,
          .co = static_cast<SVMStackOffset>(vector_offset),
          .color_offset = color_out,
          .fac_offset = factor_out,
          ._pad = {0u, 0u, 0u}};
}

[[nodiscard]] auto direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 points, BufferFloat4 output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(stack, vector_offset,
                                       points.read(index).xyz());
        svm_detail::stack_store_float3(stack, color_offset,
                                       make_float3(-92.0f, -93.0f, -94.0f));
        svm_detail::stack_store_float(stack, factor_offset, -91.0f);
        UInt cursor_offset = index * record_stride;
        const UInt begin = cursor_offset;
        svm_detail::Cursor cursor{words, cursor_offset};
        $if(index < 2u) { svm_detail::node_tex_wave(cursor, stack); }
        $elif(index == 2u) { svm_detail::node_tex_magic(cursor, stack); }
        $elif(index < 5u) { svm_detail::node_tex_checker(cursor, stack); }
        $else { svm_detail::node_tex_brick(cursor, stack); };
        output.write(
            index,
            make_float4(svm_detail::stack_load_float3(stack, color_offset),
                        svm_detail::stack_load_float(stack, factor_offset)));
        cursors.write(index, cursor_offset - begin);
      }};
}

[[nodiscard]] auto checker_formula_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<float>>{
      [](BufferFloat4 input, BufferFloat output) noexcept {
        const auto sample = input.read(dispatch_x());
        output.write(dispatch_x(),
                     psycles::luisa_backend::cycles_checker::evaluate(
                         sample.xyz() * sample.w));
      }};
}

[[nodiscard]] bool checker_formula_is_structural() {
  auto module = luisa::compute::xir::ast_to_xir_translate(
      checker_formula_kernel().function()->function(), {});
  auto add = std::size_t{};
  auto subtract = std::size_t{};
  auto multiply = std::size_t{};
  auto modulo = std::size_t{};
  auto floor_count = std::size_t{};
  auto isnan_count = std::size_t{};
  auto select_count = std::size_t{};
  for (auto *function : module->function_list()) {
    if (auto *definition = function->definition()) {
      definition->traverse_instructions(
          [&](luisa::compute::xir::Instruction *instruction) noexcept {
            if (!instruction->isa<luisa::compute::xir::ArithmeticInst>()) {
              return;
            }
            switch (static_cast<luisa::compute::xir::ArithmeticInst *>(
                        instruction)
                        ->op()) {
            case luisa::compute::xir::ArithmeticOp::BINARY_ADD:
              ++add;
              break;
            case luisa::compute::xir::ArithmeticOp::BINARY_SUB:
              ++subtract;
              break;
            case luisa::compute::xir::ArithmeticOp::BINARY_MUL:
              ++multiply;
              break;
            case luisa::compute::xir::ArithmeticOp::BINARY_MOD:
              ++modulo;
              break;
            case luisa::compute::xir::ArithmeticOp::FLOOR:
              ++floor_count;
              break;
            case luisa::compute::xir::ArithmeticOp::ISNAN:
              ++isnan_count;
              break;
            case luisa::compute::xir::ArithmeticOp::SELECT:
              ++select_count;
              break;
            default:
              break;
            }
          });
    }
  }
  const auto valid = add == 1u && subtract == 0u && multiply == 2u &&
                     modulo == 3u && floor_count == 3u && isnan_count == 0u &&
                     select_count == 2u;
  if (!valid) {
    std::cerr << "Cycles Checker operation-graph mismatch: add=" << add
              << ", sub=" << subtract << ", mul=" << multiply
              << ", mod=" << modulo << ", floor=" << floor_count
              << ", isnan=" << isnan_count << ", select=" << select_count
              << '\n';
  }
  return valid;
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

[[nodiscard]] bool test_direct_handlers(Device &device, Stream &stream,
                                        std::string_view backend) {
  std::vector<std::uint32_t> words;
  words.reserve(direct_case_count * record_stride);
  append_record(words, simple_wave_payload());
  append_record(words, distorted_wave_payload());
  append_record(words, magic_payload());
  append_record(words, checker_payload());
  append_record(words, checker_payload());
  append_record(words, brick_payload(static_cast<SVMStackOffset>(color_offset),
                                     SVM_STACK_INVALID));
  append_record(words,
                brick_payload(SVM_STACK_INVALID,
                              static_cast<SVMStackOffset>(factor_offset)));

  static constexpr std::array points{
      luisa::float4{0.173f, 0.0f, 1.375f, 0.0f},
      luisa::float4{0.512000024f, 0.0f, 1.083999991f, 0.0f},
      luisa::float4{1.0e20f, -0.375f, 0.8125f, 0.0f},
      luisa::float4{1.0f, 0.0f, 0.0f, 0.0f},
      luisa::float4{-1.0f, 0.0f, 0.0f, 0.0f},
      luisa::float4{0.471f, 0.379f, 0.0f, 0.0f},
      luisa::float4{0.811f, 0.541f, 0.0f, 0.0f}};
  static constexpr std::array expected_colors{
      luisa::float3{0.045264751f},
      luisa::float3{0.442690015f},
      luisa::float3{0.555838704f, 0.657886386f, 0.937048197f},
      luisa::float3{0.83f, 0.61f, 0.17f},
      luisa::float3{0.13f, 0.37f, 0.79f},
      luisa::float3{0.514933228f, 0.220565692f, 0.383422345f},
      luisa::float3{-92.0f, -93.0f, -94.0f}};
  static constexpr std::array expected_factors{
      0.045264751f,
      0.442690015f,
      (0.555838704f + 0.657886386f + 0.937048197f) * (1.0f / 3.0f),
      0.0f,
      1.0f,
      -91.0f,
      1.0f};
  static constexpr std::array expected_cursors{
      wave_word_count,    wave_word_count,    magic_word_count,
      checker_word_count, checker_word_count, brick_word_count,
      brick_word_count};

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto point_buffer = device.create_buffer<luisa::float4>(points.size());
  auto output_buffer = device.create_buffer<luisa::float4>(direct_case_count);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(direct_case_count);
  auto shader =
      device.compile(direct_kernel(), ShaderOption{.enable_cache = false,
                                                   .enable_fast_math = false});
  std::array<luisa::float4, direct_case_count> actual{};
  std::array<std::uint32_t, direct_case_count> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << point_buffer.copy_from(points.data())
         << shader(word_buffer, point_buffer, output_buffer, cursor_buffer)
                .dispatch(direct_case_count)
         << output_buffer.copy_to(actual.data())
         << cursor_buffer.copy_to(cursors.data()) << synchronize();

  auto valid = true;
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    const auto color =
        luisa::float3{actual[index].x, actual[index].y, actual[index].z};
    const auto case_valid = near(color, expected_colors[index]) &&
                            near(actual[index].w, expected_factors[index]) &&
                            cursors[index] == expected_cursors[index];
    valid &= case_valid;
    if (!case_valid) {
      std::cerr << "Cycles procedural handler mismatch on " << backend
                << ", case=" << index << ": got {" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ", "
                << actual[index].w << "}, cursor=" << cursors[index] << '\n';
    }
  }
  return valid;
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

template <std::size_t N, typename Shader>
[[nodiscard]] bool
run_interpreter_case(Device &device, Stream &stream, Shader &shader,
                     const std::array<std::uint32_t, N> &words,
                     luisa::float3 expected, std::string_view label,
                     std::string_view backend) {
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(1u);
  auto status_buffer = device.create_buffer<std::uint32_t>(1u);
  luisa::float4 actual{};
  std::uint32_t status{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer, status_buffer).dispatch(1u)
         << output_buffer.copy_to(&actual) << status_buffer.copy_to(&status)
         << synchronize();
  const auto actual_color = luisa::float3{actual.x, actual.y, actual.z};
  if (!near(actual_color, expected) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles " << label << " interpreter mismatch on " << backend
              << ": got {" << actual.x << ", " << actual.y << ", " << actual.z
              << "}, status=" << status << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool test_interpreter_dispatch(Device &device, Stream &stream,
                                             std::string_view backend) {
  // Complete shader-local streams captured from Cycles 5.2.1 and rebased to
  // shader zero. These retain Cycles' surrounding value, conversion, and
  // emission nodes so the test covers the real PC loop and opcode dispatch.
  static constexpr std::array wave_words{
      0x00000001u, 0x00000004u, 0x0000001du, 0x0000001eu, 0x00000013u,
      0x00000000u, 0x3e3126e9u, 0x00000000u, 0x3fb00000u, 0x00000043u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0xbfd9999au,
      0x00000000u, 0x40000000u, 0x3f800000u, 0x3f000000u, 0xbf547ae1u,
      0x00ff0300u, 0x00000007u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  static constexpr std::array magic_words{
      0x00000001u, 0x00000004u, 0x00000031u, 0x00000032u, 0x0000000bu,
      0x00000000u, 0x00000000u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x00000300u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x0000ff01u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x0000ff02u, 0x0000002cu, 0x00000000u, 0x60ad78ecu,
      0x7fc00003u, 0x00000000u, 0x00000000u, 0x00000056u, 0x7fc00000u,
      0x00000100u, 0x00000056u, 0xbec00000u, 0x00000101u, 0x00000056u,
      0x3f500000u, 0x00000102u, 0x00000044u, 0x3a83126fu, 0x3f800000u,
      0xff040102u, 0x00000007u, 0x7fc00004u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  static constexpr std::array checker_words{
      0x00000001u, 0x00000004u, 0x00000026u, 0x00000027u, 0x00000013u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000045u,
      0x3e051eb8u, 0x3ebd70a4u, 0x3f4a3d71u, 0x3f547ae1u, 0x3f1c28f6u,
      0x3e2e147bu, 0x3f800000u, 0x00060300u, 0x00000052u, 0x00000000u,
      0x7fc00003u, 0x00000000u, 0x00000000u, 0x00ff0100u, 0x00000053u,
      0x00000000u, 0x7fc00000u, 0x7fc00001u, 0x7fc00006u, 0x00000002u,
      0x00000007u, 0x7fc00002u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr std::array brick_words{
      0x00000001u, 0x00000004u, 0x00000030u, 0x00000031u, 0x00000015u,
      0x0000000bu, 0x00000000u, 0x00000000u, 0x00000046u, 0x3f47ae14u,
      0x3da3d70au, 0x3cf5c28fu, 0x3df5c28fu, 0x3edc28f6u, 0x3f68f5c3u,
      0x3d8f5c29u, 0x3e570a3du, 0x3e051eb8u, 0x40a9999au, 0x3d1374bcu,
      0xbe0f5c29u, 0x3f07ae14u, 0x3e428f5cu, 0x3c8b4396u, 0x3ebd70a4u,
      0x3f3851ecu, 0x03000203u, 0x00000006u, 0x00000052u, 0x00000000u,
      0x7fc00003u, 0x00000000u, 0x00000000u, 0x00ff0100u, 0x00000053u,
      0x00000000u, 0x7fc00000u, 0x7fc00001u, 0x7fc00006u, 0x00000002u,
      0x00000007u, 0x7fc00002u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};

  std::array<bool, NODE_NUM> node_types{};
  for (const auto type :
       {NODE_END, NODE_SHADER_JUMP, NODE_CLOSURE_EMISSION, NODE_EMISSION_WEIGHT,
        NODE_GEOMETRY, NODE_CONVERT, NODE_VALUE_V, NODE_ATTR, NODE_MATH,
        NODE_TEX_WAVE, NODE_TEX_MAGIC, NODE_TEX_CHECKER, NODE_TEX_BRICK,
        NODE_SEPARATE_COLOR, NODE_COMBINE_COLOR, NODE_SEPARATE_VECTOR,
        NODE_COMBINE_VECTOR}) {
    node_types[type] = true;
  }
  auto shader = device.compile(
      interpreter_kernel(node_types),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  return run_interpreter_case(device, stream, shader, wave_words,
                              luisa::float3{0.045264751f}, "Wave", backend) &&
         run_interpreter_case(
             device, stream, shader, magic_words,
             luisa::float3{0.555838704f, 0.657886386f, 0.937048197f}, "Magic",
             backend) &&
         run_interpreter_case(device, stream, shader, checker_words,
                              luisa::float3{0.83f, 0.61f, 0.0f}, "Checker",
                              backend) &&
         run_interpreter_case(device, stream, shader, brick_words,
                              luisa::float3{0.07f, 0.21f, 1.0f}, "Brick",
                              backend);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  if (!checker_formula_is_structural()) {
    return EXIT_FAILURE;
  }
  const auto shape = module_shape(direct_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles procedural SVM XIR: instructions="
              << shape.instructions
              << ", callables=" << shape.callable_definitions << '\n';
  }
  // The three reusable callables are the shared Cycles noise primitives used
  // by distorted Wave. A large jump here means a procedural family was
  // accidentally expanded into every call site again.
  if (shape.instructions > 9000u || shape.callable_definitions != 3u) {
    std::cerr << "Cycles procedural SVM shape regression: instructions="
              << shape.instructions
              << ", callables=" << shape.callable_definitions << '\n';
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

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

constexpr auto coordinates_offset = std::uint32_t{0u};
constexpr auto orientation_3d_offset = std::uint32_t{4u};
constexpr auto scale_offset = std::uint32_t{8u};
constexpr auto frequency_offset = std::uint32_t{9u};
constexpr auto anisotropy_offset = std::uint32_t{10u};
constexpr auto orientation_2d_offset = std::uint32_t{11u};
constexpr auto value_offset = std::uint32_t{16u};
constexpr auto phase_offset = std::uint32_t{17u};
constexpr auto intensity_offset = std::uint32_t{18u};
constexpr auto gabor_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeTexGabor) / sizeof(std::uint32_t));
static_assert(gabor_word_count == 9u);

constexpr auto value_sentinel = -91.0f;
constexpr auto phase_sentinel = -92.0f;
constexpr auto intensity_sentinel = -93.0f;

enum class OutputKind : std::uint8_t {
  value,
  phase,
  intensity,
};

struct OracleCase {
  NodeGaborType type;
  OutputKind output;
  luisa::float3 coordinates;
  float scale;
  float frequency;
  float anisotropy;
  float orientation_2d;
  luisa::float3 orientation_3d;
  float expected;
};

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

// External Cycles 5.2.1 CPU Combined pixels from svm_gabor_matrix. Geometry
// Normal keeps every coordinate path shader-varying, so all twelve materials
// retain NODE_TEX_GABOR in the final global SVM stream.
constexpr std::array oracle_cases{
    OracleCase{NODE_GABOR_TYPE_2D, OutputKind::value,
               {-0.73f, 0.41f, -0.17f}, 2.3f, 0.0f, 0.0f, -1.2f,
               {1.0f, 2.0f, 0.3f}, f32(0x3effb850u)},
    OracleCase{NODE_GABOR_TYPE_2D, OutputKind::phase,
               {0.27f, -0.91f, 1.36f}, -1.7f, 0.73f, 0.35f, 0.0f,
               {0.2f, -0.4f, 1.3f}, f32(0x3e5d71b6u)},
    OracleCase{NODE_GABOR_TYPE_2D, OutputKind::intensity,
               {1.19f, 0.05f, 0.58f}, 0.75f, 2.0f, 1.0f, 2.4f,
               {-2.0f, 0.5f, 0.7f}, f32(0x3e7826c3u)},
    OracleCase{NODE_GABOR_TYPE_2D, OutputKind::value,
               {-1.37f, 1.11f, 1.22f}, 4.1f, 5.7f, 0.68f, 5.2f,
               {0.0f, 1.0f, 0.0f}, f32(0x3f1c69e1u)},
    OracleCase{NODE_GABOR_TYPE_2D, OutputKind::phase,
               {0.0f, 0.0f, 1.0f}, 0.0f, 1.3f, 0.12f, -3.0f,
               {1.0f, 0.0f, 0.0f}, f32(0x3e8c1bfeu)},
    OracleCase{NODE_GABOR_TYPE_2D, OutputKind::intensity,
               {2.03f, -1.77f, 1.9f}, 1.9f, 11.0f, 0.92f, 0.79f,
               {0.3f, 0.8f, -0.1f}, f32(0x3e92d3bfu)},
    OracleCase{NODE_GABOR_TYPE_3D, OutputKind::value,
               {-0.83f, 0.29f, -0.41f}, 2.3f, 0.0f, 0.0f, 0.0f,
               {1.0f, 2.0f, 0.3f}, f32(0x3f0011bdu)},
    OracleCase{NODE_GABOR_TYPE_3D, OutputKind::phase,
               {0.47f, -1.21f, 1.66f}, -1.7f, 0.73f, 0.35f, 1.1f,
               {0.2f, -0.4f, 1.3f}, f32(0x3f53f108u)},
    OracleCase{NODE_GABOR_TYPE_3D, OutputKind::intensity,
               {1.49f, 0.15f, 0.28f}, 0.75f, 2.0f, 1.0f, -2.7f,
               {-2.0f, 0.5f, 0.7f}, f32(0x3e925316u)},
    OracleCase{NODE_GABOR_TYPE_3D, OutputKind::value,
               {-1.07f, 1.31f, 1.52f}, 4.1f, 5.7f, 0.68f, 2.2f,
               {0.4f, 1.1f, -0.6f}, f32(0x3ef98ab6u)},
    OracleCase{NODE_GABOR_TYPE_3D, OutputKind::phase,
               {0.2f, -0.3f, 1.4f}, 0.0f, 1.3f, 0.12f, -0.2f,
               {1.0f, 0.0f, 0.0f}, f32(0x3f2cc1a9u)},
    OracleCase{NODE_GABOR_TYPE_3D, OutputKind::intensity,
               {2.33f, -1.47f, 2.2f}, 1.9f, 11.0f, 0.92f, 0.39f,
               {0.3f, 0.8f, -0.1f}, f32(0x3e529c7eu)},
};

template <typename T>
void append_payload(std::vector<std::uint32_t> &words, const T &payload) {
  const auto encoded = std::bit_cast<
      std::array<std::uint32_t, sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] SVMStackOffset output_offset(OutputKind selected,
                                           OutputKind candidate,
                                           std::uint32_t offset) noexcept {
  return selected == candidate ? static_cast<SVMStackOffset>(offset)
                               : SVM_STACK_INVALID;
}

[[nodiscard]] SVMNodeTexGabor payload(const OracleCase &test) noexcept {
  return {
      .gabor_type = test.type,
      .orientation_3d = input_float3(
          static_cast<SVMStackOffset>(orientation_3d_offset)),
      .scale = input_float(static_cast<SVMStackOffset>(scale_offset)),
      .frequency = input_float(static_cast<SVMStackOffset>(frequency_offset)),
      .anisotropy =
          input_float(static_cast<SVMStackOffset>(anisotropy_offset)),
      .orientation_2d =
          input_float(static_cast<SVMStackOffset>(orientation_2d_offset)),
      .coordinates = static_cast<SVMStackOffset>(coordinates_offset),
      .value_offset =
          output_offset(test.output, OutputKind::value, value_offset),
      .phase_offset =
          output_offset(test.output, OutputKind::phase, phase_offset),
      .intensity_offset = output_offset(
          test.output, OutputKind::intensity, intensity_offset)};
}

[[nodiscard]] auto gabor_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::float4>, Buffer<luisa::float4>,
                  Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 coordinates_scale,
         BufferFloat4 orientation_frequency,
         BufferFloat4 anisotropy_orientation, BufferFloat4 output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        const auto first = coordinates_scale.read(index);
        const auto second = orientation_frequency.read(index);
        const auto third = anisotropy_orientation.read(index);
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(stack, coordinates_offset, first.xyz());
        svm_detail::stack_store_float3(stack, orientation_3d_offset,
                                       second.xyz());
        svm_detail::stack_store_float(stack, scale_offset, first.w);
        svm_detail::stack_store_float(stack, frequency_offset, second.w);
        svm_detail::stack_store_float(stack, anisotropy_offset, third.x);
        svm_detail::stack_store_float(stack, orientation_2d_offset, third.y);
        svm_detail::stack_store_float(stack, value_offset, value_sentinel);
        svm_detail::stack_store_float(stack, phase_offset, phase_sentinel);
        svm_detail::stack_store_float(stack, intensity_offset,
                                      intensity_sentinel);
        UInt cursor_offset = index * gabor_word_count;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::node_tex_gabor(cursor, stack);
        output.write(
            index,
            make_float4(
                svm_detail::stack_load_float(stack, value_offset),
                svm_detail::stack_load_float(stack, phase_offset),
                svm_detail::stack_load_float(stack, intensity_offset), 0.0f));
        cursors.write(index, cursor_offset - index * gabor_word_count);
      }};
}

struct ModuleShape {
  std::size_t instructions{};
  std::size_t callable_definitions{};
  std::size_t loops{};
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

[[nodiscard]] bool test_direct_oracle(Device &device, Stream &stream,
                                      std::string_view backend) {
  std::vector<std::uint32_t> words;
  words.reserve(oracle_cases.size() * gabor_word_count);
  std::array<luisa::float4, oracle_cases.size()> coordinates_scale{};
  std::array<luisa::float4, oracle_cases.size()> orientation_frequency{};
  std::array<luisa::float4, oracle_cases.size()> anisotropy_orientation{};
  for (auto index = std::size_t{}; index < oracle_cases.size(); ++index) {
    const auto &test = oracle_cases[index];
    append_payload(words, payload(test));
    coordinates_scale[index] =
        luisa::float4{test.coordinates.x, test.coordinates.y,
                      test.coordinates.z, test.scale};
    orientation_frequency[index] =
        luisa::float4{test.orientation_3d.x, test.orientation_3d.y,
                      test.orientation_3d.z, test.frequency};
    anisotropy_orientation[index] =
        luisa::float4{test.anisotropy, test.orientation_2d, 0.0f, 0.0f};
  }
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto coordinates_buffer =
      device.create_buffer<luisa::float4>(oracle_cases.size());
  auto orientation_buffer =
      device.create_buffer<luisa::float4>(oracle_cases.size());
  auto anisotropy_buffer =
      device.create_buffer<luisa::float4>(oracle_cases.size());
  auto output_buffer =
      device.create_buffer<luisa::float4>(oracle_cases.size());
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(oracle_cases.size());
  auto shader = device.compile(
      gabor_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, oracle_cases.size()> actual{};
  std::array<std::uint32_t, oracle_cases.size()> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << coordinates_buffer.copy_from(luisa::span{coordinates_scale})
         << orientation_buffer.copy_from(luisa::span{orientation_frequency})
         << anisotropy_buffer.copy_from(luisa::span{anisotropy_orientation})
         << shader(word_buffer, coordinates_buffer, orientation_buffer,
                   anisotropy_buffer, output_buffer, cursor_buffer)
                .dispatch(oracle_cases.size())
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  for (auto index = std::size_t{}; index < oracle_cases.size(); ++index) {
    const auto &test = oracle_cases[index];
    const auto selected =
        test.output == OutputKind::value
            ? actual[index].x
            : test.output == OutputKind::phase ? actual[index].y
                                               : actual[index].z;
    const auto untouched =
        (test.output == OutputKind::value ||
         actual[index].x == value_sentinel) &&
        (test.output == OutputKind::phase ||
         actual[index].y == phase_sentinel) &&
        (test.output == OutputKind::intensity ||
         actual[index].z == intensity_sentinel);
    if (!near(selected, test.expected) || !untouched ||
        cursors[index] != gabor_word_count) {
      std::cerr << "Cycles Gabor oracle mismatch on " << backend
                << ", case=" << index << ": output={" << actual[index].x
                << ", " << actual[index].y << ", " << actual[index].z
                << "}, expected=" << test.expected
                << ", cursor=" << cursors[index]
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
  // Complete Cycles 5.2.1 shader-local stream for Gabor case 0, with only
  // global jump offsets rebased. It includes Geometry Normal, Vector Add,
  // Gabor, scalar-to-color conversion, Emission, and NODE_END.
  static constexpr std::array words{
      0x00000001u, 0x00000004u, 0x00000029u, 0x0000002au,
      0x0000000bu, 0x00000001u, 0x00000000u, 0x0000002du,
      0x00000000u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0xbf3ae148u, 0x3ed1eb85u, 0xbf95c28fu, 0x00000000u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x000003ffu,
      0x00000042u, 0x00000000u, 0x3fb504f3u, 0x3fb504f3u,
      0x00000000u, 0x40133333u, 0x00000000u, 0x00000000u,
      0xbf99999au, 0xffff0003u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  std::array<bool, NODE_NUM> node_types{};
  node_types[NODE_END] = true;
  node_types[NODE_SHADER_JUMP] = true;
  node_types[NODE_GEOMETRY] = true;
  node_types[NODE_VECTOR_MATH] = true;
  node_types[NODE_TEX_GABOR] = true;
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
  const auto expected = f32(0x3effb850u);
  if (!near(actual.x, expected) || !near(actual.y, expected) ||
      !near(actual.z, expected) || actual.w != 41.0f ||
      status != static_cast<std::uint32_t>(
                    device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Gabor interpreter mismatch on " << backend
              << ": got {" << actual.x << ", " << actual.y << ", "
              << actual.z << "}, final_offset=" << actual.w
              << ", status=" << status << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(gabor_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Gabor XIR: instructions=" << shape.instructions
              << ", callables=" << shape.callable_definitions
              << ", loops=" << shape.loops << '\n';
  }
  // Cycles' fixed neighborhoods are seven nested device loops: 2D has two
  // cell loops plus its impulse loop, and 3D has three plus its impulse loop.
  // Requiring them prevents accidental host-AST unrolling.
  if (shape.callable_definitions != 0u || shape.loops != 7u ||
      shape.instructions > 7000u) {
    std::cerr << "Cycles SVM Gabor shape regression: callables="
              << shape.callable_definitions << ", loops=" << shape.loops
              << ", instructions=" << shape.instructions << '\n';
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

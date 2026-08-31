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

constexpr auto input_color_offset = std::uint32_t{0u};
constexpr auto input_factor_offset = std::uint32_t{3u};
constexpr auto output_color_offset = std::uint32_t{4u};
constexpr auto curve_header_words = static_cast<std::uint32_t>(
    sizeof(SVMNodeCurves) / sizeof(std::uint32_t));
static_assert(curve_header_words == 8u);

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

template<typename T>
void append_payload(std::vector<std::uint32_t> &words, const T &payload) {
  const auto encoded = std::bit_cast<
      std::array<std::uint32_t, sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void append_color(std::vector<std::uint32_t> &words,
                  const luisa::float4 &color) {
  words.emplace_back(std::bit_cast<std::uint32_t>(color.x));
  words.emplace_back(std::bit_cast<std::uint32_t>(color.y));
  words.emplace_back(std::bit_cast<std::uint32_t>(color.z));
  words.emplace_back(std::bit_cast<std::uint32_t>(color.w));
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-6f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

struct CurveCase {
  std::vector<luisa::float4> table;
  luisa::float3 color{};
  float factor{};
  float min_x{};
  float max_x{1.0f};
  bool extrapolate{};
  bool stack_inputs{};
};

[[nodiscard]] float lookup(const CurveCase &test, float coordinate,
                           std::size_t channel) {
  const auto value = [&](std::size_t element) {
    return test.table[element][channel];
  };
  const auto last = test.table.size() - 1u;
  if ((coordinate < 0.0f || coordinate > 1.0f) && test.extrapolate) {
    if (coordinate < 0.0f) {
      const auto first = value(0u);
      return first + (first - value(1u)) * (-coordinate) *
                         static_cast<float>(last);
    }
    const auto final_value = value(last);
    return final_value + (final_value - value(last - 1u)) *
                             (coordinate - 1.0f) *
                             static_cast<float>(last);
  }
  const auto scaled =
      std::clamp(coordinate, 0.0f, 1.0f) * static_cast<float>(last);
  const auto index = std::clamp(static_cast<int>(scaled), 0,
                                static_cast<int>(last));
  const auto t = scaled - static_cast<float>(index);
  auto result = value(static_cast<std::size_t>(index));
  if (t > 0.0f) {
    result = (1.0f - t) * result +
             t * value(static_cast<std::size_t>(index) + 1u);
  }
  return result;
}

[[nodiscard]] luisa::float3 reference(const CurveCase &test) {
  const auto range = test.max_x - test.min_x;
  const auto relative = (test.color - test.min_x) / range;
  const auto mapped = luisa::float3{
      lookup(test, relative.x, 0u), lookup(test, relative.y, 1u),
      lookup(test, relative.z, 2u)};
  return (1.0f - test.factor) * test.color + test.factor * mapped;
}

[[nodiscard]] std::vector<CurveCase> oracle_cases() {
  const std::vector<luisa::float4> two_entry{
      {0.1f, 0.7f, 0.2f, 1.0f}, {0.9f, 0.3f, 0.8f, 1.0f}};
  const std::vector<luisa::float4> three_entry{
      {0.2f, 0.8f, 0.1f, 1.0f}, {0.7f, 0.25f, 0.9f, 1.0f},
      {0.4f, 0.6f, 0.3f, 1.0f}};
  // Five exact rows (indices 0, 64, 128, 192, 256) from the external
  // Cycles 5.2.1 rgb_curve_dynamic SVM table.
  const std::vector<luisa::float4> external{
      {f32(0x3ea2b0a9u), f32(0x3f18367au), f32(0x3ee91e84u), 1.0f},
      {f32(0x3f464da4u), f32(0x3e25e891u), f32(0x3f69da02u), 1.0f},
      {f32(0x3f4be563u), f32(0x3e355d6au), f32(0x3f662261u), 1.0f},
      {f32(0x3f37c974u), f32(0x3e181eb4u), f32(0x3f6b4cb8u), 1.0f},
      {f32(0x3ed7f794u), f32(0x3ecf2e62u), f32(0x3f2a165au), 1.0f}};
  return {
      {two_entry, {0.25f, 0.5f, 0.75f}, 0.4f, 0.0f, 1.0f, false, false},
      {two_entry, {-0.5f, 1.5f, 0.5f}, 1.0f, 0.0f, 1.0f, false, false},
      {two_entry, {-0.5f, 1.5f, 0.5f}, 1.0f, 0.0f, 1.0f, true, false},
      {three_entry, {-0.5f, 0.25f, 1.5f}, 0.65f, -0.25f, 1.25f, true,
       false},
      {three_entry, {0.2f, 0.6f, 0.9f}, 0.0f, 0.0f, 1.0f, true, false},
      {three_entry, {0.2f, 0.6f, 0.9f}, 1.4f, 0.0f, 1.0f, true, false},
      {three_entry, {0.35f, 0.55f, 0.8f}, 0.37f, -0.25f, 1.3f, false,
       true},
      {external, {0.25f, 0.5f, 0.75f}, 1.0f, 0.0f, 1.0f, true, false},
      {external, {0.0f, 1.0f, 0.5f}, 1.0f, 0.0f, 1.0f, true, false}};
}

[[nodiscard]] auto curve_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<std::uint32_t>,
                  Buffer<luisa::float4>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferUInt offsets, BufferFloat4 inputs,
         BufferFloat4 output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        const UInt begin = offsets.read(index);
        UInt cursor_offset = begin;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::Stack stack;
        const Float4 input = inputs.read(index);
        svm_detail::stack_store_float3(stack, input_color_offset, input.xyz());
        svm_detail::stack_store_float(stack, input_factor_offset, input.w);
        svm_detail::node_curves(cursor, stack);
        output.write(index,
                     make_float4(svm_detail::stack_load_float3(
                                     stack, output_color_offset),
                                 1.0f));
        cursors.write(index, cursor_offset - begin);
      }};
}

struct ModuleShape {
  std::size_t instructions{};
  std::size_t loops{};
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
          [&](const luisa::compute::xir::Instruction *instruction) noexcept {
            ++result.instructions;
            result.loops +=
                instruction->isa<luisa::compute::xir::LoopInst>() ? 1u : 0u;
          });
    }
  }
  return result;
}

[[nodiscard]] bool test_direct_handler(Device &device, Stream &stream,
                                       std::string_view backend) {
  const auto cases = oracle_cases();
  std::vector<std::uint32_t> words;
  std::vector<std::uint32_t> offsets;
  std::vector<luisa::float4> inputs;
  std::vector<luisa::float3> expected;
  for (const auto &test : cases) {
    offsets.emplace_back(static_cast<std::uint32_t>(words.size()));
    append_payload(
        words,
        SVMNodeCurves{
            .color = test.stack_inputs
                         ? input_float3(static_cast<SVMStackOffset>(
                               input_color_offset))
                         : input_float3(test.color.x, test.color.y,
                                        test.color.z),
            .fac = test.stack_inputs
                       ? input_float(static_cast<SVMStackOffset>(
                             input_factor_offset))
                       : input_float(test.factor),
            .min_x = test.min_x,
            .max_x = test.max_x,
            .table_size = static_cast<std::uint32_t>(test.table.size()),
            .extrapolate = static_cast<std::uint8_t>(test.extrapolate),
            .out_offset = static_cast<SVMStackOffset>(output_color_offset),
            ._pad = {0u, 0u}});
    for (const auto &value : test.table) {
      append_color(words, value);
    }
    inputs.emplace_back(
        luisa::float4{test.color.x, test.color.y, test.color.z, test.factor});
    expected.emplace_back(reference(test));
  }

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto offset_buffer = device.create_buffer<std::uint32_t>(offsets.size());
  auto input_buffer = device.create_buffer<luisa::float4>(inputs.size());
  auto output_buffer = device.create_buffer<luisa::float4>(cases.size());
  auto cursor_buffer = device.create_buffer<std::uint32_t>(cases.size());
  auto shader = device.compile(
      curve_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::vector<luisa::float4> actual(cases.size());
  std::vector<std::uint32_t> cursors(cases.size());
  stream << word_buffer.copy_from(luisa::span{words})
         << offset_buffer.copy_from(luisa::span{offsets})
         << input_buffer.copy_from(luisa::span{inputs})
         << shader(word_buffer, offset_buffer, input_buffer, output_buffer,
                   cursor_buffer)
                .dispatch(static_cast<std::uint32_t>(cases.size()))
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  for (auto index = std::size_t{}; index < cases.size(); ++index) {
    const auto expected_cursor = curve_header_words +
                                 static_cast<std::uint32_t>(
                                     cases[index].table.size() * 4u);
    if (!near(actual[index].x, expected[index].x) ||
        !near(actual[index].y, expected[index].y) ||
        !near(actual[index].z, expected[index].z) ||
        cursors[index] != expected_cursor) {
      std::cerr << "Cycles RGB Curves direct case " << index << " failed on "
                << backend << ": value=(" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z
                << "), expected=(" << expected[index].x << ", "
                << expected[index].y << ", " << expected[index].z
                << "), cursor=" << cursors[index]
                << ", expected cursor=" << expected_cursor << '\n';
      return false;
    }
  }

  static constexpr auto external_case = std::size_t{7u};
  const auto external_expected = luisa::float3{
      f32(0x3f464da4u), f32(0x3e355d6au), f32(0x3f6b4cb8u)};
  if (!near(actual[external_case].x, external_expected.x, 2.0e-7f) ||
      !near(actual[external_case].y, external_expected.y, 2.0e-7f) ||
      !near(actual[external_case].z, external_expected.z, 2.0e-7f)) {
    std::cerr << "Cycles RGB Curves external table oracle differs on "
              << backend << '\n';
    return false;
  }

  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_RGB_CURVE_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open RGB Curves capture path " << capture_path
                << '\n';
      return false;
    }
    capture << "case\tr\tg\tb\tcursor\n" << std::setprecision(9);
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      capture << index << '\t' << actual[index].x << '\t' << actual[index].y
              << '\t' << actual[index].z << '\t' << cursors[index] << '\n';
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
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP,
      SVMNodeShaderJump{.offset_surface = 0u,
                        .offset_volume = 0u,
                        .offset_displacement = 0u});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static constexpr std::array table{
      packed_float4{0.1f, 0.7f, 0.2f, 1.0f},
      packed_float4{0.9f, 0.3f, 0.8f, 1.0f}};
  static_cast<void>(builder.add_node(
      NODE_CURVES,
      SVMNodeCurves{.color = input_float3(0.25f, 0.5f, 0.75f),
                    .fac = input_float(0.4f),
                    .min_x = 0.0f,
                    .max_x = 1.0f,
                    .table_size = static_cast<std::uint32_t>(table.size()),
                    .extrapolate = 0u,
                    .out_offset = 0u,
                    ._pad = {0u, 0u}}));
  for (const auto &value : table) {
    builder.add_node_data(value);
  }
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color = input_float3(0u),
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
  node_types[NODE_END] = true;
  node_types[NODE_SHADER_JUMP] = true;
  node_types[NODE_CURVES] = true;
  node_types[NODE_EMISSION_WEIGHT] = true;
  node_types[NODE_CLOSURE_EMISSION] = true;
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
  const CurveCase test{{luisa::float4{table[0u].x, table[0u].y, table[0u].z,
                                     table[0u].w},
                        luisa::float4{table[1u].x, table[1u].y, table[1u].z,
                                     table[1u].w}},
                       {0.25f, 0.5f, 0.75f},
                       0.4f,
                       0.0f,
                       1.0f,
                       false,
                       false};
  const auto expected = reference(test);
  if (!near(actual.x, expected.x) || !near(actual.y, expected.y) ||
      !near(actual.z, expected.z) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles RGB Curves interpreter dispatch mismatch on "
              << backend << ": status=" << status << ", value=(" << actual.x
              << ", " << actual.y << ", " << actual.z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(curve_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM RGB Curves XIR: instructions="
              << shape.instructions << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 3500u || shape.loops != 0u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles SVM RGB Curves shape regression: instructions="
              << shape.instructions << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
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

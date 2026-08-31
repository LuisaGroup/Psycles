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
#include <limits>
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

constexpr auto color_offset = std::uint32_t{0u};
constexpr auto alpha_offset = std::uint32_t{4u};
constexpr auto ramp_header_words =
    static_cast<std::uint32_t>(sizeof(SVMNodeRGBRamp) / sizeof(std::uint32_t));
static_assert(ramp_header_words == 3u);

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

template <typename T>
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

struct RampCase {
  std::vector<luisa::float4> table;
  float factor{};
  bool interpolate{};
  bool color_live{true};
  bool alpha_live{true};
};

[[nodiscard]] luisa::float4 reference(const RampCase &test) {
  const auto last = test.table.size() - 1u;
  // Cycles' SVMCompiler filters every non-finite unlinked default to zero
  // before encoding it because NaN payloads are reserved for stack offsets.
  const auto factor = std::isfinite(test.factor) ? test.factor : 0.0f;
  const auto scaled = std::clamp(factor, 0.0f, 1.0f) * static_cast<float>(last);
  const auto index =
      std::clamp(static_cast<int>(scaled), 0, static_cast<int>(last));
  const auto t = scaled - static_cast<float>(index);
  auto value = test.table[static_cast<std::size_t>(index)];
  if (test.interpolate && t > 0.0f) {
    const auto next = test.table[static_cast<std::size_t>(index) + 1u];
    value = (1.0f - t) * value + t * next;
  }
  return value;
}

[[nodiscard]] std::vector<RampCase> oracle_cases() {
  return {
      // RGB LINEAR and RGB CARDINAL adjacent samples from the external
      // Cycles 5.2.1 color_ramp_modes probe.
      {{{f32(0x3f7721cbu), f32(0x3da149b6u), f32(0x3e2cd605u),
         f32(0x3e370416u)},
        {f32(0x3f770e98u), f32(0x3da83ce9u), f32(0x3e2c1c6bu),
         f32(0x3e3a2a7du)}},
       f32(0x3e937480u),
       true},
      {{{1.0f, f32(0x3f0c3051u), 0.0f, f32(0x3f16785bu)},
        {1.0f, f32(0x3f0d307bu), 0.0f, f32(0x3f175e8fu)}},
       f32(0x3eed9200u),
       true},
      // HSV FAR samples exercise four-channel interpolation without
      // introducing any mode-specific device branch.
      {{{f32(0x3e7c21bdu), f32(0x3ce815dbu), f32(0x3f7374ffu),
         f32(0x3ea8aed8u)},
        {f32(0x3e64cc06u), f32(0x3ce87f24u), f32(0x3f7361cbu),
         f32(0x3eaa420bu)}},
       f32(0x3f73b680u),
       true},
      // Constant mode must never read or mix the next element.
      {{{0.91f, 0.11f, 0.07f, 0.32f}, {0.13f, 0.77f, 0.31f, 0.61f}},
       0.87f,
       false},
      // Clamp and exact-boundary cases, with a different table size to prove
      // that neither its cardinality nor dispatch width enters the shader AST.
      {{{0.2f, 0.3f, 0.4f, 0.5f},
        {0.6f, 0.7f, 0.8f, 0.9f},
        {1.0f, 0.9f, 0.8f, 0.7f}},
       -0.25f,
       true},
      {{{0.2f, 0.3f, 0.4f, 0.5f},
        {0.6f, 0.7f, 0.8f, 0.9f},
        {1.0f, 0.9f, 0.8f, 0.7f}},
       1.25f,
       true},
      {{{0.2f, 0.3f, 0.4f, 0.5f},
        {0.6f, 0.7f, 0.8f, 0.9f},
        {1.0f, 0.9f, 0.8f, 0.7f}},
       0.5f,
       true},
      // Cycles filters non-finite unlinked defaults before SVM encoding; this
      // also protects the NaN payload range reserved for stack offsets.
      {{{0.2f, 0.3f, 0.4f, 0.5f},
        {0.6f, 0.7f, 0.8f, 0.9f},
        {1.0f, 0.9f, 0.8f, 0.7f}},
       -std::numeric_limits<float>::infinity(),
       true},
      {{{0.2f, 0.3f, 0.4f, 0.5f},
        {0.6f, 0.7f, 0.8f, 0.9f},
        {1.0f, 0.9f, 0.8f, 0.7f}},
       std::numeric_limits<float>::infinity(),
       true},
      // Independently invalid output offsets preserve the other result.
      {{{0.1f, 0.2f, 0.3f, 0.4f}, {0.9f, 0.8f, 0.7f, 0.6f}},
       0.25f,
       true,
       false,
       true},
      {{{0.1f, 0.2f, 0.3f, 0.4f}, {0.9f, 0.8f, 0.7f, 0.6f}},
       0.75f,
       true,
       true,
       false},
      {{{0.1f, 0.2f, 0.3f, 0.4f}, {0.9f, 0.8f, 0.7f, 0.6f}},
       0.5f,
       true,
       false,
       false}};
}

[[nodiscard]] auto ramp_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<std::uint32_t>,
                  Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferUInt offsets, BufferFloat4 output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        const UInt begin = offsets.read(index);
        UInt cursor_offset = begin;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(stack, color_offset,
                                       make_float3(-91.0f, -92.0f, -93.0f));
        svm_detail::stack_store_float(stack, alpha_offset, -94.0f);
        svm_detail::node_rgb_ramp(cursor, stack);
        output.write(
            index,
            make_float4(svm_detail::stack_load_float3(stack, color_offset),
                        svm_detail::stack_load_float(stack, alpha_offset)));
        cursors.write(index, cursor_offset - begin);
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

[[nodiscard]] bool test_direct_handler(Device &device, Stream &stream,
                                       std::string_view backend) {
  const auto cases = oracle_cases();
  std::vector<std::uint32_t> words;
  std::vector<std::uint32_t> offsets;
  std::vector<luisa::float4> expected;
  for (const auto &test : cases) {
    offsets.emplace_back(static_cast<std::uint32_t>(words.size()));
    append_payload(
        words, SVMNodeRGBRamp{
                   .table_size = static_cast<std::uint32_t>(test.table.size()),
                   .fac = input_float(test.factor),
                   .interpolate = static_cast<std::uint8_t>(test.interpolate),
                   .color_offset = static_cast<SVMStackOffset>(
                       test.color_live ? color_offset : SVM_STACK_INVALID),
                   .alpha_offset = static_cast<SVMStackOffset>(
                       test.alpha_live ? alpha_offset : SVM_STACK_INVALID),
                   ._pad = {0u}});
    for (const auto &value : test.table) {
      append_color(words, value);
    }
    auto result = reference(test);
    if (!test.color_live) {
      result.x = -91.0f;
      result.y = -92.0f;
      result.z = -93.0f;
    }
    if (!test.alpha_live) {
      result.w = -94.0f;
    }
    expected.emplace_back(result);
  }

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto offset_buffer = device.create_buffer<std::uint32_t>(offsets.size());
  auto output_buffer = device.create_buffer<luisa::float4>(cases.size());
  auto cursor_buffer = device.create_buffer<std::uint32_t>(cases.size());
  auto shader =
      device.compile(ramp_kernel(), ShaderOption{.enable_cache = false,
                                                 .enable_fast_math = false});
  std::vector<luisa::float4> actual(cases.size());
  std::vector<std::uint32_t> cursors(cases.size());
  stream << word_buffer.copy_from(luisa::span{words})
         << offset_buffer.copy_from(luisa::span{offsets})
         << shader(word_buffer, offset_buffer, output_buffer, cursor_buffer)
                .dispatch(static_cast<std::uint32_t>(cases.size()))
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  for (auto index = std::size_t{}; index < cases.size(); ++index) {
    const auto expected_cursor =
        ramp_header_words +
        static_cast<std::uint32_t>(cases[index].table.size() * 4u);
    if (!near(actual[index].x, expected[index].x) ||
        !near(actual[index].y, expected[index].y) ||
        !near(actual[index].z, expected[index].z) ||
        !near(actual[index].w, expected[index].w) ||
        cursors[index] != expected_cursor) {
      std::cerr << "Cycles RGB Ramp direct case " << index << " failed on "
                << backend << ": value=(" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ", "
                << actual[index].w << "), cursor=" << cursors[index]
                << ", expected cursor=" << expected_cursor << '\n';
      return false;
    }
  }

  static constexpr std::array external_rgb{
      luisa::float3{f32(0x3f771c44u), f32(0x3da34a1fu), f32(0x3e2ca091u)},
      luisa::float3{1.0f, f32(0x3f0ca767u), 0.0f},
      luisa::float3{f32(0x3e65eab0u), f32(0x3ce879f6u), f32(0x3f7362b7u)}};
  for (auto index = std::size_t{}; index < external_rgb.size(); ++index) {
    if (!near(actual[index].x, external_rgb[index].x, 2.0e-5f) ||
        !near(actual[index].y, external_rgb[index].y, 2.0e-5f) ||
        !near(actual[index].z, external_rgb[index].z, 2.0e-5f)) {
      std::cerr << "Cycles RGB Ramp external oracle " << index
                << " differs materially on " << backend << '\n';
      return false;
    }
  }

  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_RGB_RAMP_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open RGB Ramp capture path " << capture_path
                << '\n';
      return false;
    }
    capture << "case\tr\tg\tb\ta\tcursor\n" << std::setprecision(9);
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      capture << index << '\t' << actual[index].x << '\t' << actual[index].y
              << '\t' << actual[index].z << '\t' << actual[index].w << '\t'
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
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0u,
                                          .offset_volume = 0u,
                                          .offset_displacement = 0u});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static constexpr std::array table{packed_float4{0.1f, 0.2f, 0.3f, 0.4f},
                                    packed_float4{0.9f, 0.8f, 0.7f, 0.6f}};
  static_cast<void>(builder.add_node(
      NODE_RGB_RAMP,
      SVMNodeRGBRamp{.table_size = static_cast<std::uint32_t>(table.size()),
                     .fac = input_float(0.25f),
                     .interpolate = 1u,
                     .color_offset = 0u,
                     .alpha_offset = SVM_STACK_INVALID,
                     ._pad = {0u}}));
  for (const auto &value : table) {
    builder.add_node_data(value);
  }
  static_cast<void>(
      builder.add_node(NODE_EMISSION_WEIGHT,
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
  node_types[NODE_RGB_RAMP] = true;
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
  const auto expected = luisa::float3{0.3f, 0.35f, 0.4f};
  if (!near(actual.x, expected.x) || !near(actual.y, expected.y) ||
      !near(actual.z, expected.z) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles RGB Ramp interpreter dispatch mismatch on " << backend
              << ": status=" << status << ", value=(" << actual.x << ", "
              << actual.y << ", " << actual.z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(ramp_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM RGB Ramp XIR: instructions=" << shape.instructions
              << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 1100u || shape.loops != 0u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles SVM RGB Ramp shape regression: instructions="
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

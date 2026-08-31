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

constexpr auto vector_input_offset = std::uint32_t{0u};
constexpr auto vector_factor_offset = std::uint32_t{3u};
constexpr auto vector_output_offset = std::uint32_t{4u};
constexpr auto float_input_offset = std::uint32_t{0u};
constexpr auto float_factor_offset = std::uint32_t{1u};
constexpr auto float_output_offset = std::uint32_t{2u};
constexpr auto vector_header_words =
    static_cast<std::uint32_t>(sizeof(SVMNodeCurves) / sizeof(std::uint32_t));
constexpr auto float_header_words = static_cast<std::uint32_t>(
    sizeof(SVMNodeFloatCurve) / sizeof(std::uint32_t));
static_assert(vector_header_words == 8u);
static_assert(float_header_words == 6u);

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

struct VectorCase {
  std::vector<luisa::float4> table;
  luisa::float3 value{};
  float factor{};
  float min_x{};
  float max_x{1.0f};
  bool extrapolate{};
  bool stack_inputs{};
};

struct FloatCase {
  std::vector<float> table;
  float value{};
  float factor{};
  float min_x{};
  float max_x{1.0f};
  bool extrapolate{};
  bool stack_inputs{};
};

template <typename Value>
[[nodiscard]] Value lookup(std::span<const Value> table, float coordinate,
                           bool extrapolate) {
  const auto last = table.size() - 1u;
  if ((coordinate < 0.0f || coordinate > 1.0f) && extrapolate) {
    if (coordinate < 0.0f) {
      return table.front() + (table.front() - table[1u]) * (-coordinate) *
                                 static_cast<float>(last);
    }
    return table.back() + (table.back() - table[last - 1u]) *
                              (coordinate - 1.0f) * static_cast<float>(last);
  }
  const auto scaled =
      std::clamp(coordinate, 0.0f, 1.0f) * static_cast<float>(last);
  const auto index =
      std::clamp(static_cast<int>(scaled), 0, static_cast<int>(last));
  const auto t = scaled - static_cast<float>(index);
  auto result = table[static_cast<std::size_t>(index)];
  if (t > 0.0f) {
    result =
        (1.0f - t) * result + t * table[static_cast<std::size_t>(index) + 1u];
  }
  return result;
}

[[nodiscard]] luisa::float3 reference(const VectorCase &test) {
  const auto range = test.max_x - test.min_x;
  const auto relative = (test.value - test.min_x) / range;
  std::vector<float> red;
  std::vector<float> green;
  std::vector<float> blue;
  red.reserve(test.table.size());
  green.reserve(test.table.size());
  blue.reserve(test.table.size());
  for (const auto &entry : test.table) {
    red.emplace_back(entry.x);
    green.emplace_back(entry.y);
    blue.emplace_back(entry.z);
  }
  const auto mapped =
      luisa::float3{lookup<float>(red, relative.x, test.extrapolate),
                    lookup<float>(green, relative.y, test.extrapolate),
                    lookup<float>(blue, relative.z, test.extrapolate)};
  return (1.0f - test.factor) * test.value + test.factor * mapped;
}

[[nodiscard]] float reference(const FloatCase &test) {
  const auto relative = (test.value - test.min_x) / (test.max_x - test.min_x);
  const auto mapped = lookup<float>(test.table, relative, test.extrapolate);
  return (1.0f - test.factor) * test.value + test.factor * mapped;
}

[[nodiscard]] std::vector<VectorCase> vector_cases() {
  const std::vector<luisa::float4> two_entry{{0.1f, 0.7f, 0.2f, 1.0f},
                                             {0.9f, 0.3f, 0.8f, 1.0f}};
  // Cycles 5.2.1 vector_curve_matrix rows 0, 64, 128, 192, 256.
  const std::vector<luisa::float4> external{
      {f32(0x3df5c28fu), f32(0x3f6147aeu), f32(0x3e75c28fu), 1.0f},
      {f32(0x3f540e2cu), f32(0x3e3bb925u), f32(0x3f66d4a5u), 1.0f},
      {f32(0x3f17f9deu), f32(0x3f0aae90u), f32(0x3f4bfc0eu), 1.0f},
      {f32(0x3e6ec68eu), f32(0x3f4780f0u), f32(0x3de3b8ebu), 1.0f},
      {f32(0x3f68defau), f32(0x3e23d6feu), f32(0x3f383affu), 1.0f}};
  return {{two_entry, {0.25f, 0.5f, 0.75f}, 0.4f, 0.0f, 1.0f, false, false},
          {two_entry, {-0.5f, 1.5f, 0.5f}, 1.0f, 0.0f, 1.0f, false, false},
          {two_entry, {-0.5f, 1.5f, 0.5f}, 1.0f, 0.0f, 1.0f, true, false},
          {two_entry, {0.25f, 0.5f, 0.75f}, 0.37f, -0.25f, 1.25f, true, true},
          {external, {0.25f, 0.5f, 0.75f}, 1.0f, 0.0f, 1.0f, true, false},
          {external, {0.0f, 1.0f, 0.5f}, 1.0f, 0.0f, 1.0f, true, false}};
}

[[nodiscard]] std::vector<FloatCase> float_cases() {
  const std::vector<float> two_entry{0.1f, 0.9f};
  const std::vector<float> external{f32(0x3da3d70au), f32(0x3f638f51u),
                                    f32(0x3e8c20f8u), f32(0x3f1fd781u),
                                    f32(0x3e9eb850u)};
  return {{two_entry, 0.25f, 0.4f, 0.0f, 1.0f, false, false},
          {two_entry, -0.5f, 1.0f, 0.0f, 1.0f, false, false},
          {two_entry, -0.5f, 1.0f, 0.0f, 1.0f, true, false},
          {two_entry, 1.5f, 0.65f, -0.25f, 1.25f, true, true},
          {two_entry, 0.6f, 0.0f, 0.0f, 1.0f, true, false},
          {two_entry, 0.6f, 1.4f, 0.0f, 1.0f, true, false},
          {external, 0.25f, 1.0f, 0.0f, 1.0f, true, false},
          {external, 0.5f, 1.0f, 0.0f, 1.0f, true, false},
          {external, 0.75f, 1.0f, 0.0f, 1.0f, true, false}};
}

[[nodiscard]] auto vector_kernel() {
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
        svm_detail::stack_store_float3(stack, vector_input_offset, input.xyz());
        svm_detail::stack_store_float(stack, vector_factor_offset, input.w);
        svm_detail::node_curves(cursor, stack);
        output.write(index, make_float4(svm_detail::stack_load_float3(
                                            stack, vector_output_offset),
                                        1.0f));
        cursors.write(index, cursor_offset - begin);
      }};
}

[[nodiscard]] auto float_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<std::uint32_t>,
                  Buffer<luisa::float2>, Buffer<float>, Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferUInt offsets, BufferFloat2 inputs,
         BufferFloat output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        const UInt begin = offsets.read(index);
        UInt cursor_offset = begin;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::Stack stack;
        const Float2 input = inputs.read(index);
        svm_detail::stack_store_float(stack, float_input_offset, input.x);
        svm_detail::stack_store_float(stack, float_factor_offset, input.y);
        svm_detail::node_float_curve(cursor, stack);
        output.write(index,
                     svm_detail::stack_load_float(stack, float_output_offset));
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

[[nodiscard]] bool test_vector_handler(Device &device, Stream &stream,
                                       std::string_view backend,
                                       std::ofstream *capture) {
  const auto cases = vector_cases();
  std::vector<std::uint32_t> words;
  std::vector<std::uint32_t> offsets;
  std::vector<luisa::float4> inputs;
  std::vector<luisa::float3> expected;
  for (const auto &test : cases) {
    offsets.emplace_back(static_cast<std::uint32_t>(words.size()));
    append_payload(
        words,
        SVMNodeCurves{
            .color =
                test.stack_inputs
                    ? input_float3(
                          static_cast<SVMStackOffset>(vector_input_offset))
                    : input_float3(test.value.x, test.value.y, test.value.z),
            .fac = test.stack_inputs ? input_float(static_cast<SVMStackOffset>(
                                           vector_factor_offset))
                                     : input_float(test.factor),
            .min_x = test.min_x,
            .max_x = test.max_x,
            .table_size = static_cast<std::uint32_t>(test.table.size()),
            .extrapolate = static_cast<std::uint8_t>(test.extrapolate),
            .out_offset = static_cast<SVMStackOffset>(vector_output_offset),
            ._pad = {0u, 0u}});
    for (const auto &value : test.table) {
      append_color(words, value);
    }
    inputs.emplace_back(
        luisa::float4{test.value.x, test.value.y, test.value.z, test.factor});
    expected.emplace_back(reference(test));
  }

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto offset_buffer = device.create_buffer<std::uint32_t>(offsets.size());
  auto input_buffer = device.create_buffer<luisa::float4>(inputs.size());
  auto output_buffer = device.create_buffer<luisa::float4>(cases.size());
  auto cursor_buffer = device.create_buffer<std::uint32_t>(cases.size());
  auto shader =
      device.compile(vector_kernel(), ShaderOption{.enable_cache = false,
                                                   .enable_fast_math = false});
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
    const auto expected_cursor =
        vector_header_words +
        static_cast<std::uint32_t>(cases[index].table.size() * 4u);
    if (!near(actual[index].x, expected[index].x) ||
        !near(actual[index].y, expected[index].y) ||
        !near(actual[index].z, expected[index].z) ||
        cursors[index] != expected_cursor) {
      std::cerr << "Cycles Vector Curves case " << index << " failed on "
                << backend << ": value=(" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z
                << "), cursor=" << cursors[index] << '\n';
      return false;
    }
    if (capture != nullptr) {
      *capture << "vector\t" << index << '\t' << actual[index].x << '\t'
               << actual[index].y << '\t' << actual[index].z << '\t'
               << cursors[index] << '\n';
    }
  }
  return true;
}

[[nodiscard]] bool test_float_handler(Device &device, Stream &stream,
                                      std::string_view backend,
                                      std::ofstream *capture) {
  const auto cases = float_cases();
  std::vector<std::uint32_t> words;
  std::vector<std::uint32_t> offsets;
  std::vector<luisa::float2> inputs;
  std::vector<float> expected;
  for (const auto &test : cases) {
    offsets.emplace_back(static_cast<std::uint32_t>(words.size()));
    append_payload(
        words,
        SVMNodeFloatCurve{
            .fac = test.stack_inputs ? input_float(static_cast<SVMStackOffset>(
                                           float_factor_offset))
                                     : input_float(test.factor),
            .value_in = test.stack_inputs
                            ? input_float(static_cast<SVMStackOffset>(
                                  float_input_offset))
                            : input_float(test.value),
            .min_x = test.min_x,
            .max_x = test.max_x,
            .table_size = static_cast<std::uint32_t>(test.table.size()),
            .extrapolate = static_cast<std::uint8_t>(test.extrapolate),
            .out_offset = static_cast<SVMStackOffset>(float_output_offset),
            ._pad = {0u, 0u}});
    for (const auto value : test.table) {
      words.emplace_back(std::bit_cast<std::uint32_t>(value));
    }
    inputs.emplace_back(luisa::float2{test.value, test.factor});
    expected.emplace_back(reference(test));
  }

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto offset_buffer = device.create_buffer<std::uint32_t>(offsets.size());
  auto input_buffer = device.create_buffer<luisa::float2>(inputs.size());
  auto output_buffer = device.create_buffer<float>(cases.size());
  auto cursor_buffer = device.create_buffer<std::uint32_t>(cases.size());
  auto shader =
      device.compile(float_kernel(), ShaderOption{.enable_cache = false,
                                                  .enable_fast_math = false});
  std::vector<float> actual(cases.size());
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
    const auto expected_cursor =
        float_header_words +
        static_cast<std::uint32_t>(cases[index].table.size());
    if (!near(actual[index], expected[index]) ||
        cursors[index] != expected_cursor) {
      std::cerr << "Cycles Float Curve case " << index << " failed on "
                << backend << ": value=" << actual[index]
                << ", expected=" << expected[index]
                << ", cursor=" << cursors[index] << '\n';
      return false;
    }
    if (capture != nullptr) {
      *capture << "float\t" << index << '\t' << actual[index] << "\t0\t0\t"
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
  static constexpr std::array table{0.1f, 0.9f};
  static_cast<void>(builder.add_node(
      NODE_FLOAT_CURVE,
      SVMNodeFloatCurve{.fac = input_float(0.4f),
                        .value_in = input_float(0.25f),
                        .min_x = 0.0f,
                        .max_x = 1.0f,
                        .table_size = static_cast<std::uint32_t>(table.size()),
                        .extrapolate = 0u,
                        .out_offset = 0u,
                        ._pad = {0u, 0u}}));
  for (const auto value : table) {
    builder.add_node_data_float(value);
  }
  for (auto component = std::uint8_t{}; component < 3u; ++component) {
    static_cast<void>(builder.add_node(
        NODE_COMBINE_VECTOR,
        SVMNodeCombineVector{.in = input_float(static_cast<SVMStackOffset>(0u)),
                             .vector_index = component,
                             .out_offset = 3u,
                             ._pad = {0u, 0u}}));
  }
  static_cast<void>(
      builder.add_node(NODE_EMISSION_WEIGHT,
                       SVMNodeEmissionWeight{.color = input_float3(3u),
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
  node_types[NODE_FLOAT_CURVE] = true;
  node_types[NODE_COMBINE_VECTOR] = true;
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
  const FloatCase test{
      {table[0u], table[1u]}, 0.25f, 0.4f, 0.0f, 1.0f, false, false};
  const auto expected = reference(test);
  if (!near(actual.x, expected) || !near(actual.y, expected) ||
      !near(actual.z, expected) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Float Curve interpreter mismatch on " << backend
              << ": status=" << status << ", value=(" << actual.x << ", "
              << actual.y << ", " << actual.z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto vector_shape = module_shape(vector_kernel());
  const auto float_shape = module_shape(float_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Vector Curves XIR: instructions="
              << vector_shape.instructions << ", loops=" << vector_shape.loops
              << ", callables=" << vector_shape.callable_definitions << '\n'
              << "Cycles SVM Float Curve XIR: instructions="
              << float_shape.instructions << ", loops=" << float_shape.loops
              << ", callables=" << float_shape.callable_definitions << '\n';
  }
  if (vector_shape.instructions > 3500u || vector_shape.loops != 0u ||
      vector_shape.callable_definitions != 0u ||
      float_shape.instructions > 1250u || float_shape.loops != 0u ||
      float_shape.callable_definitions != 0u) {
    std::cerr << "Cycles curve-family XIR shape regression: vector="
              << vector_shape.instructions << '/' << vector_shape.loops << '/'
              << vector_shape.callable_definitions
              << ", float=" << float_shape.instructions << '/'
              << float_shape.loops << '/' << float_shape.callable_definitions
              << '\n';
    return EXIT_FAILURE;
  }

  std::ofstream capture;
  std::ofstream *capture_stream = nullptr;
  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_CURVE_FAMILY_CAPTURE")) {
    capture.open(capture_path);
    if (!capture) {
      std::cerr << "Could not open curve-family capture path " << capture_path
                << '\n';
      return EXIT_FAILURE;
    }
    capture << "family\tcase\tx\ty\tz\tcursor\n" << std::setprecision(9);
    capture_stream = &capture;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_vector_handler(device, stream, backend, capture_stream) &&
                 test_float_handler(device, stream, backend, capture_stream) &&
                 test_interpreter_dispatch(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

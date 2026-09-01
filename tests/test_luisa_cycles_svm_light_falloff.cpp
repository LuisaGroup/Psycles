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
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto external_case_count = std::uint32_t{10u};
constexpr auto distant_case_count = std::uint32_t{3u};
constexpr auto case_count = external_case_count + distant_case_count;
constexpr auto payload_word_count = std::uint32_t{4u};

[[nodiscard]] constexpr std::uint32_t float_bits(float value) noexcept {
  return std::bit_cast<std::uint32_t>(value);
}

constexpr auto external_payloads =
    std::array<std::uint32_t, external_case_count * payload_word_count>{
        0x00000000u, 0x41000000u, 0x00000000u, 0x00000000u,
        0x00000001u, 0x40600000u, 0x40000000u, 0x00000000u,
        0x00000002u, 0x3e800000u, 0x40800000u, 0x00000000u,
        0x00000000u, 0x40c00000u, 0x3fc00000u, 0x00000000u,
        0x00000001u, 0x40c00000u, 0x3fc00000u, 0x00000001u,
        0x00000002u, 0x40c00000u, 0x3fc00000u, 0x00000002u,
        0x00000000u, 0x7fc00000u, 0x7fc00001u, 0x00000002u,
        0x00000001u, 0x7fc00000u, 0x7fc00001u, 0x00000003u,
        0x00000002u, 0x7fc00000u, 0x7fc00001u, 0x00000004u,
        0x00000002u, 0x40000000u, 0xbf800000u, 0x00000000u};

constexpr auto payloads = [] {
  std::array<std::uint32_t, case_count * payload_word_count> result{};
  std::copy(external_payloads.begin(), external_payloads.end(), result.begin());
  for (auto index = std::uint32_t{}; index < distant_case_count; ++index) {
    const auto source = index * payload_word_count;
    const auto destination =
        (external_case_count + index) * payload_word_count;
    for (auto word = std::uint32_t{}; word < payload_word_count; ++word) {
      result[destination + word] = external_payloads[source + word];
    }
  }
  return result;
}();

constexpr auto ray_lengths = std::array<float, case_count>{
    4.0f,
    2.0f,
    3.0f,
    2.0f,
    2.0f,
    2.0f,
    2.5f,
    2.5f,
    2.5f,
    3.0f,
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
};

constexpr auto expected = std::array<float, case_count>{
    8.0f,
    14.0f / 3.0f,
    81.0f / 52.0f,
    48.0f / 11.0f,
    96.0f / 11.0f,
    192.0f / 11.0f,
    100.0f / 37.0f,
    250.0f / 37.0f,
    625.0f / 37.0f,
    18.0f,
    8.0f,
    3.5f,
    0.25f,
};

[[nodiscard]] device_svm::ShaderData shader_data(Float ray_length) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
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
          ray_length,
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
}

[[nodiscard]] auto direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<float>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat lengths, BufferFloat output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        svm_detail::Stack stack;
        svm_detail::stack_store_float(stack, 0u, 4.0f);
        svm_detail::stack_store_float(stack, 1u, 3.0f);
        UInt cursor_offset = index * payload_word_count;
        const UInt begin = cursor_offset;
        svm_detail::Cursor cursor{words, cursor_offset};
        auto state = shader_data(lengths.read(index));
        svm_detail::node_light_falloff(cursor, stack, state);
        const auto output_offset =
            words.read(begin + 3u) & static_cast<std::uint32_t>(0xffu);
        output.write(index,
                     svm_detail::stack_load_float(stack, output_offset));
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

[[nodiscard]] bool distant_short_circuit_is_structural() {
  auto module = luisa::compute::xir::ast_to_xir_translate(
      direct_kernel().function()->function(), {});
  for (auto *function : module->function_list()) {
    auto *definition = function->definition();
    if (definition == nullptr) {
      continue;
    }
    std::vector<luisa::compute::xir::IfInst *> branches;
    std::vector<luisa::compute::xir::ArithmeticInst *> arithmetic;
    definition->traverse_instructions(
        [&](luisa::compute::xir::Instruction *instruction) noexcept {
          if (instruction->isa<luisa::compute::xir::IfInst>()) {
            branches.emplace_back(
                static_cast<luisa::compute::xir::IfInst *>(instruction));
            return;
          }
          if (!instruction->isa<luisa::compute::xir::ArithmeticInst>()) {
            return;
          }
          auto *candidate =
              static_cast<luisa::compute::xir::ArithmeticInst *>(instruction);
          if (candidate->type() == Type::of<float>() &&
              (candidate->op() ==
                   luisa::compute::xir::ArithmeticOp::BINARY_MUL ||
               candidate->op() ==
                   luisa::compute::xir::ArithmeticOp::BINARY_DIV)) {
            arithmetic.emplace_back(candidate);
          }
        });
    if (arithmetic.size() < 4u) {
      return false;
    }
    const auto dominance = luisa::compute::xir::compute_dom_tree(function);
    if (std::ranges::any_of(branches, [&](auto *branch) {
          return std::ranges::all_of(arithmetic, [&](auto *candidate) {
            return dominance.dominates(branch->true_block(),
                                       candidate->parent_block());
          });
        })) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool near(float actual, float expected_value,
                        float tolerance = 2.0e-6f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected_value) <=
             tolerance * std::max(1.0f, std::abs(expected_value));
}

[[nodiscard]] bool test_direct_handler(Device &device, Stream &stream,
                                       std::string_view backend) {
  auto word_buffer = device.create_buffer<std::uint32_t>(payloads.size());
  auto length_buffer = device.create_buffer<float>(ray_lengths.size());
  auto output_buffer = device.create_buffer<float>(case_count);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(case_count);
  auto shader =
      device.compile(direct_kernel(), ShaderOption{.enable_cache = false,
                                                   .enable_fast_math = false});
  std::array<float, case_count> actual{};
  std::array<std::uint32_t, case_count> cursors{};
  stream << word_buffer.copy_from(payloads.data())
         << length_buffer.copy_from(ray_lengths.data())
         << shader(word_buffer, length_buffer, output_buffer, cursor_buffer)
                .dispatch(case_count)
         << output_buffer.copy_to(actual.data())
         << cursor_buffer.copy_to(cursors.data()) << synchronize();

  auto valid = true;
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    valid &= near(actual[index], expected[index]) &&
             cursors[index] == payload_word_count;
  }
  if (!valid) {
    std::cerr << "Cycles Light Falloff direct handler mismatch on " << backend
              << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << index << ": actual=" << actual[index]
                << ", expected=" << expected[index]
                << ", cursor=" << cursors[index] << '\n';
    }
    return false;
  }
  if (const auto *capture_path = std::getenv(
          "PSYCLES_CYCLES_SVM_LIGHT_FALLOFF_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Light Falloff capture path "
                << capture_path << '\n';
      return false;
    }
    capture << "case\tvalue\tcursor\n" << std::hex << std::setfill('0');
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      capture << std::dec << index << std::hex << '\t' << std::setw(8)
              << float_bits(actual[index]) << std::dec << '\t'
              << cursors[index] << '\n';
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
        const UInt index = dispatch_x();
        const Float ray_length = select(
            2.0f, std::numeric_limits<float>::max(), index == 1u);
        auto state = shader_data(ray_length);
        const auto identity = make_float4x4(1.0f);
        const device_svm::TransformState transforms{identity, identity, identity,
                                                    identity};
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        const psycles::test_support::DefaultCyclesSvmKernelGlobals
            kernel_globals;
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
            device_svm::kernel_feature_node_emission, node_types_used,
            transforms, state, path_state, result);
        output.write(index,
                     make_float4(state.closure_emission_background,
                                 result.closure_weight.x));
        status.write(index, result.status);
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
      NODE_LIGHT_FALLOFF,
      SVMNodeLightFalloff{.falloff_type = NODE_LIGHT_FALLOFF_CONSTANT,
                          .strength = input_float(2.0f),
                          .smooth = input_float(4.0f),
                          .out_offset = 0u,
                          ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color = input_float3(1.0f, 0.5f, 0.25f),
                            .strength = input_float(SVMStackOffset{0u})}));
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
  auto output_buffer = device.create_buffer<luisa::float4>(2u);
  auto status_buffer = device.create_buffer<std::uint32_t>(2u);
  auto shader = device.compile(
      interpreter_kernel(builder.node_types_used()),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, 2u> actual{};
  std::array<std::uint32_t, 2u> status{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer, status_buffer).dispatch(2u)
         << output_buffer.copy_to(actual.data())
         << status_buffer.copy_to(status.data()) << synchronize();

  const auto ended = static_cast<std::uint32_t>(
      device_svm::EvaluationStatus::ended);
  const auto valid =
      near(actual[0u].x, 4.0f) && near(actual[0u].y, 2.0f) &&
      near(actual[0u].z, 1.0f) && near(actual[0u].w, 4.0f) &&
      near(actual[1u].x, 2.0f) && near(actual[1u].y, 1.0f) &&
      near(actual[1u].z, 0.5f) && near(actual[1u].w, 2.0f) &&
      status[0u] == ended && status[1u] == ended;
  if (!valid) {
    std::cerr << "Cycles Light Falloff interpreter mismatch on " << backend
              << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << index << ": (" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ", "
                << actual[index].w << "), status=" << status[index] << '\n';
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(direct_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Light Falloff XIR: instructions="
              << shape.instructions << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (!distant_short_circuit_is_structural() ||
      shape.instructions > 1800u || shape.loops != 0u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles Light Falloff XIR shape/control regression\n";
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

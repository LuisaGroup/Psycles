#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/cycles_svm_compiler.h>
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

constexpr auto case_count = std::uint32_t{8u};
constexpr auto payload_word_count = std::uint32_t{3u};

constexpr std::string_view type_c = R"IES(IESNA:LM-63-2002
[TEST] PSYCLES TYPE C
TILT=NONE
1 1000 1 5 5 1 1 0 0 0 1 1 100
0 20 75 120 180
0 45 130 250 360
1 2 5 9 12
2 4 8 13 17
4 7 11 16 22
3 6 10 15 20
1 2 5 9 12
)IES";

constexpr std::string_view type_b = R"IES(IESNA:LM-63-2002
[TEST] PSYCLES TYPE B
TILT=NONE
1 1000 0.75 4 3 2 1 0 0 0 1 1 100
0 30 60 90
0 45 90
2 3 5 8
4 7 11 16
6 10 15 21
)IES";

constexpr std::string_view type_a = R"IES(IESNA:LM-63-2002
[TEST] PSYCLES TYPE A
TILT=NONE
1 1000 0.5 4 3 3 1 0 0 0 1 1 100
-90 -30 30 90
0 45 90
3 5 8 13
4 7 11 17
6 10 15 22
)IES";

constexpr auto directions = std::array<luisa::float3, case_count>{
    luisa::float3{-0.023961290717124939f, -0.4788263738155365f,
                  -0.87758255004882812f},
    luisa::float3{-0.76514738798141479f, 0.35017549991607666f,
                  -0.54030227661132812f},
    luisa::float3{0.15563622117042542f, -0.29663854837417603f,
                  0.94222235679626465f},
    luisa::float3{-0.85873013734817505f, -0.43706625699996948f,
                  -0.26749882102012634f},
    luisa::float3{0.057524785399436951f, 0.98376929759979248f,
                  -0.16996714472770691f},
    luisa::float3{-0.44684332609176636f, -0.81794124841690063f,
                  -0.36235776543617249f},
    luisa::float3{-0.66983306407928467f, 0.7312474250793457f,
                  0.1288444995880127f},
    luisa::float3{0.0f, 0.0f, 0.0f}};

constexpr auto strengths = std::array<float, case_count>{
    0.4000000059604645f, 0.4000000059604645f, 0.4000000059604645f,
    0.6000000238418579f, 0.800000011920929f, 0.699999988079071f,
    0.25f, 0.300000011920929f};

constexpr auto slots = std::array<std::uint32_t, case_count>{
    0u, 0u, 0u, 1u, 2u, 2u, 3u, 0u};

// Center pixels from the canonical ies_light_values scene rendered by
// Cycles 5.2.1 CPU, Combined pass, one deterministic sample and BOX filter.
constexpr auto expected = std::array<float, case_count>{
    0.06854613870382309f, 0.26268863677978516f,
    0.3460253179073334f, 0.11535833030939102f,
    0.16630886495113373f, 0.0f, 25.0f, 0.0f};

[[nodiscard]] constexpr std::uint32_t float_bits(float value) noexcept {
  return std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] constexpr auto make_payloads() noexcept {
  std::array<std::uint32_t, case_count * payload_word_count> result{};
  for (auto index = std::uint32_t{}; index < case_count; ++index) {
    result[index * payload_word_count] = float_bits(strengths[index]);
    result[index * payload_word_count + 1u] = slots[index];
    result[index * payload_word_count + 2u] = 0x00000300u;
  }
  return result;
}

constexpr auto payloads = make_payloads();

class IESKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  Expr<Buffer<float>> _ies;

public:
  explicit IESKernelGlobals(Expr<Buffer<float>> ies) noexcept : _ies{ies} {}

  [[nodiscard]] Float ies(
      Expr<std::uint32_t> index) const noexcept override {
    return _ies.read(index);
  }
};

[[nodiscard]] auto direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>,
                  Buffer<luisa::float3>, Buffer<float>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat ies, BufferFloat3 vectors,
         BufferFloat output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        svm_detail::Stack stack;
        svm_detail::stack_store_float3(stack, 0u, vectors.read(index));
        UInt cursor_offset = index * payload_word_count;
        const UInt begin = cursor_offset;
        svm_detail::Cursor cursor{words, cursor_offset};
        const IESKernelGlobals kernel_globals{Expr<Buffer<float>>{ies}};
        svm_detail::node_ies(cursor, stack, kernel_globals);
        output.write(index, svm_detail::stack_load_float(stack, 3u));
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
                instruction->isa<luisa::compute::xir::LoopInst>() ||
                        instruction->isa<
                            luisa::compute::xir::SimpleLoopInst>()
                    ? 1u
                    : 0u;
          });
    }
  }
  return result;
}

[[nodiscard]] bool near(float actual, float expected_value,
                        float tolerance = 8.0e-6f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected_value) <=
             tolerance * std::max(1.0f, std::abs(expected_value));
}

[[nodiscard]] std::vector<float> packed_profiles() {
  IESIDMap profiles;
  if (profiles.get_ies_slot(type_c) != 0u ||
      profiles.get_ies_slot(type_b) != 1u ||
      profiles.get_ies_slot(type_a) != 2u ||
      profiles.get_ies_slot("") != 3u) {
    std::cerr << "IES fixture slot order is not deterministic\n";
    std::exit(EXIT_FAILURE);
  }
  return profiles.packed_data();
}

[[nodiscard]] bool test_direct_handler(Device &device, Stream &stream,
                                       std::string_view backend) {
  const auto packed = packed_profiles();
  auto word_buffer = device.create_buffer<std::uint32_t>(payloads.size());
  auto ies_buffer = device.create_buffer<float>(packed.size());
  auto vector_buffer = device.create_buffer<luisa::float3>(directions.size());
  auto output_buffer = device.create_buffer<float>(case_count);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(case_count);
  auto shader = device.compile(
      direct_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<float, case_count> actual{};
  std::array<std::uint32_t, case_count> cursors{};
  stream << word_buffer.copy_from(payloads.data())
         << ies_buffer.copy_from(packed.data())
         << vector_buffer.copy_from(directions.data())
         << shader(word_buffer, ies_buffer, vector_buffer, output_buffer,
                   cursor_buffer)
                .dispatch(case_count)
         << output_buffer.copy_to(actual.data())
         << cursor_buffer.copy_to(cursors.data()) << synchronize();

  auto valid = true;
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    valid &= near(actual[index], expected[index]) &&
             cursors[index] == payload_word_count;
  }
  if (!valid) {
    std::cerr << "Cycles IES handler mismatch on " << backend << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << index << ": actual=" << actual[index]
                << ", expected=" << expected[index]
                << ", cursor=" << cursors[index] << '\n';
    }
    return false;
  }
  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_IES_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
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
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>,
                  Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [node_types_used](BufferUInt words, BufferFloat ies,
                        BufferFloat4 output, BufferUInt status) noexcept {
        const auto normal = make_float3(0.0f, 0.0f, 1.0f);
        const auto identity = make_float4x4(1.0f);
        device_svm::ShaderData shader_data{
            make_float3(0.0f), normal, normal, normal,
            device_svm::primitive_triangle, 0u, 0u, 0u, 0u, 0.0f, 0.0f,
            0u, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f), identity, identity};
        const device_svm::TransformState transforms{
            identity, identity, identity, identity};
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        const IESKernelGlobals kernel_globals{Expr<Buffer<float>>{ies}};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
            device_svm::kernel_feature_node_emission, node_types_used,
            transforms, shader_data, path_state, result);
        output.write(0u, make_float4(shader_data.closure_emission_background,
                                     result.closure_weight.x));
        status.write(0u, result.status);
      }};
}

[[nodiscard]] bool test_interpreter(Device &device, Stream &stream,
                                    std::string_view backend) {
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0u,
                                          .offset_volume = 0u,
                                          .offset_displacement = 0u});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(
      NODE_VALUE_V,
      SVMNodeValueV{.out_offset = 0u,
                    ._pad = {0u, 0u, 0u},
                    .value = {directions[1u].x, directions[1u].y,
                              directions[1u].z}}));
  static_cast<void>(builder.add_node(
      NODE_IES,
      SVMNodeIES{.strength = input_float(strengths[1u]),
                 .slot = 0u,
                 .vector_offset = 0u,
                 .fac_offset = 3u,
                 ._pad = {0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color = input_float3(1.0f, 0.5f, 0.25f),
                            .strength = input_float(SVMStackOffset{3u})}));
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
  const auto packed = packed_profiles();
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto ies_buffer = device.create_buffer<float>(packed.size());
  auto output_buffer = device.create_buffer<luisa::float4>(1u);
  auto status_buffer = device.create_buffer<std::uint32_t>(1u);
  auto shader = device.compile(
      interpreter_kernel(builder.node_types_used()),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  luisa::float4 actual{};
  std::uint32_t status{};
  stream << word_buffer.copy_from(words.data())
         << ies_buffer.copy_from(packed.data())
         << shader(word_buffer, ies_buffer, output_buffer, status_buffer)
                .dispatch(1u)
         << output_buffer.copy_to(&actual) << status_buffer.copy_to(&status)
         << synchronize();
  const auto factor = expected[1u];
  const auto valid = near(actual.x, factor) &&
                     near(actual.y, factor * 0.5f) &&
                     near(actual.z, factor * 0.25f) &&
                     near(actual.w, factor) &&
                     status == static_cast<std::uint32_t>(
                                   device_svm::EvaluationStatus::ended);
  if (!valid) {
    std::cerr << "Cycles IES interpreter mismatch on " << backend << ": ("
              << actual.x << ", " << actual.y << ", " << actual.z << ", "
              << actual.w << "), status=" << status << '\n';
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(direct_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM IES XIR: instructions=" << shape.instructions
              << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 3200u || shape.loops != 2u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles IES XIR shape/control regression: instructions="
              << shape.instructions << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_direct_handler(device, stream, backend) &&
                 test_interpreter(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

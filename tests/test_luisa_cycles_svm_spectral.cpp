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
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto output_offset = SVMStackOffset{0u};
constexpr auto spectral_word_count = std::uint32_t{2u};
static_assert(sizeof(SVMNodeBlackbody) / sizeof(std::uint32_t) ==
              spectral_word_count);
static_assert(sizeof(SVMNodeWavelength) / sizeof(std::uint32_t) ==
              spectral_word_count);

struct OracleCase {
  float input;
  luisa::float3 expected;
};

// External pinned Cycles 5.2.1 svm_math_blackbody_color_rec709 values after
// the exact NODE_BLACKBODY Rec.709 projection and non-negative clamp.
constexpr std::array blackbody_cases{
    OracleCase{-100.0f, {5.41329432f, 0.0f, 0.0f}},
    OracleCase{0.0f, {5.41329432f, 0.0f, 0.0f}},
    OracleCase{799.0f, {5.41329432f, 0.0f, 0.0f}},
    OracleCase{800.0f, {5.41385937f, 0.0f, 0.0f}},
    OracleCase{964.0f, {4.73331165f, 0.0f, 0.0f}},
    OracleCase{965.0f, {4.73058176f, 0.00053691864f, 0.0f}},
    OracleCase{1166.0f, {4.06227779f, 0.199247807f, 0.0f}},
    OracleCase{1167.0f, {4.06032896f, 0.19980824f, 0.0f}},
    OracleCase{1449.0f, {3.36688447f, 0.40409857f, 0.0f}},
    OracleCase{1902.0f, {2.63755822f, 0.614063382f, 0.0f}},
    OracleCase{3315.0f, {1.63180137f, 0.877736092f, 0.355867386f}},
    OracleCase{6365.0f, {1.05364633f, 0.983039141f, 1.01417351f}},
    OracleCase{6500.0f, {1.04255211f, 0.984052539f, 1.03527153f}},
    OracleCase{11999.0f, {0.826999605f, 0.994075656f, 1.56678903f}},
    OracleCase{12000.0f, {0.826295495f, 0.994508028f, 1.56630766f}},
    OracleCase{20000.0f, {0.826295495f, 0.994508028f, 1.56630766f}},
};

// External pinned Cycles 5.2.1 wavelength node results after its film XYZ
// matrix, empirical 1/2.52 scale, and non-negative clamp. Values surrounding
// 380 nm retain Cycles' truncation-toward-zero behavior.
constexpr std::array wavelength_cases{
    OracleCase{-100.0f, {0.0f, 0.0f, 0.0f}},
    OracleCase{374.9f, {0.0f, 0.0f, 0.0f}},
    OracleCase{375.0f, {0.0f, 0.0f, 0.0f}},
    OracleCase{379.0f, {0.000479078066f, 0.0f, 0.00242034229f}},
    OracleCase{380.0f, {0.0005143577f, 0.0f, 0.00275788317f}},
    OracleCase{382.5f, {0.000602556858f, 0.0f, 0.00360173476f}},
    OracleCase{385.0f, {0.000690756249f, 0.0f, 0.00444558635f}},
    OracleCase{400.0f, {0.00471164985f, 0.0f, 0.0287697166f}},
    OracleCase{445.0f, {0.076790981f, 0.0f, 0.753134608f}},
    OracleCase{500.0f, {0.0f, 0.243057668f, 0.0880704671f}},
    OracleCase{520.1f, {0.0f, 0.506376624f, 0.0f}},
    OracleCase{555.0f, {0.0474033691f, 0.547574043f, 0.0f}},
    OracleCase{600.0f, {0.980824232f, 0.0612070598f, 0.0f}},
    OracleCase{650.0f, {0.299283713f, 0.0f, 0.0f}},
    OracleCase{700.0f, {0.0121582979f, 0.0f, 0.0f}},
    OracleCase{775.0f, {0.000128589454f, 0.0f, 2.20807124e-06f}},
    OracleCase{779.999f, {2.45223752e-08f, 0.0f, 4.21085528e-10f}},
    OracleCase{780.0f, {0.0f, 0.0f, 0.0f}},
    OracleCase{781.0f, {0.0f, 0.0f, 0.0f}},
    OracleCase{1000.0f, {0.0f, 0.0f, 0.0f}},
};

template<typename T>
void append_payload(std::vector<std::uint32_t> &words, const T &payload) {
  const auto encoded = std::bit_cast<
      std::array<std::uint32_t, sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

class NonDefaultColorKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
public:
  [[nodiscard]] Bool film_is_rec709() const noexcept override {
    return false;
  }

  [[nodiscard]] Float3 film_rec709_to_r() const noexcept override {
    return make_float3(0.0f, 1.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_g() const noexcept override {
    return make_float3(0.0f, 0.0f, 1.0f);
  }

  [[nodiscard]] Float3 film_rec709_to_b() const noexcept override {
    return make_float3(1.0f, 0.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_xyz_to_r() const noexcept override {
    return make_float3(1.0f, 0.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_xyz_to_g() const noexcept override {
    return make_float3(0.0f, 1.0f, 0.0f);
  }

  [[nodiscard]] Float3 film_xyz_to_b() const noexcept override {
    return make_float3(0.0f, 0.0f, 1.0f);
  }
};

template<bool blackbody, typename KernelGlobals>
[[nodiscard]] auto direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat4 output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        svm_detail::Stack stack;
        UInt cursor_offset = index * spectral_word_count;
        svm_detail::Cursor cursor{words, cursor_offset};
        const KernelGlobals kernel_globals;
        if constexpr (blackbody) {
          svm_detail::node_blackbody(cursor, stack, kernel_globals);
        } else {
          svm_detail::node_wavelength(cursor, stack, kernel_globals);
        }
        output.write(
            index,
            make_float4(
                svm_detail::stack_load_float3(
                    stack, static_cast<std::uint32_t>(output_offset)),
                0.0f));
        cursors.write(index,
                      cursor_offset - index * spectral_word_count);
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
                        float tolerance = 2.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 2.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

template<bool blackbody, std::size_t N>
[[nodiscard]] bool test_direct_oracle(
    Device &device, Stream &stream, std::string_view backend,
    const std::array<OracleCase, N> &cases) {
  std::vector<std::uint32_t> words;
  words.reserve(cases.size() * spectral_word_count);
  for (const auto &test : cases) {
    if constexpr (blackbody) {
      append_payload(words,
                     SVMNodeBlackbody{.temperature = input_float(test.input),
                                      .color_offset = output_offset,
                                      ._pad = {0u, 0u, 0u}});
    } else {
      append_payload(words,
                     SVMNodeWavelength{.wavelength = input_float(test.input),
                                       .color_offset = output_offset,
                                       ._pad = {0u, 0u, 0u}});
    }
  }
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(cases.size());
  auto cursor_buffer = device.create_buffer<std::uint32_t>(cases.size());
  auto shader = device.compile(
      direct_kernel<blackbody,
                    psycles::test_support::DefaultCyclesSvmKernelGlobals>(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::vector<luisa::float4> actual(cases.size());
  std::vector<std::uint32_t> cursors(cases.size());
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer)
                .dispatch(cases.size())
         << output_buffer.copy_to(luisa::span{actual})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();

  for (auto index = std::size_t{}; index < cases.size(); ++index) {
    if (!near(actual[index].xyz(), cases[index].expected) ||
        cursors[index] != spectral_word_count) {
      std::cerr << "Cycles "
                << (blackbody ? "Blackbody" : "Wavelength")
                << " oracle mismatch on " << backend << " case " << index
                << ": got {" << actual[index].x << ", " << actual[index].y
                << ", " << actual[index].z << "}, cursor=" << cursors[index]
                << '\n';
      return false;
    }
  }
  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_SPECTRAL_CAPTURE")) {
    std::ofstream capture{
        capture_path,
        blackbody ? std::ios::trunc : std::ios::app};
    if (!capture) {
      std::cerr << "Could not open spectral capture path " << capture_path
                << '\n';
      return false;
    }
    if constexpr (blackbody) {
      capture << "backend\tfamily\tindex\tinput\texpected_r\texpected_g"
                 "\texpected_b\tactual_r\tactual_g\tactual_b\n";
    }
    capture << std::setprecision(9);
    for (auto index = std::size_t{}; index < cases.size(); ++index) {
      capture << backend << '\t'
              << (blackbody ? "blackbody" : "wavelength") << '\t'
              << index << '\t' << cases[index].input << '\t'
              << cases[index].expected.x << '\t' << cases[index].expected.y
              << '\t' << cases[index].expected.z << '\t' << actual[index].x
              << '\t' << actual[index].y << '\t' << actual[index].z << '\n';
    }
  }
  return true;
}

[[nodiscard]] bool test_scene_color_transforms(Device &device, Stream &stream,
                                               std::string_view backend) {
  const std::array blackbody_words{
      input_float(6500.0f).bits, std::uint32_t{output_offset}};
  const std::array wavelength_words{
      input_float(500.0f).bits, std::uint32_t{output_offset}};
  auto blackbody_buffer =
      device.create_buffer<std::uint32_t>(blackbody_words.size());
  auto wavelength_buffer =
      device.create_buffer<std::uint32_t>(wavelength_words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(2u);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(2u);
  auto blackbody_shader = device.compile(
      direct_kernel<true, NonDefaultColorKernelGlobals>(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  auto wavelength_shader = device.compile(
      direct_kernel<false, NonDefaultColorKernelGlobals>(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, 2u> actual{};
  std::array<std::uint32_t, 2u> cursors{};
  stream << blackbody_buffer.copy_from(blackbody_words.data())
         << wavelength_buffer.copy_from(wavelength_words.data())
         << blackbody_shader(blackbody_buffer, output_buffer, cursor_buffer)
                .dispatch(1u)
         << wavelength_shader(wavelength_buffer, output_buffer.view(1u, 1u),
                              cursor_buffer.view(1u, 1u))
                .dispatch(1u)
         << output_buffer.copy_to(actual.data())
         << cursor_buffer.copy_to(cursors.data()) << synchronize();
  const auto valid =
      near(actual[0u].xyz(),
           luisa::float3{0.984052539f, 1.03527153f, 1.04255211f}) &&
      near(actual[1u].xyz(),
           luisa::float3{0.00194444447f, 0.128174609f, 0.107936516f}) &&
      cursors == std::array<std::uint32_t, 2u>{2u, 2u};
  if (!valid) {
    std::cerr << "Cycles spectral scene-color transform mismatch on "
              << backend << '\n';
  }
  return valid;
}

[[nodiscard]] auto interpreter_kernel(
    std::array<bool, NODE_NUM> node_types_used) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [node_types_used](BufferUInt words, BufferFloat4 output,
                        BufferUInt status) noexcept {
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
        const psycles::test_support::DefaultCyclesSvmKernelGlobals
            kernel_globals;
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

[[nodiscard]] bool test_interpreter(Device &device, Stream &stream,
                                    std::string_view backend) {
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0u,
                                          .offset_volume = 0u,
                                          .offset_displacement = 0u});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(
      NODE_WAVELENGTH,
      SVMNodeWavelength{.wavelength = input_float(500.0f),
                        .color_offset = 0u,
                        ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_BLACKBODY,
      SVMNodeBlackbody{.temperature = input_float(6500.0f),
                       .color_offset = 3u,
                       ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color = input_float3(SVMStackOffset{0u}),
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
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(1u);
  auto status_buffer = device.create_buffer<std::uint32_t>(1u);
  auto shader = device.compile(
      interpreter_kernel(builder.node_types_used()),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  luisa::float4 actual{};
  std::uint32_t status{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer, status_buffer).dispatch(1u)
         << output_buffer.copy_to(&actual) << status_buffer.copy_to(&status)
         << synchronize();
  const auto expected =
      wavelength_cases[9u].expected * blackbody_cases[12u].expected.x;
  const auto valid = near(actual.xyz(), expected) &&
                     actual.w == static_cast<float>(volume) &&
                     status == static_cast<std::uint32_t>(
                                   device_svm::EvaluationStatus::ended);
  if (!valid) {
    std::cerr << "Cycles spectral interpreter mismatch on " << backend
              << ": got {" << actual.x << ", " << actual.y << ", "
              << actual.z << "}, final_offset=" << actual.w
              << ", status=" << status << '\n';
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto blackbody_shape = module_shape(
      direct_kernel<true,
                    psycles::test_support::DefaultCyclesSvmKernelGlobals>());
  const auto wavelength_shape = module_shape(
      direct_kernel<false,
                    psycles::test_support::DefaultCyclesSvmKernelGlobals>());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Blackbody XIR: instructions="
              << blackbody_shape.instructions
              << ", callables=" << blackbody_shape.callable_definitions
              << ", loops=" << blackbody_shape.loops << '\n'
              << "Cycles SVM Wavelength XIR: instructions="
              << wavelength_shape.instructions
              << ", callables=" << wavelength_shape.callable_definitions
              << ", loops=" << wavelength_shape.loops << '\n';
  }
  // Neither Cycles formula contains a loop. The Blackbody ceiling prevents a
  // regression to evaluating all seven polynomial intervals eagerly.
  if (blackbody_shape.callable_definitions != 0u ||
      wavelength_shape.callable_definitions != 0u ||
      blackbody_shape.loops != 0u || wavelength_shape.loops != 0u ||
      blackbody_shape.instructions > 1050u ||
      wavelength_shape.instructions > 900u) {
    std::cerr << "Cycles spectral XIR shape regression: Blackbody="
              << blackbody_shape.instructions << ", Wavelength="
              << wavelength_shape.instructions << '\n';
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_direct_oracle<true>(device, stream, backend, blackbody_cases) &&
                 test_direct_oracle<false>(device, stream, backend,
                                           wavelength_cases) &&
                 test_scene_color_transforms(device, stream, backend) &&
                 test_interpreter(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

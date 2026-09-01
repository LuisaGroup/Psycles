#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto light_path_case_count = std::uint32_t{16u};
constexpr auto primary_case_count = std::uint32_t{15u};
constexpr auto featureless_case_count = std::uint32_t{5u};
constexpr auto output_offset = std::uint32_t{0u};

constexpr auto primary_path_types = std::array{
    NODE_LP_camera,          NODE_LP_shadow,           NODE_LP_diffuse,
    NODE_LP_glossy,          NODE_LP_singular,         NODE_LP_reflection,
    NODE_LP_transmission,    NODE_LP_volume_scatter,   NODE_LP_ray_length,
    NODE_LP_ray_depth,       NODE_LP_ray_diffuse,      NODE_LP_ray_glossy,
    NODE_LP_ray_transparent, NODE_LP_ray_transmission, NODE_LP_ray_portal};

[[nodiscard]] device_svm::ShaderData
make_shader_data(Expr<std::uint32_t> flags, Expr<float> ray_length) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          device_svm::primitive_triangle,
          0u,
          flags,
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

[[nodiscard]] constexpr auto light_path_words() noexcept {
  std::array<std::uint32_t, light_path_case_count * 2u> words{};
  for (auto index = std::uint32_t{}; index < light_path_case_count; ++index) {
    words[index * 2u] = index;
    words[index * 2u + 1u] = output_offset;
  }
  return words;
}

[[nodiscard]] constexpr auto primary_words() noexcept {
  std::array<std::uint32_t, primary_case_count * 2u> words{};
  for (auto index = std::uint32_t{}; index < primary_case_count; ++index) {
    words[index * 2u] = primary_path_types[index];
    words[index * 2u + 1u] = output_offset;
  }
  return words;
}

[[nodiscard]] auto direct_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat output, BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        $if(index < light_path_case_count) {
          svm_detail::Stack stack;
          const auto shader_data =
              make_shader_data(device_svm::shader_data_backfacing, 12.5f);
          const device_svm::PathState path_state{
              device_svm::path_ray_visibility_camera |
                  device_svm::path_ray_visibility_shadow_transparent |
                  device_svm::path_ray_visibility_diffuse |
                  device_svm::path_ray_visibility_volume_scatter,
              device_svm::path_ray_reflect | device_svm::path_ray_emission,
              3u,
              4u,
              5u,
              6u,
              7u,
              8u};
          UInt cursor_offset = index * 2u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_light_path(
              cursor, stack, shader_data, path_state,
              device_svm::kernel_feature_node_light_path);
          output.write(index,
                       svm_detail::stack_load_float(stack, output_offset));
          cursors.write(index, cursor_offset - begin);
        };
      }};
}

[[nodiscard]] auto featureless_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>>{
      [](BufferUInt words, BufferFloat output) noexcept {
        const UInt index = dispatch_x();
        $if(index < featureless_case_count) {
          UInt visibility = 0u;
          UInt flags = 0u;
          $if((index == 1u) | (index == 3u)) {
            visibility = device_svm::path_ray_visibility_shadow_opaque;
          };
          $if((index == 2u) | (index == 3u)) {
            flags = device_svm::path_ray_emission;
          };
          const auto shader_data = make_shader_data(0u, 1.0f);
          const device_svm::PathState path_state{visibility, flags, 9u,  10u,
                                                 11u,        12u,   13u, 14u};
          svm_detail::Stack stack;
          UInt cursor_offset = index * 2u;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_light_path(cursor, stack, shader_data, path_state,
                                      0u);
          output.write(index,
                       svm_detail::stack_load_float(stack, output_offset));
        };
      }};
}

[[nodiscard]] auto primary_camera_kernel() {
  return Kernel1D<Buffer<std::uint32_t>,
                  Buffer<float>>{[](BufferUInt words,
                                    BufferFloat output) noexcept {
    const UInt index = dispatch_x();
    $if(index < primary_case_count) {
      svm_detail::Stack stack;
      // External Cycles 5.2.1 CPU oracle: the canonical orthographic probe's
      // primary ray travels 2.8999814987182617 units to the surface.
      const auto shader_data = make_shader_data(0u, 2.8999814987182617f);
      const device_svm::PathState path_state{
          device_svm::path_ray_visibility_camera, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
      UInt cursor_offset = index * 2u;
      svm_detail::Cursor cursor{words, cursor_offset};
      svm_detail::node_light_path(cursor, stack, shader_data, path_state,
                                  device_svm::kernel_feature_node_light_path);
      output.write(index, svm_detail::stack_load_float(stack, output_offset));
    };
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
  static constexpr auto words = light_path_words();
  static constexpr std::array expected{1.0f, 1.0f, 1.0f, 0.0f,  0.0f, 1.0f,
                                       0.0f, 1.0f, 1.0f, 12.5f, 4.0f, 5.0f,
                                       6.0f, 4.0f, 7.0f, 8.0f};
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<float>(light_path_case_count);
  auto cursor_buffer =
      device.create_buffer<std::uint32_t>(light_path_case_count);
  auto shader =
      device.compile(direct_kernel(), ShaderOption{.enable_cache = false,
                                                   .enable_fast_math = false});
  std::array<float, light_path_case_count> actual{};
  std::array<std::uint32_t, light_path_case_count> cursors{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer, cursor_buffer)
                .dispatch(light_path_case_count)
         << output_buffer.copy_to(actual.data())
         << cursor_buffer.copy_to(cursors.data()) << synchronize();
  const auto valid = actual == expected &&
                     std::ranges::all_of(cursors, [](auto cursor) noexcept {
                       return cursor == 2u;
                     });
  if (!valid) {
    std::cerr << "Cycles Light Path direct handler mismatch on " << backend
              << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << index << ": " << actual[index] << " expected "
                << expected[index] << ", cursor " << cursors[index] << '\n';
    }
    return false;
  }
  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_LIGHT_PATH_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Light Path capture path " << capture_path
                << '\n';
      return false;
    }
    capture << "path_type\tvalue\tcursor\n" << std::setprecision(9);
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      capture << index << '\t' << actual[index] << '\t' << cursors[index]
              << '\n';
    }
  }
  return true;
}

[[nodiscard]] bool test_feature_mask(Device &device, Stream &stream,
                                     std::string_view backend) {
  static constexpr std::array words{
      static_cast<std::uint32_t>(NODE_LP_ray_depth),       output_offset,
      static_cast<std::uint32_t>(NODE_LP_ray_depth),       output_offset,
      static_cast<std::uint32_t>(NODE_LP_ray_depth),       output_offset,
      static_cast<std::uint32_t>(NODE_LP_ray_depth),       output_offset,
      static_cast<std::uint32_t>(NODE_LP_ray_transparent), output_offset};
  static constexpr std::array expected{0.0f, 1.0f, 1.0f, 1.0f, 0.0f};
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<float>(featureless_case_count);
  auto shader = device.compile(
      featureless_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<float, featureless_case_count> actual{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer).dispatch(featureless_case_count)
         << output_buffer.copy_to(actual.data()) << synchronize();
  if (actual != expected) {
    std::cerr << "Cycles Light Path feature-mask mismatch on " << backend
              << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool test_external_cycles_primary(Device &device, Stream &stream,
                                                std::string_view backend) {
  static constexpr auto words = primary_words();
  static constexpr std::array expected{
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.8999814987182617f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<float>(primary_case_count);
  auto shader = device.compile(
      primary_camera_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<float, primary_case_count> actual{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer).dispatch(primary_case_count)
         << output_buffer.copy_to(actual.data()) << synchronize();
  if (actual != expected) {
    std::cerr << "Cycles Light Path external primary-ray oracle mismatch on "
              << backend << '\n';
    return false;
  }
  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_LIGHT_PATH_PRIMARY_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Light Path primary capture path "
                << capture_path << '\n';
      return false;
    }
    capture << "output_index\tpath_type\tvalue\n" << std::setprecision(9);
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      capture << index << '\t' << primary_path_types[index] << '\t'
              << actual[index] << '\n';
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
    auto shader_data = make_shader_data(0u, 2.0f);
    const auto identity = make_float4x4(1.0f);
    const device_svm::TransformState transforms{identity, identity, identity,
                                                identity};
    const device_svm::PathState path_state{
        device_svm::path_ray_visibility_camera, 0u, 3u};
    const psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
    device_svm::EvaluationResult result;
    device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                           device_svm::kernel_feature_node_emission |
                               device_svm::kernel_feature_node_light_path,
                           node_types_used, transforms, shader_data, path_state,
                           result);
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
      NODE_LIGHT_PATH, SVMNodeLightPath{.path_type = NODE_LP_camera,
                                        .out_offset = 0u,
                                        ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_LIGHT_PATH, SVMNodeLightPath{.path_type = NODE_LP_ray_depth,
                                        .out_offset = 1u,
                                        ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_VALUE_F,
      SVMNodeValueF{.value = 0.25f, .out_offset = 2u, ._pad = {0u, 0u, 0u}}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color =
                                input_float3(static_cast<SVMStackOffset>(0u)),
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
  for (const auto type :
       {NODE_END, NODE_SHADER_JUMP, NODE_LIGHT_PATH, NODE_VALUE_F,
        NODE_EMISSION_WEIGHT, NODE_CLOSURE_EMISSION}) {
    node_types[type] = true;
  }
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
  if (actual.x != 1.0f || actual.y != 3.0f || actual.z != 0.25f ||
      actual.w != 1.0f ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Light Path interpreter mismatch on " << backend
              << ": status=" << status << ", value=(" << actual.x << ", "
              << actual.y << ", " << actual.z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(direct_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Light Path XIR: instructions="
              << shape.instructions << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 1800u || shape.loops != 0u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles Light Path XIR shape regression\n";
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_direct_handler(device, stream, backend) &&
                 test_feature_mask(device, stream, backend) &&
                 test_external_cycles_primary(device, stream, backend) &&
                 test_interpreter(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

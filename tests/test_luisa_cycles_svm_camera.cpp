#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <cmath>
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

constexpr auto case_count = std::uint32_t{5u};
constexpr auto vector_offset = std::uint32_t{0u};
constexpr auto zdepth_offset = std::uint32_t{3u};
constexpr auto distance_offset = std::uint32_t{4u};
constexpr auto sentinel_x = -101.0f;
constexpr auto sentinel_y = -102.0f;
constexpr auto sentinel_z = -103.0f;
constexpr auto sentinel_depth = -104.0f;
constexpr auto sentinel_distance = -105.0f;

[[nodiscard]] constexpr std::uint32_t
pack_offsets(std::uint32_t vector, std::uint32_t zdepth,
             std::uint32_t distance) noexcept {
  return vector | (zdepth << 8u) | (distance << 16u);
}

[[nodiscard]] auto camera_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[](BufferUInt words,
                                            BufferFloat4 output,
                                            BufferUInt cursors) noexcept {
    const UInt index = dispatch_x();
    $if(index < case_count) {
      svm_detail::Stack stack;
      svm_detail::stack_store_float3(
          stack, vector_offset,
          make_float3(sentinel_x, sentinel_y, sentinel_z));
      svm_detail::stack_store_float(stack, zdepth_offset, sentinel_depth);
      svm_detail::stack_store_float(stack, distance_offset, sentinel_distance);

      Float3 position = make_float3(1.0f, -2.0f, 6.0f);
      $if(index == 4u) { position = make_float3(-0.5f, 2.0f, 2.0f); };
      const auto identity = make_float4x4(1.0f);
      const auto world_to_camera =
          make_float4x4(make_float4(2.0f, 0.0f, 0.0f, 0.0f),
                        make_float4(0.0f, -1.0f, 0.0f, 0.0f),
                        make_float4(0.0f, 0.0f, 0.5f, 0.0f),
                        make_float4(1.0f, 2.0f, -1.0f, 1.0f));
      const device_svm::TransformState transforms{identity, world_to_camera,
                                                  identity, identity};
      device_svm::ShaderData shader_data{position,
                                         make_float3(0.0f, 0.0f, 1.0f),
                                         make_float3(0.0f, 0.0f, 1.0f),
                                         make_float3(0.0f, 0.0f, -1.0f),
                                         device_svm::primitive_triangle,
                                         0u,
                                         0u,
                                         0u,
                                         0u,
                                         0.0f,
                                         0.0f,
                                         0u,
                                         0.0f,
                                         1.0f,
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

      UInt cursor_offset = index;
      const UInt begin = cursor_offset;
      svm_detail::Cursor cursor{words, cursor_offset};
      svm_detail::node_camera(cursor, stack, transforms, shader_data);
      output.write(
          index * 2u,
          make_float4(svm_detail::stack_load_float3(stack, vector_offset),
                      svm_detail::stack_load_float(stack, zdepth_offset)));
      output.write(index * 2u + 1u, make_float4(svm_detail::stack_load_float(
                                                    stack, distance_offset),
                                                0.0f, 0.0f, 0.0f));
      cursors.write(index, cursor_offset - begin);
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

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 3.0e-6f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool test_direct_handler(Device &device, Stream &stream,
                                       std::string_view backend) {
  static constexpr std::array words{
      pack_offsets(vector_offset, zdepth_offset, distance_offset),
      pack_offsets(vector_offset, SVM_STACK_INVALID, SVM_STACK_INVALID),
      pack_offsets(SVM_STACK_INVALID, zdepth_offset, SVM_STACK_INVALID),
      pack_offsets(SVM_STACK_INVALID, SVM_STACK_INVALID, distance_offset),
      pack_offsets(vector_offset, zdepth_offset, distance_offset)};
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(case_count * 2u);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(case_count);
  auto shader =
      device.compile(camera_kernel(), ShaderOption{.enable_cache = false,
                                                   .enable_fast_math = false});
  std::array<luisa::float4, case_count * 2u> actual{};
  std::array<std::uint32_t, case_count> cursors{};
  stream
      << word_buffer.copy_from(words.data())
      << shader(word_buffer, output_buffer, cursor_buffer).dispatch(case_count)
      << output_buffer.copy_to(actual.data())
      << cursor_buffer.copy_to(cursors.data()) << synchronize();

  const auto distance = std::sqrt(29.0f);
  const auto nx = 3.0f / distance;
  const auto ny = 4.0f / distance;
  const auto nz = 2.0f / distance;
  const auto valid_vector = [&](std::size_t index) {
    return near(actual[index * 2u].x, nx) && near(actual[index * 2u].y, ny) &&
           near(actual[index * 2u].z, nz);
  };
  const auto sentinel_vector = [&](std::size_t index) {
    return actual[index * 2u].x == sentinel_x &&
           actual[index * 2u].y == sentinel_y &&
           actual[index * 2u].z == sentinel_z;
  };
  const auto valid_depth = [&](std::size_t index) {
    return near(actual[index * 2u].w, 2.0f);
  };
  const auto valid_distance = [&](std::size_t index) {
    return near(actual[index * 2u + 1u].x, distance);
  };

  const auto valid =
      valid_vector(0u) && valid_depth(0u) && valid_distance(0u) &&
      valid_vector(1u) && actual[2u].w == sentinel_depth &&
      actual[3u].x == sentinel_distance && sentinel_vector(2u) &&
      valid_depth(2u) && actual[5u].x == sentinel_distance &&
      sentinel_vector(3u) && actual[6u].w == sentinel_depth &&
      valid_distance(3u) && std::isnan(actual[8u].x) &&
      std::isnan(actual[8u].y) && std::isnan(actual[8u].z) &&
      actual[8u].w == 0.0f && actual[9u].x == 0.0f &&
      std::all_of(cursors.begin(), cursors.end(),
                  [](auto cursor) noexcept { return cursor == 1u; });
  if (!valid) {
    std::cerr << "Cycles Camera Data direct handler mismatch on " << backend
              << '\n';
    return false;
  }

  if (const auto *capture_path =
          std::getenv("PSYCLES_CYCLES_SVM_CAMERA_CAPTURE")) {
    std::ofstream capture{capture_path};
    if (!capture) {
      std::cerr << "Could not open Camera Data capture path " << capture_path
                << '\n';
      return false;
    }
    capture << "case\tx\ty\tz\tzdepth\tdistance\tcursor\n"
            << std::setprecision(9);
    for (auto index = std::size_t{}; index < case_count; ++index) {
      capture << index << '\t' << actual[index * 2u].x << '\t'
              << actual[index * 2u].y << '\t' << actual[index * 2u].z << '\t'
              << actual[index * 2u].w << '\t' << actual[index * 2u + 1u].x
              << '\t' << cursors[index] << '\n';
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
    const auto world_to_camera =
        make_float4x4(make_float4(2.0f, 0.0f, 0.0f, 0.0f),
                      make_float4(0.0f, -1.0f, 0.0f, 0.0f),
                      make_float4(0.0f, 0.0f, 0.5f, 0.0f),
                      make_float4(1.0f, 2.0f, -1.0f, 1.0f));
    const device_svm::TransformState transforms{identity, world_to_camera,
                                                identity, identity};
    device_svm::ShaderData shader_data{make_float3(1.0f, -2.0f, 6.0f),
                                       make_float3(0.0f, 0.0f, 1.0f),
                                       make_float3(0.0f, 0.0f, 1.0f),
                                       make_float3(0.0f, 0.0f, -1.0f),
                                       device_svm::primitive_triangle,
                                       0u,
                                       0u,
                                       0u,
                                       0u,
                                       0.0f,
                                       0.0f,
                                       0u,
                                       0.0f,
                                       1.0f,
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
    const psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
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
  static_cast<void>(builder.add_node(
      NODE_CAMERA,
      SVMNodeCamera{.vector_offset = static_cast<SVMStackOffset>(vector_offset),
                    .zdepth_offset = SVM_STACK_INVALID,
                    .distance_offset = SVM_STACK_INVALID,
                    ._pad = {0u}}));
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{
          .color = input_float3(static_cast<SVMStackOffset>(vector_offset)),
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
  node_types[NODE_CAMERA] = true;
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

  const auto distance = std::sqrt(29.0f);
  if (!near(actual.x, 3.0f / distance) || !near(actual.y, 4.0f / distance) ||
      !near(actual.z, 2.0f / distance) ||
      status !=
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended)) {
    std::cerr << "Cycles Camera Data interpreter mismatch on " << backend
              << ": status=" << status << ", value=(" << actual.x << ", "
              << actual.y << ", " << actual.z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(camera_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Camera Data XIR: instructions="
              << shape.instructions << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 1200u || shape.loops != 0u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles Camera Data XIR shape regression: "
              << shape.instructions << '/' << shape.loops << '/'
              << shape.callable_definitions << '\n';
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

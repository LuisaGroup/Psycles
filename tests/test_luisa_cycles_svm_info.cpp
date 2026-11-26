#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

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

constexpr auto object_case_count = std::uint32_t{6u};
constexpr auto particle_case_count = std::uint32_t{8u};
constexpr auto hair_case_count = std::uint32_t{6u};
constexpr auto point_case_count = std::uint32_t{3u};
constexpr auto output_offset = std::uint32_t{0u};
constexpr auto untouched = -91.0f;

template<std::size_t N>
[[nodiscard]] constexpr auto payload_words(
    const std::array<std::uint32_t, N> &types) noexcept {
  std::array<std::uint32_t, N * 2u> words{};
  for (auto index = std::size_t{}; index < N; ++index) {
    words[index * 2u] = types[index];
    words[index * 2u + 1u] = output_offset;
  }
  return words;
}

constexpr auto object_types = std::array{
    std::uint32_t{NODE_INFO_OB_LOCATION},
    std::uint32_t{NODE_INFO_OB_COLOR},
    std::uint32_t{NODE_INFO_OB_ALPHA},
    std::uint32_t{NODE_INFO_OB_INDEX},
    std::uint32_t{NODE_INFO_MAT_INDEX},
    std::uint32_t{NODE_INFO_OB_RANDOM},
};

constexpr auto particle_types = std::array{
    std::uint32_t{NODE_INFO_PAR_INDEX},
    std::uint32_t{NODE_INFO_PAR_RANDOM},
    std::uint32_t{NODE_INFO_PAR_AGE},
    std::uint32_t{NODE_INFO_PAR_LIFETIME},
    std::uint32_t{NODE_INFO_PAR_LOCATION},
    std::uint32_t{NODE_INFO_PAR_SIZE},
    std::uint32_t{NODE_INFO_PAR_VELOCITY},
    std::uint32_t{NODE_INFO_PAR_ANGULAR_VELOCITY},
};

constexpr auto hair_types = std::array{
    std::uint32_t{NODE_INFO_CURVE_IS_STRAND},
    std::uint32_t{NODE_INFO_CURVE_INTERCEPT},
    std::uint32_t{NODE_INFO_CURVE_LENGTH},
    std::uint32_t{NODE_INFO_CURVE_THICKNESS},
    std::uint32_t{NODE_INFO_CURVE_TANGENT_NORMAL},
    std::uint32_t{NODE_INFO_CURVE_RANDOM},
};

constexpr auto point_types = std::array{
    std::uint32_t{NODE_INFO_POINT_POSITION},
    std::uint32_t{NODE_INFO_POINT_RADIUS},
    std::uint32_t{NODE_INFO_POINT_RANDOM},
};

class ProbeKernelGlobals
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals,
      public device_svm::InfoServices,
      public device_svm::AmbientOcclusionServices {
public:
  [[nodiscard]] const device_svm::InfoServices *
  info_services() const noexcept override {
    return this;
  }

  [[nodiscard]] const device_svm::AmbientOcclusionServices *
  ambient_occlusion_services() const noexcept override {
    return this;
  }

  [[nodiscard]] Float global_ao_distance(
      const device_svm::ShaderData &) const noexcept override {
    return 5.0f;
  }

  [[nodiscard]] Bool ambient_occluded(
      const device_svm::ShaderData &,
      const Var<luisa::compute::Ray> &ray,
      Expr<bool> only_local) const noexcept override {
    return only_local | (ray->direction().z < 0.0f) | (ray->t_max() < 4.0f);
  }

  [[nodiscard]] Float3 object_location(
      const device_svm::ShaderData &shader_data) const noexcept override {
    return shader_data.P + make_float3(1.0f, 2.0f, 3.0f);
  }

  [[nodiscard]] Float3 object_color(
      Expr<std::uint32_t> object) const noexcept override {
    const auto value = object.cast<float>();
    return make_float3(value, value + 0.5f, value + 1.0f);
  }

  [[nodiscard]] Float object_alpha(
      Expr<std::uint32_t>) const noexcept override {
    return 0.25f;
  }

  [[nodiscard]] Float object_pass_id(
      Expr<std::uint32_t>) const noexcept override {
    return 7.0f;
  }

  [[nodiscard]] Float shader_pass_id(
      const device_svm::ShaderData &shader_data) const noexcept override {
    return shader_data.shader.cast<float>();
  }

  [[nodiscard]] Float object_random_number(
      Expr<std::uint32_t>) const noexcept override {
    return 0.75f;
  }

  [[nodiscard]] Int object_particle_id(
      Expr<std::uint32_t> object) const noexcept override {
    return object.cast<std::int32_t>() + 28;
  }

  [[nodiscard]] UInt particle_index(
      Expr<std::int32_t>) const noexcept override {
    // Index zero has an independently pinned Cycles hash oracle below.
    return 0u;
  }

  [[nodiscard]] Float particle_age(
      Expr<std::int32_t> particle) const noexcept override {
    return particle.cast<float>() + 0.25f;
  }

  [[nodiscard]] Float particle_lifetime(
      Expr<std::int32_t> particle) const noexcept override {
    return particle.cast<float>() + 1.5f;
  }

  [[nodiscard]] Float particle_size(
      Expr<std::int32_t> particle) const noexcept override {
    return particle.cast<float>() + 0.125f;
  }

  [[nodiscard]] Float3 particle_location(
      Expr<std::int32_t> particle) const noexcept override {
    const auto value = particle.cast<float>();
    return make_float3(value + 1.0f, value + 2.0f, value + 3.0f);
  }

  [[nodiscard]] Float3 particle_velocity(
      Expr<std::int32_t> particle) const noexcept override {
    const auto value = particle.cast<float>();
    return make_float3(-value, value * 2.0f, value * 3.0f);
  }

  [[nodiscard]] Float3 particle_angular_velocity(
      Expr<std::int32_t> particle) const noexcept override {
    const auto value = particle.cast<float>();
    return make_float3(value * 0.5f, -value, value + 4.0f);
  }

  [[nodiscard]] Float curve_thickness(
      const device_svm::ShaderData &shader_data) const noexcept override {
    return select(0.0f, shader_data.u + 2.0f,
                  (shader_data.type & device_svm::primitive_curve) != 0u);
  }

  [[nodiscard]] Float3 point_position(
      const device_svm::ShaderData &shader_data) const noexcept override {
    return select(make_float3(0.0f),
                  shader_data.P + make_float3(4.0f, 5.0f, 6.0f),
                  (shader_data.type & device_svm::primitive_point) != 0u);
  }

  [[nodiscard]] Float point_radius(
      const device_svm::ShaderData &shader_data) const noexcept override {
    return select(0.0f, 1.25f,
                  (shader_data.type & device_svm::primitive_point) != 0u);
  }
};

[[nodiscard]] device_svm::ShaderData make_shader_data(
    Expr<std::uint32_t> primitive_type) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(10.0f, 20.0f, 30.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          primitive_type,
          13u,
          0u,
          0u,
          0u,
          0.25f,
          0.75f,
          9u,
          0.0f,
          2.0f,
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

inline void initialize_stack(svm_detail::Stack &stack) noexcept {
  svm_detail::stack_store_float(stack, 0u, untouched);
  svm_detail::stack_store_float(stack, 1u, untouched);
  svm_detail::stack_store_float(stack, 2u, untouched);
}

[[nodiscard]] Float4 capture_stack(svm_detail::Stack &stack,
                                   UInt consumed) noexcept {
  return make_float4(svm_detail::stack_load_float(stack, 0u),
                     svm_detail::stack_load_float(stack, 1u),
                     svm_detail::stack_load_float(stack, 2u),
                     consumed.cast<float>());
}

[[nodiscard]] auto object_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
      [](BufferUInt words, BufferFloat4 output) noexcept {
        const UInt index = dispatch_x();
        $if(index < object_case_count) {
          svm_detail::Stack stack{SVM_STACK_SIZE};
          initialize_stack(stack);
          auto shader_data =
              make_shader_data(device_svm::primitive_triangle);
          const ProbeKernelGlobals kernel_globals;
          UInt cursor_offset = index * 2u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_object_info(cursor, stack, kernel_globals,
                                       shader_data);
          output.write(index, capture_stack(stack, cursor_offset - begin));
        };
      }};
}

[[nodiscard]] auto particle_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
      [](BufferUInt words, BufferFloat4 output) noexcept {
        const UInt index = dispatch_x();
        $if(index < particle_case_count) {
          svm_detail::Stack stack{SVM_STACK_SIZE};
          initialize_stack(stack);
          auto shader_data =
              make_shader_data(device_svm::primitive_triangle);
          const ProbeKernelGlobals kernel_globals;
          UInt cursor_offset = index * 2u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_particle_info(cursor, stack, kernel_globals,
                                         shader_data);
          output.write(index, capture_stack(stack, cursor_offset - begin));
        };
      }};
}

[[nodiscard]] auto hair_kernel(bool curve) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
      [curve](BufferUInt words, BufferFloat4 output) noexcept {
        const UInt index = dispatch_x();
        $if(index < hair_case_count) {
          svm_detail::Stack stack{SVM_STACK_SIZE};
          initialize_stack(stack);
          auto shader_data = make_shader_data(
              curve ? device_svm::primitive_curve
                    : device_svm::primitive_triangle);
          const ProbeKernelGlobals kernel_globals;
          UInt cursor_offset = index * 2u;
          const UInt begin = cursor_offset;
          Bool supported = true;
          const svm_detail::EvaluationTransition transition{&supported};
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_hair_info(cursor, stack, &kernel_globals,
                                     shader_data, transition);
          output.write(index, capture_stack(stack, cursor_offset - begin));
        };
      }};
}

[[nodiscard]] auto point_kernel(bool point) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
      [point](BufferUInt words, BufferFloat4 output) noexcept {
        const UInt index = dispatch_x();
        $if(index < point_case_count) {
          svm_detail::Stack stack{SVM_STACK_SIZE};
          initialize_stack(stack);
          auto shader_data = make_shader_data(
              point ? device_svm::primitive_point
                    : device_svm::primitive_triangle);
          const ProbeKernelGlobals kernel_globals;
          UInt cursor_offset = index * 2u;
          const UInt begin = cursor_offset;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_point_info(cursor, stack, kernel_globals,
                                      shader_data);
          output.write(index, capture_stack(stack, cursor_offset - begin));
        };
      }};
}

[[nodiscard]] bool is_near(float actual, float expected) noexcept {
  return std::abs(actual - expected) <= 2.0e-6f;
}

[[nodiscard]] bool equal(luisa::float4 actual,
                         luisa::float4 expected) noexcept {
  return is_near(actual.x, expected.x) && is_near(actual.y, expected.y) &&
         is_near(actual.z, expected.z) && is_near(actual.w, expected.w);
}

void capture_direct_results(std::string_view family,
                            std::span<const luisa::float4> values) {
  const auto *path = std::getenv("PSYCLES_CYCLES_SVM_INFO_CAPTURE");
  if (path == nullptr) {
    return;
  }
  static bool started = false;
  std::ofstream capture{
      path, std::ios::out | (started ? std::ios::app : std::ios::trunc)};
  if (!capture) {
    std::cerr << "Could not open Info capture path " << path << '\n';
    std::exit(EXIT_FAILURE);
  }
  started = true;
  capture << family << '\n';
  for (auto index = std::size_t{}; index < values.size(); ++index) {
    capture << index;
    for (const auto component : {values[index].x, values[index].y,
                                 values[index].z, values[index].w}) {
      capture << '\t' << std::hex << std::setw(8) << std::setfill('0')
              << std::bit_cast<std::uint32_t>(component) << std::dec;
    }
    capture << '\n';
  }
}

template<typename Kernel, std::size_t WordCount, std::size_t CaseCount>
[[nodiscard]] bool run_direct(Device &device, Stream &stream,
                              std::string_view backend,
                              std::string_view family, const Kernel &kernel,
                              const std::array<std::uint32_t, WordCount> &words,
                              const std::array<luisa::float4, CaseCount> &expected) {
  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(expected.size());
  auto shader = device.compile(
      kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float4, CaseCount> actual{};
  stream << word_buffer.copy_from(words.data())
         << shader(word_buffer, output_buffer).dispatch(expected.size())
         << output_buffer.copy_to(actual.data()) << synchronize();
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    if (!equal(actual[index], expected[index])) {
      std::cerr << "Cycles " << family << " mismatch on " << backend
                << " at " << index << ": (" << actual[index].x << ", "
                << actual[index].y << ", " << actual[index].z << ", "
                << actual[index].w << ") expected (" << expected[index].x
                << ", " << expected[index].y << ", " << expected[index].z
                << ", " << expected[index].w << ")\n";
      return false;
    }
  }
  capture_direct_results(family, actual);
  return true;
}

[[nodiscard]] bool test_direct_handlers(Device &device, Stream &stream,
                                        std::string_view backend) {
  static constexpr auto object_words = payload_words(object_types);
  static constexpr auto particle_words = payload_words(particle_types);
  static constexpr auto hair_words = payload_words(hair_types);
  static constexpr auto point_words = payload_words(point_types);
  static constexpr auto scalar_tail =
      luisa::float4{untouched, untouched, untouched, 2.0f};
  static constexpr auto object_expected = std::array{
      luisa::float4{11.0f, 22.0f, 33.0f, 2.0f},
      luisa::float4{9.0f, 9.5f, 10.0f, 2.0f},
      luisa::float4{0.25f, untouched, untouched, 2.0f},
      luisa::float4{7.0f, untouched, untouched, 2.0f},
      luisa::float4{13.0f, untouched, untouched, 2.0f},
      luisa::float4{0.75f, untouched, untouched, 2.0f},
  };
  // Cycles 5.2.1 hash_uint2_to_float(0, 0), pinned independently of this
  // implementation and already used by the particle-hair export oracle.
  static constexpr auto particle_random = 0.860312759f;
  static constexpr auto particle_expected = std::array{
      luisa::float4{0.0f, untouched, untouched, 2.0f},
      luisa::float4{particle_random, untouched, untouched, 2.0f},
      luisa::float4{37.25f, untouched, untouched, 2.0f},
      luisa::float4{38.5f, untouched, untouched, 2.0f},
      luisa::float4{38.0f, 39.0f, 40.0f, 2.0f},
      luisa::float4{37.125f, untouched, untouched, 2.0f},
      luisa::float4{-37.0f, 74.0f, 111.0f, 2.0f},
      luisa::float4{18.5f, -37.0f, 41.0f, 2.0f},
  };
  static constexpr auto hair_expected = std::array{
      luisa::float4{1.0f, untouched, untouched, 2.0f},
      scalar_tail,
      scalar_tail,
      luisa::float4{2.25f, untouched, untouched, 2.0f},
      luisa::float4{0.0f, 0.0f, 1.0f, 2.0f},
      scalar_tail,
  };
  static constexpr auto hair_noncurve_expected = std::array{
      luisa::float4{0.0f, untouched, untouched, 2.0f},
      scalar_tail,
      scalar_tail,
      luisa::float4{0.0f, untouched, untouched, 2.0f},
      luisa::float4{0.0f, 0.0f, 0.0f, 2.0f},
      scalar_tail,
  };
  static constexpr auto point_expected = std::array{
      luisa::float4{14.0f, 25.0f, 36.0f, 2.0f},
      luisa::float4{1.25f, untouched, untouched, 2.0f},
      scalar_tail,
  };
  static constexpr auto point_nonpoint_expected = std::array{
      luisa::float4{0.0f, 0.0f, 0.0f, 2.0f},
      luisa::float4{0.0f, untouched, untouched, 2.0f},
      scalar_tail,
  };
  return run_direct(device, stream, backend, "Object Info", object_kernel(),
                    object_words, object_expected) &&
         run_direct(device, stream, backend, "Particle Info",
                    particle_kernel(), particle_words, particle_expected) &&
         run_direct(device, stream, backend, "Hair Info", hair_kernel(true),
                    hair_words, hair_expected) &&
         run_direct(device, stream, backend, "Hair Info non-curve",
                    hair_kernel(false), hair_words,
                    hair_noncurve_expected) &&
         run_direct(device, stream, backend, "Point Info", point_kernel(true),
                    point_words, point_expected) &&
         run_direct(device, stream, backend, "Point Info non-point",
                    point_kernel(false), point_words,
                    point_nonpoint_expected);
}

struct InterpreterProgram {
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types{};
  std::uint32_t surface_offset{};
  std::uint32_t after_info_offset{};
};

[[nodiscard]] InterpreterProgram object_interpreter_program() {
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0u,
                                          .offset_volume = 0u,
                                          .offset_displacement = 0u});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(
      NODE_OBJECT_INFO,
      SVMNodeObjectInfo{.info_type = NODE_INFO_OB_LOCATION,
                        .out_offset = 0u,
                        ._pad = {0u, 0u, 0u}}));
  const auto after_info = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(
      NODE_EMISSION_WEIGHT,
      SVMNodeEmissionWeight{.color = input_float3(SVMStackOffset{0u}),
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
  InterpreterProgram result;
  result.words.assign(builder.words().begin(), builder.words().end());
  result.surface_offset = surface;
  result.after_info_offset = after_info;
  for (const auto type : {NODE_END, NODE_SHADER_JUMP, NODE_OBJECT_INFO,
                          NODE_EMISSION_WEIGHT, NODE_CLOSURE_EMISSION}) {
    result.node_types[type] = true;
  }
  return result;
}

[[nodiscard]] auto interpreter_kernel(
    std::array<bool, NODE_NUM> node_types, bool with_services) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [node_types, with_services](BufferUInt words, BufferFloat4 output,
                                  BufferUInt state) noexcept {
        auto shader_data =
            make_shader_data(device_svm::primitive_triangle);
        // The interpreter's global jump table is rooted at shader zero. The
        // direct-handler fixture keeps shader 13 to test Material Index.
        shader_data.shader = 0u;
        const auto identity = make_float4x4(1.0f);
        const device_svm::TransformState transforms{identity, identity,
                                                    identity, identity};
        const device_svm::PathState path_state{0u, 0u, 0u};
        device_svm::EvaluationResult result;
        if (with_services) {
          const ProbeKernelGlobals kernel_globals;
          device_svm::eval_nodes(
              kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
              device_svm::kernel_feature_node_emission, node_types,
              transforms, shader_data, path_state, result);
        } else {
          const psycles::test_support::DefaultCyclesSvmKernelGlobals
              kernel_globals;
          device_svm::eval_nodes(
              kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
              device_svm::kernel_feature_node_emission, node_types,
              transforms, shader_data, path_state, result);
        }
        output.write(0u, make_float4(shader_data.closure_emission_background,
                                    result.closure_weight.x));
        state.write(0u, result.status);
        state.write(1u, result.final_offset);
      }};
}

[[nodiscard]] bool test_interpreter(Device &device, Stream &stream,
                                    std::string_view backend) {
  const auto program = object_interpreter_program();
  auto word_buffer =
      device.create_buffer<std::uint32_t>(program.words.size());
  auto output_buffer = device.create_buffer<luisa::float4>(1u);
  auto state_buffer = device.create_buffer<std::uint32_t>(2u);
  for (const auto with_services : {true, false}) {
    auto shader = device.compile(
        interpreter_kernel(program.node_types, with_services),
        ShaderOption{.enable_cache = false, .enable_fast_math = false});
    luisa::float4 output{};
    std::array<std::uint32_t, 2u> state{};
    stream << word_buffer.copy_from(program.words.data())
           << shader(word_buffer, output_buffer, state_buffer).dispatch(1u)
           << output_buffer.copy_to(&output) << state_buffer.copy_to(state.data())
           << synchronize();
    const auto expected_status = static_cast<std::uint32_t>(
        with_services ? device_svm::EvaluationStatus::ended
                      : device_svm::EvaluationStatus::unsupported_node);
    const auto expected_offset =
        with_services ? static_cast<std::uint32_t>(program.words.size() - 2u)
                      : program.after_info_offset;
    const auto valid_output =
        with_services ? equal(output, {11.0f, 22.0f, 33.0f, 11.0f})
                      : true;
    if (state[0u] != expected_status || state[1u] != expected_offset ||
        !valid_output) {
      std::cerr << "Cycles Info interpreter mismatch on " << backend
                << ": services=" << with_services << ", status="
                << state[0u] << ", offset=" << state[1u] << ", value=("
                << output.x << ", " << output.y << ", " << output.z << ", "
                << output.w << ")\n";
      return false;
    }
  }
  return true;
}

struct GatedProgram {
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types{};
  std::uint32_t surface_offset{};
  std::uint32_t after_info_offset{};
};

[[nodiscard]] GatedProgram gated_program(ShaderNodeType opcode,
                                         std::uint32_t info_type) {
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0u,
                                          .offset_volume = 0u,
                                          .offset_displacement = 0u});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  if (opcode == NODE_OBJECT_INFO) {
    static_cast<void>(builder.add_node(
        opcode,
        SVMNodeObjectInfo{
            .info_type = static_cast<NodeObjectInfo>(info_type),
            .out_offset = 0u,
            ._pad = {0u, 0u, 0u}}));
  } else if (opcode == NODE_PARTICLE_INFO) {
    static_cast<void>(builder.add_node(
        opcode,
        SVMNodeParticleInfo{
            .info_type = static_cast<NodeParticleInfo>(info_type),
            .out_offset = 0u,
            ._pad = {0u, 0u, 0u}}));
  } else if (opcode == NODE_HAIR_INFO) {
    static_cast<void>(builder.add_node(
        opcode, SVMNodeHairInfo{.info_type = static_cast<NodeHairInfo>(info_type),
                                .out_offset = 0u,
                                ._pad = {0u, 0u, 0u}}));
  } else {
    static_cast<void>(builder.add_node(
        opcode,
        SVMNodePointInfo{.info_type = static_cast<NodePointInfo>(info_type),
                         .out_offset = 0u,
                         ._pad = {0u, 0u, 0u}}));
  }
  const auto after_info = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  const auto volume = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  const auto displacement = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  builder.set_word(jump + 1u, surface);
  builder.set_word(jump + 2u, volume);
  builder.set_word(jump + 3u, displacement);
  GatedProgram result;
  result.words.assign(builder.words().begin(), builder.words().end());
  result.surface_offset = surface;
  result.after_info_offset = after_info;
  result.node_types[NODE_END] = true;
  result.node_types[NODE_SHADER_JUMP] = true;
  result.node_types[opcode] = true;
  return result;
}

[[nodiscard]] auto gate_kernel(std::array<bool, NODE_NUM> node_types,
                               std::uint32_t kernel_features,
                               bool with_services,
                               std::uint32_t primitive_type) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<std::uint32_t>>{
      [node_types, kernel_features, with_services,
       primitive_type](BufferUInt words, BufferUInt state) noexcept {
        auto shader_data = make_shader_data(primitive_type);
        shader_data.shader = 0u;
        const auto identity = make_float4x4(1.0f);
        const device_svm::TransformState transforms{identity, identity,
                                                    identity, identity};
        const device_svm::PathState path_state{0u, 0u, 0u};
        device_svm::EvaluationResult result;
        if (with_services) {
          const ProbeKernelGlobals kernel_globals;
          device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE,
                                 kernel_features, 0u, node_types, transforms,
                                 shader_data, path_state, result);
        } else {
          const psycles::test_support::DefaultCyclesSvmKernelGlobals
              kernel_globals;
          device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE,
                                 kernel_features, 0u, node_types, transforms,
                                 shader_data, path_state, result);
        }
        state.write(0u, result.status);
        state.write(1u, result.final_offset);
      }};
}

[[nodiscard]] bool run_gate(
    Device &device, Stream &stream, std::string_view backend,
    std::string_view label, const GatedProgram &program,
    std::uint32_t kernel_features, bool with_services,
    std::uint32_t primitive_type, device_svm::EvaluationStatus expected_status,
    std::uint32_t expected_offset) {
  auto word_buffer =
      device.create_buffer<std::uint32_t>(program.words.size());
  auto state_buffer = device.create_buffer<std::uint32_t>(2u);
  auto shader = device.compile(
      gate_kernel(program.node_types, kernel_features, with_services,
                  primitive_type),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<std::uint32_t, 2u> state{};
  stream << word_buffer.copy_from(program.words.data())
         << shader(word_buffer, state_buffer).dispatch(1u)
         << state_buffer.copy_to(state.data()) << synchronize();
  if (state[0u] != static_cast<std::uint32_t>(expected_status) ||
      state[1u] != expected_offset) {
    std::cerr << "Cycles Info " << label << " mismatch on " << backend
              << ": status=" << state[0u] << ", offset=" << state[1u]
              << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool test_feature_gates(Device &device, Stream &stream,
                                      std::string_view backend) {
  const auto particle = gated_program(NODE_PARTICLE_INFO,
                                      NODE_INFO_PAR_INDEX);
  const auto hair = gated_program(NODE_HAIR_INFO,
                                  NODE_INFO_CURVE_THICKNESS);
  const auto hair_is_strand = gated_program(
      NODE_HAIR_INFO, NODE_INFO_CURVE_IS_STRAND);
  const auto hair_tangent = gated_program(
      NODE_HAIR_INFO, NODE_INFO_CURVE_TANGENT_NORMAL);
  const auto point = gated_program(NODE_POINT_INFO,
                                   NODE_INFO_POINT_RADIUS);
  const auto hair_is_strand_ended =
      hair_is_strand.after_info_offset + 1u;
  const auto hair_tangent_ended = hair_tangent.after_info_offset + 1u;
  const auto hair_ended = hair.after_info_offset + 1u;
  const auto point_ended = point.after_info_offset + 1u;
  return run_gate(device, stream, backend, "Particle service missing",
                  particle, 0u, false, device_svm::primitive_triangle,
                  device_svm::EvaluationStatus::unsupported_node,
                  particle.after_info_offset) &&
         run_gate(device, stream, backend, "Hair feature disabled", hair, 0u,
                  true, device_svm::primitive_curve,
                  device_svm::EvaluationStatus::invalid_node,
                  hair.surface_offset + 1u) &&
         run_gate(device, stream, backend, "Hair feature enabled", hair,
                  device_svm::kernel_feature_hair, true,
                  device_svm::primitive_curve,
                  device_svm::EvaluationStatus::ended, hair_ended) &&
         run_gate(device, stream, backend, "Hair service missing", hair,
                  device_svm::kernel_feature_hair, false,
                  device_svm::primitive_curve,
                  device_svm::EvaluationStatus::unsupported_node,
                  hair.after_info_offset) &&
         run_gate(device, stream, backend,
                  "Hair Is Strand without service", hair_is_strand,
                  device_svm::kernel_feature_hair, false,
                  device_svm::primitive_curve,
                  device_svm::EvaluationStatus::ended,
                  hair_is_strand_ended) &&
         run_gate(device, stream, backend,
                  "Hair tangent without service", hair_tangent,
                  device_svm::kernel_feature_hair, false,
                  device_svm::primitive_curve,
                  device_svm::EvaluationStatus::ended,
                  hair_tangent_ended) &&
         run_gate(device, stream, backend, "Point feature disabled", point,
                  0u, true, device_svm::primitive_point,
                  device_svm::EvaluationStatus::invalid_node,
                  point.surface_offset + 1u) &&
         run_gate(device, stream, backend, "Point feature enabled", point,
                  device_svm::kernel_feature_pointcloud, true,
                  device_svm::primitive_point,
                  device_svm::EvaluationStatus::ended, point_ended) &&
         run_gate(device, stream, backend, "Point service missing", point,
                  device_svm::kernel_feature_pointcloud, false,
                  device_svm::primitive_point,
                  device_svm::EvaluationStatus::unsupported_node,
                  point.after_info_offset);
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

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto shape = module_shape(particle_kernel());
  if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
    std::cout << "Cycles SVM Particle Info XIR: instructions="
              << shape.instructions << ", loops=" << shape.loops
              << ", callables=" << shape.callable_definitions << '\n';
  }
  if (shape.instructions > 2400u || shape.loops != 0u ||
      shape.callable_definitions != 0u) {
    std::cerr << "Cycles Info XIR shape regression\n";
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_direct_handlers(device, stream, backend) &&
                 test_interpreter(device, stream, backend) &&
                 test_feature_gates(device, stream, backend) &&
                 [&] {
                   BytecodeBuilder builder;
                   builder.add_node_data(
                       SVMNodeAmbientOcclusion{
                           .color = input_float3(0.5f, 0.25f, 0.75f),
                           .dist = input_float(1.0f),
                           .flags = NODE_AO_GLOBAL_RADIUS,
                           .samples = 4u,
                           .normal_offset = SVM_STACK_INVALID,
                           .out_ao_offset = 0u,
                           .out_color_offset = 1u,
                           ._pad = {0u, 0u, 0u}});
                   builder.add_node_data(
                       SVMNodeAmbientOcclusion{
                           .color = input_float3(1.0f, 1.0f, 1.0f),
                           .dist = input_float(5.0f),
                           .flags = NODE_AO_INSIDE,
                           .samples = 4u,
                           .normal_offset = SVM_STACK_INVALID,
                           .out_ao_offset = 4u,
                           .out_color_offset = SVM_STACK_INVALID,
                           ._pad = {0u, 0u, 0u}});
                   builder.add_node_data(
                       SVMNodeAmbientOcclusion{
                           .color = input_float3(1.0f, 1.0f, 1.0f),
                           .dist = input_float(5.0f),
                           .flags = NODE_AO_ONLY_LOCAL,
                           .samples = 4u,
                           .normal_offset = SVM_STACK_INVALID,
                           .out_ao_offset = 5u,
                           .out_color_offset = SVM_STACK_INVALID,
                           ._pad = {0u, 0u, 0u}});
                   std::vector<std::uint32_t> words(builder.words().begin(),
                                                    builder.words().end());
                   auto word_buffer =
                       device.create_buffer<std::uint32_t>(words.size());
                   auto out_buffer = device.create_buffer<luisa::float4>(2u);
                   Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>> k =
                       [](BufferUInt w, BufferFloat4 out) noexcept {
                         svm_detail::Stack stack{SVM_STACK_SIZE};
                         auto sd = make_shader_data(device_svm::primitive_triangle);
                         const ProbeKernelGlobals kg;
                         UInt pc = 0u;
                         svm_detail::Cursor cursor{w, pc};
                         svm_detail::node_ambient_occlusion(cursor, stack, kg, sd);
                         svm_detail::node_ambient_occlusion(cursor, stack, kg, sd);
                         svm_detail::node_ambient_occlusion(cursor, stack, kg, sd);
                         out.write(0u, make_float4(
                                           svm_detail::stack_load_float(stack, 0u),
                                           svm_detail::stack_load_float(stack, 1u),
                                           svm_detail::stack_load_float(stack, 2u),
                                           svm_detail::stack_load_float(stack, 3u)));
                         out.write(1u, make_float4(
                                           svm_detail::stack_load_float(stack, 4u),
                                           svm_detail::stack_load_float(stack, 5u),
                                           pc.cast<float>(),
                                           0.0f));
                       };
                   auto shader = device.compile(
                       k, ShaderOption{.enable_cache = false,
                                       .enable_fast_math = false});
                   std::array<luisa::float4, 2u> actual{};
                   stream << word_buffer.copy_from(words.data())
                          << shader(word_buffer, out_buffer).dispatch(1u)
                          << out_buffer.copy_to(actual.data()) << synchronize();
                   const bool ok =
                       equal(actual[0u], {1.0f, 0.5f, 0.25f, 0.75f}) &&
                       equal(actual[1u], {0.0f, 0.0f, 18.0f, 0.0f});
                   if (!ok) {
                     std::cerr << "Cycles AO Info test mismatch on " << backend
                               << '\n';
                   }
                   return ok;
                 }() &&
                 [&] {
                   class BevelProbeKernelGlobals final
                       : public ProbeKernelGlobals {
                   public:
                     Float min_ray_pdf{10.0f};
                     UInt tri_shader{0u};

                     [[nodiscard]] device_svm::TriangleVertices
                     triangle_vertices(Expr<std::uint32_t>,
                                       Expr<std::uint32_t> prim) const noexcept
                         override {
                       device_svm::TriangleVertices v{
                           .v0 = make_float3(0.0f, 0.0f, 0.0f),
                           .v1 = make_float3(0.0f, 1.0f, 0.0f),
                           .v2 = make_float3(0.0f, 0.0f, 1.0f)};
                       $if(prim == 1u) {
                         v.v0 = make_float3(0.0f, 0.0f, 0.0f);
                         v.v1 = make_float3(0.0f, 1.0f, 0.0f);
                         v.v2 = make_float3(-1.0f, 0.0f, 1.0f);
                       };
                       return v;
                     }

                     [[nodiscard]] UInt
                     triangle_shader(Expr<std::uint32_t>) const noexcept
                         override {
                       return tri_shader;
                     }

                     [[nodiscard]] Var<packed_normal>
                     attribute_normal(Expr<std::int32_t> offset) const noexcept
                         override {
                       Var<packed_normal> packed;
                       // (65535, 32768) decodes to (+1, 0, 0), whereas
                       // (32768, 32768) decodes to ~ (0, 0, +1).
                       packed.value = select(
                           static_cast<std::uint32_t>(32768u | (32768u << 16u)),
                           static_cast<std::uint32_t>(65535u | (32768u << 16u)),
                           offset == 0);
                       return packed;
                     }

                     [[nodiscard]] Float bevel_min_ray_pdf() const noexcept
                         override {
                       return min_ray_pdf;
                     }

                     [[nodiscard]] UInt bevel_lcg_state_init() const noexcept
                         override {
                       return 1234u;
                     }

                     [[nodiscard]] Float2
                     bevel_sample_disk(Expr<std::uint32_t>,
                                       Expr<std::uint32_t>) const noexcept
                         override {
                       return make_float2(0.1f, 0.02f);
                     }

                     [[nodiscard]] device_svm::BevelLocalHitReservoir
                     trace_bevel_local(const device_svm::ShaderData &sd,
                                       const Var<luisa::compute::Ray> &,
                                       UInt &) const noexcept override {
                       device_svm::BevelLocalHitReservoir out{};
                       out.prim[0u] = 0u;
                       out.u[0u] = 0.25f;
                       out.v[0u] = 0.05f;
                       out.num_hits = select(1u, 2u, sd.dPdu.z > 0.5f);
                       out.prim[1u] = 1u;
                       out.u[1u] = 0.25f;
                       out.v[1u] = 0.05f;
                       return out;
                     }
                   };

                   const auto make_unlinked_word = [](float radius) {
                     BytecodeBuilder b;
                     b.add_node_data(SVMNodeBevel{
                         .radius = input_float(radius),
                         .num_samples = 1u,
                         .normal_offset = SVM_STACK_INVALID,
                         .out_offset = 0u,
                         ._pad = {0u}});
                     const auto words = b.words();
                     return std::vector<std::uint32_t>{words.begin(),
                                                       words.end()};
                   };
                   const auto make_linked_word = [](float radius) {
                     BytecodeBuilder b;
                     b.add_node_data(SVMNodeBevel{
                         .radius = input_float(radius),
                         .num_samples = 1u,
                         .normal_offset = 4u,
                         .out_offset = 0u,
                         ._pad = {0u}});
                     const auto words = b.words();
                     return std::vector<std::uint32_t>{words.begin(),
                                                       words.end()};
                   };

                   const auto unlinked_words = make_unlinked_word(0.5f);
                   const auto linked_words = make_linked_word(0.5f);
                   auto word_buf_unlinked =
                       device.create_buffer<std::uint32_t>(unlinked_words.size());
                   auto word_buf_linked =
                       device.create_buffer<std::uint32_t>(linked_words.size());
                   auto out_buf = device.create_buffer<luisa::float4>(6u);

                   Kernel1D bevel_k =
                       [](BufferUInt wu, BufferUInt wl,
                          BufferFloat4 out) noexcept {
                         const UInt idx = dispatch_x();
                         svm_detail::Stack stack{SVM_STACK_SIZE};
                         svm_detail::stack_store_float3(
                             stack, 4u, make_float3(0.0f, 1.0f, 0.0f));
                         auto sd = make_shader_data(
                             device_svm::primitive_triangle);
                         sd.type = device_svm::primitive_triangle;
                         sd.object = 0u;
                         sd.prim = 0u;
                         sd.u = 0.25f;
                         sd.v = 0.05f;
                         sd.P = make_float3(0.0f, 0.25f, 0.05f);
                         sd.N = make_float3(1.0f, 0.0f, 0.0f);
                         sd.Ng = make_float3(1.0f, 0.0f, 0.0f);
                         sd.object_flag =
                             device_svm::shader_data_object_transform_applied;
                         sd.dPdu.z = select(0.0f, 1.0f,
                                            (idx >= 1u) & (idx <= 4u));
                         sd.flag = select(0u,
                                          device_svm::shader_data_backfacing,
                                          idx == 3u);

                         BevelProbeKernelGlobals kg;
                         kg.min_ray_pdf = select(10.0f, 7.5f, idx == 4u);
                         kg.tri_shader = select(
                             0u, device_svm::shader_smooth_normal, idx == 5u);

                         const auto identity = make_float4x4(1.0f);
                         const device_svm::TransformState transforms{
                             identity, identity, identity, identity};
                         UInt pc = 0u;
                         $if((idx == 2u) | (idx == 4u)) {
                           svm_detail::Cursor c{wl, pc};
                           svm_detail::node_bevel(c, stack, kg, transforms, kg,
                                                  sd, false);
                         }
                         $else {
                           svm_detail::Cursor c{wu, pc};
                           svm_detail::node_bevel(c, stack, kg, transforms, kg,
                                                  sd, false);
                         };
                         const auto out_n =
                             svm_detail::stack_load_float3(stack, 0u);
                         out.write(idx, make_float4(out_n, pc.cast<float>()));
                       };

                   auto shader = device.compile(
                       bevel_k, ShaderOption{.enable_cache = false,
                                             .enable_fast_math = false});
                   std::array<luisa::float4, 6u> actual{};
                   stream << word_buf_unlinked.copy_from(unlinked_words.data())
                          << word_buf_linked.copy_from(linked_words.data())
                          << shader(word_buf_unlinked, word_buf_linked, out_buf)
                                 .dispatch(6u)
                          << out_buf.copy_to(actual.data()) << synchronize();

                   const auto unit_len = [](luisa::float4 v) noexcept {
                     const float l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                     return std::abs(l - 1.0f) <= 1.0e-4f;
                   };
                   const auto norm3 = [](float x, float y, float z) noexcept {
                     const float inv = 1.0f / std::sqrt(x * x + y * y + z * z);
                     return luisa::make_float4(x * inv, y * inv, z * inv, 2.0f);
                   };
                   const luisa::float4 flat = actual[0u];
                   const luisa::float4 corner = actual[1u];
                   const luisa::float4 linked = actual[2u];
                   const luisa::float4 backfacing = actual[3u];
                   const luisa::float4 low_pdf = actual[4u];
                   const luisa::float4 smooth_hit = actual[5u];

                   const auto expected_linked = norm3(
                       0.0f + (corner.x - 1.0f),
                       1.0f + (corner.y - 0.0f),
                       0.0f + (corner.z - 0.0f));
                   bool ok =
                       equal(flat, {1.0f, 0.0f, 0.0f, 2.0f}) &&
                       unit_len(corner) && corner.x > 0.1f && corner.z > 0.01f &&
                       is_near(corner.y, 0.0f) && is_near(corner.w, 2.0f) &&
                       equal(linked, expected_linked) &&
                       equal(backfacing, {-corner.x, -corner.y, -corner.z, 2.0f}) &&
                       equal(low_pdf, {0.0f, 1.0f, 0.0f, 2.0f}) &&
                       unit_len(smooth_hit) && smooth_hit.x > 0.85f &&
                       smooth_hit.z > 0.35f && smooth_hit.z < 0.45f &&
                       is_near(smooth_hit.w, 2.0f);

                   // Check NODE_BEVEL missing service gate transitions to unsupported_node.
                   BytecodeBuilder gate_builder;
                   const auto jump = gate_builder.add_node(
                       NODE_SHADER_JUMP,
                       SVMNodeShaderJump{.offset_surface = 0u,
                                         .offset_volume = 0u,
                                         .offset_displacement = 0u});
                   const auto surface = static_cast<std::uint32_t>(
                       gate_builder.size());
                   static_cast<void>(gate_builder.add_node(
                       NODE_BEVEL, SVMNodeBevel{.radius = input_float(0.05f),
                                                .num_samples = 4u,
                                                .normal_offset = SVM_STACK_INVALID,
                                                .out_offset = 0u,
                                                ._pad = {0u}}));
                   const auto after = static_cast<std::uint32_t>(
                       gate_builder.size());
                   static_cast<void>(gate_builder.add_node(NODE_END));
                   const auto vol = static_cast<std::uint32_t>(
                       gate_builder.size());
                   static_cast<void>(gate_builder.add_node(NODE_END));
                   const auto disp = static_cast<std::uint32_t>(
                       gate_builder.size());
                   static_cast<void>(gate_builder.add_node(NODE_END));
                   gate_builder.set_word(jump + 1u, surface);
                   gate_builder.set_word(jump + 2u, vol);
                   gate_builder.set_word(jump + 3u, disp);
                   GatedProgram bevel_prog;
                   bevel_prog.words.assign(gate_builder.words().begin(),
                                           gate_builder.words().end());
                   bevel_prog.surface_offset = surface;
                   bevel_prog.after_info_offset = after;
                   bevel_prog.node_types[NODE_END] = true;
                   bevel_prog.node_types[NODE_SHADER_JUMP] = true;
                   bevel_prog.node_types[NODE_BEVEL] = true;
                   ok = ok &&
                        run_gate(device, stream, backend,
                                 "Bevel raytrace mask off", bevel_prog,
                                 device_svm::kernel_feature_node_raytrace,
                                 false, device_svm::primitive_triangle,
                                 device_svm::EvaluationStatus::ended,
                                 after + 1u) &&
                        [&]() {
                          auto wbuf = device.create_buffer<std::uint32_t>(
                              bevel_prog.words.size());
                          auto sbuf = device.create_buffer<std::uint32_t>(2u);
                          Kernel1D gate_raytrace =
                              [nt = bevel_prog.node_types](
                                  BufferUInt w, BufferUInt st) noexcept {
                                auto sd = make_shader_data(
                                    device_svm::primitive_triangle);
                                sd.shader = 0u;
                                const auto id = make_float4x4(1.0f);
                                const device_svm::TransformState tf{
                                    id, id, id, id};
                                const device_svm::PathState ps{0u, 0u, 0u};
                                const psycles::test_support::
                                    DefaultCyclesSvmKernelGlobals kg;
                                device_svm::EvaluationResult res;
                                device_svm::eval_nodes(
                                    kg, w, SHADER_TYPE_SURFACE,
                                    device_svm::kernel_feature_node_raytrace,
                                    device_svm::kernel_feature_node_raytrace,
                                    nt, tf, sd, ps, res);
                                st.write(0u, res.status);
                                st.write(1u, res.final_offset);
                              };
                          auto sh = device.compile(
                              gate_raytrace,
                              ShaderOption{.enable_cache = false,
                                           .enable_fast_math = false});
                          std::array<std::uint32_t, 2u> st{};
                          stream << wbuf.copy_from(bevel_prog.words.data())
                                 << sh(wbuf, sbuf).dispatch(1u)
                                 << sbuf.copy_to(st.data()) << synchronize();
                          return st[0u] ==
                                     static_cast<std::uint32_t>(
                                         device_svm::EvaluationStatus::
                                             unsupported_node) &&
                                 st[1u] == after;
                        }();
                   if (!ok) {
                     std::cerr << "Cycles Bevel Info test mismatch on " << backend
                               << ": flat=(" << flat.x << ',' << flat.y << ','
                               << flat.z << ',' << flat.w << ") corner=("
                               << corner.x << ',' << corner.y << ',' << corner.z
                               << ',' << corner.w << ") linked=(" << linked.x
                               << ',' << linked.y << ',' << linked.z << ','
                               << linked.w << ") backfacing=(" << backfacing.x
                               << ',' << backfacing.y << ',' << backfacing.z
                               << ',' << backfacing.w << ") low_pdf=("
                               << low_pdf.x << ',' << low_pdf.y << ','
                               << low_pdf.z << ',' << low_pdf.w
                               << ") smooth=(" << smooth_hit.x << ','
                               << smooth_hit.y << ',' << smooth_hit.z << ','
                               << smooth_hit.w << ")\n";
                   }
                   return ok;
                 }()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

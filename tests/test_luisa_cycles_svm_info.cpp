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

class ProbeKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals,
      public device_svm::InfoServices {
public:
  [[nodiscard]] const device_svm::InfoServices *
  info_services() const noexcept override {
    return this;
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
  stack[0u] = untouched;
  stack[1u] = untouched;
  stack[2u] = untouched;
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
          svm_detail::Stack stack;
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
          svm_detail::Stack stack;
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
          svm_detail::Stack stack;
          initialize_stack(stack);
          auto shader_data = make_shader_data(
              curve ? device_svm::primitive_curve
                    : device_svm::primitive_triangle);
          const ProbeKernelGlobals kernel_globals;
          UInt cursor_offset = index * 2u;
          const UInt begin = cursor_offset;
          Bool supported = true;
          svm_detail::Cursor cursor{words, cursor_offset};
          svm_detail::node_hair_info(cursor, stack, &kernel_globals,
                                     shader_data, supported);
          output.write(index, capture_stack(stack, cursor_offset - begin));
        };
      }};
}

[[nodiscard]] auto point_kernel(bool point) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>>{
      [point](BufferUInt words, BufferFloat4 output) noexcept {
        const UInt index = dispatch_x();
        $if(index < point_case_count) {
          svm_detail::Stack stack;
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

[[nodiscard]] bool near(float actual, float expected) noexcept {
  return std::abs(actual - expected) <= 2.0e-6f;
}

[[nodiscard]] bool equal(luisa::float4 actual,
                         luisa::float4 expected) noexcept {
  return near(actual.x, expected.x) && near(actual.y, expected.y) &&
         near(actual.z, expected.z) && near(actual.w, expected.w);
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
                 test_feature_gates(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

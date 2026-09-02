#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace device_svm = psycles::luisa_backend::cycles_svm;

static_assert(closure::type_hair_chiang ==
              static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_CHIANG_ID));
static_assert(closure::type_hair_huang ==
              static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_HUANG_ID));

inline constexpr std::uint32_t output_stride = 4u;
inline constexpr std::uint32_t meta_stride = 6u;
inline constexpr std::uint32_t scenario_count = 8u;

struct Program {
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types{};
  std::uint32_t final_offset{};
};

[[nodiscard]] SVMNodePrincipledHairBsdfData reflectance_payload() noexcept {
  return {.parametrization = NODE_PRINCIPLED_HAIR_REFLECTANCE,
          .color = input_float3(0.13f, 0.47f, 0.81f),
          .tint = input_float3(0.9f, 0.8f, 0.7f),
          .absorption_coefficient = input_float3(0.1f, 0.2f, 0.3f),
          .roughness = input_float(0.37f),
          .random_roughness = input_float(0.4f),
          .offset = input_float(0.19f),
          .ior = input_float(1.55f),
          .random = input_float(0.73f),
          .melanin = input_float(0.5f),
          .melanin_redness = input_float(0.2f),
          .coat = input_float(0.28f),
          .aspect_ratio = input_float(1.0f),
          .radial_roughness = input_float(0.21f),
          .random_color = input_float(0.0f),
          .R = input_float(1.0f),
          .TT = input_float(1.0f),
          .TRT = input_float(1.0f),
          .attr_random = ATTR_STD_CURVE_RANDOM,
          .attr_normal = ATTR_STD_NONE};
}

[[nodiscard]] SVMNodePrincipledHairBsdfData pigment_payload() noexcept {
  return {
      .parametrization = NODE_PRINCIPLED_HAIR_PIGMENT_CONCENTRATION,
      .color = input_float3(0.1f, 0.2f, 0.3f),
      .tint = input_float3(0.72f, 0.55f, 0.29f),
      .absorption_coefficient = input_float3(0.3f, 0.4f, 0.5f),
      .roughness = input_float(0.58f),
      .random_roughness = input_float(0.7f),
      .offset = input_float(-0.23f),
      .ior = input_float(1.42f),
      /* The attribute value 0.12 must override this stack/immediate value. */
      .random = input_float(0.91f),
      .melanin = input_float(0.62f),
      .melanin_redness = input_float(0.31f),
      .coat = input_float(1.3f),
      .aspect_ratio = input_float(1.0f),
      .radial_roughness = input_float(0.45f),
      .random_color = input_float(0.8f),
      .R = input_float(1.0f),
      .TT = input_float(1.0f),
      .TRT = input_float(1.0f),
      .attr_random = ATTR_STD_CURVE_RANDOM,
      .attr_normal = ATTR_STD_NONE};
}

[[nodiscard]] SVMNodePrincipledHairBsdfData absorption_payload() noexcept {
  return {.parametrization = NODE_PRINCIPLED_HAIR_DIRECT_ABSORPTION,
          .color = input_float3(0.1f, 0.2f, 0.3f),
          .tint = input_float3(0.4f, 0.5f, 0.6f),
          .absorption_coefficient = input_float3(0.2f, 0.7f, 1.1f),
          .roughness = input_float(-0.2f),
          .random_roughness = input_float(0.0f),
          .offset = input_float(0.41f),
          .ior = input_float(1.33f),
          .random = input_float(0.5f),
          .melanin = input_float(0.5f),
          .melanin_redness = input_float(0.5f),
          .coat = input_float(-0.3f),
          .aspect_ratio = input_float(1.0f),
          .radial_roughness = input_float(1.7f),
          .random_color = input_float(0.0f),
          .R = input_float(1.0f),
          .TT = input_float(1.0f),
          .TRT = input_float(1.0f),
          .attr_random = ATTR_STD_CURVE_RANDOM,
          .attr_normal = ATTR_STD_NONE};
}

[[nodiscard]] SVMNodePrincipledHairBsdfData invalid_payload() noexcept {
  auto result = reflectance_payload();
  result.parametrization = static_cast<NodePrincipledHairParametrization>(99u);
  result.roughness = input_float(0.33f);
  result.random_roughness = input_float(0.0f);
  result.radial_roughness = input_float(0.29f);
  result.coat = input_float(0.5f);
  result.offset = input_float(0.0f);
  result.ior = input_float(1.5f);
  return result;
}

[[nodiscard]] Program make_program(const SVMNodePrincipledHairBsdfData &payload,
                                   packed_float3 weight = {0.8f, 0.4f, 0.2f},
                                   bool zero_mix = false) {
  BytecodeBuilder builder;
  const auto jump = builder.add_node(
      NODE_SHADER_JUMP, SVMNodeShaderJump{.offset_surface = 0,
                                          .offset_volume = 0,
                                          .offset_displacement = 0});
  const auto surface = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_CLOSURE_SET_WEIGHT,
                                     SVMNodeClosureSetWeight{.rgb = weight}));
  if (zero_mix) {
    static_cast<void>(builder.add_node(
        NODE_VALUE_F,
        SVMNodeValueF{.value = 0.0f, .out_offset = 0u, ._pad = {0u, 0u, 0u}}));
  }
  static_cast<void>(builder.add_node(
      NODE_CLOSURE_BSDF,
      SVMNodeClosureBsdf{.closure_type = CLOSURE_BSDF_HAIR_CHIANG_ID,
                         .mix_weight_offset =
                             zero_mix ? SVMStackOffset{0u} : SVM_STACK_INVALID,
                         ._pad = {0u, 0u, 0u}}));
  builder.add_node_data(payload);
  const auto surface_end = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  const auto volume = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  const auto displacement = static_cast<std::uint32_t>(builder.size());
  static_cast<void>(builder.add_node(NODE_END));
  builder.set_word(jump + 1u, surface);
  builder.set_word(jump + 2u, volume);
  builder.set_word(jump + 3u, displacement);
  return {.words = {builder.words().begin(), builder.words().end()},
          .node_types = builder.node_types_used(),
          .final_offset = surface_end + 1u};
}

class HairKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  Bool _attribute_found;

public:
  explicit HairKernelGlobals(Expr<bool> attribute_found) noexcept
      : _attribute_found{attribute_found} {}

  [[nodiscard]] Var<AttributeMap>
  attribute_map(Expr<std::uint32_t>) const noexcept override {
    Var<AttributeMap> entry;
    entry.id = select(static_cast<luisa::ulong>(ATTR_STD_NONE),
                      static_cast<luisa::ulong>(ATTR_STD_CURVE_RANDOM),
                      _attribute_found);
    entry.offset = 0;
    entry.element = select(static_cast<std::uint16_t>(ATTR_ELEMENT_NONE),
                           static_cast<std::uint16_t>(ATTR_ELEMENT_OBJECT),
                           _attribute_found);
    entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT);
    entry.pad = static_cast<std::uint8_t>(0u);
    return entry;
  }

  [[nodiscard]] Float
  attribute_float(Expr<std::int32_t>) const noexcept override {
    return 0.12f;
  }
};

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *closures,
                 Expr<std::uint32_t> primitive_type) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          normalize(make_float3(0.1f, -0.2f, 1.0f)),
          normalize(make_float3(0.3f, -0.1f, 0.95f)),
          primitive_type,
          0u,
          0u,
          0u,
          0u,
          0.0f,
          0.27f,
          0u,
          0.0f,
          1.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          make_float3(3.0f, 4.0f, 0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          identity,
          identity,
          0u,
          closures};
}

template <std::size_t Capacity,
          std::uint32_t NodeFeatureMask = device_svm::kernel_feature_node_bsdf>
[[nodiscard]] auto transition_kernel(std::array<bool, NODE_NUM> used) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>, std::uint32_t>{
      [used](BufferUInt words, BufferFloat4 output, BufferUInt meta,
             UInt scenario) noexcept {
        HairKernelGlobals kernel_globals{scenario == 1u};
        device_svm::ClosurePool closures{Capacity};
        const auto primitive_type =
            select(device_svm::primitive_curve_thick,
                   device_svm::primitive_curve_ribbon, scenario == 1u);
        auto shader_data = make_shader_data(&closures, primitive_type);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            kernel_globals, words, SHADER_TYPE_SURFACE,
            device_svm::kernel_feature_node_principled_hair, NodeFeatureMask,
            used, identity_transform_state(), shader_data, path_state, result);

        const auto output_base = scenario * output_stride;
        const auto meta_base = scenario * meta_stride;
        for (auto field = 0u; field < output_stride; ++field) {
          output.write(output_base + field, make_float4(0.0f));
        }
        UInt type = static_cast<std::uint32_t>(CLOSURE_NONE_ID);
        $if(closures.count() != 0u) {
          const auto hair = closures.chiang_hair(0u);
          type = hair.common.type;
          output.write(
              output_base + 0u,
              make_float4(hair.common.weight, hair.common.sample_weight));
          output.write(output_base + 1u,
                       make_float4(hair.common.N, hair.param.h));
          output.write(output_base + 2u,
                       make_float4(hair.param.sigma, hair.param.v));
          output.write(output_base + 3u,
                       make_float4(hair.param.s, hair.param.alpha,
                                   hair.param.eta, hair.param.m0_roughness));
        };
        meta.write(meta_base + 0u, type);
        meta.write(meta_base + 1u, closures.count());
        meta.write(meta_base + 2u, closures.left());
        meta.write(meta_base + 3u, shader_data.flag);
        meta.write(meta_base + 4u, result.status);
        meta.write(meta_base + 5u, result.final_offset);
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 3.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 3.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  const std::array programs{
      make_program(reflectance_payload()),
      make_program(pigment_payload()),
      make_program(absorption_payload()),
      make_program(invalid_payload()),
      make_program(reflectance_payload(), {1.0e-6f, 1.0e-6f, 1.0e-6f}),
      make_program(reflectance_payload()),
      make_program(reflectance_payload(), {0.8f, 0.4f, 0.2f}, true),
      make_program(reflectance_payload())};

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  std::array<Buffer<std::uint32_t>, scenario_count> word_buffers;
  for (auto index = 0u; index < scenario_count; ++index) {
    word_buffers[index] =
        device.create_buffer<std::uint32_t>(programs[index].words.size());
  }
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(scenario_count * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = true};
  auto full_shader =
      device.compile(transition_kernel<8u>(programs[0u].node_types), options);
  auto zero_capacity_shader =
      device.compile(transition_kernel<0u>(programs[5u].node_types), options);
  auto zero_mix_shader =
      device.compile(transition_kernel<8u>(programs[6u].node_types), options);
  auto feature_erased_shader = device.compile(
      transition_kernel<8u, 0u>(programs[7u].node_types), options);

  for (auto index = 0u; index < scenario_count; ++index) {
    stream << word_buffers[index].copy_from(programs[index].words.data());
  }
  for (auto index : {0u, 1u, 2u, 3u, 4u}) {
    stream
        << full_shader(word_buffers[index], output, meta, index).dispatch(1u);
  }
  stream
      << zero_capacity_shader(word_buffers[5u], output, meta, 5u).dispatch(1u)
      << zero_mix_shader(word_buffers[6u], output, meta, 6u).dispatch(1u)
      << feature_erased_shader(word_buffers[7u], output, meta, 7u).dispatch(1u);
  std::array<luisa::float4, scenario_count * output_stride> actual{};
  std::array<std::uint32_t, scenario_count * meta_stride> actual_meta{};
  stream << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  const auto value = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual[scenario * output_stride + field];
  };
  const auto state = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual_meta[scenario * meta_stride + field];
  };
  constexpr auto flags = device_svm::shader_data_bsdf |
                         device_svm::shader_data_bsdf_has_eval |
                         device_svm::shader_data_bsdf_has_transmission;
  constexpr auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  constexpr auto chiang = closure::type_hair_chiang;
  constexpr auto common = luisa::float4{0.8f, 0.4f, 0.2f, 0.4666666667f};
  constexpr std::array<luisa::float4, 12u> oracle{
      luisa::float4{0.762866139f, -0.572149634f, -0.301131397f, 0.107753888f},
      luisa::float4{0.118423782f, 0.0162182655f, 0.00126328191f, 0.224562809f},
      luisa::float4{0.0875470489f, -0.189999998f, 1.54999995f, 0.0959622115f},
      luisa::float4{0.762866139f, -0.572149634f, -0.301131397f, -0.270000011f},
      luisa::float4{0.129874706f, 0.234955996f, 0.526990354f, 0.0659941882f},
      luisa::float4{0.0681587979f, 0.230000004f, 1.41999996f, 5.28255782e-7f},
      luisa::float4{0.762866139f, -0.572149634f, -0.301131397f, 0.107753888f},
      luisa::float4{0.2f, 0.7f, 1.1f, 5.28255782e-7f},
      luisa::float4{4.28069448f, -0.409999996f, 1.33000004f, 5.28255782e-7f},
      luisa::float4{0.762866139f, -0.572149634f, -0.301131397f, 0.107753888f},
      luisa::float4{0.276265055f, 0.590385675f, 1.54966176f, 0.107588485f},
      luisa::float4{0.11108461f, -0.0f, 1.5f, 0.0201346762f}};

  bool valid = true;
  for (auto scenario = 0u; scenario < 4u; ++scenario) {
    valid &= state(scenario, 0u) == chiang && state(scenario, 1u) == 1u &&
             state(scenario, 2u) == 7u && state(scenario, 3u) == flags &&
             state(scenario, 4u) == ended &&
             state(scenario, 5u) == programs[scenario].final_offset &&
             near(value(scenario, 0u), common) &&
             near(value(scenario, 1u), oracle[scenario * 3u + 0u]) &&
             near(value(scenario, 2u), oracle[scenario * 3u + 1u]) &&
             near(value(scenario, 3u), oracle[scenario * 3u + 2u]);
  }
  for (auto scenario : {4u, 5u, 6u, 7u}) {
    valid &=
        state(scenario, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
        state(scenario, 1u) == 0u && state(scenario, 3u) == 0u &&
        state(scenario, 4u) == ended &&
        state(scenario, 5u) == programs[scenario].final_offset;
  }
  valid &= state(4u, 2u) == 8u && state(5u, 2u) == 0u && state(6u, 2u) == 8u &&
           state(7u, 2u) == 8u;

  if (!valid) {
    std::cerr << "Cycles Principled Hair Chiang transition mismatch on "
              << backend << '\n';
    for (auto scenario = 0u; scenario < scenario_count; ++scenario) {
      std::cerr << "scenario " << scenario << ": type=" << state(scenario, 0u)
                << ", count=" << state(scenario, 1u)
                << ", left=" << state(scenario, 2u)
                << ", flags=" << state(scenario, 3u)
                << ", status=" << state(scenario, 4u)
                << ", offset=" << state(scenario, 5u) << '\n';
      for (auto field = 0u; field < output_stride; ++field) {
        const auto item = value(scenario, field);
        std::cerr << "  value[" << field << "] = (" << item.x << ", " << item.y
                  << ", " << item.z << ", " << item.w << ")\n";
      }
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

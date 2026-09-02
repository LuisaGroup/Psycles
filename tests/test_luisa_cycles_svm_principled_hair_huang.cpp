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
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace device_svm = psycles::luisa_backend::cycles_svm;

static_assert(closure::type_hair_huang ==
              static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_HUANG_ID));

inline constexpr std::int32_t normal_attribute_id = 0x13572468;
inline constexpr std::uint32_t output_stride = 8u;
inline constexpr std::uint32_t meta_stride = 6u;
inline constexpr std::uint32_t scenario_count = 11u;

struct Program {
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types{};
  std::uint32_t final_offset{};
};

struct HuangInput {
  luisa::float3 sigma;
  float roughness;
  float tilt;
  float eta;
  float aspect_ratio;
  float R;
  float TT;
  float TRT;
  std::int32_t attr_normal;
};

[[nodiscard]] SVMNodePrincipledHairBsdfData
huang_payload(const HuangInput &input) noexcept {
  return {
      .parametrization = NODE_PRINCIPLED_HAIR_DIRECT_ABSORPTION,
      .color = input_float3(0.13f, 0.47f, 0.81f),
      .tint = input_float3(0.9f, 0.8f, 0.7f),
      .absorption_coefficient =
          input_float3(input.sigma.x, input.sigma.y, input.sigma.z),
      .roughness = input_float(input.roughness),
      .random_roughness = input_float(0.0f),
      .offset = input_float(input.tilt),
      .ior = input_float(input.eta),
      .random = input_float(0.73f),
      .melanin = input_float(0.5f),
      .melanin_redness = input_float(0.2f),
      .coat = input_float(0.28f),
      .aspect_ratio = input_float(input.aspect_ratio),
      /* Huang uses longitudinal roughness for both directions. This distinct
       * immediate detects accidental reuse of the Chiang-only field. */
      .radial_roughness = input_float(0.91f),
      .random_color = input_float(0.0f),
      .R = input_float(input.R),
      .TT = input_float(input.TT),
      .TRT = input_float(input.TRT),
      .attr_random = ATTR_STD_CURVE_RANDOM,
      .attr_normal = input.attr_normal};
}

[[nodiscard]] Program make_program(const HuangInput &input,
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
      SVMNodeClosureBsdf{.closure_type = CLOSURE_BSDF_HAIR_HUANG_ID,
                         .mix_weight_offset =
                             zero_mix ? SVMStackOffset{0u} : SVM_STACK_INVALID,
                         ._pad = {0u, 0u, 0u}}));
  builder.add_node_data(huang_payload(input));
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

class HuangKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  UInt _scenario;

public:
  explicit HuangKernelGlobals(Expr<std::uint32_t> scenario) noexcept
      : _scenario{scenario} {}

  [[nodiscard]] Var<AttributeMap>
  attribute_map(Expr<std::uint32_t> offset) const noexcept override {
    const Bool has_normal = (_scenario == 1u) | (_scenario == 2u);
    const Bool normal_entry = has_normal & (offset == 0u);
    Var<AttributeMap> entry;
    entry.id = select(static_cast<luisa::ulong>(ATTR_STD_NONE),
                      static_cast<luisa::ulong>(normal_attribute_id),
                      normal_entry);
    entry.offset = 0;
    entry.element = select(
        static_cast<std::uint16_t>(ATTR_ELEMENT_NONE),
        static_cast<std::uint16_t>(ATTR_ELEMENT_CURVE_KEY), normal_entry);
    entry.type = static_cast<std::uint8_t>(NODE_ATTR_FLOAT3);
    entry.pad = static_cast<std::uint8_t>(0u);
    return entry;
  }

  [[nodiscard]] Var<packed_float3>
  attribute_float3(Expr<std::int32_t>) const noexcept override {
    const auto first = normalize(make_float3(0.4f, -0.1f, 0.9f));
    const auto second = normalize(make_float3(-0.2f, 0.7f, 0.5f));
    const auto value = select(first, second, _scenario == 2u);
    Var<packed_float3> packed;
    packed.x = value.x;
    packed.y = value.y;
    packed.z = value.z;
    return packed;
  }

  [[nodiscard]] Var<KernelCurve>
  curve(Expr<std::uint32_t> prim) const noexcept override {
    Var<KernelCurve> value;
    value.shader_id = 0;
    value.first_key = select(99, 3, prim == 0u);
    value.num_keys = 2;
    value.type = 0;
    return value;
  }

  [[nodiscard]] Int object_position_offset(
      Expr<std::uint32_t> object) const noexcept override {
    return select(101, 4, object == 2u);
  }

  [[nodiscard]] Float4
  curve_key(Expr<std::int32_t> key) const noexcept override {
    Float radius = 17.0f;
    $if(key == 7) { radius = 0.2f; }
    $elif(key == 8) { radius = 0.4f; };
    return make_float4(0.0f, 0.0f, 0.0f, radius);
  }
};

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *closures,
                 Expr<std::uint32_t> scenario) noexcept {
  const auto identity = make_float4x4(1.0f);
  const auto ribbon = (scenario == 2u) | (scenario == 3u) | (scenario == 9u);
  const auto primitive_type =
      select(device_svm::primitive_curve_thick,
             device_svm::primitive_curve_ribbon, ribbon);
  const auto v = select(0.27f, 1.0f, (scenario == 3u) | (scenario == 9u));
  return {make_float3(0.0f),
          normalize(make_float3(0.2f, -0.3f, 0.93f)),
          normalize(make_float3(-0.1f, 0.25f, 0.96f)),
          normalize(make_float3(0.35f, -0.18f, 0.92f)),
          primitive_type,
          0u,
          0u,
          0u,
          0u,
          0.25f,
          v,
          2u,
          0.0f,
          1.0f,
          0.1f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          make_float3(2.0f, 3.0f, 0.5f),
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
        HuangKernelGlobals kernel_globals{scenario};
        device_svm::ClosurePool closures{Capacity};
        auto shader_data = make_shader_data(&closures, scenario);
        const auto visibility =
            select(device_svm::path_ray_visibility_camera,
                   device_svm::path_ray_visibility_diffuse, scenario == 2u);
        const auto path_flag =
            select(0u, device_svm::path_ray_terminate, scenario == 9u);
        const device_svm::PathState path_state{visibility, path_flag};
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
          const auto common = closures.common(0u);
          type = common.type;
          output.write(output_base + 0u,
                       make_float4(common.weight, common.sample_weight));
          output.write(output_base + 1u, make_float4(common.N, 0.0f));
          $if(type == closure::type_hair_huang) {
            const auto hair = closures.huang_hair(0u);
            output.write(output_base + 1u,
                         make_float4(hair.common.N, hair.param.h));
            output.write(output_base + 2u,
                         make_float4(hair.param.sigma, hair.param.roughness));
            output.write(
                output_base + 3u,
                make_float4(hair.param.tilt, hair.param.eta,
                            hair.param.aspect_ratio,
                            hair.extra.pixel_coverage));
            output.write(output_base + 4u,
                         make_float4(hair.extra.R, hair.extra.TT,
                                     hair.extra.TRT, hair.extra.radius));
            output.write(output_base + 5u,
                         make_float4(hair.extra.Y, hair.extra.e2));
            output.write(output_base + 6u, make_float4(hair.extra.Z, 0.0f));
            output.write(output_base + 7u, make_float4(hair.extra.wi, 0.0f));
          }
          $else {
            output.write(output_base + 2u,
                         make_float4(
                             shader_data.closure_transparent_extinction, 0.0f));
          };
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
                        float tolerance = 4.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 4.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  constexpr HuangInput circular{.sigma = {0.2f, 0.7f, 1.1f},
                                .roughness = -0.2f,
                                .tilt = 0.41f,
                                .eta = 1.33f,
                                .aspect_ratio = 1.0f,
                                .R = -0.2f,
                                .TT = 0.7f,
                                .TRT = 1.1f,
                                .attr_normal = ATTR_STD_NONE};
  constexpr HuangInput major_axis{.sigma = {0.12f, 0.34f, 0.56f},
                                  .roughness = 0.58f,
                                  .tilt = -0.23f,
                                  .eta = 1.42f,
                                  .aspect_ratio = 2.0f,
                                  .R = 0.3f,
                                  .TT = 0.6f,
                                  .TRT = 0.9f,
                                  .attr_normal = normal_attribute_id};
  constexpr HuangInput minor_axis{.sigma = {0.08f, 0.16f, 0.32f},
                                  .roughness = 1.4f,
                                  .tilt = 0.19f,
                                  .eta = 1.55f,
                                  .aspect_ratio = 0.5f,
                                  .R = 1.0f,
                                  .TT = 0.5f,
                                  .TRT = 0.25f,
                                  .attr_normal = normal_attribute_id};
  constexpr HuangInput transparent{.sigma = {0.2f, 0.7f, 1.1f},
                                   .roughness = 0.37f,
                                   .tilt = 0.19f,
                                   .eta = 1.55f,
                                   .aspect_ratio = 1.0f,
                                   .R = 1.0f,
                                   .TT = 1.0f,
                                   .TRT = 1.0f,
                                   .attr_normal = ATTR_STD_NONE};
  constexpr HuangInput disabled{.sigma = {0.2f, 0.7f, 1.1f},
                                .roughness = 0.37f,
                                .tilt = 0.19f,
                                .eta = 1.55f,
                                .aspect_ratio = 1.0f,
                                .R = 0.0f,
                                .TT = -0.5f,
                                .TRT = 0.0f,
                                .attr_normal = ATTR_STD_NONE};
  const std::array programs{
      make_program(circular),
      make_program(major_axis),
      make_program(minor_axis),
      make_program(transparent),
      make_program(disabled),
      make_program(circular),
      make_program(circular, {1.0e-6f, 1.0e-6f, 1.0e-6f}),
      make_program(circular, {0.8f, 0.4f, 0.2f}, true),
      make_program(circular),
      make_program(transparent),
      make_program(circular)};

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
      device.compile(transition_kernel<2u>(programs[0u].node_types), options);
  auto one_slot_shader =
      device.compile(transition_kernel<1u>(programs[5u].node_types), options);
  auto zero_mix_shader =
      device.compile(transition_kernel<2u>(programs[7u].node_types), options);
  auto feature_erased_shader = device.compile(
      transition_kernel<2u, 0u>(programs[8u].node_types), options);
  auto zero_capacity_shader =
      device.compile(transition_kernel<0u>(programs[10u].node_types), options);

  for (auto index = 0u; index < scenario_count; ++index) {
    stream << word_buffers[index].copy_from(programs[index].words.data());
  }
  for (auto index : {0u, 1u, 2u, 3u, 4u, 6u, 9u}) {
    stream
        << full_shader(word_buffers[index], output, meta, index).dispatch(1u);
  }
  stream << one_slot_shader(word_buffers[5u], output, meta, 5u).dispatch(1u)
         << zero_mix_shader(word_buffers[7u], output, meta, 7u).dispatch(1u)
         << feature_erased_shader(word_buffers[8u], output, meta, 8u)
                .dispatch(1u)
         << zero_capacity_shader(word_buffers[10u], output, meta, 10u)
                .dispatch(1u);
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
  constexpr auto hair_flags = device_svm::shader_data_bsdf |
                              device_svm::shader_data_bsdf_has_eval |
                              device_svm::shader_data_bsdf_has_transmission;
  constexpr auto transparent_flags =
      device_svm::shader_data_bsdf | device_svm::shader_data_transparent;
  constexpr auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  constexpr std::array<luisa::float4, 32u> oracle{
      luisa::float4{0.800000012f, 0.400000006f, 0.200000003f, 0.466666698f},
      luisa::float4{0.794034362f, -0.463883251f, -0.392838061f, 0.0675399899f},
      luisa::float4{0.200000003f, 0.699999988f, 1.10000002f, 0.00100000005f},
      luisa::float4{-0.409999996f, 1.33000004f, 1.0f, 0.200000003f},
      luisa::float4{0.0f, 0.699999988f, 1.10000002f, 1.0f},
      luisa::float4{0.549442232f, 0.824163318f, 0.137360558f, 0.0f},
      luisa::float4{0.260043472f, -0.32491082f, 0.909291029f, 0.0f},
      luisa::float4{7.50358709e-09f, 0.170216486f, 0.985406637f, 0.0f},
      luisa::float4{0.800000012f, 0.400000006f, 0.200000003f, 0.466666698f},
      luisa::float4{0.268715411f, -0.329966754f, 0.904938698f, 0.0675399899f},
      luisa::float4{0.119999997f, 0.340000004f, 0.560000002f, 0.579999983f},
      luisa::float4{0.230000004f, 1.41999996f, 0.5f, 0.200000003f},
      luisa::float4{0.300000012f, 0.600000024f, 0.899999976f, 0.500089765f},
      luisa::float4{0.549442232f, 0.824163318f, 0.137360558f, 0.75f},
      luisa::float4{-0.791141689f, 0.460300654f, 0.402763069f, 0.0f},
      luisa::float4{0.985347748f, 0.170216486f, 0.0107813049f, 0.0f},
      luisa::float4{0.800000012f, 0.400000006f, 0.200000003f, 0.466666698f},
      luisa::float4{-0.70403403f, 0.368139923f, 0.607296586f, -0.270000011f},
      luisa::float4{0.0799999982f, 0.159999996f, 0.319999993f, 1.0f},
      luisa::float4{-0.189999998f, 1.54999995f, 0.5f, 1.0f},
      luisa::float4{1.0f, 0.5f, 0.25f, 0.976373792f},
      luisa::float4{0.549442232f, 0.824163318f, 0.137360558f, 0.75f},
      luisa::float4{-0.449943662f, 0.430380911f, -0.782510698f, 0.0f},
      luisa::float4{0.245876014f, 0.170216486f, -0.954238653f, 0.0f},
      luisa::float4{0.800000012f, 0.400000006f, 0.200000003f, 0.466666698f},
      luisa::float4{0.200511962f, -0.300767958f, 0.932380617f, 0.0f},
      luisa::float4{0.800000012f, 0.400000006f, 0.200000003f, 0.0f},
      luisa::float4{0.0f}, luisa::float4{0.0f}, luisa::float4{0.0f},
      luisa::float4{0.0f}, luisa::float4{0.0f}};

  bool valid = true;
  for (auto scenario = 0u; scenario < 3u; ++scenario) {
    valid &= state(scenario, 0u) == closure::type_hair_huang &&
             state(scenario, 1u) == 1u && state(scenario, 2u) == 0u &&
             state(scenario, 3u) == hair_flags &&
             state(scenario, 4u) == ended &&
             state(scenario, 5u) == programs[scenario].final_offset;
    for (auto field = 0u; field < output_stride; ++field) {
      valid &= near(value(scenario, field),
                    oracle[scenario * output_stride + field]);
    }
  }
  for (auto scenario : {3u, 9u}) {
    valid &= state(scenario, 0u) == closure::type_transparent &&
             state(scenario, 1u) == 1u &&
             state(scenario, 2u) == (scenario == 9u ? 0u : 1u) &&
             state(scenario, 3u) == transparent_flags &&
             state(scenario, 4u) == ended &&
             state(scenario, 5u) == programs[scenario].final_offset;
    for (auto field = 0u; field < output_stride; ++field) {
      valid &= near(value(scenario, field), oracle[24u + field]);
    }
  }
  for (auto scenario : {4u, 5u, 6u, 7u, 8u, 10u}) {
    valid &=
        state(scenario, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
        state(scenario, 1u) == 0u && state(scenario, 3u) == 0u &&
        state(scenario, 4u) == ended &&
        state(scenario, 5u) == programs[scenario].final_offset;
  }
  valid &= state(4u, 2u) == 2u && state(5u, 2u) == 1u &&
           state(6u, 2u) == 2u && state(7u, 2u) == 2u &&
           state(8u, 2u) == 2u && state(10u, 2u) == 0u;

  if (!valid) {
    std::cerr << "Cycles Principled Hair Huang transition mismatch on "
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
        std::cerr << "  value[" << field << "] = (" << item.x << ", "
                  << item.y << ", " << item.z << ", " << item.w << ")\n";
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

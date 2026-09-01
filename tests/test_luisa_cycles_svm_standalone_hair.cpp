#include <psycles/luisa/cycles_svm.h>

#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;

/* Exact compact images from unmodified Cycles 5.2.1. Only each shader jump
 * is relocated from its final global offsets to the local test image. */
constexpr std::array<std::uint32_t, 18u> reflection_words{
    0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u, 0x00000005u,
    0x3f4ccccdu, 0x3f4ccccdu, 0x3f4ccccdu, 0x00000002u, 0x00000013u,
    0x000000ffu, 0x3dcccccdu, 0x3f800000u, 0x00000000u, 0x000000ffu,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 23u> transmission_words{
    0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u, 0x00000013u,
    0x00000000u, 0x3e99999au, 0x3ecccccdu, 0x00000000u, 0x00000005u,
    0x3f547ae1u, 0x3e2e147bu, 0x3f051eb8u, 0x00000002u, 0x00000017u,
    0x000000ffu, 0x3951b717u, 0x3fb33333u, 0x3e8a3d71u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u};

[[nodiscard]] consteval std::array<std::uint32_t, 21u>
make_mixed_reflection_words(std::uint32_t mix_bits) noexcept {
  std::array<std::uint32_t, 21u> result{};
  result[0u] = reflection_words[0u];
  result[1u] = 4u;
  result[2u] = 19u;
  result[3u] = 20u;
  result[4u] = static_cast<std::uint32_t>(NODE_VALUE_F);
  result[5u] = mix_bits;
  result[6u] = 3u;
  for (auto index = 4u; index < reflection_words.size(); ++index) {
    result[index + 3u] = reflection_words[index];
  }
  result[13u] = 3u;
  return result;
}

constexpr auto quarter_mix_words =
    make_mixed_reflection_words(std::bit_cast<std::uint32_t>(0.25f));
constexpr auto zero_mix_words = make_mixed_reflection_words(0u);
inline constexpr std::uint32_t output_stride = 4u;
inline constexpr std::uint32_t meta_stride = 6u;
inline constexpr std::uint32_t scenario_count = 9u;

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *closures,
                 Expr<std::uint32_t> primitive_type) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(4.0f, -2.0f, 0.5f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          primitive_type,
          0u,
          0u,
          0u,
          0u,
          0.25f,
          0.25f,
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
          make_float3(0.0f, 3.0f, 4.0f),
          identity,
          identity,
          0u,
          closures};
}

[[nodiscard]] std::array<bool, NODE_NUM> node_types() noexcept {
  std::array<bool, NODE_NUM> result{};
  result[NODE_END] = true;
  result[NODE_SHADER_JUMP] = true;
  result[NODE_VALUE_F] = true;
  result[NODE_VALUE_V] = true;
  result[NODE_CLOSURE_SET_WEIGHT] = true;
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

template <std::size_t Capacity,
          std::uint32_t FeatureMask = device_svm::kernel_feature_node_bsdf>
[[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>, std::uint32_t>{
      [used](BufferUInt words, BufferFloat4 output, BufferUInt meta,
             UInt scenario) noexcept {
        psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
        device_svm::ClosurePool closures{Capacity};
        const UInt primitive_type =
            select(device_svm::primitive_triangle,
                   device_svm::primitive_curve_thick, scenario == 2u);
        auto shader_data = make_shader_data(&closures, primitive_type);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                               FeatureMask, used, identity_transform_state(),
                               shader_data, path_state, result);

        const auto output_base = scenario * output_stride;
        const auto meta_base = scenario * meta_stride;
        for (auto index = 0u; index < output_stride; ++index) {
          output.write(output_base + index, make_float4(0.0f));
        }
        meta.write(meta_base + 0u, static_cast<std::uint32_t>(CLOSURE_NONE_ID));
        $if(closures.count() != 0u) {
          const auto closure = closures.hair(0u);
          output.write(
              output_base + 0u,
              make_float4(closure.common.weight, closure.common.sample_weight));
          output.write(output_base + 1u, make_float4(closure.common.N, 0.0f));
          output.write(output_base + 2u,
                       make_float4(closure.param.T, closure.param.roughness1));
          output.write(output_base + 3u,
                       make_float4(closure.param.roughness2,
                                   closure.param.offset, 0.0f, 0.0f));
          meta.write(meta_base + 0u, closure.common.type);
        };
        meta.write(meta_base + 1u, closures.count());
        meta.write(meta_base + 2u, closures.left());
        meta.write(meta_base + 3u, shader_data.flag);
        meta.write(meta_base + 4u, result.status);
        meta.write(meta_base + 5u, result.final_offset);
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-6f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 2.0e-6f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();

  auto curve_words = reflection_words;
  curve_words[11u] = std::bit_cast<std::uint32_t>(0.25f);
  curve_words[12u] = std::bit_cast<std::uint32_t>(0.5f);
  curve_words[13u] = std::bit_cast<std::uint32_t>(0.31f);
  auto signed_weight = reflection_words;
  signed_weight[5u] = std::bit_cast<std::uint32_t>(-0.3f);
  signed_weight[6u] = std::bit_cast<std::uint32_t>(0.6f);
  signed_weight[7u] = std::bit_cast<std::uint32_t>(0.3f);
  auto below_cutoff = reflection_words;
  below_cutoff[5u] = std::bit_cast<std::uint32_t>(1.0e-6f);
  below_cutoff[6u] = std::bit_cast<std::uint32_t>(1.0e-6f);
  below_cutoff[7u] = std::bit_cast<std::uint32_t>(1.0e-6f);

  auto reflection_buffer =
      device.create_buffer<std::uint32_t>(reflection_words.size());
  auto transmission_buffer =
      device.create_buffer<std::uint32_t>(transmission_words.size());
  auto curve_buffer = device.create_buffer<std::uint32_t>(curve_words.size());
  auto quarter_buffer =
      device.create_buffer<std::uint32_t>(quarter_mix_words.size());
  auto signed_buffer =
      device.create_buffer<std::uint32_t>(signed_weight.size());
  auto cutoff_buffer = device.create_buffer<std::uint32_t>(below_cutoff.size());
  auto zero_mix_buffer =
      device.create_buffer<std::uint32_t>(zero_mix_words.size());
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(scenario_count * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto full_shader = device.compile(transition_kernel<8u>(), options);
  auto empty_shader = device.compile(transition_kernel<0u>(), options);
  auto feature_erased_shader =
      device.compile(transition_kernel<8u, 0u>(), options);

  std::array<luisa::float4, scenario_count * output_stride> actual{};
  std::array<std::uint32_t, scenario_count * meta_stride> actual_meta{};
  stream
      << reflection_buffer.copy_from(reflection_words.data())
      << transmission_buffer.copy_from(transmission_words.data())
      << curve_buffer.copy_from(curve_words.data())
      << quarter_buffer.copy_from(quarter_mix_words.data())
      << signed_buffer.copy_from(signed_weight.data())
      << cutoff_buffer.copy_from(below_cutoff.data())
      << zero_mix_buffer.copy_from(zero_mix_words.data())
      << full_shader(reflection_buffer, output, meta, 0u).dispatch(1u)
      << full_shader(transmission_buffer, output, meta, 1u).dispatch(1u)
      << full_shader(curve_buffer, output, meta, 2u).dispatch(1u)
      << full_shader(quarter_buffer, output, meta, 3u).dispatch(1u)
      << full_shader(signed_buffer, output, meta, 4u).dispatch(1u)
      << full_shader(cutoff_buffer, output, meta, 5u).dispatch(1u)
      << empty_shader(reflection_buffer, output, meta, 6u).dispatch(1u)
      << full_shader(zero_mix_buffer, output, meta, 7u).dispatch(1u)
      << feature_erased_shader(reflection_buffer, output, meta, 8u).dispatch(1u)
      << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
      << synchronize();

  auto value = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual[scenario * output_stride + field];
  };
  auto state = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual_meta[scenario * meta_stride + field];
  };
  constexpr auto reflection_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_REFLECTION_ID);
  constexpr auto transmission_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_TRANSMISSION_ID);
  constexpr auto reflection_flags =
      device_svm::shader_data_bsdf | device_svm::shader_data_bsdf_has_eval;
  constexpr auto transmission_flags =
      reflection_flags | device_svm::shader_data_bsdf_has_transmission;
  constexpr auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);

  auto valid = state(0u, 0u) == reflection_type && state(0u, 1u) == 1u &&
               state(0u, 2u) == 7u && state(0u, 3u) == reflection_flags &&
               state(0u, 4u) == ended && state(0u, 5u) == 16u &&
               near(value(0u, 0u).xyz(), luisa::float3{0.8f}) &&
               near(value(0u, 0u).w, 0.8f) &&
               near(value(0u, 1u).xyz(), luisa::float3{0.0f, 0.0f, 1.0f}) &&
               near(value(0u, 2u).xyz(), luisa::float3{0.0f, 0.6f, 0.8f}) &&
               near(value(0u, 2u).w, 0.1f) && near(value(0u, 3u).x, 1.0f) &&
               near(value(0u, 3u).y, 0.0f);

  valid &= state(1u, 0u) == transmission_type && state(1u, 1u) == 1u &&
           state(1u, 2u) == 7u && state(1u, 3u) == transmission_flags &&
           state(1u, 4u) == ended && state(1u, 5u) == 21u &&
           near(value(1u, 0u).xyz(), luisa::float3{0.83f, 0.17f, 0.52f}) &&
           near(value(1u, 0u).w, 0.5066666603088379f) &&
           near(value(1u, 2u).xyz(), luisa::float3{0.6f, 0.8f, 0.0f}) &&
           near(value(1u, 2u).w, 0.001f) && near(value(1u, 3u).x, 1.0f) &&
           near(value(1u, 3u).y, -0.27f);

  valid &= state(2u, 0u) == reflection_type && state(2u, 5u) == 16u &&
           near(value(2u, 2u).xyz(), luisa::float3{0.6f, 0.8f, 0.0f}) &&
           near(value(2u, 2u).w, 0.25f) && near(value(2u, 3u).x, 0.5f) &&
           near(value(2u, 3u).y, -0.31f);

  valid &= state(3u, 0u) == reflection_type && state(3u, 5u) == 19u &&
           near(value(3u, 0u).xyz(), luisa::float3{0.2f}) &&
           near(value(3u, 0u).w, 0.2f);

  valid &= state(4u, 0u) == reflection_type &&
           near(value(4u, 0u).xyz(), luisa::float3{0.0f, 0.6f, 0.3f}) &&
           near(value(4u, 0u).w, 0.3f);

  for (auto scenario : {5u, 7u}) {
    valid &=
        state(scenario, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
        state(scenario, 1u) == 0u && state(scenario, 2u) == 8u &&
        state(scenario, 3u) == 0u && state(scenario, 4u) == ended;
  }
  valid &= state(5u, 5u) == 16u && state(7u, 5u) == 19u;

  valid &= state(6u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           state(6u, 1u) == 0u && state(6u, 2u) == 0u && state(6u, 3u) == 0u &&
           state(6u, 4u) == ended && state(6u, 5u) == 16u;

  valid &= state(8u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           state(8u, 1u) == 0u && state(8u, 2u) == 8u && state(8u, 3u) == 0u &&
           state(8u, 4u) == ended && state(8u, 5u) == 16u;

  if (!valid) {
    std::cerr << "Cycles standalone Hair transition mismatch on " << backend
              << '\n';
    for (auto scenario = 0u; scenario < scenario_count; ++scenario) {
      const auto common = value(scenario, 0u);
      const auto tangent = value(scenario, 2u);
      const auto scalars = value(scenario, 3u);
      std::cerr << "scenario " << scenario << ": type=" << state(scenario, 0u)
                << ", count=" << state(scenario, 1u)
                << ", left=" << state(scenario, 2u)
                << ", flags=" << state(scenario, 3u)
                << ", status=" << state(scenario, 4u)
                << ", offset=" << state(scenario, 5u) << ", common=("
                << common.x << ", " << common.y << ", " << common.z << ", "
                << common.w << "), T/r1=(" << tangent.x << ", " << tangent.y
                << ", " << tangent.z << ", " << tangent.w << "), r2/o=("
                << scalars.x << ", " << scalars.y << ")\n";
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

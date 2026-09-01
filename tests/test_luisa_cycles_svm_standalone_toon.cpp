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

/* Compact surface images copied word-for-word from the final global buffers
 * of unmodified Cycles 5.2.1 standalone Toon probes. Only the four-word
 * shader jump is relocated from (89, 103, 104) to (4, 18, 19). */
constexpr std::array<std::uint32_t, 20u> diffuse_words{
    0x00000001u, 0x00000004u, 0x00000012u, 0x00000013u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3e9eb852u, 0x3f3ae148u,
    0x3e428f5cu, 0x00000002u, 0x00000008u, 0x000000ffu, 0x3ebd70a4u,
    0x3e570a3du, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 20u> glossy_words{
    0x00000001u, 0x00000004u, 0x00000012u, 0x00000013u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3f570a3du, 0x3e6147aeu,
    0x3f0f5c29u, 0x00000002u, 0x00000012u, 0x000000ffu, 0x3f2147aeu,
    0x3e0f5c29u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

[[nodiscard]] consteval std::array<std::uint32_t, 23u>
make_mixed_diffuse_words() noexcept {
  std::array<std::uint32_t, 23u> result{};
  result[0u] = diffuse_words[0u];
  result[1u] = 4u;
  result[2u] = 21u;
  result[3u] = 22u;
  result[4u] = static_cast<std::uint32_t>(NODE_VALUE_F);
  result[5u] = std::bit_cast<std::uint32_t>(0.25f);
  result[6u] = 3u;
  for (auto index = 4u; index < diffuse_words.size(); index++) {
    result[index + 3u] = diffuse_words[index];
  }
  /* SVMNodeClosureBsdf::mix_weight_offset is its low byte. */
  result[16u] = 3u;
  return result;
}

[[nodiscard]] consteval std::array<std::uint32_t, 25u>
make_authored_normal_words() noexcept {
  std::array<std::uint32_t, 25u> result{};
  result[0u] = glossy_words[0u];
  result[1u] = 4u;
  result[2u] = 23u;
  result[3u] = 24u;
  result[4u] = static_cast<std::uint32_t>(NODE_VALUE_V);
  result[5u] = 3u;
  result[6u] = std::bit_cast<std::uint32_t>(0.0f);
  result[7u] = std::bit_cast<std::uint32_t>(3.0f);
  result[8u] = std::bit_cast<std::uint32_t>(4.0f);
  for (auto index = 4u; index < glossy_words.size(); index++) {
    result[index + 5u] = glossy_words[index];
  }
  /* SVMNodeToonBsdfData::normal_offset is its low byte. */
  result[21u] = 3u;
  return result;
}

constexpr auto mixed_diffuse_words = make_mixed_diffuse_words();
constexpr auto authored_normal_words = make_authored_normal_words();
inline constexpr std::uint32_t size_word = 14u;
inline constexpr std::uint32_t smooth_word = 15u;
inline constexpr std::uint32_t output_stride = 3u;
inline constexpr std::uint32_t meta_stride = 6u;
inline constexpr std::uint32_t scenario_count = 10u;
inline constexpr float half_pi = 1.57079632679489661923f;

class ToonKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  Bool _reflective_caustics;

public:
  explicit ToonKernelGlobals(Expr<bool> reflective_caustics) noexcept
      : _reflective_caustics{reflective_caustics} {}

  [[nodiscard]] Bool caustics_reflective() const noexcept override {
    return _reflective_caustics;
  }
};

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *closures) noexcept {
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
          make_float3(1.0f, 0.0f, 0.0f),
          make_float3(0.0f, 1.0f, 0.0f),
          identity,
          identity,
          0u,
          closures};
}

[[nodiscard]] std::array<bool, NODE_NUM> node_types() noexcept {
  std::array<bool, NODE_NUM> result{};
  result[NODE_END] = true;
  result[NODE_SHADER_JUMP] = true;
  result[NODE_GEOMETRY] = true;
  result[NODE_VALUE_F] = true;
  result[NODE_VALUE_V] = true;
  result[NODE_CLOSURE_SET_WEIGHT] = true;
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

template <std::size_t Capacity> [[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>, std::uint32_t, bool, std::uint32_t>{
      [used](BufferUInt words, BufferFloat4 output, BufferUInt meta,
             UInt scenario, Bool reflective_caustics,
             UInt visibility) noexcept {
        const ToonKernelGlobals kernel_globals{reflective_caustics};
        device_svm::ClosurePool closures{Capacity};
        auto shader_data = make_shader_data(&closures);
        const device_svm::PathState path_state{visibility, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                               device_svm::kernel_feature_node_bsdf, used,
                               identity_transform_state(), shader_data,
                               path_state, result);

        const auto output_base = scenario * output_stride;
        const auto meta_base = scenario * meta_stride;
        for (auto index = 0u; index < output_stride; index++) {
          output.write(output_base + index, make_float4(0.0f));
        }
        meta.write(meta_base + 0u, static_cast<std::uint32_t>(CLOSURE_NONE_ID));
        $if(closures.count() != 0u) {
          const auto closure = closures.toon(0u);
          output.write(
              output_base + 0u,
              make_float4(closure.common.weight, closure.common.sample_weight));
          output.write(output_base + 1u, make_float4(closure.common.N, 0.0f));
          output.write(output_base + 2u,
                       make_float4(closure.param.size, closure.param.smooth,
                                   0.0f, 0.0f));
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

  auto low = diffuse_words;
  low[size_word] = std::bit_cast<std::uint32_t>(-0.25f);
  low[smooth_word] = std::bit_cast<std::uint32_t>(-0.40f);
  auto high = glossy_words;
  high[size_word] = std::bit_cast<std::uint32_t>(1.40f);
  high[smooth_word] = std::bit_cast<std::uint32_t>(1.30f);

  auto diffuse = device.create_buffer<std::uint32_t>(diffuse_words.size());
  auto glossy = device.create_buffer<std::uint32_t>(glossy_words.size());
  auto low_buffer = device.create_buffer<std::uint32_t>(low.size());
  auto high_buffer = device.create_buffer<std::uint32_t>(high.size());
  auto mixed = device.create_buffer<std::uint32_t>(mixed_diffuse_words.size());
  auto normal =
      device.create_buffer<std::uint32_t>(authored_normal_words.size());
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(scenario_count * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto full_shader = device.compile(transition_kernel<8u>(), options);
  auto empty_shader = device.compile(transition_kernel<0u>(), options);

  std::array<luisa::float4, scenario_count * output_stride> actual{};
  std::array<std::uint32_t, scenario_count * meta_stride> actual_meta{};
  const auto camera = device_svm::path_ray_visibility_camera;
  const auto diffuse_ray = device_svm::path_ray_visibility_diffuse;
  stream
      << diffuse.copy_from(diffuse_words.data())
      << glossy.copy_from(glossy_words.data())
      << low_buffer.copy_from(low.data()) << high_buffer.copy_from(high.data())
      << mixed.copy_from(mixed_diffuse_words.data())
      << normal.copy_from(authored_normal_words.data())
      << full_shader(diffuse, output, meta, 0u, true, camera).dispatch(1u)
      << full_shader(glossy, output, meta, 1u, true, camera).dispatch(1u)
      << full_shader(low_buffer, output, meta, 2u, true, camera).dispatch(1u)
      << full_shader(high_buffer, output, meta, 3u, true, camera).dispatch(1u)
      << full_shader(mixed, output, meta, 4u, true, camera).dispatch(1u)
      << full_shader(normal, output, meta, 5u, true, camera).dispatch(1u)
      << full_shader(glossy, output, meta, 6u, false, diffuse_ray).dispatch(1u)
      << full_shader(diffuse, output, meta, 7u, false, diffuse_ray).dispatch(1u)
      << full_shader(glossy, output, meta, 8u, false, camera).dispatch(1u)
      << empty_shader(diffuse, output, meta, 9u, true, camera).dispatch(1u)
      << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
      << synchronize();

  auto value = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual[scenario * output_stride + field];
  };
  auto state = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual_meta[scenario * meta_stride + field];
  };
  constexpr auto expected_flags =
      device_svm::shader_data_bsdf | device_svm::shader_data_bsdf_has_eval;
  constexpr auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};

  auto valid = state(0u, 0u) ==
                   static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_TOON_ID) &&
               state(0u, 1u) == 1u && state(0u, 2u) == 7u &&
               state(0u, 3u) == expected_flags && state(0u, 4u) == ended &&
               state(0u, 5u) == 18u &&
               near(value(0u, 0u).xyz(), luisa::float3{0.31f, 0.73f, 0.19f}) &&
               near(value(0u, 0u).w, 0.4100000262260437f) &&
               near(value(0u, 1u).xyz(), up) &&
               near(value(0u, 2u).x, 0.37f * half_pi) &&
               near(value(0u, 2u).y, 0.21f * half_pi);

  valid &= state(1u, 0u) ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_GLOSSY_TOON_ID) &&
           state(1u, 1u) == 1u && state(1u, 2u) == 7u &&
           state(1u, 3u) == expected_flags && state(1u, 4u) == ended &&
           state(1u, 5u) == 18u &&
           near(value(1u, 0u).xyz(), luisa::float3{0.84f, 0.22f, 0.56f}) &&
           near(value(1u, 0u).w, 0.5399999618530273f) &&
           near(value(1u, 1u).xyz(), up) &&
           near(value(1u, 2u).x, 0.63f * half_pi) &&
           near(value(1u, 2u).y, 0.14f * half_pi);

  valid &= state(2u, 0u) ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_TOON_ID) &&
           near(value(2u, 2u).x, 1.0e-5f * half_pi) &&
           near(value(2u, 2u).y, 0.0f);
  valid &= state(3u, 0u) ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_GLOSSY_TOON_ID) &&
           near(value(3u, 2u).x, half_pi) && near(value(3u, 2u).y, half_pi);

  valid &=
      state(4u, 0u) ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_TOON_ID) &&
      state(4u, 5u) == 21u &&
      near(value(4u, 0u).xyz(), luisa::float3{0.31f, 0.73f, 0.19f} * 0.25f) &&
      near(value(4u, 0u).w, 0.4100000262260437f * 0.25f);

  valid &= state(5u, 0u) ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_GLOSSY_TOON_ID) &&
           state(5u, 5u) == 23u &&
           near(value(5u, 1u).xyz(), luisa::float3{0.0f, 0.6f, 0.8f});

  /* Cycles' caustics trick is a Glossy-only allocation guard. It still
   * consumes the typed payload; Diffuse Toon and non-diffuse rays remain
   * active under the same integrator setting. */
  valid &= state(6u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           state(6u, 1u) == 0u && state(6u, 2u) == 8u && state(6u, 3u) == 0u &&
           state(6u, 4u) == ended && state(6u, 5u) == 18u;
  valid &= state(7u, 0u) ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_TOON_ID) &&
           state(7u, 1u) == 1u && state(7u, 3u) == expected_flags;
  valid &= state(8u, 0u) ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_GLOSSY_TOON_ID) &&
           state(8u, 1u) == 1u && state(8u, 3u) == expected_flags;

  valid &= state(9u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           state(9u, 1u) == 0u && state(9u, 2u) == 0u && state(9u, 3u) == 0u &&
           state(9u, 4u) == ended && state(9u, 5u) == 18u;

  if (!valid) {
    std::cerr << "Cycles standalone Toon transition mismatch on " << backend
              << '\n';
    for (auto scenario = 0u; scenario < scenario_count; scenario++) {
      const auto common = value(scenario, 0u);
      const auto payload = value(scenario, 2u);
      std::cerr << "scenario " << scenario << ": type=" << state(scenario, 0u)
                << ", count=" << state(scenario, 1u)
                << ", left=" << state(scenario, 2u)
                << ", flags=" << state(scenario, 3u)
                << ", status=" << state(scenario, 4u)
                << ", offset=" << state(scenario, 5u) << ", common=("
                << common.x << ", " << common.y << ", " << common.z << ", "
                << common.w << "), payload=(" << payload.x << ", " << payload.y
                << ")\n";
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

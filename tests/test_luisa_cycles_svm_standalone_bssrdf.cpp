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

/* Compact tails copied word-for-word from the final global SVM buffers of the
 * unmodified Cycles 5.2.1 standalone BSSRDF probes. Only the four-word shader
 * jump is relocated from global offsets (89, 108, 109) to (4, 23, 24). */
constexpr std::array<std::uint32_t, 25u> random_walk_words{
    0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3ebd70a4u, 0x3f1eb852u,
    0x3e0f5c29u, 0x00000002u, 0x00000020u, 0x000000ffu, 0x3f8ccccdu,
    0x3ee66666u, 0x3db851ecu, 0x3cfdf3b6u, 0x3fc66666u, 0x3f99999au,
    0x3e8f5c29u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 25u> burley_words{
    0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3e570a3du, 0x3f3d70a4u,
    0x3ef5c28fu, 0x00000002u, 0x0000001fu, 0x000000ffu, 0x3f4ccccdu,
    0x3ea3d70au, 0x3d4ccccdu, 0x3cbc6a7fu, 0x3fb33333u, 0x00000000u,
    0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

[[nodiscard]] consteval std::array<std::uint32_t, 28u>
make_mixed_random_walk_words() noexcept {
  std::array<std::uint32_t, 28u> result{};
  result[0] = random_walk_words[0];
  result[1] = 4u;
  result[2] = 26u;
  result[3] = 27u;
  result[4] = static_cast<std::uint32_t>(NODE_VALUE_F);
  result[5] = std::bit_cast<std::uint32_t>(0.25f);
  /* The following external NODE_GEOMETRY owns stack [0, 3). */
  result[6] = 3u;
  for (auto index = 4u; index < random_walk_words.size(); index++) {
    result[index + 3u] = random_walk_words[index];
  }
  /* SVMNodeClosureBsdf::mix_weight_offset is the low byte of this word. */
  result[16] = 3u;
  return result;
}

constexpr auto mixed_random_walk_words = make_mixed_random_walk_words();

inline constexpr std::uint32_t closure_type_word = 12u;
inline constexpr std::uint32_t scale_word = 17u;
inline constexpr std::uint32_t output_stride = 7u;
inline constexpr std::uint32_t meta_stride = 7u;
inline constexpr std::uint32_t scenario_count = 9u;

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *closures,
                 Expr<std::uint32_t> tilted_normal) noexcept {
  const auto identity = make_float4x4(1.0f);
  const auto normal = select(make_float3(0.0f, 0.0f, 1.0f),
                             make_float3(0.0f, 2.0f, 0.0f),
                             tilted_normal != 0u);
  return {make_float3(0.0f), normal, normal, normal,
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
  result[NODE_CLOSURE_SET_WEIGHT] = true;
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

template<std::size_t Capacity> [[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>, std::uint32_t, std::uint32_t,
                  std::uint32_t>{
      [used](BufferUInt words, BufferFloat4 output, BufferUInt meta,
             UInt scenario, UInt path_flag, UInt tilted_normal) noexcept {
        const psycles::test_support::DefaultCyclesSvmKernelGlobals
            kernel_globals;
        device_svm::ClosurePool closures{Capacity};
        auto shader_data = make_shader_data(&closures, tilted_normal);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, path_flag};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
            device_svm::kernel_feature_node_bsdf, used,
            identity_transform_state(), shader_data, path_state, result);

        const auto output_base = scenario * output_stride;
        const auto meta_base = scenario * meta_stride;
        for (auto i = 0u; i < output_stride; i++) {
          output.write(output_base + i, make_float4(0.0f));
        }
        for (auto i = 0u; i < 2u; i++) {
          meta.write(meta_base + i,
                     static_cast<std::uint32_t>(CLOSURE_NONE_ID));
          $if(i < closures.count()) {
            const auto closure = closures.common(i);
            output.write(output_base + i,
                         make_float4(closure.weight, closure.sample_weight));
            output.write(output_base + 2u + i,
                         make_float4(closure.N, 0.0f));
            meta.write(meta_base + i, closure.type);
          };
        }
        $if(closures.count() != 0u) {
          const auto bssrdf = closures.bssrdf(0u);
          output.write(output_base + 4u,
                       make_float4(bssrdf.param.radius,
                                   bssrdf.param.anisotropy));
          output.write(output_base + 5u,
                       make_float4(bssrdf.param.albedo, bssrdf.param.ior));
          output.write(output_base + 6u,
                       make_float4(bssrdf.param.alpha));
        };
        meta.write(meta_base + 2u, closures.count());
        meta.write(meta_base + 3u, closures.left());
        meta.write(meta_base + 4u, shader_data.flag);
        meta.write(meta_base + 5u, result.status);
        meta.write(meta_base + 6u, result.final_offset);
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 5.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 5.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();

  auto skin_words = random_walk_words;
  skin_words[closure_type_word] =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID);
  auto legacy_words = random_walk_words;
  legacy_words[closure_type_word] =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID);
  auto negative_scale_words = random_walk_words;
  negative_scale_words[scale_word] = std::bit_cast<std::uint32_t>(-0.031f);

  auto random = device.create_buffer<std::uint32_t>(random_walk_words.size());
  auto burley = device.create_buffer<std::uint32_t>(burley_words.size());
  auto skin = device.create_buffer<std::uint32_t>(skin_words.size());
  auto legacy = device.create_buffer<std::uint32_t>(legacy_words.size());
  auto negative =
      device.create_buffer<std::uint32_t>(negative_scale_words.size());
  auto mixed =
      device.create_buffer<std::uint32_t>(mixed_random_walk_words.size());
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(scenario_count * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto full_shader = device.compile(transition_kernel<8u>(), options);
  auto empty_shader = device.compile(transition_kernel<0u>(), options);
  std::array<luisa::float4, scenario_count * output_stride> actual{};
  std::array<std::uint32_t, scenario_count * meta_stride> actual_meta{};

  stream << random.copy_from(random_walk_words.data())
         << burley.copy_from(burley_words.data())
         << skin.copy_from(skin_words.data())
         << legacy.copy_from(legacy_words.data())
         << negative.copy_from(negative_scale_words.data())
         << mixed.copy_from(mixed_random_walk_words.data())
         << full_shader(random, output, meta, 0u, 0u, 0u).dispatch(1u)
         << full_shader(burley, output, meta, 1u, 0u, 0u).dispatch(1u)
         << full_shader(skin, output, meta, 2u, 0u, 0u).dispatch(1u)
         << full_shader(legacy, output, meta, 3u, 0u, 0u).dispatch(1u)
         << full_shader(burley, output, meta, 4u,
                        device_svm::path_ray_diffuse_ancestor, 0u)
                .dispatch(1u)
         << full_shader(negative, output, meta, 5u, 0u, 0u).dispatch(1u)
         << full_shader(random, output, meta, 6u,
                        device_svm::path_ray_diffuse_ancestor, 1u)
                .dispatch(1u)
         << empty_shader(random, output, meta, 7u, 0u, 0u).dispatch(1u)
         << full_shader(mixed, output, meta, 8u, 0u, 0u).dispatch(1u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  auto common = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + index];
  };
  auto normal = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + 2u + index].xyz();
  };
  auto radius = [&](std::uint32_t scenario) {
    return actual[scenario * output_stride + 4u];
  };
  auto albedo = [&](std::uint32_t scenario) {
    return actual[scenario * output_stride + 5u];
  };
  auto alpha = [&](std::uint32_t scenario) {
    return actual[scenario * output_stride + 6u].x;
  };
  auto type = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual_meta[scenario * meta_stride + index];
  };
  auto state = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual_meta[scenario * meta_stride + 2u + field];
  };

  constexpr auto random_type =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_ID);
  constexpr auto burley_type =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_BURLEY_ID);
  constexpr auto skin_type =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID);
  constexpr auto legacy_type =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID);
  constexpr auto diffuse_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);
  constexpr auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  constexpr auto diffuse_flags = device_svm::shader_data_bsdf |
                                 device_svm::shader_data_bsdf_has_eval;
  constexpr auto inverse_pi = 0.31830988618379067154f;
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  const auto tilted = luisa::float3{0.0f, 1.0f, 0.0f};
  const auto random_color = luisa::float3{0.37f, 0.62f, 0.14f};
  const auto random_radius = luisa::float3{1.10f, 0.45f, 0.09f} * 0.031f;
  const auto burley_color = luisa::float3{0.21f, 0.74f, 0.48f};
  const auto burley_radius =
      luisa::float3{0.80f, 0.32f, 0.05f} * 0.023f;

  auto valid = true;
  /* Common closure state is observed directly from unmodified Cycles. */
  valid &= type(0u, 0u) == random_type &&
           near(common(0u, 0u).xyz(), random_color) &&
           near(common(0u, 0u).w, 1.1299999952f) && near(normal(0u, 0u), up) &&
           near(radius(0u).xyz(), random_radius) && near(radius(0u).w, 0.99f) &&
           near(albedo(0u).xyz(), random_color) && near(albedo(0u).w, 1.55f) &&
           near(alpha(0u), 0.28f) && state(0u, 0u) == 1u &&
           state(0u, 1u) == 7u &&
           state(0u, 2u) == device_svm::shader_data_bssrdf &&
           state(0u, 3u) == ended && state(0u, 4u) == 23u;

  valid &= type(1u, 0u) == burley_type &&
           near(common(1u, 0u).xyz(), burley_color) &&
           near(common(1u, 0u).w, 1.4299999475f) && near(normal(1u, 0u), up) &&
           near(radius(1u).xyz(), burley_radius * (0.25f * inverse_pi)) &&
           near(radius(1u).w, 0.0f) && near(albedo(1u).xyz(), burley_color) &&
           near(albedo(1u).w, 1.4f) && near(alpha(1u), 1.0f) &&
           state(1u, 0u) == 1u && state(1u, 1u) == 7u &&
           state(1u, 2u) == device_svm::shader_data_bssrdf &&
           state(1u, 3u) == ended && state(1u, 4u) == 23u;

  /* The same typed payload is partitioned only by Cycles' method state. */
  valid &= type(2u, 0u) == skin_type && near(common(2u, 0u).w, 1.13f) &&
           near(radius(2u).xyz(),
                {0.0125522302f, 0.0019617188f, 0.0026057979f}) &&
           near(radius(2u).w, 0.9f) && near(alpha(2u), 1.0f) &&
           state(2u, 2u) == device_svm::shader_data_bssrdf;
  valid &= type(3u, 0u) == legacy_type && near(common(3u, 0u).w, 1.13f) &&
           near(radius(3u).xyz(), random_radius * (0.25f * inverse_pi)) &&
           near(radius(3u).w, 0.9f) && near(alpha(3u), 0.28f) &&
           state(3u, 2u) == device_svm::shader_data_bssrdf;

  /* Burley's diffuse-ancestor fallback and a non-positive radius are two
   * different source transitions, including the retained NONE-slot weight. */
  valid &= type(4u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           type(4u, 1u) == diffuse_type &&
           near(common(4u, 0u).xyz(), burley_color) &&
           near(common(4u, 0u).w, 0.0f) &&
           near(common(4u, 1u).xyz(), burley_color) &&
           near(radius(4u).xyz(), burley_radius) && state(4u, 0u) == 2u &&
           state(4u, 1u) == 6u && state(4u, 2u) == diffuse_flags;
  valid &= type(5u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           type(5u, 1u) == diffuse_type &&
           near(common(5u, 0u).xyz(), {0.0f, 0.0f, 0.0f}) &&
           near(common(5u, 1u).xyz(), random_color) &&
           near(radius(5u).xyz(), {0.0f, 0.0f, 0.0f}) &&
           state(5u, 0u) == 2u && state(5u, 1u) == 6u &&
           state(5u, 2u) == diffuse_flags;

  /* Non-Burley methods ignore DIFFUSE_ANCESTOR, and the SVM normal payload is
   * normalized before the valid-reflection correction. */
  valid &= type(6u, 0u) == random_type && near(normal(6u, 0u), tilted) &&
           near(radius(6u).xyz(), random_radius) &&
           state(6u, 2u) == device_svm::shader_data_bssrdf;

  /* Allocation failure never changes cursor progression or invents flags. */
  valid &= state(7u, 0u) == 0u && state(7u, 1u) == 0u &&
           state(7u, 2u) == 0u && state(7u, 3u) == ended &&
           state(7u, 4u) == 23u;

  /* Cycles mixes the common closure weight but retains the unmixed shader
   * color as BSSRDF albedo. This stream prepends one legal NODE_VALUE_F. */
  valid &= type(8u, 0u) == random_type &&
           near(common(8u, 0u).xyz(), random_color * 0.25f) &&
           near(common(8u, 0u).w, 1.13f * 0.25f) &&
           near(radius(8u).xyz(), random_radius) &&
           near(albedo(8u).xyz(), random_color) &&
           state(8u, 0u) == 1u && state(8u, 1u) == 7u &&
           state(8u, 2u) == device_svm::shader_data_bssrdf &&
           state(8u, 3u) == ended && state(8u, 4u) == 26u;

  if (!valid || std::getenv("PSYCLES_DUMP_STANDALONE_BSSRDF_REGRESSION")) {
    for (auto scenario = 0u; scenario < scenario_count; scenario++) {
      const auto c0 = common(scenario, 0u);
      const auto c1 = common(scenario, 1u);
      const auto r = radius(scenario);
      const auto a = albedo(scenario);
      std::cerr << "scenario " << scenario << " types " << type(scenario, 0u)
                << ' ' << type(scenario, 1u) << " count "
                << state(scenario, 0u) << " left " << state(scenario, 1u)
                << " flags 0x" << std::hex << state(scenario, 2u) << std::dec
                << " status " << state(scenario, 3u) << " pc "
                << state(scenario, 4u) << "\n  c0 " << c0.x << ' ' << c0.y
                << ' ' << c0.z << ' ' << c0.w << " c1 " << c1.x << ' '
                << c1.y << ' ' << c1.z << ' ' << c1.w << "\n  radius/a "
                << r.x << ' ' << r.y << ' ' << r.z << ' ' << r.w
                << " albedo/ior " << a.x << ' ' << a.y << ' ' << a.z << ' '
                << a.w << " alpha " << alpha(scenario) << '\n';
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <backend>\n";
    return 2;
  }
  return run(argv[1], argv) ? 0 : 1;
}

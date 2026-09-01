#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_bssrdf.h"
#include "luisa_cycles_svm_test_kernel_globals.h"
#include "path_tracer_bsdf_tables.h"

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

/* Compact final global streams from the unmodified Cycles 5.2.1 oracle
 * probes. Only the four-word shader jump is relocated. */
constexpr std::array<std::uint32_t, 55u> random_walk_skin_words{
    0x00000001u, 0x00000004u, 0x00000037u, 0x00000038u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000002u, 0x0000002bu, 0x000000ffu,
    0x0000001au, 0x3f800000u, 0x3ec28f5cu, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x3f266666u, 0x3edc28f6u, 0x3e570a3du,
    0x3f428f5cu, 0x3f51eb85u, 0x00000000u, 0x0000ff00u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x3e19999au, 0x3d4ccccdu, 0x3ecccccdu, 0x3f99999au, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x3f800000u, 0x3f800000u,
    0x3f800000u, 0x3cf5c28fu, 0x3fc00000u, 0x00000022u, 0x3f99999au,
    0x3eb33333u, 0x3da3d70au, 0x3ccccccdu, 0x3faf5c29u, 0x3e8a3d71u,
    0x00000000u, 0x3faa3d71u, 0x00000000u, 0x000000ffu, 0x00000000u};

constexpr std::array<std::uint32_t, 55u> burley_words{
    0x00000001u, 0x00000004u, 0x00000037u, 0x00000038u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000002u, 0x0000002bu, 0x000000ffu,
    0x0000001au, 0x3f800000u, 0x3f028f5cu, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x3f147ae1u, 0x3e75c28fu, 0x3f2e147bu,
    0x3ec7ae14u, 0x3f428f5cu, 0x00000000u, 0x0000ff00u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x3ea3d70au, 0x3da3d70au, 0x3e428f5cu, 0x3f666666u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x3f800000u, 0x3f800000u,
    0x3f800000u, 0x3cf5c28fu, 0x3fc00000u, 0x0000001fu, 0x3f666666u,
    0x3e99999au, 0x3d23d70au, 0x3c9374bcu, 0x3fb33333u, 0xbed70a3du,
    0x00000000u, 0x3faa3d71u, 0x00000000u, 0x000000ffu, 0x00000000u};

inline constexpr std::uint32_t diffuse_roughness_word = 22u;
inline constexpr std::uint32_t subsurface_radius_y_word = 45u;
inline constexpr std::uint32_t thin_wall_word = 52u;

class SubsurfaceKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;

public:
  explicit SubsurfaceKernelGlobals(const BufferFloat &table) noexcept
      : _table{&table} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return _table->read(index);
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
          device_svm::shader_data_use_bump_map_correction,
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
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

inline constexpr std::uint32_t output_stride = 36u;
inline constexpr std::uint32_t meta_stride = 10u;
inline constexpr std::uint32_t scenario_count = 7u;

template <std::size_t Capacity> [[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>, std::uint32_t, std::uint32_t>{
      [used](BufferUInt words, BufferFloat table, BufferFloat4 output,
             BufferUInt meta, UInt scenario, UInt path_flag) noexcept {
        const SubsurfaceKernelGlobals kernel_globals{table};
        device_svm::ClosurePool closures{Capacity};
        auto shader_data = make_shader_data(&closures);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera |
                device_svm::path_ray_visibility_diffuse,
            path_flag};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                               device_svm::kernel_feature_node_bsdf |
                                   device_svm::kernel_feature_node_emission,
                               used, identity_transform_state(), shader_data,
                               path_state, result);

        const auto output_base = scenario * output_stride;
        const auto meta_base = scenario * meta_stride;
        for (auto i = 0u; i < 5u; i++) {
          output.write(output_base + i, make_float4(0.0f));
          output.write(output_base + 5u + i, make_float4(0.0f));
          output.write(output_base + 10u + i, make_float4(0.0f));
          output.write(output_base + 15u + i, make_float4(0.0f));
          output.write(output_base + 20u + i, make_float4(0.0f));
          output.write(output_base + 25u + i, make_float4(0.0f));
          output.write(output_base + 30u + i, make_float4(0.0f));
          meta.write(meta_base + i,
                     static_cast<std::uint32_t>(CLOSURE_NONE_ID));
          $if(i < closures.count()) {
            const auto closure = closures.common(i);
            output.write(output_base + i,
                         make_float4(closure.weight, closure.sample_weight));
            output.write(output_base + 5u + i, make_float4(closure.N, 0.0f));
            meta.write(meta_base + i, closure.type);
            const Bool is_bssrdf =
                ((closure.type >=
                  static_cast<std::uint32_t>(CLOSURE_BSSRDF_BURLEY_ID)) &
                 (closure.type <= static_cast<std::uint32_t>(
                                      CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID))) |
                ((closure.type == static_cast<std::uint32_t>(CLOSURE_NONE_ID)) &
                 (i == 1u));
            $if(is_bssrdf) {
              const auto bssrdf = closures.bssrdf(i);
              output.write(
                  output_base + 10u + i,
                  make_float4(bssrdf.param.radius, bssrdf.param.anisotropy));
              output.write(output_base + 15u + i,
                           make_float4(bssrdf.param.albedo, bssrdf.param.ior));
              output.write(output_base + 20u + i,
                           make_float4(bssrdf.param.alpha));
            };
            const Bool is_oren =
                (closure.type ==
                 static_cast<std::uint32_t>(CLOSURE_BSDF_OREN_NAYAR_ID)) |
                (closure.type ==
                 static_cast<std::uint32_t>(CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID));
            $if(is_oren) {
              const auto oren = closures.oren_nayar(i);
              output.write(output_base + 25u + i,
                           make_float4(oren.param.roughness, oren.param.a,
                                       oren.param.b, 0.0f));
              output.write(output_base + 30u + i,
                           make_float4(oren.param.multiscatter_term, 0.0f));
            };
          };
        }
        output.write(output_base + 35u,
                     make_float4(shader_data.closure_emission_background,
                                 shader_data.closure_transparent_extinction.x));
        meta.write(meta_base + 5u, closures.count());
        meta.write(meta_base + 6u, closures.left());
        meta.write(meta_base + 7u, shader_data.flag);
        meta.write(meta_base + 8u, result.status);
        meta.write(meta_base + 9u, result.final_offset);
      }};
}

[[nodiscard]] auto nonfinite_allocation_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 output, BufferUInt meta) noexcept {
        constexpr auto scenario = 6u;
        const auto output_base = scenario * output_stride;
        const auto meta_base = scenario * meta_stride;
        device_svm::ClosurePool closures{1u};
        auto shader_data = make_shader_data(&closures);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        const Float nan = as<float>(UInt{0x7fc00000u});
        device_svm::detail::bssrdf_setup(
            shader_data, path_state,
            static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_ID),
            make_float3(nan), make_float3(0.01f), make_float3(0.5f),
            make_float3(0.0f, 0.0f, 1.0f), 0.25f, 1.5f, 0.0f);
        for (auto i = 0u; i < output_stride; i++) {
          output.write(output_base + i, make_float4(0.0f));
        }
        for (auto i = 0u; i < 5u; i++) {
          meta.write(meta_base + i,
                     static_cast<std::uint32_t>(CLOSURE_NONE_ID));
        }
        $if(closures.count() != 0u) {
          const auto closure = closures.common(0u);
          output.write(output_base,
                       make_float4(closure.weight, closure.sample_weight));
          output.write(output_base + 5u, make_float4(closure.N, 0.0f));
          meta.write(meta_base, closure.type);
        };
        meta.write(meta_base + 5u, closures.count());
        meta.write(meta_base + 6u, closures.left());
        meta.write(meta_base + 7u, shader_data.flag);
        meta.write(meta_base + 8u, 0u);
        meta.write(meta_base + 9u, 0u);
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

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 5.0e-5f) noexcept {
  return near(actual.xyz(), expected.xyz(), tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table_values =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values(
          psycles::contract::ShaderColorSpace{});
  auto small_radius_words = random_walk_skin_words;
  small_radius_words[subsurface_radius_y_word] = 0u;
  auto thin_rough_words = random_walk_skin_words;
  thin_rough_words[diffuse_roughness_word] = std::bit_cast<std::uint32_t>(0.4f);
  thin_rough_words[thin_wall_word] = 1u;

  auto table = device.create_buffer<float>(table_values.size());
  auto skin =
      device.create_buffer<std::uint32_t>(random_walk_skin_words.size());
  auto burley = device.create_buffer<std::uint32_t>(burley_words.size());
  auto small = device.create_buffer<std::uint32_t>(small_radius_words.size());
  auto thin = device.create_buffer<std::uint32_t>(thin_rough_words.size());
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(scenario_count * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto full_shader = device.compile(transition_kernel<8u>(), options);
  auto exhausted_shader = device.compile(transition_kernel<1u>(), options);
  auto nonfinite_shader =
      device.compile(nonfinite_allocation_kernel(), options);
  std::array<luisa::float4, scenario_count * output_stride> actual{};
  std::array<std::uint32_t, scenario_count * meta_stride> actual_meta{};
  stream << table.copy_from(luisa::span{table_values})
         << skin.copy_from(random_walk_skin_words.data())
         << burley.copy_from(burley_words.data())
         << small.copy_from(small_radius_words.data())
         << thin.copy_from(thin_rough_words.data())
         << full_shader(skin, table, output, meta, 0u, 0u).dispatch(1u)
         << full_shader(burley, table, output, meta, 1u, 0u).dispatch(1u)
         << full_shader(small, table, output, meta, 2u, 0u).dispatch(1u)
         << full_shader(burley, table, output, meta, 3u,
                        device_svm::path_ray_diffuse_ancestor)
                .dispatch(1u)
         << full_shader(thin, table, output, meta, 4u, 0u).dispatch(1u)
         << exhausted_shader(skin, table, output, meta, 5u, 0u).dispatch(1u)
         << nonfinite_shader(output, meta).dispatch(1u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  constexpr auto transparent_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID);
  constexpr auto diffuse_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);
  constexpr auto oren_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_OREN_NAYAR_ID);
  constexpr auto rough_translucent_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID);
  constexpr auto skin_type =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID);
  constexpr auto burley_type =
      static_cast<std::uint32_t>(CLOSURE_BSSRDF_BURLEY_ID);
  constexpr auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  constexpr auto base_flags = device_svm::shader_data_use_bump_map_correction |
                              device_svm::shader_data_emission |
                              device_svm::shader_data_bsdf |
                              device_svm::shader_data_bsdf_has_eval |
                              device_svm::shader_data_transparent;
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  const auto down = luisa::float3{0.0f, 0.0f, -1.0f};
  auto common = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + index];
  };
  auto normal = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + 5u + index].xyz();
  };
  auto radius = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + 10u + index];
  };
  auto albedo = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + 15u + index];
  };
  auto alpha = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + 20u + index].x;
  };
  auto oren = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + 25u + index];
  };
  auto oren_ms = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual[scenario * output_stride + 30u + index].xyz();
  };
  auto type = [&](std::uint32_t scenario, std::uint32_t index) {
    return actual_meta[scenario * meta_stride + index];
  };
  auto state = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual_meta[scenario * meta_stride + 5u + field];
  };

  auto valid = true;
  /* External Cycles common closure oracle plus source-derived typed state. */
  valid &= type(0u, 0u) == transparent_type && type(0u, 1u) == skin_type &&
           type(0u, 2u) == diffuse_type &&
           near(common(0u, 0u).xyz(), {0.18f, 0.18f, 0.18f}) &&
           near(common(0u, 0u).w, 0.1800000072f) &&
           near(common(0u, 1u).xyz(),
                {0.2291900069f, 0.1119299904f, 0.4050799906f}) &&
           near(common(0u, 1u).w, 0.7461999655f) &&
           near(common(0u, 2u).xyz(),
                {0.1234100088f, 0.06027000025f, 0.2181199938f}) &&
           near(common(0u, 2u).w, 0.1339333355f) &&
           near(radius(0u, 1u).xyz(),
                {0.0101080587f, 0.0063496992f, 0.0001951562f}) &&
           near(radius(0u, 1u).w, 0.27f) &&
           near(albedo(0u, 1u).xyz(), {0.43f, 0.21f, 0.76f}) &&
           near(albedo(0u, 1u).w, 1.37f) && near(alpha(0u, 1u), 1.0f) &&
           near(normal(0u, 1u), up) && state(0u, 0u) == 3u &&
           state(0u, 1u) == 5u &&
           state(0u, 2u) == (base_flags | device_svm::shader_data_bssrdf) &&
           state(0u, 3u) == ended && state(0u, 4u) == 55u;

  valid &= type(1u, 0u) == transparent_type && type(1u, 1u) == burley_type &&
           type(1u, 2u) == diffuse_type &&
           near(common(1u, 0u).xyz(), {0.24f, 0.24f, 0.24f}) &&
           near(common(1u, 1u).xyz(),
                {0.1057919860f, 0.2997440100f, 0.1719119847f}) &&
           near(common(1u, 1u).w, 0.5774480104f) &&
           near(common(1u, 2u).xyz(),
                {0.0766080022f, 0.2170560062f, 0.1244879961f}) &&
           near(radius(1u, 1u).xyz(),
                {0.0012891550f, 0.0004297183f, 0.0000572958f}) &&
           near(radius(1u, 1u).w, -0.42f) &&
           near(albedo(1u, 1u).xyz(), {0.24f, 0.68f, 0.39f}) &&
           near(albedo(1u, 1u).w, 1.01f) && near(alpha(1u, 1u), 0.2601f) &&
           state(1u, 0u) == 3u && state(1u, 1u) == 5u &&
           state(1u, 2u) == (base_flags | device_svm::shader_data_bssrdf);

  /* One small channel is transferred to a newly appended Diffuse closure. */
  valid &= type(2u, 1u) == skin_type && type(2u, 2u) == diffuse_type &&
           type(2u, 3u) == diffuse_type &&
           near(common(2u, 1u).xyz(), {0.2291900069f, 0.0f, 0.4050799906f}) &&
           near(common(2u, 1u).w, 0.4228466650f) &&
           near(common(2u, 2u).xyz(), {0.0f, 0.1119299904f, 0.0f}) &&
           near(common(2u, 2u).w, 0.0373099968f) &&
           near(radius(2u, 1u).xyz(), {0.0101080587f, 0.0f, 0.0001951562f}) &&
           state(2u, 0u) == 4u && state(2u, 1u) == 4u &&
           state(2u, 2u) == (base_flags | device_svm::shader_data_bssrdf);

  /* Burley after a diffuse ancestor keeps the allocated NONE slot and moves
   * its entire weight to Diffuse before the ordinary residual Diffuse. */
  valid &= type(3u, 1u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           type(3u, 2u) == diffuse_type && type(3u, 3u) == diffuse_type &&
           near(common(3u, 1u).xyz(), common(1u, 1u).xyz()) &&
           near(common(3u, 1u).w, 0.0f) &&
           near(common(3u, 2u).xyz(), common(1u, 1u).xyz()) &&
           near(radius(3u, 1u).xyz(), {0.0162f, 0.0054f, 0.00072f}) &&
           near(albedo(3u, 1u).xyz(), {0.24f, 0.68f, 0.39f}) &&
           state(3u, 0u) == 4u && state(3u, 1u) == 4u &&
           state(3u, 2u) == base_flags;

  /* Thin Wall is a disjoint two-BSDF transition: no BSSRDF tag/flag survives.
   */
  valid &=
      type(4u, 1u) == oren_type && type(4u, 2u) == rough_translucent_type &&
      type(4u, 3u) == oren_type &&
      near(common(4u, 1u).xyz(),
           {0.0836543525f, 0.0408544465f, 0.1478541966f}) &&
      near(common(4u, 2u).xyz(),
           {0.1455356544f, 0.0710755439f, 0.2572257940f}) &&
      near(normal(4u, 1u), up) && near(normal(4u, 2u), down) &&
      near(oren(4u, 1u).xyz(), {0.4f, 0.2854496724f, 0.1141798690f}) &&
      near(oren(4u, 2u).xyz(), oren(4u, 1u).xyz()) &&
      near(oren_ms(4u, 1u), {0.0750885648f, 0.0175998613f, 0.2409164660f}) &&
      near(oren_ms(4u, 2u), oren_ms(4u, 1u)) && state(4u, 0u) == 4u &&
      state(4u, 1u) == 4u &&
      state(4u, 2u) ==
          (base_flags | device_svm::shader_data_bsdf_has_transmission);

  /* With one slot, alpha consumes the pool. BSSRDF and residual Diffuse both
   * fail allocation without inventing state or rolling back transparency. */
  constexpr auto exhausted_flags =
      device_svm::shader_data_use_bump_map_correction |
      device_svm::shader_data_emission | device_svm::shader_data_bsdf |
      device_svm::shader_data_transparent;
  valid &= type(5u, 0u) == transparent_type && state(5u, 0u) == 1u &&
           state(5u, 1u) == 0u && state(5u, 2u) == exhausted_flags &&
           state(5u, 3u) == ended && state(5u, 4u) == 55u;

  /* bssrdf_alloc rejects only `sample_weight < cutoff`; Cycles therefore
   * allocates a non-finite weight. This locks the predicate shape itself. */
  valid &= type(6u, 0u) ==
               static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_ID) &&
           state(6u, 0u) == 1u && state(6u, 1u) == 0u &&
           state(6u, 2u) == (device_svm::shader_data_use_bump_map_correction |
                             device_svm::shader_data_bssrdf);

  valid &=
      near(actual[35u].xyz(), {0.1476000100f, 0.0492000021f, 0.3936000168f}) &&
      near(actual[output_stride + 35u].xyz(),
           {0.2188799828f, 0.0547199957f, 0.1299599856f});

  if (!valid || std::getenv("PSYCLES_DUMP_PRINCIPLED_SUBSURFACE_REGRESSION")) {
    for (auto scenario = 0u; scenario < scenario_count; scenario++) {
      std::cerr << "scenario " << scenario << " count " << state(scenario, 0u)
                << " left " << state(scenario, 1u) << " flags 0x" << std::hex
                << state(scenario, 2u) << std::dec << " status "
                << state(scenario, 3u) << " pc " << state(scenario, 4u) << '\n';
      for (auto i = 0u; i < state(scenario, 0u); i++) {
        const auto c = common(scenario, i);
        const auto r = radius(scenario, i);
        const auto a = albedo(scenario, i);
        std::cerr << "  closure " << i << " type " << type(scenario, i) << " w "
                  << c.x << ' ' << c.y << ' ' << c.z << " sw " << c.w << " N "
                  << normal(scenario, i).x << ' ' << normal(scenario, i).y
                  << ' ' << normal(scenario, i).z << " radius/a " << r.x << ' '
                  << r.y << ' ' << r.z << ' ' << r.w << " albedo/ior " << a.x
                  << ' ' << a.y << ' ' << a.z << ' ' << a.w << " alpha "
                  << alpha(scenario, i) << '\n';
      }
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

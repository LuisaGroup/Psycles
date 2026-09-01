#include <psycles/luisa/cycles_svm.h>

#include "luisa_cycles_svm_test_kernel_globals.h"
#include "path_tracer_bsdf_tables.h"

#include <algorithm>
#include <array>
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

/* Compact global image from the unmodified Cycles 5.2.1
 * `principled_thin_wall_svm_oracle`. Only the four-word shader jump is
 * relocated. The Thin Wall literal at word 52 is the sole payload difference
 * from the paired thick-transmission oracle. */
constexpr std::array<std::uint32_t, 57u> principled_thin_wall_words{
    0x00000001u, 0x00000004u, 0x00000037u, 0x00000038u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000002u, 0x0000002bu, 0x000000ffu,
    0x0000001au, 0x3fbd70a4u, 0x3e9eb852u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x3f3851ecu, 0x00000000u, 0x3eb851ecu, 0x3f23d70au,
    0x3f68f5c3u, 0x3f4f5c29u, 0x00000000u, 0x0000ff00u, 0x3f428f5cu,
    0x3ed1eb85u, 0x3f6e147bu, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x3e2e147bu, 0x3db851ecu, 0x3f028f5cu, 0x3fb33333u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x3f800000u, 0x3f800000u,
    0x3f800000u, 0x3cf5c28fu, 0x3fc00000u, 0x00000020u, 0x3f800000u,
    0x3e4ccccdu, 0x3dcccccdu, 0x3ba3d70au, 0x3fb33333u, 0x00000000u,
    0x43a50000u, 0x3fc7ae14u, 0x00000001u, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u};

inline constexpr std::uint32_t roughness_word = 12u;

class ThinWallKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;
  Bool _reflective_caustics;
  Bool _refractive_caustics;

public:
  ThinWallKernelGlobals(const BufferFloat &table,
                        Expr<bool> reflective_caustics,
                        Expr<bool> refractive_caustics) noexcept
      : _table{&table}, _reflective_caustics{reflective_caustics},
        _refractive_caustics{refractive_caustics} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return _table->read(index);
  }

  [[nodiscard]] Bool caustics_reflective() const noexcept override {
    return _reflective_caustics;
  }

  [[nodiscard]] Bool caustics_refractive() const noexcept override {
    return _refractive_caustics;
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

inline constexpr std::uint32_t output_stride = 14u;
inline constexpr std::uint32_t meta_stride = 13u;

[[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>, std::uint32_t, std::uint32_t>{
      [used](BufferUInt words, BufferFloat table, BufferFloat4 output,
             BufferUInt meta, UInt visibility, UInt case_offset) noexcept {
        const auto case_index = dispatch_x() + case_offset;
        const Bool reflective_caustics =
            (case_index == 0u) | (case_index == 1u) | (case_index == 4u);
        const Bool refractive_caustics =
            (case_index == 0u) | (case_index == 2u) | (case_index == 4u);
        const ThinWallKernelGlobals kernel_globals{
            table, reflective_caustics, refractive_caustics};
        device_svm::ClosurePool closures{8u};
        auto shader_data = make_shader_data(&closures);
        const device_svm::PathState path_state{visibility, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
            device_svm::kernel_feature_node_bsdf |
                device_svm::kernel_feature_node_emission,
            used, identity_transform_state(), shader_data, path_state, result);

        const auto output_base = case_index * output_stride;
        const auto meta_base = case_index * meta_stride;
        for (auto i = 0u; i < 5u; i++) {
          output.write(output_base + i, make_float4(0.0f));
          output.write(output_base + 5u + i, make_float4(0.0f));
          meta.write(meta_base + 5u + i,
                     static_cast<std::uint32_t>(CLOSURE_NONE_ID));
          $if (i < closures.count()) {
            const auto closure = closures.common(i);
            output.write(output_base + i,
                         make_float4(closure.weight, closure.sample_weight));
            output.write(output_base + 5u + i,
                         make_float4(closure.N, 0.0f));
            meta.write(meta_base + 5u + i, closure.type);
          };
        }
        for (auto i = 1u; i < 4u; i++) {
          output.write(output_base + 9u + i, make_float4(0.0f));
          meta.write(meta_base + 9u + i,
                     static_cast<std::uint32_t>(
                         device_svm::MicrofacetFresnel::none));
          $if (i < closures.count()) {
            const auto closure = closures.common(i);
            const auto microfacet =
                (closure.type == static_cast<std::uint32_t>(
                                     CLOSURE_BSDF_MICROFACET_GGX_ID)) |
                (closure.type == static_cast<std::uint32_t>(
                                     CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID));
            $if (microfacet) {
              const auto param = closures.microfacet_param(i);
              output.write(output_base + 9u + i,
                           make_float4(param.alpha_x, param.alpha_y, param.ior,
                                       param.energy_scale));
              meta.write(meta_base + 9u + i, param.fresnel_type);
            };
          };
        }
        output.write(output_base + 13u,
                     make_float4(shader_data.closure_emission_background,
                                 shader_data.closure_transparent_extinction.x));
        meta.write(meta_base + 0u, closures.count());
        meta.write(meta_base + 1u, closures.left());
        meta.write(meta_base + 2u, shader_data.flag);
        meta.write(meta_base + 3u, result.status);
        meta.write(meta_base + 4u, result.final_offset);
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

[[nodiscard]] bool positive_finite(luisa::float4 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z) && std::isfinite(value.w) && value.x > 0.0f &&
         value.y > 0.0f && value.z > 0.0f && value.w > 0.0f;
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table_values =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values(
          psycles::contract::ShaderColorSpace{});
  auto singular_words = principled_thin_wall_words;
  singular_words[roughness_word] = 0u;
  auto table = device.create_buffer<float>(table_values.size());
  auto words =
      device.create_buffer<std::uint32_t>(principled_thin_wall_words.size());
  auto singular = device.create_buffer<std::uint32_t>(singular_words.size());
  auto output = device.create_buffer<luisa::float4>(5u * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(5u * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto shader = device.compile(transition_kernel(), options);

  std::array<luisa::float4, 5u * output_stride> actual{};
  std::array<std::uint32_t, 5u * meta_stride> actual_meta{};
  constexpr auto camera_visibility = device_svm::path_ray_visibility_camera |
                                     device_svm::path_ray_visibility_diffuse;
  constexpr auto diffuse_visibility = device_svm::path_ray_visibility_diffuse;
  stream << table.copy_from(luisa::span{table_values})
         << words.copy_from(principled_thin_wall_words.data())
         << singular.copy_from(singular_words.data())
         << shader(words, table, output, meta, camera_visibility, 0u).dispatch(4u)
         << shader(singular, table, output, meta, diffuse_visibility, 4u)
                .dispatch(1u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  constexpr auto base_flags = device_svm::shader_data_use_bump_map_correction |
                              device_svm::shader_data_emission |
                              device_svm::shader_data_bsdf |
                              device_svm::shader_data_bsdf_has_eval |
                              device_svm::shader_data_transparent;
  constexpr auto transmission_flags =
      base_flags | device_svm::shader_data_bsdf_has_transmission;
  constexpr auto transparent_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID);
  constexpr auto ggx_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID);
  constexpr auto thin_type = static_cast<std::uint32_t>(
      CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID);
  constexpr auto diffuse_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);
  constexpr auto no_fresnel =
      static_cast<std::uint32_t>(device_svm::MicrofacetFresnel::none);
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  const auto down = luisa::float3{0.0f, 0.0f, -1.0f};

  auto valid = true;
  for (auto case_index = 0u; case_index < 4u; case_index++) {
    const auto base = case_index * output_stride;
    const auto meta_base = case_index * meta_stride;
    const auto reflective = case_index < 2u;
    const auto refractive = case_index == 0u || case_index == 2u;
    const auto expected_count = 5u - case_index;
    const auto expected_left =
        case_index == 0u ? 2u : (case_index == 1u ? 3u :
                                 (case_index == 2u ? 5u : 6u));
    const auto common_valid =
        near(actual[base + 0u].xyz(), luisa::float3{0.1899999976f}) &&
        near(actual[base + 0u].w, 0.1899999976f) &&
        near(actual[base + 13u].xyz(),
             luisa::float3{0.1927800030f, 0.1020600051f, 0.5783399940f}) &&
        near(actual[base + 13u].w, 0.1899999976f) &&
        actual_meta[meta_base + 0u] == expected_count &&
        actual_meta[meta_base + 1u] == expected_left &&
        actual_meta[meta_base + 2u] ==
            (refractive ? transmission_flags : base_flags) &&
        actual_meta[meta_base + 3u] ==
            static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
        actual_meta[meta_base + 4u] == 55u &&
        actual_meta[meta_base + 5u] == transparent_type;
    if (!common_valid) {
      std::cerr << "common case failed: " << case_index << '\n';
    }
    valid &= common_valid;

    if (reflective) {
      const auto reflection_valid = positive_finite(actual[base + 1u]) &&
               near(actual[base + 6u].xyz(), up) &&
               actual_meta[meta_base + 6u] == ggx_type &&
               actual_meta[meta_base + 10u] == no_fresnel &&
               near(actual[base + 10u].x, 0.0961f) &&
               near(actual[base + 10u].y, 0.0961f) &&
               near(actual[base + 10u].z, 1.0f) &&
               std::isfinite(actual[base + 10u].w) &&
               actual[base + 10u].w >= 1.0f;
      if (!reflection_valid) {
        std::cerr << "reflection case failed: " << case_index << '\n';
      }
      valid &= reflection_valid;
    }

    const auto transmission_index = reflective ? 2u : 1u;
    if (refractive) {
      const auto transmission_valid =
          positive_finite(actual[base + transmission_index]) &&
               near(actual[base + 5u + transmission_index].xyz(), down) &&
               actual_meta[meta_base + 5u + transmission_index] == thin_type &&
               actual_meta[meta_base + 9u + transmission_index] == no_fresnel &&
               near(actual[base + 9u + transmission_index].x,
                    0.0668216149f) &&
               near(actual[base + 9u + transmission_index].y,
                    0.0668216149f) &&
               near(actual[base + 9u + transmission_index].z, 1.0f) &&
               std::isfinite(actual[base + 9u + transmission_index].w) &&
          actual[base + 9u + transmission_index].w >= 1.0f;
      if (!transmission_valid) {
        std::cerr << "transmission case failed: " << case_index << '\n';
      }
      valid &= transmission_valid;
    }
  }

  /* The externally observed event-0 common records. Typed microfacet state
   * above is source-derived because Cycles' diagnostic trace intentionally
   * observes ShaderClosure common fields only. */
  const auto external_valid =
      near(actual[1u].xyz(),
           luisa::float3{0.0274434481f, 0.0141683603f, 0.0517196022f}) &&
      near(actual[1u].w, 0.0310758036f) &&
      near(actual[2u].xyz(),
           luisa::float3{0.1949812770f, 0.3594840169f, 0.4788061380f}) &&
      near(actual[2u].w, 0.3444238305f) &&
      near(actual[3u].xyz(),
           luisa::float3{0.2245076895f, 0.2244775295f, 0.2245223373f}) &&
      near(actual[3u].w, 0.0080270125f) &&
      near(actual[4u].xyz(),
           luisa::float3{0.0778046995f, 0.1383194625f, 0.1966729909f}) &&
      near(actual[4u].w, 0.1375990510f) &&
      actual_meta[7u] == thin_type && actual_meta[8u] == ggx_type &&
      actual_meta[9u] == diffuse_type;
  if (!external_valid) {
    std::cerr << "external common state failed\n";
  }
  valid &= external_valid;

  /* A source-derived zero-roughness variant proves the exact Cycles
   * non-camera transition: type 22 is not allocated, and its full weight is
   * merged into the pre-existing transparent record without a second slot. */
  constexpr auto singular_case = 4u;
  const auto singular_base = singular_case * output_stride;
  const auto singular_meta = singular_case * meta_stride;
  const auto singular_valid =
      near(actual[singular_base + 0u].xyz(),
           luisa::float3{0.3849812746f, 0.5494840145f, 0.6688061357f}) &&
      near(actual[singular_base + 0u].w, 0.5344238281f) &&
      near(actual[singular_base + 13u].w, 0.3849812746f) &&
      actual_meta[singular_meta + 0u] == 4u &&
      actual_meta[singular_meta + 1u] == 3u &&
      actual_meta[singular_meta + 2u] == base_flags &&
      actual_meta[singular_meta + 3u] ==
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
      actual_meta[singular_meta + 4u] == 55u &&
      actual_meta[singular_meta + 5u] == transparent_type &&
      actual_meta[singular_meta + 6u] == ggx_type &&
      actual_meta[singular_meta + 7u] == ggx_type &&
      actual_meta[singular_meta + 8u] == diffuse_type;
  if (!singular_valid) {
    std::cerr << "singular transition failed\n";
  }
  valid &= singular_valid;

  if (!valid) {
    std::cerr << "Cycles thin-wall Principled Transmission mismatch on "
              << backend << '\n';
    for (auto case_index = 0u; case_index < 5u; case_index++) {
      const auto base = case_index * output_stride;
      const auto meta_base = case_index * meta_stride;
      std::cerr << "case " << case_index << ": c0=(" << actual[base].x
                << ", " << actual[base].y << ", " << actual[base].z << ", "
                << actual[base].w << "), c1=(" << actual[base + 1u].x
                << ", " << actual[base + 1u].y << ", "
                << actual[base + 1u].z << ", " << actual[base + 1u].w
                << "), c2=(" << actual[base + 2u].x << ", "
                << actual[base + 2u].y << ", " << actual[base + 2u].z
                << ", " << actual[base + 2u].w << "), c3=("
                << actual[base + 3u].x << ", " << actual[base + 3u].y
                << ", " << actual[base + 3u].z << ", "
                << actual[base + 3u].w << "), c4=("
                << actual[base + 4u].x << ", " << actual[base + 4u].y
                << ", " << actual[base + 4u].z << ", "
                << actual[base + 4u].w << "), emission=("
                << actual[base + 13u].x << ", "
                << actual[base + 13u].y << ", "
                << actual[base + 13u].z << ", "
                << actual[base + 13u].w << "), meta=("
                << actual_meta[meta_base] << ", "
                << actual_meta[meta_base + 1u] << ", "
                << actual_meta[meta_base + 2u] << ", "
                << actual_meta[meta_base + 3u] << ", "
                << actual_meta[meta_base + 4u] << "), types=("
                << actual_meta[meta_base + 5u] << ", "
                << actual_meta[meta_base + 6u] << ", "
                << actual_meta[meta_base + 7u] << ", "
                << actual_meta[meta_base + 8u] << ", "
                << actual_meta[meta_base + 9u] << "), p1=("
                << actual[base + 10u].x << ", "
                << actual[base + 10u].y << ", "
                << actual[base + 10u].z << ", "
                << actual[base + 10u].w << "), p2=("
                << actual[base + 11u].x << ", "
                << actual[base + 11u].y << ", "
                << actual[base + 11u].z << ", "
                << actual[base + 11u].w << "), fresnel=("
                << actual_meta[meta_base + 10u] << ", "
                << actual_meta[meta_base + 11u] << ", "
                << actual_meta[meta_base + 12u] << ")\n";
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

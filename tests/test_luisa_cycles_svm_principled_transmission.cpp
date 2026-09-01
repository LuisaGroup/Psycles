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
 * `principled_transmission_svm_oracle`. The four-word shader jump is
 * relocated; both Geometry records and the typed Principled payload are
 * word-identical to the external dump. */
constexpr std::array<std::uint32_t, 57u> principled_transmission_words{
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
    0x43a50000u, 0x3fc7ae14u, 0x00000000u, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u};

class TransmissionKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;
  Bool _reflective_caustics;
  Bool _refractive_caustics;

public:
  TransmissionKernelGlobals(const BufferFloat &table,
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

inline constexpr std::uint32_t output_stride = 13u;
inline constexpr std::uint32_t meta_stride = 10u;

[[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[used](BufferUInt words,
                                                BufferFloat table,
                                                BufferFloat4 output,
                                                BufferUInt meta) noexcept {
    const auto case_index = dispatch_x();
    const Bool reflective_caustics = (case_index == 0u) | (case_index == 1u);
    const Bool refractive_caustics = (case_index == 0u) | (case_index == 2u);
    const TransmissionKernelGlobals kernel_globals{table, reflective_caustics,
                                                   refractive_caustics};
    device_svm::ClosurePool closures{8u};
    auto shader_data = make_shader_data(&closures);
    const device_svm::PathState path_state{
        device_svm::path_ray_visibility_camera |
            device_svm::path_ray_visibility_diffuse,
        0u};
    device_svm::EvaluationResult result;
    device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                           device_svm::kernel_feature_node_bsdf |
                               device_svm::kernel_feature_node_emission,
                           used, identity_transform_state(), shader_data,
                           path_state, result);

    const auto output_base = case_index * output_stride;
    const auto meta_base = case_index * meta_stride;
    for (auto i = 0u; i < output_stride; i++) {
      output.write(output_base + i, make_float4(0.0f));
    }
    for (auto i = 0u; i < meta_stride; i++) {
      meta.write(meta_base + i, 0u);
    }

    const auto transparent = closures.common(0u);
    output.write(output_base + 0u,
                 make_float4(transparent.weight, transparent.sample_weight));
    output.write(output_base + 12u,
                 make_float4(shader_data.closure_emission_background,
                             shader_data.closure_transparent_extinction.x));
    meta.write(meta_base + 0u, closures.count());
    meta.write(meta_base + 1u, closures.left());
    meta.write(meta_base + 2u, shader_data.flag);
    meta.write(meta_base + 3u, result.status);
    meta.write(meta_base + 4u, result.final_offset);
    meta.write(meta_base + 5u, transparent.type);
    meta.write(meta_base + 6u, static_cast<std::uint32_t>(CLOSURE_NONE_ID));
    meta.write(meta_base + 7u, static_cast<std::uint32_t>(CLOSURE_NONE_ID));
    meta.write(meta_base + 8u, static_cast<std::uint32_t>(CLOSURE_NONE_ID));
    meta.write(meta_base + 9u,
               static_cast<std::uint32_t>(device_svm::MicrofacetFresnel::none));

    $if(reflective_caustics | refractive_caustics) {
      const auto glass = closures.microfacet(1u);
      output.write(output_base + 1u, make_float4(glass.common.weight,
                                                 glass.common.sample_weight));
      output.write(output_base + 4u, make_float4(glass.common.N, 0.0f));
      output.write(output_base + 5u,
                   make_float4(glass.param.alpha_x, glass.param.alpha_y,
                               glass.param.ior, glass.param.energy_scale));
      output.write(output_base + 6u, make_float4(glass.param.T, 0.0f));
      output.write(output_base + 7u,
                   make_float4(glass.generalized_schlick.thin_film.thickness,
                               glass.generalized_schlick.thin_film.ior,
                               glass.generalized_schlick.exponent, 0.0f));
      output.write(
          output_base + 8u,
          make_float4(glass.generalized_schlick.reflection_tint, 0.0f));
      output.write(
          output_base + 9u,
          make_float4(glass.generalized_schlick.transmission_tint, 0.0f));
      output.write(output_base + 10u,
                   make_float4(glass.generalized_schlick.f0, 0.0f));
      output.write(output_base + 11u,
                   make_float4(glass.generalized_schlick.f90, 0.0f));
      meta.write(meta_base + 6u, glass.common.type);
      meta.write(meta_base + 9u, glass.param.fresnel_type);

      $if(reflective_caustics) {
        const auto specular = closures.common(2u);
        const auto diffuse = closures.common(3u);
        output.write(output_base + 2u,
                     make_float4(specular.weight, specular.sample_weight));
        output.write(output_base + 3u,
                     make_float4(diffuse.weight, diffuse.sample_weight));
        meta.write(meta_base + 7u, specular.type);
        meta.write(meta_base + 8u, diffuse.type);
      }
      $else {
        const auto diffuse = closures.common(2u);
        output.write(output_base + 3u,
                     make_float4(diffuse.weight, diffuse.sample_weight));
        meta.write(meta_base + 8u, diffuse.type);
      };
    }
    $else {
      const auto diffuse = closures.common(1u);
      output.write(output_base + 3u,
                   make_float4(diffuse.weight, diffuse.sample_weight));
      meta.write(meta_base + 8u, diffuse.type);
    };
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

[[nodiscard]] bool finite_positive(luisa::float4 value) noexcept {
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
  auto table = device.create_buffer<float>(table_values.size());
  auto words =
      device.create_buffer<std::uint32_t>(principled_transmission_words.size());
  auto output = device.create_buffer<luisa::float4>(4u * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(4u * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto shader = device.compile(transition_kernel(), options);

  std::array<luisa::float4, 4u * output_stride> actual{};
  std::array<std::uint32_t, 4u * meta_stride> actual_meta{};
  stream << table.copy_from(luisa::span{table_values})
         << words.copy_from(principled_transmission_words.data())
         << shader(words, table, output, meta).dispatch(4u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  constexpr auto base_flags = device_svm::shader_data_use_bump_map_correction |
                              device_svm::shader_data_emission |
                              device_svm::shader_data_bsdf |
                              device_svm::shader_data_bsdf_has_eval |
                              device_svm::shader_data_transparent;
  constexpr auto transmission_flags =
      base_flags | device_svm::shader_data_bsdf_has_transmission;
  constexpr auto transparent_weight = 0.1899999976158142f;
  constexpr auto glass_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID);
  constexpr auto specular_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID);
  constexpr auto diffuse_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);
  constexpr auto transparent_type =
      static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID);
  constexpr auto generalized_schlick = static_cast<std::uint32_t>(
      device_svm::MicrofacetFresnel::generalized_schlick);
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  const auto zero = luisa::float3{0.0f};
  const auto one = luisa::float3{1.0f};
  const auto transmission_tint =
      luisa::float3{0.6000000238f, 0.8000000119f, 0.9539391994f};
  const auto f0 = luisa::float3{0.0284703467f, 0.0153590031f, 0.0348387137f};

  auto valid = true;
  for (auto case_index = 0u; case_index < 4u; case_index++) {
    const auto base = case_index * output_stride;
    const auto meta_base = case_index * meta_stride;
    const auto reflective = case_index < 2u;
    const auto refractive = (case_index == 0u) || (case_index == 2u);
    const auto has_glass = reflective || refractive;
    valid &=
        near(actual[base + 0u].xyz(), luisa::float3{transparent_weight}) &&
        near(actual[base + 0u].w, transparent_weight) &&
        near(actual[base + 12u].xyz(),
             luisa::float3{0.1927800030f, 0.1020600051f, 0.5783399940f}) &&
        near(actual[base + 12u].w, transparent_weight) &&
        actual_meta[meta_base + 2u] ==
            (has_glass ? transmission_flags : base_flags) &&
        actual_meta[meta_base + 3u] ==
            static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
        actual_meta[meta_base + 4u] == 55u &&
        actual_meta[meta_base + 5u] == transparent_type &&
        actual_meta[meta_base + 8u] == diffuse_type;

    if (has_glass) {
      valid &=
          finite_positive(actual[base + 1u]) &&
          near(actual[base + 4u].xyz(), up) &&
          near(actual[base + 5u].x, 0.0961f) &&
          near(actual[base + 5u].y, 0.0961f) &&
          near(actual[base + 5u].z, 1.48f) &&
          std::isfinite(actual[base + 5u].w) && actual[base + 5u].w >= 1.0f &&
          near(actual[base + 6u].xyz(), zero) &&
          near(actual[base + 7u].xyz(), luisa::float3{330.0f, 1.56f, -1.48f}) &&
          near(actual[base + 8u].xyz(), reflective ? one : zero) &&
          near(actual[base + 9u].xyz(),
               refractive ? transmission_tint : zero) &&
          near(actual[base + 10u].xyz(), f0) &&
          near(actual[base + 11u].xyz(), one) &&
          actual_meta[meta_base + 6u] == glass_type &&
          actual_meta[meta_base + 9u] == generalized_schlick;
    } else {
      valid &= near(actual[base + 1u].xyz(), zero) &&
               actual_meta[meta_base + 6u] ==
                   static_cast<std::uint32_t>(CLOSURE_NONE_ID);
    }

    if (reflective) {
      valid &=
          near(actual[base + 2u].xyz(),
               luisa::float3{0.2245076895f, 0.2244775295f, 0.2245223373f}) &&
          near(actual[base + 2u].w, 0.0080270125f) &&
          near(actual[base + 3u].xyz(),
               luisa::float3{0.0778046995f, 0.1383194625f, 0.1966729909f}) &&
          near(actual[base + 3u].w, 0.1375990510f) &&
          actual_meta[meta_base + 0u] == 4u &&
          actual_meta[meta_base + 1u] == 2u &&
          actual_meta[meta_base + 7u] == specular_type;
    } else {
      valid &= near(actual[base + 2u].xyz(), zero) &&
               near(actual[base + 3u].xyz(),
                    luisa::float3{0.081648f, 0.1451519877f, 0.2063879967f}) &&
               near(actual[base + 3u].w, 0.1443959922f) &&
               actual_meta[meta_base + 0u] == (has_glass ? 3u : 2u) &&
               actual_meta[meta_base + 1u] == (has_glass ? 4u : 6u) &&
               actual_meta[meta_base + 7u] ==
                   static_cast<std::uint32_t>(CLOSURE_NONE_ID);
    }
  }

  valid &= near(actual[1u].xyz(),
                luisa::float3{0.5829386115f, 0.5830691457f, 0.5831698179f}) &&
           near(actual[1u].w, 0.4619406760f);

  if (!valid) {
    std::cerr << "Cycles thick Principled Transmission mismatch on " << backend
              << '\n';
    for (auto case_index = 0u; case_index < 4u; case_index++) {
      const auto base = case_index * output_stride;
      const auto meta_base = case_index * meta_stride;
      std::cerr << "case " << case_index << ": glass=(" << actual[base + 1u].x
                << ", " << actual[base + 1u].y << ", " << actual[base + 1u].z
                << ", " << actual[base + 1u].w << "), spec=("
                << actual[base + 2u].x << ", " << actual[base + 2u].y << ", "
                << actual[base + 2u].z << ", " << actual[base + 2u].w
                << "), diffuse=(" << actual[base + 3u].x << ", "
                << actual[base + 3u].y << ", " << actual[base + 3u].z << ", "
                << actual[base + 3u].w << "), meta=("
                << actual_meta[meta_base + 0u] << ", "
                << actual_meta[meta_base + 1u] << ", "
                << actual_meta[meta_base + 2u] << ", "
                << actual_meta[meta_base + 3u] << ", "
                << actual_meta[meta_base + 4u] << ")\n";
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

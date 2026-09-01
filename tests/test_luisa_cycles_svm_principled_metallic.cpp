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
 * `principled_metallic_svm_oracle`. The four-word shader jump is relocated;
 * both Geometry records and the typed Principled payload are word-identical
 * to the external dump. */
constexpr std::array<std::uint32_t, 60u> principled_metallic_words{
    0x00000001u, 0x00000004u, 0x0000003au, 0x0000003bu, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x0000000bu, 0x03000002u, 0x00000000u,
    0x00000002u, 0x0000002bu, 0x000000ffu, 0x0000001au, 0x3faf5c29u,
    0x3ef0a3d7u, 0x00000000u, 0x00000000u, 0x3f23d70au, 0x00000000u,
    0x00000000u, 0x3e3851ecu, 0x3f0ccccdu, 0x3f51eb85u, 0x3f47ae14u,
    0x00000000u, 0x00000300u, 0x3f3851ecu, 0x3eb33333u, 0x3f68f5c3u,
    0x3f000000u, 0x3ec28f5cu, 0x3e570a3du, 0x3e6147aeu, 0x3d8f5c29u,
    0x3ef5c28fu, 0x3fa00000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
    0x3f000000u, 0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3cf5c28fu,
    0x3fc00000u, 0x00000020u, 0x3f800000u, 0x3e4ccccdu, 0x3dcccccdu,
    0x3ba3d70au, 0x3fb33333u, 0x00000000u, 0x43b40000u, 0x3fc28f5cu,
    0x00000000u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};

class MetallicKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;
  Bool _reflective_caustics;

public:
  MetallicKernelGlobals(const BufferFloat &table,
                        Expr<bool> reflective_caustics) noexcept
      : _table{&table}, _reflective_caustics{reflective_caustics} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return _table->read(index);
  }

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

inline constexpr std::uint32_t output_stride = 11u;
inline constexpr std::uint32_t meta_stride = 10u;

[[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[used](BufferUInt words,
                                                BufferFloat table,
                                                BufferFloat4 output,
                                                BufferUInt meta) noexcept {
    const auto case_index = dispatch_x();
    const Bool reflective_caustics = case_index == 0u;
    const MetallicKernelGlobals kernel_globals{table, reflective_caustics};
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
    const auto transparent = closures.common(0u);
    output.write(output_base + 0u,
                 make_float4(transparent.weight, transparent.sample_weight));
    for (auto i = 1u; i < 10u; i++) {
      output.write(output_base + i, make_float4(0.0f));
    }
    output.write(output_base + 10u,
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

    $if(reflective_caustics) {
      const auto metallic = closures.microfacet_f82_tint(1u);
      const auto dielectric = closures.common(2u);
      const auto diffuse = closures.common(3u);
      output.write(
          output_base + 1u,
          make_float4(metallic.common.weight, metallic.common.sample_weight));
      output.write(output_base + 2u,
                   make_float4(dielectric.weight, dielectric.sample_weight));
      output.write(output_base + 3u,
                   make_float4(diffuse.weight, diffuse.sample_weight));
      output.write(output_base + 4u, make_float4(metallic.common.N, 0.0f));
      output.write(output_base + 5u,
                   make_float4(metallic.param.alpha_x, metallic.param.alpha_y,
                               metallic.param.ior,
                               metallic.param.energy_scale));
      output.write(output_base + 6u, make_float4(metallic.param.T, 0.0f));
      output.write(output_base + 7u,
                   make_float4(metallic.f82_tint.thin_film.thickness,
                               metallic.f82_tint.thin_film.ior, 0.0f, 0.0f));
      output.write(output_base + 8u, make_float4(metallic.f82_tint.f0, 0.0f));
      output.write(output_base + 9u, make_float4(metallic.f82_tint.b, 0.0f));
      meta.write(meta_base + 6u, metallic.common.type);
      meta.write(meta_base + 7u, dielectric.type);
      meta.write(meta_base + 8u, diffuse.type);
      meta.write(meta_base + 9u, metallic.param.fresnel_type);
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

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table_values =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values(
          psycles::contract::ShaderColorSpace{});
  auto table = device.create_buffer<float>(table_values.size());
  auto words =
      device.create_buffer<std::uint32_t>(principled_metallic_words.size());
  auto output = device.create_buffer<luisa::float4>(2u * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(2u * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto shader = device.compile(transition_kernel(), options);

  std::array<luisa::float4, 2u * output_stride> actual{};
  std::array<std::uint32_t, 2u * meta_stride> actual_meta{};
  stream << table.copy_from(luisa::span{table_values})
         << words.copy_from(principled_metallic_words.data())
         << shader(words, table, output, meta).dispatch(2u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  constexpr auto expected_flags =
      device_svm::shader_data_use_bump_map_correction |
      device_svm::shader_data_emission | device_svm::shader_data_bsdf |
      device_svm::shader_data_bsdf_has_eval |
      device_svm::shader_data_transparent;
  constexpr auto transparent_weight = 0.2200000286102295f;
  const auto reflective = 0u;
  const auto disabled = output_stride;
  const auto reflective_meta = 0u;
  const auto disabled_meta = meta_stride;
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  const auto zero = luisa::float3{0.0f};
  auto valid =
      near(actual[reflective + 0u].xyz(), luisa::float3{transparent_weight}) &&
      near(actual[reflective + 0u].w, transparent_weight) &&
      near(actual[reflective + 1u].xyz(),
           luisa::float3{0.4725275338f, 0.4821479917f, 0.4927247167f}) &&
      near(actual[reflective + 1u].w, 0.2328373790f) &&
      near(actual[reflective + 2u].xyz(),
           luisa::float3{0.2634941339f, 0.2633501291f, 0.2635681629f}) &&
      near(actual[reflective + 2u].w, 0.0085809864f) &&
      near(actual[reflective + 3u].xyz(),
           luisa::float3{0.0478230119f, 0.1461258680f, 0.2178603709f}) &&
      near(actual[reflective + 3u].w, 0.1372697651f) &&
      near(actual[reflective + 4u].xyz(), up) &&
      near(actual[reflective + 5u].x, 0.2723220289f) &&
      near(actual[reflective + 5u].y, 0.1791878939f) &&
      near(actual[reflective + 5u].z, 1.0f) &&
      std::isfinite(actual[reflective + 5u].w) &&
      actual[reflective + 5u].w >= 1.0f &&
      near(actual[reflective + 6u].xyz(),
           luisa::float3{0.2486899495f, 0.9685831666f, 0.0f}) &&
      near(actual[reflective + 7u].x, 360.0f) &&
      near(actual[reflective + 7u].y, 1.52f) &&
      near(actual[reflective + 8u].xyz(), luisa::float3{0.18f, 0.55f, 0.82f}) &&
      near(actual[reflective + 9u].xyz(),
           luisa::float3{2.76469636f, 8.69911861f, 1.43497205f}) &&
      near(actual[reflective + 10u].xyz(),
           luisa::float3{0.2144999951f, 0.0682499930f, 0.4679999650f}) &&
      near(actual[reflective + 10u].w, transparent_weight) &&
      actual_meta[reflective_meta + 0u] == 4u &&
      actual_meta[reflective_meta + 1u] == 2u &&
      actual_meta[reflective_meta + 2u] == expected_flags &&
      actual_meta[reflective_meta + 3u] ==
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
      actual_meta[reflective_meta + 4u] == 58u &&
      actual_meta[reflective_meta + 5u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID) &&
      actual_meta[reflective_meta + 6u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID) &&
      actual_meta[reflective_meta + 7u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID) &&
      actual_meta[reflective_meta + 8u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID) &&
      actual_meta[reflective_meta + 9u] ==
          static_cast<std::uint32_t>(device_svm::MicrofacetFresnel::f82_tint);

  /* Cycles attenuates lower BSDF components even when reflective caustics
   * suppress both reflective allocations. Emission precedes Metallic and is
   * therefore unchanged. */
  valid &= near(actual[disabled + 1u].xyz(), zero) &&
           near(actual[disabled + 2u].xyz(), zero) &&
           near(actual[disabled + 3u].xyz(),
                luisa::float3{0.050544f, 0.15444f, 0.230256f}) &&
           near(actual[disabled + 3u].w, 0.14508f) &&
           near(actual[disabled + 10u].xyz(),
                luisa::float3{0.2145f, 0.06825f, 0.468f}) &&
           actual_meta[disabled_meta + 0u] == 2u &&
           actual_meta[disabled_meta + 1u] == 6u &&
           actual_meta[disabled_meta + 2u] == expected_flags &&
           actual_meta[disabled_meta + 4u] == 58u &&
           actual_meta[disabled_meta + 6u] ==
               static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           actual_meta[disabled_meta + 7u] ==
               static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           actual_meta[disabled_meta + 8u] ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);

  if (!valid) {
    std::cerr << "Cycles Principled Metallic transition mismatch on " << backend
              << "\nmetallic=(" << actual[reflective + 1u].x << ", "
              << actual[reflective + 1u].y << ", " << actual[reflective + 1u].z
              << ", " << actual[reflective + 1u].w << "), alpha-energy=("
              << actual[reflective + 5u].x << ", " << actual[reflective + 5u].y
              << ", " << actual[reflective + 5u].z << ", "
              << actual[reflective + 5u].w << "), tangent=("
              << actual[reflective + 6u].x << ", " << actual[reflective + 6u].y
              << ", " << actual[reflective + 6u].z << ")\n";
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

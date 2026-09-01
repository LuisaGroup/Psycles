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

/* Compact surface image from the unmodified Cycles 5.2.1
 * `principled_coat_svm_oracle`. Only its global shader jump is relocated;
 * the typed Principled payload is word-identical to the external dump. */
constexpr std::array<std::uint32_t, 53u> principled_coat_words{
    0x0000000bu, 0x00000001u, 0x00000000u, 0x00000002u, 0x0000002bu,
    0x000000ffu, 0x0000001au, 0x3f800000u, 0x3ed70a3du, 0x00000000u,
    0x3f266666u, 0x00000000u, 0x00000000u, 0x00000000u, 0x3e9eb852u,
    0x3f11eb85u, 0x3f547ae1u, 0x3f4ccccdu, 0x00000000u, 0x0000ff00u,
    0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f000000u, 0x00000000u,
    0x00000000u, 0x3e4ccccdu, 0x3dcccccdu, 0x3f19999au, 0x3fa66666u,
    0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f000000u, 0x3f333333u,
    0x3f59999au, 0x3f0ccccdu, 0x3e8f5c29u, 0x3fb9999au, 0x00000020u,
    0x3f800000u, 0x3e4ccccdu, 0x3dcccccdu, 0x3ba3d70au, 0x3fb33333u,
    0x00000000u, 0x00000000u, 0x3faa3d71u, 0x00000000u, 0x000000ffu,
    0x00000000u, 0x00000000u, 0x00000000u};

class CoatKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;
  Bool _reflective_caustics;

public:
  CoatKernelGlobals(const BufferFloat &table,
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

inline constexpr std::uint32_t output_stride = 7u;
inline constexpr std::uint32_t meta_stride = 9u;

[[nodiscard]] auto bsdf_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[used](BufferUInt words,
                                                BufferFloat table,
                                                BufferFloat4 output,
                                                BufferUInt meta) noexcept {
    const auto case_index = dispatch_x();
    const Bool reflective_caustics = case_index == 0u;
    const CoatKernelGlobals kernel_globals{table, reflective_caustics};
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

    const auto transparent = closures.common(0u);
    const auto output_base = case_index * output_stride;
    const auto meta_base = case_index * meta_stride;
    output.write(output_base + 0u,
                 make_float4(transparent.weight, transparent.sample_weight));
    output.write(output_base + 1u, make_float4(0.0f));
    output.write(output_base + 2u, make_float4(0.0f));
    output.write(output_base + 3u, make_float4(0.0f));
    output.write(output_base + 4u, make_float4(0.0f));
    output.write(output_base + 5u, make_float4(0.0f));
    output.write(output_base + 6u,
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
    meta.write(meta_base + 8u,
               static_cast<std::uint32_t>(device_svm::MicrofacetFresnel::none));

    $if(reflective_caustics) {
      const auto coat = closures.microfacet(1u);
      const auto diffuse = closures.common(2u);
      output.write(output_base + 1u,
                   make_float4(coat.common.weight, coat.common.sample_weight));
      output.write(output_base + 2u,
                   make_float4(diffuse.weight, diffuse.sample_weight));
      output.write(output_base + 3u, make_float4(coat.common.N, 0.0f));
      output.write(output_base + 4u,
                   make_float4(coat.param.alpha_x, coat.param.alpha_y,
                               coat.param.ior, coat.param.energy_scale));
      output.write(output_base + 5u, make_float4(coat.param.T, 0.0f));
      meta.write(meta_base + 6u, coat.common.type);
      meta.write(meta_base + 7u, diffuse.type);
      meta.write(meta_base + 8u, coat.param.fresnel_type);
    }
    $else {
      const auto diffuse = closures.common(1u);
      output.write(output_base + 2u,
                   make_float4(diffuse.weight, diffuse.sample_weight));
      meta.write(meta_base + 7u, diffuse.type);
    };
  }};
}

[[nodiscard]] auto emission_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[used](BufferUInt words,
                                                BufferFloat table,
                                                BufferFloat4 output,
                                                BufferUInt meta) noexcept {
    const CoatKernelGlobals kernel_globals{table, true};
    device_svm::ClosurePool closures{8u};
    auto shader_data = make_shader_data(&closures);
    const device_svm::PathState path_state{
        device_svm::path_ray_visibility_camera |
            device_svm::path_ray_visibility_diffuse,
        device_svm::path_ray_emission};
    device_svm::EvaluationResult result;
    device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                           device_svm::kernel_feature_node_emission, used,
                           identity_transform_state(), shader_data, path_state,
                           result);
    const auto transparent = closures.common(0u);
    output.write(0u, make_float4(shader_data.closure_emission_background,
                                 shader_data.closure_transparent_extinction.x));
    output.write(1u,
                 make_float4(transparent.weight, transparent.sample_weight));
    meta.write(0u, closures.count());
    meta.write(1u, closures.left());
    meta.write(2u, shader_data.flag);
    meta.write(3u, result.status);
    meta.write(4u, result.final_offset);
    meta.write(5u, transparent.type);
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
      device.create_buffer<std::uint32_t>(principled_coat_words.size());
  auto output = device.create_buffer<luisa::float4>(2u * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(2u * meta_stride);
  auto emission_output = device.create_buffer<luisa::float4>(2u);
  auto emission_meta = device.create_buffer<std::uint32_t>(6u);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto bsdf = device.compile(bsdf_kernel(), options);
  auto emission = device.compile(emission_kernel(), options);

  std::array<luisa::float4, 2u * output_stride> actual{};
  std::array<std::uint32_t, 2u * meta_stride> actual_meta{};
  std::array<luisa::float4, 2u> actual_emission{};
  std::array<std::uint32_t, 6u> actual_emission_meta{};
  stream << table.copy_from(luisa::span{table_values})
         << words.copy_from(principled_coat_words.data())
         << bsdf(words, table, output, meta).dispatch(2u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << emission(words, table, emission_output, emission_meta).dispatch(1u)
         << emission_output.copy_to(actual_emission.data())
         << emission_meta.copy_to(actual_emission_meta.data()) << synchronize();

  constexpr auto expected_flags =
      device_svm::shader_data_use_bump_map_correction |
      device_svm::shader_data_emission | device_svm::shader_data_bsdf |
      device_svm::shader_data_bsdf_has_eval |
      device_svm::shader_data_transparent;
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  const auto zero = luisa::float3{0.0f};
  const auto reflective_base = 0u;
  const auto disabled_base = output_stride;
  const auto reflective_meta = 0u;
  const auto disabled_meta = meta_stride;
  auto valid =
      near(actual[reflective_base + 0u].xyz(), luisa::float3{0.1999999881f}) &&
      near(actual[reflective_base + 0u].w, 0.1999999881f) &&
      near(actual[reflective_base + 1u].xyz(), luisa::float3{0.5166048408f}) &&
      near(actual[reflective_base + 1u].w, 0.0174650792f) &&
      near(actual[reflective_base + 2u].xyz(),
           luisa::float3{0.1952815950f, 0.4025555849f, 0.4595240653f}) &&
      near(actual[reflective_base + 2u].w, 0.3524537683f) &&
      near(actual[reflective_base + 3u].xyz(), up) &&
      near(actual[reflective_base + 4u].x, 0.0784f) &&
      near(actual[reflective_base + 4u].y, 0.0784f) &&
      near(actual[reflective_base + 4u].z, 1.45f) &&
      std::isfinite(actual[reflective_base + 4u].w) &&
      actual[reflective_base + 4u].w >= 1.0f &&
      near(actual[reflective_base + 5u].xyz(), zero) &&
      near(actual[reflective_base + 6u].xyz(),
           luisa::float3{0.1637845635f, 0.0918109193f, 0.4318419099f}) &&
      near(actual[reflective_base + 6u].w, 0.1999999881f) &&
      actual_meta[reflective_meta + 0u] == 3u &&
      actual_meta[reflective_meta + 1u] == 5u &&
      actual_meta[reflective_meta + 2u] == expected_flags &&
      actual_meta[reflective_meta + 3u] ==
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
      actual_meta[reflective_meta + 4u] == 51u &&
      actual_meta[reflective_meta + 5u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID) &&
      actual_meta[reflective_meta + 6u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID) &&
      actual_meta[reflective_meta + 7u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID) &&
      actual_meta[reflective_meta + 8u] ==
          static_cast<std::uint32_t>(device_svm::MicrofacetFresnel::dielectric);

  /* Cycles' coat tint is outside the reflective-caustics branch. Disabling
   * the GGX closure must therefore leave the Beer attenuation of emission
   * and diffuse active, while consuming no coat slot. */
  valid &= near(actual[disabled_base + 1u].xyz(), zero) &&
           near(actual[disabled_base + 2u].xyz(),
                luisa::float3{0.19964f, 0.41154f, 0.46978f}) &&
           near(actual[disabled_base + 2u].w, 0.36032f) &&
           near(actual[disabled_base + 6u].xyz(),
                luisa::float3{0.16744f, 0.09386f, 0.44148f}) &&
           actual_meta[disabled_meta + 0u] == 2u &&
           actual_meta[disabled_meta + 1u] == 6u &&
           actual_meta[disabled_meta + 2u] == expected_flags &&
           actual_meta[disabled_meta + 4u] == 51u &&
           actual_meta[disabled_meta + 6u] ==
               static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           actual_meta[disabled_meta + 7u] ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);

  /* PATH_RAY_EMISSION uses Cycles' stack-local MicrofacetBsdf. Coat changes
   * emission and flags without consuming a closure slot; transparency keeps
   * its ordinary slot and lower BSDF layers are not evaluated. */
  valid &=
      near(actual_emission[0u].xyz(),
           luisa::float3{0.1637845635f, 0.0918109193f, 0.4318419099f}) &&
      near(actual_emission[0u].w, 0.1999999881f) &&
      near(actual_emission[1u].xyz(), luisa::float3{0.1999999881f}) &&
      near(actual_emission[1u].w, 0.1999999881f) &&
      actual_emission_meta[0u] == 1u && actual_emission_meta[1u] == 7u &&
      actual_emission_meta[2u] == expected_flags &&
      actual_emission_meta[3u] ==
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
      actual_emission_meta[4u] == 51u &&
      actual_emission_meta[5u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID);

  if (!valid) {
    std::cerr << "Cycles Principled Coat transition mismatch on " << backend
              << "\ncoat=(" << actual[reflective_base + 1u].x << ", "
              << actual[reflective_base + 1u].y << ", "
              << actual[reflective_base + 1u].z << ", "
              << actual[reflective_base + 1u].w << "), diffuse=("
              << actual[reflective_base + 2u].x << ", "
              << actual[reflective_base + 2u].y << ", "
              << actual[reflective_base + 2u].z << ", "
              << actual[reflective_base + 2u].w << "), emission=("
              << actual[reflective_base + 6u].x << ", "
              << actual[reflective_base + 6u].y << ", "
              << actual[reflective_base + 6u].z << "), param=("
              << actual[reflective_base + 4u].x << ", "
              << actual[reflective_base + 4u].y << ", "
              << actual[reflective_base + 4u].z << ", "
              << actual[reflective_base + 4u].w << ")\n";
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

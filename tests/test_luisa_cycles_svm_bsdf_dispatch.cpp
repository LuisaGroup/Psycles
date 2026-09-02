#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_bsdf.h"
#include "cycles_svm_microfacet.h"
#include "cycles_svm_microfacet_scattering.h"
#include "cycles_svm_principled_hair_chiang.h"
#include "cycles_svm_ray_portal.h"
#include "cycles_svm_simple_closure.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace detail = psycles::luisa_backend::cycles_svm::detail;
namespace device_svm = psycles::luisa_backend::cycles_svm;

inline constexpr std::uint32_t scenario_count = 6u;
inline constexpr std::uint32_t values_per_scenario = 10u;
inline constexpr std::uint32_t meta_per_scenario = 3u;
inline constexpr std::uint32_t table_probe_count = 2u;
inline constexpr std::uint32_t modifier_value_count = 6u;
inline constexpr std::uint32_t modifier_meta_count = 2u;
inline constexpr std::uint32_t table_probe_base =
    scenario_count * values_per_scenario;
inline constexpr std::uint32_t modifier_value_base =
    table_probe_base + table_probe_count;
inline constexpr std::uint32_t modifier_meta_base =
    scenario_count * meta_per_scenario;
inline constexpr std::uint32_t value_count =
    modifier_value_base + modifier_value_count;
inline constexpr std::uint32_t meta_count =
    modifier_meta_base + modifier_meta_count;
inline constexpr auto closure_mask =
    (detail::ClosureTypeMask{1u} << closure::type_diffuse) |
    (detail::ClosureTypeMask{1u} << closure::type_microfacet_beckmann) |
    (detail::ClosureTypeMask{1u} << closure::type_hair_chiang) |
    (detail::ClosureTypeMask{1u} << closure::type_ray_portal) |
    (detail::ClosureTypeMask{1u} << closure::type_microfacet_ggx_glass);
inline constexpr auto modifier_closure_mask =
    (detail::ClosureTypeMask{1u} << closure::type_diffuse) |
    (detail::ClosureTypeMask{1u} << closure::type_rough_translucent);
inline constexpr auto diffuse_only_mask = detail::ClosureTypeMask{1u}
                                          << closure::type_diffuse;

class TableSentinelKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
public:
  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    /* Cycles 5.2.1 uses distinct 4096-entry tables for the negative
     * dielectric exponent domain and the non-negative generalized
     * Schlick domain. Constant sentinels make the selected address range
     * observable without duplicating the interpolation implementation. */
    return select(
        0.75f, 0.25f,
        index >=
            psycles::luisa_backend::cycles45_tables::ggx_gen_schlick_s_offset);
  }
};

class ModifierKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
public:
  [[nodiscard]] Float
  transparent_roughness_squared_threshold() const noexcept override {
    return 1.0f;
  }

  [[nodiscard]] Float object_shadow_terminator_shading_offset(
      Expr<std::uint32_t>) const noexcept override {
    return 2.0f;
  }
};

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *pool) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          normalize(make_float3(0.32f, 0.88f, 0.35f)),
          0u,
          0u,
          0u,
          0u,
          0u,
          0.0f,
          0.0f,
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
          0x12345678u,
          pool};
}

void assign(detail::BsdfEvaluation &destination,
            const detail::BsdfEvaluation &source) noexcept {
  destination.value = source.value;
  destination.pdf = source.pdf;
}

void assign(detail::BsdfSample &destination,
            const detail::BsdfSample &source) noexcept {
  destination.value = source.value;
  destination.wo = source.wo;
  destination.pdf = source.pdf;
  destination.sampled_roughness = source.sampled_roughness;
  destination.eta = source.eta;
  destination.label = source.label;
}

[[nodiscard]] auto dispatch_kernel() {
  return Kernel1D<Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[](BufferFloat4 values,
                                            BufferUInt meta) noexcept {
    const UInt scenario = dispatch_id().x;
    UInt type = closure::type_diffuse;
    $switch(scenario) {
      $case(0u) { type = closure::type_diffuse; };
      $case(1u) { type = closure::type_microfacet_beckmann; };
      $case(2u) { type = closure::type_microfacet_beckmann; };
      $case(3u) { type = closure::type_hair_chiang; };
      $case(4u) { type = closure::type_ray_portal; };
      $default { type = closure::type_microfacet_ggx_glass; };
    };

    device_svm::ClosurePool pool{2u};
    const auto allocated = pool.allocate(type, make_float3(0.8f, 0.4f, 0.2f));
    pool.set_sample_weight(allocated.index, 0.466666698f);

    const auto ordinary_normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
    pool.set_normal(allocated.index, ordinary_normal);
    $if(scenario == 1u) {
      static_cast<void>(pool.allocate_extra(allocated, 1u));
      pool.set_microfacet_param(
          allocated.index, {.alpha_x = 0.31f,
                            .alpha_y = 0.57f,
                            .ior = 1.0f,
                            .energy_scale = 1.0f,
                            .fresnel_type = static_cast<std::uint32_t>(
                                device_svm::MicrofacetFresnel::conductor),
                            .T = normalize(make_float3(0.9f, 0.2f, -0.1f))});
      pool.set_fresnel_conductor(allocated.index,
                                 {.thin_film = {.thickness = 0.0f, .ior = 1.4f},
                                  .ior = make_float3(0.3f, 0.8f, 1.2f),
                                  .extinction = make_float3(3.0f, 2.0f, 1.5f)});
    }
    $elif(scenario == 2u) {
      static_cast<void>(pool.allocate_extra(allocated, 1u));
      pool.set_microfacet_param(
          allocated.index, {.alpha_x = 0.42f,
                            .alpha_y = 0.18f,
                            .ior = 1.0f,
                            .energy_scale = 1.0f,
                            .fresnel_type = static_cast<std::uint32_t>(
                                device_svm::MicrofacetFresnel::f82_tint),
                            .T = normalize(make_float3(0.9f, 0.2f, -0.1f))});
      pool.set_fresnel_f82_tint(allocated.index,
                                {.thin_film = {.thickness = 0.0f, .ior = 1.4f},
                                 .f0 = make_float3(0.8f, 0.4f, 0.1f),
                                 .b = make_float3(0.12f, -0.08f, 0.04f)});
    }
    $elif(scenario == 3u) {
      const auto incoming = normalize(make_float3(0.32f, 0.88f, 0.35f));
      const auto tangent = make_float3(1.0f, 0.0f, 0.0f);
      pool.set_normal(allocated.index, normalize(cross(tangent, incoming)));
      pool.set_chiang_hair_param(allocated.index,
                                 {.sigma = make_float3(0.12f, 0.34f, 0.56f),
                                  .v = 0.36f,
                                  .s = 0.31f,
                                  .alpha = -0.23f,
                                  .eta = 1.42f,
                                  .m0_roughness = 0.07f,
                                  .h = 0.23f});
    }
    $elif(scenario == 4u) {
      pool.set_ray_portal_param(
          allocated.index, {.P = make_float3(1.0f, 2.0f, 3.0f),
                            .D = normalize(make_float3(-0.2f, 0.4f, 0.7f))});
    }
    $elif(scenario == 5u) {
      pool.set_microfacet_param(allocated.index,
                                {.alpha_x = 0.28f,
                                 .alpha_y = 0.28f,
                                 .ior = 1.45f,
                                 .energy_scale = 1.09f,
                                 .fresnel_type = static_cast<std::uint32_t>(
                                     device_svm::MicrofacetFresnel::dielectric),
                                 .T = make_float3(0.0f)});
    };

    psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
    auto generic_shader_data = make_shader_data(&pool);
    auto direct_shader_data = make_shader_data(&pool);
    const auto common = pool.common(allocated.index);
    generic_shader_data.N = common.N;
    generic_shader_data.Ng = common.N;
    direct_shader_data.N = common.N;
    direct_shader_data.Ng = common.N;

    const auto random = make_float3(0.37f, 0.73f, 0.42f);
    const auto evaluation_direction =
        normalize(make_float3(-0.41f, 0.37f, 0.83f));
    const auto generic_sample =
        detail::bsdf_sample(kernel_globals, generic_shader_data,
                            allocated.index, random, closure_mask);
    const auto generic_evaluation =
        detail::bsdf_eval(kernel_globals, generic_shader_data, allocated.index,
                          evaluation_direction, closure_mask);

    detail::BsdfSample direct_sample{.value = make_float3(0.0f),
                                     .wo = make_float3(0.0f),
                                     .pdf = 0.0f,
                                     .sampled_roughness = make_float2(0.0f),
                                     .eta = 0.0f,
                                     .label = closure::label_none};
    detail::BsdfEvaluation direct_evaluation{.value = make_float3(0.0f),
                                             .pdf = 0.0f};
    $switch(scenario) {
      $case(0u) {
        assign(direct_sample,
               detail::bsdf_diffuse_sample(common, common.N,
                                           direct_shader_data.wi, random.xy()));
        assign(direct_evaluation,
               detail::bsdf_diffuse_eval(common, direct_shader_data.wi,
                                         evaluation_direction));
      };
      $case(1u) {
        const auto closure_value = pool.microfacet_conductor(allocated.index);
        assign(direct_sample, detail::bsdf_microfacet_beckmann_sample(
                                  kernel_globals, closure_value, common.N,
                                  direct_shader_data.wi, random));
        assign(direct_evaluation,
               detail::bsdf_microfacet_beckmann_eval(
                   kernel_globals, closure_value, direct_shader_data.wi,
                   evaluation_direction));
      };
      $case(2u) {
        const auto closure_value = pool.microfacet_f82_tint(allocated.index);
        assign(direct_sample, detail::bsdf_microfacet_beckmann_sample(
                                  kernel_globals, closure_value, common.N,
                                  direct_shader_data.wi, random));
        assign(direct_evaluation,
               detail::bsdf_microfacet_beckmann_eval(
                   kernel_globals, closure_value, direct_shader_data.wi,
                   evaluation_direction));
      };
      $case(3u) {
        const auto closure_value = pool.chiang_hair(allocated.index);
        assign(direct_sample,
               detail::bsdf_hair_chiang_sample(kernel_globals, closure_value,
                                               direct_shader_data, random));
        assign(direct_evaluation,
               detail::bsdf_hair_chiang_eval(kernel_globals, closure_value,
                                             direct_shader_data,
                                             evaluation_direction));
      };
      $case(4u) {
        assign(direct_evaluation,
               detail::bsdf_ray_portal_eval(pool.ray_portal(allocated.index),
                                            direct_shader_data.wi,
                                            evaluation_direction));
      };
      $default {
        const auto closure_value = pool.microfacet(allocated.index);
        assign(direct_sample, detail::bsdf_microfacet_ggx_sample(
                                  kernel_globals, closure_value, common.N,
                                  direct_shader_data.wi, random));
        assign(direct_evaluation,
               detail::bsdf_microfacet_ggx_eval(kernel_globals, closure_value,
                                                direct_shader_data.wi,
                                                evaluation_direction));
      };
    };

    const auto generic_albedo =
        detail::bsdf_albedo(kernel_globals, generic_shader_data,
                            allocated.index, true, true, closure_mask);
    Float3 direct_albedo = common.weight;
    $if(scenario == 1u) {
      direct_albedo *= detail::bsdf_microfacet_estimate_albedo(
          kernel_globals, pool.microfacet_conductor(allocated.index),
          direct_shader_data.wi, true, true);
    }
    $elif(scenario == 2u) {
      direct_albedo *= detail::bsdf_microfacet_estimate_albedo(
          kernel_globals, pool.microfacet_f82_tint(allocated.index),
          direct_shader_data.wi, true, true);
    }
    $elif(scenario == 3u) {
      direct_albedo *= detail::bsdf_hair_chiang_albedo(
          pool.chiang_hair(allocated.index), direct_shader_data);
    }
    $elif(scenario == 5u) {
      direct_albedo *= detail::bsdf_microfacet_estimate_albedo(
          kernel_globals, pool.microfacet(allocated.index),
          direct_shader_data.wi, true, true);
    };

    const auto roughness_eta = detail::bsdf_roughness_eta(
        pool, allocated.index, common.N, closure_mask);
    const auto label = detail::bsdf_label(kernel_globals, pool, allocated.index,
                                          common.N, closure_mask);
    detail::bsdf_blur(pool, allocated.index, 0.73f, closure_mask);

    Float4 blurred = make_float4(0.0f);
    $if((scenario == 1u) | (scenario == 2u) | (scenario == 5u)) {
      const auto param = pool.microfacet_param(allocated.index);
      blurred = make_float4(param.alpha_x, param.alpha_y, 0.0f, 0.0f);
    }
    $elif(scenario == 3u) {
      const auto param = pool.chiang_hair(allocated.index).param;
      blurred = make_float4(param.v, param.s, param.m0_roughness, 0.0f);
    };

    const auto base = scenario * values_per_scenario;
    values.write(base + 0u, make_float4(generic_sample.wo, generic_sample.pdf));
    values.write(base + 1u, make_float4(direct_sample.wo, direct_sample.pdf));
    values.write(base + 2u, make_float4(generic_sample.value,
                                        generic_sample.sampled_roughness.x));
    values.write(base + 3u, make_float4(direct_sample.value,
                                        direct_sample.sampled_roughness.x));
    values.write(base + 4u,
                 make_float4(generic_evaluation.value, generic_evaluation.pdf));
    values.write(base + 5u,
                 make_float4(direct_evaluation.value, direct_evaluation.pdf));
    values.write(base + 6u, make_float4(generic_albedo, generic_sample.eta));
    values.write(base + 7u, make_float4(direct_albedo, direct_sample.eta));
    values.write(base + 8u, blurred);
    values.write(base + 9u,
                 make_float4(roughness_eta.roughness, roughness_eta.eta, 0.0f));

    const auto meta_base = scenario * meta_per_scenario;
    meta.write(meta_base + 0u, generic_sample.label);
    meta.write(meta_base + 1u, direct_sample.label);
    meta.write(meta_base + 2u, label);

    $if(scenario == 0u) {
      TableSentinelKernelGlobals table_kernel_globals;
      device_svm::MicrofacetClosure table_closure{
          .common = {.weight = make_float3(1.0f),
                     .type = closure::type_microfacet_ggx,
                     .sample_weight = 1.0f,
                     .N = ordinary_normal},
          .param = {.alpha_x = 0.35f,
                    .alpha_y = 0.55f,
                    .ior = 1.5f,
                    .energy_scale = 1.0f,
                    .fresnel_type = static_cast<std::uint32_t>(
                        device_svm::MicrofacetFresnel::generalized_schlick),
                    .T = make_float3(0.0f)},
          .generalized_schlick = {.thin_film = {.thickness = 0.0f, .ior = 1.4f},
                                  .reflection_tint = make_float3(1.0f),
                                  .transmission_tint = make_float3(0.0f),
                                  .f0 = make_float3(0.1f, 0.2f, 0.3f),
                                  .f90 = make_float3(0.9f, 0.8f, 0.7f),
                                  .exponent = 2.0f}};
      const auto positive = detail::bsdf_microfacet_estimate_albedo(
          table_kernel_globals, table_closure, direct_shader_data.wi, true,
          false);
      table_closure.generalized_schlick.exponent = -1.5f;
      const auto negative = detail::bsdf_microfacet_estimate_albedo(
          table_kernel_globals, table_closure, direct_shader_data.wi, true,
          false);
      values.write(table_probe_base + 0u, make_float4(positive, 0.0f));
      values.write(table_probe_base + 1u, make_float4(negative, 0.0f));

      ModifierKernelGlobals modifier_kernel_globals;
      device_svm::ClosurePool rough_translucent_pool{1u};
      const auto rough_translucent = rough_translucent_pool.allocate(
          closure::type_rough_translucent, make_float3(1.0f));
      rough_translucent_pool.set_normal(rough_translucent.index,
                                        ordinary_normal);
      rough_translucent_pool.set_oren_nayar_param(
          rough_translucent.index, {.roughness = 0.4f,
                                    .a = 0.8f,
                                    .b = 0.2f,
                                    .multiscatter_term = make_float3(0.1f)});
      auto rough_shader_data = make_shader_data(&rough_translucent_pool);
      rough_shader_data.N = ordinary_normal;
      rough_shader_data.Ng = ordinary_normal;
      const auto rough_sample = detail::bsdf_sample(
          modifier_kernel_globals, rough_shader_data, rough_translucent.index,
          random, modifier_closure_mask);
      const auto rough_label = detail::bsdf_label(
          modifier_kernel_globals, rough_translucent_pool,
          rough_translucent.index, -ordinary_normal, modifier_closure_mask);
      meta.write(modifier_meta_base + 0u, rough_sample.label);
      meta.write(modifier_meta_base + 1u, rough_label);

      device_svm::ClosurePool bump_pool{1u};
      const auto bump_closure =
          bump_pool.allocate(closure::type_diffuse, make_float3(1.0f));
      const auto bump_normal = make_float3(0.8f, 0.0f, 0.6f);
      bump_pool.set_normal(bump_closure.index, bump_normal);
      auto bump_shader_data = make_shader_data(&bump_pool);
      bump_shader_data.N = make_float3(0.0f, 0.0f, 1.0f);
      bump_shader_data.Ng = bump_normal;
      const auto bump_direction = make_float3(0.8f, 0.0f, -0.6f);
      const auto corrected_bump = detail::bsdf_eval(
          kernel_globals, bump_shader_data, bump_closure.index, bump_direction,
          modifier_closure_mask);
      const auto raw_bump =
          detail::bsdf_diffuse_eval(bump_pool.common(bump_closure.index),
                                    bump_shader_data.wi, bump_direction);
      values.write(modifier_value_base + 0u,
                   make_float4(corrected_bump.value, corrected_bump.pdf));
      values.write(modifier_value_base + 1u,
                   make_float4(raw_bump.value, raw_bump.pdf));

      auto sample_shader_data = make_shader_data(&bump_pool);
      sample_shader_data.N = bump_normal;
      sample_shader_data.Ng = bump_normal;
      const auto neutral_sample = detail::bsdf_sample(
          kernel_globals, sample_shader_data, bump_closure.index, random,
          modifier_closure_mask);
      const auto shifted_sample = detail::bsdf_sample(
          modifier_kernel_globals, sample_shader_data, bump_closure.index,
          random, modifier_closure_mask);
      values.write(modifier_value_base + 2u,
                   make_float4(neutral_sample.wo, neutral_sample.pdf));
      values.write(modifier_value_base + 3u,
                   make_float4(shifted_sample.wo, shifted_sample.pdf));
      values.write(modifier_value_base + 4u,
                   make_float4(neutral_sample.value,
                               neutral_sample.sampled_roughness.x));
      values.write(modifier_value_base + 5u,
                   make_float4(shifted_sample.value,
                               shifted_sample.sampled_roughness.x));
    };
  }};
}

/* This separate shader keeps the ClosureType device value dynamic while its
 * host/JIT reachability domain contains only Diffuse. Backend code-size logs
 * therefore measure whether absent Cycles switch families were eliminated,
 * rather than constant-folding the entire dispatch from a literal type. */
[[nodiscard]] auto diffuse_only_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 values, BufferUInt meta) noexcept {
        const UInt type = closure::type_diffuse + dispatch_id().x;
        device_svm::ClosurePool pool{1u};
        const auto allocated =
            pool.allocate(type, make_float3(0.8f, 0.4f, 0.2f));
        const auto normal = normalize(make_float3(0.2f, -0.3f, 1.0f));
        pool.set_normal(allocated.index, normal);

        psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
        auto shader_data = make_shader_data(&pool);
        shader_data.N = normal;
        shader_data.Ng = normal;
        const auto random = make_float3(0.37f, 0.73f, 0.42f);
        const auto sample =
            detail::bsdf_sample(kernel_globals, shader_data, allocated.index,
                                random, diffuse_only_mask);
        const auto evaluation =
            detail::bsdf_eval(kernel_globals, shader_data, allocated.index,
                              sample.wo, diffuse_only_mask);
        const auto albedo =
            detail::bsdf_albedo(kernel_globals, shader_data, allocated.index,
                                true, true, diffuse_only_mask);
        const auto roughness_eta = detail::bsdf_roughness_eta(
            pool, allocated.index, sample.wo, diffuse_only_mask);
        const auto specular_roughness =
            detail::bsdf_get_specular_roughness_squared(pool, allocated.index);
        detail::bsdf_blur(pool, allocated.index, 0.73f, diffuse_only_mask);

        values.write(0u, make_float4(sample.value, sample.pdf));
        values.write(1u, make_float4(evaluation.value, evaluation.pdf));
        values.write(2u, make_float4(albedo, sample.sampled_roughness.x));
        values.write(3u, make_float4(roughness_eta.roughness, roughness_eta.eta,
                                     specular_roughness));
        meta.write(0u, sample.label);
        meta.write(1u, detail::bsdf_label(kernel_globals, pool, allocated.index,
                                          sample.wo, diffuse_only_mask));
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 8.0e-4f) noexcept {
  return std::isfinite(actual) && std::abs(actual - expected) <=
                                      2.0e-6f + tolerance * std::abs(expected);
}

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected) noexcept {
  return near(actual.x, expected.x) && near(actual.y, expected.y) &&
         near(actual.z, expected.z) && near(actual.w, expected.w);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto values = device.create_buffer<luisa::float4>(value_count);
  auto meta = device.create_buffer<std::uint32_t>(meta_count);
  std::array<luisa::float4, value_count> actual_values{};
  std::array<std::uint32_t, meta_count> actual_meta{};
  const auto shader =
      device.compile(dispatch_kernel(), ShaderOption{.enable_cache = false,
                                                     .enable_fast_math = true});
  const auto diffuse_shader = device.compile(
      diffuse_only_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  auto diffuse_values = device.create_buffer<luisa::float4>(4u);
  auto diffuse_meta = device.create_buffer<std::uint32_t>(2u);
  std::array<luisa::float4, 4u> actual_diffuse_values{};
  std::array<std::uint32_t, 2u> actual_diffuse_meta{};
  stream << shader(values, meta).dispatch(scenario_count)
         << values.copy_to(actual_values.data())
         << meta.copy_to(actual_meta.data())
         << diffuse_shader(diffuse_values, diffuse_meta).dispatch(1u)
         << diffuse_values.copy_to(actual_diffuse_values.data())
         << diffuse_meta.copy_to(actual_diffuse_meta.data()) << synchronize();

  constexpr std::array<std::uint32_t, scenario_count> expected_labels{
      closure::label_reflect | closure::label_diffuse,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_reflect | closure::label_glossy,
      closure::label_transmit | closure::label_ray_portal,
      closure::label_reflect | closure::label_glossy,
  };
  constexpr std::array<luisa::float4, scenario_count> expected_blur{
      luisa::float4{0.0f},
      luisa::float4{0.73f, 0.73f, 0.0f, 0.0f},
      luisa::float4{0.73f, 0.73f, 0.0f, 0.0f},
      luisa::float4{0.73f, 0.73f, 0.73f, 0.0f},
      luisa::float4{0.0f},
      luisa::float4{0.73f, 0.73f, 0.0f, 0.0f},
  };
  constexpr std::array<luisa::float4, scenario_count> expected_roughness_eta{
      luisa::float4{1.0f, 1.0f, 1.0f, 0.0f},
      luisa::float4{0.31f, 0.57f, 1.0f, 0.0f},
      luisa::float4{0.42f, 0.18f, 1.0f, 0.0f},
      luisa::float4{0.07f, 0.07f, 1.0f, 0.0f},
      luisa::float4{0.0f, 0.0f, 1.0f, 0.0f},
      luisa::float4{0.28f, 0.28f, 1.0f, 0.0f},
  };

  auto valid = true;
  for (auto scenario = std::size_t{0u}; scenario < scenario_count; ++scenario) {
    const auto base = scenario * values_per_scenario;
    valid &= near(actual_values[base + 0u], actual_values[base + 1u]);
    valid &= near(actual_values[base + 2u], actual_values[base + 3u]);
    valid &= near(actual_values[base + 4u], actual_values[base + 5u]);
    valid &= near(actual_values[base + 6u], actual_values[base + 7u]);
    valid &= near(actual_values[base + 8u], expected_blur[scenario]);
    valid &= near(actual_values[base + 9u], expected_roughness_eta[scenario]);

    const auto meta_base = scenario * meta_per_scenario;
    valid &= actual_meta[meta_base + 0u] == actual_meta[meta_base + 1u];
    valid &= actual_meta[meta_base + 2u] == expected_labels[scenario];
  }
  valid &= near(actual_values[table_probe_base + 0u],
                luisa::float4{0.3f, 0.35f, 0.4f, 0.0f});
  valid &= near(actual_values[table_probe_base + 1u],
                luisa::float4{0.7f, 0.65f, 0.6f, 0.0f});
  valid &= actual_meta[modifier_meta_base + 0u] ==
           (closure::label_transmit | closure::label_diffuse);
  valid &= actual_meta[modifier_meta_base + 1u] ==
           (closure::label_transmit | closure::label_diffuse |
            closure::label_transmit_transparent);
  valid &= near(actual_values[modifier_value_base + 0u], luisa::float4{0.0f});
  const auto raw_bump = actual_values[modifier_value_base + 1u];
  valid &= raw_bump.x > 0.0f && raw_bump.y > 0.0f && raw_bump.z > 0.0f &&
           raw_bump.w > 0.0f;
  valid &= near(actual_values[modifier_value_base + 2u],
                actual_values[modifier_value_base + 3u]);
  const auto neutral_sample = actual_values[modifier_value_base + 4u];
  const auto shifted_sample = actual_values[modifier_value_base + 5u];
  valid &= near(neutral_sample.w, shifted_sample.w);
  valid &= shifted_sample.x > 0.0f && shifted_sample.x < neutral_sample.x;
  valid &= shifted_sample.y > 0.0f && shifted_sample.y < neutral_sample.y;
  valid &= shifted_sample.z > 0.0f && shifted_sample.z < neutral_sample.z;
  valid &=
      near(actual_diffuse_values[2u], luisa::float4{0.8f, 0.4f, 0.2f, 1.0f});
  valid &=
      near(actual_diffuse_values[3u], luisa::float4{1.0f, 1.0f, 1.0f, 1.0f});
  valid &= actual_diffuse_meta[0u] ==
           (closure::label_reflect | closure::label_diffuse);
  valid &= actual_diffuse_meta[1u] ==
           (closure::label_reflect | closure::label_diffuse);
  if (!valid) {
    std::cerr << "Cycles SVM BSDF dispatch mismatch on " << backend << '\n';
    for (auto index = std::size_t{0u}; index < actual_values.size(); ++index) {
      const auto value = actual_values[index];
      std::cerr << "value[" << index << "] = {" << value.x << ", " << value.y
                << ", " << value.z << ", " << value.w << "}\n";
    }
    for (auto index = std::size_t{0u}; index < actual_meta.size(); ++index) {
      std::cerr << "meta[" << index << "] = " << actual_meta[index] << '\n';
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

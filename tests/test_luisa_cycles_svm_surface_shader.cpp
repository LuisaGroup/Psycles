#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_surface_shader.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace closure = psycles::luisa_backend::cycles_closure;
namespace detail = psycles::luisa_backend::cycles_svm::detail;
namespace device_svm = psycles::luisa_backend::cycles_svm;

inline constexpr std::uint32_t value_count = 18u;
inline constexpr std::uint32_t meta_count = 22u;
inline constexpr std::uint32_t predicate_count = 44u;
inline constexpr auto closure_mask =
    (detail::ClosureTypeMask{1u} << closure::type_diffuse) |
    (detail::ClosureTypeMask{1u} << closure::type_bssrdf_random_walk);

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *pool) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
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

void write_pick(BufferUInt &meta, std::uint32_t base,
                const device_svm::ClosurePool &pool,
                const detail::SurfaceShaderClosurePick &pick) noexcept {
  const auto common = pool.common(pick.index);
  meta.write(base + 0u, pick.index);
  meta.write(base + 1u, common.type);
  meta.write(base + 2u,
             select(0u, 1u, closure::is_bsdf_or_bssrdf(common.type)));
  meta.write(base + 3u, select(0u, 1u, closure::is_bssrdf(common.type)));
}

[[nodiscard]] auto surface_kernel() {
  return Kernel1D<Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [](BufferFloat4 values, BufferUInt meta) noexcept {
        device_svm::ClosurePool pool{4u};
        const auto diffuse_a =
            pool.allocate(closure::type_diffuse, make_float3(0.2f, 0.4f, 0.6f));
        pool.set_sample_weight(diffuse_a.index, 0.25f);
        pool.set_normal(diffuse_a.index, make_float3(0.0f, 0.0f, 1.0f));

        const auto diffuse_b =
            pool.allocate(closure::type_diffuse, make_float3(0.3f, 0.2f, 0.1f));
        pool.set_sample_weight(diffuse_b.index, 0.75f);
        pool.set_normal(diffuse_b.index, make_float3(0.0f, 0.0f, 1.0f));

        const auto bssrdf = pool.allocate(closure::type_bssrdf_random_walk,
                                          make_float3(0.4f, 0.5f, 0.6f));
        pool.set_sample_weight(bssrdf.index, 0.5f);
        pool.set_normal(bssrdf.index, make_float3(0.0f, 0.0f, 1.0f));
        pool.set_bssrdf_param(bssrdf.index, {.radius = make_float3(1.0f),
                                             .albedo = make_float3(0.7f),
                                             .anisotropy = 0.0f,
                                             .ior = 1.4f,
                                             .alpha = 1.0f});

        auto shader_data = make_shader_data(&pool);
        psycles::test_support::DefaultCyclesSvmKernelGlobals kernel_globals;
        const auto wo = normalize(make_float3(0.3f, -0.2f, 1.0f));
        const auto evaluation = detail::surface_shader_bsdf_eval(
            kernel_globals, shader_data, wo, device_svm::shader_use_mis,
            closure_mask);
        const auto no_mis = detail::surface_shader_bsdf_eval(
            kernel_globals, shader_data, wo, 0u, closure_mask);
        const auto excluded = detail::surface_shader_bsdf_eval(
            kernel_globals, shader_data, wo,
            device_svm::shader_use_mis | device_svm::shader_exclude_diffuse,
            closure_mask);

        const auto pick_a = detail::surface_shader_bsdf_bssrdf_pick(
            shader_data, make_float3(0.2f, 0.4f, 0.1f));
        const auto pick_b = detail::surface_shader_bsdf_bssrdf_pick(
            shader_data, make_float3(0.2f, 0.4f, 0.5f));
        const auto pick_bssrdf = detail::surface_shader_bsdf_bssrdf_pick(
            shader_data, make_float3(0.2f, 0.4f, 0.9f));
        const auto bssrdf_weight = detail::surface_shader_bssrdf_sample_weight(
            shader_data, pick_bssrdf.index);
        const auto sample = detail::surface_shader_bsdf_sample_closure(
            kernel_globals, shader_data, pick_b, closure_mask);
        const auto sampled_common = pool.common(pick_b.index);
        const auto sample_evaluation = detail::surface_shader_bsdf_eval(
            kernel_globals, shader_data, sample.wo, device_svm::shader_use_mis,
            closure_mask);

        values.write(0u, make_float4(evaluation.sum, evaluation.pdf));
        values.write(1u, make_float4(evaluation.diffuse,
                                     evaluation.average_roughness_squared));
        values.write(2u, make_float4(evaluation.glossy, 0.0f));
        values.write(3u, make_float4(no_mis.sum, no_mis.pdf));
        values.write(4u, make_float4(excluded.sum, excluded.pdf));
        values.write(5u, make_float4(pick_a.random.z, 0.0f, 0.0f, 0.0f));
        values.write(6u, make_float4(pick_b.random.z, 0.0f, 0.0f, 0.0f));
        values.write(7u, make_float4(pick_bssrdf.random.z, 0.0f, 0.0f, 0.0f));
        values.write(8u, make_float4(bssrdf_weight, 0.0f));
        values.write(9u,
                     make_float4(sample.evaluation.sum, sample.evaluation.pdf));
        values.write(10u,
                     make_float4(sample.evaluation.diffuse,
                                 sample.evaluation.average_roughness_squared));
        values.write(11u, make_float4(sample.wo, pick_b.random.z));
        values.write(12u,
                     make_float4(sample_evaluation.sum, sample_evaluation.pdf));
        values.write(13u, make_float4(sampled_common.weight,
                                      sampled_common.sample_weight));

        write_pick(meta, 0u, pool, pick_a);
        write_pick(meta, 4u, pool, pick_b);
        write_pick(meta, 8u, pool, pick_bssrdf);
        meta.write(12u, sample.label);
        meta.write(13u, select(0u, 1u, sample.evaluation.pdf != 0.0f));
        meta.write(14u, pick_b.index);
        meta.write(15u, sampled_common.type);

        device_svm::ClosurePool single_pool{1u};
        const auto single = single_pool.allocate(closure::type_diffuse,
                                                 make_float3(0.7f, 0.5f, 0.3f));
        single_pool.set_sample_weight(single.index, 0.4f);
        single_pool.set_normal(single.index, make_float3(0.0f, 0.0f, 1.0f));
        auto single_shader_data = make_shader_data(&single_pool);
        const auto single_pick = detail::surface_shader_bsdf_bssrdf_pick(
            single_shader_data, make_float3(0.3f, 0.7f, 0.625f));
        const auto single_sample = detail::surface_shader_bsdf_sample_closure(
            kernel_globals, single_shader_data, single_pick, closure_mask);
        values.write(14u, make_float4(single_pick.random.z, 0.0f, 0.0f, 0.0f));
        values.write(15u, make_float4(single_sample.evaluation.sum,
                                      single_sample.evaluation.pdf));
        values.write(
            16u,
            make_float4(single_sample.evaluation.diffuse,
                        single_sample.evaluation.average_roughness_squared));
        write_pick(meta, 16u, single_pool, single_pick);

        device_svm::ClosurePool prefixed_pool{2u};
        const auto holdout = prefixed_pool.allocate(35u, make_float3(1.0f));
        prefixed_pool.set_sample_weight(holdout.index, 0.0f);
        const auto prefixed_diffuse =
            prefixed_pool.allocate(closure::type_diffuse, make_float3(0.5f));
        prefixed_pool.set_sample_weight(prefixed_diffuse.index, 0.8f);
        prefixed_pool.set_normal(prefixed_diffuse.index,
                                 make_float3(0.0f, 0.0f, 1.0f));
        auto prefixed_shader_data = make_shader_data(&prefixed_pool);
        const auto prefixed_pick = detail::surface_shader_bsdf_bssrdf_pick(
            prefixed_shader_data, make_float3(0.1f, 0.2f, 0.3f));
        values.write(17u,
                     make_float4(prefixed_pick.random.z, 0.0f, 0.0f, 0.0f));
        meta.write(20u, prefixed_pick.index);
        meta.write(21u, prefixed_pool.common(prefixed_pick.index).type);
      }};
}

[[nodiscard]] auto predicate_kernel() {
  return Kernel1D<Buffer<std::uint32_t>>{[](BufferUInt result) noexcept {
    const UInt type = dispatch_id().x;
    UInt bits = 0u;
    bits |= select(0u, 1u << 0u, closure::is_bsdf(type));
    bits |= select(0u, 1u << 1u, closure::is_bsdf_diffuse(type));
    bits |= select(0u, 1u << 2u, closure::is_bsdf_glossy(type));
    bits |= select(0u, 1u << 3u, closure::is_bsdf_transmission(type));
    bits |= select(0u, 1u << 4u, closure::is_glass(type));
    bits |= select(0u, 1u << 5u, closure::is_bsdf_or_bssrdf(type));
    bits |= select(0u, 1u << 6u, closure::is_bssrdf(type));
    result.write(type, bits);
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

[[nodiscard]] std::uint32_t expected_predicates(std::uint32_t type) noexcept {
  const auto bsdf = type != 0u && type <= 30u;
  const auto diffuse = type >= 2u && type <= 9u;
  const auto glossy =
      (type >= 12u && type <= 19u) || type == 27u || type == 28u;
  const auto transmission = type >= 20u && type <= 23u;
  const auto glass = type >= 24u && type <= 26u;
  const auto bsdf_or_bssrdf = type != 0u && type <= 34u;
  const auto bssrdf = type >= 31u && type <= 34u;
  return (static_cast<std::uint32_t>(bsdf) << 0u) |
         (static_cast<std::uint32_t>(diffuse) << 1u) |
         (static_cast<std::uint32_t>(glossy) << 2u) |
         (static_cast<std::uint32_t>(transmission) << 3u) |
         (static_cast<std::uint32_t>(glass) << 4u) |
         (static_cast<std::uint32_t>(bsdf_or_bssrdf) << 5u) |
         (static_cast<std::uint32_t>(bssrdf) << 6u);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto values = device.create_buffer<luisa::float4>(value_count);
  auto meta = device.create_buffer<std::uint32_t>(meta_count);
  auto predicates = device.create_buffer<std::uint32_t>(predicate_count);
  std::array<luisa::float4, value_count> actual_values{};
  std::array<std::uint32_t, meta_count> actual_meta{};
  std::array<std::uint32_t, predicate_count> actual_predicates{};
  const auto surface_shader =
      device.compile(surface_kernel(), ShaderOption{.enable_cache = false,
                                                    .enable_fast_math = true});
  const auto predicate_shader = device.compile(
      predicate_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = true});
  stream << surface_shader(values, meta).dispatch(1u)
         << predicate_shader(predicates).dispatch(predicate_count)
         << values.copy_to(actual_values.data())
         << meta.copy_to(actual_meta.data())
         << predicates.copy_to(actual_predicates.data()) << synchronize();

  const auto cosine = 1.0f / std::sqrt(1.13f);
  const auto closure_pdf = cosine / std::numbers::pi_v<float>;
  const auto mixture_pdf = closure_pdf / 1.5f;
  const auto weighted_sum = luisa::float3{0.5f, 0.6f, 0.7f} * closure_pdf;

  auto valid = true;
  valid &= near(actual_values[0u], luisa::float4{weighted_sum.x, weighted_sum.y,
                                                 weighted_sum.z, mixture_pdf});
  valid &= near(actual_values[1u], luisa::float4{weighted_sum.x, weighted_sum.y,
                                                 weighted_sum.z, 1.0f});
  valid &= near(actual_values[2u], luisa::float4{0.0f});
  valid &= near(actual_values[3u], luisa::float4{weighted_sum.x, weighted_sum.y,
                                                 weighted_sum.z, 0.0f});
  valid &=
      near(actual_values[4u], luisa::float4{0.0f, 0.0f, 0.0f, mixture_pdf});
  valid &= near(actual_values[5u], luisa::float4{0.6f, 0.0f, 0.0f, 0.0f});
  valid &=
      near(actual_values[6u], luisa::float4{2.0f / 3.0f, 0.0f, 0.0f, 0.0f});
  valid &= near(actual_values[7u], luisa::float4{0.7f, 0.0f, 0.0f, 0.0f});
  valid &= near(actual_values[8u], luisa::float4{1.2f, 1.5f, 1.8f, 0.0f});
  valid &= near(actual_values[9u], actual_values[12u]);
  valid &= near(actual_values[10u].x, actual_values[12u].x) &&
           near(actual_values[10u].y, actual_values[12u].y) &&
           near(actual_values[10u].z, actual_values[12u].z) &&
           near(actual_values[10u].w, 1.0f);
  valid &= near(actual_values[11u].w, 2.0f / 3.0f);
  valid &= near(actual_values[13u], luisa::float4{0.3f, 0.2f, 0.1f, 0.75f});
  valid &= near(actual_values[14u], luisa::float4{0.625f, 0.0f, 0.0f, 0.0f});
  valid &= near(actual_values[15u].w, actual_values[16u].x / 0.7f);
  valid &= near(actual_values[16u].w, 1.0f);
  valid &= near(actual_values[17u], luisa::float4{0.3f, 0.0f, 0.0f, 0.0f});

  constexpr std::array expected_pick_meta{0u, closure::type_diffuse,
                                          1u, 0u,
                                          1u, closure::type_diffuse,
                                          1u, 0u,
                                          2u, closure::type_bssrdf_random_walk,
                                          1u, 1u};
  for (auto index = std::size_t{0u}; index < expected_pick_meta.size();
       ++index) {
    valid &= actual_meta[index] == expected_pick_meta[index];
  }
  valid &=
      actual_meta[12u] == (closure::label_reflect | closure::label_diffuse);
  valid &= actual_meta[13u] == 1u;
  valid &= actual_meta[14u] == 1u;
  valid &= actual_meta[15u] == closure::type_diffuse;
  valid &= actual_meta[16u] == 0u;
  valid &= actual_meta[17u] == closure::type_diffuse;
  valid &= actual_meta[18u] == 1u;
  valid &= actual_meta[19u] == 0u;
  valid &= actual_meta[20u] == 1u;
  valid &= actual_meta[21u] == closure::type_diffuse;

  for (auto type = std::uint32_t{0u}; type < predicate_count; ++type) {
    valid &= actual_predicates[type] == expected_predicates(type);
  }
  if (!valid) {
    std::cerr << "Cycles SVM surface shader mismatch on " << backend << '\n';
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

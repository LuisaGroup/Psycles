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
 * `principled_sheen_svm_oracle`. Only its global shader jump is relocated;
 * the typed Principled payload is word-identical to the external dump. */
constexpr std::array<std::uint32_t, 57u> principled_sheen_words{
    0x00000001u, 0x00000004u, 0x00000037u, 0x00000038u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000002u, 0x0000002bu, 0x000000ffu,
    0x0000001au, 0x3f800000u, 0x3ed70a3du, 0x3f19999au, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x3e9eb852u, 0x3f11eb85u,
    0x3f547ae1u, 0x3f4ccccdu, 0x00000000u, 0x0000ff00u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x3e4ccccdu, 0x3dcccccdu, 0x3f19999au, 0x3fa66666u, 0x3e800000u,
    0x3f333333u, 0x3ee66666u, 0x3ebd70a4u, 0x3f800000u, 0x3f800000u,
    0x3f800000u, 0x3cf5c28fu, 0x3fc00000u, 0x00000020u, 0x3f800000u,
    0x3e4ccccdu, 0x3dcccccdu, 0x3ba3d70au, 0x3fb33333u, 0x00000000u,
    0x00000000u, 0x3faa3d71u, 0x00000000u, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u};

class TableKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;
  Bool _valid;

public:
  TableKernelGlobals(const BufferFloat &table, Expr<bool> valid) noexcept
      : _table{&table}, _valid{valid} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return select(0.0f, _table->read(index), _valid);
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

inline constexpr std::uint32_t output_stride = 9u;
inline constexpr std::uint32_t meta_stride = 8u;

[[nodiscard]] auto bsdf_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [used](BufferUInt words, BufferFloat table, BufferFloat4 output,
             BufferUInt meta) noexcept {
        const auto case_index = dispatch_x();
        const TableKernelGlobals kernel_globals{table, case_index == 0u};
        device_svm::ClosurePool closures{8u};
        auto shader_data = make_shader_data(&closures);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                               device_svm::kernel_feature_node_bsdf |
                                   device_svm::kernel_feature_node_emission,
                               used, identity_transform_state(), shader_data,
                               path_state, result);

        const auto closure0 = closures.common(0u);
        const auto sheen = closures.sheen(1u);
        const auto closure2 = closures.common(2u);
        const auto output_base = case_index * output_stride;
        output.write(output_base + 0u,
                     make_float4(closure0.weight, closure0.sample_weight));
        output.write(output_base + 1u, make_float4(sheen.common.weight,
                                                   sheen.common.sample_weight));
        output.write(output_base + 2u,
                     make_float4(closure2.weight, closure2.sample_weight));
        output.write(output_base + 3u, make_float4(closure0.N, 0.0f));
        output.write(output_base + 4u, make_float4(sheen.common.N, 0.0f));
        output.write(output_base + 5u,
                     make_float4(sheen.param.roughness, sheen.param.transform_a,
                                 sheen.param.transform_b, 0.0f));
        output.write(output_base + 6u, make_float4(sheen.param.T, 0.0f));
        output.write(output_base + 7u, make_float4(sheen.param.B, 0.0f));
        output.write(output_base + 8u,
                     make_float4(shader_data.closure_emission_background,
                                 shader_data.closure_transparent_extinction.x));

        const auto meta_base = case_index * meta_stride;
        meta.write(meta_base + 0u, closures.count());
        meta.write(meta_base + 1u, closures.left());
        meta.write(meta_base + 2u, shader_data.flag);
        meta.write(meta_base + 3u, result.status);
        meta.write(meta_base + 4u, result.final_offset);
        meta.write(meta_base + 5u, closure0.type);
        meta.write(meta_base + 6u, sheen.common.type);
        meta.write(meta_base + 7u, closure2.type);
      }};
}

[[nodiscard]] auto emission_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[used](BufferUInt words,
                                                BufferFloat table,
                                                BufferFloat4 output,
                                                BufferUInt meta) noexcept {
    const TableKernelGlobals kernel_globals{table, true};
    device_svm::ClosurePool closures{8u};
    auto shader_data = make_shader_data(&closures);
    const device_svm::PathState path_state{
        device_svm::path_ray_visibility_camera, device_svm::path_ray_emission};
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
      device.create_buffer<std::uint32_t>(principled_sheen_words.size());
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
         << words.copy_from(principled_sheen_words.data())
         << bsdf(words, table, output, meta).dispatch(2u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << emission(words, table, emission_output, emission_meta).dispatch(1u)
         << emission_output.copy_to(actual_emission.data())
         << emission_meta.copy_to(actual_emission_meta.data()) << synchronize();

  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  constexpr auto expected_flags =
      device_svm::shader_data_use_bump_map_correction |
      device_svm::shader_data_emission | device_svm::shader_data_bsdf |
      device_svm::shader_data_bsdf_has_eval |
      device_svm::shader_data_transparent;
  const auto basis_t = luisa::float3{0.7071067812f, -0.7071067812f, 0.0f};
  const auto basis_b = luisa::float3{0.7071067812f, 0.7071067812f, 0.0f};
  const auto valid_base = 0u;
  const auto invalid_base = output_stride;
  const auto valid_meta = 0u;
  const auto invalid_meta = meta_stride;
  auto valid =
      near(actual[valid_base + 0u].xyz(), luisa::float3{0.1999999881f}) &&
      near(actual[valid_base + 0u].w, 0.1999999881f) &&
      near(actual[valid_base + 1u].xyz(),
           luisa::float3{0.00036242406f, 0.00101478747f, 0.00065236329f}) &&
      near(actual[valid_base + 1u].w, 0.00067652494f) &&
      near(actual[valid_base + 2u].xyz(),
           luisa::float3{0.247685403f, 0.455421537f, 0.663157701f}) &&
      near(actual[valid_base + 2u].w, 0.455421537f) &&
      near(actual[valid_base + 3u].xyz(), up) &&
      near(actual[valid_base + 4u].xyz(), up) &&
      near(actual[valid_base + 5u].x, 0.37f) &&
      std::abs(actual[valid_base + 5u].y) >= 1.0e-5f &&
      std::isfinite(actual[valid_base + 5u].z) &&
      near(actual[valid_base + 6u].xyz(), basis_t) &&
      near(actual[valid_base + 7u].xyz(), basis_b) &&
      near(actual[valid_base + 8u].xyz(),
           luisa::float3{0.207736135f, 0.103868067f, 0.623208463f}) &&
      near(actual[valid_base + 8u].w, 0.1999999881f) &&
      actual_meta[valid_meta + 0u] == 3u &&
      actual_meta[valid_meta + 1u] == 5u &&
      actual_meta[valid_meta + 2u] == expected_flags &&
      actual_meta[valid_meta + 3u] ==
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
      actual_meta[valid_meta + 4u] == 55u &&
      actual_meta[valid_meta + 5u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID) &&
      actual_meta[valid_meta + 6u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID) &&
      actual_meta[valid_meta + 7u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);

  /* Source edge oracle: an invalid LTC consumes the already allocated slot,
   * retains its input weight and typed setup payload, and clears only its
   * sample weight. Lower layers must remain unattenuated. */
  valid &= near(actual[invalid_base + 1u].xyz(),
                luisa::float3{0.12f, 0.336f, 0.216f}) &&
           near(actual[invalid_base + 1u].w, 0.0f) &&
           near(actual[invalid_base + 2u].xyz(),
                luisa::float3{0.248f, 0.456f, 0.664f}) &&
           near(actual[invalid_base + 2u].w, 0.456f) &&
           near(actual[invalid_base + 5u].x, 0.37f) &&
           near(actual[invalid_base + 5u].y, 0.0f) &&
           near(actual[invalid_base + 5u].z, 0.0f) &&
           near(actual[invalid_base + 6u].xyz(), basis_t) &&
           near(actual[invalid_base + 7u].xyz(), basis_b) &&
           near(actual[invalid_base + 8u].xyz(),
                luisa::float3{0.208f, 0.104f, 0.624f}) &&
           actual_meta[invalid_meta + 0u] == 3u &&
           actual_meta[invalid_meta + 1u] == 5u &&
           actual_meta[invalid_meta + 2u] == expected_flags &&
           actual_meta[invalid_meta + 6u] ==
               static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           actual_meta[invalid_meta + 7u] ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID);

  /* PATH_RAY_EMISSION uses Cycles' stack-local SheenBsdf: transparency is
   * still an ordinary closure, while Sheen attenuates emission without
   * consuming a pool slot and the BSDF-only lower layers are not evaluated. */
  valid &=
      near(actual_emission[0u].xyz(),
           luisa::float3{0.207736135f, 0.103868067f, 0.623208463f}) &&
      near(actual_emission[0u].w, 0.1999999881f) &&
      near(actual_emission[1u].xyz(), luisa::float3{0.1999999881f}) &&
      near(actual_emission[1u].w, 0.1999999881f) &&
      actual_emission_meta[0u] == 1u && actual_emission_meta[1u] == 7u &&
      actual_emission_meta[2u] == expected_flags &&
      actual_emission_meta[3u] ==
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
      actual_emission_meta[4u] == 55u &&
      actual_emission_meta[5u] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_TRANSPARENT_ID);

  if (!valid) {
    std::cerr << "Cycles Principled Sheen transition mismatch on " << backend
              << "\nvalid sheen=(" << actual[valid_base + 1u].x << ", "
              << actual[valid_base + 1u].y << ", " << actual[valid_base + 1u].z
              << ", " << actual[valid_base + 1u].w << "), diffuse=("
              << actual[valid_base + 2u].x << ", " << actual[valid_base + 2u].y
              << ", " << actual[valid_base + 2u].z << ", "
              << actual[valid_base + 2u].w << "), meta=(" << actual_meta[0u]
              << ", " << actual_meta[1u] << ", " << actual_meta[2u] << ", "
              << actual_meta[3u] << ", " << actual_meta[4u] << ", "
              << actual_meta[5u] << ", " << actual_meta[6u] << ", "
              << actual_meta[7u] << ")\n";
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

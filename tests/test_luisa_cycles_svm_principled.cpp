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

/* Complete compact surface image from the unmodified Cycles 5.2.1
 * `principled_svm_oracle` default material. Only the global shader jump was
 * relocated to this compact buffer; the 176-byte typed Principled payload is
 * word-identical to the external dump whose SHA-256 is documented beside the
 * closure trace. */
constexpr std::array<std::uint32_t, 57u> principled_default_words{
    0x00000001u, 0x00000004u, 0x00000037u, 0x00000038u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000002u, 0x0000002bu, 0x000000ffu,
    0x0000001au, 0x3fc00000u, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x3f4ccccdu, 0x3f4ccccdu,
    0x3f4ccccdu, 0x3f800000u, 0x00000000u, 0x0000ff00u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x3f800000u, 0x3f800000u, 0x3f800000u, 0x00000000u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x3f800000u, 0x3f800000u,
    0x3f800000u, 0x3cf5c28fu, 0x3fc00000u, 0x00000020u, 0x3f800000u,
    0x3e4ccccdu, 0x3dcccccdu, 0x3ba3d70au, 0x3fb33333u, 0x00000000u,
    0x00000000u, 0x3faa3d71u, 0x00000000u, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u};

class TableKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;

public:
  explicit TableKernelGlobals(const BufferFloat &table) noexcept
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
make_shader_data(device_svm::ClosurePool *closure) noexcept {
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
          closure};
}

[[nodiscard]] std::array<bool, NODE_NUM> node_types() noexcept {
  std::array<bool, NODE_NUM> result{};
  result[NODE_END] = true;
  result[NODE_SHADER_JUMP] = true;
  result[NODE_GEOMETRY] = true;
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

[[nodiscard]] auto principled_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{[used](BufferUInt words,
                                                BufferFloat table,
                                                BufferFloat4 output,
                                                BufferUInt meta) noexcept {
    const TableKernelGlobals kernel_globals{table};
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
    const auto closure1 = closures.common(1u);
    const auto microfacet = closures.microfacet(0u);
    output.write(0u, make_float4(closure0.weight, closure0.sample_weight));
    output.write(1u, make_float4(closure0.N, 0.0f));
    output.write(2u, make_float4(closure1.weight, closure1.sample_weight));
    output.write(3u, make_float4(closure1.N, 0.0f));
    output.write(4u, make_float4(microfacet.param.alpha_x,
                                 microfacet.param.alpha_y, microfacet.param.ior,
                                 microfacet.param.energy_scale));
    output.write(5u, make_float4(microfacet.param.T, 0.0f));
    output.write(6u,
                 make_float4(microfacet.generalized_schlick.thin_film.thickness,
                             microfacet.generalized_schlick.thin_film.ior,
                             microfacet.generalized_schlick.exponent, 0.0f));
    output.write(
        7u, make_float4(microfacet.generalized_schlick.reflection_tint, 0.0f));
    output.write(
        8u,
        make_float4(microfacet.generalized_schlick.transmission_tint, 0.0f));
    output.write(9u, make_float4(microfacet.generalized_schlick.f0, 0.0f));
    output.write(10u, make_float4(microfacet.generalized_schlick.f90, 0.0f));
    output.write(11u,
                 make_float4(shader_data.closure_transparent_extinction, 0.0f));
    output.write(12u,
                 make_float4(shader_data.closure_emission_background, 0.0f));

    meta.write(0u, closures.count());
    meta.write(1u, closures.left());
    meta.write(2u, shader_data.flag);
    meta.write(3u, result.status);
    meta.write(4u, result.final_offset);
    meta.write(5u, closure0.type);
    meta.write(6u, closure1.type);
    meta.write(7u, microfacet.param.fresnel_type);
  }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 4.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 4.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, int argc, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table_values =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values(
          psycles::contract::ShaderColorSpace{});
  auto table = device.create_buffer<float>(table_values.size());
  auto words =
      device.create_buffer<std::uint32_t>(principled_default_words.size());
  auto output = device.create_buffer<luisa::float4>(13u);
  auto meta = device.create_buffer<std::uint32_t>(8u);
  auto shader = device.compile(
      principled_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});

  std::array<luisa::float4, 13u> actual{};
  std::array<std::uint32_t, 8u> actual_meta{};
  stream << table.copy_from(luisa::span{table_values})
         << words.copy_from(principled_default_words.data())
         << shader(words, table, output, meta).dispatch(1u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  const auto zero = luisa::float3{0.0f};
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  auto valid =
      near(actual[0].xyz(), luisa::float3{0.9220424294471741f}) &&
      near(actual[0].w, 0.03738487884402275f) && near(actual[1].xyz(), up) &&
      near(actual[2].xyz(), luisa::float3{0.7700921297073364f}) &&
      near(actual[2].w, 0.7700921893119812f) && near(actual[3].xyz(), up) &&
      near(actual[4].x, 0.25f) && near(actual[4].y, 0.25f) &&
      near(actual[4].z, 1.5f) && actual[4].w > 1.0f &&
      near(actual[5].xyz(), zero) &&
      near(actual[6].xyz(), luisa::float3{0.0f, 0.0f, -1.5f}) &&
      near(actual[7].xyz(), luisa::float3{1.0f}) &&
      near(actual[8].xyz(), zero) &&
      near(actual[9].xyz(), luisa::float3{0.04f}) &&
      near(actual[10].xyz(), luisa::float3{1.0f}) &&
      near(actual[11].xyz(), zero) && near(actual[12].xyz(), zero) &&
      actual_meta[0] == 2u && actual_meta[1] == 5u &&
      actual_meta[2] == (device_svm::shader_data_use_bump_map_correction |
                         device_svm::shader_data_bsdf |
                         device_svm::shader_data_bsdf_has_eval) &&
      actual_meta[3] ==
          static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended) &&
      actual_meta[4] == 55u &&
      actual_meta[5] ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID) &&
      actual_meta[6] == static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_ID) &&
      actual_meta[7] == static_cast<std::uint32_t>(
                            device_svm::MicrofacetFresnel::generalized_schlick);

  if (!valid) {
    std::cerr << "Cycles default Principled closure oracle mismatch on "
              << backend << "\nclosure0=(" << actual[0].x << ", " << actual[0].y
              << ", " << actual[0].z << ", " << actual[0].w << "), closure1=("
              << actual[2].x << ", " << actual[2].y << ", " << actual[2].z
              << ", " << actual[2].w << ")\n"
              << "microfacet=(" << actual[4].x << ", " << actual[4].y << ", "
              << actual[4].z << ", " << actual[4].w << "), meta=("
              << actual_meta[0] << ", " << actual_meta[1] << ", "
              << actual_meta[2] << ", " << actual_meta[3] << ", "
              << actual_meta[4] << ", " << actual_meta[5] << ", "
              << actual_meta[6] << ", " << actual_meta[7] << ")\n";
  }
  static_cast<void>(argc);
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

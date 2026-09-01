#include <psycles/luisa/cycles_svm.h>

#include "luisa_cycles_svm_test_kernel_globals.h"

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

/* These tails are copied word-for-word from the final global SVM buffers of
 * unmodified Cycles 5.2.1 probe scenes. Only the four-word shader jump is
 * relocated to address the compact test buffer; the surface words remain
 * byte-identical. Volume and displacement each point at a terminal END. */
constexpr std::array<std::uint32_t, 22u> diffuse_surface_words{
    0x00000001u, 0x00000004u, 0x00000014u, 0x00000015u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3f2e147bu, 0x3e75c28fu,
    0x3db851ecu, 0x00000002u, 0x00000002u, 0x000000ffu, 0x3f2e147bu,
    0x3e75c28fu, 0x3db851ecu, 0x3edc28f6u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 19u> translucent_surface_words{
    0x00000001u, 0x00000004u, 0x00000011u, 0x00000012u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3f3ae148u, 0x3e8f5c29u,
    0x3de147aeu, 0x00000002u, 0x00000009u, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 25u> transparent_mix_words{
    0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u, 0x00000008u,
    0x3f1eb852u, 0x000100ffu, 0x00000005u, 0x3f400000u, 0x3f666666u,
    0x3f19999au, 0x00000002u, 0x0000001eu, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000005u, 0x3f828f5du, 0x3dc49ba6u, 0x3d1374bdu,
    0x00000003u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u};

/* Source-derived state-machine regression for
 * bsdf_transparent_setup's unique-closure merge transition. Each individual
 * record uses the externally frozen Transparent payload above; the pair is a
 * legal Add Shader ordering and must still consume one closure slot. */
constexpr std::array<std::uint32_t, 25u> transparent_merge_words{
    0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u, 0x00000005u,
    0x3e4ccccdu, 0x3e99999au, 0x3ecccccdu, 0x00000002u, 0x0000001eu,
    0x000000ffu, 0x00000000u, 0x00000000u, 0x00000005u, 0x3d4ccccdu,
    0x3d8f5c29u, 0x3de147aeu, 0x00000002u, 0x0000001eu, 0x000000ffu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr auto sphere_normal = luisa::float3{
    0.02134036459028721f, 0.021340366452932358f, 0.9995445013046265f};
constexpr auto sphere_geometric_normal = luisa::float3{
    0.03299136832356453f, 0.03640035167336464f, 0.9987925887107849f};

struct ExpectedClosure {
  std::string_view name;
  luisa::float3 normal;
  luisa::float3 geometric_normal;
  luisa::float3 weight;
  float sample_weight;
  std::uint32_t type;
  std::uint32_t flag;
  std::uint32_t final_offset;
  luisa::float3 transparent_extinction;
  luisa::float3 emission;
  bool oren_nayar;
};

constexpr ExpectedClosure diffuse_expected{
    .name = "Diffuse Probe",
    .normal = sphere_normal,
    .geometric_normal = sphere_geometric_normal,
    .weight = {0.6800000071525574f, 0.23999999463558197f, 0.09000000357627869f},
    .sample_weight = 0.33666667342185974f,
    .type = 3u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 20u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = true};

constexpr ExpectedClosure translucent_expected{
    .name = "Translucent Probe",
    .normal = sphere_normal,
    .geometric_normal = sphere_geometric_normal,
    .weight = {0.7300000190734863f, 0.2800000011920929f, 0.10999999940395355f},
    .sample_weight = 0.3733333349227905f,
    .type = 9u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval |
            device_svm::shader_data_bsdf_has_transmission,
    .final_offset = 17u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false};

constexpr ExpectedClosure transparent_expected{
    .name = "Transparent Probe",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.2849999964237213f, 0.34199997782707214f, 0.2280000001192093f},
    .sample_weight = 0.2849999964237213f,
    .type = 30u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_emission | device_svm::shader_data_bsdf |
            device_svm::shader_data_transparent,
    .final_offset = 23u,
    .transparent_extinction = {0.2849999964237213f, 0.34199997782707214f,
                               0.2280000001192093f},
    .emission = {0.6324000954627991f, 0.05952000245451927f,
                 0.02232000231742859f},
    .oren_nayar = false};

constexpr ExpectedClosure transparent_merge_expected{
    .name = "Transparent merge state machine",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.25f, 0.37f, 0.51f},
    .sample_weight = 0.3766666650772095f,
    .type = 30u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf | device_svm::shader_data_transparent,
    .final_offset = 23u,
    .transparent_extinction = {0.25f, 0.37f, 0.51f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false};

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

[[nodiscard]] bool near(luisa::float2 actual, luisa::float2 expected,
                        float tolerance = 4.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance);
}

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(Expr<luisa::float3> normal,
                 Expr<luisa::float3> geometric_normal,
                 device_svm::ClosurePool *closure) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          normal,
          geometric_normal,
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

[[nodiscard]] std::array<bool, NODE_NUM> closure_node_types() noexcept {
  std::array<bool, NODE_NUM> result{};
  result[NODE_END] = true;
  result[NODE_SHADER_JUMP] = true;
  result[NODE_GEOMETRY] = true;
  result[NODE_CLOSURE_SET_WEIGHT] = true;
  result[NODE_CLOSURE_WEIGHT] = true;
  result[NODE_CLOSURE_EMISSION] = true;
  result[NODE_MIX_CLOSURE] = true;
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

[[nodiscard]] auto closure_kernel(std::array<bool, NODE_NUM> node_types_used) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::float4>, Buffer<std::uint32_t>>{
      [node_types_used](BufferUInt words, BufferFloat4 state,
                        BufferFloat4 output, BufferUInt meta) noexcept {
        const psycles::test_support::DefaultCyclesSvmKernelGlobals
            kernel_globals;
        device_svm::ClosurePool closures{8u};
        const auto normal = state.read(0u).xyz();
        const auto geometric_normal = state.read(1u).xyz();
        auto shader_data =
            make_shader_data(normal, geometric_normal, &closures);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                               device_svm::kernel_feature_node_bsdf |
                                   device_svm::kernel_feature_node_emission,
                               node_types_used, identity_transform_state(),
                               shader_data, path_state, result);

        Float3 weight = make_float3(0.0f);
        Float sample_weight = 0.0f;
        UInt type = static_cast<std::uint32_t>(CLOSURE_NONE_ID);
        Float3 closure_normal = make_float3(0.0f);
        Float roughness = 0.0f;
        Float oren_a = 0.0f;
        Float oren_b = 0.0f;
        Float3 multiscatter = make_float3(0.0f);
        $if(closures.count() != 0u) {
          const auto common = closures.common(0u);
          weight = common.weight;
          sample_weight = common.sample_weight;
          type = common.type;
          closure_normal = common.N;
          $if(common.type ==
              static_cast<std::uint32_t>(CLOSURE_BSDF_OREN_NAYAR_ID)) {
            const auto oren = closures.oren_nayar(0u);
            roughness = oren.param.roughness;
            oren_a = oren.param.a;
            oren_b = oren.param.b;
            multiscatter = oren.param.multiscatter_term;
          };
        };

        output.write(0u, make_float4(weight, sample_weight));
        output.write(1u, make_float4(closure_normal, roughness));
        output.write(
            2u, make_float4(oren_a, oren_b, multiscatter.x, multiscatter.y));
        output.write(3u,
                     make_float4(multiscatter.z,
                                 shader_data.closure_transparent_extinction));
        output.write(
            4u, make_float4(shader_data.closure_emission_background, 0.0f));
        meta.write(0u, type);
        meta.write(1u, closures.count());
        meta.write(2u, closures.left());
        meta.write(3u, shader_data.flag);
        meta.write(4u, result.status);
        meta.write(5u, result.final_offset);
      }};
}

template <std::size_t word_count>
[[nodiscard]] bool run_oracle(
    Device &device, Stream &stream, std::string_view backend,
    const std::array<std::uint32_t, word_count> &word_image,
    const ExpectedClosure &expected,
    const Shader1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                   Buffer<luisa::float4>, Buffer<std::uint32_t>> &shader) {
  auto words = device.create_buffer<std::uint32_t>(word_count);
  auto state = device.create_buffer<luisa::float4>(2u);
  auto output = device.create_buffer<luisa::float4>(5u);
  auto meta = device.create_buffer<std::uint32_t>(6u);
  const std::array state_data{
      luisa::float4{expected.normal.x, expected.normal.y, expected.normal.z,
                    0.0f},
      luisa::float4{expected.geometric_normal.x, expected.geometric_normal.y,
                    expected.geometric_normal.z, 0.0f}};
  std::array<luisa::float4, 5u> actual{};
  std::array<std::uint32_t, 6u> actual_meta{};
  stream << words.copy_from(word_image.data())
         << state.copy_from(state_data.data())
         << shader(words, state, output, meta).dispatch(1u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  auto valid = near(actual[0].xyz(), expected.weight) &&
               near(actual[0].w, expected.sample_weight) &&
               near(actual[1].xyz(), expected.normal) &&
               near(actual[3].yzw(), expected.transparent_extinction) &&
               near(actual[4].xyz(), expected.emission) &&
               actual_meta[0] == expected.type && actual_meta[1] == 1u &&
               actual_meta[2] == 7u && actual_meta[3] == expected.flag &&
               actual_meta[4] == static_cast<std::uint32_t>(
                                     device_svm::EvaluationStatus::ended) &&
               actual_meta[5] == expected.final_offset;
  if (expected.oren_nayar) {
    valid &= near(actual[1].w, 0.43f) &&
             near(actual[2].x, 0.28325653076171875f) &&
             near(actual[2].y, 0.12180031090974808f) &&
             near(actual[2].zw(),
                  luisa::float2{0.19124409556388855f, 0.022941801697015762f}) &&
             near(actual[3].x, 0.0031860244926065207f);
  }

  if (!valid) {
    std::cerr << "Cycles closure oracle mismatch for " << expected.name
              << " on " << backend << "\nweight/sample=(" << actual[0].x << ", "
              << actual[0].y << ", " << actual[0].z << ", " << actual[0].w
              << "), normal/roughness=(" << actual[1].x << ", " << actual[1].y
              << ", " << actual[1].z << ", " << actual[1].w << "), meta=("
              << actual_meta[0] << ", " << actual_meta[1] << ", "
              << actual_meta[2] << ", " << actual_meta[3] << ", "
              << actual_meta[4] << ", " << actual_meta[5] << ")\n";
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto shader = device.compile(
      closure_kernel(closure_node_types()),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  return run_oracle(device, stream, backend, diffuse_surface_words,
                    diffuse_expected, shader) &&
                 run_oracle(device, stream, backend, translucent_surface_words,
                            translucent_expected, shader) &&
                 run_oracle(device, stream, backend, transparent_mix_words,
                            transparent_expected, shader) &&
                 run_oracle(device, stream, backend, transparent_merge_words,
                            transparent_merge_expected, shader)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_bsdf.h"
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

/* Exact surface tail of `Glass Transport 02` from the unmodified Cycles
 * 5.2.1 diagnostic dump
 * d84f339e9d25276cd8086105c47353e85f8187dea0535aac0bb4cbea7da33c5e.
 * The source global jump (123,142,143) is relocated to (4,23,24); all node
 * payload words are unchanged. */
constexpr std::array<std::uint32_t, 25u> glass_beckmann_words{
    0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000005u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
    0x00000002u, 0x00000018u, 0x000000ffu,
    0x3f800000u, 0x3f800000u, 0x3f800000u,
    0x3e315cacu, 0x3fc00000u, 0x00000000u,
    0x3faa3d71u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 24u> glossy_ggx_words{
    0x00000001u, 0x00000004u, 0x00000016u, 0x00000017u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000005u, 0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu,
    0x00000002u, 0x0000000cu, 0x000000ffu,
    0x00000000u, 0x00000000u, 0x00000000u,
    0x3ecccccdu, 0x00000000u, 0x00000000u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 24u> glossy_ashikhmin_shirley_words{
    0x00000001u, 0x00000004u, 0x00000016u, 0x00000017u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000005u, 0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu,
    0x00000002u, 0x0000000fu, 0x000000ffu,
    0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

/* Exact surface tail from the external Cycles 5.2.1 anisotropic-default-
 * Tangent oracle. Geometry.Tangent is written to stack offset 3; the Glossy
 * payload consumes it with Anisotropy=0.5 and Rotation=0.25. */
constexpr std::array<std::uint32_t, 27u>
    glossy_anisotropic_default_tangent_words{
        0x00000001u, 0x00000004u, 0x00000019u, 0x0000001au,
        0x0000000bu, 0x00000001u, 0x00000000u,
        0x0000000bu, 0x03000002u, 0x00000000u,
        0x00000005u, 0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu,
        0x00000002u, 0x0000000cu, 0x000000ffu,
        0x00000000u, 0x00000000u, 0x00000000u,
        0x3ecccccdu, 0x3f000000u, 0x3e800000u, 0x00000300u,
        0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 27u> glossy_beckmann_words{
    0x00000001u, 0x00000004u, 0x00000019u, 0x0000001au,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000008u, 0x3f4f5c29u, 0x0003ffffu,
    0x00000005u, 0x3f333333u, 0x3e99999au, 0x3dcccccdu,
    0x00000002u, 0x0000000du, 0x00000003u,
    0x00000000u, 0x00000000u, 0x00000000u,
    0x3e4ccccdu, 0x00000000u, 0x00000000u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 27u> glossy_multi_ggx_words{
    0x00000001u, 0x00000004u, 0x00000019u, 0x0000001au,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000008u, 0x3f4f5c29u, 0x0003ffffu,
    0x00000005u, 0x3f333333u, 0x3e99999au, 0x3dcccccdu,
    0x00000002u, 0x0000000eu, 0x00000003u,
    0x3f333333u, 0x3e99999au, 0x3dcccccdu,
    0x3f333333u, 0x00000000u, 0x00000000u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 20u> refraction_beckmann_words{
    0x00000001u, 0x00000004u, 0x00000012u, 0x00000013u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000005u, 0x3ec3b96bu, 0x3f26ba28u, 0x3f5e35b5u,
    0x00000002u, 0x00000014u, 0x000000ffu,
    0x3e0c6480u, 0x3f947ae1u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 20u> refraction_ggx_words{
    0x00000001u, 0x00000004u, 0x00000012u, 0x00000013u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000005u, 0x3ec3b96bu, 0x3f26ba28u, 0x3f5e35b5u,
    0x00000002u, 0x00000015u, 0x000000ffu,
    0x3e0c6480u, 0x3f947ae1u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 20u>
    refraction_beckmann_backface_words{
        0x00000001u, 0x00000004u, 0x00000012u, 0x00000013u,
        0x0000000bu, 0x00000001u, 0x00000000u,
        0x00000005u, 0x3f23d70au, 0x3db851ecu, 0x3efae148u,
        0x00000002u, 0x00000014u, 0x000000ffu,
        0x3ee66666u, 0x3fc00000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 26u> metallic_f82_ggx_words{
    0x00000001u, 0x00000004u, 0x00000018u, 0x00000019u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000002u, 0x0000000bu, 0x000000ffu,
    0x0000000cu, 0x3f3851ecu, 0x3e0f5c29u, 0x3d0f5c29u,
    0x3f6b851fu, 0x3ed70a3du, 0x3df5c28fu,
    0x3e3851ecu, 0x00000000u, 0x00000000u, 0x00000000u,
    0x3faa3d71u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 26u> metallic_f82_beckmann_words{
    0x00000001u, 0x00000004u, 0x00000018u, 0x00000019u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000002u, 0x0000000bu, 0x000000ffu,
    0x0000000du, 0x3e23d70au, 0x3f1eb852u, 0x3f51eb85u,
    0x3f3d70a4u, 0x3f5c28f6u, 0x3f7ae148u,
    0x3eb33333u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x3faa3d71u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 29u>
    metallic_f82_multi_anisotropic_words{
        0x00000001u, 0x00000004u, 0x0000001bu, 0x0000001cu,
        0x0000000bu, 0x00000001u, 0x00000000u,
        0x0000000bu, 0x03000002u, 0x00000000u,
        0x00000002u, 0x0000000bu, 0x000000ffu,
        0x0000000eu, 0x3f1eb852u, 0x3e6147aeu, 0x3d75c28fu,
        0x3f75c28fu, 0x3ef5c28fu, 0x3e3851ecu,
        0x3eeb851fu, 0x3f0ccccdu, 0x3e3851ecu,
        0x43d20000u, 0x3fc28f5cu, 0x00000300u,
        0x00000000u, 0x00000000u, 0x00000000u};

/* Exact global surface tail for `Metallic BSDF Matrix 06` from the
 * unmodified Cycles 5.2.1 linked-scene dump
 * ca569e6e0fdf31b04109689ef8b3fb70a6655d464348f595de334cc66d637cfd.
 * The source jump (316,343,344) is relocated to (4,31,32); every surface
 * payload word is unchanged. Unlike the literal oracle above, this retains
 * the two linked Value Vector nodes that feed Normal and Tangent. */
constexpr std::array<std::uint32_t, 33u>
    metallic_f82_multi_linked_anisotropic_words{
        0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
        0x00000013u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x3f800000u, 0x00000013u, 0x00000003u, 0x3f19999au,
        0x3f4ccccdu, 0x00000000u, 0x00000002u, 0x0000000bu,
        0x000000ffu, 0x0000000eu, 0x3e3851ecu, 0x3f0a3d71u,
        0x3f6147aeu, 0x3f23d70au, 0x3f666666u, 0x3f7d70a4u,
        0x3f2147aeu, 0x3f666666u, 0x3e000000u, 0x00000000u,
        0x3faa3d71u, 0x00000300u, 0x00000000u, 0x00000000u,
        0x00000000u};

constexpr std::array<std::uint32_t, 26u> metallic_conductor_ggx_words{
    0x00000001u, 0x00000004u, 0x00000018u, 0x00000019u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000002u, 0x0000000au, 0x000000ffu,
    0x0000000cu, 0x3e8a3d71u, 0x3f2e147bu, 0x3fa8f5c3u,
    0x40670a3du, 0x4027ae14u, 0x3ff47ae1u,
    0x3e3851ecu, 0x00000000u, 0x00000000u, 0x00000000u,
    0x3faa3d71u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 26u> metallic_conductor_beckmann_words{
    0x00000001u, 0x00000004u, 0x00000018u, 0x00000019u,
    0x0000000bu, 0x00000001u, 0x00000000u,
    0x00000002u, 0x0000000au, 0x000000ffu,
    0x0000000du, 0x3fb9999au, 0x3f570a3du, 0x3ec28f5cu,
    0x3ff5c28fu, 0x40251eb8u, 0x405ae148u,
    0x3eb33333u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x3faa3d71u, 0x0000ff00u,
    0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 29u>
    metallic_conductor_multi_anisotropic_words{
        0x00000001u, 0x00000004u, 0x0000001bu, 0x0000001cu,
        0x0000000bu, 0x00000001u, 0x00000000u,
        0x0000000bu, 0x03000002u, 0x00000000u,
        0x00000002u, 0x0000000au, 0x000000ffu,
        0x0000000eu, 0x3ed70a3du, 0x3f6147aeu, 0x3fd1eb85u,
        0x406e147bu, 0x40228f5cu, 0x3fae147bu,
        0x3eeb851fu, 0x3f0ccccdu, 0x3e3851ecu,
        0x44188000u, 0x3fd9999au, 0x00000300u,
        0x00000000u, 0x00000000u, 0x00000000u};

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
  std::uint32_t initial_flag{device_svm::shader_data_use_bump_map_correction};
  std::uint32_t count{1u};
  std::uint32_t left{7u};
  bool microfacet{false};
  bool generalized_schlick{false};
  bool fresnel_payload{false};
  luisa::float4 alpha_ior_energy{};
  luisa::float3 tangent{};
  std::uint32_t fresnel_type{};
  luisa::float4 thin_film_exponent{};
  luisa::float3 reflection_tint{};
  luisa::float3 transmission_tint{};
  luisa::float3 f0{};
  luisa::float3 f90{};
  bool direct_evaluation{};
  luisa::float3 direct_value{};
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

constexpr ExpectedClosure glass_beckmann_expected{
    .name = "Glass Transport 02",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {1.0f, 1.0f, 1.0f},
    .sample_weight = 1.0f,
    .type = 24u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval |
            device_svm::shader_data_bsdf_has_transmission,
    .final_offset = 23u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .generalized_schlick = true,
    .alpha_ior_energy = {0.03f, 0.03f, 1.5f, 1.0f},
    .tangent = {0.0f, 0.0f, 0.0f},
    .fresnel_type = 4u,
    .thin_film_exponent = {0.0f, 1.33f, -1.5f, 0.0f},
    .reflection_tint = {1.0f, 1.0f, 1.0f},
    .transmission_tint = {1.0f, 1.0f, 1.0f},
    .f0 = {0.04f, 0.04f, 0.04f},
    .f90 = {1.0f, 1.0f, 1.0f}};

constexpr ExpectedClosure glossy_ggx_expected{
    .name = "Glossy BSDF Matrix 00",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.6800000071525574f, 0.23999999463558197f,
               0.09000000357627869f},
    .sample_weight = 0.33666667342185974f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 22u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .microfacet = true,
    .alpha_ior_energy = {0.1600000113248825f, 0.1600000113248825f,
                         1.0f, 1.0f}};

constexpr ExpectedClosure glossy_ashikhmin_shirley_expected{
    .name = "Glossy Ashikhmin-Shirley zero roughness",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.6800000071525574f, 0.23999999463558197f,
               0.09000000357627869f},
    .sample_weight = 0.33666667342185974f,
    .type = 15u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 22u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .microfacet = true,
    .alpha_ior_energy = {0.00009999999747378752f,
                         0.00009999999747378752f, 1.0f, 1.0f}};

constexpr ExpectedClosure glossy_anisotropic_default_tangent_expected{
    .name = "Glossy anisotropic default Tangent",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.6800000071525574f, 0.23999999463558197f,
               0.09000000357627869f},
    .sample_weight = 0.33666667342185974f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 25u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .microfacet = true,
    .alpha_ior_energy = {0.08000000566244125f, 0.320000022649765f,
                         1.0f, 1.0f},
    .tangent = {0.0f, 1.0f, 0.0f}};

constexpr ExpectedClosure glossy_beckmann_expected{
    .name = "Glossy BSDF Matrix 08",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.5669999718666077f, 0.24300001561641693f,
               0.08100000023841858f},
    .sample_weight = 0.2970000207424164f,
    .type = 13u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 25u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .microfacet = true,
    .alpha_ior_energy = {0.04000000283122063f, 0.04000000283122063f,
                         1.0f, 1.0f}};

constexpr ExpectedClosure glossy_multi_ggx_expected{
    .name = "Glossy BSDF Matrix 09",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.5023012757301331f, 0.18660412728786469f,
               0.05828767269849777f},
    .sample_weight = 0.23496782779693604f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 25u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .microfacet = true,
    .alpha_ior_energy = {0.4899999797344208f, 0.4899999797344208f,
                         1.0f, 1.4335616827011108f}};

constexpr ExpectedClosure refraction_beckmann_expected{
    .name = "Refraction BSDF Matrix 00",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.38227400183677673f, 0.651278018951416f,
               0.8680070042610168f},
    .sample_weight = 0.6338530778884888f,
    .type = 20u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval |
            device_svm::shader_data_bsdf_has_transmission,
    .final_offset = 18u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .microfacet = true,
    .alpha_ior_energy = {0.018796993419528008f,
                         0.018796993419528008f,
                         1.159999966621399f, 1.0f}};

constexpr ExpectedClosure refraction_ggx_expected{
    .name = "Refraction BSDF Matrix 01",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.38227400183677673f, 0.651278018951416f,
               0.8680070042610168f},
    .sample_weight = 0.6338530778884888f,
    .type = 21u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval |
            device_svm::shader_data_bsdf_has_transmission,
    .final_offset = 18u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .microfacet = true,
    .alpha_ior_energy = {0.018796993419528008f,
                         0.018796993419528008f,
                         1.159999966621399f, 1.0f}};

constexpr ExpectedClosure refraction_beckmann_backface_expected{
    .name = "Refraction BSDF Matrix 07 backface",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.6399999856948853f, 0.09000000357627869f,
               0.49000000953674316f},
    .sample_weight = 0.40666669607162476f,
    .type = 20u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_backfacing |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval |
            device_svm::shader_data_bsdf_has_transmission,
    .final_offset = 18u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .initial_flag = device_svm::shader_data_use_bump_map_correction |
                    device_svm::shader_data_backfacing,
    .microfacet = true,
    .alpha_ior_energy = {0.20249998569488525f, 0.20249998569488525f,
                         0.6666666865348816f, 1.0f}};

constexpr ExpectedClosure metallic_f82_ggx_expected{
    .name = "Metallic BSDF Matrix 00 F82 GGX",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {1.0f, 1.0f, 1.0f},
    .sample_weight = 0.29833468794822693f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 24u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .fresnel_payload = true,
    .alpha_ior_energy = {0.03240000084042549f, 0.03240000084042549f,
                         1.0f, 1.0f},
    .fresnel_type = 5u,
    .thin_film_exponent = {0.000009999999747378752f, 1.3300000429153442f,
                           0.0f, 0.0f},
    .reflection_tint = {0.7200000286102295f, 0.14000000059604645f,
                        0.03500000014901161f},
    .transmission_tint = {1.1996527910232544f, 5.506825923919678f,
                          7.478795051574707f}};

constexpr ExpectedClosure metallic_f82_beckmann_expected{
    .name = "Metallic BSDF Matrix 01 F82 Beckmann",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {1.0f, 1.0f, 1.0f},
    .sample_weight = 0.5333424210548401f,
    .type = 13u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 24u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .fresnel_payload = true,
    .alpha_ior_energy = {0.1224999949336052f, 0.1224999949336052f,
                         1.0f, 1.0f},
    .fresnel_type = 5u,
    .thin_film_exponent = {0.000009999999747378752f, 1.3300000429153442f,
                           0.0f, 0.0f},
    .reflection_tint = {0.1599999964237213f, 0.6200000047683716f,
                        0.8199999928474426f},
    .transmission_tint = {2.517897367477417f, 1.9666064977645874f,
                          0.31888237595558167f}};

constexpr ExpectedClosure metallic_f82_multi_anisotropic_expected{
    .name = "Metallic BSDF Matrix 07 F82 Multi-GGX",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.9768685698509216f, 0.9523739218711853f,
               0.9431397318840027f},
    .sample_weight = 0.23154392838478088f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 27u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .fresnel_payload = true,
    .alpha_ior_energy = {0.2977624833583832f, 0.1503700613975525f,
                         1.0f, 1.06322f},
    .tangent = {0.42577922344207764f, 0.9048271179199219f, 0.0f},
    .fresnel_type = 5u,
    .thin_film_exponent = {420.0f, 1.5199999809265137f, 0.0f, 0.0f},
    .reflection_tint = {0.6200000047683716f, 0.2199999988079071f,
                        0.05999999865889549f},
    .transmission_tint = {0.5618879199028015f, 5.3317179679870605f,
                          7.163314342498779f}};

constexpr ExpectedClosure metallic_f82_multi_linked_anisotropic_expected{
    .name = "Metallic BSDF Matrix 06 linked Multi-GGX",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.8217415809631348f, 0.8920326232910156f,
               0.9695853590965271f},
    .sample_weight = 0.4770972430706024f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 31u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .fresnel_payload = true,
    .alpha_ior_energy = {0.9105510711669922f, 0.17300468683242798f,
                         1.0f, 1.2672f},
    .tangent = {-0.1414213627576828f, 0.9899494647979736f, 0.0f},
    .fresnel_type = 5u,
    .thin_film_exponent = {0.000009999999747378752f, 1.3300000429153442f,
                           0.0f, 0.0f},
    .reflection_tint = {0.18000000715255737f, 0.5400000214576721f,
                        0.8799999952316284f},
    .transmission_tint = {3.55461f, 1.32884f, 0.165132f},
    .direct_evaluation = true,
    .direct_value = {0.0021440336f, 0.0064319385f, 0.010481611f}};

constexpr ExpectedClosure metallic_conductor_ggx_expected{
    .name = "Metallic BSDF Matrix 08 physical GGX",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {1.0f, 1.0f, 1.0f},
    .sample_weight = 0.6869250535964966f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 24u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .fresnel_payload = true,
    .alpha_ior_energy = {0.03240000084042549f, 0.03240000084042549f,
                         1.0f, 1.0f},
    .fresnel_type = 3u,
    .thin_film_exponent = {0.000009999999747378752f, 1.3300000429153442f,
                           0.0f, 0.0f},
    .reflection_tint = {0.27000001072883606f, 0.6800000071525574f,
                        1.3200000524520874f},
    .transmission_tint = {3.609999895095825f, 2.619999885559082f,
                          1.909999966621399f}};

constexpr ExpectedClosure metallic_conductor_beckmann_expected{
    .name = "Metallic BSDF Matrix 09 physical Beckmann",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {1.0f, 1.0f, 1.0f},
    .sample_weight = 0.651674747467041f,
    .type = 13u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 24u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .fresnel_payload = true,
    .alpha_ior_energy = {0.1224999949336052f, 0.1224999949336052f,
                         1.0f, 1.0f},
    .fresnel_type = 3u,
    .thin_film_exponent = {0.000009999999747378752f, 1.3300000429153442f,
                           0.0f, 0.0f},
    .reflection_tint = {1.4500000476837158f, 0.8399999737739563f,
                        0.3799999952316284f},
    .transmission_tint = {1.9199999570846558f, 2.5799999237060547f,
                          3.4200000762939453f}};

constexpr ExpectedClosure metallic_conductor_multi_anisotropic_expected{
    .name = "Metallic BSDF Matrix 15 physical Multi-GGX",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.9931033253669739f, 0.9784969091415405f,
               0.9566794633865356f},
    .sample_weight = 0.528117299079895f,
    .type = 12u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 27u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 1u,
    .left = 6u,
    .microfacet = true,
    .fresnel_payload = true,
    .alpha_ior_energy = {0.2977624833583832f, 0.1503700613975525f,
                         1.0f, 1.06322f},
    .tangent = {0.42577922344207764f, 0.9048271179199219f, 0.0f},
    .fresnel_type = 3u,
    .thin_film_exponent = {610.0f, 1.7000000476837158f, 0.0f, 0.0f},
    .reflection_tint = {0.41999998688697815f, 0.8799999952316284f,
                        1.6399999856948853f},
    .transmission_tint = {3.7200000286102295f, 2.5399999618530273f,
                          1.3600000143051147f}};

constexpr ExpectedClosure metallic_extra_rollback_expected{
    .name = "Metallic extra allocation rollback",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.0f, 0.0f, 0.0f},
    .sample_weight = 0.0f,
    .type = 0u,
    .flag = device_svm::shader_data_use_bump_map_correction |
            device_svm::shader_data_bsdf |
            device_svm::shader_data_bsdf_has_eval,
    .final_offset = 24u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 0u,
    .left = 1u};

/* closure_alloc_extra must remove the immediately preceding ordinary
 * closure when its one-slot Fresnel payload does not fit. This source-derived
 * state transition is observable independently of any rendered color. */
constexpr ExpectedClosure glass_extra_rollback_expected{
    .name = "Glass extra allocation rollback",
    .normal = {0.0f, 0.0f, 1.0f},
    .geometric_normal = {0.0f, 0.0f, 1.0f},
    .weight = {0.0f, 0.0f, 0.0f},
    .sample_weight = 0.0f,
    .type = 0u,
    .flag = device_svm::shader_data_use_bump_map_correction,
    .final_offset = 23u,
    .transparent_extinction = {0.0f, 0.0f, 0.0f},
    .emission = {0.0f, 0.0f, 0.0f},
    .oren_nayar = false,
    .count = 0u,
    .left = 1u};

class TableKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;

public:
  explicit TableKernelGlobals(const BufferFloat &table) noexcept
      : _table{&table} {}

  [[nodiscard]] Float cycles_bsdf_data(
      Expr<std::uint32_t> index) const noexcept override {
    return _table->read(index);
  }
};

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

[[nodiscard]] bool near(luisa::float4 actual, luisa::float4 expected,
                        float tolerance = 4.0e-5f) noexcept {
  return near(actual.xyz(), expected.xyz(), tolerance) &&
         near(actual.w, expected.w, tolerance);
}

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(Expr<luisa::float3> normal,
                 Expr<luisa::float3> geometric_normal,
                 Expr<std::uint32_t> initial_flag,
                 device_svm::ClosurePool *closure) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          normal,
          geometric_normal,
          make_float3(0.0f, 0.0f, 1.0f),
          device_svm::primitive_triangle,
          0u,
          initial_flag,
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
  result[NODE_VALUE_V] = true;
  result[NODE_CLOSURE_SET_WEIGHT] = true;
  result[NODE_CLOSURE_WEIGHT] = true;
  result[NODE_CLOSURE_EMISSION] = true;
  result[NODE_MIX_CLOSURE] = true;
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

[[nodiscard]] auto closure_kernel(std::array<bool, NODE_NUM> node_types_used,
                                  std::size_t closure_capacity) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>,
                  Buffer<luisa::float4>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>>{
      [node_types_used, closure_capacity](
          BufferUInt words, BufferFloat table, BufferFloat4 state,
          BufferFloat4 output, BufferUInt meta) noexcept {
        const TableKernelGlobals kernel_globals{table};
        device_svm::ClosurePool closures{closure_capacity};
        const auto normal = state.read(0u).xyz();
        const auto geometric_normal = state.read(1u).xyz();
        const auto initial_flag = cast<luisa::uint>(state.read(0u).w);
        auto shader_data = make_shader_data(
            normal, geometric_normal, initial_flag, &closures);
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
        Float4 alpha_ior_energy = make_float4(0.0f);
        Float3 tangent = make_float3(0.0f);
        UInt fresnel_type = 0u;
        Float4 thin_film_exponent = make_float4(0.0f);
        Float3 reflection_tint = make_float3(0.0f);
        Float3 transmission_tint = make_float3(0.0f);
        Float3 f0 = make_float3(0.0f);
        Float3 f90 = make_float3(0.0f);
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
          const Bool is_microfacet =
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_GGX_ID)) |
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_BECKMANN_ID)) |
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID)) |
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID)) |
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID)) |
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)) |
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID));
          $if(is_microfacet) {
            const auto microfacet = closures.microfacet_param(0u);
            alpha_ior_energy =
                make_float4(microfacet.alpha_x, microfacet.alpha_y,
                            microfacet.ior, microfacet.energy_scale);
            tangent = microfacet.T;
            fresnel_type = microfacet.fresnel_type;
          };
          $if((common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)) |
              (common.type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID))) {
            const auto microfacet = closures.microfacet(0u);
            thin_film_exponent = make_float4(
                microfacet.generalized_schlick.thin_film.thickness,
                microfacet.generalized_schlick.thin_film.ior,
                microfacet.generalized_schlick.exponent, 0.0f);
            reflection_tint =
                microfacet.generalized_schlick.reflection_tint;
            transmission_tint =
                microfacet.generalized_schlick.transmission_tint;
            f0 = microfacet.generalized_schlick.f0;
            f90 = microfacet.generalized_schlick.f90;
          };
          $if(fresnel_type == static_cast<std::uint32_t>(
                                  device_svm::MicrofacetFresnel::conductor)) {
            const auto microfacet = closures.microfacet_conductor(0u);
            thin_film_exponent = make_float4(
                microfacet.conductor.thin_film.thickness,
                microfacet.conductor.thin_film.ior, 0.0f, 0.0f);
            reflection_tint = microfacet.conductor.ior;
            transmission_tint = microfacet.conductor.extinction;
          }
          $elif(fresnel_type == static_cast<std::uint32_t>(
                                     device_svm::MicrofacetFresnel::f82_tint)) {
            const auto microfacet = closures.microfacet_f82_tint(0u);
            thin_film_exponent = make_float4(
                microfacet.f82_tint.thin_film.thickness,
                microfacet.f82_tint.thin_film.ior, 0.0f, 0.0f);
            reflection_tint = microfacet.f82_tint.f0;
            transmission_tint = microfacet.f82_tint.b;
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
        output.write(5u, alpha_ior_energy);
        output.write(6u, make_float4(tangent, 0.0f));
        output.write(7u, thin_film_exponent);
        output.write(8u, make_float4(reflection_tint, 0.0f));
        output.write(9u, make_float4(transmission_tint, 0.0f));
        output.write(10u, make_float4(f0, 0.0f));
        output.write(11u, make_float4(f90, 0.0f));
        const auto direct = device_svm::detail::bsdf_eval(
            kernel_globals, shader_data, 0u,
            make_float3(0.7956017255783081f, 0.0f,
                        0.6058201789855957f),
            device_svm::detail::all_closure_types);
        output.write(12u, make_float4(direct.value, direct.pdf));
        meta.write(0u, type);
        meta.write(1u, closures.count());
        meta.write(2u, closures.left());
        meta.write(3u, shader_data.flag);
        meta.write(4u, result.status);
        meta.write(5u, result.final_offset);
        meta.write(6u, fresnel_type);
      }};
}

template <std::size_t word_count>
[[nodiscard]] bool run_oracle(
    Device &device, Stream &stream, std::string_view backend,
    Buffer<float> &table,
    const std::array<std::uint32_t, word_count> &word_image,
    const ExpectedClosure &expected,
    const Shader1D<Buffer<std::uint32_t>, Buffer<float>,
                   Buffer<luisa::float4>, Buffer<luisa::float4>,
                   Buffer<std::uint32_t>> &shader) {
  auto words = device.create_buffer<std::uint32_t>(word_count);
  auto state = device.create_buffer<luisa::float4>(2u);
  auto output = device.create_buffer<luisa::float4>(13u);
  auto meta = device.create_buffer<std::uint32_t>(7u);
  const std::array state_data{
      luisa::float4{expected.normal.x, expected.normal.y, expected.normal.z,
                    static_cast<float>(expected.initial_flag)},
      luisa::float4{expected.geometric_normal.x, expected.geometric_normal.y,
                    expected.geometric_normal.z, 0.0f}};
  std::array<luisa::float4, 13u> actual{};
  std::array<std::uint32_t, 7u> actual_meta{};
  stream << words.copy_from(word_image.data())
         << state.copy_from(state_data.data())
         << shader(words, table, state, output, meta).dispatch(1u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  const auto expected_closure_normal =
      expected.count == 0u ? luisa::float3{0.0f} : expected.normal;
  auto valid = near(actual[0].xyz(), expected.weight) &&
               near(actual[0].w, expected.sample_weight) &&
               near(actual[1].xyz(), expected_closure_normal) &&
               near(actual[3].yzw(), expected.transparent_extinction) &&
               near(actual[4].xyz(), expected.emission) &&
               actual_meta[0] == expected.type &&
               actual_meta[1] == expected.count &&
               actual_meta[2] == expected.left &&
               actual_meta[3] == expected.flag &&
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
  if (expected.microfacet) {
    valid &= near(actual[5], expected.alpha_ior_energy) &&
             near(actual[6].xyz(), expected.tangent) &&
             actual_meta[6] == expected.fresnel_type;
  }
  if (expected.generalized_schlick) {
    valid &= near(actual[7], expected.thin_film_exponent) &&
             near(actual[8].xyz(), expected.reflection_tint) &&
             near(actual[9].xyz(), expected.transmission_tint) &&
             near(actual[10].xyz(), expected.f0) &&
             near(actual[11].xyz(), expected.f90);
  }
  if (expected.fresnel_payload) {
    valid &= near(actual[7], expected.thin_film_exponent) &&
             near(actual[8].xyz(), expected.reflection_tint) &&
             near(actual[9].xyz(), expected.transmission_tint);
  }
  if (expected.direct_evaluation) {
    valid &= near(actual[12].xyz(), expected.direct_value);
  }

  if (!valid) {
    std::cerr << "Cycles closure oracle mismatch for " << expected.name
              << " on " << backend << "\nweight/sample=(" << actual[0].x << ", "
              << actual[0].y << ", " << actual[0].z << ", " << actual[0].w
              << "), normal/roughness=(" << actual[1].x << ", " << actual[1].y
              << ", " << actual[1].z << ", " << actual[1].w << "), meta=("
              << actual_meta[0] << ", " << actual_meta[1] << ", "
              << actual_meta[2] << ", " << actual_meta[3] << ", "
              << actual_meta[4] << ", " << actual_meta[5] << ", "
              << actual_meta[6] << ")\n";
    if (expected.microfacet) {
      std::cerr << "microfacet=(" << actual[5].x << ", " << actual[5].y
                << ", " << actual[5].z << ", " << actual[5].w
                << "), tangent=(" << actual[6].x << ", " << actual[6].y
                << ", " << actual[6].z << "), film=(" << actual[7].x
                << ", " << actual[7].y << ", " << actual[7].z
                << "), payload0=(" << actual[8].x << ", " << actual[8].y
                << ", " << actual[8].z << "), payload1=(" << actual[9].x
                << ", " << actual[9].y << ", " << actual[9].z << ")\n";
    }
    if (expected.direct_evaluation) {
      std::cerr << "direct=(" << actual[12].x << ", " << actual[12].y
                << ", " << actual[12].z << ", pdf=" << actual[12].w
                << ")\n";
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table_values =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values(
          psycles::contract::ShaderColorSpace{});
  auto table = device.create_buffer<float>(table_values.size());
  stream << table.copy_from(luisa::span{table_values}) << synchronize();
  auto shader = device.compile(
      closure_kernel(closure_node_types(), 8u),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  auto rollback_shader = device.compile(
      closure_kernel(closure_node_types(), 1u),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  return run_oracle(device, stream, backend, table, diffuse_surface_words,
                    diffuse_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            translucent_surface_words, translucent_expected,
                            shader) &&
                 run_oracle(device, stream, backend, table,
                            transparent_mix_words, transparent_expected,
                            shader) &&
                 run_oracle(device, stream, backend, table,
                            transparent_merge_words,
                            transparent_merge_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            glass_beckmann_words, glass_beckmann_expected,
                            shader) &&
                 run_oracle(device, stream, backend, table, glossy_ggx_words,
                            glossy_ggx_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            glossy_ashikhmin_shirley_words,
                            glossy_ashikhmin_shirley_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            glossy_anisotropic_default_tangent_words,
                            glossy_anisotropic_default_tangent_expected,
                            shader) &&
                 run_oracle(device, stream, backend, table,
                            glossy_beckmann_words,
                            glossy_beckmann_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            glossy_multi_ggx_words,
                            glossy_multi_ggx_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            refraction_beckmann_words,
                            refraction_beckmann_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            refraction_ggx_words,
                            refraction_ggx_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            refraction_beckmann_backface_words,
                            refraction_beckmann_backface_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_f82_ggx_words,
                            metallic_f82_ggx_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_f82_beckmann_words,
                            metallic_f82_beckmann_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_f82_multi_anisotropic_words,
                            metallic_f82_multi_anisotropic_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_f82_multi_linked_anisotropic_words,
                            metallic_f82_multi_linked_anisotropic_expected,
                            shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_conductor_ggx_words,
                            metallic_conductor_ggx_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_conductor_beckmann_words,
                            metallic_conductor_beckmann_expected, shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_conductor_multi_anisotropic_words,
                            metallic_conductor_multi_anisotropic_expected,
                            shader) &&
                 run_oracle(device, stream, backend, table,
                            metallic_f82_ggx_words,
                            metallic_extra_rollback_expected,
                            rollback_shader) &&
                 run_oracle(device, stream, backend, table,
                            glass_beckmann_words,
                            glass_extra_rollback_expected, rollback_shader)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}

#pragma once

#include <psycles/compiler/cycles_svm_osl_binding.h>
#include <psycles/luisa/cycles_svm.h>

#include <array>
#include <cstdint>

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::cycles_svm::detail {

inline constexpr std::uint32_t max_osl_closures = 8u;

enum class OslClosureTag : std::uint32_t {
  none = 0u,
  diffuse = 1u,
  oren_nayar = 2u,
  translucent = 3u,
  transparent = 4u,
  emission = 5u,
  background = 6u,
  holdout = 7u,
  microfacet_ggx = 8u,
  microfacet_beckmann = 9u,
  microfacet_ggx_refraction = 10u,
  microfacet_beckmann_refraction = 11u,
  microfacet_ggx_glass = 12u,
  microfacet_beckmann_glass = 13u,
  microfacet_multi_ggx = 14u,
  microfacet_multi_ggx_glass = 15u,
  ashikhmin_shirley = 16u,
  ashikhmin_velvet = 17u,
  sheen = 18u,
  diffuse_toon = 19u,
  glossy_toon = 20u,
  reflection = 21u,
  refraction = 22u,
  dielectric_bsdf = 23u,
  conductor_bsdf = 24u,
  generalized_schlick_bsdf = 25u,
};

struct OslEvalGlobals {
  luisa::float3 P{};
  luisa::float3 dPdx{};
  luisa::float3 dPdy{};
  luisa::float3 I{};
  luisa::float3 dIdx{};
  luisa::float3 dIdy{};
  luisa::float3 N{};
  luisa::float3 Ng{};
  luisa::float3 dPdu{};
  luisa::float3 dPdv{};
  float u{};
  float dudx{};
  float dudy{};
  float v{};
  float dvdx{};
  float dvdy{};
  float time{};
  float dtime{};
  float ray_length{};
  std::uint32_t ray_visibility{};
  std::uint32_t shader_raw{};
  std::uint32_t sd_flag{};
  std::uint32_t prim_type{};
  std::uint32_t bounce{};
  std::uint32_t diffuse_bounce{};
  std::uint32_t glossy_bounce{};
  std::uint32_t transmission_bounce{};
  std::uint32_t transparent_bounce{};
  std::int32_t backfacing{};
  float object_index{};
  float object_random{};
  float object_alpha{};
  float material_index{};
  luisa::float3 object_location{};
  luisa::float3 object_color{};
};

struct OslTransformGlobals {
  luisa::float4x4 object_to_world{};
  luisa::float4x4 world_to_object{};
  luisa::float4x4 camera_to_world{};
  luisa::float4x4 world_to_camera{};
};

struct OslAttrInput {
  luisa::float4 val{};
  luisa::float4 dx{};
  luisa::float4 dy{};
  std::int32_t found{};
};

struct OslMutableState {
  luisa::float3 N{};
  std::uint32_t wrote_N{};
};

struct OslClosureEntry {
  std::uint32_t tag{};
  luisa::float3 weight{};
  luisa::float3 N{};
  luisa::float3 T{};
  luisa::float3 color_arg{};
  luisa::float3 color_arg2{};
  float p0{};
  float p1{};
  float p2{};
  float p3{};
  float p4{};
};

struct OslClosureList {
  std::uint32_t count{};
  std::array<OslClosureEntry, max_osl_closures> entries{};
};

} // namespace psycles::luisa_backend::cycles_svm::detail

LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::OslEvalGlobals, P, dPdx,
             dPdy, I, dIdx, dIdy, N, Ng, dPdu, dPdv, u, dudx, dudy, v, dvdx,
             dvdy, time, dtime, ray_length, ray_visibility, shader_raw, sd_flag,
             prim_type, bounce, diffuse_bounce, glossy_bounce,
             transmission_bounce, transparent_bounce, backfacing, object_index,
             object_random, object_alpha, material_index, object_location,
             object_color){};

LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::OslTransformGlobals,
             object_to_world, world_to_object, camera_to_world,
             world_to_camera){};

LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::OslAttrInput, val, dx,
             dy, found){};

LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::OslMutableState, N,
             wrote_N){};

LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::OslClosureEntry, tag,
             weight, N, T, color_arg, color_arg2, p0, p1, p2, p3, p4){};

LUISA_STRUCT(psycles::luisa_backend::cycles_svm::detail::OslClosureList, count,
             entries){};

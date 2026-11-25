#include "virtual_texture_preflight.h"

#include "path_kernel_scene_geometry_plan.h"
#include "path_kernel_scene_traversal.h"
#include "path_tracer_camera.h"
#include "path_tracer_cycles_svm_background.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_shadow.h"
#include "path_tracer_internal.h"
#include "virtual_texture_cache.h"
#include "virtual_texture_cache_profile.h"

#include <psycles/compiler/cycles_svm_ir.h>
#include <psycles/luisa/camera_sampling.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/cycles_svm_translator.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>

namespace psycles::luisa_backend::detail {

namespace abi = ::psycles::compiler::cycles_svm;
namespace svm = ::psycles::luisa_backend::cycles_svm;

namespace {

[[nodiscard]] bool is_vt_image_opcode(abi::ShaderNodeType op) noexcept {
  switch (op) {
    case abi::NODE_TEX_IMAGE:
    case abi::NODE_TEX_IMAGE_DERIVATIVE:
    case abi::NODE_TEX_IMAGE_BOX:
    case abi::NODE_TEX_IMAGE_BOX_DERIVATIVE:
    case abi::NODE_TEX_ENVIRONMENT:
    case abi::NODE_TEX_ENVIRONMENT_DERIVATIVE:
    case abi::NODE_TEX_SKY:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool is_pruned_drop_opcode(
    abi::ShaderNodeType op,
    std::uint32_t kernel_features) noexcept {
  switch (op) {
    case abi::NODE_CLOSURE_BSDF:
    case abi::NODE_CLOSURE_EMISSION:
    case abi::NODE_CLOSURE_BACKGROUND:
    case abi::NODE_CLOSURE_SET_WEIGHT:
    case abi::NODE_CLOSURE_WEIGHT:
    case abi::NODE_EMISSION_WEIGHT:
    case abi::NODE_MIX_CLOSURE:
    case abi::NODE_CLOSURE_HOLDOUT:
    case abi::NODE_CLOSURE_VOLUME:
    case abi::NODE_VOLUME_COEFFICIENTS:
    case abi::NODE_PRINCIPLED_VOLUME:
    case abi::NODE_CLOSURE_SET_NORMAL:
    case abi::NODE_SET_BUMP:
    case abi::NODE_ENTER_BUMP_EVAL:
    case abi::NODE_LEAVE_BUMP_EVAL:
    case abi::NODE_SET_DISPLACEMENT:
    case abi::NODE_DISPLACEMENT:
    case abi::NODE_VECTOR_DISPLACEMENT:
    case abi::NODE_NORMAL_MAP:
    case abi::NODE_AMBIENT_OCCLUSION:
    case abi::NODE_BEVEL:
    case abi::NODE_RAYCAST:
    case abi::NODE_AOV_START:
    case abi::NODE_AOV_COLOR:
    case abi::NODE_AOV_VALUE:
    case abi::NODE_RADIAL_TILING:
    case abi::NODE_SCENE_TIME:
    case abi::NODE_NONE:
    case abi::NODE_PAD1:
      return true;
    case abi::NODE_HAIR_INFO:
      return (kernel_features & abi::kernel_feature_hair) == 0u;
    case abi::NODE_POINT_INFO:
      return (kernel_features & abi::kernel_feature_pointcloud) == 0u;
    default:
      return false;
  }
}

inline constexpr std::uint32_t preflight_bounce_diffuse = 0u;
inline constexpr std::uint32_t preflight_bounce_glossy_reflect = 1u;
inline constexpr std::uint32_t preflight_bounce_transparent_pass = 2u;
inline constexpr std::uint32_t preflight_bounce_mixed_glossy_diffuse = 3u;
inline constexpr std::uint32_t preflight_bounce_mode_mask = 0x3u;
inline constexpr std::uint32_t preflight_has_surface_vt_bit = 1u << 4u;

[[nodiscard]] bool flatten_svm_block_for_vt(
    const abi::SvmBlockIr &in_block,
    std::uint32_t kernel_features,
    std::vector<abi::SvmInstructionIr> &out) {
  out.clear();
  std::size_t last_image_end = 0u;
  for (const auto &op : in_block) {
    if (op.is_conditional) {
      std::vector<abi::SvmInstructionIr> child_insts;
      if (flatten_svm_block_for_vt(op.children, kernel_features, child_insts)) {
        out.insert(out.end(), child_insts.begin(), child_insts.end());
        last_image_end = out.size();
      }
      continue;
    }
    const auto opc = op.instruction.opcode;
    if (opc == abi::NODE_END || opc == abi::NODE_SHADER_JUMP ||
        opc == abi::NODE_JUMP_IF_ZERO || opc == abi::NODE_JUMP_IF_ONE ||
        is_pruned_drop_opcode(opc, kernel_features)) {
      continue;
    }
    out.push_back(op.instruction);
    if (is_vt_image_opcode(opc)) {
      last_image_end = out.size();
    }
  }
  if (last_image_end == 0u) {
    out.clear();
    return false;
  }
  out.resize(last_image_end);
  return true;
}

[[nodiscard]] std::uint32_t classify_shader_bounce_mode(
    const abi::ShaderCompileMetadata &meta,
    const abi::KernelShader &kshader) noexcept {
  using CU = abi::ClosureUsage;
  const auto &cu = meta.surface_closure_usage;
  const std::uint64_t cmask = cu.surface_closure_mask;
  const std::uint64_t bmask = cu.svm_bsdf_node_mask;
  const std::uint64_t combined_mask = cmask | bmask;

  const std::uint64_t transparent_mask =
      CU::bit(abi::CLOSURE_BSDF_TRANSPARENT_ID) |
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID) |
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) |
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID) |
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID) |
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID) |
      CU::bit(abi::CLOSURE_BSDF_THIN_GLASS_TRANSMISSION_ID) |
      CU::bit(abi::CLOSURE_BSDF_RAY_PORTAL_ID);
  const bool has_transparent =
      meta.has_surface_transparent ||
      (static_cast<std::uint32_t>(kshader.flags) &
       static_cast<std::uint32_t>(abi::SD_HAS_TRANSPARENT_SHADOW)) != 0u ||
      (combined_mask & transparent_mask) != 0ull ||
      (cu.principled_mask &
       (CU::principled_transmission | CU::principled_alpha)) != 0u;
  if (has_transparent) {
    return preflight_bounce_transparent_pass;
  }

  const std::uint64_t glossy_mask =
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_GGX_ID) |
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_BECKMANN_ID) |
      CU::bit(abi::CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID) |
      CU::bit(abi::CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) |
      CU::bit(abi::CLOSURE_BSDF_PHYSICAL_CONDUCTOR) |
      CU::bit(abi::CLOSURE_BSDF_F82_CONDUCTOR) |
      CU::bit(abi::CLOSURE_BSDF_HAIR_REFLECTION_ID) |
      CU::bit(abi::CLOSURE_BSDF_HAIR_CHIANG_ID) |
      CU::bit(abi::CLOSURE_BSDF_HAIR_HUANG_ID);
  const bool has_glossy =
      (combined_mask & glossy_mask) != 0ull ||
      (cu.principled_mask &
       (CU::principled_metallic | CU::principled_coat |
        CU::principled_single_ggx | CU::principled_multi_ggx)) != 0u;

  const std::uint64_t diffuse_mask =
      CU::bit(abi::CLOSURE_BSDF_DIFFUSE_ID) |
      CU::bit(abi::CLOSURE_BSDF_OREN_NAYAR_ID) |
      CU::bit(abi::CLOSURE_BSDF_ROUGH_TRANSLUCENT_ID) |
      CU::bit(abi::CLOSURE_BSDF_BURLEY_ID) |
      CU::bit(abi::CLOSURE_BSDF_DIFFUSE_RAMP_ID) |
      CU::bit(abi::CLOSURE_BSDF_SHEEN_ID) |
      CU::bit(abi::CLOSURE_BSDF_DIFFUSE_TOON_ID) |
      CU::bit(abi::CLOSURE_BSDF_TRANSLUCENT_ID) |
      CU::bit(abi::CLOSURE_BSSRDF_BURLEY_ID) |
      CU::bit(abi::CLOSURE_BSSRDF_RANDOM_WALK_ID) |
      CU::bit(abi::CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID) |
      CU::bit(abi::CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID);
  const bool principled_diffuse =
      ((cu.svm_bsdf_node_mask & CU::bit(abi::CLOSURE_BSDF_PRINCIPLED_ID)) !=
       0ull) &&
      ((cu.principled_mask & CU::principled_metallic) == 0u);
  const bool has_diffuse =
      meta.has_surface_bssrdf || cu.diffuse_node_mask != 0u ||
      (combined_mask & diffuse_mask) != 0ull || principled_diffuse;

  if (has_glossy && !has_diffuse) {
    return preflight_bounce_glossy_reflect;
  }
  if (has_glossy && has_diffuse) {
    return preflight_bounce_mixed_glossy_diffuse;
  }
  return preflight_bounce_diffuse;
}

struct PreflightSvmBytecode {
  abi::ShaderTableImage table;
  std::vector<std::uint8_t> surface_active_mask;
  std::array<bool, abi::NODE_NUM> surface_nodes_used{};
  std::size_t surface_stack_size{1u};
  abi::NoiseUsage surface_noise_usage{};
  bool has_surface_vt{false};
  std::vector<std::uint8_t> bg_active_mask;
  std::array<bool, abi::NODE_NUM> bg_nodes_used{};
  std::size_t bg_stack_size{1u};
  abi::NoiseUsage bg_noise_usage{};
  bool has_bg_vt{false};
  std::uint32_t shader_count{0u};
  std::uint32_t bg_shader_index{~0u};
  std::vector<std::uint32_t> shader_bounce_modes;
};

[[nodiscard]] PreflightSvmBytecode build_preflight_svm_bytecode(
    const LuisaSceneData &scene) {
  PreflightSvmBytecode result;
  if (!scene.cycles_svm) {
    result.table.words = {static_cast<std::uint32_t>(abi::NODE_END)};
    result.shader_bounce_modes = {preflight_bounce_diffuse};
    return result;
  }
  const auto &comp = scene.cycles_svm->compilation;
  const auto &src_table = comp.table;
  const std::uint32_t kernel_features = scene.cycles_svm->kernel_features;
  if (!src_table.valid || !src_table.ir.valid) {
    result.table.words = {static_cast<std::uint32_t>(abi::NODE_END)};
    result.shader_bounce_modes = {preflight_bounce_diffuse};
    return result;
  }
  const std::uint32_t shader_count = src_table.shader_count;
  if (src_table.ir.shaders.size() != static_cast<std::size_t>(shader_count)) {
    result.table.words = {static_cast<std::uint32_t>(abi::NODE_END)};
    result.shader_bounce_modes = {preflight_bounce_diffuse};
    return result;
  }

  result.shader_count = shader_count;
  result.table.valid = true;
  result.table.peak_stack_usage = src_table.peak_stack_usage;
  result.table.shader_count = shader_count;
  result.table.entry_usage = src_table.entry_usage;

  const auto domain_surface_mask = cycles_svm_domain_shader_mask(
      scene, PathCyclesSvmShaderDomain::surface_only);
  const auto bg_shader = scene.cycles_background_shader_id;
  const std::uint32_t bg_idx =
      (bg_shader != ~0u) ? (bg_shader & svm::shader_mask) : ~0u;
  result.bg_shader_index = bg_idx;

  result.surface_active_mask.assign(shader_count, 0u);
  result.bg_active_mask.assign(shader_count, 0u);
  result.shader_bounce_modes.assign(
      std::max<std::size_t>(1u, shader_count), preflight_bounce_diffuse);

  result.surface_nodes_used[abi::NODE_SHADER_JUMP] = true;
  result.surface_nodes_used[abi::NODE_END] = true;
  result.bg_nodes_used[abi::NODE_SHADER_JUMP] = true;
  result.bg_nodes_used[abi::NODE_END] = true;
  result.table.node_types_used[abi::NODE_SHADER_JUMP] = true;
  result.table.node_types_used[abi::NODE_END] = true;

  constexpr std::uint32_t jump_stride =
      1u + static_cast<std::uint32_t>(sizeof(abi::SVMNodeShaderJump) /
                                      sizeof(std::uint32_t));
  auto &words = result.table.words;
  words.resize(static_cast<std::size_t>(shader_count) * jump_stride);
  const std::uint32_t shared_end_offset = static_cast<std::uint32_t>(words.size());
  words.push_back(static_cast<std::uint32_t>(abi::NODE_END));
  for (std::uint32_t sid = 0u; sid < shader_count; ++sid) {
    const std::size_t jbase = static_cast<std::size_t>(sid) * jump_stride;
    words[jbase + 0u] = static_cast<std::uint32_t>(abi::NODE_SHADER_JUMP);
    words[jbase + 1u] = shared_end_offset;
    words[jbase + 2u] = shared_end_offset;
    words[jbase + 3u] = shared_end_offset;
  }

  std::size_t surf_peak = 0u;
  std::size_t bg_peak = 0u;
  abi::NoiseUsage surf_noise = abi::NoiseUsage::none();
  abi::NoiseUsage bg_noise = abi::NoiseUsage::none();
  const bool has_entry_usage =
      comp.shader_surface_entry_usage.size() == shader_count;
  const bool has_meta = comp.shader_metadata.size() == shader_count;
  const bool has_kshaders = comp.kernel_shaders.size() == shader_count;

  std::vector<abi::SvmInstructionIr> flat_insts;
  for (std::uint32_t sid = 0u; sid < shader_count; ++sid) {
    std::uint32_t bmode = preflight_bounce_diffuse;
    if (has_meta) {
      const abi::KernelShader empty_ks{};
      const auto &ks = has_kshaders ? comp.kernel_shaders[sid] : empty_ks;
      bmode = classify_shader_bounce_mode(comp.shader_metadata[sid], ks);
    }

    const auto &in_routine =
        src_table.ir.shaders[sid].routine_for(abi::SHADER_TYPE_SURFACE);
    const bool has_vt =
        !in_routine.empty() &&
        flatten_svm_block_for_vt(in_routine, kernel_features, flat_insts);
    const bool is_bg = (sid == bg_idx);
    const bool in_surface_domain =
        !is_bg &&
        (domain_surface_mask.empty() ||
         (sid < domain_surface_mask.size() && domain_surface_mask[sid] != 0u));

    if (has_vt && (is_bg || in_surface_domain)) {
      const std::uint32_t entry_offset = static_cast<std::uint32_t>(words.size());
      words[static_cast<std::size_t>(sid) * jump_stride + 1u] = entry_offset;
      for (const auto &inst : flat_insts) {
        const std::size_t p_off = static_cast<std::size_t>(inst.payload_offset);
        const std::size_t p_cnt =
            static_cast<std::size_t>(inst.payload_word_count);
        LUISA_ASSERT(
            p_off + p_cnt <= src_table.words.size(),
            "Preflight SVM instruction payload [{}, +{}) exceeds word table {}.",
            p_off, p_cnt, src_table.words.size());
        words.push_back(static_cast<std::uint32_t>(inst.opcode));
        if (p_cnt > 0u) {
          words.insert(words.end(), src_table.words.begin() + p_off,
                       src_table.words.begin() + p_off + p_cnt);
        }
        const auto idx = static_cast<std::size_t>(inst.opcode);
        if (idx < result.table.node_types_used.size()) {
          result.table.node_types_used[idx] = true;
          if (is_bg) {
            result.bg_nodes_used[idx] = true;
          }
          if (in_surface_domain) {
            result.surface_nodes_used[idx] = true;
          }
        }
      }
      words.push_back(static_cast<std::uint32_t>(abi::NODE_END));

      const auto fallback_stack = std::max<std::uint32_t>(
          1u, src_table.usage_for(abi::SHADER_TYPE_SURFACE).peak_stack_usage);
      const std::size_t st_sz =
          has_entry_usage
              ? std::max<std::size_t>(
                    1u, comp.shader_surface_entry_usage[sid].peak_stack_usage)
              : static_cast<std::size_t>(fallback_stack);
      const auto n_usage =
          has_entry_usage ? comp.shader_surface_entry_usage[sid].noise_usage
                          : abi::NoiseUsage::none();
      if (is_bg) {
        result.bg_active_mask[sid] = 1u;
        result.has_bg_vt = true;
        bg_peak = std::max(bg_peak, st_sz);
        bg_noise.shape_mask |= n_usage.shape_mask;
      }
      if (in_surface_domain) {
        result.surface_active_mask[sid] = 1u;
        result.has_surface_vt = true;
        surf_peak = std::max(surf_peak, st_sz);
        surf_noise.shape_mask |= n_usage.shape_mask;
        bmode |= preflight_has_surface_vt_bit;
      }
    }
    result.shader_bounce_modes[sid] = bmode;
  }

  result.surface_stack_size = std::clamp<std::size_t>(
      std::max<std::size_t>(1u, surf_peak), 1u, static_cast<std::size_t>(SVM_STACK_SIZE));
  result.bg_stack_size = std::clamp<std::size_t>(
      std::max<std::size_t>(1u, bg_peak), 1u, static_cast<std::size_t>(SVM_STACK_SIZE));
  result.surface_noise_usage = surf_noise;
  result.bg_noise_usage = bg_noise;
  return result;
}

using PreflightShader = luisa::compute::Shader2D<
    luisa::compute::Buffer<float>,
    luisa::compute::Buffer<luisa::uint>,
    luisa::compute::Buffer<luisa::uint>,
    RenderKernelParameters,
    luisa::uint,
    luisa::uint,
    luisa::uint,
    luisa::uint>;

struct VtPreflightCacheKey {
  CameraProjection projection{CameraProjection::perspective};
  bool depth_of_field{false};
  std::uint32_t aperture_blades{};
  float aperture_rotation{};
  bool enable_cache{true};
  bool enable_fast_math{false};
  bool assume_no_packed_textures{false};
  bool enable_driver_optimization{false};
  LuisaTextureMipMode mip_mode{LuisaTextureMipMode::mip0};

  [[nodiscard]] bool operator==(
      const VtPreflightCacheKey &other) const noexcept {
    return projection == other.projection &&
           depth_of_field == other.depth_of_field &&
           aperture_blades == other.aperture_blades &&
           aperture_rotation == other.aperture_rotation &&
           enable_cache == other.enable_cache &&
           enable_fast_math == other.enable_fast_math &&
           assume_no_packed_textures == other.assume_no_packed_textures &&
           enable_driver_optimization == other.enable_driver_optimization &&
           mip_mode == other.mip_mode;
  }
};

struct VtPreflightCacheEntry {
  VtPreflightCacheKey key{};
  PreflightShader shader;
};

[[nodiscard]] PreflightShader compile_vt_preflight_shader(
    const std::shared_ptr<LuisaSceneData> &scene,
    const PreflightSvmBytecode &preflight_svm,
    const VtPreflightCacheKey &key,
    const SafeNormalizeCallable &safe_normalize,
    const luisa::compute::ShaderOption &shader_option) {
  using namespace luisa::compute;
  const auto traversal_plan = make_scene_traversal_stage_plan(
      scene->geometries.size(), scene->curve_geometries.size());
  const auto traversal = make_scene_traversal_component(traversal_plan);
  const std::uint32_t node_feature_mask =
      svm::kernel_feature_node_mask_surface &
      ~(svm::kernel_feature_node_raytrace |
        svm::kernel_feature_node_bump |
        svm::kernel_feature_node_bump_state);
  const std::uint32_t bg_node_feature_mask =
      svm::kernel_feature_node_mask_surface_light &
      ~(svm::kernel_feature_node_raytrace |
        svm::kernel_feature_node_bump |
        svm::kernel_feature_node_bump_state);

  const auto surf_usage = preflight_svm.has_surface_vt
                              ? cycles_svm_masked_entry_usage(
                                    *scene, preflight_svm.surface_active_mask)
                              : abi::ShaderEntryUsage{};
  const auto bg_usage = preflight_svm.has_bg_vt
                            ? cycles_svm_masked_entry_usage(
                                  *scene, preflight_svm.bg_active_mask)
                            : abi::ShaderEntryUsage{};
  const std::size_t surface_stack_size = std::clamp<std::size_t>(
      std::max<std::size_t>(preflight_svm.surface_stack_size,
                            surf_usage.peak_stack_usage),
      1u, static_cast<std::size_t>(SVM_STACK_SIZE));
  const std::size_t bg_stack_size = std::clamp<std::size_t>(
      std::max<std::size_t>(preflight_svm.bg_stack_size,
                            bg_usage.peak_stack_usage),
      1u, static_cast<std::size_t>(SVM_STACK_SIZE));
  abi::NoiseUsage surface_noise = preflight_svm.surface_noise_usage;
  surface_noise.shape_mask |= surf_usage.noise_usage.shape_mask;
  abi::NoiseUsage bg_noise = preflight_svm.bg_noise_usage;
  bg_noise.shape_mask |= bg_usage.noise_usage.shape_mask;

  const std::uint32_t shader_count_u32 = preflight_svm.shader_count;
  const std::uint32_t last_shader_id =
      shader_count_u32 > 0u ? (shader_count_u32 - 1u) : 0u;
  const std::uint32_t bg_shader_idx = preflight_svm.bg_shader_index;
  constexpr std::uint32_t max_transparent_steps = 6u;

  auto kernel = [&](BufferFloat filter_table,
                    BufferUInt preflight_words,
                    BufferUInt shader_bounce_modes,
                    Var<RenderKernelParameters> parameters,
                    UInt stride,
                    UInt width,
                    UInt height,
                    UInt max_bounces) noexcept {
    set_block_size(8u, 8u, 1u);
    const auto gid = dispatch_id();
    const auto gx = gid.x;
    const auto gy = gid.y;
    const auto half_s = stride >> 1u;
    const auto px = min(gx * stride + half_s, width - 1u);
    const auto py = min(gy * stride + half_s, height - 1u);
    const auto full_x = parameters.window_x + px;
    const auto full_y = parameters.window_y + py;
    const auto cycles_y =
        camera_sampling::cycles_pixel_y(full_y, parameters.full_height);
    const auto rng_hash =
        cycles_sampler::pixel_hash(full_x, cycles_y, parameters.seed);

    const CameraDimensionSample cam_dim{
        .cycles_y = cycles_y,
        .rng_hash = rng_hash,
        .filter_sample = make_float2(0.5f, 0.5f),
        .lens_time_sample = make_float3(0.5f, 0.5f, 0.5f)};
    const CameraRaySample cam_ray = construct_camera_ray(
        filter_table, parameters, full_x, full_y, cam_dim,
        key.projection, key.depth_of_field,
        key.aperture_blades, key.aperture_rotation, safe_normalize);

    auto ray = cam_ray.ray;
    Float ray_dP = cam_ray.differential_position;
    Float ray_dD = cam_ray.differential_direction;
    ScenePrimitiveIdentity prev_src = ScenePrimitiveIdentity::invalid();

    const bool has_geom =
        !scene->geometries.empty() || !scene->curve_geometries.empty();
    const auto &translator =
        svm::svm_translator_for(svm::SvmBackendStrategy::bytecode_vm);
    const auto identity = make_float4x4(1.0f);
    Bool active = def(true);
    UInt bounce = 0u;
    UInt transparent_steps = 0u;
    UInt total_steps = 0u;
    const UInt max_total_steps = max_bounces + max_transparent_steps + 1u;

    $while(active & (total_steps < max_total_steps)) {
      const Bool is_camera_step = (bounce == 0u);
      const UInt trav_vis = ite(
          is_camera_step,
          UInt(contract::visibility_bit(contract::RayVisibility::camera)),
          UInt(contract::visibility_bit(contract::RayVisibility::diffuse) |
               contract::visibility_bit(contract::RayVisibility::glossy) |
               contract::visibility_bit(contract::RayVisibility::transmission)));
      const UInt svm_vis = ite(
          is_camera_step,
          UInt(static_cast<std::uint32_t>(abi::PATH_RAY_VISIBILITY_CAMERA)),
          UInt(static_cast<std::uint32_t>(abi::PATH_RAY_VISIBILITY_DIFFUSE)));
      const UInt demand_weight = ite(is_camera_step, 4u, 1u);
      const UInt bg_demand_weight = ite(is_camera_step, 2u, 1u);
      const auto step_lcg = cycles_noise::hash_uint3(
          rng_hash ^ 0xb4bc3953u, total_steps, 0u);

      auto eval_background = [&]() noexcept {
        if (!preflight_svm.has_bg_vt || bg_shader_idx >= shader_count_u32) {
          return;
        }
        const PathCyclesSvmKernelGlobals bg_kg{
            scene, parameters, key.projection, true, true,
            PathCyclesSvmShaderDomain::background,
            std::span<const std::uint8_t>{preflight_svm.bg_active_mask},
            false, nullptr, &bg_demand_weight};
        auto bg_sd = setup_cycles_svm_background_shader_data(
            *scene, ray->origin(), ray->direction(), ray_dD, 0.0f,
            svm_vis, CyclesSvmBackgroundEvaluation::forward);
        bg_sd.lcg_state = step_lcg;
        const auto bg_sid = bg_sd.shader & svm::shader_mask;
        $if(bg_sid == bg_shader_idx) {
          const svm::TransformState transforms{
              parameters.camera_transform,
              parameters.camera_inverse_transform,
              identity, identity};
          const svm::PathState bg_path_state{
              svm_vis, UInt(0u), bounce, transparent_steps, bounce,
              UInt(0u), UInt(0u), UInt(0u)};
          translator.translate(svm::SvmTranslationRequest{
              .kernel_globals = bg_kg,
              .table = preflight_svm.table,
              .words = preflight_words,
              .shader_type = abi::SHADER_TYPE_SURFACE,
              .kernel_features = scene->cycles_svm->kernel_features,
              .node_feature_mask = bg_node_feature_mask,
              .node_types_used = preflight_svm.bg_nodes_used,
              .transform_state = transforms,
              .shader_data = bg_sd,
              .path_state = bg_path_state,
              .stack_size = bg_stack_size,
              .noise_usage = bg_noise,
              .active_shader_mask = preflight_svm.bg_active_mask,
              .max_groups_per_outlined_chunk = 0u,
              .outline_chunk_dispatcher = false});
        };
      };

      if (!has_geom) {
        eval_background();
        active = false;
      } else {
        const auto hit = traversal->closest(scene, ray, trav_vis, prev_src);
        $if(hit->miss()) {
          eval_background();
          active = false;
        } $else {
          const PathCyclesSvmKernelGlobals surf_kg{
              scene, parameters, key.projection, true, true,
              PathCyclesSvmShaderDomain::surface_only,
              std::span<const std::uint8_t>{preflight_svm.surface_active_mask},
              false, nullptr, &demand_weight};
          auto setup = setup_cycles_svm_ray_shader_data(
              scene, surf_kg, ray, hit->inst, hit->prim, hit->bary,
              hit->committed_ray_t, ray_dP, ray_dD, 0.0f, step_lcg, parameters,
              nullptr, nullptr, nullptr);
          auto &sd = setup.shader_data;
          const auto sid = sd.shader & svm::shader_mask;
          const UInt bmode_word =
              shader_bounce_modes.read(min(sid, last_shader_id));
          if (preflight_svm.has_surface_vt) {
            $if((sid < shader_count_u32) &
                ((bmode_word & preflight_has_surface_vt_bit) != 0u)) {
              const svm::PathState surf_path_state{
                  svm_vis, UInt(0u), bounce, transparent_steps, bounce,
                  UInt(0u), UInt(0u), UInt(0u)};
              translator.translate(svm::SvmTranslationRequest{
                  .kernel_globals = surf_kg,
                  .table = preflight_svm.table,
                  .words = preflight_words,
                  .shader_type = abi::SHADER_TYPE_SURFACE,
                  .kernel_features = scene->cycles_svm->kernel_features,
                  .node_feature_mask = node_feature_mask,
                  .node_types_used = preflight_svm.surface_nodes_used,
                  .transform_state = setup.transforms,
                  .shader_data = sd,
                  .path_state = surf_path_state,
                  .stack_size = surface_stack_size,
                  .noise_usage = surface_noise,
                  .active_shader_mask = preflight_svm.surface_active_mask,
                  .max_groups_per_outlined_chunk = 0u,
                  .outline_chunk_dispatcher = false});
            };
          }

          const UInt bmode = bmode_word & preflight_bounce_mode_mask;
          const auto face_Ng =
              select(-sd.Ng, sd.Ng, dot(sd.Ng, ray->direction()) <= 0.0f);
          const auto h0 = cycles_noise::hash_uint3(
              gx ^ 0x9e3779b9u, gy ^ 0x85ebca6bu, total_steps + 1u);
          const auto h1 = cycles_noise::hash_uint(h0 ^ 0xc2b2ae35u);
          const Bool can_bounce = bounce < max_bounces;
          const Bool do_pass_through =
              (bmode == preflight_bounce_transparent_pass) &
              (transparent_steps < max_transparent_steps) &
              (!can_bounce | ((h1 & 3u) != 0u));
          $if(do_pass_through) {
            const auto next_org = offset_ray_origin(sd.P, -face_Ng);
            ray = make_ray(next_org, ray->direction(), 0.0f, 1.0e30f);
            ray_dP = sd.dP;
            prev_src.object = sd.object;
            prev_src.primitive = sd.prim;
            transparent_steps += 1u;
          } $elif(can_bounce) {
            const Bool do_reflect =
                (bmode == preflight_bounce_glossy_reflect) |
                ((bmode == preflight_bounce_transparent_pass) &
                 ((h1 & 3u) == 0u)) |
                ((bmode == preflight_bounce_mixed_glossy_diffuse) &
                 ((h1 & 3u) == 0u));
            const auto rd = ray->direction();
            const auto ndotr = dot(rd, face_Ng);
            const auto refl_dir = safe_normalize(
                rd - (2.0f * ndotr) * face_Ng, face_Ng);
            const auto u1 = cycles_noise::uint_to_float_inclusive(h0);
            const auto u2 = cycles_noise::uint_to_float_inclusive(h1);
            const auto r_s = sqrt(clamp(u1, 0.0f, 1.0f));
            const auto phi = (2.0f * 3.14159265359f) * u2;
            const auto lx = r_s * cos(phi);
            const auto ly = r_s * sin(phi);
            const auto lz = sqrt(max(0.0f, 1.0f - u1));
            const auto ref = select(
                make_float3(1.0f, 0.0f, 0.0f),
                make_float3(0.0f, 1.0f, 0.0f),
                abs(face_Ng.x) > 0.9f);
            const auto t_u = safe_normalize(
                cross(ref, face_Ng), make_float3(1.0f, 0.0f, 0.0f));
            const auto t_v = cross(face_Ng, t_u);
            const auto diff_dir = safe_normalize(
                t_u * lx + t_v * ly + face_Ng * lz, face_Ng);
            const auto next_dir = select(diff_dir, refl_dir, do_reflect);
            const auto next_org = offset_ray_origin(sd.P, face_Ng);
            ray = make_ray(next_org, next_dir, 0.0f, 1.0e30f);
            ray_dP = sd.dP;
            ray_dD = select(ray_dD, max(ray_dD, 0.05f), !do_reflect);
            prev_src.object = sd.object;
            prev_src.primitive = sd.prim;
            bounce += 1u;
          } $else {
            active = false;
          };
        };
      }
      total_steps += 1u;
    };
  };

  auto shader = scene->device.compile<2>(std::move(kernel), shader_option);
  if (scene->device.impl()->query("vk_timeline_active") == "1" &&
      shader.handle() != luisa::compute::invalid_resource_handle) {
    char qbuf[192];
    std::snprintf(qbuf, sizeof(qbuf),
                  "vk_timeline_label:shader:%llu:vt_preflight",
                  static_cast<unsigned long long>(shader.handle()));
    static_cast<void>(scene->device.impl()->query(qbuf));
  }
  return shader;
}

} // namespace

struct VtPreflightShaderCache {
  std::mutex mtx;
  luisa::compute::Stream stream;
  bool bytecode_ready{false};
  PreflightSvmBytecode bytecode;
  luisa::compute::Buffer<luisa::uint> preflight_words_buf;
  luisa::compute::Buffer<luisa::uint> shader_bounce_modes_buf;
  std::vector<VtPreflightCacheEntry> entries;
};

struct PreflightOverlapHandle::State {
  std::shared_ptr<LuisaSceneData> scene;
  std::thread worker;
  std::atomic<bool> joined{false};
  bool overlapped{false};
  std::uint32_t stride{2u};
  std::uint32_t bounces{1u};
  std::uint64_t start_ns{0ull};
  std::uint64_t work_end_ns{0ull};
  double compile_ms{0.0};
  double trace_ms{0.0};
  double service_ms{0.0};
};

PreflightOverlapHandle::~PreflightOverlapHandle() noexcept { join(); }

void PreflightOverlapHandle::join() noexcept {
  if (!_state) {
    return;
  }
  if (_state->joined.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  if (_state->worker.joinable()) {
    _state->worker.join();
  }
  const auto fallback_end_ns = VtProfileRecorder::steady_ns();
  const auto end_ns =
      (_state->work_end_ns >= _state->start_ns)
          ? _state->work_end_ns
          : fallback_end_ns;
  const double total_ms =
      static_cast<double>(end_ns - _state->start_ns) / 1.0e6;
  if (_state->scene && _state->scene->virtual_texture_cache) {
    _state->scene->virtual_texture_cache->set_preflight_timing_stats(
        true, _state->overlapped, _state->stride, _state->bounces,
        _state->compile_ms, _state->trace_ms, _state->service_ms, total_ms);
  }
}

PreflightOverlapHandle start_vt_preflight(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SessionKernelSignature &signature,
    const luisa::compute::Buffer<float> &filter_table,
    const RenderKernelParameters &parameters,
    const SafeNormalizeCallable &safe_normalize,
    const VtPreflightConfig &config) noexcept {
  PreflightOverlapHandle handle;
  if (!scene || !scene->virtual_texture_cache ||
      !scene->virtual_texture_cache->has_virtual_images() ||
      scene->virtual_texture_cache->eager_fallback_active() ||
      !scene->cycles_svm || !scene->cycles_svm->word_buffer) {
    return handle;
  }

  auto &vt = *scene->virtual_texture_cache;
  vt.prepare_timeline_labels_host();

  const auto start_ns = VtProfileRecorder::steady_ns();
  if (!scene->vt_preflight_cache) {
    scene->vt_preflight_cache = std::make_shared<VtPreflightShaderCache>();
  }
  auto &cache = *scene->vt_preflight_cache;

  luisa::compute::ShaderOption pf_shader_option = config.shader_option;
  pf_shader_option.enable_driver_optimization = false;

  const VtPreflightCacheKey key{
      .projection = signature.camera_projection,
      .depth_of_field = signature.camera_depth_of_field,
      .aperture_blades = signature.camera_aperture_blades,
      .aperture_rotation = signature.camera_aperture_rotation,
      .enable_cache = pf_shader_option.enable_cache,
      .enable_fast_math = pf_shader_option.enable_fast_math,
      .assume_no_packed_textures = pf_shader_option.assume_no_packed_textures,
      .enable_driver_optimization = pf_shader_option.enable_driver_optimization,
      .mip_mode = scene->virtual_texture_cache->mip_mode()};

  PreflightShader *shader_ptr = nullptr;
  const auto compile_t0 = VtProfileRecorder::steady_ns();
  {
    std::lock_guard lock{cache.mtx};
    if (!cache.stream) {
      cache.stream = scene->device.create_stream();
    }
    if (!cache.bytecode_ready) {
      cache.bytecode = build_preflight_svm_bytecode(*scene);
      const auto &words = cache.bytecode.table.words;
      const auto &bmodes = cache.bytecode.shader_bounce_modes;
      cache.preflight_words_buf = scene->device.create_buffer<luisa::uint>(
          std::max<std::size_t>(1u, words.size()));
      cache.shader_bounce_modes_buf = scene->device.create_buffer<luisa::uint>(
          std::max<std::size_t>(1u, bmodes.size()));
      luisa::compute::CommandList init_cmd;
      if (!words.empty()) {
        init_cmd << cache.preflight_words_buf.copy_from(words.data());
      } else {
        const luisa::uint end_w = static_cast<luisa::uint>(abi::NODE_END);
        init_cmd << cache.preflight_words_buf.copy_from(&end_w);
      }
      if (!bmodes.empty()) {
        init_cmd << cache.shader_bounce_modes_buf.copy_from(bmodes.data());
      } else {
        const luisa::uint zero_w = preflight_bounce_diffuse;
        init_cmd << cache.shader_bounce_modes_buf.copy_from(&zero_w);
      }
      cache.stream << init_cmd.commit() << luisa::compute::synchronize();
      cache.bytecode_ready = true;
    }
    for (auto &entry : cache.entries) {
      if (entry.key == key) {
        shader_ptr = &entry.shader;
        break;
      }
    }
    if (shader_ptr == nullptr) {
      VtPreflightCacheEntry entry;
      entry.key = key;
      entry.shader = compile_vt_preflight_shader(
          scene, cache.bytecode, key, safe_normalize, pf_shader_option);
      cache.entries.push_back(std::move(entry));
      shader_ptr = &cache.entries.back().shader;
    }
  }
  const auto compile_t1 = VtProfileRecorder::steady_ns();
  const double compile_ms =
      static_cast<double>(compile_t1 - compile_t0) / 1.0e6;
  vt.record_preflight_event(compile_t0, compile_t1, "Preflight Compile");

  const std::uint32_t stride = std::max<std::uint32_t>(config.stride, 1u);
  const std::uint32_t width = std::max<std::uint32_t>(config.width, 1u);
  const std::uint32_t height = std::max<std::uint32_t>(config.height, 1u);
  const std::uint32_t grid_w = (width + stride - 1u) / stride;
  const std::uint32_t grid_h = (height + stride - 1u) / stride;

  luisa::compute::CommandList cmd;
  vt.clear_preflight_demand_buffers(cmd);
  cmd << (*shader_ptr)(filter_table, cache.preflight_words_buf,
                       cache.shader_bounce_modes_buf, parameters, stride,
                       width, height, config.bounces)
             .dispatch(grid_w, grid_h);
  vt.enqueue_preflight_demand_readback(cmd);
  cache.stream << cmd.commit();

  auto st = std::make_shared<PreflightOverlapHandle::State>();
  st->scene = scene;
  st->overlapped = config.overlap_pso;
  st->stride = stride;
  st->bounces = config.bounces;
  st->start_ns = start_ns;
  st->compile_ms = compile_ms;

  auto run_sync_and_service = [st]() noexcept {
    auto &stream = st->scene->vt_preflight_cache->stream;
    auto &vtc = *st->scene->virtual_texture_cache;
    const auto sync_t0 = VtProfileRecorder::steady_ns();
    stream << luisa::compute::synchronize();
    const auto sync_t1 = VtProfileRecorder::steady_ns();
    st->trace_ms = static_cast<double>(sync_t1 - sync_t0) / 1.0e6;
    vtc.record_preflight_event(sync_t0, sync_t1, "Preflight GPU Trace + Sync");

    const auto svc_t0 = VtProfileRecorder::steady_ns();
    vtc.service_preflight_misses(stream, true);
    const auto svc_t1 = VtProfileRecorder::steady_ns();
    st->service_ms = static_cast<double>(svc_t1 - svc_t0) / 1.0e6;
    st->work_end_ns = svc_t1;
  };

  handle._state = st;
  if (config.overlap_pso) {
    st->worker = std::thread(std::move(run_sync_and_service));
  } else {
    run_sync_and_service();
    handle.join();
  }
  return handle;
}

void VirtualTextureCache::clear_preflight_demand_buffers(
    luisa::compute::CommandList &cmd) noexcept {
  if (_pages.empty() || !_clear_u32_shader) {
    return;
  }
  const auto total_pages = static_cast<luisa::uint>(_pages.size());
  cmd << _clear_u32_shader(_aux_demand_hist, total_pages).dispatch(total_pages);
}

void VirtualTextureCache::enqueue_preflight_demand_readback(
    luisa::compute::CommandList &cmd) noexcept {
  if (_pages.empty() || !_use_pinned_readback) {
    return;
  }
  cmd << _readback_aux_demand_hist.copy_from(_aux_demand_hist);
}

void VirtualTextureCache::record_preflight_event(
    std::uint64_t start_ns,
    std::uint64_t end_ns,
    std::string name,
    std::string detail) const noexcept {
  if (_profiler) {
    _profiler->record_preflight_event(
        start_ns, end_ns, std::move(name), std::move(detail));
  }
}

void VirtualTextureCache::record_pso_compile_event(
    std::uint64_t start_ns,
    std::uint64_t end_ns,
    std::string name,
    std::string detail) const noexcept {
  if (_profiler) {
    _profiler->record_pso_compile_event(
        start_ns, end_ns, std::move(name), std::move(detail));
  }
}

void VirtualTextureCache::set_preflight_timing_stats(
    bool enabled,
    bool overlapped,
    std::uint32_t stride,
    std::uint32_t bounces,
    double compile_ms,
    double trace_ms,
    double service_ms,
    double total_ms) noexcept {
  _stats.preflight_enabled = enabled;
  _stats.preflight_overlapped = overlapped;
  _stats.preflight_stride = stride;
  _stats.preflight_bounces = bounces;
  _stats.preflight_compile_ms = compile_ms;
  _stats.preflight_trace_ms = trace_ms;
  _stats.preflight_service_ms = service_ms;
  _stats.preflight_total_ms = total_ms;
}

void VirtualTextureCache::set_pso_compile_ms(double ms) noexcept {
  _stats.pso_compile_ms = ms;
}

void VirtualTextureCache::service_preflight_misses(
    luisa::compute::Stream &stream,
    bool readback_already_synced) noexcept {
  if (_pages.empty()) {
    return;
  }
  using namespace luisa::compute;
  const std::uint32_t total_pages = static_cast<std::uint32_t>(_pages.size());

  const luisa::uint *hist_ptr = nullptr;
  if (_use_pinned_readback) {
    if (!readback_already_synced) {
      stream << _readback_aux_demand_hist.copy_from(_aux_demand_hist)
             << synchronize();
    }
    _pinned_ext->flush_range(
        _readback_aux_demand_hist.handle(), 0ull,
        static_cast<std::uint64_t>(total_pages) * sizeof(luisa::uint));
    hist_ptr = _pinned_aux_demand_hist_ptr;
  } else {
    stream << _aux_demand_hist.copy_to(_host_aux_demand_hist.data())
           << synchronize();
    hist_ptr = _host_aux_demand_hist.data();
  }

  std::vector<std::vector<MissCandidate>> pool_misses(_pools.size());
  std::uint32_t discovered_total = 0u;
  for (std::uint32_t pid = 0u; pid < total_pages; ++pid) {
    const luisa::uint demand = hist_ptr[pid];
    if (demand == 0u) {
      continue;
    }
    if ((_host_page_table[pid] & vt_page_resident_bit) != 0u) {
      continue;
    }
    ++discovered_total;
    pool_misses[_pages[pid].pool_index].push_back(
        MissCandidate{.page = pid, .primary_demand = demand});
  }
  _stats.preflight_tiles_discovered = discovered_total;

  if (discovered_total == 0u) {
    CommandList clear_cmd;
    clear_cmd << _clear_u32_shader(_aux_demand_hist, total_pages)
                     .dispatch(total_pages);
    stream << clear_cmd.commit() << synchronize();
    _stats.preflight_tiles_loaded = 0u;
    _stats.preflight_upload_bytes = 0ull;
    _stats.preflight_bytes_loaded = 0ull;
    return;
  }

  // Plan strictly within free slots (zero evictions during preflight).
  auto &plans = _sync_planning_scratch.plans;
  plans.clear();
  plans.resize(_pools.size());
  std::size_t last_active_pool = _pools.size();
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    auto &pool_rt = _pools[p];
    const auto cap = pool_rt.config.capacity;
    std::vector<std::uint32_t> free_slots;
    free_slots.reserve(cap);
    for (std::uint32_t s = 0u; s < cap; ++s) {
      if (pool_rt.host_slots[s] == vt_invalid_page) {
        free_slots.push_back(s);
      }
    }
    auto &misses = pool_misses[p];
    if (misses.empty() || free_slots.empty()) {
      continue;
    }
    const std::size_t chosen = std::min(misses.size(), free_slots.size());
    if (misses.size() > chosen) {
      std::nth_element(
          misses.begin(), misses.begin() + chosen, misses.end(),
          [](const MissCandidate &a, const MissCandidate &b) {
            if (a.primary_demand != b.primary_demand) {
              return a.primary_demand > b.primary_demand;
            }
            return a.page < b.page;
          });
      misses.resize(chosen);
    }
    std::sort(misses.begin(), misses.end(),
              [&](const MissCandidate &a, const MissCandidate &b) {
                const auto &pa = _pages[a.page];
                const auto &pb = _pages[b.page];
                return std::tie(pa.source_index, pa.tile_y, pa.tile_x) <
                       std::tie(pb.source_index, pb.tile_y, pb.tile_x);
              });

    auto &plan = plans[p];
    plan.misses = std::move(misses);
    plan.victim_slots.assign(free_slots.begin(), free_slots.begin() + chosen);
    plan.wf_seq = pool_rt.wavefront_seq;
    plan.load_seq = ++pool_rt.load_seq;
    plan.new_slot_epoch = pool_rt.config.eviction_possible
                              ? (_host_current_epoch[p] > 0u
                                     ? _host_current_epoch[p] - 1u
                                     : 0u)
                              : 0u;
    plan.has_work = true;
    pool_rt.used_slots += static_cast<std::uint32_t>(chosen);
    last_active_pool = p;
  }

  if (last_active_pool == _pools.size()) {
    CommandList clear_cmd;
    clear_cmd << _clear_u32_shader(_aux_demand_hist, total_pages)
                     .dispatch(total_pages);
    stream << clear_cmd.commit() << synchronize();
    _stats.preflight_tiles_loaded = 0u;
    _stats.preflight_upload_bytes = 0ull;
    _stats.preflight_bytes_loaded = 0ull;
    return;
  }

  const double saved_decode_ms = _stats.decode_ms;
  const double saved_pack_ms = _stats.pack_ms;
  const double saved_upload_ms = _stats.upload_ms;
  const std::uint64_t saved_gpu_upload_bytes = _stats.gpu_upload_bytes;
  const std::uint64_t saved_bytes_loaded = _stats.bytes_loaded_total;
  const std::uint64_t saved_tiles_loaded = _stats.tiles_loaded;
  const std::uint64_t saved_tiles_evicted = _stats.tiles_evicted;
  const std::uint64_t saved_cq_batches = _stats.copy_queue_batches;
  const std::uint64_t saved_cq_h2d_bytes = _stats.copy_queue_h2d_bytes;
  const std::uint64_t saved_cq_atlas_bytes = _stats.copy_queue_atlas_bytes;
  const double saved_cq_submit_ms = _stats.copy_queue_submit_ms;
  const std::uint64_t saved_decode_evictions = _stats.host_decode_evictions;

  const bool use_prepack =
      (_options.prepack_mode != LuisaVirtualTexturePrepackMode::off) &&
      (_prepack_cache != nullptr) &&
      (_page_prepack_keys.size() == _pages.size());
  const bool use_disk_prepack =
      use_prepack &&
      (_options.prepack_mode == LuisaVirtualTexturePrepackMode::disk) &&
      _prepack_disk && _prepack_disk->enabled();

  // Batch decode all unique needed sources upfront once across all active pools.
  std::vector<std::uint32_t> unique_sources;
  std::vector<std::uint8_t> src_seen(_sources.size(), 0u);
  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    if (!plans[p].has_work) {
      continue;
    }
    for (const auto &m : plans[p].misses) {
      const auto &pm = _pages[m.page];
      const auto si = pm.source_index;
      if (si >= _sources.size() || src_seen[si] != 0u) {
        continue;
      }
      const auto &src = _sources[si];
      if (src.decoded || src.load_failed) {
        src_seen[si] = 1u;
        continue;
      }
      if (use_disk_prepack &&
          _prepack_disk->has_valid_file(_page_prepack_keys[m.page].image,
                                        pm.tiles_x, pm.tiles_y)) {
        continue;
      }
      src_seen[si] = 1u;
      unique_sources.push_back(si);
    }
  }
  if (!unique_sources.empty()) {
    const auto dec_t0 = VtProfileRecorder::steady_ns();
    decode_sources_batch(unique_sources);
    const auto dec_t1 = VtProfileRecorder::steady_ns();
    record_preflight_event(dec_t0, dec_t1, "Preflight Batch Decode");
  }

  const auto pack_up_t0 = VtProfileRecorder::steady_ns();
  const std::uint64_t cur_stream_h = static_cast<std::uint64_t>(stream.handle());
  const std::size_t meta_slot = acquire_meta_slot(cur_stream_h);
  luisa::uint2 *pt_upd_dst =
      _use_pinned_upload ? _pinned_page_table_updates_ptr[meta_slot] : nullptr;
  luisa::uint2 *ep_upd_dst =
      _use_pinned_upload ? _pinned_slot_epoch_updates_ptr[meta_slot] : nullptr;
  std::size_t pt_upd_count = 0u;
  std::size_t ep_upd_count = 0u;
  if (!_use_pinned_upload) {
    _host_page_table_updates.clear();
    _host_slot_epoch_updates.clear();
  }

  bool gpu_state_dirty = false;
  std::uint32_t dummy_b4 = 0u;
  std::uint32_t dummy_f4 = 0u;

  for (std::size_t p = 0u; p < _pools.size(); ++p) {
    const auto &plan = plans[p];
    if (!plan.has_work) {
      continue;
    }
    auto &pool_rt = _pools[p];
    const auto &pool = pool_rt.config;
    const std::size_t total_load = plan.misses.size();
    const std::size_t staging_cap =
        std::max<std::size_t>(pool.staging_capacity, 1u);
    std::uint32_t dummy_pool = 0u;

    for (std::size_t chunk_begin = 0u; chunk_begin < total_load;
         chunk_begin += staging_cap) {
      const std::size_t chunk_end =
          std::min(chunk_begin + staging_cap, total_load);
      const std::size_t chunk_count = chunk_end - chunk_begin;
      const std::size_t chunk_slot =
          acquire_chunk_slot(static_cast<std::uint32_t>(p), cur_stream_h);

      ChunkStagingTarget target{.pool_index = static_cast<std::uint32_t>(p),
                                .chunk_slot = chunk_slot};
      if (_use_pinned_upload) {
        target.pack_base = pool_rt.pinned_staging_ptr[chunk_slot];
        target.um_dst = pool.use_batch_copy_upload
                            ? nullptr
                            : pool_rt.pinned_upload_meta_ptr[chunk_slot];
      } else {
        target.pack_base = pool_rt.host_staging.data();
        target.um_dst =
            pool.use_batch_copy_upload ? nullptr : _host_upload_meta.data();
      }

      const PackedChunkMeta packed = prepare_chunk_staging(
          static_cast<std::uint32_t>(p), pool_rt, plan, chunk_begin, chunk_end,
          staging_cap, target, pt_upd_dst, _host_page_table_updates,
          pt_upd_count, ep_upd_dst, _host_slot_epoch_updates, ep_upd_count,
          false, gpu_state_dirty, dummy_b4, dummy_f4, dummy_pool,
          _sync_planning_scratch);

      const auto chunk_u32 = static_cast<luisa::uint>(chunk_count);
      const auto elem_count = static_cast<luisa::uint>(
          chunk_count * pool.tile_staging_u32s);
      if (_use_pinned_upload) {
        if (!pool.use_batch_copy_upload) {
          _pinned_ext->flush_range(
              pool_rt.upload_meta[chunk_slot].handle(), 0ull,
              static_cast<std::uint64_t>(chunk_count) * sizeof(luisa::uint4));
        }
        _pinned_ext->flush_range(
            pool_rt.staging[chunk_slot].handle(), 0ull,
            static_cast<std::uint64_t>(elem_count) * sizeof(luisa::uint));
      }

      std::uint64_t h2d_fence = 0ull;
      if (_use_host_cached_staging && _use_copy_stream && chunk_count > 0u) {
        h2d_fence = submit_staged_chunk_h2d(
            static_cast<std::uint32_t>(p), chunk_slot, chunk_count, true);
      }

      auto emit_fallback_uploads = [&](CommandList &target_cmd) {
        if (_use_pinned_upload) {
          return;
        }
        if (!pool.use_batch_copy_upload) {
          target_cmd << pool_rt.upload_meta[chunk_slot]
                            .view(0u, chunk_u32)
                            .copy_from(luisa::span{_host_upload_meta}
                                           .subspan(0u, chunk_count));
        }
        target_cmd << pool_rt.staging[chunk_slot]
                          .view(0u, elem_count)
                          .copy_from(luisa::span{pool_rt.host_staging}
                                         .subspan(0u, elem_count));
      };

      std::uint64_t atlas_bytes = 0ull;
      if (_copy_queue_mode == LuisaVirtualTextureCopyQueueMode::direct) {
        if (!_use_pinned_upload) {
          CommandList pre_cmd;
          emit_fallback_uploads(pre_cmd);
          stream << pre_cmd.commit();
        }
        std::array<PackedChunkMeta, 1u> one_chunk{packed};
        submit_direct_copy_chunks(stream, one_chunk, &atlas_bytes);
        if (_use_pinned_upload) {
          auto sig = _upload_event.signal();
          const auto fence = sig.fence;
          stream << std::move(sig);
          pool_rt.slot_fence[chunk_slot] = fence;
        } else {
          stream << synchronize();
        }
      } else {
        if (h2d_fence != 0ull) {
          stream << _copy_h2d_event.wait(h2d_fence);
        }
        CommandList upload_cmd;
        emit_fallback_uploads(upload_cmd);
        emit_chunk_atlas_upload(upload_cmd, packed, &atlas_bytes);
        if (_use_pinned_upload) {
          auto sig = _upload_event.signal();
          const auto fence = sig.fence;
          stream << upload_cmd.commit() << std::move(sig);
          pool_rt.slot_fence[chunk_slot] = fence;
        } else {
          stream << upload_cmd.commit() << synchronize();
        }
      }
      _stats.gpu_upload_bytes += atlas_bytes;
    }
  }

  CommandList finalize_cmd;
  finalize_cmd << _clear_u32_shader(_aux_demand_hist, total_pages)
                      .dispatch(total_pages);
  if (gpu_state_dirty && pt_upd_count > 0u) {
    std::uint32_t b4_resident = 0u;
    std::uint32_t f4_resident = 0u;
    for (const auto &pr : _pools) {
      std::uint32_t cnt = 0u;
      for (const auto s : pr.host_slots) {
        if (s != vt_invalid_page) {
          ++cnt;
        }
      }
      if (pr.config.is_high_precision) {
        f4_resident += cnt;
      } else {
        b4_resident += cnt;
      }
    }
    _stats.byte4_peak_resident =
        std::max(_stats.byte4_peak_resident, b4_resident);
    _stats.float4_peak_resident =
        std::max(_stats.float4_peak_resident, f4_resident);

    const auto upd_count = static_cast<luisa::uint>(pt_upd_count);
    if (_use_pinned_upload) {
      _pinned_ext->flush_range(
          _page_table_updates[meta_slot].handle(), 0ull,
          static_cast<std::uint64_t>(upd_count) * sizeof(luisa::uint2));
    } else {
      finalize_cmd
          << _page_table_updates[meta_slot]
                 .view(0u, upd_count)
                 .copy_from(luisa::span{_host_page_table_updates});
    }
    finalize_cmd
        << _apply_page_table_updates_shader(
               _page_table, _page_table_updates[meta_slot], upd_count)
               .dispatch(upd_count);
    if (ep_upd_count > 0u) {
      const auto ep_upd_count_u32 = static_cast<luisa::uint>(ep_upd_count);
      if (_use_pinned_upload) {
        _pinned_ext->flush_range(
            _slot_epoch_updates[meta_slot].handle(), 0ull,
            static_cast<std::uint64_t>(ep_upd_count_u32) *
                sizeof(luisa::uint2));
      } else {
        finalize_cmd
            << _slot_epoch_updates[meta_slot]
                   .view(0u, ep_upd_count_u32)
                   .copy_from(luisa::span{_host_slot_epoch_updates});
      }
      finalize_cmd
          << _apply_page_table_updates_shader(
                 _slot_epoch, _slot_epoch_updates[meta_slot],
                 ep_upd_count_u32)
                 .dispatch(ep_upd_count_u32);
    }
    {
      std::lock_guard lock{_vt_worker_mtx};
      const std::uint64_t next_pub_epoch = ++_publish_epoch;
      for (std::size_t p = 0u; p < _pools.size(); ++p) {
        if (!plans[p].has_work) {
          continue;
        }
        auto &pr = _pools[p];
        for (const auto v : plans[p].victim_slots) {
          if (v < pr.slot_publish_epoch.size()) {
            pr.slot_publish_epoch[v] = next_pub_epoch;
          }
        }
      }
    }
  }
  stream << finalize_cmd.commit() << synchronize();
  wait_all_upload_fences();

  for (auto &pr : _pools) {
    std::fill(pr.slot_fence.begin(), pr.slot_fence.end(), 0ull);
    std::fill(pr.slot_stream.begin(), pr.slot_stream.end(), 0ull);
  }
  std::fill(_meta_slot_fence.begin(), _meta_slot_fence.end(), 0ull);
  std::fill(_meta_slot_stream.begin(), _meta_slot_stream.end(), 0ull);

  const auto pack_up_t1 = VtProfileRecorder::steady_ns();
  record_preflight_event(pack_up_t0, pack_up_t1, "Preflight Pack + Upload");

  sync_prepack_stats();
  _stats.preflight_tiles_loaded =
      static_cast<std::uint32_t>(_stats.tiles_loaded - saved_tiles_loaded);
  _stats.preflight_upload_bytes =
      _stats.gpu_upload_bytes - saved_gpu_upload_bytes;
  _stats.preflight_bytes_loaded =
      _stats.bytes_loaded_total - saved_bytes_loaded;

  // Preserve runtime integration stats at 0 prior to render_samples.
  _stats.decode_ms = saved_decode_ms;
  _stats.pack_ms = saved_pack_ms;
  _stats.upload_ms = saved_upload_ms;
  _stats.gpu_upload_bytes = saved_gpu_upload_bytes;
  _stats.tiles_loaded = saved_tiles_loaded;
  _stats.tiles_evicted = saved_tiles_evicted;
  _stats.copy_queue_batches = saved_cq_batches;
  _stats.copy_queue_h2d_bytes = saved_cq_h2d_bytes;
  _stats.copy_queue_atlas_bytes = saved_cq_atlas_bytes;
  _stats.copy_queue_submit_ms = saved_cq_submit_ms;
  _stats.host_decode_evictions = saved_decode_evictions;
}

} // namespace psycles::luisa_backend::detail

#include "path_tracer_cycles_svm_kernel_globals.h"

#include "cycles_svm_image_sampling.h"
#include "cycles_svm_internal.h"
#include "virtual_texture_cache.h"

#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/cycles_transform.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <algorithm>
#include <cmath>
#include <utility>

#include <luisa/dsl/struct.h>

namespace psycles::luisa_backend::detail {

struct KernelObjDual3 {
  luisa::float3 val;
  luisa::float3 dx;
  luisa::float3 dy;
};

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::KernelObjDual3, val, dx, dy){};

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;
namespace svm = ::psycles::luisa_backend::cycles_svm;
namespace svm_detail = ::psycles::luisa_backend::cycles_svm::detail;
namespace abi = ::psycles::compiler::cycles_svm;

[[nodiscard]] Float3 unpack(Var<abi::packed_float3> value) noexcept {
  return make_float3(value.x, value.y, value.z);
}

[[nodiscard]] Float4 unpack(Var<abi::packed_float4> value) noexcept {
  return make_float4(value.x, value.y, value.z, value.w);
}

[[nodiscard]] UInt3 unpack(Var<abi::packed_uint3> value) noexcept {
  return make_uint3(value.x, value.y, value.z);
}

[[nodiscard]] Float4x4 unpack_transform(
    Var<abi::PackedTransform> value) noexcept {
  return svm_detail::transform_from_rows(
      make_float4(value.x.x, value.x.y, value.x.z, value.x.w),
      make_float4(value.y.x, value.y.y, value.y.z, value.y.w),
      make_float4(value.z.x, value.z.y, value.z.z, value.z.w));
}

[[nodiscard]] Var<abi::KernelObject> object_record(
    const LuisaSceneData &scene, Expr<std::uint32_t> object) noexcept {
  const UInt index = select(0u, object,
                            object != svm::object_none);
  return scene.cycles_svm->objects->object_buffer->read(index);
}

[[nodiscard]] Var<CyclesSvmParticleGpu> particle_record(
    const LuisaSceneData &scene, Expr<std::int32_t> particle) noexcept {
  const Expr<Buffer<CyclesSvmParticleGpu>> records{
      *scene.cycles_svm->particle_buffer};
  return records->read(
      max(particle, 0).cast<std::uint32_t>());
}

[[nodiscard]] Float3 host_vector(const Vec3f &value) noexcept {
  return make_float3(value.x, value.y, value.z);
}

[[nodiscard]] Vec3f rgb_to_y_coefficients(
    const contract::ShaderColorSpace &color_space) noexcept {
  // ShaderColorSpace stores the three rows of XYZ -> scene-linear RGB.
  // Film::rgb_to_y is row one of its inverse. Computing that row here keeps
  // arbitrary OCIO working spaces exact without adding a second serialized
  // color-space representation.
  const auto a = static_cast<double>(color_space.xyz_to_r.x);
  const auto b = static_cast<double>(color_space.xyz_to_r.y);
  const auto c = static_cast<double>(color_space.xyz_to_r.z);
  const auto d = static_cast<double>(color_space.xyz_to_g.x);
  const auto e = static_cast<double>(color_space.xyz_to_g.y);
  const auto f = static_cast<double>(color_space.xyz_to_g.z);
  const auto g = static_cast<double>(color_space.xyz_to_b.x);
  const auto h = static_cast<double>(color_space.xyz_to_b.y);
  const auto i = static_cast<double>(color_space.xyz_to_b.z);
  const auto determinant =
      a * (e * i - f * h) - b * (d * i - f * g) +
      c * (d * h - e * g);
  LUISA_ASSERT(std::abs(determinant) > 1.0e-20,
               "Shader XYZ-to-RGB matrix is singular.");
  return {static_cast<float>((f * g - d * i) / determinant),
          static_cast<float>((a * i - c * g) / determinant),
          static_cast<float>((c * d - a * f) / determinant)};
}

[[nodiscard]] bool is_rec709(
    const contract::ShaderColorSpace &color_space) noexcept {
  return color_space.rec709_to_r == Vec3f{1.0f, 0.0f, 0.0f} &&
         color_space.rec709_to_g == Vec3f{0.0f, 1.0f, 0.0f} &&
         color_space.rec709_to_b == Vec3f{0.0f, 0.0f, 1.0f};
}

[[nodiscard]] Bool attribute_found(
    const svm::AttributeDescriptor &descriptor) noexcept {
  return descriptor.offset !=
         static_cast<std::int32_t>(abi::ATTR_STD_NOT_FOUND);
}

[[nodiscard]] svm::Dual3 dual_cross_right(
    const svm::Dual3 &lhs, Expr<luisa::float3> rhs) noexcept {
  return {.val = cross(lhs.val, rhs),
          .dx = cross(lhs.dx, rhs),
          .dy = cross(lhs.dy, rhs)};
}

[[nodiscard]] svm::Dual3 dual_cross_left(
    Expr<luisa::float3> lhs, const svm::Dual3 &rhs) noexcept {
  return {.val = cross(lhs, rhs.val),
          .dx = cross(lhs, rhs.dx),
          .dy = cross(lhs, rhs.dy)};
}

[[nodiscard]] svm::Dual3 transform_direction_transposed(
    Expr<luisa::float4x4> transform,
    const svm::Dual3 &value) noexcept {
  return {.val = cycles_transform::direction_transposed(transform, value.val),
          .dx = cycles_transform::direction_transposed(transform, value.dx),
          .dy = cycles_transform::direction_transposed(transform, value.dy)};
}

[[nodiscard]] bool shader_in_domain(
    const LuisaSceneData &scene, std::uint32_t shader_index,
    PathCyclesSvmShaderDomain domain) noexcept {
  const auto &compilation = scene.cycles_svm->compilation;
  const auto bg_idx =
      scene.cycles_background_shader_id != ~0u
          ? (scene.cycles_background_shader_id & svm::shader_mask)
          : ~0u;
  const auto flags =
      static_cast<std::uint32_t>(compilation.kernel_shaders[shader_index].flags);
  const auto &entry = compilation.shader_surface_entry_usage[shader_index];
  switch (domain) {
    case PathCyclesSvmShaderDomain::all_shaders:
      return true;
    case PathCyclesSvmShaderDomain::surface_only:
      return !(shader_index == bg_idx &&
               entry.node_types_used[abi::NODE_CLOSURE_BACKGROUND]);
    case PathCyclesSvmShaderDomain::shadow_surface:
      return (flags &
              static_cast<std::uint32_t>(abi::SD_HAS_TRANSPARENT_SHADOW |
                                         abi::SD_RAY_PORTAL)) != 0u &&
             (flags & static_cast<std::uint32_t>(abi::SD_HAS_ONLY_VOLUME)) == 0u;
    case PathCyclesSvmShaderDomain::light_emission:
      return (flags & static_cast<std::uint32_t>(abi::SD_HAS_CONSTANT_EMISSION)) ==
                 0u &&
             ((flags & static_cast<std::uint32_t>(abi::SD_HAS_EMISSION)) != 0u ||
              entry.node_types_used[abi::NODE_CLOSURE_EMISSION] ||
              entry.node_types_used[abi::NODE_CLOSURE_BACKGROUND] ||
              shader_index == bg_idx);
    case PathCyclesSvmShaderDomain::background:
      return shader_index == bg_idx;
  }
  return true;
}

} // namespace

bool cycles_svm_has_domain_metadata(const LuisaSceneData &scene) noexcept {
  if (!scene.cycles_svm) {
    return false;
  }
  const auto &compilation = scene.cycles_svm->compilation;
  const auto count = static_cast<std::size_t>(compilation.table.shader_count);
  return compilation.table.valid &&
         compilation.shader_metadata.size() == count &&
         compilation.shader_surface_entry_usage.size() == count &&
         compilation.kernel_shaders.size() == count;
}

abi::ShaderEntryUsage cycles_svm_domain_entry_usage(
    const LuisaSceneData &scene,
    PathCyclesSvmShaderDomain domain) noexcept {
  const auto &compilation = scene.cycles_svm->compilation;
  const auto table_usage =
      compilation.table.usage_for(abi::SHADER_TYPE_SURFACE);
  if (!cycles_svm_has_domain_metadata(scene) ||
      domain == PathCyclesSvmShaderDomain::all_shaders) {
    return table_usage;
  }
  abi::ShaderEntryUsage result{};
  result.node_types_used[abi::NODE_SHADER_JUMP] = true;
  result.node_types_used[abi::NODE_END] = true;
  result.noise_usage = abi::NoiseUsage::none();
  const auto count = compilation.table.shader_count;
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (!shader_in_domain(scene, i, domain)) {
      continue;
    }
    const auto &entry = compilation.shader_surface_entry_usage[i];
    for (std::size_t op = 0u; op < entry.node_types_used.size(); ++op) {
      result.node_types_used[op] =
          result.node_types_used[op] || entry.node_types_used[op];
    }
    result.peak_stack_usage =
        std::max(result.peak_stack_usage, entry.peak_stack_usage);
    result.noise_usage.shape_mask |= entry.noise_usage.shape_mask;
  }
  if (domain == PathCyclesSvmShaderDomain::shadow_surface ||
      domain == PathCyclesSvmShaderDomain::light_emission) {
    // Preserve the pre-optimization AST Stack array extent checked by
    // test_luisa_cycles_svm_stack_extent; LLVM SROA drops unused lanes.
    result.peak_stack_usage = table_usage.peak_stack_usage;
  }
  return result;
}

abi::ShaderEntryUsage cycles_svm_masked_entry_usage(
    const LuisaSceneData &scene,
    std::span<const std::uint8_t> active_shader_mask) noexcept {
  const auto &compilation = scene.cycles_svm->compilation;
  const auto table_usage =
      compilation.table.usage_for(abi::SHADER_TYPE_SURFACE);
  const auto count = static_cast<std::size_t>(compilation.table.shader_count);
  if (!cycles_svm_has_domain_metadata(scene) ||
      active_shader_mask.size() != count) {
    return table_usage;
  }
  abi::ShaderEntryUsage result{};
  result.node_types_used[abi::NODE_SHADER_JUMP] = true;
  result.node_types_used[abi::NODE_END] = true;
  result.noise_usage = abi::NoiseUsage::none();
  for (std::size_t i = 0u; i < count; ++i) {
    if (active_shader_mask[i] == 0u) {
      continue;
    }
    const auto &entry = compilation.shader_surface_entry_usage[i];
    for (std::size_t op = 0u; op < entry.node_types_used.size(); ++op) {
      result.node_types_used[op] =
          result.node_types_used[op] || entry.node_types_used[op];
    }
    result.peak_stack_usage =
        std::max(result.peak_stack_usage, entry.peak_stack_usage);
    result.noise_usage.shape_mask |= entry.noise_usage.shape_mask;
  }
  return result;
}

std::vector<std::uint8_t> cycles_svm_domain_shader_mask(
    const LuisaSceneData &scene,
    PathCyclesSvmShaderDomain domain) noexcept {
  if (!cycles_svm_has_domain_metadata(scene) ||
      domain == PathCyclesSvmShaderDomain::all_shaders) {
    return {};
  }
  const auto count = scene.cycles_svm->compilation.table.shader_count;
  std::vector<std::uint8_t> mask(count, 0u);
  for (std::uint32_t i = 0u; i < count; ++i) {
    mask[i] = shader_in_domain(scene, i, domain) ? 1u : 0u;
  }
  return mask;
}

PathCyclesSvmKernelGlobals::PathCyclesSvmKernelGlobals(
    std::shared_ptr<LuisaSceneData> scene,
    const SurfacePopulationContext &context,
    std::span<const std::uint8_t> explicit_shader_mask,
    bool force_direct_textures,
    UInt *vt_miss_page,
    bool spectral_rendering) noexcept
    : PathCyclesSvmKernelGlobals(
          std::move(scene), context.parameters, context.camera_projection,
          context.query.reflective_caustics,
          context.query.refractive_caustics,
          PathCyclesSvmShaderDomain::surface_only,
          explicit_shader_mask,
          force_direct_textures,
          vt_miss_page != nullptr ? vt_miss_page : context.vt_miss_page,
          nullptr,
          spectral_rendering) {
  _ambient_occlusion = context.ambient_occlusion;
}

PathCyclesSvmKernelGlobals::PathCyclesSvmKernelGlobals(
    std::shared_ptr<LuisaSceneData> scene,
    const Var<RenderKernelParameters> &parameters,
    CameraProjection camera_projection, Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics,
    PathCyclesSvmShaderDomain domain,
    std::span<const std::uint8_t> explicit_shader_mask,
    bool force_direct_textures,
    UInt *vt_miss_page,
    const UInt *vt_preflight_weight,
    bool spectral_rendering) noexcept
    : _scene{std::move(scene)},
      _parameters{parameters},
      _camera_projection{camera_projection},
      _spectral_rendering{spectral_rendering},
      _caustics_reflective{reflective_caustics},
      _caustics_refractive{refractive_caustics},
      _camera_to_world{parameters.camera_transform},
      _world_to_camera{parameters.camera_inverse_transform},
      _force_direct_textures{force_direct_textures},
      _vt_miss_page{vt_miss_page},
      _vt_preflight_weight{vt_preflight_weight} {
  LUISA_ASSERT(_scene && _scene->cycles_svm &&
                   _scene->cycles_svm->geometry &&
                   _scene->cycles_svm->objects,
               "Native Cycles SVM surface requires a finalized scene image.");
  _has_shadow_terminator_shading_offset =
      _scene->cycles_svm->objects->image.has_shadow_terminator_shading_offset();
  _has_curves = !_scene->curve_geometries.empty();
  const auto &compilation = _scene->cycles_svm->compilation;
  if (compilation.table.valid &&
      compilation.shader_metadata.size() == compilation.table.shader_count) {
    const bool has_explicit_mask =
        cycles_svm_has_domain_metadata(*_scene) &&
        explicit_shader_mask.size() ==
            static_cast<std::size_t>(compilation.table.shader_count);
    const bool filter_domain =
        has_explicit_mask ||
        (cycles_svm_has_domain_metadata(*_scene) &&
         domain != PathCyclesSvmShaderDomain::all_shaders);
    bool thin_film = false;
    auto closure_u = abi::ClosureUsage::none();
    std::uint32_t mix_mask = 0u;
    std::uint32_t nm_space_mask = 0u;
    bool nm_orig_base = false;
    bool noise_dist = false;
    bool noise_col = false;
    std::uint32_t tc_mask = 0u;
    bool bump_obj = false;
    bool curves_extrap = false;
    std::uint64_t math_mask = 0ull;
    std::uint64_t vec_math_mask = 0ull;
    std::uint32_t map_type_mask = 0u;
    bool map_rot = false;
    std::uint32_t mr_type_mask = 0u;
    std::uint32_t vor_dim_mask = 0u;
    std::uint32_t vor_feat_mask = 0u;
    std::uint32_t vor_met_mask = 0u;
    bool vor_detail = false;
    bool vor_norm = false;
    bool vor_color = false;
    bool vor_pos = false;
    bool vor_rad = false;
    std::uint32_t wave_ty_mask = 0u;
    std::uint32_t wave_prof_mask = 0u;
    bool wave_dist = false;
    const auto count = compilation.table.shader_count;
    for (std::uint32_t i = 0u; i < count; ++i) {
      if (has_explicit_mask) {
        if (explicit_shader_mask[i] == 0u) {
          continue;
        }
      } else if (filter_domain && !shader_in_domain(*_scene, i, domain)) {
        continue;
      }
      const auto &meta = compilation.shader_metadata[i];
      thin_film = thin_film || meta.may_have_thin_film;
      closure_u.merge(meta.surface_closure_usage);
      mix_mask |= meta.mix_blend_mask;
      nm_space_mask |= meta.normal_map_space_mask;
      nm_orig_base = nm_orig_base || meta.normal_map_uses_original_base;
      noise_dist = noise_dist || meta.noise_may_have_distortion;
      noise_col = noise_col || meta.noise_may_need_color;
      tc_mask |= meta.tex_coord_mask;
      bump_obj = bump_obj || meta.bump_uses_object_space;
      curves_extrap = curves_extrap || meta.curves_may_extrapolate;
      math_mask |= meta.math_type_mask;
      vec_math_mask |= meta.vector_math_type_mask;
      map_type_mask |= meta.mapping_type_mask;
      map_rot = map_rot || meta.mapping_may_rotate;
      mr_type_mask |= meta.map_range_type_mask;
      vor_dim_mask |= meta.voronoi_dimension_mask;
      vor_feat_mask |= meta.voronoi_feature_mask;
      vor_met_mask |= meta.voronoi_metric_mask;
      vor_detail = vor_detail || meta.voronoi_may_have_detail;
      vor_norm = vor_norm || meta.voronoi_may_normalize;
      vor_color = vor_color || meta.voronoi_may_need_color;
      vor_pos = vor_pos || meta.voronoi_may_need_position;
      vor_rad = vor_rad || meta.voronoi_may_need_radius;
      wave_ty_mask |= meta.wave_type_mask;
      wave_prof_mask |= meta.wave_profile_mask;
      wave_dist = wave_dist || meta.wave_may_have_distortion;
    }
    _may_have_thin_film = thin_film;
    _closure_usage = filter_domain ? closure_u : compilation.surface_closure_usage;
    _mix_blend_mask = mix_mask;
    _normal_map_usage = {
        .space_mask = nm_space_mask,
        .uses_original_base = nm_orig_base};
    _noise_static_options = {
        .may_have_distortion = noise_dist,
        .may_need_color = noise_col};
    _tex_coord_mask = tc_mask;
    _bump_uses_object_space = bump_obj;
    _curves_may_extrapolate = curves_extrap;
    _math_type_mask = math_mask;
    _vector_math_type_mask = vec_math_mask;
    _mapping_static_options = {
        .type_mask = map_type_mask,
        .may_rotate = map_rot};
    _map_range_type_mask = mr_type_mask;
    _voronoi_static_options = {
        .dimension_mask = vor_dim_mask,
        .feature_mask = vor_feat_mask,
        .metric_mask = vor_met_mask,
        .may_have_detail = vor_detail,
        .may_normalize = vor_norm,
        .may_need_color = vor_color,
        .may_need_position = vor_pos,
        .may_need_radius = vor_rad};
    _wave_static_options = {
        .type_mask = wave_ty_mask,
        .profile_mask = wave_prof_mask,
        .may_have_distortion = wave_dist};
  }
  const auto &geom_img = _scene->cycles_svm->geometry->image;
  const bool has_curves = !geom_img.curves.empty();
  const bool has_points = !geom_img.attributes.points.empty();
  std::uint32_t elem_mask = 0u;
  std::uint32_t type_mask = 0u;
  bool has_undisp = false;
  bool has_undisp_pos = false;
  bool has_float4 = false;
  for (const auto &entry : geom_img.attributes.attribute_map) {
    if (entry.id == static_cast<std::uint64_t>(abi::ATTR_STD_NONE)) {
      continue;
    }
    const auto elem = static_cast<std::uint32_t>(entry.element);
    if (elem == static_cast<std::uint32_t>(abi::ATTR_ELEMENT_NONE)) {
      continue;
    }
    elem_mask |= elem;
    const auto ty = static_cast<std::uint32_t>(entry.type);
    const std::uint32_t ty_bit = (ty < 32u) ? (1u << ty) : 0u;
    type_mask |= ty_bit;
    const bool is_float4_entry =
        ((ty == static_cast<std::uint32_t>(abi::NODE_ATTR_FLOAT4)) ||
         (ty == static_cast<std::uint32_t>(abi::NODE_ATTR_RGBA))) &&
        ((elem & static_cast<std::uint32_t>(abi::ATTR_ELEMENT_IS_BYTE)) == 0u);
    if (is_float4_entry) {
      has_float4 = true;
    }
    if (entry.id == static_cast<std::uint64_t>(abi::ATTR_STD_NORMAL_UNDISPLACED)) {
      has_undisp = true;
    }
    if (entry.id == static_cast<std::uint64_t>(abi::ATTR_STD_POSITION_UNDISPLACED)) {
      has_undisp_pos = true;
    }
    auto &per_id = _attribute_summaries_by_id[entry.id];
    if (per_id.element_mask == ~0u && per_id.type_mask == ~0u) {
      per_id = {
          .element_mask = 0u,
          .type_mask = 0u,
          .has_curves = has_curves,
          .has_points = has_points,
          .has_undisplaced_normal = false,
          .has_undisplaced_position = false,
          .has_float4 = false};
    }
    per_id.element_mask |= elem;
    per_id.type_mask |= ty_bit;
    if (is_float4_entry) {
      per_id.has_float4 = true;
    }
  }
  for (auto &[id, per_id] : _attribute_summaries_by_id) {
    (void)id;
    per_id.has_undisplaced_normal = has_undisp;
    per_id.has_undisplaced_position = has_undisp_pos;
  }
  _has_attribute_summary_table = true;
  _surface_attribute_summary = {
      .element_mask = elem_mask,
      .type_mask = type_mask,
      .has_curves = has_curves,
      .has_points = has_points,
      .has_undisplaced_normal = has_undisp,
      .has_undisplaced_position = has_undisp_pos,
      .has_float4 = has_float4};

  const auto &bindings = _scene->cycles_svm->image_bindings;
  _has_bicubic_images = false;
  _has_failed_images = false;
  _has_compact_channel_images = false;
  _contiguous_texture_slots = !bindings.empty();
  _base_texture_slot = bindings.empty() ? 0u : bindings.front().texture_slot;
  _image_binding_count = static_cast<std::uint32_t>(bindings.size());
  for (std::size_t i = 0u; i < bindings.size(); ++i) {
    const auto &b = bindings[i];
    if ((b.sampler & cycles_svm_image_interpolation_mask) == 2u) {
      _has_bicubic_images = true;
    }
    if ((b.sampler & cycles_svm_image_load_failed_flag) != 0u) {
      _has_failed_images = true;
    }
    if ((b.sampler & cycles_svm_image_channel_mode_mask) != 0u) {
      _has_compact_channel_images = true;
    }
    if (b.texture_slot != _base_texture_slot + static_cast<std::uint32_t>(i)) {
      _contiguous_texture_slots = false;
    }
  }
}

Bool PathCyclesSvmKernelGlobals::caustics_reflective() const noexcept {
  return _caustics_reflective;
}

Bool PathCyclesSvmKernelGlobals::caustics_refractive() const noexcept {
  return _caustics_refractive;
}

Bool PathCyclesSvmKernelGlobals::background_use_sun_guiding() const noexcept {
  return _scene->background_guided_sun_weight > 0.0f;
}

Float PathCyclesSvmKernelGlobals::object_shadow_terminator_shading_offset(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(1.0f, record.shadow_terminator_shading_offset,
                object != svm::object_none);
}

bool PathCyclesSvmKernelGlobals::has_shadow_terminator_shading_offset() const noexcept {
  return _has_shadow_terminator_shading_offset;
}

bool PathCyclesSvmKernelGlobals::may_have_thin_film() const noexcept {
  return _may_have_thin_film;
}

bool PathCyclesSvmKernelGlobals::is_spectral() const noexcept {
  return _spectral_rendering;
}

const cycles_spectrum::SpectralTablesGpu *
PathCyclesSvmKernelGlobals::spectral_tables() const noexcept {
  return _spectral_rendering && _scene ? _scene->spectral_tables.get()
                                       : nullptr;
}

bool PathCyclesSvmKernelGlobals::has_dispersive_materials() const noexcept {
  return _spectral_rendering && _scene && _scene->has_dispersive_materials;
}

Float PathCyclesSvmKernelGlobals::shader_dispersion_abbe(
    Expr<std::uint32_t> shader_id) const noexcept {
  if (!has_dispersive_materials() || !_scene->cycles_svm ||
      !_scene->cycles_svm->dispersion_abbe_buffer) {
    return 0.0f;
  }
  const auto count =
      static_cast<std::uint32_t>(_scene->cycles_svm->dispersion_abbe_values.size());
  if (count == 0u) {
    return 0.0f;
  }
  const Expr<Buffer<float>> abbes{*_scene->cycles_svm->dispersion_abbe_buffer};
  const auto index = shader_id & svm::shader_mask;
  return select(0.0f, abbes->read(min(index, count - 1u)), index < count);
}

abi::ClosureUsage PathCyclesSvmKernelGlobals::closure_usage() const noexcept {
  return _closure_usage;
}

svm::SurfaceAttributeSummary
PathCyclesSvmKernelGlobals::surface_attribute_summary() const noexcept {
  return _surface_attribute_summary;
}

svm::SurfaceAttributeSummary
PathCyclesSvmKernelGlobals::attribute_summary_for_id(
    std::uint64_t id) const noexcept {
  if (!_has_attribute_summary_table) {
    return _surface_attribute_summary;
  }
  if (const auto it = _attribute_summaries_by_id.find(id);
      it != _attribute_summaries_by_id.end()) {
    return it->second;
  }
  return {
      .element_mask = 0u,
      .type_mask = 0u,
      .has_curves = _surface_attribute_summary.has_curves,
      .has_points = _surface_attribute_summary.has_points,
      .has_undisplaced_normal =
          _surface_attribute_summary.has_undisplaced_normal,
      .has_undisplaced_position =
          _surface_attribute_summary.has_undisplaced_position,
      .has_float4 = false};
}

svm::NormalMapUsage
PathCyclesSvmKernelGlobals::normal_map_usage() const noexcept {
  return _normal_map_usage;
}

std::uint32_t PathCyclesSvmKernelGlobals::mix_blend_mask() const noexcept {
  return _mix_blend_mask;
}

svm::NoiseStaticOptions
PathCyclesSvmKernelGlobals::noise_static_options() const noexcept {
  return _noise_static_options;
}

std::uint32_t PathCyclesSvmKernelGlobals::tex_coord_mask() const noexcept {
  return _tex_coord_mask;
}

bool PathCyclesSvmKernelGlobals::bump_uses_object_space() const noexcept {
  return _bump_uses_object_space;
}

bool PathCyclesSvmKernelGlobals::curves_may_extrapolate() const noexcept {
  return _curves_may_extrapolate;
}

std::uint64_t PathCyclesSvmKernelGlobals::math_type_mask() const noexcept {
  return _math_type_mask;
}

std::uint64_t PathCyclesSvmKernelGlobals::vector_math_type_mask() const noexcept {
  return _vector_math_type_mask;
}

svm::MappingStaticOptions
PathCyclesSvmKernelGlobals::mapping_static_options() const noexcept {
  return _mapping_static_options;
}

std::uint32_t PathCyclesSvmKernelGlobals::map_range_type_mask() const noexcept {
  return _map_range_type_mask;
}

svm::VoronoiStaticOptions
PathCyclesSvmKernelGlobals::voronoi_static_options() const noexcept {
  return _voronoi_static_options;
}

svm::WaveStaticOptions
PathCyclesSvmKernelGlobals::wave_static_options() const noexcept {
  return _wave_static_options;
}

bool PathCyclesSvmKernelGlobals::has_curves() const noexcept {
  return _has_curves;
}

std::optional<Float> PathCyclesSvmKernelGlobals::object_volume_density(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(1.0f, record.volume_density, object != svm::object_none);
}

Float PathCyclesSvmKernelGlobals::ies(
    Expr<std::uint32_t> offset) const noexcept {
  if (_scene->cycles_svm->ies_buffer) {
    const Expr<Buffer<float>> values{*_scene->cycles_svm->ies_buffer};
    return values->read(offset);
  }
  dsl::unreachable("NODE_IES reached without a Cycles IES table");
  return 0.0f;
}

const svm::InfoServices *
PathCyclesSvmKernelGlobals::info_services() const noexcept {
  return this;
}

const svm::AmbientOcclusionServices *
PathCyclesSvmKernelGlobals::ambient_occlusion_services() const noexcept {
  return (_ambient_occlusion != nullptr &&
          _ambient_occlusion->traversal != nullptr)
             ? this
             : nullptr;
}

std::span<const abi::OslScriptBinding>
PathCyclesSvmKernelGlobals::osl_scripts() const noexcept {
  if (_scene->cycles_svm) {
    return _scene->cycles_svm->compilation.table.osl_scripts;
  }
  return {};
}

Float PathCyclesSvmKernelGlobals::cycles_bsdf_data(
    Expr<std::uint32_t> index) const noexcept {
  return _scene->cycles_bsdf_table_buffer->read(index);
}

svm::TriangleVertices PathCyclesSvmKernelGlobals::triangle_vertices(
    Expr<std::uint32_t> object, Expr<std::uint32_t> prim) const noexcept {
  const auto record = object_record(*_scene, object);
  const auto indices = triangle_vertex_indices(prim);
  const auto base = record.position_offset.cast<std::uint32_t>();
  const auto &vertices = _scene->cycles_svm->geometry->triangle_vertex_buffer;
  return {.v0 = unpack(vertices->read(base + indices.x)),
          .v1 = unpack(vertices->read(base + indices.y)),
          .v2 = unpack(vertices->read(base + indices.z))};
}

svm::TriangleVertices
PathCyclesSvmKernelGlobals::motion_triangle_vertices(
    Expr<std::uint32_t> object, Expr<std::uint32_t> prim,
    Expr<float>) const noexcept {
  dsl::unreachable(
      "motion triangle SVM service reached in a static-scene kernel");
  return triangle_vertices(object, prim);
}

Float3 PathCyclesSvmKernelGlobals::film_rgb_to_y() const noexcept {
  return host_vector(rgb_to_y_coefficients(_scene->shader_color_space));
}

Float3 PathCyclesSvmKernelGlobals::primitive_tangent(
    const svm::ShaderData &shader_data) const noexcept {
  Float3 result = svm_detail::normalize_cycles(shader_data.dPdu);
  if (attribute_summary_for_id(
          static_cast<std::uint64_t>(abi::ATTR_STD_GENERATED))
          .element_mask == 0u) {
    return result;
  }
  const auto generated = svm::find_attribute(
      *this, shader_data, static_cast<luisa::ulong>(abi::ATTR_STD_GENERATED));
  const auto curve_or_point =
      (shader_data.type & (svm::primitive_curve | svm::primitive_point)) != 0u;
  $if((!curve_or_point) & attribute_found(generated)) {
    auto value = svm::primitive_surface_attribute_float3(
        *this, shader_data, generated);
    value = make_float3(-(value.y - 0.5f), value.x - 0.5f, 0.0f);
    const auto object = object_record(*_scene, shader_data.object);
    value = svm_detail::normalize_cycles(
        cycles_transform::direction_transposed(
            unpack_transform(object.itfm), value));
    result = cross(shader_data.N,
                   svm_detail::normalize_cycles(cross(value, shader_data.N)));
  };
  return result;
}

svm::Dual3 PathCyclesSvmKernelGlobals::primitive_tangent_derivative(
    const svm::ShaderData &shader_data) const noexcept {
  svm::Dual3 result{.val = svm_detail::normalize_cycles(shader_data.dPdu),
                    .dx = make_float3(0.0f),
                    .dy = make_float3(0.0f)};
  if (attribute_summary_for_id(
          static_cast<std::uint64_t>(abi::ATTR_STD_GENERATED))
          .element_mask == 0u) {
    return result;
  }
  const auto generated = svm::find_attribute(
      *this, shader_data, static_cast<luisa::ulong>(abi::ATTR_STD_GENERATED));
  const auto curve_or_point =
      (shader_data.type & (svm::primitive_curve | svm::primitive_point)) != 0u;
  $if((!curve_or_point) & attribute_found(generated)) {
    auto value = svm::primitive_surface_attribute_float3_derivative(
        *this, shader_data, generated);
    value.val = make_float3(-(value.val.y - 0.5f), value.val.x - 0.5f, 0.0f);
    value.dx = make_float3(-value.dx.y, value.dx.x, 0.0f);
    value.dy = make_float3(-value.dy.y, value.dy.x, 0.0f);
    const auto object = object_record(*_scene, shader_data.object);
    value = svm_detail::normalize_dual_cycles(transform_direction_transposed(
        unpack_transform(object.itfm), value));
    result = dual_cross_left(
        shader_data.N,
        svm_detail::normalize_dual_cycles(
            dual_cross_right(value, shader_data.N)));
  };
  return result;
}

UInt PathCyclesSvmKernelGlobals::object_attribute_map_offset(
    Expr<std::uint32_t> object) const noexcept {
  return object_record(*_scene, object).attribute_map_offset;
}

Var<abi::AttributeMap> PathCyclesSvmKernelGlobals::attribute_map(
    Expr<std::uint32_t> offset) const noexcept {
  return _scene->cycles_svm->geometry->attribute_map_buffer->read(offset);
}

std::optional<luisa::compute::UInt2>
PathCyclesSvmKernelGlobals::find_attribute_map_offset(
    Expr<std::uint32_t> start_offset,
    Expr<luisa::ulong> id) const noexcept {
  static const Callable<luisa::uint2(
      Buffer<abi::AttributeMap>, luisa::uint, luisa::ulong)>
      find_attr_callable = [](BufferVar<abi::AttributeMap> attr_maps,
                              UInt attr_offset, ULong target_id) noexcept {
        auto attr_map = attr_maps.read(attr_offset);
        UInt found = 1u;
        $while(attr_map.id != target_id) {
          $if(attr_map.id == static_cast<luisa::ulong>(abi::ATTR_STD_NONE)) {
            $if(attr_map.element.cast<std::uint32_t>() == 0u) {
              found = 0u;
              $break;
            }
            $else { attr_offset = attr_map.offset.cast<std::uint32_t>(); };
          }
          $else {
            attr_offset += static_cast<std::uint32_t>(abi::ATTR_PRIM_TYPES);
          };
          attr_map = attr_maps.read(attr_offset);
        };
        return make_uint2(found, attr_offset);
      };
  return find_attr_callable(
      _scene->cycles_svm->geometry->attribute_map_buffer, start_offset, id);
}

std::optional<svm::SurfaceAttributeBuffers>
PathCyclesSvmKernelGlobals::surface_attribute_buffers() const noexcept {
  if (!_scene || !_scene->cycles_svm || !_scene->cycles_svm->geometry ||
      !_scene->cycles_svm->objects) {
    return std::nullopt;
  }
  const auto &geom = *_scene->cycles_svm->geometry;
  const auto &cs = _scene->shader_color_space;
  return svm::SurfaceAttributeBuffers{
      .object_buffer = &_scene->cycles_svm->objects->object_buffer,
      .attribute_map_buffer = &geom.attribute_map_buffer,
      .triangle_index_buffer = &geom.triangle_index_buffer,
      .curve_buffer = &geom.curve_buffer,
      .attribute_float_buffer = &geom.attribute_float_buffer,
      .attribute_float2_buffer = &geom.attribute_float2_buffer,
      .attribute_float3_buffer = &geom.attribute_float3_buffer,
      .attribute_float4_buffer = &geom.attribute_float4_buffer,
      .attribute_uchar4_buffer = &geom.attribute_uchar4_buffer,
      .attribute_normal_buffer = &geom.attribute_normal_buffer,
      .is_rec709 = is_rec709(cs),
      .rec709_to_r = luisa::make_float3(cs.rec709_to_r.x, cs.rec709_to_r.y, cs.rec709_to_r.z),
      .rec709_to_g = luisa::make_float3(cs.rec709_to_g.x, cs.rec709_to_g.y, cs.rec709_to_g.z),
      .rec709_to_b = luisa::make_float3(cs.rec709_to_b.x, cs.rec709_to_b.y, cs.rec709_to_b.z),
  };
}

bool PathCyclesSvmKernelGlobals::prefers_outlined_loops() const noexcept {
  return _scene->device.backend_name() == "simd" &&
         _scene->device.compute_warp_size() > 1u;
}

Float PathCyclesSvmKernelGlobals::attribute_float(
    Expr<std::int32_t> offset) const noexcept {
  return _scene->cycles_svm->geometry->attribute_float_buffer->read(
      offset.cast<std::uint32_t>());
}

Float2 PathCyclesSvmKernelGlobals::attribute_float2(
    Expr<std::int32_t> offset) const noexcept {
  const auto value =
      _scene->cycles_svm->geometry->attribute_float2_buffer->read(
          offset.cast<std::uint32_t>());
  return make_float2(value.x, value.y);
}

Var<abi::packed_float3> PathCyclesSvmKernelGlobals::attribute_float3(
    Expr<std::int32_t> offset) const noexcept {
  return _scene->cycles_svm->geometry->attribute_float3_buffer->read(
      offset.cast<std::uint32_t>());
}

Float4 PathCyclesSvmKernelGlobals::attribute_float4(
    Expr<std::int32_t> offset) const noexcept {
  return unpack(_scene->cycles_svm->geometry->attribute_float4_buffer->read(
      offset.cast<std::uint32_t>()));
}

Var<abi::uchar4> PathCyclesSvmKernelGlobals::attribute_uchar4(
    Expr<std::int32_t> offset) const noexcept {
  return _scene->cycles_svm->geometry->attribute_uchar4_buffer->read(
      offset.cast<std::uint32_t>());
}

Var<abi::packed_normal> PathCyclesSvmKernelGlobals::attribute_normal(
    Expr<std::int32_t> offset) const noexcept {
  return _scene->cycles_svm->geometry->attribute_normal_buffer->read(
      offset.cast<std::uint32_t>());
}

UInt3 PathCyclesSvmKernelGlobals::triangle_vertex_indices(
    Expr<std::uint32_t> prim) const noexcept {
  return unpack(
      _scene->cycles_svm->geometry->triangle_index_buffer->read(prim));
}

Int PathCyclesSvmKernelGlobals::object_normal_offset(
    Expr<std::uint32_t> object) const noexcept {
  return object_record(*_scene, object).normal_offset;
}

UInt PathCyclesSvmKernelGlobals::object_num_geom_steps(
    Expr<std::uint32_t> object) const noexcept {
  return object_record(*_scene, object).num_geom_steps.cast<std::uint32_t>();
}

Int PathCyclesSvmKernelGlobals::object_num_vertices(
    Expr<std::uint32_t> object) const noexcept {
  return object_record(*_scene, object).numverts;
}

Int PathCyclesSvmKernelGlobals::object_num_primitives(
    Expr<std::uint32_t> object) const noexcept {
  return object_record(*_scene, object).numprims;
}

Float3 PathCyclesSvmKernelGlobals::object_dupli_generated(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(make_float3(0.0f), unpack(record.dupli_generated),
                object != svm::object_none);
}

Float3 PathCyclesSvmKernelGlobals::object_dupli_uv(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(make_float3(0.0f),
                make_float3(record.dupli_uv.x, record.dupli_uv.y, 0.0f),
                object != svm::object_none);
}

UInt PathCyclesSvmKernelGlobals::camera_type() const noexcept {
  switch (_camera_projection) {
    case CameraProjection::perspective:
      return svm::camera_perspective;
    case CameraProjection::orthographic:
      return svm::camera_orthographic;
    case CameraProjection::panorama:
      return svm::camera_panorama;
  }
  return svm::camera_custom;
}

Float PathCyclesSvmKernelGlobals::camera_width() const noexcept {
  return _parameters.full_width.cast<float>();
}

Float PathCyclesSvmKernelGlobals::camera_height() const noexcept {
  return _parameters.full_height.cast<float>();
}

Float3 PathCyclesSvmKernelGlobals::camera_world_to_ndc(
    const svm::ShaderData &shader_data,
    Expr<luisa::float3> position) const noexcept {
  if (_camera_projection != CameraProjection::panorama) {
    Float3 P = position;
    if (_camera_projection == CameraProjection::perspective) {
      $if(shader_data.object == svm::object_none) { P += _camera_to_world[3u].xyz(); };
    }
    return cycles_transform::perspective(_parameters.camera_world_to_ndc, P);
  }

  Float3 direction = cycles_transform::point(_world_to_camera, position);
  $if(shader_data.object == svm::object_none) {
    direction = cycles_transform::direction(_world_to_camera, position);
  };
  direction = svm_detail::safe_normalize_cycles(direction);
  const auto longitude = atan2(direction.x, -direction.z);
  const auto latitude = asin(clamp(direction.y, -1.0f, 1.0f));
  return make_float3(longitude * (0.5f / 3.14159265358979323846f) + 0.5f,
                     latitude * (1.0f / 3.14159265358979323846f) + 0.5f,
                     0.0f);
}

Var<abi::KernelCurve> PathCyclesSvmKernelGlobals::curve(
    Expr<std::uint32_t> prim) const noexcept {
  return _scene->cycles_svm->geometry->curve_buffer->read(prim);
}

Int PathCyclesSvmKernelGlobals::object_position_offset(
    Expr<std::uint32_t> object) const noexcept {
  return object_record(*_scene, object).position_offset;
}

Float4 PathCyclesSvmKernelGlobals::curve_key(
    Expr<std::int32_t> key) const noexcept {
  return unpack(_scene->cycles_svm->geometry->curve_key_buffer->read(
      key.cast<std::uint32_t>()));
}

Bool PathCyclesSvmKernelGlobals::film_is_rec709() const noexcept {
  return is_rec709(_scene->shader_color_space);
}

Float3 PathCyclesSvmKernelGlobals::film_xyz_to_r() const noexcept {
  return host_vector(_scene->shader_color_space.xyz_to_r);
}

Float3 PathCyclesSvmKernelGlobals::film_xyz_to_g() const noexcept {
  return host_vector(_scene->shader_color_space.xyz_to_g);
}

Float3 PathCyclesSvmKernelGlobals::film_xyz_to_b() const noexcept {
  return host_vector(_scene->shader_color_space.xyz_to_b);
}

Float3 PathCyclesSvmKernelGlobals::film_rec709_to_r() const noexcept {
  return host_vector(_scene->shader_color_space.rec709_to_r);
}

Float3 PathCyclesSvmKernelGlobals::film_rec709_to_g() const noexcept {
  return host_vector(_scene->shader_color_space.rec709_to_g);
}

Float3 PathCyclesSvmKernelGlobals::film_rec709_to_b() const noexcept {
  return host_vector(_scene->shader_color_space.rec709_to_b);
}

Float3
PathCyclesSvmKernelGlobals::object_inverse_position_transform_if_object(
    const svm::ShaderData &shader_data,
    Expr<luisa::float3> value) const noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<abi::KernelObject> objects, UInt object,
                       Float3 p) noexcept {
      Float3 out = p;
      $if(object != svm::object_none) {
        const auto t = objects.read(object).itfm;
        out = make_float3(
            dot(make_float3(t.x.x, t.x.y, t.x.z), p) + t.x.w,
            dot(make_float3(t.y.x, t.y.y, t.y.z), p) + t.y.w,
            dot(make_float3(t.z.x, t.z.y, t.z.z), p) + t.z.w);
      };
      return out;
    }};
    result.set_name("cycles_obj_inv_pos_if_obj");
    return result;
  }();
  return callable(_scene->cycles_svm->objects->object_buffer,
                  shader_data.object, value);
}

svm::Dual3 PathCyclesSvmKernelGlobals::
    object_inverse_position_transform_if_object_derivative(
        const svm::ShaderData &shader_data,
        const svm::Dual3 &value) const noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<abi::KernelObject> objects, UInt object,
                       Float3 val, Float3 dx, Float3 dy) noexcept {
      Var<KernelObjDual3> out;
      out.val = val;
      out.dx = dx;
      out.dy = dy;
      $if(object != svm::object_none) {
        const auto t = objects.read(object).itfm;
        const Float3 r0 = make_float3(t.x.x, t.x.y, t.x.z);
        const Float3 r1 = make_float3(t.y.x, t.y.y, t.y.z);
        const Float3 r2 = make_float3(t.z.x, t.z.y, t.z.z);
        out.val = make_float3(dot(r0, val) + t.x.w, dot(r1, val) + t.y.w,
                              dot(r2, val) + t.z.w);
        out.dx = make_float3(dot(r0, dx), dot(r1, dx), dot(r2, dx));
        out.dy = make_float3(dot(r0, dy), dot(r1, dy), dot(r2, dy));
      };
      return out;
    }};
    result.set_name("cycles_obj_inv_pos_deriv_if_obj");
    return result;
  }();
  const auto res = callable(_scene->cycles_svm->objects->object_buffer,
                            shader_data.object, value.val, value.dx, value.dy);
  return {.val = res.val, .dx = res.dx, .dy = res.dy};
}

Float3 PathCyclesSvmKernelGlobals::object_inverse_position_transform(
    const svm::ShaderData &shader_data,
    Expr<luisa::float3> value) const noexcept {
  static auto callable = []() noexcept {
    Callable result{[](BufferVar<abi::KernelObject> objects, UInt object,
                       Float3 p) noexcept {
      const UInt index = select(0u, object, object != svm::object_none);
      const auto t = objects.read(index).itfm;
      return make_float3(
          dot(make_float3(t.x.x, t.x.y, t.x.z), p) + t.x.w,
          dot(make_float3(t.y.x, t.y.y, t.y.z), p) + t.y.w,
          dot(make_float3(t.z.x, t.z.y, t.z.z), p) + t.z.w);
    }};
    result.set_name("cycles_obj_inv_pos");
    return result;
  }();
  return callable(_scene->cycles_svm->objects->object_buffer,
                  shader_data.object, value);
}

Float4 PathCyclesSvmKernelGlobals::kernel_image_interp_with_udim(
    svm::ShaderData &, Expr<std::int32_t> image_texture_id,
    const svm::Dual2 &uv) const noexcept {
  const auto &vt = _scene->virtual_texture_cache;
  auto sample_uv = uv.val;
  sample_uv.y = 1.0f - sample_uv.y;
  const auto duv_dx = make_float2(uv.dx.x, -uv.dx.y);
  const auto duv_dy = make_float2(uv.dy.x, -uv.dy.y);
  if (_vt_preflight_weight != nullptr) {
    if (vt && !_force_direct_textures && !vt->eager_fallback_active()) {
      if (vt->mip_mode() == LuisaTextureMipMode::isotropic) {
        vt->record_preflight_demand_2d(
            image_texture_id, sample_uv, duv_dx, duv_dy,
            *_vt_preflight_weight);
      } else {
        vt->record_preflight_demand_2d(
            image_texture_id, sample_uv, *_vt_preflight_weight);
      }
    }
    return make_float4(0.5f, 0.5f, 0.5f, 0.5f);
  }
  const bool use_vt =
      vt && !_force_direct_textures && _vt_miss_page != nullptr &&
      !vt->eager_fallback_active();
  const bool vt_isotropic =
      use_vt && (vt->mip_mode() == LuisaTextureMipMode::isotropic);
  const bool direct_isotropic =
      (_scene->direct_texture_mip_mode == LuisaTextureMipMode::isotropic) ||
      vt_isotropic;
  if (const auto cid = svm_detail::try_const_int(image_texture_id)) {
    constexpr auto magenta = luisa::make_float4(1.0f, 0.0f, 1.0f, 1.0f);
    if (*cid < 0 ||
        static_cast<std::size_t>(*cid) >=
            _scene->cycles_svm->image_bindings.size()) {
      return make_float4(magenta);
    }
    const auto &b =
        _scene->cycles_svm->image_bindings[static_cast<std::size_t>(*cid)];
    if ((b.sampler & cycles_svm_image_load_failed_flag) != 0u) {
      return make_float4(magenta);
    }
    if (use_vt && vt->is_virtual_binding(*cid)) {
      return vt_isotropic
                 ? vt->sample_2d_const_mip(
                       *cid, sample_uv, duv_dx, duv_dy, _vt_miss_page, false)
                 : vt->sample_2d_const(*cid, sample_uv, _vt_miss_page, false);
    }
    auto texture = _scene->texture_heap->tex2d(b.texture_slot);
    const std::uint32_t ch_mode =
        (b.sampler & cycles_svm_image_channel_mode_mask) >>
        cycles_svm_image_channel_mode_shift;
    const auto expand_channels = [&](Float4 s) noexcept -> Float4 {
      if (ch_mode == cycles_svm_image_channel_mode_gray_opaque) {
        return make_float4(s.x, s.x, s.x, 1.0f);
      }
      if (ch_mode == cycles_svm_image_channel_mode_gray_alpha) {
        return make_float4(s.x, s.x, s.x, s.y);
      }
      return s;
    };
    if (direct_isotropic && _scene->direct_image_mip_descs) {
      const auto mip_desc = _scene->direct_image_mip_descs->read(
          static_cast<luisa::uint>(*cid));
      return expand_channels(sample_direct_texture_2d_isotropic(
          texture, sample_uv, duv_dx, duv_dy, mip_desc.x, mip_desc.y,
          mip_desc.z, UInt(b.sampler & cycles_svm_image_interpolation_mask)));
    }
    if ((b.sampler & cycles_svm_image_interpolation_mask) == 2u) {
      return expand_channels(sample_cycles_texture_2d_bicubic(
          texture, sample_uv, [&](Float u, Float v) noexcept {
            return texture.sample(make_float2(u, v));
          }));
    }
    return expand_channels(texture.sample(sample_uv));
  }
  if (use_vt && vt->has_virtual_images()) {
    return vt_isotropic
               ? vt->sample_2d_dynamic_mip(
                     image_texture_id, sample_uv, duv_dx, duv_dy,
                     _scene->texture_heap, _vt_miss_page, false)
               : vt->sample_2d_dynamic(
                     image_texture_id, sample_uv, _scene->texture_heap,
                     _vt_miss_page, false);
  }
  if (!_scene->cycles_svm->image_binding_buffer) {
    dsl::unreachable(
        "image SVM node reached without a Cycles image binding table");
    return make_float4(0.0f);
  }
  if (direct_isotropic && _scene->direct_image_mip_descs) {
    return svm_detail::sample_scene_image_2d_isotropic(
        _scene->texture_heap,
        *_scene->cycles_svm->image_binding_buffer,
        _scene->direct_image_mip_descs,
        image_texture_id, uv,
        _has_failed_images, _has_compact_channel_images);
  }
  return svm_detail::sample_scene_image_2d(
      _scene->texture_heap,
      *_scene->cycles_svm->image_binding_buffer,
      image_texture_id, uv,
      _has_bicubic_images, _has_failed_images,
      _contiguous_texture_slots, _base_texture_slot,
      _image_binding_count, _has_compact_channel_images);
}

Float4 PathCyclesSvmKernelGlobals::kernel_image_interp_3d(
    svm::ShaderData &, Expr<std::int32_t>, Expr<luisa::float3>,
    Expr<std::int32_t>, Expr<bool>) const noexcept {
  dsl::unreachable(
      "3D image SVM service reached before voxel scene-image support");
  return make_float4(0.0f);
}

Float3 PathCyclesSvmKernelGlobals::object_location(
    const svm::ShaderData &shader_data) const noexcept {
  const auto object = object_record(*_scene, shader_data.object);
  return select(make_float3(0.0f),
                make_float3(object.tfm.x.w, object.tfm.y.w,
                            object.tfm.z.w),
                shader_data.object != svm::object_none);
}

Float3 PathCyclesSvmKernelGlobals::object_color(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(make_float3(0.0f), unpack(record.color),
                object != svm::object_none);
}

Float PathCyclesSvmKernelGlobals::object_alpha(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(0.0f, record.alpha, object != svm::object_none);
}

Float PathCyclesSvmKernelGlobals::object_pass_id(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(0.0f, record.pass_id, object != svm::object_none);
}

Float PathCyclesSvmKernelGlobals::shader_pass_id(
    const svm::ShaderData &shader_data) const noexcept {
  const Expr<Buffer<abi::KernelShader>> shaders{
      *_scene->cycles_svm->kernel_shader_buffer};
  return shaders->read(shader_data.shader & svm::shader_mask)
      .pass_id.cast<float>();
}

Float PathCyclesSvmKernelGlobals::object_random_number(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(0.0f, record.random_number, object != svm::object_none);
}

Int PathCyclesSvmKernelGlobals::object_particle_id(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(0, record.particle_index, object != svm::object_none);
}

UInt PathCyclesSvmKernelGlobals::particle_index(
    Expr<std::int32_t> particle) const noexcept {
  return particle_record(*_scene, particle).index;
}

Float PathCyclesSvmKernelGlobals::particle_age(
    Expr<std::int32_t> particle) const noexcept {
  return particle_record(*_scene, particle).age;
}

Float PathCyclesSvmKernelGlobals::particle_lifetime(
    Expr<std::int32_t> particle) const noexcept {
  return particle_record(*_scene, particle).lifetime;
}

Float PathCyclesSvmKernelGlobals::particle_size(
    Expr<std::int32_t> particle) const noexcept {
  return particle_record(*_scene, particle).size;
}

Float3 PathCyclesSvmKernelGlobals::particle_location(
    Expr<std::int32_t> particle) const noexcept {
  return particle_record(*_scene, particle).location.xyz();
}

Float3 PathCyclesSvmKernelGlobals::particle_velocity(
    Expr<std::int32_t> particle) const noexcept {
  return particle_record(*_scene, particle).velocity.xyz();
}

Float3 PathCyclesSvmKernelGlobals::particle_angular_velocity(
    Expr<std::int32_t> particle) const noexcept {
  return particle_record(*_scene, particle).angular_velocity.xyz();
}

Float PathCyclesSvmKernelGlobals::curve_thickness(
    const svm::ShaderData &shader_data) const noexcept {
  Float result = 0.0f;
  $if((shader_data.type & svm::primitive_curve) != 0u) {
    const auto curve_record = curve(shader_data.prim);
    const auto segment = shader_data.type >> svm::primitive_num_bits;
    const auto object = object_record(*_scene, shader_data.object);
    const auto k0 = object.position_offset + curve_record.first_key +
                    segment.cast<std::int32_t>();
    const auto p0 = curve_key(k0);
    const auto p1 = curve_key(k0 + 1);
    result = 2.0f * ((p1.w - p0.w) * shader_data.u + p0.w);
    $if((shader_data.object_flag &
         svm::shader_data_object_transform_applied) == 0u) {
      const auto radius = result * 0.5773502691896258f;
      const auto direction = cycles_transform::direction(
          unpack_transform(object.tfm), make_float3(radius));
      result = length(direction);
    };
  };
  return result;
}

Float3 PathCyclesSvmKernelGlobals::point_position(
    const svm::ShaderData &shader_data) const noexcept {
  Float3 result = make_float3(0.0f);
  $if((shader_data.type & svm::primitive_point) != 0u) {
    const auto object = object_record(*_scene, shader_data.object);
    const auto point = _scene->cycles_svm->geometry->point_buffer->read(
        (object.position_offset + shader_data.prim.cast<std::int32_t>())
            .cast<std::uint32_t>());
    result = make_float3(point.x, point.y, point.z);
    $if((shader_data.object_flag &
         svm::shader_data_object_transform_applied) == 0u) {
      result = cycles_transform::point(unpack_transform(object.tfm), result);
    };
  };
  return result;
}

Float PathCyclesSvmKernelGlobals::point_radius(
    const svm::ShaderData &shader_data) const noexcept {
  Float result = 0.0f;
  $if((shader_data.type & svm::primitive_point) != 0u) {
    const auto object = object_record(*_scene, shader_data.object);
    const auto point = _scene->cycles_svm->geometry->point_buffer->read(
        (object.position_offset + shader_data.prim.cast<std::int32_t>())
            .cast<std::uint32_t>());
    result = point.w;
    $if((shader_data.object_flag &
         svm::shader_data_object_transform_applied) == 0u) {
      const auto radius = result * 0.5773502691896258f;
      result = length(cycles_transform::direction(
          unpack_transform(object.tfm), make_float3(radius)));
    };
  };
  return result;
}

Float PathCyclesSvmKernelGlobals::global_ao_distance(
    const svm::ShaderData &shader_data) const noexcept {
  Float distance = _parameters.ambient_occlusion_distance;
  if (_scene->ambient_occlusion_distance_buffer) {
    const Expr<Buffer<float>> distance_buffer{
        *_scene->ambient_occlusion_distance_buffer};
    distance = distance_buffer->read(0u);
  }
  if (_scene->ambient_occlusion_object_distance_buffer) {
    const Expr<Buffer<float>> object_distance_buffer{
        _scene->ambient_occlusion_object_distance_buffer};
    const auto has_object_distance =
        (_parameters.ambient_occlusion_object_distance_count != 0u) &
        (shader_data.object != svm::object_none) &
        (shader_data.object <
         _parameters.ambient_occlusion_object_distance_count);
    $if(has_object_distance) {
      const auto object_distance =
          object_distance_buffer->read(shader_data.object);
      distance = select(distance, object_distance, object_distance != 0.0f);
    };
  }
  return distance;
}

Bool PathCyclesSvmKernelGlobals::ambient_occluded(
    const svm::ShaderData &shader_data,
    const Var<luisa::compute::Ray> &ray,
    Expr<bool> only_local) const noexcept {
  LUISA_ASSERT(_ambient_occlusion != nullptr &&
                   _ambient_occlusion->traversal != nullptr,
               "NODE_AMBIENT_OCCLUSION requires an AmbientOcclusion traversal context.");
  static_cast<void>(shader_data);
  return _ambient_occlusion->traversal->ambient_occluded(
      _scene, ray, _ambient_occlusion->source, only_local);
}

UInt PathCyclesSvmKernelGlobals::triangle_shader(
    Expr<std::uint32_t> prim) const noexcept {
  return _scene->cycles_svm->geometry->triangle_shader_buffer->read(prim);
}

Float PathCyclesSvmKernelGlobals::bevel_min_ray_pdf() const noexcept {
  if (_ambient_occlusion == nullptr) {
    return 1.0e30f;
  }
  return _ambient_occlusion->minimum_bsdf_pdf;
}

UInt PathCyclesSvmKernelGlobals::bevel_lcg_state_init() const noexcept {
  if (_ambient_occlusion == nullptr) {
    return 0u;
  }
  return cycles_noise::hash_uint3(
      _ambient_occlusion->rng_hash ^ 0x64c6a40eu,
      _ambient_occlusion->rng_offset, _ambient_occlusion->sample_index);
}

Float2 PathCyclesSvmKernelGlobals::bevel_sample_disk(
    Expr<std::uint32_t> sample,
    Expr<std::uint32_t> num_samples) const noexcept {
  if (_ambient_occlusion == nullptr ||
      _ambient_occlusion->sobol_table == nullptr) {
    return make_float2(0.0f);
  }
  return cycles_sampler::sample_branched_2d(
      *_ambient_occlusion->sobol_table, _parameters.sobol_sequence_size,
      _ambient_occlusion->sample_index, _ambient_occlusion->rng_hash,
      _ambient_occlusion->rng_offset, sample, num_samples,
      sampling::tabulated_sobol::surface_bevel_dimension);
}

svm::BevelLocalHitReservoir PathCyclesSvmKernelGlobals::trace_bevel_local(
    const svm::ShaderData &shader_data, const Var<luisa::compute::Ray> &ray,
    UInt &lcg_state) const noexcept {
  LUISA_ASSERT(_ambient_occlusion != nullptr &&
                   _ambient_occlusion->traversal != nullptr,
               "NODE_BEVEL requires a local traversal context.");
  return _ambient_occlusion->traversal->trace_bevel_local(
      _scene, ray, shader_data.object, lcg_state);
}

} // namespace psycles::luisa_backend::detail

#pragma once

#include "path_tracer_surfaces.h"

#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/luisa/cycles_svm.h>

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace psycles::luisa_backend::detail {

enum class PathCyclesSvmShaderDomain : std::uint8_t {
  all_shaders = 0u,
  surface_only = 1u,
  shadow_surface = 2u,
  light_emission = 3u,
  background = 4u,
};

[[nodiscard]] bool cycles_svm_has_domain_metadata(
    const LuisaSceneData &scene) noexcept;

[[nodiscard]] compiler::cycles_svm::ShaderEntryUsage
cycles_svm_domain_entry_usage(const LuisaSceneData &scene,
                              PathCyclesSvmShaderDomain domain) noexcept;

[[nodiscard]] compiler::cycles_svm::ShaderEntryUsage
cycles_svm_masked_entry_usage(const LuisaSceneData &scene,
                              std::span<const std::uint8_t> active_shader_mask) noexcept;

[[nodiscard]] std::vector<std::uint8_t>
cycles_svm_domain_shader_mask(const LuisaSceneData &scene,
                              PathCyclesSvmShaderDomain domain) noexcept;

// Host/JIT adapter from Psycles' uploaded scene resources to the exact
// KernelGlobals services observed by the copied Cycles SVM handlers. Virtual
// calls execute only while recording the shader AST; no vtable or provider
// object enters device memory.
class PathCyclesSvmKernelGlobals final
    : public cycles_svm::KernelGlobals,
      public cycles_svm::InfoServices {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    const Var<RenderKernelParameters> &_parameters;
    CameraProjection _camera_projection;
    bool _has_shadow_terminator_shading_offset{true};
    bool _may_have_thin_film{true};
    compiler::cycles_svm::ClosureUsage _closure_usage{};
    cycles_svm::SurfaceAttributeSummary _surface_attribute_summary{};
    std::unordered_map<std::uint64_t, cycles_svm::SurfaceAttributeSummary>
        _attribute_summaries_by_id;
    bool _has_attribute_summary_table{false};
    cycles_svm::NormalMapUsage _normal_map_usage{};
    std::uint32_t _mix_blend_mask{~0u};
    cycles_svm::NoiseStaticOptions _noise_static_options{};
    std::uint32_t _tex_coord_mask{~0u};
    bool _bump_uses_object_space{true};
    bool _curves_may_extrapolate{true};
    std::uint64_t _math_type_mask{~0ull};
    std::uint64_t _vector_math_type_mask{~0ull};
    cycles_svm::MappingStaticOptions _mapping_static_options{};
    std::uint32_t _map_range_type_mask{~0u};
    cycles_svm::VoronoiStaticOptions _voronoi_static_options{};
    cycles_svm::WaveStaticOptions _wave_static_options{};
    bool _has_curves{true};
    bool _has_bicubic_images{true};
    bool _has_failed_images{true};
    bool _contiguous_texture_slots{false};
    std::uint32_t _base_texture_slot{};
    std::uint32_t _image_binding_count{};
    Bool _caustics_reflective;
    Bool _caustics_refractive;
    luisa::compute::Float4x4 _camera_to_world;
    luisa::compute::Float4x4 _world_to_camera;

  public:
    PathCyclesSvmKernelGlobals(
        std::shared_ptr<LuisaSceneData> scene,
        const Var<RenderKernelParameters> &parameters,
        CameraProjection camera_projection,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        PathCyclesSvmShaderDomain domain =
            PathCyclesSvmShaderDomain::all_shaders,
        std::span<const std::uint8_t> explicit_shader_mask = {}) noexcept;
    PathCyclesSvmKernelGlobals(
        std::shared_ptr<LuisaSceneData> scene,
        const SurfacePopulationContext &context,
        std::span<const std::uint8_t> explicit_shader_mask = {}) noexcept;

    [[nodiscard]] Bool caustics_reflective() const noexcept override;
    [[nodiscard]] Bool caustics_refractive() const noexcept override;
    [[nodiscard]] Bool background_use_sun_guiding() const noexcept override;
    [[nodiscard]] bool has_shadow_terminator_shading_offset() const noexcept override;
    [[nodiscard]] bool may_have_thin_film() const noexcept override;
    [[nodiscard]] compiler::cycles_svm::ClosureUsage closure_usage() const noexcept override;
    [[nodiscard]] cycles_svm::SurfaceAttributeSummary surface_attribute_summary() const noexcept override;
    [[nodiscard]] cycles_svm::SurfaceAttributeSummary attribute_summary_for_id(
        std::uint64_t id) const noexcept override;
    [[nodiscard]] cycles_svm::NormalMapUsage normal_map_usage() const noexcept override;
    [[nodiscard]] std::uint32_t mix_blend_mask() const noexcept override;
    [[nodiscard]] cycles_svm::NoiseStaticOptions noise_static_options() const noexcept override;
    [[nodiscard]] std::uint32_t tex_coord_mask() const noexcept override;
    [[nodiscard]] bool bump_uses_object_space() const noexcept override;
    [[nodiscard]] bool curves_may_extrapolate() const noexcept override;
    [[nodiscard]] std::uint64_t math_type_mask() const noexcept override;
    [[nodiscard]] std::uint64_t vector_math_type_mask() const noexcept override;
    [[nodiscard]] cycles_svm::MappingStaticOptions mapping_static_options() const noexcept override;
    [[nodiscard]] std::uint32_t map_range_type_mask() const noexcept override;
    [[nodiscard]] cycles_svm::VoronoiStaticOptions voronoi_static_options() const noexcept override;
    [[nodiscard]] cycles_svm::WaveStaticOptions wave_static_options() const noexcept override;
    [[nodiscard]] bool has_curves() const noexcept override;
    [[nodiscard]] Float object_shadow_terminator_shading_offset(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] std::optional<Float> object_volume_density(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Float ies(
        Expr<std::uint32_t> offset) const noexcept override;
    [[nodiscard]] const cycles_svm::InfoServices *
    info_services() const noexcept override;

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t> index) const noexcept override;
    [[nodiscard]] cycles_svm::TriangleVertices triangle_vertices(
        Expr<std::uint32_t> object,
        Expr<std::uint32_t> prim) const noexcept override;
    [[nodiscard]] cycles_svm::TriangleVertices motion_triangle_vertices(
        Expr<std::uint32_t> object,
        Expr<std::uint32_t> prim,
        Expr<float> time) const noexcept override;
    [[nodiscard]] Float3 film_rgb_to_y() const noexcept override;
    [[nodiscard]] Float3 primitive_tangent(
        const cycles_svm::ShaderData &shader_data) const noexcept override;
    [[nodiscard]] cycles_svm::Dual3 primitive_tangent_derivative(
        const cycles_svm::ShaderData &shader_data) const noexcept override;
    [[nodiscard]] UInt object_attribute_map_offset(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Var<compiler::cycles_svm::AttributeMap> attribute_map(
        Expr<std::uint32_t> offset) const noexcept override;
    [[nodiscard]] std::optional<luisa::compute::UInt2>
    find_attribute_map_offset(
        Expr<std::uint32_t> start_offset,
        Expr<luisa::ulong> id) const noexcept override;
    [[nodiscard]] std::optional<cycles_svm::SurfaceAttributeBuffers>
    surface_attribute_buffers() const noexcept override;
    [[nodiscard]] bool prefers_outlined_loops() const noexcept override;
    [[nodiscard]] Float attribute_float(
        Expr<std::int32_t> offset) const noexcept override;
    [[nodiscard]] Float2 attribute_float2(
        Expr<std::int32_t> offset) const noexcept override;
    [[nodiscard]] Var<compiler::cycles_svm::packed_float3> attribute_float3(
        Expr<std::int32_t> offset) const noexcept override;
    [[nodiscard]] Float4 attribute_float4(
        Expr<std::int32_t> offset) const noexcept override;
    [[nodiscard]] Var<compiler::cycles_svm::uchar4> attribute_uchar4(
        Expr<std::int32_t> offset) const noexcept override;
    [[nodiscard]] Var<compiler::cycles_svm::packed_normal> attribute_normal(
        Expr<std::int32_t> offset) const noexcept override;
    [[nodiscard]] luisa::compute::UInt3 triangle_vertex_indices(
        Expr<std::uint32_t> prim) const noexcept override;
    [[nodiscard]] Int object_normal_offset(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] UInt object_num_geom_steps(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Int object_num_vertices(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Int object_num_primitives(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Float3 object_dupli_generated(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Float3 object_dupli_uv(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] UInt camera_type() const noexcept override;
    [[nodiscard]] Float camera_width() const noexcept override;
    [[nodiscard]] Float camera_height() const noexcept override;
    [[nodiscard]] Float3 camera_world_to_ndc(
        const cycles_svm::ShaderData &shader_data,
        Expr<luisa::float3> position) const noexcept override;
    [[nodiscard]] Var<compiler::cycles_svm::KernelCurve> curve(
        Expr<std::uint32_t> prim) const noexcept override;
    [[nodiscard]] Int object_position_offset(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Float4 curve_key(
        Expr<std::int32_t> key) const noexcept override;
    [[nodiscard]] Bool film_is_rec709() const noexcept override;
    [[nodiscard]] Float3 film_xyz_to_r() const noexcept override;
    [[nodiscard]] Float3 film_xyz_to_g() const noexcept override;
    [[nodiscard]] Float3 film_xyz_to_b() const noexcept override;
    [[nodiscard]] Float3 film_rec709_to_r() const noexcept override;
    [[nodiscard]] Float3 film_rec709_to_g() const noexcept override;
    [[nodiscard]] Float3 film_rec709_to_b() const noexcept override;
    [[nodiscard]] Float3 object_inverse_position_transform_if_object(
        const cycles_svm::ShaderData &shader_data,
        Expr<luisa::float3> value) const noexcept override;
    [[nodiscard]] cycles_svm::Dual3
    object_inverse_position_transform_if_object_derivative(
        const cycles_svm::ShaderData &shader_data,
        const cycles_svm::Dual3 &value) const noexcept override;
    [[nodiscard]] Float3 object_inverse_position_transform(
        const cycles_svm::ShaderData &shader_data,
        Expr<luisa::float3> value) const noexcept override;
    [[nodiscard]] Float4 kernel_image_interp_with_udim(
        cycles_svm::ShaderData &shader_data,
        Expr<std::int32_t> image_texture_id,
        const cycles_svm::Dual2 &uv) const noexcept override;
    [[nodiscard]] Float4 kernel_image_interp_3d(
        cycles_svm::ShaderData &shader_data,
        Expr<std::int32_t> image_texture_id,
        Expr<luisa::float3> position,
        Expr<std::int32_t> interpolation,
        Expr<bool> stochastic) const noexcept override;

    [[nodiscard]] Float3 object_location(
        const cycles_svm::ShaderData &shader_data) const noexcept override;
    [[nodiscard]] Float3 object_color(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Float object_alpha(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Float object_pass_id(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Float shader_pass_id(
        const cycles_svm::ShaderData &shader_data) const noexcept override;
    [[nodiscard]] Float object_random_number(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] Int object_particle_id(
        Expr<std::uint32_t> object) const noexcept override;
    [[nodiscard]] UInt particle_index(
        Expr<std::int32_t> particle) const noexcept override;
    [[nodiscard]] Float particle_age(
        Expr<std::int32_t> particle) const noexcept override;
    [[nodiscard]] Float particle_lifetime(
        Expr<std::int32_t> particle) const noexcept override;
    [[nodiscard]] Float particle_size(
        Expr<std::int32_t> particle) const noexcept override;
    [[nodiscard]] Float3 particle_location(
        Expr<std::int32_t> particle) const noexcept override;
    [[nodiscard]] Float3 particle_velocity(
        Expr<std::int32_t> particle) const noexcept override;
    [[nodiscard]] Float3 particle_angular_velocity(
        Expr<std::int32_t> particle) const noexcept override;
    [[nodiscard]] Float curve_thickness(
        const cycles_svm::ShaderData &shader_data) const noexcept override;
    [[nodiscard]] Float3 point_position(
        const cycles_svm::ShaderData &shader_data) const noexcept override;
    [[nodiscard]] Float point_radius(
        const cycles_svm::ShaderData &shader_data) const noexcept override;
};

} // namespace psycles::luisa_backend::detail

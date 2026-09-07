#pragma once

#include "path_tracer_surfaces.h"

#include <psycles/luisa/cycles_svm.h>

#include <memory>

namespace psycles::luisa_backend::detail {

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
        Expr<bool> refractive_caustics) noexcept;
    PathCyclesSvmKernelGlobals(
        std::shared_ptr<LuisaSceneData> scene,
        const SurfacePopulationContext &context) noexcept;

    [[nodiscard]] Bool caustics_reflective() const noexcept override;
    [[nodiscard]] Bool caustics_refractive() const noexcept override;
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

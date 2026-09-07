#include "path_tracer_cycles_svm_kernel_globals.h"

#include "cycles_svm_image_sampling.h"
#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>

#include <cmath>
#include <utility>

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

} // namespace

PathCyclesSvmKernelGlobals::PathCyclesSvmKernelGlobals(
    std::shared_ptr<LuisaSceneData> scene,
    const SurfacePopulationContext &context) noexcept
    : PathCyclesSvmKernelGlobals(
          std::move(scene), context.parameters, context.camera_projection,
          context.query.reflective_caustics,
          context.query.refractive_caustics) {}

PathCyclesSvmKernelGlobals::PathCyclesSvmKernelGlobals(
    std::shared_ptr<LuisaSceneData> scene,
    const Var<RenderKernelParameters> &parameters,
    CameraProjection camera_projection, Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics) noexcept
    : _scene{std::move(scene)},
      _parameters{parameters},
      _camera_projection{camera_projection},
      _caustics_reflective{reflective_caustics},
      _caustics_refractive{refractive_caustics},
      _camera_to_world{parameters.camera_transform},
      _world_to_camera{parameters.camera_inverse_transform} {
  LUISA_ASSERT(_scene && _scene->cycles_svm &&
                   _scene->cycles_svm->geometry &&
                   _scene->cycles_svm->objects,
               "Native Cycles SVM surface requires a finalized scene image.");
}

Bool PathCyclesSvmKernelGlobals::caustics_reflective() const noexcept {
  return _caustics_reflective;
}

Bool PathCyclesSvmKernelGlobals::caustics_refractive() const noexcept {
  return _caustics_refractive;
}

Float PathCyclesSvmKernelGlobals::object_shadow_terminator_shading_offset(
    Expr<std::uint32_t> object) const noexcept {
  const auto record = object_record(*_scene, object);
  return select(1.0f, record.shadow_terminator_shading_offset,
                object != svm::object_none);
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
  const auto camera = cycles_transform::point(_world_to_camera, position);
  if (_camera_projection == CameraProjection::perspective) {
    const auto screen_x = camera.x /
                          (-camera.z * _parameters.camera_horizontal_tangent);
    const auto screen_y = camera.y /
                          (-camera.z * _parameters.camera_vertical_tangent);
    return make_float3(
        0.5f * (screen_x - 2.0f * _parameters.camera_shift_x + 1.0f),
        0.5f * (screen_y - 2.0f * _parameters.camera_shift_y + 1.0f),
        -camera.z);
  }
  if (_camera_projection == CameraProjection::orthographic) {
    const auto aspect = camera_width() / camera_height();
    const auto screen_x = 2.0f * camera.x /
                          (_parameters.camera_ortho_vertical_span * aspect);
    const auto screen_y =
        2.0f * camera.y / _parameters.camera_ortho_vertical_span;
    return make_float3(
        0.5f * (screen_x - 2.0f * _parameters.camera_shift_x + 1.0f),
        0.5f * (screen_y - 2.0f * _parameters.camera_shift_y + 1.0f),
        -camera.z);
  }

  Float3 direction = camera;
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
  Float3 result = value;
  $if(shader_data.object != svm::object_none) {
    result = object_inverse_position_transform(shader_data, value);
  };
  return result;
}

svm::Dual3 PathCyclesSvmKernelGlobals::
    object_inverse_position_transform_if_object_derivative(
        const svm::ShaderData &shader_data,
        const svm::Dual3 &value) const noexcept {
  auto result = value;
  $if(shader_data.object != svm::object_none) {
    const auto object = object_record(*_scene, shader_data.object);
    const auto transform = unpack_transform(object.itfm);
    result.val = cycles_transform::point(transform, value.val);
    result.dx = cycles_transform::direction(transform, value.dx);
    result.dy = cycles_transform::direction(transform, value.dy);
  };
  return result;
}

Float3 PathCyclesSvmKernelGlobals::object_inverse_position_transform(
    const svm::ShaderData &shader_data,
    Expr<luisa::float3> value) const noexcept {
  const auto object = object_record(*_scene, shader_data.object);
  return cycles_transform::point(unpack_transform(object.itfm), value);
}

Float4 PathCyclesSvmKernelGlobals::kernel_image_interp_with_udim(
    svm::ShaderData &, Expr<std::int32_t> image_texture_id,
    const svm::Dual2 &uv) const noexcept {
  if (!_scene->cycles_svm->image_binding_buffer) {
    dsl::unreachable(
        "image SVM node reached without a Cycles image binding table");
    return make_float4(0.0f);
  }
  return svm_detail::sample_scene_image_2d(
      _scene->texture_heap,
      *_scene->cycles_svm->image_binding_buffer,
      image_texture_id, uv);
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

} // namespace psycles::luisa_backend::detail

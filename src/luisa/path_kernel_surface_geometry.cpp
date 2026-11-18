#include "path_kernel_builder.h"
#include "path_kernel_surface_primitive.h"
#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>

#include <tuple>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceGeometryStageImpl final : public SurfaceGeometryStage {

private:
  ScenePrimitiveStagePlan _primitive_plan;
  std::shared_ptr<const SurfacePrimitiveGeometryComponent> _geometry;

public:
  explicit SurfaceGeometryStageImpl(
      ScenePrimitiveStagePlan primitive_plan)
      : _primitive_plan{primitive_plan},
        _geometry{make_surface_primitive_geometry_component(
            primitive_plan)} {}

  SurfaceGeometryContext
  emit(PathBounceContext &bounce) const noexcept override {
    auto &sample = bounce.sample;
    auto &invocation = sample.invocation;
    const auto &config = invocation.config;
    const auto &scene = config.scene;
    auto &hit = bounce.hit;
    auto &ray = sample.ray;
    auto &ray_dP = sample.ray_dP;
    auto &ray_dD = sample.ray_dD;
    const auto shader_ray_visibility = sample.shader_ray_visibility();
    auto &ray_events = sample.ray_events;
    auto &path_depth = sample.path_depth;
    auto &diffuse_depth = sample.diffuse_depth;
    auto &glossy_depth = sample.glossy_depth;
    auto &transparent_depth = sample.transparent_depth;
    auto &transmission_depth = sample.transmission_depth;
    const auto terminate_after_transparent =
        sample.terminate_after_transparent_requested();
    auto &minimum_bsdf_pdf = sample.minimum_bsdf_pdf;
    const auto &surface_query = invocation.surface_query;
    const auto &kernel_parameters = invocation.parameters;
    const auto &safe_normalize = config.light_transport.safe_normalize;
    const auto reflective_caustics = config.reflective_caustics;
    const auto refractive_caustics = config.refractive_caustics;

    // Cycles camera_sample: shuttertime == -1 gives time 0.5 for the
    // currently admitted non-motion production path.
    auto primitive =
        _geometry->emit(scene, hit, ray, ray_dP, ray_dD, 0.5f,
                        kernel_parameters, safe_normalize);
    auto point = std::move(primitive.point);
    point.ray_visibility = shader_ray_visibility;
    point.ray_events = ray_events;
    point.ray_depth = path_depth;
    point.diffuse_depth = diffuse_depth;
    point.glossy_depth = glossy_depth;
    point.transparent_depth = transparent_depth;
    point.transmission_depth = transmission_depth;

    Float3 shadow_shading_normal = point.shading_normal;

    UInt path_lobe_mask = surface_query.lobe_mask;
    const Bool previous_ray_was_diffuse = scene->native_cycles_svm_surface
        ? (sample.cycles_path_visibility & cycles_path_state::visibility_diffuse) != 0u
        : (ray_events & static_cast<std::uint32_t>(contract::event_diffuse)) != 0u;
    const auto path_reflective_caustics =
        Bool{reflective_caustics} | !previous_ray_was_diffuse;
    const auto path_refractive_caustics =
        Bool{refractive_caustics} | !previous_ray_was_diffuse;
    if (!reflective_caustics) {
      path_lobe_mask = select(
          path_lobe_mask,
          path_lobe_mask &
              ~static_cast<std::uint32_t>(contract::event_glossy),
          previous_ray_was_diffuse);
    }
    if (!refractive_caustics) {
      path_lobe_mask = select(
          path_lobe_mask,
          path_lobe_mask &
              ~static_cast<std::uint32_t>(contract::event_transmission),
          previous_ray_was_diffuse);
    }
    path_lobe_mask = select(
        path_lobe_mask,
        static_cast<std::uint32_t>(contract::event_transparent),
        terminate_after_transparent);
    Float3 subsurface_normal = point.shading_normal;
    if (!scene->native_cycles_svm_surface) {
      $if(bounce.subsurface_exit) {
        subsurface_normal = invocation.surface_bssrdf_normal(
            primitive.surface_tag,
            point,
            primitive.surface_has_bssrdf_bump,
            path_reflective_caustics,
            path_refractive_caustics);
      };
    }
    SurfaceQuery path_surface_query{
        .lobe_mask = path_lobe_mask,
        .transport_mode = surface_query.transport_mode,
        .glossy_filter_roughness = 0.0f,
        .reflective_caustics = path_reflective_caustics,
        .refractive_caustics = path_refractive_caustics,
        .subsurface_exit = bounce.subsurface_exit,
        .subsurface_normal = subsurface_normal};
    if (config.has_filter_glossy) {
      const auto blur_pdf = kernel_parameters.filter_glossy * minimum_bsdf_pdf;
      const auto filter_glossy_enabled =
          kernel_parameters.filter_glossy < std::numeric_limits<float>::max();
      path_surface_query.glossy_filter_roughness = select(
          0.0f, sqrt(max(1.0f - blur_pdf, 0.0f)) * 0.5f,
          filter_glossy_enabled & (blur_pdf < 1.0f));
    }

    return {.bounce = bounce,
            .primitive_plan = _primitive_plan,
            .emission_sampling = std::move(primitive.emission_sampling),
            .instance = std::move(primitive.instance),
            .p0 = std::move(primitive.p0),
            .p1 = std::move(primitive.p1),
            .p2 = std::move(primitive.p2),
            .n0 = std::move(primitive.n0),
            .n1 = std::move(primitive.n1),
            .n2 = std::move(primitive.n2),
            .object_to_world = primitive.object_to_world,
            .world_to_object = primitive.world_to_object,
            .wp0 = std::move(primitive.wp0),
            .wp1 = std::move(primitive.wp1),
            .wp2 = std::move(primitive.wp2),
            .geometric_normal = point.geometric_normal,
            .hit_position = std::move(primitive.hit_position),
            .object_hit_position = std::move(primitive.object_hit_position),
            .differential_radius = primitive.differential_radius,
            .is_curve = primitive.is_curve,
            .cycles_primitive_type = primitive.cycles_primitive_type,
            .cycles_transform_applied =
                primitive.cycles_transform_applied,
            .triangle_smooth = std::move(primitive.triangle_smooth),
            .shadow_shading_normal = std::move(shadow_shading_normal),
            .surface_tag = std::move(primitive.surface_tag),
            .cycles_surface_shader =
                std::move(primitive.cycles_surface_shader),
            .cycles_object_index = std::move(primitive.cycles_object_index),
            .cycles_primitive_index =
                std::move(primitive.cycles_primitive_index),
            .cycles_shader_flags = std::move(primitive.cycles_shader_flags),
            .cycles_object_flags = std::move(primitive.cycles_object_flags),
            .volume_stack_entry = std::move(primitive.volume_stack_entry),
            .surface_has_volume = std::move(primitive.surface_has_volume),
            .point = std::move(point),
            .path_surface_query = std::move(path_surface_query),
            .shadow_terminator_geometry_offset =
                std::move(primitive.shadow_terminator_geometry_offset),
            .lazy_cycles_geometry = primitive.lazy_cycles_geometry};
  }
};

namespace svm_abi = compiler::cycles_svm;

[[nodiscard]] luisa::compute::Float4x4 unpack_packed_transform(
    Var<svm_abi::PackedTransform> t) noexcept {
  return cycles_svm::detail::transform_from_rows(
      make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
      make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
      make_float4(t.z.x, t.z.y, t.z.z, t.z.w));
}

[[nodiscard]] std::tuple<Float3, Float3, Float3> fetch_cycles_triangle_vertices(
    const LuisaSceneData &scene, Expr<std::uint32_t> object_index,
    Expr<std::uint32_t> prim_index) noexcept {
  const auto object =
      scene.cycles_svm->objects->object_buffer->read(object_index);
  const auto base = object.position_offset.cast<std::uint32_t>();
  const auto tri =
      scene.cycles_svm->geometry->triangle_index_buffer->read(prim_index);
  const auto &verts = scene.cycles_svm->geometry->triangle_vertex_buffer;
  const auto uv = [](Var<svm_abi::packed_float3> v) noexcept {
    return make_float3(v.x, v.y, v.z);
  };
  return {uv(verts->read(base + tri.x)), uv(verts->read(base + tri.y)),
          uv(verts->read(base + tri.z))};
}

[[nodiscard]] bool scene_has_shadow_terminator_geometry_offset(
    const LuisaSceneData &scene) noexcept {
  if (!scene.cycles_svm || !scene.cycles_svm->objects ||
      !scene.cycles_svm->objects->image.valid) {
    return true;
  }
  const auto &objects = scene.cycles_svm->objects->image.objects;
  for (std::size_t i = 0u; i < objects.size(); ++i) {
    if (i == static_cast<std::size_t>(scene.cycles_background_object_index)) {
      continue;
    }
    const float offset = objects[i].shadow_terminator_geometry_offset;
    if (!std::isfinite(offset) || offset > 0.0f) {
      return true;
    }
  }
  return false;
}

} // namespace

void SurfaceGeometryContext::ensure_world_triangle_vertices() noexcept {
  if (lazy_cycles_geometry) {
    const auto &scene = *bounce.sample.invocation.config.scene;
    auto [v0, v1, v2] = fetch_cycles_triangle_vertices(
        scene, cycles_object_index, cycles_primitive_index);
    $if(!cycles_transform_applied) {
      const auto o2w = unpack_packed_transform(
          scene.cycles_svm->objects->object_buffer->read(cycles_object_index)
              .tfm);
      v0 = cycles_transform::point(o2w, v0);
      v1 = cycles_transform::point(o2w, v1);
      v2 = cycles_transform::point(o2w, v2);
    };
    wp0 = std::move(v0);
    wp1 = std::move(v1);
    wp2 = std::move(v2);
    return;
  }
  $if(!cycles_transform_applied) {
    wp0 = cycles_transform::point(object_to_world, p0);
    wp1 = cycles_transform::point(object_to_world, p1);
    wp2 = cycles_transform::point(object_to_world, p2);
  };
}

std::tuple<Float3, Float3, Float3>
SurfaceGeometryContext::resolve_world_triangle_vertices() noexcept {
  if (lazy_cycles_geometry) {
    const auto &scene = *bounce.sample.invocation.config.scene;
    auto [v0, v1, v2] = fetch_cycles_triangle_vertices(
        scene, cycles_object_index, cycles_primitive_index);
    $if(!cycles_transform_applied) {
      const auto o2w = unpack_packed_transform(
          scene.cycles_svm->objects->object_buffer->read(cycles_object_index)
              .tfm);
      v0 = cycles_transform::point(o2w, v0);
      v1 = cycles_transform::point(o2w, v1);
      v2 = cycles_transform::point(o2w, v2);
    };
    return {std::move(v0), std::move(v1), std::move(v2)};
  }
  ensure_world_triangle_vertices();
  return {wp0, wp1, wp2};
}

luisa::compute::Float4x4
SurfaceGeometryContext::fetch_world_to_object() const noexcept {
  if (lazy_cycles_geometry) {
    const auto &scene = *bounce.sample.invocation.config.scene;
    return unpack_packed_transform(
        scene.cycles_svm->objects->object_buffer->read(cycles_object_index)
            .itfm);
  }
  return world_to_object;
}

void SurfaceGeometryContext::set_evaluated_shadow_shading_normal(
    Float3 evaluated_normal) noexcept {
  if (lazy_cycles_geometry &&
      !scene_has_shadow_terminator_geometry_offset(
          *bounce.sample.invocation.config.scene)) {
    return;
  }
  const Float geom_offset =
      lazy_cycles_geometry
          ? shadow_terminator_geometry_offset
          : Float{instance.shadow_terminator_geometry_offset};
  const auto record_shadow_terminator = [&] {
    $if((!bounce.subsurface_exit) & triangle_smooth & (geom_offset > 0.0f)) {
      shadow_shading_normal = evaluated_normal;
    };
  };
  if (primitive_plan.triangles) {
    if (primitive_plan.curves) {
      $if(!is_curve) {
        record_shadow_terminator();
      };
    } else {
      record_shadow_terminator();
    }
  }
}

Float3 SurfaceGeometryContext::make_ray_origin(Float3 direction) const noexcept {
  Float3 origin = hit_position;
  const auto resolve_triangle_origin = [&] {
    if (lazy_cycles_geometry) {
      const auto &scene = *bounce.sample.invocation.config.scene;
      const auto [v0, v1, v2] = fetch_cycles_triangle_vertices(
          scene, cycles_object_index, cycles_primitive_index);
      Float3 cycles_origin = hit_position;
      Float3 cycles_direction = direction;
      $if(!cycles_transform_applied) {
        const auto w2o = unpack_packed_transform(
            scene.cycles_svm->objects->object_buffer->read(cycles_object_index)
                .itfm);
        cycles_origin = cycles_transform::point(w2o, hit_position);
        cycles_direction = cycles_transform::direction(w2o, direction);
      };
      origin = surface_ray::origin_with_explicit_self_exclusion(
          hit_position, geometric_normal, cycles_origin, cycles_direction,
          v0, v1, v2);
    } else {
      const Float3 object_direction =
          cycles_transform::direction(world_to_object, direction);
      const Float3 cycles_origin = select(
          object_hit_position, hit_position, cycles_transform_applied);
      const Float3 cycles_direction = select(
          object_direction, direction, cycles_transform_applied);
      origin = surface_ray::origin_with_explicit_self_exclusion(
          hit_position, geometric_normal, cycles_origin, cycles_direction,
          p0, p1, p2);
    }
  };
  if (primitive_plan.triangles) {
    if (primitive_plan.curves) {
      $if(!is_curve) {
        resolve_triangle_origin();
      };
    } else {
      resolve_triangle_origin();
    }
  }
  return origin;
}

surface_ray::ShadowOrigin
SurfaceGeometryContext::make_shadow_origin(Float3 direction) const noexcept {
  if (lazy_cycles_geometry) {
    auto origin = make_shadow_terminator_origin(direction);
    origin.position =
        apply_shadow_certificate(origin.position, direction, origin.skip_self);
    return origin;
  }
  Float3 position = hit_position;
  Bool skip_self = true;
  const auto resolve_triangle_origin = [&] {
    const auto triangle = surface_ray::surface_shadow_origin(
        hit_position, shadow_shading_normal, geometric_normal, direction,
        instance.shadow_terminator_geometry_offset, triangle_smooth,
        object_to_world, world_to_object, cycles_transform_applied,
        bounce.hit->bary, p0, p1, p2, n0, n1, n2);
    position = triangle.position;
    skip_self = triangle.skip_self;
  };
  if (primitive_plan.triangles) {
    if (primitive_plan.curves) {
      $if(!is_curve) {
        resolve_triangle_origin();
      };
    } else {
      resolve_triangle_origin();
    }
  }
  return {.position = std::move(position), .skip_self = std::move(skip_self)};
}

surface_ray::ShadowOrigin
SurfaceGeometryContext::make_shadow_terminator_origin(
    Float3 direction) const noexcept {
  Float3 position = hit_position;
  Bool skip_self = true;
  const auto resolve_triangle_origin = [&] {
    if (lazy_cycles_geometry) {
      if (!scene_has_shadow_terminator_geometry_offset(
              *bounce.sample.invocation.config.scene)) {
        return;
      }
      $if(triangle_smooth & (shadow_terminator_geometry_offset > 0.0f)) {
        auto normal_light = dot(shadow_shading_normal, direction);
        const auto transmit = normal_light < 0.0f;
        normal_light = abs(normal_light);
        const auto offset_normal =
            select(geometric_normal, -geometric_normal, transmit);
        const auto geometric_light = dot(offset_normal, direction);
        const auto near_terminator = clamp(
            2.0f - (geometric_light + normal_light) /
                       shadow_terminator_geometry_offset,
            0.0f, 1.0f);
        const auto regular = clamp(
            1.0f - geometric_light / shadow_terminator_geometry_offset,
            0.0f, 1.0f);
        const auto amount = select(
            regular, near_terminator,
            normal_light < shadow_terminator_geometry_offset);
        $if(amount > 0.0f) {
          const auto &scene = *bounce.sample.invocation.config.scene;
          const auto object =
              scene.cycles_svm->objects->object_buffer->read(
                  cycles_object_index);
          const auto tri =
              scene.cycles_svm->geometry->triangle_index_buffer->read(
                  cycles_primitive_index);
          const auto base = object.position_offset.cast<std::uint32_t>();
          const auto &verts =
              scene.cycles_svm->geometry->triangle_vertex_buffer;
          const auto uv = [](Var<svm_abi::packed_float3> v) noexcept {
            return make_float3(v.x, v.y, v.z);
          };
          const auto v0 = uv(verts->read(base + tri.x));
          const auto v1 = uv(verts->read(base + tri.y));
          const auto v2 = uv(verts->read(base + tri.z));
          const UInt obj_flags =
              cycles_object_flags
                  ? UInt{*cycles_object_flags}
                  : scene.cycles_svm->objects->object_flag_buffer->read(
                        cycles_object_index);
          const Bool corner_normals =
              (obj_flags &
               cycles_svm::shader_data_object_has_corner_normals) != 0u;
          const UInt ni0 =
              select(tri.x, cycles_primitive_index * 3u, corner_normals);
          const UInt ni1 =
              select(tri.y, cycles_primitive_index * 3u + 1u, corner_normals);
          const UInt ni2 =
              select(tri.z, cycles_primitive_index * 3u + 2u, corner_normals);
          const auto norm_base = object.normal_offset;
          const auto &norm_buf =
              scene.cycles_svm->geometry->attribute_normal_buffer;
          const auto vn0 = cycles_svm::decode_packed_normal(
              norm_buf->read(
                  (norm_base + ni0.cast<std::int32_t>()).cast<std::uint32_t>()));
          const auto vn1 = cycles_svm::decode_packed_normal(
              norm_buf->read(
                  (norm_base + ni1.cast<std::int32_t>()).cast<std::uint32_t>()));
          const auto vn2 = cycles_svm::decode_packed_normal(
              norm_buf->read(
                  (norm_base + ni2.cast<std::int32_t>()).cast<std::uint32_t>()));
          luisa::compute::Float4x4 o2w = make_float4x4(1.0f);
          $if(!cycles_transform_applied) {
            o2w = unpack_packed_transform(object.tfm);
          };
          const auto offset = surface_ray::smooth_surface_offset(
              o2w, cycles_transform_applied, bounce.hit->bary,
              v0, v1, v2, vn0, vn1, vn2, offset_normal);
          position = hit_position + offset * amount;
          skip_self = geometric_light > 0.0f;
        };
      };
    } else {
      const auto triangle = surface_ray::surface_shadow_terminator_origin(
          hit_position, shadow_shading_normal, geometric_normal, direction,
          instance.shadow_terminator_geometry_offset, triangle_smooth,
          object_to_world, cycles_transform_applied,
          bounce.hit->bary, p0, p1, p2, n0, n1, n2);
      position = triangle.position;
      skip_self = triangle.skip_self;
    }
  };
  if (primitive_plan.triangles) {
    if (primitive_plan.curves) {
      $if(!is_curve) { resolve_triangle_origin(); };
    } else {
      resolve_triangle_origin();
    }
  }
  return {.position = std::move(position), .skip_self = std::move(skip_self)};
}

Float3 SurfaceGeometryContext::apply_shadow_certificate(
    Float3 position, Float3 direction, Bool skip_self) const noexcept {
  auto result = position;
  const auto resolve_triangle_certificate = [&] {
    if (lazy_cycles_geometry) {
      $if(skip_self) {
        const auto &scene = *bounce.sample.invocation.config.scene;
        const auto [v0, v1, v2] = fetch_cycles_triangle_vertices(
            scene, cycles_object_index, cycles_primitive_index);
        Float3 object_origin = position;
        Float3 object_direction = direction;
        $if(!cycles_transform_applied) {
          const auto w2o = unpack_packed_transform(
              scene.cycles_svm->objects->object_buffer->read(cycles_object_index)
                  .itfm);
          object_origin = cycles_transform::point(w2o, position);
          object_direction = cycles_transform::direction(w2o, direction);
        };
        result = surface_ray::origin_with_explicit_self_exclusion(
            position, geometric_normal, object_origin, object_direction,
            v0, v1, v2);
      };
    } else {
      result = surface_ray::surface_shadow_certificate(
          position, geometric_normal, direction, world_to_object,
          cycles_transform_applied, skip_self, p0, p1, p2);
    }
  };
  if (primitive_plan.triangles) {
    if (primitive_plan.curves) {
      $if(!is_curve) { resolve_triangle_certificate(); };
    } else {
      resolve_triangle_certificate();
    }
  }
  return result;
}

std::unique_ptr<SurfaceGeometryStage> make_surface_geometry_stage(
    ScenePrimitiveStagePlan plan) {
  return std::make_unique<SurfaceGeometryStageImpl>(plan);
}

} // namespace psycles::luisa_backend::detail

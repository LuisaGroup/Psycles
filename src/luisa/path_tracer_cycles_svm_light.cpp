#include "path_tracer_cycles_svm_light.h"
#include "path_tracer_cycles_svm_shader_data.h"

#include "cycles_svm_internal.h"
#include "path_tracer_cycles_svm_kernel_globals.h"

#include <psycles/luisa/analytic_light_sampling.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_transform.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;
namespace svm = ::psycles::luisa_backend::cycles_svm;
namespace sd_detail = ::psycles::luisa_backend::cycles_svm::detail;
namespace abi = ::psycles::compiler::cycles_svm;

[[nodiscard]] Float4x4 unpack_transform(Var<abi::PackedTransform> t) noexcept {
  return sd_detail::transform_from_rows(
      make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
      make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
      make_float4(t.z.x, t.z.y, t.z.z, t.z.w));
}

// Cycles util/math_intersect.h::ray_triangle_uv. The intersection is already
// known to exist; reconstituting UV from the final shadow ray avoids carrying
// the pre-offset light sample through the surface -> shadow cut.
[[nodiscard]] Float2
triangle_light_uv(Float3 ray_p, Float3 ray_d,
                  const svm::TriangleVertices &vertices) noexcept {
  const auto v0 = vertices.v0 - ray_p;
  const auto v1 = vertices.v1 - ray_p;
  const auto v2 = vertices.v2 - ray_p;
  const auto e0 = v2 - v0;
  const auto e1 = v0 - v1;
  const auto e2 = v1 - v2;
  const auto u = dot(cross(e0, v2 + v0), ray_d);
  const auto v = dot(cross(e1, v0 + v1), ray_d);
  const auto w = dot(cross(e2, v1 + v2), ray_d);
  const auto uvw = u + v + w;
  const auto inverse = select(1.0f / uvw, 0.0f, abs(uvw) < 1.0e-18f);
  return make_float2(clamp(u * inverse, 0.0f, 1.0f),
                     clamp(v * inverse, 0.0f, 1.0f));
}

void setup_triangle(const LuisaSceneData &scene, const svm::KernelGlobals &kg,
                    const svm::TransformState &transforms,
                    const Var<DirectLightTaskCall> &task,
                    svm::ShaderData &sd) noexcept {
  // The uploaded production geometry image is static, as is its existing
  // motion_triangle_vertices service. Do not reinterpret another primitive
  // as a static triangle if that scene-domain invariant ever changes.
  assume(sd.type == static_cast<unsigned>(abi::PRIMITIVE_TRIANGLE));
  const auto vertices = kg.triangle_vertices(sd.object, sd.prim);
  auto world_vertices = vertices;
  const auto transform_applied =
      (sd.object_flag & svm::shader_data_object_transform_applied) != 0u;
  $if(!transform_applied) {
    world_vertices.v0 =
        cycles_transform::point(transforms.object_to_world, vertices.v0);
    world_vertices.v1 =
        cycles_transform::point(transforms.object_to_world, vertices.v1);
    world_vertices.v2 =
        cycles_transform::point(transforms.object_to_world, vertices.v2);
  };
  const auto uv =
      triangle_light_uv(task.ray_origin, task.ray_direction, world_vertices);
  sd.u = uv.x;
  sd.v = uv.y;
  sd.shader = scene.cycles_svm->geometry->triangle_shader_buffer->read(sd.prim);
  cycles_svm_triangle_shader_setup(kg, transforms, vertices, sd);
  cycles_svm_shader_setup_backfacing(sd);
  sd.dP = task.ray_dP + task.ray_maximum * task.ray_dD;
  sd.dI = task.ray_dD;
  cycles_svm_shader_setup_dudv(sd);
}

void setup_lamp(const LuisaSceneData &scene,
                const Var<DirectLightTaskCall> &task,
                svm::ShaderData &sd) noexcept {
  const auto light = scene.light_buffer->read(task.light_primitive);
  sd.shader = light.cycles_shader_id;
  sd.P = select(task.ray_origin + task.ray_maximum * task.ray_direction,
                -task.ray_direction, task.ray_maximum == ray_maximum);
  Float3 normal = -task.ray_direction;
  Float2 uv = make_float2(0.0f);
  const auto transform = analytic_light_sampling::light_linear_transform(
      light.axis_x, light.axis_y, light.axis_z, light.axis_scale);
  $if((light.type == static_cast<unsigned>(LightType::point)) |
      (light.type == static_cast<unsigned>(LightType::spot))) {
    $if(((light.flags & light_flag_sphere) != 0u) & (light.radius != 0.0f)) {
      normal = sd_detail::normalize_cycles(sd.P - light.position);
    };
    $if(light.type == static_cast<unsigned>(LightType::spot)) {
      auto local = sd_detail::safe_normalize_cycles(
          analytic_light_sampling::world_to_light_direction(-task.ray_direction,
                                                            transform));
      local.z = -local.z;
      uv = analytic_light_sampling::spot_light_uv(local, light.spot_angle);
    }
    $else { uv = analytic_light_sampling::point_light_uv(normal, transform); };
  }
  $elif(light.type == static_cast<unsigned>(LightType::area)) {
    normal = -light.axis_z;
    const auto inplane = sd.P - light.position;
    const auto u =
        clamp(dot(inplane, light.axis_x / light.size_u), -0.5f, 0.5f);
    const auto v =
        clamp(dot(inplane, light.axis_y / light.size_v), -0.5f, 0.5f);
    uv = make_float2(v + 0.5f, -u - v);
  }
  $else {
    // KernelSunLight uses co = -axis_z and angle = half the authored angle.
    const auto half_inverse =
        select(0.5f / sin(0.25f * light.angle), 0.0f, light.angle == 0.0f);
    const auto factor =
        half_inverse / length(task.ray_direction + light.axis_z);
    const auto local = analytic_light_sampling::world_to_light_direction(
        task.ray_direction, transform);
    const auto u = local.x * factor;
    const auto v = local.y * factor;
    uv = make_float2(v + 0.5f, -u - v);
  };
  sd.N = sd.Ng = normal;
  sd.u = uv.x;
  sd.v = uv.y;
  // shader_setup_from_sample leaves lamp derivatives zero, including dI.
  cycles_svm_shader_setup_backfacing(sd);
}

} // namespace

CyclesSvmLightShaderData setup_cycles_svm_light_shader_data(
    const std::shared_ptr<LuisaSceneData> &scene, const svm::KernelGlobals &kg,
    const Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) noexcept {
  const auto identity = make_float4x4(1.0f);
  svm::TransformState transforms{parameters.camera_transform,
                                 parameters.camera_inverse_transform, identity,
                                 identity};
  svm::ShaderData sd{make_float3(0.0f),
                     make_float3(0.0f),
                     make_float3(0.0f),
                     -task.ray_direction,
                     0u,
                     0u,
                     0u,
                     0u,
                     task.light_primitive,
                     0.0f,
                     0.0f,
                     task.light_object,
                     task.ray_time,
                     task.ray_maximum,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     make_float3(0.0f),
                     make_float3(0.0f),
                     identity,
                     identity,
                     cycles_noise::hash_uint3(task.rng_hash ^ 0xb4bc3953u,
                                              task.rng_offset,
                                              task.sample_index),
                     nullptr};
  sd.ray_P = task.ray_origin;
  Bool background = false;
  const auto setup_nonbackground = [&] {
    const auto object =
        scene->cycles_svm->objects->object_buffer->read(task.light_object);
    sd.object_flag =
        scene->cycles_svm->objects->object_flag_buffer->read(task.light_object);
    sd.type = object.primitive_type.cast<unsigned>();
    transforms.object_to_world = unpack_transform(object.tfm);
    transforms.world_to_object = unpack_transform(object.itfm);
    if (scene->light_count != 0u && scene->emissive_triangle_count != 0u) {
      $if(sd.type == static_cast<unsigned>(abi::PRIMITIVE_LAMP)) {
        setup_lamp(*scene, task, sd);
      }
      $else { setup_triangle(*scene, kg, transforms, task, sd); };
    } else if (scene->light_count != 0u) {
      setup_lamp(*scene, task, sd);
    } else if (scene->emissive_triangle_count != 0u) {
      setup_triangle(*scene, kg, transforms, task, sd);
    } else {
      dsl::unreachable("non-background NEE without a scene emitter");
    }
  };
  if (scene->cycles_background_shader_id != ~0u) {
    $if(task.light_object == scene->cycles_background_object_index) {
      background = true;
      sd.P = task.ray_direction;
      sd.N = sd.Ng = -task.ray_direction;
      sd.shader = scene->cycles_background_shader_id;
      sd.object = svm::object_none;
      sd.prim = ~0u;
      sd.type = 0u;
      sd.ray_length = ray_maximum;
      const auto map_dD =
          scene->background_map_weight > 0.0f &&
                  scene->background_map_width > 0u &&
                  scene->background_map_height > 0u
              ? std::min(pi / scene->background_map_height,
                         2.0f * pi / scene->background_map_width)
              : std::numeric_limits<float>::max();
      sd.dP = sd.dI = min(task.ray_dD, map_dD);
      const auto basis = sd_detail::differential_from_compact(sd.Ng, 1.0f);
      sd.dPdu = basis.dx;
      sd.dPdv = basis.dy;
      sd.du.dx = sd.dv.dy = sd.dP;
    }
    $else { setup_nonbackground(); };
  } else {
    setup_nonbackground();
  }
  sd.flag |= (*scene->cycles_svm->kernel_shader_buffer)
                 ->read(sd.shader & svm::shader_mask)
                 .flags.cast<unsigned>();
  return {.shader_data = std::move(sd),
          .transforms = std::move(transforms),
          .background = std::move(background)};
}

namespace {

class CyclesSvmLightEmission final : public DirectLightEmissionComponent {
  std::shared_ptr<LuisaSceneData> _scene;
  CameraProjection _camera_projection;
  bool _reflective_caustics;
  bool _refractive_caustics;

public:
  CyclesSvmLightEmission(std::shared_ptr<LuisaSceneData> scene,
                         CameraProjection camera_projection,
                         bool reflective_caustics, bool refractive_caustics)
      : _scene{std::move(scene)}, _camera_projection{camera_projection},
        _reflective_caustics{reflective_caustics},
        _refractive_caustics{refractive_caustics} {}

  [[nodiscard]] Float3 evaluate(
      const Var<DirectLightTaskCall> &task,
      const Var<RenderKernelParameters> &parameters) const noexcept override {
    const PathCyclesSvmKernelGlobals kg{_scene, parameters, _camera_projection,
                                        Bool{_reflective_caustics},
                                        Bool{_refractive_caustics}};
    auto setup =
        setup_cycles_svm_light_shader_data(_scene, kg, task, parameters);
    auto &sd = setup.shader_data;
    const auto &transforms = setup.transforms;
    const auto &background = setup.background;
    // The shadow state's bounce counters are copied before surface scatter.
    // NODE_LIGHT_PATH performs the emission-ray +1 itself, exactly once.
    const svm::PathState state{0u,
                               svm::path_ray_emission,
                               task.path_depth,
                               task.transparent_depth,
                               task.diffuse_depth,
                               task.glossy_depth,
                               task.transmission_depth,
                               0u};
    svm::EvaluationResult result;
    svm::eval_nodes(kg, *_scene->cycles_svm->word_buffer,
                    abi::SHADER_TYPE_SURFACE,
                    _scene->cycles_svm->kernel_features,
                    svm::kernel_feature_node_mask_surface_light,
                    _scene->cycles_svm->compilation.table.node_types_used,
                    transforms, sd, state, result);
    $if(result.status !=
        static_cast<unsigned>(svm::EvaluationStatus::ended)) {
      dsl::unreachable("native Cycles light SVM did not reach NODE_END");
    };
    Float3 emission = make_float3(0.0f);
    $if((sd.flag & static_cast<unsigned>(abi::SD_EMISSION)) != 0u) {
      const auto evaluable = background | (abs(dot(sd.Ng, sd.wi)) > 0.0f);
      emission =
          select(make_float3(0.0f), sd.closure_emission_background, evaluable);
    };
    return emission;
  }
};

} // namespace

std::shared_ptr<const DirectLightEmissionComponent>
make_cycles_svm_light_emission_component(
    const std::shared_ptr<LuisaSceneData> &scene,
    CameraProjection camera_projection, bool reflective_caustics,
    bool refractive_caustics) noexcept {
  return std::make_shared<CyclesSvmLightEmission>(
      scene, camera_projection, reflective_caustics, refractive_caustics);
}

} // namespace psycles::luisa_backend::detail

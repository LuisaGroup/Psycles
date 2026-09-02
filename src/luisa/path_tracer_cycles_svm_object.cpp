#include "path_tracer_cycles_svm_object.h"

#include "path_tracer_curve_scene.h"
#include "path_tracer_internal.h"
#include "path_tracer_scene_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

using namespace compiler::cycles_svm;

struct ObjectBounds {
  bool valid{};
  Vec3f minimum{};
  Vec3f maximum{};
};

struct PendingSceneObject {
  PendingKernelObject object;
  FinalizedKernelObjectGeometry geometry;
  ObjectBounds bounds;
};

[[nodiscard]] CyclesSvmObjectSceneImage reject(std::string diagnostic) {
  CyclesSvmObjectSceneImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] bool finite(Vec3f value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

void grow(ObjectBounds &bounds, Vec3f value) noexcept {
  // Cycles retries invalid geometry bounds through BoundBox::grow_safe. The
  // finite projection below is the corresponding host operation: invalid
  // source values cannot make a valid object's interval non-finite.
  if (!finite(value)) {
    return;
  }
  if (!bounds.valid) {
    bounds.valid = true;
    bounds.minimum = value;
    bounds.maximum = value;
    return;
  }
  bounds.minimum.x = std::min(bounds.minimum.x, value.x);
  bounds.minimum.y = std::min(bounds.minimum.y, value.y);
  bounds.minimum.z = std::min(bounds.minimum.z, value.z);
  bounds.maximum.x = std::max(bounds.maximum.x, value.x);
  bounds.maximum.y = std::max(bounds.maximum.y, value.y);
  bounds.maximum.z = std::max(bounds.maximum.z, value.z);
}

[[nodiscard]] ObjectBounds transform_bounds(const ObjectBounds &local,
                                            const Mat4f &transform) noexcept {
  if (!local.valid) {
    return {};
  }
  ObjectBounds world;
  for (const auto x : {local.minimum.x, local.maximum.x}) {
    for (const auto y : {local.minimum.y, local.maximum.y}) {
      for (const auto z : {local.minimum.z, local.maximum.z}) {
        grow(world, cycles_transform_point(transform, {x, y, z}));
      }
    }
  }
  return world;
}

[[nodiscard]] bool intersects(const ObjectBounds &lhs,
                              const ObjectBounds &rhs) noexcept {
  return lhs.valid && rhs.valid && lhs.minimum.x <= rhs.maximum.x &&
         lhs.maximum.x >= rhs.minimum.x && lhs.minimum.y <= rhs.maximum.y &&
         lhs.maximum.y >= rhs.minimum.y && lhs.minimum.z <= rhs.maximum.z &&
         lhs.maximum.z >= rhs.minimum.z;
}

[[nodiscard]] ObjectBounds mesh_bounds(const GeometryUpload &upload,
                                       const contract::InstanceDesc &instance,
                                       bool transform_applied) noexcept {
  ObjectBounds bounds;
  if (transform_applied) {
    // The geometry transaction has already proved that this is the exact
    // world-space vertex image used by the accelerator and Cycles tri_verts.
    for (const auto position : upload.cycles_intersection_positions) {
      grow(bounds, {position.x, position.y, position.z});
    }
    return bounds;
  }
  for (const auto position : upload.positions) {
    grow(bounds, {position.x, position.y, position.z});
  }
  return transform_bounds(bounds, instance.transform);
}

[[nodiscard]] ObjectBounds
curve_bounds(const contract::CurveGeometryDesc &geometry,
             const contract::InstanceDesc &instance) noexcept {
  const auto geometry_bounds = build_curve_geometry_bounds(geometry);
  const auto local = ObjectBounds{.valid = geometry_bounds.valid,
                                  .minimum = geometry_bounds.minimum,
                                  .maximum = geometry_bounds.maximum};
  return transform_bounds(local, instance.transform);
}

[[nodiscard]] bool material_has_volume(const contract::SceneSnapshot &snapshot,
                                       contract::MaterialId material,
                                       bool &has_volume,
                                       std::string &diagnostic) {
  const auto iter = snapshot.materials.find(material);
  if (iter == snapshot.materials.end()) {
    diagnostic =
        "Cycles object volume domain references unavailable material " +
        std::to_string(material.value);
    return false;
  }
  has_volume |=
      iter->second.shader.root(contract::ShaderDomain::volume).has_value();
  return true;
}

template <typename Geometry>
[[nodiscard]] bool
geometry_has_volume(const contract::SceneSnapshot &snapshot,
                    contract::GeometryId geometry_id, const Geometry &geometry,
                    bool &has_volume, std::string &diagnostic) {
  has_volume = false;
  for (const auto material : geometry.material_slots) {
    if (!material_has_volume(snapshot, material, has_volume, diagnostic)) {
      return false;
    }
  }
  // Cycles Geometry::used_shaders is shared by every user. Instance overrides
  // therefore contribute to one geometry-level union, not an object-local
  // volume flag.
  for (const auto &[instance_id, instance] : snapshot.instances) {
    static_cast<void>(instance_id);
    if (instance.geometry != geometry_id) {
      continue;
    }
    for (const auto material : instance.material_overrides) {
      if (!material_has_volume(snapshot, material, has_volume, diagnostic)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] std::optional<std::uint32_t>
cycles_visibility(std::uint32_t visibility) noexcept {
  if (visibility == std::numeric_limits<std::uint32_t>::max()) {
    visibility = contract::all_ray_visibility;
  }
  if ((visibility & ~contract::all_ray_visibility) != 0u) {
    return std::nullopt;
  }
  auto result = std::uint32_t{};
  const auto include = [&](contract::RayVisibility source,
                           std::uint32_t destination) {
    if ((visibility & contract::visibility_bit(source)) != 0u) {
      result |= destination;
    }
  };
  include(contract::RayVisibility::camera, PATH_RAY_VISIBILITY_CAMERA);
  include(contract::RayVisibility::transmission, PATH_RAY_VISIBILITY_TRANSMIT);
  include(contract::RayVisibility::diffuse, PATH_RAY_VISIBILITY_DIFFUSE);
  include(contract::RayVisibility::glossy, PATH_RAY_VISIBILITY_GLOSSY);
  include(contract::RayVisibility::volume_scatter,
          PATH_RAY_VISIBILITY_VOLUME_SCATTER);
  include(contract::RayVisibility::shadow, PATH_RAY_VISIBILITY_SHADOW);
  return result;
}

[[nodiscard]] std::uint32_t quantize_legacy_random(float value) noexcept {
  const auto normalized = std::clamp(static_cast<double>(value), 0.0, 1.0);
  return static_cast<std::uint32_t>(std::llround(normalized * 4294967295.0));
}

[[nodiscard]] std::string
source_asset_name(const contract::SceneSnapshot &snapshot,
                  std::string_view name, std::string_view asset_name) {
  if (!asset_name.empty() || snapshot.cycles_object_count) {
    return std::string{asset_name};
  }
  return std::string{name};
}

[[nodiscard]] std::optional<FinalizedGeometryAttribute>
finalized_attributes(const CyclesSvmGeometrySceneImage &image,
                     std::uint32_t index) noexcept {
  if (index >= image.attributes.geometries.size()) {
    return std::nullopt;
  }
  return image.attributes.geometries[index];
}

[[nodiscard]] FinalizedKernelObjectGeometry
geometry_record(const FinalizedGeometryAttribute &attributes,
                std::uint64_t numverts, std::uint64_t numprims,
                std::int32_t primitive_type, bool transform_applied,
                bool has_corner_normals, bool has_volume) noexcept {
  return {.volume_density = 1.0f,
          .geometry_motion_steps = 0u,
          .numverts = numverts,
          .numprims = numprims,
          .attribute_map_offset = attributes.attribute_map_offset,
          .position_offset = attributes.position_offset,
          .normal_offset = attributes.normal_offset,
          .primitive_type = primitive_type,
          .velocity_scale = 0.0f,
          .transform_applied = transform_applied,
          .has_vertex_motion = false,
          .has_corner_normals = has_corner_normals,
          .has_volume = has_volume,
          .intersects_volume = false,
          .has_volume_attributes = false,
          .has_volume_motion = false};
}

[[nodiscard]] std::optional<std::uint32_t>
particle_index(const ParticleTableImage &particles,
               std::uint32_t object_index) noexcept {
  const auto iter = particles.object_particle_indices.find(object_index);
  return iter == particles.object_particle_indices.end()
             ? std::nullopt
             : std::optional<std::uint32_t>{iter->second};
}

[[nodiscard]] KernelObjectSource
instance_source(const contract::SceneSnapshot &snapshot,
                const contract::InstanceDesc &instance, std::uint32_t particle,
                std::uint32_t visibility) {
  return {.transform = instance.transform,
          .name = instance.name,
          .asset_name = source_asset_name(snapshot, instance.name,
                                          instance.cycles_asset_name),
          .color = instance.object_color,
          .alpha = instance.object_alpha,
          .pass_id = instance.object_pass_id,
          .random_id = instance.cycles_random_id.value_or(
              quantize_legacy_random(instance.random)),
          .particle_index = particle,
          .dupli_generated = instance.dupli_generated,
          .dupli_uv = instance.dupli_uv,
          .motion_offset = 0u,
          .transform_motion_steps = 0u,
          .has_object_motion = false,
          .shadow_terminator_shading_offset =
              instance.shadow_terminator_shading_offset,
          .shadow_terminator_geometry_offset =
              instance.shadow_terminator_geometry_offset,
          .ao_distance = instance.ambient_occlusion_distance,
          .lightgroup = instance.cycles_light_group,
          .visibility = visibility,
          .use_holdout = instance.use_holdout,
          .is_shadow_catcher = instance.is_shadow_catcher,
          .is_caustics_caster = instance.is_caustics_caster,
          .is_caustics_receiver = instance.is_caustics_receiver};
}

[[nodiscard]] KernelObjectSource
light_source(const contract::SceneSnapshot &snapshot,
             const contract::LightDesc &light, std::uint32_t particle,
             std::uint32_t visibility) {
  return {.transform = light.transform,
          .name = light.name,
          .asset_name =
              source_asset_name(snapshot, light.name, light.cycles_asset_name),
          .color = light.object_color,
          .alpha = light.object_alpha,
          .pass_id = light.object_pass_id,
          .random_id = light.cycles_random_id.value_or(
              quantize_legacy_random(light.object_random)),
          .particle_index = particle,
          .dupli_generated = light.dupli_generated,
          .dupli_uv = light.dupli_uv,
          .motion_offset = 0u,
          .transform_motion_steps = 0u,
          .has_object_motion = false,
          .shadow_terminator_shading_offset =
              light.shadow_terminator_shading_offset,
          .shadow_terminator_geometry_offset =
              light.shadow_terminator_geometry_offset,
          .ao_distance = 0.0f,
          .lightgroup = light.cycles_light_group,
          .visibility = visibility,
          .use_holdout = light.use_holdout,
          .is_shadow_catcher = light.is_shadow_catcher,
          .is_caustics_caster = light.is_caustics_caster,
          .is_caustics_receiver = light.is_caustics_receiver};
}

[[nodiscard]] KernelObjectSource
background_source(const contract::SceneSnapshot &snapshot) {
  KernelObjectSource source;
  source.asset_name = snapshot.cycles_background_asset_name;
  source.shadow_terminator_geometry_offset = 0.1f;
  source.lightgroup = snapshot.cycles_background_light_group;
  source.visibility = PATH_RAY_VISIBILITY_ALL;
  source.is_shadow_catcher = true;
  return source;
}

} // namespace

CyclesSvmObjectSceneImage build_cycles_svm_object_scene_image(
    const contract::SceneSnapshot &snapshot,
    const ObjectIdentityPlan &object_identities,
    const ParticleTableImage &particles,
    const CyclesSvmGeometrySceneImage &geometry_image,
    std::span<const CyclesInstanceIntersectionPlan> intersection_plans,
    std::span<const GeometryUpload> mesh_uploads,
    const std::map<contract::GeometryId, std::uint32_t>
        &resource_geometry_indices) {
  if (!object_identities.valid) {
    return reject("Cycles object image received an invalid object domain");
  }
  if (!particles.valid) {
    return reject("Cycles object image received an invalid particle table");
  }
  if (!geometry_image.valid || !geometry_image.attributes.valid) {
    return reject("Cycles object image received an invalid geometry image");
  }
  if (snapshot.cycles_uses_light_linking) {
    return reject("Cycles light/shadow linking is enabled, but the scene "
                  "contract does not preserve native object link sets");
  }
  if (intersection_plans.size() != snapshot.instances.size()) {
    return reject("Cycles object image received an intersection plan with the "
                  "wrong instance extent");
  }

  std::map<contract::GeometryId, bool> geometry_volumes;
  std::string diagnostic;
  for (const auto &[id, geometry] : snapshot.geometries) {
    bool has_volume = false;
    if (!geometry_has_volume(snapshot, id, geometry, has_volume, diagnostic)) {
      return reject(std::move(diagnostic));
    }
    geometry_volumes.emplace(id, has_volume);
  }
  for (const auto &[id, geometry] : snapshot.curve_geometries) {
    bool has_volume = false;
    if (!geometry_has_volume(snapshot, id, geometry, has_volume, diagnostic)) {
      return reject(std::move(diagnostic));
    }
    geometry_volumes.emplace(id, has_volume);
  }

  std::vector<std::optional<PendingSceneObject>> pending(
      object_identities.object_count);
  const auto install = [&](std::uint32_t index, KernelObjectSource source,
                           FinalizedKernelObjectGeometry geometry,
                           ObjectBounds bounds) -> std::optional<std::string> {
    if (index >= pending.size()) {
      return "Cycles represented object lies outside its dense domain";
    }
    if (pending[index]) {
      return "Cycles represented object index was installed more than once";
    }
    auto object = prepare_kernel_object(std::move(source));
    if (!object.valid) {
      return object.diagnostic;
    }
    pending[index] = PendingSceneObject{.object = std::move(object),
                                        .geometry = std::move(geometry),
                                        .bounds = bounds};
    return std::nullopt;
  };

  auto plan_index = std::size_t{};
  for (const auto &[instance_id, instance] : snapshot.instances) {
    if (!instance.motion.empty()) {
      return reject("Cycles object motion is present, but the native "
                    "object_motion table is not finalized");
    }
    const auto &plan = intersection_plans[plan_index++];
    if (plan.instance != instance_id ||
        plan.world_to_object != cycles_inverse_transform(instance.transform)) {
      return reject("Cycles object image received a permuted or inconsistent "
                    "instance intersection plan");
    }
    const auto identity = object_identities.instance_indices.find(instance_id);
    if (identity == object_identities.instance_indices.end()) {
      return reject("Cycles object image cannot resolve an instance identity");
    }
    const auto resolved_particle = particle_index(particles, identity->second);
    if (!resolved_particle) {
      return reject("Cycles particle table omits an instance object");
    }
    const auto visibility = cycles_visibility(instance.visibility_mask);
    if (!visibility) {
      return reject("Cycles instance visibility leaves the contract domain");
    }

    FinalizedKernelObjectGeometry geometry;
    ObjectBounds bounds;
    if (const auto mesh = snapshot.geometries.find(instance.geometry);
        mesh != snapshot.geometries.end()) {
      const auto resource = resource_geometry_indices.find(instance.geometry);
      const auto image_index =
          geometry_image.attribute_geometry_indices.find(instance.geometry);
      if (resource == resource_geometry_indices.end() ||
          resource->second >= mesh_uploads.size() ||
          image_index == geometry_image.attribute_geometry_indices.end()) {
        return reject("Cycles object image cannot resolve finalized mesh "
                      "geometry " +
                      std::to_string(instance.geometry.value));
      }
      const auto attributes =
          finalized_attributes(geometry_image, image_index->second);
      if (!attributes) {
        return reject("Cycles mesh attribute geometry index is out of range");
      }
      const auto &upload = mesh_uploads[resource->second];
      geometry = geometry_record(
          *attributes, upload.positions.size(), upload.triangles.size(),
          PRIMITIVE_TRIANGLE, plan.transform_applied,
          (upload.attribute_domains & geometry_normal_corner) != 0u,
          geometry_volumes.at(instance.geometry));
      bounds = mesh_bounds(upload, instance, plan.transform_applied);
    } else if (const auto curve =
                   snapshot.curve_geometries.find(instance.geometry);
               curve != snapshot.curve_geometries.end()) {
      if (plan.transform_applied) {
        return reject("Cycles object image received a statically transformed "
                      "curve without transformed radius storage");
      }
      const auto image_index =
          geometry_image.attribute_geometry_indices.find(instance.geometry);
      if (image_index == geometry_image.attribute_geometry_indices.end()) {
        return reject("Cycles object image cannot resolve finalized curve "
                      "geometry " +
                      std::to_string(instance.geometry.value));
      }
      const auto attributes =
          finalized_attributes(geometry_image, image_index->second);
      if (!attributes) {
        return reject("Cycles curve attribute geometry index is out of range");
      }
      geometry =
          geometry_record(*attributes, curve->second.keys.size(), 0u,
                          cycles_svm_curve_primitive_type(curve->second.shape),
                          false, false, geometry_volumes.at(instance.geometry));
      bounds = curve_bounds(curve->second, instance);
    } else {
      return reject("Cycles object image cannot resolve instance geometry " +
                    std::to_string(instance.geometry.value));
    }

    if (auto error = install(identity->second,
                             instance_source(snapshot, instance,
                                             *resolved_particle, *visibility),
                             geometry, bounds)) {
      return reject("Cycles instance object " +
                    std::to_string(identity->second) + ": " + *error);
    }
  }

  for (const auto &[light_id, light] : snapshot.lights) {
    const auto identity = object_identities.light_indices.find(light_id);
    const auto image_index =
        geometry_image.light_attribute_geometry_indices.find(light_id);
    if (identity == object_identities.light_indices.end() ||
        image_index == geometry_image.light_attribute_geometry_indices.end()) {
      return reject("Cycles object image cannot resolve an analytic light");
    }
    const auto resolved_particle = particle_index(particles, identity->second);
    const auto attributes =
        finalized_attributes(geometry_image, image_index->second);
    const auto visibility = cycles_visibility(light.visibility_mask);
    if (!resolved_particle || !attributes || !visibility) {
      return reject("Cycles analytic-light object has unresolved native state");
    }
    bool has_volume = false;
    if (light.shader &&
        !material_has_volume(snapshot, *light.shader, has_volume, diagnostic)) {
      return reject(std::move(diagnostic));
    }
    auto geometry = geometry_record(*attributes, 0u, 0u, PRIMITIVE_LAMP, false,
                                    false, has_volume);
    if (auto error = install(
            identity->second,
            light_source(snapshot, light, *resolved_particle, *visibility),
            geometry, {})) {
      return reject("Cycles light object " + std::to_string(identity->second) +
                    ": " + *error);
    }
  }

  if (object_identities.background_index) {
    if (!geometry_image.background_attribute_geometry_index) {
      return reject("Cycles object image cannot resolve background geometry");
    }
    const auto attributes = finalized_attributes(
        geometry_image, *geometry_image.background_attribute_geometry_index);
    const auto resolved_particle =
        particle_index(particles, *object_identities.background_index);
    if (!attributes || !resolved_particle) {
      return reject("Cycles background object has unresolved native state");
    }
    bool has_volume = false;
    if (snapshot.world_shader &&
        !material_has_volume(snapshot, *snapshot.world_shader, has_volume,
                             diagnostic)) {
      return reject(std::move(diagnostic));
    }
    auto geometry = geometry_record(*attributes, 0u, 0u, PRIMITIVE_LAMP, false,
                                    false, has_volume);
    if (auto error = install(*object_identities.background_index,
                             background_source(snapshot), geometry, {})) {
      return reject("Cycles background object: " + *error);
    }
  }

  auto installed_count = std::size_t{};
  for (auto index = std::size_t{}; index < pending.size(); ++index) {
    if (!pending[index]) {
      if (object_identities.occupied(static_cast<std::uint32_t>(index))) {
        return reject("Cycles occupied object identity was not finalized");
      }
      continue;
    }
    ++installed_count;
    if (!object_identities.occupied(static_cast<std::uint32_t>(index))) {
      return reject("Cycles object image populated a source-domain hole");
    }
  }
  if (installed_count != object_identities.occupied_indices.size()) {
    return reject("Cycles object image and identity domain have different "
                  "represented extents");
  }

  std::vector<std::size_t> volume_objects;
  for (auto index = std::size_t{}; index < pending.size(); ++index) {
    if (pending[index] && pending[index]->geometry.has_volume) {
      volume_objects.emplace_back(index);
    }
  }
  for (auto index = std::size_t{}; index < pending.size(); ++index) {
    if (!pending[index]) {
      continue;
    }
    for (const auto volume_index : volume_objects) {
      if (index != volume_index &&
          intersects(pending[index]->bounds, pending[volume_index]->bounds)) {
        pending[index]->geometry.intersects_volume = true;
        break;
      }
    }
  }

  CyclesSvmObjectSceneImage result;
  result.objects.resize(object_identities.object_count);
  result.object_flags.resize(object_identities.object_count);
  for (auto index = std::size_t{}; index < pending.size(); ++index) {
    if (!pending[index]) {
      continue;
    }
    const auto finalized = finalize_kernel_object(pending[index]->object,
                                                  pending[index]->geometry);
    if (!finalized.valid) {
      return reject("Cycles object " + std::to_string(index) + ": " +
                    finalized.diagnostic);
    }
    result.objects[index] = finalized.object;
    result.object_flags[index] = finalized.object_flag;
  }
  result.valid = true;
  return result;
}

} // namespace psycles::luisa_backend::detail

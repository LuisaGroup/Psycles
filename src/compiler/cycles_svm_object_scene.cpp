#include <psycles/compiler/cycles_svm_object_scene.h>

#include <psycles/compiler/cycles_transform.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] ObjectIdentityPlan reject(std::string diagnostic) {
  ObjectIdentityPlan result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] ParticleTableImage reject_particles(std::string diagnostic) {
  ParticleTableImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] PendingKernelObject reject_object(std::string diagnostic) {
  PendingKernelObject result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] FinalizedKernelObject
reject_final_object(std::string diagnostic) {
  FinalizedKernelObject result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] PackedTransform pack_transform(const Mat4f &matrix) noexcept {
  const auto &e = matrix.elements;
  return {
      .x = {e[0u], e[4u], e[8u], e[12u]},
      .y = {e[1u], e[5u], e[9u], e[13u]},
      .z = {e[2u], e[6u], e[10u], e[14u]},
  };
}

[[nodiscard]] Vec3f cross(Vec3f lhs, Vec3f rhs) noexcept {
  return {lhs.y * rhs.z - lhs.z * rhs.y,
          lhs.z * rhs.x - lhs.x * rhs.z,
          lhs.x * rhs.y - lhs.y * rhs.x};
}

[[nodiscard]] float dot(Vec3f lhs, Vec3f rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

[[nodiscard]] bool transform_has_negative_scale(
    const Mat4f &transform) noexcept {
  const auto &e = transform.elements;
  const auto c0 = Vec3f{e[0u], e[1u], e[2u]};
  const auto c1 = Vec3f{e[4u], e[5u], e[6u]};
  const auto c2 = Vec3f{e[8u], e[9u], e[10u]};
  return dot(cross(c0, c1), c2) < 0.0f;
}

[[nodiscard]] std::uint32_t murmur_hash3(std::string_view value) noexcept {
  constexpr auto c1 = std::uint32_t{0xcc9e2d51u};
  constexpr auto c2 = std::uint32_t{0x1b873593u};
  auto h1 = std::uint32_t{};
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(value.data());
  const auto blocks = value.size() / 4u;
  for (auto i = std::size_t{}; i < blocks; ++i) {
    const auto base = i * 4u;
    auto k1 = static_cast<std::uint32_t>(bytes[base]) |
              (static_cast<std::uint32_t>(bytes[base + 1u]) << 8u) |
              (static_cast<std::uint32_t>(bytes[base + 2u]) << 16u) |
              (static_cast<std::uint32_t>(bytes[base + 3u]) << 24u);
    k1 *= c1;
    k1 = std::rotl(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = std::rotl(h1, 13);
    h1 = h1 * 5u + 0xe6546b64u;
  }
  const auto *tail = bytes + blocks * 4u;
  auto k1 = std::uint32_t{};
  switch (value.size() & 3u) {
  case 3u:
    k1 ^= static_cast<std::uint32_t>(tail[2u]) << 16u;
    [[fallthrough]];
  case 2u:
    k1 ^= static_cast<std::uint32_t>(tail[1u]) << 8u;
    [[fallthrough]];
  case 1u:
    k1 ^= static_cast<std::uint32_t>(tail[0u]);
    k1 *= c1;
    k1 = std::rotl(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    break;
  default:
    break;
  }
  h1 ^= static_cast<std::uint32_t>(value.size());
  h1 ^= h1 >> 16u;
  h1 *= 0x85ebca6bu;
  h1 ^= h1 >> 13u;
  h1 *= 0xc2b2ae35u;
  h1 ^= h1 >> 16u;
  return h1;
}

[[nodiscard]] float cryptomatte_hash(std::string_view value) noexcept {
  const auto hash = murmur_hash3(value);
  const auto mantissa = hash & ((1u << 23u) - 1u);
  auto exponent = (hash >> 23u) & ((1u << 8u) - 1u);
  exponent = std::clamp(exponent, 1u, 254u) << 23u;
  const auto sign = (hash >> 31u) << 31u;
  return std::bit_cast<float>(sign | exponent | mantissa);
}

[[nodiscard]] std::string instance_label(contract::InstanceId id) {
  return "instance " + std::to_string(id.value);
}

[[nodiscard]] std::string light_label(contract::LightId id) {
  return "light " + std::to_string(id.value);
}

} // namespace

ObjectIdentityPlan
plan_object_identities(const contract::SceneSnapshot &scene) {
  const auto declared_count = scene.cycles_object_count;
  const auto has_background = scene.world_shader.has_value() ||
                              scene.cycles_background_object_index.has_value();
  std::map<std::uint32_t, std::string> owners;

  const auto claim = [&](std::uint32_t index,
                         std::string owner) -> std::optional<std::string> {
    if (declared_count && index >= *declared_count) {
      return owner + " has Cycles object index " + std::to_string(index) +
             " outside declared object domain [0, " +
             std::to_string(*declared_count) + ")";
    }
    if (!declared_count &&
        index == std::numeric_limits<std::uint32_t>::max()) {
      return owner + " has Cycles object index UINT32_MAX, whose dense "
                     "domain extent is not representable";
    }
    const auto [iter, inserted] = owners.emplace(index, owner);
    if (!inserted) {
      return owner + " and " + iter->second +
             " name the same Cycles object index " +
             std::to_string(index);
    }
    return std::nullopt;
  };

  ObjectIdentityPlan result;
  for (const auto &[instance_id, instance] : scene.instances) {
    if (instance.cycles_object_index) {
      if (auto error = claim(*instance.cycles_object_index,
                             instance_label(instance_id))) {
        return reject(std::move(*error));
      }
      result.instance_indices.emplace(instance_id,
                                      *instance.cycles_object_index);
    } else if (declared_count) {
      return reject(instance_label(instance_id) +
                    " omits its index in a declared Cycles object domain");
    }
  }
  for (const auto &[light_id, light] : scene.lights) {
    if (light.cycles_object_index) {
      if (auto error =
              claim(*light.cycles_object_index, light_label(light_id))) {
        return reject(std::move(*error));
      }
      result.light_indices.emplace(light_id, *light.cycles_object_index);
    } else if (declared_count) {
      return reject(light_label(light_id) +
                    " omits its index in a declared Cycles object domain");
    }
  }
  if (has_background) {
    if (scene.cycles_background_object_index) {
      if (auto error = claim(*scene.cycles_background_object_index,
                             "background object")) {
        return reject(std::move(*error));
      }
      result.background_index = *scene.cycles_background_object_index;
    } else if (declared_count) {
      return reject("background object omits its index in a declared Cycles "
                    "object domain");
    }
  }

  if (declared_count && has_background &&
      *result.background_index + 1u != *declared_count) {
    return reject("Cycles background object is not the final entry in the "
                  "declared object domain");
  }

  if (!declared_count) {
    auto candidate = std::uint64_t{};
    const auto allocate = [&]() -> std::optional<std::uint32_t> {
      constexpr auto maximum_index =
          static_cast<std::uint64_t>(
              std::numeric_limits<std::uint32_t>::max()) -
          1u;
      while (candidate <= maximum_index &&
             owners.contains(static_cast<std::uint32_t>(candidate))) {
        ++candidate;
      }
      if (candidate > maximum_index) {
        return std::nullopt;
      }
      const auto index = static_cast<std::uint32_t>(candidate++);
      return index;
    };

    for (const auto &[instance_id, instance] : scene.instances) {
      if (instance.cycles_object_index) {
        continue;
      }
      const auto index = allocate();
      if (!index) {
        return reject("renderer-authored Cycles object domain is exhausted");
      }
      owners.emplace(*index, instance_label(instance_id));
      result.instance_indices.emplace(instance_id, *index);
    }
    for (const auto &[light_id, light] : scene.lights) {
      if (light.cycles_object_index) {
        continue;
      }
      const auto index = allocate();
      if (!index) {
        return reject("renderer-authored Cycles object domain is exhausted");
      }
      owners.emplace(*index, light_label(light_id));
      result.light_indices.emplace(light_id, *index);
    }
    if (has_background && !scene.cycles_background_object_index) {
      const auto index = allocate();
      if (!index) {
        return reject("renderer-authored Cycles object domain is exhausted");
      }
      owners.emplace(*index, "background object");
      result.background_index = *index;
    }
  }

  result.object_count = declared_count.value_or(
      owners.empty() ? 0u : std::prev(owners.end())->first + 1u);
  for (const auto &[index, owner] : owners) {
    static_cast<void>(owner);
    result.occupied_indices.emplace(index);
  }
  result.valid = true;
  return result;
}

ParticleTableImage
pack_particle_table(std::span<const ParticleTableObject> objects) {
  std::vector<ParticleTableObject> ordered{objects.begin(), objects.end()};
  std::sort(ordered.begin(), ordered.end(), [](const auto &lhs,
                                                const auto &rhs) noexcept {
    return lhs.object_index < rhs.object_index;
  });
  for (auto index = std::size_t{1u}; index < ordered.size(); ++index) {
    if (ordered[index - 1u].object_index == ordered[index].object_index) {
      return reject_particles("duplicate Cycles object identity in particle "
                              "table input");
    }
  }

  struct PendingReference {
    std::uint32_t object_index{};
    std::uint32_t system{};
    std::uint32_t local_index{};
  };
  struct Group {
    std::uint32_t source_system{};
    std::vector<contract::CyclesParticleSource> particles;
  };

  std::vector<Group> groups;
  std::map<std::uint32_t, std::size_t> group_indices;
  std::vector<PendingReference> pending;
  ParticleTableImage result;
  for (const auto &object : ordered) {
    result.object_particle_indices.emplace(object.object_index, 0u);
    if (!object.needs_particle || !object.source) {
      continue;
    }
    const auto system = object.source->system;
    auto [iter, inserted] = group_indices.emplace(system, groups.size());
    if (inserted) {
      groups.emplace_back(
          Group{.source_system = system, .particles = {}});
    }
    auto &group = groups[iter->second];
    if (group.particles.size() >=
        std::numeric_limits<std::uint32_t>::max()) {
      return reject_particles("Cycles particle-system local index overflows "
                              "uint32");
    }
    const auto local_index =
        static_cast<std::uint32_t>(group.particles.size());
    group.particles.emplace_back(*object.source);
    pending.emplace_back(PendingReference{
        .object_index = object.object_index,
        .system = system,
        .local_index = local_index});
  }

  // Cycles ParticleSystemManager always creates dummy entry zero, including
  // scenes with no qualifying particle systems.
  result.particles.emplace_back(contract::CyclesParticleSource{});
  std::map<std::uint32_t, std::uint32_t> group_offsets;
  for (const auto &group : groups) {
    const auto offset = result.particles.size();
    if (offset > std::numeric_limits<std::uint32_t>::max() ||
        group.particles.size() >
            std::numeric_limits<std::uint32_t>::max() - offset) {
      return reject_particles("Cycles global particle table overflows uint32");
    }
    group_offsets.emplace(group.source_system,
                          static_cast<std::uint32_t>(offset));
    result.particles.insert(result.particles.end(), group.particles.begin(),
                            group.particles.end());
  }
  for (const auto &reference : pending) {
    result.object_particle_indices.at(reference.object_index) =
        group_offsets.at(reference.system) + reference.local_index;
  }
  result.valid = true;
  return result;
}

PendingKernelObject prepare_kernel_object(KernelObjectSource source) {
  if (source.name.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      source.asset_name.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return reject_object("Cycles cryptomatte object name exceeds int32 length");
  }
  if (source.particle_index >
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    return reject_object("Cycles particle table index exceeds int32");
  }
  if ((source.visibility & ~PATH_RAY_VISIBILITY_ALL) != 0u) {
    return reject_object("Cycles object visibility leaves the low 7-bit domain");
  }
  if (source.transform_motion_steps >
      std::numeric_limits<std::uint16_t>::max()) {
    return reject_object("Cycles object transform step count exceeds uint16");
  }

  auto flags = std::uint32_t{};
  if (transform_has_negative_scale(source.transform)) {
    flags |= SD_OBJECT_NEGATIVE_SCALE;
  }
  if (source.has_object_motion) {
    flags |= SD_OBJECT_MOTION;
  }
  if (source.use_holdout) {
    flags |= SD_OBJECT_HOLDOUT_MASK;
  }
  if (source.is_shadow_catcher) {
    flags |= SD_OBJECT_SHADOW_CATCHER;
  }
  if (source.is_caustics_caster) {
    flags |= SD_OBJECT_CAUSTICS_CASTER;
  }
  if (source.is_caustics_receiver) {
    flags |= SD_OBJECT_CAUSTICS_RECEIVER;
  }

  PendingKernelObject result;
  result.valid = true;
  result.source = std::move(source);
  result.tfm = pack_transform(result.source.transform);
  result.itfm = pack_transform(
      ::psycles::compiler::cycles_inverse_affine_transform(
          result.source.transform));
  result.cryptomatte_object = cryptomatte_hash(result.source.name);
  result.cryptomatte_asset = cryptomatte_hash(result.source.asset_name);
  result.object_flag = flags;
  return result;
}

FinalizedKernelObject finalize_kernel_object(
    const PendingKernelObject &pending,
    const FinalizedKernelObjectGeometry &geometry) {
  if (!pending.valid) {
    return reject_final_object("cannot finalize rejected Cycles object: " +
                               pending.diagnostic);
  }
  if (!geometry.attribute_map_offset || !geometry.position_offset ||
      !geometry.normal_offset) {
    return reject_final_object(
        "Cycles geometry attribute offsets have not been finalized");
  }
  if (geometry.geometry_motion_steps >
      std::numeric_limits<std::uint16_t>::max()) {
    return reject_final_object("Cycles geometry motion step count exceeds uint16");
  }
  if (geometry.numverts >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      geometry.numprims >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return reject_final_object("Cycles object geometry count exceeds int32");
  }

  const auto &source = pending.source;
  KernelObject object{};
  object.tfm = pending.tfm;
  object.itfm = pending.itfm;
  object.volume_density = geometry.volume_density;
  object.pass_id = static_cast<float>(source.pass_id);
  object.random_number =
      static_cast<float>(source.random_id) *
      (1.0f / static_cast<float>(0xffffffffu));
  object.color = {source.color.x, source.color.y, source.color.z};
  object.alpha = source.alpha;
  object.particle_index = static_cast<std::int32_t>(source.particle_index);
  object.dupli_generated = {source.dupli_generated.x,
                            source.dupli_generated.y,
                            source.dupli_generated.z};
  object.dupli_uv = {source.dupli_uv.x, source.dupli_uv.y};
  object.num_geom_steps =
      static_cast<std::uint16_t>(geometry.geometry_motion_steps);
  object.num_tfm_steps =
      static_cast<std::uint16_t>(source.transform_motion_steps);
  object.numverts = static_cast<std::int32_t>(geometry.numverts);
  object.numprims = static_cast<std::int32_t>(geometry.numprims);
  object.attribute_map_offset = *geometry.attribute_map_offset;
  object.motion_offset = source.motion_offset;
  object.position_offset = *geometry.position_offset;
  object.normal_offset = *geometry.normal_offset;
  object.cryptomatte_object = pending.cryptomatte_object;
  object.cryptomatte_asset = pending.cryptomatte_asset;
  object.shadow_terminator_shading_offset =
      1.0f / (1.0f - 0.5f * source.shadow_terminator_shading_offset);
  object.shadow_terminator_geometry_offset =
      source.shadow_terminator_geometry_offset;
  object.ao_distance = source.ao_distance;
  object.lightgroup = source.lightgroup;
  object.visibility = source.visibility |
                      (source.is_shadow_catcher ? source.visibility << 16u : 0u);
  object.primitive_type = geometry.primitive_type;
  object.velocity_scale = geometry.velocity_scale;
  object.light_set_membership = source.light_set_membership;
  object.receiver_light_set =
      source.receiver_light_set >= LIGHT_LINK_SET_MAX
          ? 0u
          : source.receiver_light_set;
  object.shadow_set_membership = source.shadow_set_membership;
  object.blocker_shadow_set =
      source.blocker_shadow_set >= LIGHT_LINK_SET_MAX
          ? 0u
          : source.blocker_shadow_set;

  auto flags = pending.object_flag;
  flags |= geometry.transform_applied ? SD_OBJECT_TRANSFORM_APPLIED : 0u;
  flags |= geometry.has_vertex_motion ? SD_OBJECT_HAS_VERTEX_MOTION : 0u;
  flags |= geometry.has_corner_normals ? SD_OBJECT_HAS_CORNER_NORMALS : 0u;
  flags |= geometry.has_volume ? SD_OBJECT_HAS_VOLUME : 0u;
  flags |= geometry.intersects_volume ? SD_OBJECT_INTERSECTS_VOLUME : 0u;
  flags |= geometry.has_volume_attributes
               ? SD_OBJECT_HAS_VOLUME_ATTRIBUTES
               : 0u;
  flags |= geometry.has_volume_motion ? SD_OBJECT_HAS_VOLUME_MOTION : 0u;

  FinalizedKernelObject result;
  result.valid = true;
  result.object = object;
  result.object_flag = flags;
  return result;
}

} // namespace psycles::compiler::cycles_svm

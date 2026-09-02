#include <psycles/compiler/cycles_svm_object_scene.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
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

} // namespace psycles::compiler::cycles_svm

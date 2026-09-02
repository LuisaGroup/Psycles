#include <psycles/compiler/cycles_svm_object_scene.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] ObjectIdentityPlan reject(std::string diagnostic) {
  ObjectIdentityPlan result;
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

} // namespace psycles::compiler::cycles_svm

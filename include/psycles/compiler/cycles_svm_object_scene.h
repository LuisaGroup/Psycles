#pragma once

#include <psycles/contract/scene.h>

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace psycles::compiler::cycles_svm {

// A Cycles Scene::objects identity is a coordinate in one scene-global dense
// domain. It is not a Psycles instance/light array offset: unsupported source
// objects may leave holes, while geometry, analytic lights, and the synthetic
// background object all share the same index space.
struct ObjectIdentityPlan {
  bool valid{};
  std::string diagnostic;
  std::uint32_t object_count{};
  std::map<contract::InstanceId, std::uint32_t> instance_indices;
  std::map<contract::LightId, std::uint32_t> light_indices;
  std::optional<std::uint32_t> background_index;
  std::set<std::uint32_t> occupied_indices;

  [[nodiscard]] bool occupied(std::uint32_t index) const noexcept {
    return occupied_indices.contains(index);
  }
};

// Preserve an exported Cycles object domain exactly when object_count is
// declared. Every represented object must then carry its source index; holes
// are retained and never compacted. Renderer-authored scenes without a source
// domain receive the least available indices in stable instance, light, then
// background order after all explicit indices have been reserved.
[[nodiscard]] ObjectIdentityPlan
plan_object_identities(const contract::SceneSnapshot &scene);

} // namespace psycles::compiler::cycles_svm

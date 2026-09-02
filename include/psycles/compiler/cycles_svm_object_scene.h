#pragma once

#include <psycles/contract/scene.h>

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

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

struct ParticleTableObject {
  std::uint32_t object_index{};
  bool needs_particle{};
  std::optional<contract::CyclesParticleSource> source;
};

// Exact host image of Cycles' particles array plus the final device-table
// index stored in each represented KernelObject. Element zero is always the
// dummy particle. Source system IDs are equality keys only; group order is
// induced by the first qualifying object in Cycles object-sync order.
struct ParticleTableImage {
  bool valid{};
  std::string diagnostic;
  std::vector<contract::CyclesParticleSource> particles;
  std::map<std::uint32_t, std::uint32_t> object_particle_indices;
};

// Preserve an exported Cycles object domain exactly when object_count is
// declared. Every represented object must then carry its source index; holes
// are retained and never compacted. Renderer-authored scenes without a source
// domain receive the least available indices in stable instance, light, then
// background order after all explicit indices have been reserved.
[[nodiscard]] ObjectIdentityPlan
plan_object_identities(const contract::SceneSnapshot &scene);

[[nodiscard]] ParticleTableImage
pack_particle_table(std::span<const ParticleTableObject> objects);

} // namespace psycles::compiler::cycles_svm

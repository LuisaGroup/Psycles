#pragma once

#include <psycles/compiler/cycles_svm_types.h>
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

// Non-geometry inputs copied by Cycles' device_update_object_transform. The
// raw integer random identity is retained until packing because converting it
// to float earlier loses the exact Cycles uint-to-float rounding operation.
struct KernelObjectSource {
  Mat4f transform;
  std::string name;
  std::string asset_name;
  Vec3f color{};
  float alpha{};
  std::int32_t pass_id{};
  std::uint32_t random_id{};
  std::uint32_t particle_index{};
  Vec3f dupli_generated{};
  Vec2f dupli_uv{};
  std::uint32_t motion_offset{};
  std::uint32_t transform_motion_steps{};
  bool has_object_motion{};
  float shadow_terminator_shading_offset{};
  float shadow_terminator_geometry_offset{};
  float ao_distance{};
  std::int32_t lightgroup{-1};
  // Exact low 7-bit Cycles PathRayVisibilityFlag domain. Shadow-catcher
  // duplication into the high 16 bits is performed only during finalization.
  std::uint32_t visibility{PATH_RAY_VISIBILITY_ALL};
  bool use_holdout{};
  bool is_shadow_catcher{};
  bool is_caustics_caster{};
  bool is_caustics_receiver{};
  std::uint64_t light_set_membership{LIGHT_LINK_MASK_ALL};
  std::uint32_t receiver_light_set{};
  std::uint64_t shadow_set_membership{LIGHT_LINK_MASK_ALL};
  std::uint32_t blocker_shadow_set{};
};

// Values owned by finalized geometry/attribute images. Optionals distinguish
// "not finalized" from Cycles' legitimate ATTR_STD_NOT_FOUND sentinel.
struct FinalizedKernelObjectGeometry {
  float volume_density{1.0f};
  std::uint32_t geometry_motion_steps{};
  std::uint64_t numverts{};
  std::uint64_t numprims{};
  std::optional<std::uint32_t> attribute_map_offset;
  std::optional<std::int32_t> position_offset;
  std::optional<std::int32_t> normal_offset;
  std::int32_t primitive_type{PRIMITIVE_NONE};
  float velocity_scale{};
  bool transform_applied{};
  bool has_vertex_motion{};
  bool has_corner_normals{};
  bool has_volume{};
  bool intersects_volume{};
  bool has_volume_attributes{};
  bool has_volume_motion{};
};

// A pending object contains no KernelObject and therefore cannot accidentally
// be uploaded with placeholder geometry offsets. Only finalize_kernel_object
// crosses that type-state boundary.
struct PendingKernelObject {
  bool valid{};
  std::string diagnostic;
  KernelObjectSource source;
  PackedTransform tfm{};
  PackedTransform itfm{};
  float cryptomatte_object{};
  float cryptomatte_asset{};
  std::uint32_t object_flag{};
};

struct FinalizedKernelObject {
  bool valid{};
  std::string diagnostic;
  KernelObject object{};
  std::uint32_t object_flag{};
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

[[nodiscard]] PendingKernelObject
prepare_kernel_object(KernelObjectSource source);

[[nodiscard]] FinalizedKernelObject finalize_kernel_object(
    const PendingKernelObject &pending,
    const FinalizedKernelObjectGeometry &geometry);

} // namespace psycles::compiler::cycles_svm

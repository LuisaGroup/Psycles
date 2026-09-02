#include <psycles/compiler/cycles_svm_object_scene.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

[[nodiscard]] InstanceDesc instance(std::uint32_t object_index) {
  InstanceDesc result;
  result.cycles_object_index = object_index;
  return result;
}

[[nodiscard]] LightDesc light(std::uint32_t object_index) {
  LightDesc result;
  result.cycles_object_index = object_index;
  return result;
}

void test_declared_sparse_source_domain_is_not_compacted() {
  SceneSnapshot scene;
  scene.cycles_object_count = 9u;
  scene.instances.emplace(InstanceId{2u}, instance(1u));
  scene.instances.emplace(InstanceId{7u}, instance(6u));
  scene.lights.emplace(LightId{3u}, light(3u));
  scene.world_shader = MaterialId{5u};
  scene.cycles_background_object_index = 8u;

  const auto plan = plan_object_identities(scene);
  require(plan.valid, "valid sparse Cycles object domain was rejected");
  require(plan.object_count == 9u,
          "declared Cycles object extent was compacted");
  require(plan.instance_indices.at(InstanceId{2u}) == 1u &&
              plan.instance_indices.at(InstanceId{7u}) == 6u &&
              plan.light_indices.at(LightId{3u}) == 3u &&
              plan.background_index == 8u,
          "source Cycles object identities changed");
  require(plan.occupied(1u) && plan.occupied(3u) && plan.occupied(6u) &&
              plan.occupied(8u) && !plan.occupied(0u) &&
              !plan.occupied(2u) && !plan.occupied(7u),
          "unsupported source objects were not retained as holes");
}

void test_declared_domain_rejects_ambiguous_or_impossible_identity() {
  {
    SceneSnapshot scene;
    scene.cycles_object_count = 4u;
    scene.instances.emplace(InstanceId{0u}, instance(1u));
    scene.lights.emplace(LightId{0u}, light(1u));
    require(!plan_object_identities(scene).valid,
            "duplicate cross-kind Cycles object identity was accepted");
  }
  {
    SceneSnapshot scene;
    scene.cycles_object_count = 4u;
    scene.instances.emplace(InstanceId{0u}, instance(4u));
    require(!plan_object_identities(scene).valid,
            "out-of-domain Cycles object identity was accepted");
  }
  {
    SceneSnapshot scene;
    scene.cycles_object_count = 4u;
    scene.instances.emplace(InstanceId{0u}, InstanceDesc{});
    require(!plan_object_identities(scene).valid,
            "missing identity was guessed inside a source domain");
  }
  {
    SceneSnapshot scene;
    scene.cycles_object_count = 4u;
    scene.world_shader = MaterialId{0u};
    scene.cycles_background_object_index = 2u;
    require(!plan_object_identities(scene).valid,
            "non-final Cycles background object was accepted");
  }
}

void test_renderer_authored_domain_is_total_and_deterministic() {
  SceneSnapshot scene;
  scene.instances.emplace(InstanceId{3u}, InstanceDesc{});
  scene.instances.emplace(InstanceId{9u}, instance(4u));
  scene.lights.emplace(LightId{2u}, light(2u));
  scene.lights.emplace(LightId{8u}, LightDesc{});
  scene.world_shader = MaterialId{7u};

  const auto first = plan_object_identities(scene);
  const auto second = plan_object_identities(scene);
  require(first.valid && second.valid,
          "renderer-authored Cycles object domain was rejected");
  require(first.instance_indices == second.instance_indices &&
              first.light_indices == second.light_indices &&
              first.background_index == second.background_index,
          "renderer-authored object allocation was not deterministic");
  require(first.instance_indices.at(InstanceId{3u}) == 0u &&
              first.instance_indices.at(InstanceId{9u}) == 4u &&
              first.light_indices.at(LightId{2u}) == 2u &&
              first.light_indices.at(LightId{8u}) == 1u &&
              first.background_index == 3u && first.object_count == 5u,
          "least-available object allocation violated semantic order");
}

void test_unrepresentable_dense_extent_is_rejected() {
  SceneSnapshot scene;
  scene.instances.emplace(
      InstanceId{0u},
      instance(std::numeric_limits<std::uint32_t>::max()));
  require(!plan_object_identities(scene).valid,
          "UINT32_MAX object index overflowed the dense extent");
}

[[nodiscard]] CyclesParticleSource particle(std::uint32_t system,
                                             std::uint32_t source_index) {
  return {.system = system, .source_index = source_index};
}

void test_particle_table_copies_cycles_group_prefix_algebra() {
  const std::array inputs{
      ParticleTableObject{.object_index = 5u,
                          .needs_particle = true,
                          .source = particle(7u, 42u)},
      ParticleTableObject{.object_index = 1u,
                          .needs_particle = true,
                          .source = particle(9u, 3u)},
      ParticleTableObject{.object_index = 2u,
                          .needs_particle = false,
                          .source = particle(9u, 8u)},
      ParticleTableObject{.object_index = 3u,
                          .needs_particle = true,
                          .source = particle(7u, 4u)},
      ParticleTableObject{.object_index = 4u,
                          .needs_particle = true,
                          .source = std::nullopt}};
  const auto packed = pack_particle_table(inputs);
  require(packed.valid, "valid Cycles particle groups were rejected");
  require(packed.particles.size() == 4u &&
              packed.particles[0u].source_index == 0u &&
              packed.particles[1u].source_index == 3u &&
              packed.particles[2u].source_index == 4u &&
              packed.particles[3u].source_index == 42u,
          "Cycles dummy/group/local particle order changed");
  require(packed.object_particle_indices.at(1u) == 1u &&
              packed.object_particle_indices.at(2u) == 0u &&
              packed.object_particle_indices.at(3u) == 2u &&
              packed.object_particle_indices.at(4u) == 0u &&
              packed.object_particle_indices.at(5u) == 3u,
          "KernelObject particle prefix resolution changed");

  const std::array duplicates{
      ParticleTableObject{.object_index = 1u},
      ParticleTableObject{.object_index = 1u}};
  require(!pack_particle_table(duplicates).valid,
          "duplicate object input to particle packing was accepted");
}

void test_kernel_object_requires_finalized_geometry_state() {
  KernelObjectSource source;
  source.transform.elements = {
      -2.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 3.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 4.0f, 0.0f,
      5.0f, 6.0f, 7.0f, 1.0f};
  source.name = "Cube";
  source.asset_name = "Lone Monk";
  source.color = {0.125f, 0.25f, 0.5f};
  source.alpha = 0.75f;
  source.pass_id = 19;
  source.random_id = 0x12345678u;
  source.particle_index = 17u;
  source.dupli_generated = {1.0f, 2.0f, 3.0f};
  source.dupli_uv = {0.2f, 0.8f};
  source.motion_offset = 31u;
  source.transform_motion_steps = 3u;
  source.has_object_motion = true;
  source.shadow_terminator_shading_offset = 0.5f;
  source.shadow_terminator_geometry_offset = 0.625f;
  source.ao_distance = 2.5f;
  source.lightgroup = 7;
  source.visibility = PATH_RAY_VISIBILITY_CAMERA |
                      PATH_RAY_VISIBILITY_DIFFUSE |
                      PATH_RAY_VISIBILITY_SHADOW;
  source.use_holdout = true;
  source.is_shadow_catcher = true;
  source.is_caustics_caster = true;
  source.is_caustics_receiver = true;
  source.light_set_membership = 0x55aa55aa55aa55aaull;
  source.receiver_light_set = LIGHT_LINK_SET_MAX;
  source.shadow_set_membership = 0xaa55aa55aa55aa55ull;
  source.blocker_shadow_set = 63u;

  const auto pending = prepare_kernel_object(source);
  require(pending.valid, "valid Cycles object source was rejected");
  require(std::bit_cast<std::uint32_t>(pending.cryptomatte_object) ==
              0xa8fce865u &&
              std::bit_cast<std::uint32_t>(pending.cryptomatte_asset) ==
                  0x04c3b823u,
          "Cycles MurmurHash3/cryptomatte projection changed");

  FinalizedKernelObjectGeometry geometry;
  require(!finalize_kernel_object(pending, geometry).valid,
          "unresolved Cycles attribute offsets were accepted");
  geometry.volume_density = 0.25f;
  geometry.geometry_motion_steps = 2u;
  geometry.numverts = 1024u;
  geometry.numprims = 2048u;
  geometry.attribute_map_offset = 11u;
  geometry.position_offset = 123;
  geometry.normal_offset = 456;
  geometry.primitive_type = PRIMITIVE_MOTION_TRIANGLE;
  geometry.velocity_scale = 8.0f;
  geometry.transform_applied = true;
  geometry.has_vertex_motion = true;
  geometry.has_corner_normals = true;
  geometry.has_volume = true;
  geometry.intersects_volume = true;
  geometry.has_volume_attributes = true;
  geometry.has_volume_motion = true;

  const auto finalized = finalize_kernel_object(pending, geometry);
  require(finalized.valid, "finalized Cycles object was rejected");
  const auto &object = finalized.object;
  require(object.tfm.x.x == -2.0f && object.tfm.x.w == 5.0f &&
              object.tfm.y.y == 3.0f && object.tfm.y.w == 6.0f &&
              object.tfm.z.z == 4.0f && object.tfm.z.w == 7.0f,
          "Cycles object transform row packing changed");
  require(object.itfm.x.x == -0.5f && object.itfm.x.w == 2.5f &&
              object.itfm.y.y == (1.0f / 3.0f) &&
              object.itfm.y.w == -2.0f && object.itfm.z.z == 0.25f &&
              object.itfm.z.w == -1.75f,
          "Cycles inverse object transform changed");
  require(object.volume_density == 0.25f && object.pass_id == 19.0f &&
              std::bit_cast<std::uint32_t>(object.random_number) ==
                  0x3d91a2b4u &&
              object.color.x == 0.125f && object.color.y == 0.25f &&
              object.color.z == 0.5f && object.alpha == 0.75f &&
              object.particle_index == 17,
          "Cycles scalar object projection changed");
  require(object.dupli_generated.x == 1.0f &&
              object.dupli_generated.y == 2.0f &&
              object.dupli_generated.z == 3.0f &&
              object.dupli_uv.x == 0.2f && object.dupli_uv.y == 0.8f,
          "Cycles dupli coordinate projection changed");
  require(object.num_geom_steps == 2u && object.num_tfm_steps == 3u &&
              object.numverts == 1024 && object.numprims == 2048 &&
              object.attribute_map_offset == 11u &&
              object.motion_offset == 31u && object.position_offset == 123 &&
              object.normal_offset == 456,
          "Cycles finalized geometry projection changed");
  require(std::bit_cast<std::uint32_t>(object.cryptomatte_object) ==
              0xa8fce865u &&
              std::bit_cast<std::uint32_t>(object.cryptomatte_asset) ==
                  0x04c3b823u &&
              object.shadow_terminator_shading_offset ==
                  (1.0f / (1.0f - 0.5f * 0.5f)) &&
              object.shadow_terminator_geometry_offset == 0.625f,
          "Cycles cryptomatte/shadow-terminator projection changed");
  require(object.ao_distance == 2.5f && object.lightgroup == 7 &&
              object.visibility ==
                  (source.visibility | (source.visibility << 16u)) &&
              object.primitive_type == PRIMITIVE_MOTION_TRIANGLE &&
              object.velocity_scale == 8.0f,
          "Cycles visibility/primitive projection changed");
  require(object.light_set_membership == 0x55aa55aa55aa55aaull &&
              object.receiver_light_set == 0u &&
              object.shadow_set_membership == 0xaa55aa55aa55aa55ull &&
              object.blocker_shadow_set == 63u,
          "Cycles light-link clamping or masks changed");
  const auto expected_flags =
      SD_OBJECT_HOLDOUT_MASK | SD_OBJECT_MOTION |
      SD_OBJECT_TRANSFORM_APPLIED | SD_OBJECT_NEGATIVE_SCALE |
      SD_OBJECT_HAS_VOLUME | SD_OBJECT_INTERSECTS_VOLUME |
      SD_OBJECT_HAS_VERTEX_MOTION | SD_OBJECT_SHADOW_CATCHER |
      SD_OBJECT_HAS_VOLUME_ATTRIBUTES | SD_OBJECT_CAUSTICS_CASTER |
      SD_OBJECT_CAUSTICS_RECEIVER | SD_OBJECT_HAS_VOLUME_MOTION |
      SD_OBJECT_HAS_CORNER_NORMALS;
  require(finalized.object_flag == expected_flags,
          "Cycles object flag composition changed");
  require(object._pad_light_set_alignment == 0u &&
              object._pad_shadow_set_alignment == 0u &&
              object._pad_tail[0u] == 0u && object._pad_tail[1u] == 0u &&
              object._pad_tail[2u] == 0u,
          "KernelObject ABI padding was left indeterminate");
}

void test_kernel_object_domain_boundaries_are_rejected() {
  KernelObjectSource source;
  source.visibility = PATH_RAY_VISIBILITY_ALL | (1u << 7u);
  require(!prepare_kernel_object(source).valid,
          "out-of-domain Cycles visibility was accepted");
  source.visibility = PATH_RAY_VISIBILITY_ALL;
  source.transform_motion_steps = 65536u;
  require(!prepare_kernel_object(source).valid,
          "overflowing Cycles transform step count was accepted");
  source.transform_motion_steps = 0u;
  source.particle_index = 0x80000000u;
  require(!prepare_kernel_object(source).valid,
          "overflowing Cycles particle index was accepted");

  source.particle_index = 0u;
  const auto pending = prepare_kernel_object(source);
  require(pending.valid, "boundary-test Cycles source was rejected");
  FinalizedKernelObjectGeometry geometry;
  geometry.attribute_map_offset = 0u;
  geometry.position_offset = ATTR_STD_NOT_FOUND;
  geometry.normal_offset = ATTR_STD_NOT_FOUND;
  geometry.geometry_motion_steps = 65536u;
  require(!finalize_kernel_object(pending, geometry).valid,
          "overflowing Cycles geometry step count was accepted");
  geometry.geometry_motion_steps = 0u;
  geometry.numverts =
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) +
      1u;
  require(!finalize_kernel_object(pending, geometry).valid,
          "overflowing Cycles geometry size was accepted");
  geometry.numverts = 0u;
  require(finalize_kernel_object(pending, geometry).valid,
          "resolved ATTR_STD_NOT_FOUND sentinels were confused with pending offsets");
}

} // namespace

int main() {
  test_declared_sparse_source_domain_is_not_compacted();
  test_declared_domain_rejects_ambiguous_or_impossible_identity();
  test_renderer_authored_domain_is_total_and_deterministic();
  test_unrepresentable_dense_extent_is_rejected();
  test_particle_table_copies_cycles_group_prefix_algebra();
  test_kernel_object_requires_finalized_geometry_state();
  test_kernel_object_domain_boundaries_are_rejected();
  std::cout << "Cycles SVM object identity tests passed\n";
  return 0;
}

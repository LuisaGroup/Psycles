#include <psycles/compiler/cycles_svm_object_scene.h>

#include <array>
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

} // namespace

int main() {
  test_declared_sparse_source_domain_is_not_compacted();
  test_declared_domain_rejects_ambiguous_or_impossible_identity();
  test_renderer_authored_domain_is_total_and_deterministic();
  test_unrepresentable_dense_extent_is_rejected();
  test_particle_table_copies_cycles_group_prefix_algebra();
  std::cout << "Cycles SVM object identity tests passed\n";
  return 0;
}

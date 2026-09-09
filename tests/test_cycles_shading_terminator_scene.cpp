#include "path_tracer_cycles_svm_geometry.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_object.h"
#include "path_tracer_internal.h"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::luisa_backend::detail;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

void check_production_kernel_globals(const CyclesSvmObjectSceneImage &image,
                                     bool expected) {
  auto scene = std::make_shared<LuisaSceneData>();
  scene->cycles_svm = std::make_unique<CyclesSvmRuntime>();
  scene->cycles_svm->geometry = std::make_unique<CyclesSvmGeometryRuntime>();
  scene->cycles_svm->objects = std::make_unique<CyclesSvmObjectRuntime>();
  scene->cycles_svm->objects->image = image;
  // Deliberately no device or uploaded buffers. This exercises the actual
  // production constructor/cache wiring, not another fake KernelGlobals.
  // Its uniform host query must not record any object-buffer service access.
  bool recorded = false, actual = !expected;
  const luisa::compute::Kernel1D kernel = [&] {
    luisa::compute::Var<RenderKernelParameters> parameters;
    const PathCyclesSvmKernelGlobals kg{
        scene, parameters, CameraProjection::perspective, true, true};
    actual = kg.has_shadow_terminator_shading_offset();
    recorded = true;
  };
  require(recorded && actual == expected,
          "production KernelGlobals lost the finalized image capability");
  require(!kernel.function()->function().propagated_builtin_callables().test(
              luisa::compute::CallOp::BUFFER_READ),
          "host capability query recorded an object resource read");
}

// Exercise the real geometry/object transaction from authored contract data.
// Analytic lights need no mesh upload or device; they use the same finalized
// KernelObject field as geometry and deliberately participate in the proof.
CyclesSvmObjectSceneImage finalize(float first_offset, float second_offset,
                                  bool reject_geometry = false) {
  SceneSnapshot snapshot;
  snapshot.cycles_object_count = 5u;
  LightDesc first;
  first.name = "arbitrary first object";
  first.cycles_object_index = 1u;
  first.shadow_terminator_shading_offset = first_offset;
  first.shadow_terminator_geometry_offset = 0.9f;
  auto second = first;
  second.name = "different object and identity";
  second.cycles_object_index = 3u;
  second.shadow_terminator_shading_offset = second_offset;
  snapshot.lights.emplace(LightId{19u}, first);
  snapshot.lights.emplace(LightId{7u}, second);
  const auto identities = plan_object_identities(snapshot);
  require(identities.valid, identities.diagnostic);
  const std::array particle_sources{ParticleTableObject{.object_index = 1u},
                                    ParticleTableObject{.object_index = 3u}};
  const auto particles = pack_particle_table(particle_sources);
  require(particles.valid, particles.diagnostic);
  CompiledShaderTable compilation;
  compilation.table.valid = true;
  auto geometry = build_cycles_svm_geometry_scene_image(
      snapshot, compilation, {}, identities, {}, {}, {}, {}, {});
  require(geometry.valid, geometry.diagnostic);
  if (reject_geometry) { geometry.valid = false; }
  return build_cycles_svm_object_scene_image(
      snapshot, identities, particles, geometry, {}, {}, {});
}

void check_transaction_regeneration() {
  const auto off = finalize(0.0f, 0.0f);
  const auto mixed = finalize(0.0f, 1.0f);
  const auto on = finalize(1.0f, 0.5f);
  const auto off_again = finalize(0.0f, 0.0f);
  for (const auto *image : {&off, &mixed, &on, &off_again}) {
    require(image->valid, image->diagnostic);
    require(image->objects.size() == 5u, "sparse object domain changed");
  }
  require(off.objects[1u].shadow_terminator_shading_offset == 1.0f &&
              mixed.objects[3u].shadow_terminator_shading_offset == 2.0f,
          "certificate did not use the finalized native frequency");
  require(!off.has_shadow_terminator_shading_offset(), "all-off table retained frequency work");
  require(mixed.has_shadow_terminator_shading_offset(), "mixed table lost enabled object");
  require(on.has_shadow_terminator_shading_offset(), "all-on table lost frequency work");
  require(!off_again.has_shadow_terminator_shading_offset(), "new finalization reused old capability");
  require(!off.has_shadow_terminator_shading_offset(), "new table mutated an existing snapshot");
  check_production_kernel_globals(off, false);
  check_production_kernel_globals(mixed, true);
  check_production_kernel_globals(on, true);
  check_production_kernel_globals(off_again, false);
  require(off.objects[1u].shadow_terminator_geometry_offset == 0.9f,
          "unrelated geometry-offset field changed");

  const auto failed = finalize(0.0f, 0.0f, true);
  require(!failed.valid && failed.has_shadow_terminator_shading_offset(),
          "rejected geometry transaction was treated as all-off");
}

void check_unknown_and_noop_domains() {
  CyclesSvmObjectSceneImage unknown;
  require(unknown.has_shadow_terminator_shading_offset(), "unknown image was pruned");
  check_production_kernel_globals(unknown, true);
  auto image = finalize(0.0f, 0.0f);
  for (const float frequency : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()}) {
    image.objects[1u].shadow_terminator_shading_offset = frequency;
    require(image.has_shadow_terminator_shading_offset(), "nonfinite metadata was pruned");
  }
  for (const float frequency : {-2.0f, -0.0f, 0.5f, 1.0f}) {
    image.objects[1u].shadow_terminator_shading_offset = frequency;
    require(!image.has_shadow_terminator_shading_offset(), "native no-op domain was narrowed");
  }
  image.object_flags.pop_back();
  require(image.has_shadow_terminator_shading_offset(), "incomplete image was pruned");
  CyclesSvmObjectSceneImage empty;
  empty.valid = true;
  require(!empty.has_shadow_terminator_shading_offset(), "finalized empty table needs no object work");
}
} // namespace

int main() {
  try {
    check_transaction_regeneration();
    check_unknown_and_noop_domains();
    std::cout << "Finalized object frequency proof: off/mixed/on/unknown and regeneration passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

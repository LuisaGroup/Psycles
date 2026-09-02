#include "path_tracer_cycles_svm_scene.h"

#include "cycles_svm_scene_image.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool material_uses_particle(
    const CyclesSvmRuntime &runtime,
    contract::MaterialId material,
    bool &uses_particle,
    std::string &diagnostic) {
  const auto shader = runtime.material_shader_indices.find(material);
  if (shader == runtime.material_shader_indices.end() ||
      shader->second >= runtime.compilation.shader_node_types_used.size()) {
    diagnostic = "Cycles particle demand references unavailable material " +
                 std::to_string(material.value);
    return false;
  }
  const auto &node_types =
      runtime.compilation.shader_node_types_used[shader->second];
  uses_particle = node_types[compiler::cycles_svm::NODE_PARTICLE_INFO];
  return true;
}

[[nodiscard]] bool geometry_uses_particle(
    const CyclesSvmRuntime &runtime,
    const contract::SceneSnapshot &snapshot,
    const contract::InstanceDesc &instance,
    bool &uses_particle,
    std::string &diagnostic) {
  uses_particle = false;
  const std::vector<contract::MaterialId> *materials = nullptr;
  if (const auto mesh = snapshot.geometries.find(instance.geometry);
      mesh != snapshot.geometries.end()) {
    materials = &mesh->second.material_slots;
  } else if (const auto curves =
                 snapshot.curve_geometries.find(instance.geometry);
             curves != snapshot.curve_geometries.end()) {
    materials = &curves->second.material_slots;
  } else {
    diagnostic = "Cycles particle demand references unavailable geometry " +
                 std::to_string(instance.geometry.value);
    return false;
  }

  const auto include = [&](contract::MaterialId material) {
    bool shader_uses_particle = false;
    if (!material_uses_particle(runtime, material, shader_uses_particle,
                                diagnostic)) {
      return false;
    }
    uses_particle |= shader_uses_particle;
    return true;
  };
  for (const auto material : *materials) {
    if (!include(material)) {
      return false;
    }
  }
  for (const auto material : instance.material_overrides) {
    if (!include(material)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool build_particle_table(
    CyclesSvmRuntime &runtime,
    const contract::SceneSnapshot &snapshot,
    std::string &diagnostic) {
  std::vector<compiler::cycles_svm::ParticleTableObject> objects;
  objects.reserve(snapshot.instances.size() + snapshot.lights.size() + 1u);
  for (const auto &[instance_id, instance] : snapshot.instances) {
    bool needs_particle = false;
    if (!geometry_uses_particle(runtime, snapshot, instance, needs_particle,
                                diagnostic)) {
      return false;
    }
    objects.emplace_back(compiler::cycles_svm::ParticleTableObject{
        .object_index = runtime.object_identities.instance_indices.at(
            instance_id),
        .needs_particle = needs_particle,
        .source = instance.cycles_particle_source});
  }
  for (const auto &[light_id, light] : snapshot.lights) {
    bool needs_particle = false;
    if (light.shader &&
        !material_uses_particle(runtime, *light.shader, needs_particle,
                                diagnostic)) {
      return false;
    }
    objects.emplace_back(compiler::cycles_svm::ParticleTableObject{
        .object_index =
            runtime.object_identities.light_indices.at(light_id),
        .needs_particle = needs_particle,
        .source = light.cycles_particle_source});
  }
  if (runtime.object_identities.background_index) {
    objects.emplace_back(compiler::cycles_svm::ParticleTableObject{
        .object_index = *runtime.object_identities.background_index});
  }
  runtime.particles = compiler::cycles_svm::pack_particle_table(objects);
  if (!runtime.particles.valid) {
    diagnostic = runtime.particles.diagnostic;
    return false;
  }
  runtime.particle_records.reserve(runtime.particles.particles.size());
  for (const auto &particle : runtime.particles.particles) {
    runtime.particle_records.emplace_back(make_cycles_svm_particle(particle));
  }
  return true;
}

} // namespace

std::unique_ptr<CyclesSvmRuntime>
build_cycles_svm_runtime(const std::shared_ptr<LuisaSceneData> &scene,
                         const contract::SceneSnapshot &snapshot,
                         std::string &diagnostic) {
  diagnostic.clear();
  auto runtime = std::make_unique<CyclesSvmRuntime>();
  runtime->object_identities =
      compiler::cycles_svm::plan_object_identities(snapshot);
  if (!runtime->object_identities.valid) {
    diagnostic = runtime->object_identities.diagnostic;
    return nullptr;
  }

  std::set<std::uint32_t> occupied_indices;
  auto maximum_source_index = std::uint32_t{};
  auto has_source_index = false;
  for (const auto &[material_id, material] : scene->materials.materials()) {
    static_cast<void>(material);
    const auto &source = snapshot.materials.at(material_id);
    if (source.cycles_shader_index) {
      occupied_indices.emplace(*source.cycles_shader_index);
      maximum_source_index =
          std::max(maximum_source_index, *source.cycles_shader_index);
      has_source_index = true;
    }
  }

  auto next_authored_index =
      has_source_index ? static_cast<std::uint64_t>(maximum_source_index) + 1u
                       : std::uint64_t{};
  for (const auto &[material_id, material] : scene->materials.materials()) {
    const auto &source = snapshot.materials.at(material_id);
    auto shader_index = source.cycles_shader_index;
    if (!shader_index) {
      while (next_authored_index <= std::numeric_limits<std::uint32_t>::max() &&
             occupied_indices.contains(
                 static_cast<std::uint32_t>(next_authored_index))) {
        ++next_authored_index;
      }
      if (next_authored_index > std::numeric_limits<std::uint32_t>::max()) {
        diagnostic =
            "renderer-authored Cycles shader identity overflows uint32";
        return nullptr;
      }
      shader_index = static_cast<std::uint32_t>(next_authored_index++);
      occupied_indices.emplace(*shader_index);
    }
    runtime->material_shader_indices.emplace(material_id, *shader_index);
  }

  std::vector<compiler::cycles_svm::ShaderTableCompileUnit> units;
  units.reserve(scene->materials.materials().size());
  for (const auto &[material_id, material] : scene->materials.materials()) {
    units.emplace_back(compiler::cycles_svm::ShaderTableCompileUnit{
        .shader_index = runtime->material_shader_indices.at(material_id),
        .shader = &material.shader(),
        .context = {.background = snapshot.world_shader == material_id,
                    .color_space = snapshot.shader_color_space}});
  }

  runtime->compilation = compiler::cycles_svm::compile_shader_table(units);
  if (!runtime->compilation.table.valid) {
    diagnostic = runtime->compilation.table.diagnostic;
    return nullptr;
  }
  if (!build_particle_table(*runtime, snapshot, diagnostic)) {
    return nullptr;
  }
  if (!runtime->compilation.table.words.empty()) {
    runtime->word_buffer.emplace(scene->device.create_buffer<luisa::uint>(
        runtime->compilation.table.words.size()));
  }
  if (!runtime->compilation.ies.empty()) {
    runtime->ies_buffer.emplace(
        scene->device.create_buffer<float>(runtime->compilation.ies.size()));
  }
  runtime->image_bindings.reserve(runtime->compilation.images.size());
  for (const auto &binding : runtime->compilation.images) {
    if (binding.resource_id > std::numeric_limits<std::uint32_t>::max()) {
      diagnostic = "Cycles SVM image resource identity exceeds the 32-bit "
                   "Luisa bindless address space";
      return nullptr;
    }
    const contract::ImageId image_id{binding.resource_id};
    if (!snapshot.images.contains(image_id)) {
      diagnostic = "Cycles SVM image handle references unavailable scene "
                   "resource " +
                   std::to_string(binding.resource_id);
      return nullptr;
    }
    runtime->image_bindings.emplace_back(make_cycles_svm_image_binding(
        static_cast<std::uint32_t>(binding.resource_id),
        binding.interpolation, binding.extension));
  }
  if (!runtime->image_bindings.empty()) {
    runtime->image_binding_buffer.emplace(
        scene->device.create_buffer<CyclesSvmImageBindingGpu>(
            runtime->image_bindings.size()));
  }
  // Cycles always materializes dummy particle entry zero, even when no object
  // has particle data. The formal host packer therefore makes this non-empty.
  runtime->particle_buffer.emplace(
      scene->device.create_buffer<CyclesSvmParticleGpu>(
          runtime->particle_records.size()));
  return runtime;
}

void upload_cycles_svm_runtime(Stream &stream,
                               CyclesSvmRuntime &runtime) noexcept {
  if (runtime.word_buffer) {
    stream << runtime.word_buffer->copy_from(
        luisa::span{runtime.compilation.table.words});
  }
  if (runtime.ies_buffer) {
    stream << runtime.ies_buffer->copy_from(
        luisa::span{runtime.compilation.ies});
  }
  if (runtime.image_binding_buffer) {
    stream << runtime.image_binding_buffer->copy_from(
        luisa::span{runtime.image_bindings});
  }
  if (runtime.particle_buffer) {
    stream << runtime.particle_buffer->copy_from(
        luisa::span{runtime.particle_records});
  }
}

} // namespace psycles::luisa_backend::detail

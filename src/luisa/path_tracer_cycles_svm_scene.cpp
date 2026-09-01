#include "path_tracer_cycles_svm_scene.h"

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

std::unique_ptr<CyclesSvmRuntime>
build_cycles_svm_runtime(const std::shared_ptr<LuisaSceneData> &scene,
                         const contract::SceneSnapshot &snapshot,
                         std::string &diagnostic) {
  diagnostic.clear();
  auto runtime = std::make_unique<CyclesSvmRuntime>();

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
  if (!runtime->compilation.table.words.empty()) {
    runtime->word_buffer.emplace(scene->device.create_buffer<luisa::uint>(
        runtime->compilation.table.words.size()));
  }
  if (!runtime->compilation.ies.empty()) {
    runtime->ies_buffer.emplace(
        scene->device.create_buffer<float>(runtime->compilation.ies.size()));
  }
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
}

} // namespace psycles::luisa_backend::detail

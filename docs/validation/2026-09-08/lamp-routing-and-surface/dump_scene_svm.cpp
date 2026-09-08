// Compile the production used-shader domain, without evaluating any shader.
// stdout is a PSYSVM52 image for structural comparison, never an oracle.
#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include <algorithm>
#include <bit>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace psycles::luisa_backend::detail {
// The diagnostic links the production collector; do not invent another
// used-material domain or admit only the currently visible materials.
std::set<contract::MaterialId> collect_cycles_svm_shader_materials(
    const contract::SceneSnapshot &scene);
}

template<typename T>
void write_scalar(T value) {
  static_assert(std::endian::native == std::endian::little);
  std::cout.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

// Optional sidecar added by the resource-identity audit. These are the actual
// compiler's handles and imported resource descriptions, not an inferred
// resource renumbering. The historical stdout word image stays unchanged.
void write_bindings(const char *path,
                    const psycles::compiler::cycles_svm::CompiledShaderTable &table,
                    const psycles::contract::SceneSnapshot &scene) {
  std::ofstream output{path, std::ios::binary};
  const auto scalar = [&output]<typename T>(T value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
  };
  const auto text = [&output, &scalar](const std::string &value) {
    if (value.size() > UINT32_MAX) { throw std::runtime_error("string overflow"); }
    scalar(static_cast<std::uint32_t>(value.size()));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
  };
  output.write("PSYPBD52", 8);
  scalar(std::uint32_t{1});
  scalar(static_cast<std::uint32_t>(table.named_attributes.size()));
  for (const auto &[name, id] : table.named_attributes) {
    scalar(id);
    text(name);
  }
  scalar(static_cast<std::uint32_t>(table.images.size()));
  for (std::uint32_t id = 0; id < table.images.size(); ++id) {
    const auto &binding = table.images[id];
    scalar(id);
    scalar(binding.resource_id);
    scalar(static_cast<std::uint32_t>(binding.interpolation));
    scalar(static_cast<std::uint32_t>(binding.extension));
    scalar(static_cast<std::uint32_t>(binding.nishita.has_value()));
    if (binding.nishita) {
      scalar(static_cast<std::uint32_t>(binding.nishita->multiple_scattering));
      for (auto bits : binding.nishita->parameter_bits) { scalar(bits); }
    } else {
      const auto &image = scene.images.at(
          psycles::contract::ImageId{binding.resource_id});
      text(image.name);
      scalar(static_cast<std::uint32_t>(image.color_space));
      scalar(static_cast<std::uint32_t>(image.alpha_type));
      scalar(image.width);
      scalar(image.height);
      scalar(static_cast<std::uint32_t>(image.load_failed));
    }
  }
  output.flush();
  if (!output.good()) { throw std::runtime_error("binding sidecar write failed"); }
}

int main(int argc, char **argv) {
  using namespace psycles;
  namespace svm = compiler::cycles_svm;
  try {
    if (argc != 2 && argc != 3) { return 2; }
    auto imported = adapter::load_blender_scene_bundle(argv[1]);
    if (!imported.ok()) { throw std::runtime_error("scene import failed"); }
    const auto &scene = *imported.scene;
    const auto materials =
        luisa_backend::detail::collect_cycles_svm_shader_materials(scene);
    compiler::ShaderCompiler validator{compiler::make_core_node_registry()};
    std::map<contract::MaterialId, std::shared_ptr<const compiler::ShaderProgram>> programs;
    std::uint64_t next_authored = 0;
    for (auto id : materials) {
      const auto &source = scene.materials.at(id);
      if (source.cycles_shader_index) {
        next_authored = std::max(next_authored,
            static_cast<std::uint64_t>(*source.cycles_shader_index) + 1u);
      }
      auto result = validator.compile(source.shader);
      if (!result.ok()) {
        throw std::runtime_error("graph validation failed: " + source.name);
      }
      programs.emplace(id, std::move(result.program));
    }
    std::vector<svm::ShaderTableCompileUnit> units;
    for (auto id : materials) {
      const auto &source = scene.materials.at(id);
      const auto index = source.cycles_shader_index ?
          static_cast<std::uint64_t>(*source.cycles_shader_index) : next_authored++;
      if (index > UINT32_MAX) { throw std::runtime_error("shader index overflow"); }
      units.push_back({
          .shader_index = static_cast<std::uint32_t>(index),
          .shader = programs.at(id).get(),
          .context = {.background = scene.world_shader == id,
                      .displacement_method = source.displacement_method,
                      .color_space = scene.shader_color_space},
          .kernel = {.name = source.name,
                     .use_transparent_shadow = source.use_transparent_shadow,
                     .use_bump_map_correction = source.use_bump_map_correction,
                     .emission_sampling = source.emission_sampling,
                     .volume_sampling = source.volume_sampling,
                     .volume_interpolation = source.volume_interpolation,
                     .pass_id = source.cycles_pass_id}});
    }
    std::ranges::stable_sort(units, {}, &svm::ShaderTableCompileUnit::shader_index);
    const auto compiled = svm::compile_shader_table(units);
    if (!compiled.table.valid) { throw std::runtime_error(compiled.table.diagnostic); }
    if (argc == 3) { write_bindings(argv[2], compiled, scene); }
    std::vector<std::string_view> names(compiled.table.shader_count);
    for (const auto &unit : units) { names.at(unit.shader_index) = unit.kernel.name; }
    std::cout.write("PSYSVM52", 8);
    write_scalar(std::uint32_t{1});
    write_scalar(static_cast<std::uint64_t>(compiled.table.words.size()));
    write_scalar(compiled.table.shader_count);
    for (std::uint32_t i = 0; i < names.size(); ++i) {
      write_scalar(i);
      write_scalar(static_cast<std::uint32_t>(names[i].size()));
      std::cout.write(names[i].data(), static_cast<std::streamsize>(names[i].size()));
    }
    for (auto word : compiled.table.words) { write_scalar(word); }
    std::cerr << "Psycles compiled " << units.size() << " used units; "
              << compiled.table.shader_count << " dense shaders; "
              << compiled.table.words.size() << " words; stack "
              << compiled.table.peak_stack_usage << "; closures "
              << compiled.max_closures << '\n';
    return std::cout.good() ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

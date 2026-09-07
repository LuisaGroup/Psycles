#include "cycles_svm_map_range_fixture.h"

#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>

namespace {
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

struct Bundle {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("psycles-map-range-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  explicit Bundle(const char *scene) {
    require(std::filesystem::create_directory(path), "cannot create bundle");
    std::filesystem::copy_file(scene, path / "scene.json");
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
    require(geometry.good(), "cannot write empty geometry fixture");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

void run(const char *scene, const char *words, unsigned count) {
  Bundle bundle{scene};
  const auto imported =
      psycles::adapter::load_blender_scene_bundle(bundle.path);
  require(imported.ok(), "Map Range fixture did not import");
  const ShaderCompiler compiler{make_core_node_registry()};
  const auto oracles =
      psycles::test_support::read_map_range_images(words, count);
  for (auto index = 0u; index < oracles.size(); ++index) {
    const auto &oracle = oracles[index];
    // The probe shaders are 5..28; Monster's light is shader 5. Source
    // identity is authoritative, including when the adapter decorates names.
    const auto shader_index = 5u + index;
    const auto material = std::find_if(
        imported.scene->materials.begin(), imported.scene->materials.end(),
        [&](const auto &entry) {
          return entry.second.cycles_shader_index == shader_index;
        });
    require(material != imported.scene->materials.end(),
            "external material is missing");
    const auto shader = compiler.compile(material->second.shader);
    require(shader.ok(), "Map Range source graph is invalid");
    for (const auto &node : shader.program->graph().nodes()) {
      if (node.type == node_type::map_range &&
          std::get<std::string>(node.properties.at("DataType").value) ==
              "FLOAT" &&
          std::get<std::string>(node.properties.at("Interpolation").value) !=
              "STEPPED") {
        // Cycles' node prototype keeps Steps=4 when Blender marks that
        // socket unavailable, even if its hidden authored default is 3.
        const auto &steps = node.inputs.at("Steps").value;
        require(steps && std::get<float>(steps->value) == 4.0f,
                "unavailable socket replaced the Cycles prototype default");
      }
    }
    AttributeIDMap attributes;
    const auto image =
        compile_shader(*shader.program, attributes, ShaderCompileContext{});
    require(image.valid, image.diagnostic);
    if (image.words != oracle.words) {
      std::cerr << oracle.name << ": actual=" << image.words.size()
                << " Cycles=" << oracle.words.size() << " words\n";
      for (auto i = 0u; i < std::max(image.words.size(), oracle.words.size());
           ++i) {
        std::cerr << i << std::hex << ": "
                  << (i < image.words.size() ? image.words[i] : ~0u) << " / "
                  << (i < oracle.words.size() ? oracle.words[i] : ~0u)
                  << std::dec << '\n';
      }
      throw std::runtime_error{
          "Map Range differs from the complete Cycles image"};
    }
  }
}
} // namespace

int main() {
  try {
    run(PSYCLES_MAP_RANGE_SCENE, PSYCLES_MAP_RANGE_WORDS, 24u);
    run(PSYCLES_MONSTER_MAP_RANGE_SCENE, PSYCLES_MONSTER_MAP_RANGE_WORDS, 1u);
    std::cout << "24 Map Range probes and Monster light image match original Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

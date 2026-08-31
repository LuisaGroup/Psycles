#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include "cycles_svm_test_compile.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error{message};
  }
}

class TemporaryDirectory {
private:
  std::filesystem::path _path;

public:
  TemporaryDirectory() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
            ("psycles-background-import-" + std::to_string(nonce));
    std::filesystem::create_directories(_path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return _path;
  }
};

void write_scene(const std::filesystem::path &path) {
  {
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  std::ofstream scene{path / "scene.json"};
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],"materials":[],
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":{
    "name":"SVM Background Import",
    "color":[0.05,0.05,0.05],
    "cycles_sync":{"shader_index":3,"object_index":12},
    "node_tree":{
      "name":"SVM Background Import",
      "surface_root":{"node":"Background","socket":"Background"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Background","from_socket":"Background",
         "to_node":"Output","to_socket":"Surface"}
      ],
      "nodes":[
        {
          "name":"Background","type":"BACKGROUND","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.16,0.48,0.77,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":2.3}
          ],
          "outputs":[
            {"identifier":"Background","name":"Background",
             "type":"NodeSocketShader","linked":true}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Output","type":"OUTPUT_WORLD","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Surface","name":"Surface",
             "type":"NodeSocketShader","linked":true},
            {"identifier":"Volume","name":"Volume",
             "type":"NodeSocketShader","linked":false}
          ],
          "outputs":[],"properties":{"is_active_output":true},"special":{}
        }
      ]
    }
  },
  "world_environment":null,
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0}
})JSON";
}

void test_background_import_matches_cycles_5_2_1() {
  TemporaryDirectory temporary;
  write_scene(temporary.path());
  const auto imported =
      psycles::adapter::load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Background scene did not import");

  require(imported.scene->world_shader.has_value(),
          "imported Background world has no shader identity");
  const auto imported_world = imported.scene->materials.find(
      *imported.scene->world_shader);
  require(imported_world != imported.scene->materials.end(),
          "imported Background world is absent");
  const auto *world = &imported_world->second;
  bool has_background = false;
  for (const auto &node : world->shader.nodes()) {
    has_background |=
        node.type == psycles::compiler::node_type::background;
  }
  require(has_background,
          "Blender Background was not retained as a background closure");

  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(world->shader);
  require(shader.ok(), "imported Background graph did not validate");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program);
  require(image.valid, "imported Background graph did not compile to SVM");
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
      0x00000005u, 0x3ebc6a7eu, 0x3f8d4fdfu, 0x3fe2b020u,
      0x00000004u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  require(image.words ==
              std::vector<std::uint32_t>(expected.begin(), expected.end()),
          "imported Background stream differs from Cycles 5.2.1");
}

} // namespace

int main() {
  test_background_import_matches_cycles_5_2_1();
  return 0;
}

#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_test_compile.h"
#include <psycles/compiler/shader_program.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

void expect(bool condition, const char *message) {
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
            ("psycles-wireframe-import-" + std::to_string(nonce));
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

void write_scene(const std::filesystem::path &path, bool use_pixel_size) {
  {
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  std::ofstream scene{path / "scene.json"};
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],
  "materials":[{
    "name":"SVM Wireframe Import",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"SVM Wireframe Import",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Wireframe","from_socket":"Fac",
         "to_node":"Emission","to_socket":"Color"}
      ],
      "nodes":[
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1.0,1.0,1.0,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1.0}
          ],
          "outputs":[
            {"identifier":"Emission","name":"Emission",
             "type":"NodeSocketShader","linked":true}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Wireframe","type":"WIREFRAME","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Size","name":"Size",
             "type":"NodeSocketFloatDistance","linked":false,
             "default":)JSON"
        << (use_pixel_size ? "2.5" : "0.09") << R"JSON(}
          ],
          "outputs":[
            {"identifier":"Fac","name":"Fac",
             "type":"NodeSocketFloat","linked":true,"default":0.0}
          ],
          "properties":{"use_pixel_size":)JSON"
        << (use_pixel_size ? "true" : "false") << R"JSON(},
          "special":{}
        }
      ]
    }
  }],
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

void test_wireframe_import(bool use_pixel_size) {
  TemporaryDirectory temporary;
  write_scene(temporary.path(), use_pixel_size);
  const auto imported =
      psycles::adapter::load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "Wireframe scene did not import");

  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "SVM Wireframe Import") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "Wireframe imported material is absent");

  const psycles::contract::ShaderNode *wireframe = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.label == "Wireframe") {
      wireframe = &node;
      break;
    }
  }
  expect(wireframe != nullptr, "Blender Wireframe node is absent");
  expect(wireframe->type == psycles::compiler::node_type::wireframe,
         "Blender Wireframe node type differs from Cycles projection");
  const auto property = wireframe->properties.find("Use Pixel Size");
  expect(property != wireframe->properties.end() &&
             std::get<bool>(property->second.value) == use_pixel_size,
         "Blender Wireframe use_pixel_size property was not preserved");

  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  expect(shader.ok(), "imported Wireframe graph did not validate");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program);
  expect(image.valid, "imported Wireframe graph did not compile to SVM");

  static constexpr std::array world{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x0000005au,
      0x3db851ecu, 0x00000000u, 0x00000000u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  auto expected = world;
  if (use_pixel_size) {
    expected[5] = 0x40200000u;
    expected[7] = 0x00000001u;
  }
  expect(image.words ==
             std::vector<std::uint32_t>(expected.begin(), expected.end()),
         "imported Wireframe stream differs from Cycles 5.2.1");
  expect(image.peak_stack_usage == 4u,
         "imported Wireframe stack lifetime differs from Cycles");
}

} // namespace

int main() {
  test_wireframe_import(false);
  test_wireframe_import(true);
  return 0;
}

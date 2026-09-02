#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string &message) {
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
            ("psycles-light-path-import-" + std::to_string(nonce));
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

} // namespace

void test_blender_light_path_portal_depth_import() {
  TemporaryDirectory temporary;
  {
    std::ofstream geometry{temporary.path() / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],
  "node_groups":[],
  "materials":[{
    "name":"Portal Depth Material",
    "cycles_sync":{"shader_index":7},
    "node_tree":{
      "name":"Portal Depth Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,
      "displacement_root":null,
      "links":[
        {"from_node":"Light Path","from_socket":"Portal Depth",
         "to_node":"Emission","to_socket":"Strength"}
      ],
      "nodes":[
        {
          "name":"Light Path","type":"LIGHT_PATH","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Portal Depth","name":"Portal Depth",
            "type":"NodeSocketFloat","linked":true,"default":0.0}],
          "properties":{},"special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[1.0,1.0,1.0,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":true,"default":1.0}
          ],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}],
          "properties":{},"special":{}
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

  const auto imported =
      psycles::adapter::load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "Light Path Portal Depth scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Portal Depth Material") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "Light Path Portal Depth material is missing");

  psycles::compiler::ShaderCompiler compiler{
      psycles::compiler::make_core_node_registry()};
  const auto shader = compiler.compile(material->shader);
  expect(shader.ok(), "Light Path Portal Depth graph did not validate");
  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  expect(surface.ok(), "Light Path Portal Depth graph did not lower");
  auto portal_count = 0u;
  auto transmission_count = 0u;
  for (const auto &instruction : surface.program->value_instructions()) {
    portal_count += instruction.operation ==
                    psycles::compiler::ValueOperation::path_portal_depth;
    transmission_count +=
        instruction.operation ==
        psycles::compiler::ValueOperation::path_transmission_depth;
  }
  expect(portal_count == 1u && transmission_count == 0u,
         "Blender Portal Depth was aliased to Transmission Depth");
}

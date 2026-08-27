#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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
            ("psycles-hair-import-" + std::to_string(nonce));
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

void test_legacy_hair_closure_import() {
  TemporaryDirectory temporary;
  {
    std::ofstream geometry{temporary.path() / "geometry.bin",
                           std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],
  "node_groups":[],
  "materials":[{
    "name":"Legacy Hair Transmission",
    "cycles_sync":{"shader_index":12},
    "node_tree":{
      "name":"Legacy Hair Transmission",
      "surface_root":{"node":"Hair","socket":"BSDF"},
      "volume_root":null,
      "displacement_root":null,
      "links":[],
      "nodes":[{
        "name":"Hair","type":"BSDF_HAIR","mute":false,
        "internal_links":[],
        "inputs":[
          {"identifier":"Color","name":"Color",
           "type":"NodeSocketColor","linked":false,
           "default":[0.7,0.3,0.1,1.0]},
          {"identifier":"Offset","name":"Offset",
           "type":"NodeSocketFloatAngle","linked":false,
           "default":0.12},
          {"identifier":"RoughnessU","name":"RoughnessU",
           "type":"NodeSocketFloatFactor","linked":false,
           "default":0.21},
          {"identifier":"RoughnessV","name":"RoughnessV",
           "type":"NodeSocketFloatFactor","linked":false,
           "default":0.43},
          {"identifier":"Tangent","name":"Tangent",
           "type":"NodeSocketVector","linked":false,
           "default":[0.0,0.0,0.0]},
          {"identifier":"Weight","name":"Weight",
           "type":"NodeSocketFloat","linked":false,"default":0.0}
        ],
        "outputs":[{"identifier":"BSDF","name":"BSDF",
          "type":"NodeSocketShader","linked":true}],
        "properties":{"component":"Transmission"},
        "special":{}
      }]
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
  expect(imported.ok(), "legacy Hair scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Legacy Hair Transmission") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "legacy Hair material is missing");
  const psycles::contract::ShaderNode *hair = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.type == psycles::compiler::node_type::hair_bsdf) {
      hair = &node;
      break;
    }
  }
  expect(
      hair != nullptr && hair->properties.contains("Component") &&
          hair->properties.at("Component") ==
              psycles::contract::SocketValue::string("Transmission") &&
          hair->inputs.contains("Color") &&
          hair->inputs.contains("Offset") &&
          hair->inputs.contains("RoughnessU") &&
          hair->inputs.contains("RoughnessV") &&
          hair->inputs.contains("Tangent") &&
          !hair->inputs.contains("Weight") &&
          !hair->inputs.at("Tangent").source.has_value(),
      "raw Hair closure topology or static component was pre-baked");

  psycles::compiler::ShaderCompiler compiler{
      psycles::compiler::make_core_node_registry()};
  const auto shader = compiler.compile(material->shader);
  expect(shader.ok(), "legacy Hair graph did not validate");
  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  expect(surface.ok() &&
             surface.program->closure_instructions().size() == 1u,
         "legacy Hair graph did not lower");
  const auto &closure = surface.program->closure_instructions().front();
  expect(
      closure.operation ==
              psycles::compiler::ClosureOperation::hair_transmission &&
          closure.color.valid() && closure.hair_offset.valid() &&
          closure.roughness.valid() &&
          closure.diffuse_roughness.valid() && closure.tangent.valid() &&
          !closure.hair_tangent_linked,
      "legacy Hair closure was not preserved as raw typed inputs");
}

} // namespace

int main() {
  try {
    test_legacy_hair_closure_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

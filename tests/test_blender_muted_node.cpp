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
            ("psycles-muted-node-" + std::to_string(nonce));
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

void test_muted_node_uses_internal_bypass() {
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
    "name":"Muted Mix Material",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Muted Mix Material",
      "surface_root":{"node":"Diffuse","socket":"BSDF"},
      "volume_root":null,
      "displacement_root":null,
      "links":[
        {"from_node":"Live Source","from_socket":"Color",
         "to_node":"Muted Mix","to_socket":"A_Color"},
        {"from_node":"Dead Unsupported Source","from_socket":"Color",
         "to_node":"Muted Mix","to_socket":"B_Color"},
        {"from_node":"Muted Mix","from_socket":"Result_Color",
         "to_node":"Diffuse","to_socket":"Color"}
      ],
      "nodes":[
        {
          "name":"Live Source","type":"RGB","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Color","name":"Color",
            "type":"NodeSocketColor","linked":true,
            "default":[0.9,0.7,0.5,1.0]}],
          "properties":{},"special":{}
        },
        {
          "name":"Dead Unsupported Source","type":"UNSUPPORTED",
          "mute":false,"internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Color","name":"Color",
            "type":"NodeSocketColor","linked":true,
            "default":[0.1,0.2,0.3,1.0]}],
          "properties":{},"special":{}
        },
        {
          "name":"Muted Mix","type":"MIX","mute":true,
          "internal_links":[
            {"from_socket":"A_Color","to_socket":"Result_Color"}
          ],
          "inputs":[
            {"identifier":"Factor_Float","name":"Factor",
             "type":"NodeSocketFloat","linked":false,"default":0.5},
            {"identifier":"A_Color","name":"A",
             "type":"NodeSocketColor","linked":true,
             "default":[0.0,0.0,0.0,1.0]},
            {"identifier":"B_Color","name":"B",
             "type":"NodeSocketColor","linked":true,
             "default":[0.0,0.0,0.0,1.0]}
          ],
          "outputs":[
            {"identifier":"Result_Color","name":"Result",
             "type":"NodeSocketColor","linked":true,
             "default":[0.0,0.0,0.0,0.0]}
          ],
          "properties":{"data_type":"RGBA","blend_type":"MIX",
                        "clamp_factor":true,"clamp_result":false},
          "special":{}
        },
        {
          "name":"Diffuse","type":"BSDF_DIFFUSE","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[0.8,0.8,0.8,1.0]},
            {"identifier":"Roughness","name":"Roughness",
             "type":"NodeSocketFloat","linked":false,"default":0.0},
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
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
  expect(imported.ok(), "muted-node scene did not import");

  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Muted Mix Material") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "muted-node material is missing");

  const auto &graph = material->shader;
  const psycles::contract::ShaderNode *diffuse = nullptr;
  for (const auto &node : graph.nodes()) {
    expect(node.type != psycles::compiler::node_type::mix_color,
           "muted Mix was evaluated instead of bypassed");
    expect(node.label != "Dead Unsupported Source",
           "dead muted-node input was lowered");
    if (node.type == psycles::compiler::node_type::diffuse_bsdf) {
      diffuse = &node;
    }
  }
  expect(diffuse != nullptr, "diffuse closure is missing");
  const auto color = diffuse->inputs.find("Color");
  expect(color != diffuse->inputs.end() && color->second.source.has_value(),
         "muted bypass did not connect the live color source");
  const auto *source = graph.find(color->second.source->node);
  expect(source != nullptr &&
             source->type == psycles::compiler::node_type::constant_color &&
             source->label == "Live Source",
         "muted bypass selected the wrong input");

  psycles::compiler::ShaderCompiler compiler{
      psycles::compiler::make_core_node_registry()};
  const auto shader = compiler.compile(graph);
  expect(shader.ok(), "muted bypass graph did not validate");
  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  expect(surface.ok(), "muted bypass graph did not lower");
  for (const auto &instruction : surface.program->value_instructions()) {
    expect(instruction.operation !=
               psycles::compiler::ValueOperation::mix,
           "muted Mix emitted a value instruction");
  }
  for (const auto &diagnostic : imported.diagnostics) {
    expect(diagnostic.message.find("UNSUPPORTED") == std::string::npos,
           "dead muted-node branch emitted a diagnostic");
  }
}

} // namespace

int main() {
  try {
    test_muted_node_uses_internal_bypass();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
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
            ("psycles-attribute-import-" + std::to_string(nonce));
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
  "images":[],"node_groups":[],
  "materials":[
    {
      "name":"SVM Vertex Color Import",
      "cycles_sync":{"shader_index":5},
      "node_tree":{
        "name":"SVM Vertex Color Import",
        "surface_root":{"node":"Emission","socket":"Emission"},
        "volume_root":null,"displacement_root":null,
        "links":[
          {"from_node":"Vertex Color","from_socket":"Color",
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
            "name":"Vertex Color","type":"VERTEX_COLOR","mute":false,
            "internal_links":[],"inputs":[],
            "outputs":[
              {"identifier":"Color","name":"Color",
               "type":"NodeSocketColor","linked":true},
              {"identifier":"Alpha","name":"Alpha",
               "type":"NodeSocketFloat","linked":false}
            ],
            "properties":{"layer_name":""},"special":{}
          }
        ]
      }
    },
    {
      "name":"SVM Empty Attribute Import",
      "cycles_sync":{"shader_index":6},
      "node_tree":{
        "name":"SVM Empty Attribute Import",
        "surface_root":{"node":"Emission","socket":"Emission"},
        "volume_root":null,"displacement_root":null,
        "links":[
          {"from_node":"Attribute","from_socket":"Color",
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
            "name":"Attribute","type":"ATTRIBUTE","mute":false,
            "internal_links":[],"inputs":[],
            "outputs":[
              {"identifier":"Color","name":"Color",
               "type":"NodeSocketColor","linked":true},
              {"identifier":"Vector","name":"Vector",
               "type":"NodeSocketVector","linked":false},
              {"identifier":"Fac","name":"Fac",
               "type":"NodeSocketFloat","linked":false},
              {"identifier":"Alpha","name":"Alpha",
               "type":"NodeSocketFloat","linked":false}
            ],
            "properties":{"attribute_name":"","attribute_type":"GEOMETRY"},
            "special":{}
          }
        ]
      }
    }
  ],
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

[[nodiscard]] const psycles::contract::MaterialDesc &
find_material(const psycles::contract::SceneSnapshot &scene,
              const std::string &name) {
  for (const auto &[id, material] : scene.materials) {
    static_cast<void>(id);
    if (material.name == name) {
      return material;
    }
  }
  throw std::runtime_error{"imported material is absent: " + name};
}

[[nodiscard]] const psycles::contract::ShaderNode &
find_node(const psycles::contract::MaterialDesc &material,
          const std::string &label) {
  for (const auto &node : material.shader.nodes()) {
    if (node.label == label) {
      return node;
    }
  }
  throw std::runtime_error{"imported node is absent: " + label};
}

[[nodiscard]] psycles::compiler::cycles_svm::ShaderImage
compile_material(
    const psycles::contract::MaterialDesc &material,
    psycles::compiler::cycles_svm::AttributeIDMap &attribute_ids) {
  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(material.shader);
  expect(shader.ok(), "imported attribute graph did not validate");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program,
                                                    attribute_ids);
  expect(image.valid, image.diagnostic.c_str());
  return image;
}

void test_attribute_import() {
  TemporaryDirectory temporary;
  write_scene(temporary.path());
  const auto imported =
      psycles::adapter::load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "attribute scene did not import");

  const auto &vertex_material =
      find_material(*imported.scene, "SVM Vertex Color Import");
  const auto &vertex = find_node(vertex_material, "Vertex Color");
  expect(vertex.type == psycles::compiler::node_type::vertex_color,
         "Blender Vertex Color node type differs from Cycles projection");
  const auto layer = vertex.properties.find("Layer Name");
  expect(layer != vertex.properties.end() &&
             std::get<std::string>(layer->second.value).empty(),
         "Blender default Vertex Color layer was not preserved");

  const auto &attribute_material =
      find_material(*imported.scene, "SVM Empty Attribute Import");
  const auto &attribute = find_node(attribute_material, "Attribute");
  expect(attribute.type == psycles::compiler::node_type::attribute,
         "Blender Attribute was projected as another node family");
  const auto name = attribute.properties.find("Attribute");
  expect(name != attribute.properties.end() &&
             std::get<std::string>(name->second.value).empty(),
         "Blender empty Attribute name was discarded");

  static constexpr std::array<std::uint32_t, 17u> vertex_oracle{
      0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u, 0x00000017u,
      0x00ff000au, 0x00000000u, 0x00000007u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  static constexpr std::array<std::uint32_t, 18u> attribute_oracle{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u, 0x00000015u,
      0x00000023u, 0x00000000u, 0x00000000u, 0x00000007u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  psycles::compiler::cycles_svm::AttributeIDMap attribute_ids;
  const auto vertex_image = compile_material(vertex_material, attribute_ids);
  const auto attribute_image =
      compile_material(attribute_material, attribute_ids);
  expect(vertex_image.words ==
             std::vector<std::uint32_t>(vertex_oracle.begin(),
                                        vertex_oracle.end()),
         "imported default Vertex Color stream differs from Cycles 5.2.1");
  expect(attribute_image.words ==
             std::vector<std::uint32_t>(attribute_oracle.begin(),
                                        attribute_oracle.end()),
         "imported empty Attribute stream differs from Cycles 5.2.1");
}

} // namespace

int main() {
  test_attribute_import();
  return 0;
}

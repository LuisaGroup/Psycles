#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

class TemporaryDirectory {
private:
  std::filesystem::path _path;

public:
  TemporaryDirectory() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
            ("psycles-normal-map-tangent-import-" +
             std::to_string(nonce));
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
      "name":"Normal Map Material","cycles_sync":{"shader_index":3},
      "node_tree":{
        "name":"Normal Map Material",
        "surface_root":{"node":"Emission","socket":"Emission"},
        "volume_root":null,"displacement_root":null,
        "links":[
          {"from_node":"Normal Map","from_socket":"Normal",
           "to_node":"Emission","to_socket":"Color"}
        ],
        "nodes":[
          {
            "name":"Normal Map","type":"NORMAL_MAP","mute":false,
            "internal_links":[],
            "inputs":[
              {"identifier":"Strength","name":"Strength",
               "type":"NodeSocketFloat","linked":false,"default":0.7},
              {"identifier":"Color","name":"Color",
               "type":"NodeSocketColor","linked":false,
               "default":[0.65,0.35,0.95,1.0]}
            ],
            "outputs":[
              {"identifier":"Normal","name":"Normal",
               "type":"NodeSocketVector","linked":true,
               "default":[0.0,0.0,0.0]}
            ],
            "properties":{"space":"TANGENT","uv_map":"MappedUV",
              "convention":"DIRECTX","base":"ORIGINAL"},
            "special":{}
          },
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
          }
        ]
      }
    },
    {
      "name":"Tangent Material","cycles_sync":{"shader_index":4},
      "node_tree":{
        "name":"Tangent Material",
        "surface_root":{"node":"Emission","socket":"Emission"},
        "volume_root":null,"displacement_root":null,
        "links":[
          {"from_node":"Tangent","from_socket":"Tangent",
           "to_node":"Emission","to_socket":"Color"}
        ],
        "nodes":[
          {
            "name":"Tangent","type":"TANGENT","mute":false,
            "internal_links":[],"inputs":[],
            "outputs":[
              {"identifier":"Tangent","name":"Tangent",
               "type":"NodeSocketVector","linked":true,
               "default":[0.0,0.0,0.0]}
            ],
            "properties":{"direction_type":"UV_MAP","axis":"Y",
              "uv_map":"MappedUV"},
            "special":{}
          },
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
          }
        ]
      }
    }
  ],
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null,
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0}
})JSON";
}

[[nodiscard]] const psycles::contract::MaterialDesc &find_material(
    const psycles::contract::SceneSnapshot &scene, std::string_view name) {
  for (const auto &[id, material] : scene.materials) {
    static_cast<void>(id);
    if (material.name == name) { return material; }
  }
  throw std::runtime_error{"imported normal-family material is missing"};
}

[[nodiscard]] const psycles::contract::ShaderNode &find_node(
    const psycles::contract::MaterialDesc &material, std::string_view type) {
  for (const auto &node : material.shader.nodes()) {
    if (node.type == type) { return node; }
  }
  throw std::runtime_error{"imported normal-family node is missing"};
}

[[nodiscard]] std::string property_string(
    const psycles::contract::ShaderNode &node, std::string_view name) {
  const auto found = node.properties.find(std::string{name});
  require(found != node.properties.end(), "imported node property is missing");
  return std::get<std::string>(found->second.value);
}

template<std::size_t N>
[[nodiscard]] std::array<std::uint32_t, N> node_record(
    const ShaderImage &image, ShaderNodeType opcode) {
  for (auto index = std::size_t{}; index + N <= image.words.size(); ++index) {
    if (image.words[index] == static_cast<std::uint32_t>(opcode)) {
      std::array<std::uint32_t, N> result{};
      std::copy_n(image.words.begin() + index, N, result.begin());
      return result;
    }
  }
  throw std::runtime_error{"compiled import has no expected SVM record"};
}

[[nodiscard]] ShaderImage compile_material(
    const psycles::contract::MaterialDesc &material) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material.shader);
  require(shader.ok(), "imported normal-family graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  auto image = compile_shader(
      *shader.program, attributes, images,
      ShaderCompileContext{.background = false});
  require(image.valid, "imported normal-family graph did not compile to SVM");
  return image;
}

void test_blender_normal_map_tangent_import() {
  TemporaryDirectory temporary;
  write_scene(temporary.path());
  const auto imported =
      psycles::adapter::load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Normal Map/Tangent scene did not import");

  const auto &normal_material =
      find_material(*imported.scene, "Normal Map Material");
  const auto &normal_map = find_node(normal_material, node_type::normal_map);
  require(property_string(normal_map, "Space") == "TANGENT" &&
              property_string(normal_map, "Convention") == "DIRECTX" &&
              property_string(normal_map, "Base") == "ORIGINAL" &&
              property_string(normal_map, "Attribute") == "MappedUV",
          "Blender Normal Map lost exact Cycles properties");
  const auto normal_image = compile_material(normal_material);
  const auto normal_record = node_record<11u>(normal_image, NODE_NORMAL_MAP);
  require(normal_record[1u] == NODE_NORMAL_MAP_TANGENT &&
              normal_record[2u] == 1u && normal_record[3u] == 1u &&
              normal_record[4u] == ATTR_STD_NUM &&
              normal_record[5u] == ATTR_STD_NUM + 1u,
          "imported Normal Map SVM attribute payload differs from Cycles");

  const auto &tangent_material =
      find_material(*imported.scene, "Tangent Material");
  const auto &tangent = find_node(tangent_material, node_type::tangent);
  require(property_string(tangent, "Direction Type") == "UV_MAP" &&
              property_string(tangent, "Axis") == "Y" &&
              property_string(tangent, "Attribute") == "MappedUV",
          "Blender Tangent lost exact Cycles properties");
  const auto tangent_image = compile_material(tangent_material);
  const auto tangent_record = node_record<5u>(tangent_image, NODE_TANGENT);
  require(tangent_record[1u] == NODE_TANGENT_UVMAP &&
              tangent_record[2u] == NODE_TANGENT_AXIS_Y &&
              tangent_record[3u] == ATTR_STD_NUM,
          "imported Tangent SVM attribute payload differs from Cycles");
}

} // namespace

int main() {
  test_blender_normal_map_tangent_import();
  return 0;
}

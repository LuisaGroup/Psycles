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
            ("psycles-vector-import-" + std::to_string(nonce));
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

void test_blender_separate_combine_xyz_import() {
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
  "images":[],"node_groups":[],
  "materials":[{
    "name":"SVM Separate Combine Vector",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"SVM Separate Combine Vector",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Geometry","from_socket":"Normal",
         "to_node":"Separate XYZ","to_socket":"Vector"},
        {"from_node":"Separate XYZ","from_socket":"Z",
         "to_node":"Combine XYZ","to_socket":"X"},
        {"from_node":"Separate XYZ","from_socket":"X",
         "to_node":"Combine XYZ","to_socket":"Y"},
        {"from_node":"Separate XYZ","from_socket":"Y",
         "to_node":"Combine XYZ","to_socket":"Z"},
        {"from_node":"Combine XYZ","from_socket":"Vector",
         "to_node":"Emission","to_socket":"Color"}
      ],
      "nodes":[
        {
          "name":"Combine XYZ","type":"COMBXYZ","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":true,"default":0.0}
          ],
          "outputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{},"special":{}
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
        },
        {
          "name":"Geometry","type":"NEW_GEOMETRY","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Separate XYZ","type":"SEPXYZ","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":true,"default":0.0}
          ],
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
  expect(imported.ok(), "Separate/Combine XYZ scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "SVM Separate Combine Vector") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr,
         "Separate/Combine XYZ imported material is absent");
  const psycles::contract::ShaderNode *separate = nullptr;
  const psycles::contract::ShaderNode *combine = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.label == "Separate XYZ") {
      separate = &node;
    } else if (node.label == "Combine XYZ") {
      combine = &node;
    }
  }
  if (separate == nullptr) {
    auto available = std::string{"Blender Separate XYZ node is absent; nodes:"};
    for (const auto &node : material->shader.nodes()) {
      available += " [" + node.label + ":" + node.type + "]";
    }
    throw std::runtime_error{available};
  }
  if (separate->type != psycles::compiler::node_type::separate_xyz) {
    throw std::runtime_error{"Blender Separate XYZ projected as " +
                             separate->type};
  }
  expect(combine != nullptr, "Blender Combine XYZ node is absent");
  if (combine->type != psycles::compiler::node_type::combine_xyz) {
    throw std::runtime_error{"Blender Combine XYZ projected as " +
                             combine->type};
  }
  expect(separate->inputs.at("Vector").source.has_value() &&
             combine->inputs.at("X").source.has_value() &&
             combine->inputs.at("Y").source.has_value() &&
             combine->inputs.at("Z").source.has_value(),
         "Blender vector split/pack links were not preserved");

  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  expect(shader.ok(), "imported Separate/Combine XYZ graph did not validate");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program);
  expect(image.valid,
         "imported Separate/Combine XYZ graph did not compile to SVM");
  static constexpr std::array cycles_5_2_1_oracle{
      0x00000001u, 0x00000004u, 0x00000027u, 0x00000028u,
      0x0000000bu, 0x00000001u, 0x00000000u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000300u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000401u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000502u,
      0x00000056u, 0x7fc00005u, 0x00000000u,
      0x00000056u, 0x7fc00003u, 0x00000001u,
      0x00000056u, 0x7fc00004u, 0x00000002u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u,
  };
  expect(image.words.size() == cycles_5_2_1_oracle.size(),
         "imported vector split/pack word count differs from Cycles");
  for (auto index = std::size_t{}; index < image.words.size(); ++index) {
    expect(image.words[index] == cycles_5_2_1_oracle[index],
           "imported vector split/pack stream differs from Cycles 5.2.1");
  }
  expect(image.peak_stack_usage == 6u,
         "imported vector split/pack stack lifetime differs from Cycles");
}

void test_blender_vector_rotate_import() {
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
  "images":[],"node_groups":[],
  "materials":[{
    "name":"SVM Vector Rotate Import",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"SVM Vector Rotate Import",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Geometry","from_socket":"Normal",
         "to_node":"Vector Rotate","to_socket":"Vector"},
        {"from_node":"Vector Rotate","from_socket":"Vector",
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
          "name":"Geometry","type":"NEW_GEOMETRY","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Vector Rotate","type":"VECTOR_ROTATE","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]},
            {"identifier":"Center","name":"Center",
             "type":"NodeSocketVector","linked":false,
             "default":[0.17,-0.23,0.31]},
            {"identifier":"Axis","name":"Axis",
             "type":"NodeSocketVector","linked":false,
             "default":[0.29,0.73,-0.41]},
            {"identifier":"Angle","name":"Angle",
             "type":"NodeSocketFloatAngle","linked":false,
             "default":0.71},
            {"identifier":"Rotation","name":"Rotation",
             "type":"NodeSocketVectorEuler","linked":false,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{"rotation_type":"AXIS_ANGLE","invert":false},
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

  const auto imported =
      psycles::adapter::load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "Vector Rotate scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "SVM Vector Rotate Import") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "Vector Rotate imported material is absent");
  const psycles::contract::ShaderNode *rotate = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.label == "Vector Rotate") {
      rotate = &node;
      break;
    }
  }
  expect(rotate != nullptr, "Blender Vector Rotate node is absent");
  expect(rotate->type == psycles::compiler::node_type::vector_rotate,
         "Blender Vector Rotate node type differs from Cycles projection");
  expect(rotate->inputs.at("Vector").source.has_value(),
         "Blender Vector Rotate input link was not preserved");

  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  expect(shader.ok(), "imported Vector Rotate graph did not validate");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program);
  expect(image.valid, "imported Vector Rotate graph did not compile to SVM");
  static constexpr std::array cycles_5_2_1_oracle{
      0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
      0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
      0x00000000u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x3e947ae1u,
      0x3f3ae148u, 0xbed1eb85u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x3f35c28fu, 0x00000300u, 0x00000007u,
      0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  expect(image.words.size() == cycles_5_2_1_oracle.size(),
         "imported Vector Rotate word count differs from Cycles");
  for (auto index = std::size_t{}; index < image.words.size(); ++index) {
    expect(image.words[index] == cycles_5_2_1_oracle[index],
           "imported Vector Rotate stream differs from Cycles 5.2.1");
  }
  expect(image.peak_stack_usage == 6u,
         "imported Vector Rotate stack lifetime differs from Cycles");
}

void test_blender_vector_transform_import() {
  TemporaryDirectory temporary;
  {
    std::ofstream geometry{temporary.path() / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  {
    std::ofstream scene{temporary.path() / "scene.json"};
    scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],
  "materials":[{
    "name":"SVM Vector Transform Import",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"SVM Vector Transform Import",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Vector Transform","from_socket":"Vector",
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
          "name":"Vector Transform","type":"VECTOR_TRANSFORM",
          "mute":false,"internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":false,
             "default":[0.37,-0.21,0.63]}
          ],
          "outputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{"vector_type":"NORMAL",
                        "convert_from":"CAMERA",
                        "convert_to":"OBJECT"},
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

  const auto imported =
      psycles::adapter::load_blender_scene_bundle(temporary.path());
  expect(imported.ok(), "Vector Transform scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "SVM Vector Transform Import") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "Vector Transform imported material is absent");
  const psycles::contract::ShaderNode *transform = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.label == "Vector Transform") {
      transform = &node;
      break;
    }
  }
  expect(transform != nullptr, "Blender Vector Transform node is absent");
  expect(transform->type == psycles::compiler::node_type::vector_transform,
         "Blender Vector Transform type differs from Cycles projection");
  expect(transform->properties.at("Type").value ==
                 psycles::contract::SocketValue::string("NORMAL").value &&
             transform->properties.at("Convert From").value ==
                 psycles::contract::SocketValue::string("CAMERA").value &&
             transform->properties.at("Convert To").value ==
                 psycles::contract::SocketValue::string("OBJECT").value,
         "Blender Vector Transform properties were not preserved");

  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  expect(shader.ok(), "imported Vector Transform graph did not validate");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program);
  expect(image.valid, "imported Vector Transform graph did not compile to SVM");
  static constexpr std::array cycles_5_2_1_oracle{
      0x00000001u, 0x00000004u, 0x00000014u, 0x00000015u, 0x00000059u,
      0x00000002u, 0x00000002u, 0x00000001u, 0x3ebd70a4u, 0xbe570a3du,
      0x3f2147aeu, 0x00000000u, 0x00000007u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };
  expect(image.words == std::vector<std::uint32_t>(cycles_5_2_1_oracle.begin(),
                                                   cycles_5_2_1_oracle.end()),
         "imported Vector Transform stream differs from Cycles 5.2.1");
  expect(image.peak_stack_usage == 3u,
         "imported Vector Transform stack lifetime differs from Cycles");
}

void test_blender_geometry_tangent_parametric_import() {
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
  "images":[],"node_groups":[],
  "materials":[{
    "name":"SVM Geometry Tangent Parametric Import",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"SVM Geometry Tangent Parametric Import",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Geometry","from_socket":"Tangent",
         "to_node":"Separate Tangent","to_socket":"Vector"},
        {"from_node":"Geometry","from_socket":"Parametric",
         "to_node":"Separate Parametric","to_socket":"Vector"},
        {"from_node":"Separate Tangent","from_socket":"X",
         "to_node":"Combine XYZ","to_socket":"X"},
        {"from_node":"Separate Parametric","from_socket":"Y",
         "to_node":"Combine XYZ","to_socket":"Y"},
        {"from_node":"Combine XYZ","from_socket":"Vector",
         "to_node":"Emission","to_socket":"Color"}
      ],
      "nodes":[
        {
          "name":"Geometry","type":"NEW_GEOMETRY","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[
            {"identifier":"Tangent","name":"Tangent",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]},
            {"identifier":"Parametric","name":"Parametric",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Separate Tangent","type":"SEPXYZ","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":false,"default":0.0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":0.0}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Separate Parametric","type":"SEPXYZ","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":false,"default":0.0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":0.0}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Combine XYZ","type":"COMBXYZ","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":true,"default":0.0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":0.0}
          ],
          "outputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{},"special":{}
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
  expect(imported.ok(), "Geometry Tangent/Parametric scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "SVM Geometry Tangent Parametric Import") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr,
         "Geometry Tangent/Parametric imported material is absent");
  const psycles::contract::ShaderNode *separate_tangent = nullptr;
  const psycles::contract::ShaderNode *separate_parametric = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.label == "Separate Tangent") {
      separate_tangent = &node;
    } else if (node.label == "Separate Parametric") {
      separate_parametric = &node;
    }
  }
  expect(separate_tangent != nullptr && separate_parametric != nullptr,
         "Geometry Tangent/Parametric consumers are absent");

  const auto tangent_source =
      separate_tangent->inputs.at("Vector").source.value();
  const auto *tangent_to_color = material->shader.find(tangent_source.node);
  expect(tangent_to_color != nullptr &&
             tangent_to_color->type ==
                 psycles::compiler::node_type::vector_to_color,
         "Geometry.Tangent lost Cycles' Separate XYZ color conversion");
  const auto tangent_geometry_source =
      tangent_to_color->inputs.at("Vector").source.value();
  const auto *tangent_geometry =
      material->shader.find(tangent_geometry_source.node);
  expect(tangent_geometry != nullptr &&
             tangent_geometry->type == psycles::compiler::node_type::geometry &&
             tangent_geometry_source.socket == "Tangent",
         "Blender Geometry.Tangent was not preserved as the Cycles output");

  const auto parametric_vector_source =
      separate_parametric->inputs.at("Vector").source.value();
  const auto *parametric_to_color =
      material->shader.find(parametric_vector_source.node);
  expect(parametric_to_color != nullptr &&
             parametric_to_color->type ==
                 psycles::compiler::node_type::vector_to_color,
         "Geometry.Parametric lost Cycles' Separate XYZ color conversion");
  const auto point_to_vector_source =
      parametric_to_color->inputs.at("Vector").source.value();
  const auto *point_to_vector =
      material->shader.find(point_to_vector_source.node);
  expect(point_to_vector != nullptr &&
             point_to_vector->type ==
                 psycles::compiler::node_type::point_to_vector,
         "Blender Geometry.Parametric lost its point-to-vector conversion");
  const auto parametric_source =
      point_to_vector->inputs.at("Point").source.value();
  const auto *parametric_geometry =
      material->shader.find(parametric_source.node);
  expect(parametric_geometry != nullptr &&
             parametric_geometry->type ==
                 psycles::compiler::node_type::geometry &&
             parametric_source.socket == "Parametric",
         "Blender Geometry.Parametric was not preserved as the Cycles output");

  const psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  expect(shader.ok(), "imported Geometry Tangent/Parametric graph is invalid");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program);
  expect(image.valid && image.node_types_used[
                            psycles::compiler::cycles_svm::NODE_GEOMETRY] &&
             !image.node_types_used[
                 psycles::compiler::cycles_svm::NODE_GEOMETRY_DERIVATIVE],
         "plain Geometry Tangent/Parametric did not select Cycles' opcode");
}

} // namespace

void test_blender_vector_import() {
  test_blender_separate_combine_xyz_import();
  test_blender_vector_rotate_import();
  test_blender_vector_transform_import();
  test_blender_geometry_tangent_parametric_import();
}

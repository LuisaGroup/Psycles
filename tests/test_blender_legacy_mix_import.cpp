#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <bit>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <variant>

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
            ("psycles-legacy-mix-import-" + std::to_string(nonce));
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

void test_blender_legacy_mix_import() {
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
    "name":"Legacy Mix Import","cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Legacy Mix Import",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Legacy Mix","from_socket":"Color",
         "to_node":"Emission","to_socket":"Color"}
      ],
      "nodes":[
        {
          "name":"Legacy Mix","type":"MIX_RGB","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Fac","name":"Fac","type":"NodeSocketFloat",
             "linked":false,"default":0.37},
            {"identifier":"Color1","name":"Color1",
             "type":"NodeSocketColor","linked":false,
             "default":[0.17,0.63,0.89,0.2]},
            {"identifier":"Color2","name":"Color2",
             "type":"NodeSocketColor","linked":false,
             "default":[0.82,0.24,0.51,0.9]}
          ],
          "outputs":[{"identifier":"Color","name":"Color",
            "type":"NodeSocketColor","linked":true}],
          "properties":{"blend_type":"OVERLAY","use_clamp":true,
                        "use_alpha":true},"special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[0.0,0.0,0.0,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1.0}
          ],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}],
          "properties":{},"special":{}
        }
      ]
    }
  },{
    "name":"Modern Mix Import","cycles_sync":{"shader_index":6},
    "node_tree":{
      "name":"Modern Mix Import",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Geometry","from_socket":"Backfacing",
         "to_node":"Mix Float","to_socket":"Factor_Float"},
        {"from_node":"Mix Float","from_socket":"Result_Float",
         "to_node":"Mix Vector Uniform","to_socket":"Factor_Float"},
        {"from_node":"Geometry","from_socket":"Normal",
         "to_node":"Mix Vector Non Uniform","to_socket":"Factor_Vector"},
        {"from_node":"Geometry","from_socket":"Backfacing",
         "to_node":"Mix Color","to_socket":"Factor_Float"},
        {"from_node":"Mix Vector Uniform","from_socket":"Result_Vector",
         "to_node":"Mix Color","to_socket":"A_Color"},
        {"from_node":"Mix Vector Non Uniform","from_socket":"Result_Vector",
         "to_node":"Mix Color","to_socket":"B_Color"},
        {"from_node":"Mix Color","from_socket":"Result_Color",
         "to_node":"Emission","to_socket":"Color"}
      ],
      "nodes":[
        {
          "name":"Geometry","type":"NEW_GEOMETRY","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]},
            {"identifier":"Backfacing","name":"Backfacing",
             "type":"NodeSocketFloat","linked":true,"default":0.0}
          ],
          "properties":{},"special":{}
        },
        {
          "name":"Mix Float","type":"MIX","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Factor_Float","name":"Factor",
             "type":"NodeSocketFloatFactor","linked":true,"default":1.0},
            {"identifier":"A_Float","name":"A",
             "type":"NodeSocketFloat","linked":false,"default":0.2},
            {"identifier":"B_Float","name":"B",
             "type":"NodeSocketFloat","linked":false,"default":0.8}
          ],
          "outputs":[
            {"identifier":"Result_Float","name":"Result",
             "type":"NodeSocketFloat","linked":true,"default":0.0}
          ],
          "properties":{"blend_type":"MIX","clamp_factor":false,
                        "clamp_result":false,"data_type":"FLOAT",
                        "factor_mode":"UNIFORM"},
          "special":{}
        },
        {
          "name":"Mix Vector Uniform","type":"MIX","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Factor_Float","name":"Factor",
             "type":"NodeSocketFloatFactor","linked":true,"default":1.0},
            {"identifier":"A_Vector","name":"A",
             "type":"NodeSocketVector","linked":false,
             "default":[0.1,0.7,-0.2]},
            {"identifier":"B_Vector","name":"B",
             "type":"NodeSocketVector","linked":false,
             "default":[0.9,-0.1,0.6]}
          ],
          "outputs":[
            {"identifier":"Result_Vector","name":"Result",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{"blend_type":"MIX","clamp_factor":true,
                        "clamp_result":false,"data_type":"VECTOR",
                        "factor_mode":"UNIFORM"},
          "special":{}
        },
        {
          "name":"Mix Vector Non Uniform","type":"MIX","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Factor_Vector","name":"Factor",
             "type":"NodeSocketVectorFactor","linked":true,
             "default":[0.5,0.5,0.5]},
            {"identifier":"A_Vector","name":"A",
             "type":"NodeSocketVector","linked":false,
             "default":[0.3,-0.4,0.8]},
            {"identifier":"B_Vector","name":"B",
             "type":"NodeSocketVector","linked":false,
             "default":[-0.2,0.6,0.1]}
          ],
          "outputs":[
            {"identifier":"Result_Vector","name":"Result",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "properties":{"blend_type":"MIX","clamp_factor":false,
                        "clamp_result":false,"data_type":"VECTOR",
                        "factor_mode":"NON_UNIFORM"},
          "special":{}
        },
        {
          "name":"Mix Color","type":"MIX","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Factor_Float","name":"Factor",
             "type":"NodeSocketFloatFactor","linked":true,"default":1.0},
            {"identifier":"A_Color","name":"A",
             "type":"NodeSocketColor","linked":true,
             "default":[0.5,0.5,0.5,1.0]},
            {"identifier":"B_Color","name":"B",
             "type":"NodeSocketColor","linked":true,
             "default":[0.5,0.5,0.5,1.0]}
          ],
          "outputs":[
            {"identifier":"Result_Color","name":"Result",
             "type":"NodeSocketColor","linked":true,
             "default":[0.8,0.8,0.8,1.0]}
          ],
          "properties":{"blend_type":"OVERLAY","clamp_factor":false,
                        "clamp_result":true,"data_type":"RGBA",
                        "factor_mode":"UNIFORM"},
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
  expect(imported.ok(), "legacy MixRGB scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Legacy Mix Import") {
      material = &candidate;
      break;
    }
  }
  expect(material != nullptr, "legacy MixRGB material is missing");

  const psycles::contract::ShaderNode *legacy_mix = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.label == "Legacy Mix") {
      legacy_mix = &node;
      break;
    }
  }
  expect(legacy_mix != nullptr, "legacy MixRGB graph node is missing");
  expect(legacy_mix->type == psycles::compiler::node_type::legacy_mix_color,
         "legacy MixRGB was conflated with modern Mix Color");
  const auto blend = legacy_mix->properties.find("BlendMode");
  const auto clamp = legacy_mix->properties.find("ClampResult");
  expect(blend != legacy_mix->properties.end() &&
             std::get<std::string>(blend->second.value) == "OVERLAY",
         "legacy MixRGB blend type was not preserved");
  expect(clamp != legacy_mix->properties.end() &&
             std::get<bool>(clamp->second.value),
         "legacy MixRGB result clamp was not preserved");
  expect(legacy_mix->properties.find("ClampFactor") ==
             legacy_mix->properties.end(),
         "legacy MixRGB invented a ClampFactor property absent from Cycles");

  psycles::compiler::ShaderCompiler frontend{
      psycles::compiler::make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  expect(shader.ok(), "imported legacy MixRGB graph did not validate");
  const auto image =
      psycles::compiler::cycles_svm::compile_shader(*shader.program);
  expect(image.valid, "imported legacy MixRGB graph did not compile to SVM");
  expect(image.words.size() == 13u,
         "folded legacy MixRGB SVM word count differs from Cycles");
  expect(image.words[5] == 0x3e574d59u &&
             image.words[6] == 0x3f0f0e4eu &&
             image.words[7] == 0x3f640c63u,
         "imported legacy MixRGB fold differs from Cycles 5.2.1");

  const psycles::contract::MaterialDesc *modern_material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Modern Mix Import") {
      modern_material = &candidate;
      break;
    }
  }
  expect(modern_material != nullptr, "modern Mix material is missing");

  const auto find_node = [&](std::string_view label) {
    for (const auto &node : modern_material->shader.nodes()) {
      if (node.label == label) {
        return &node;
      }
    }
    return static_cast<const psycles::contract::ShaderNode *>(nullptr);
  };
  const auto *mix_float = find_node("Mix Float");
  const auto *mix_uniform = find_node("Mix Vector Uniform");
  const auto *mix_nonuniform = find_node("Mix Vector Non Uniform");
  const auto *mix_color = find_node("Mix Color");
  expect(mix_float != nullptr &&
             mix_float->type == psycles::compiler::node_type::mix_float,
         "Blender Mix FLOAT did not map to Cycles MixFloatNode");
  expect(mix_uniform != nullptr &&
             mix_uniform->type == psycles::compiler::node_type::mix_vector,
         "Blender Mix VECTOR UNIFORM did not map to Cycles MixVectorNode");
  expect(mix_nonuniform != nullptr &&
             mix_nonuniform->type ==
                 psycles::compiler::node_type::mix_vector_nonuniform,
         "Blender Mix VECTOR NON_UNIFORM did not map to Cycles node");
  expect(mix_color != nullptr &&
             mix_color->type == psycles::compiler::node_type::mix_color,
         "Blender Mix RGBA did not map to Cycles MixColorNode");

  const auto bool_property = [](const auto *node, std::string_view name) {
    const auto iter = node->properties.find(name);
    return iter != node->properties.end() &&
           std::get<bool>(iter->second.value);
  };
  expect(!bool_property(mix_float, "ClampFactor") &&
             bool_property(mix_uniform, "ClampFactor") &&
             !bool_property(mix_nonuniform, "ClampFactor"),
         "Blender Mix factor clamp flags were not preserved by data type");
  expect(!bool_property(mix_color, "ClampFactor") &&
             bool_property(mix_color, "ClampResult"),
         "Blender MixColor clamp flags were not preserved");
  const auto modern_blend = mix_color->properties.find("BlendMode");
  expect(modern_blend != mix_color->properties.end() &&
             std::get<std::string>(modern_blend->second.value) == "OVERLAY",
         "Blender MixColor blend mode was not preserved");
  expect(mix_float->inputs.at("Factor").source.has_value() &&
             mix_uniform->inputs.at("Factor").source.has_value() &&
             mix_nonuniform->inputs.at("Factor").source.has_value() &&
             mix_color->inputs.at("Factor").source.has_value() &&
             mix_color->inputs.at("A").source.has_value() &&
             mix_color->inputs.at("B").source.has_value(),
         "Blender modern Mix socket identifiers were not linked exactly");
  const auto blender_linked_color_default =
      psycles::contract::SocketValue::color({0.5f, 0.5f, 0.5f});
  expect(mix_color->inputs.at("A").value == blender_linked_color_default &&
             mix_color->inputs.at("B").value ==
                 blender_linked_color_default,
         "Blender linked MixColor socket defaults were not retained");

  const auto modern_shader = frontend.compile(modern_material->shader);
  expect(modern_shader.ok(), "imported modern Mix graph did not validate");
  const auto modern_image =
      psycles::compiler::cycles_svm::compile_shader(*modern_shader.program);
  expect(modern_image.valid,
         "imported modern Mix graph did not compile to Cycles SVM");
  static constexpr std::array cycles_5_2_1_oracle{
      0x00000001u, 0x00000004u, 0x00000035u, 0x00000036u,
      0x0000000bu, 0x00000001u, 0x00000000u,
      0x00000032u, 0x00000008u, 0x00000003u,
      0x0000006au, 0x3e99999au, 0xbecccccdu, 0x3f4ccccdu,
      0xbe4ccccdu, 0x3f19999au, 0x3dcccccdu,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x00000400u,
      0x00000068u, 0x7fc00003u, 0x3e4ccccdu, 0x3f4ccccdu,
      0x00000000u,
      0x00000069u, 0x3dcccccdu, 0x3f333333u, 0xbe4ccccdu,
      0x3f666666u, 0xbdcccccdu, 0x3f19999au,
      0x7fc00000u, 0x00000701u,
      0x00000067u, 0x00000009u,
      0x7fc00007u, 0x00000000u, 0x00000000u,
      0x7fc00004u, 0x00000000u, 0x00000000u,
      0x7fc00003u, 0x00000100u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u,
  };
  expect(modern_image.words.size() == cycles_5_2_1_oracle.size(),
         "imported modern Mix SVM word count differs from Cycles 5.2.1");
  for (auto index = std::size_t{};
       index < cycles_5_2_1_oracle.size(); ++index) {
    expect(modern_image.words[index] == cycles_5_2_1_oracle[index],
           "imported modern Mix SVM stream differs from Cycles 5.2.1");
  }
  expect(modern_image.peak_stack_usage == 10u,
         "imported modern Mix stack lifetime differs from Cycles 5.2.1");
}

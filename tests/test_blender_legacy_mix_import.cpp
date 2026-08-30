#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <bit>
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
}

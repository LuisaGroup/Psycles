#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

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
            ("psycles-rgb-curve-import-" + std::to_string(nonce));
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

void write_scene_bundle(const std::filesystem::path &directory) {
  {
    std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  std::ofstream scene{directory / "scene.json"};
  scene << std::setprecision(9);
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],
  "materials":[{
    "name":"RGB Curves Material",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"RGB Curves Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Coordinates","from_socket":"Generated",
         "to_node":"Curves","to_socket":"Color"},
        {"from_node":"Curve Factor","from_socket":"Value",
         "to_node":"Curves","to_socket":"Fac"},
        {"from_node":"Curves","from_socket":"Color",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Coordinates","type":"TEX_COORD","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Generated","name":"Generated",
            "type":"NodeSocketVector","linked":true,
            "default":[0,0,0]}],
          "properties":{"from_instancer":false},"special":{}
        },
        {
          "name":"Curve Factor","type":"VALUE","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Value","name":"Value",
            "type":"NodeSocketFloat","linked":true,
            "default":0.37}],
          "properties":{},"special":{}
        },
        {
          "name":"Curves","type":"CURVE_RGB","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Fac","name":"Factor",
             "type":"NodeSocketFloatFactor","linked":true,
             "default":0.37},
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1,1,1,1]}],
          "outputs":[{"identifier":"Color","name":"Color",
            "type":"NodeSocketColor","linked":true,
            "default":[1,1,1,1]}],
          "properties":{},
          "special":{"curve_mapping":{
            "min_x":-0.25,"max_x":1.3,"extrapolate":false,
            "samples":[
)JSON";
  for (auto index = std::uint32_t{}; index <= 256u; ++index) {
    const auto t = static_cast<float>(index) / 256.0f;
    if (index != 0u) {
      scene << ",\n";
    }
    scene << '[' << t << ',' << (0.25f + 0.5f * t) << ',' << (1.0f - t)
          << ']';
  }
  scene << R"JSON(
            ],"curves":[]}}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1}],
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
    "clip_start":0.01,"clip_end":100},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

void test_sampled_table_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "RGB Curves scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "RGB Curves Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "RGB Curves material is missing");

  const ShaderNode *curve = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.type == node_type::rgb_curve) {
      require(curve == nullptr,
              "RGB Curves import duplicated the semantic node");
      curve = &node;
    }
  }
  require(curve != nullptr, "RGB Curves import lost the semantic node");
  const auto sampled = curve->properties.find("Sampled");
  const auto min_x = curve->properties.find("MinX");
  const auto max_x = curve->properties.find("MaxX");
  const auto extrapolate = curve->properties.find("Extrapolate");
  const auto table = curve->properties.find("Table");
  require(sampled != curve->properties.end() &&
              std::get<bool>(sampled->second.value) &&
              min_x != curve->properties.end() &&
              std::get<float>(min_x->second.value) == -0.25f &&
              max_x != curve->properties.end() &&
              std::get<float>(max_x->second.value) == 1.3f &&
              extrapolate != curve->properties.end() &&
              !std::get<bool>(extrapolate->second.value),
          "RGB Curves import lost sampled/domain/extend metadata");
  require(table != curve->properties.end() &&
              std::count(std::get<std::string>(table->second.value).begin(),
                         std::get<std::string>(table->second.value).end(),
                         ';') == 256,
          "RGB Curves import did not preserve all 257 Cycles samples");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "RGB Curves graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());
  const auto opcode = std::find(image.words.begin(), image.words.end(),
                                static_cast<std::uint32_t>(NODE_CURVES));
  require(opcode != image.words.end(), "imported RGB Curves opcode is missing");
  const auto begin = static_cast<std::size_t>(opcode - image.words.begin());
  require(begin + 9u + 257u * 4u <= image.words.size(),
          "imported RGB Curves table is truncated");
  require((image.words[begin + 1u] >> 8u) ==
                  (SVM_INPUT_STACK_OFFSET_MASK >> 8u) &&
              image.words[begin + 4u] ==
                  std::bit_cast<std::uint32_t>(0.37f) &&
              image.words[begin + 5u] ==
                  std::bit_cast<std::uint32_t>(-0.25f) &&
              image.words[begin + 6u] ==
                  std::bit_cast<std::uint32_t>(1.3f) &&
              image.words[begin + 7u] == 257u,
          "imported RGB Curves header differs from Cycles");
  const auto packed = image.words[begin + 8u];
  require((packed & 0xffu) == 0u &&
              ((packed >> 8u) & 0xffu) != SVM_STACK_INVALID &&
              (packed >> 16u) == 0u,
          "imported RGB Curves extend/output bytes differ from Cycles");

  const auto sample_word = [&](std::uint32_t sample,
                               std::uint32_t component) {
    return image.words[begin + 9u + sample * 4u + component];
  };
  require(sample_word(0u, 0u) == std::bit_cast<std::uint32_t>(0.0f) &&
              sample_word(128u, 0u) ==
                  std::bit_cast<std::uint32_t>(0.5f) &&
              sample_word(128u, 1u) ==
                  std::bit_cast<std::uint32_t>(0.5f) &&
              sample_word(256u, 2u) ==
                  std::bit_cast<std::uint32_t>(0.0f) &&
              sample_word(0u, 3u) ==
                  std::bit_cast<std::uint32_t>(1.0f) &&
              sample_word(256u, 3u) ==
                  std::bit_cast<std::uint32_t>(1.0f),
          "imported RGB Curves payload changed samples or float4 padding");
}

} // namespace

int main() {
  try {
    test_sampled_table_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

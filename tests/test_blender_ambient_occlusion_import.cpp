#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/cycles_svm_node_types.h>
#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/compiler/shader_program.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using psycles::contract::feature_bit;
using psycles::contract::ShaderFeature;

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
            ("psycles-ao-import-" + std::to_string(nonce));
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
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],
  "node_groups":[],
  "materials":[{
    "name":"AO Material",
    "cycles_sync":{"shader_index":3},
    "node_tree":{
      "name":"AO Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,
      "displacement_root":null,
      "links":[
        {"from_node":"Geometry","from_socket":"Normal",
         "to_node":"AO","to_socket":"Normal"},
        {"from_node":"AO","from_socket":"Color",
         "to_node":"Emission","to_socket":"Color"},
        {"from_node":"AO","from_socket":"AO",
         "to_node":"Emission","to_socket":"Strength"}
      ],
      "nodes":[
        {
          "name":"Geometry","type":"NEW_GEOMETRY","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Normal","name":"Normal",
            "type":"NodeSocketVector","linked":true,
            "default":[0.0,0.0,0.0]}],
          "properties":{},"special":{}
        },
        {
          "name":"AO","type":"AMBIENT_OCCLUSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.25,0.5,0.75,1.0]},
            {"identifier":"Distance","name":"Distance",
             "type":"NodeSocketFloat","linked":false,"default":0.0},
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":true,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[0.0,0.0,0.0,1.0]},
            {"identifier":"AO","name":"AO",
             "type":"NodeSocketFloat","linked":true,"default":0.0}
          ],
          "properties":{"samples":273,"inside":true,"only_local":true},
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

void test_ambient_occlusion_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());

  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Ambient Occlusion scene did not import");
  require(imported.scene->ambient_occlusion_distance == 10.0f,
          "old scene bundles did not retain Cycles' AO-distance default");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "AO Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Ambient Occlusion material is missing");

  const psycles::contract::ShaderNode *ao_node = nullptr;
  auto ao_node_count = 0u;
  auto multiply_node_count = 0u;
  for (const auto &node : material->shader.nodes()) {
    if (node.type == psycles::compiler::node_type::ambient_occlusion) {
      ++ao_node_count;
      ao_node = &node;
    }
    multiply_node_count +=
        node.type == psycles::compiler::node_type::multiply_color;
  }
  require(ao_node_count == 1u && ao_node != nullptr,
          "AO Color and AO outputs did not share one semantic node");
  require(multiply_node_count == 1u,
          "AO Color output did not retain its raw color multiplication");

  const auto property_u64 = [&](const char *name) {
    return std::get<std::uint64_t>(ao_node->properties.at(name).value);
  };
  const auto property_bool = [&](const char *name) {
    return std::get<bool>(ao_node->properties.at(name).value);
  };
  require(property_u64("Samples") == 17u,
          "AO Samples did not preserve Cycles' uint8 SVM narrowing");
  require(property_bool("NormalLinked") && property_bool("Inside") &&
              property_bool("OnlyLocal") && property_bool("GlobalRadius"),
          "AO immutable configuration was not preserved");

  psycles::compiler::ShaderCompiler compiler{
      psycles::compiler::make_core_node_registry()};
  const auto shader = compiler.compile(material->shader);
  require(shader.ok(), "Ambient Occlusion graph did not validate");
  require((shader.program->analysis().required_features &
           feature_bit(ShaderFeature::ambient_occlusion)) != 0u,
          "Ambient Occlusion graph lost its traversal capability");

  namespace abi = psycles::compiler::cycles_svm;
  const std::array units{
      abi::ShaderTableCompileUnit{.shader_index = 0u,
                                  .shader = shader.program.get()}};
  const auto table = abi::compile_shader_table(units);
  require(table.table.valid, "Ambient Occlusion SVM shader table failed to compile");
  require(table.table.node_types_used[abi::NODE_AMBIENT_OCCLUSION],
          "NODE_AMBIENT_OCCLUSION was not emitted into SVM bytecode");
  require((table.kernel_features & abi::kernel_feature_node_raytrace) != 0u,
          "NODE_AMBIENT_OCCLUSION did not request kernel_feature_node_raytrace");
  const auto &routine =
      table.table.ir.shaders.at(0u).routine_for(abi::SHADER_TYPE_SURFACE);
  bool found_ao = false;
  for (const auto &op : routine) {
    if (!op.is_conditional &&
        op.instruction.opcode == abi::NODE_AMBIENT_OCCLUSION) {
      found_ao = true;
      require(op.instruction.payload_word_count == 6u,
              "SVMNodeAmbientOcclusion payload size mismatch");
      const auto word4 = table.table.words.at(op.instruction.payload_offset + 4u);
      const auto flags = word4 & 0xffu;
      const auto samples = (word4 >> 8u) & 0xffu;
      require(flags == (abi::NODE_AO_ONLY_LOCAL | abi::NODE_AO_INSIDE |
                        abi::NODE_AO_GLOBAL_RADIUS),
              "NODE_AMBIENT_OCCLUSION flags mismatch");
      require(samples == 17u, "NODE_AMBIENT_OCCLUSION samples mismatch");
    }
  }
  require(found_ao, "NODE_AMBIENT_OCCLUSION op missing from surface IR");
}

void write_closure_zone_bundle(const std::filesystem::path &directory) {
  {
    std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  std::ofstream scene{directory / "scene.json"};
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],
  "node_groups":[{
    "name":"EvalClosureGroup",
    "nodes":[
      {
        "name":"Group Input","type":"GROUP_INPUT","mute":false,"internal_links":[],
        "inputs":[],
        "outputs":[{"identifier":"Socket_0","name":"Fn","type":"NodeSocketClosure","linked":true}],
        "properties":{},"special":{}
      },
      {
        "name":"Eval A","type":"NodeEvaluateClosure","mute":false,"internal_links":[],
        "inputs":[
          {"identifier":"Closure","name":"Closure","type":"NodeSocketClosure","linked":true},
          {"identifier":"Item_0","name":"Vector","type":"NodeSocketVector","linked":false,"default":[1.0,0.0,0.0]}
        ],
        "outputs":[
          {"identifier":"Item_0","name":"Color","type":"NodeSocketColor","linked":true,"default":[0.0,0.0,0.0,1.0]}
        ],
        "properties":{},"special":{}
      },
      {
        "name":"Eval B","type":"NodeEvaluateClosure","mute":false,"internal_links":[],
        "inputs":[
          {"identifier":"Closure","name":"Closure","type":"NodeSocketClosure","linked":true},
          {"identifier":"Item_0","name":"Vector","type":"NodeSocketVector","linked":false,"default":[0.0,1.0,0.0]}
        ],
        "outputs":[
          {"identifier":"Item_0","name":"Color","type":"NodeSocketColor","linked":true,"default":[0.0,0.0,0.0,1.0]}
        ],
        "properties":{},"special":{}
      },
      {
        "name":"Sum","type":"VECT_MATH","mute":false,"internal_links":[],
        "inputs":[
          {"identifier":"Vector","name":"Vector","type":"NodeSocketVector","linked":true,"default":[0.0,0.0,0.0]},
          {"identifier":"Vector_001","name":"Vector","type":"NodeSocketVector","linked":true,"default":[0.0,0.0,0.0]}
        ],
        "outputs":[{"identifier":"Vector","name":"Vector","type":"NodeSocketVector","linked":true,"default":[0.0,0.0,0.0]}],
        "properties":{"operation":"ADD"},"special":{}
      },
      {
        "name":"Group Output","type":"GROUP_OUTPUT","mute":false,"internal_links":[],
        "inputs":[{"identifier":"Socket_1","name":"Out","type":"NodeSocketColor","linked":true,"default":[0.0,0.0,0.0,1.0]}],
        "outputs":[],
        "properties":{"is_active_output":true},"special":{}
      }
    ],
    "links":[
      {"from_node":"Group Input","from_socket":"Socket_0","to_node":"Eval A","to_socket":"Closure"},
      {"from_node":"Group Input","from_socket":"Socket_0","to_node":"Eval B","to_socket":"Closure"},
      {"from_node":"Eval A","from_socket":"Item_0","to_node":"Sum","to_socket":"Vector"},
      {"from_node":"Eval B","from_socket":"Item_0","to_node":"Sum","to_socket":"Vector_001"},
      {"from_node":"Sum","from_socket":"Vector","to_node":"Group Output","to_socket":"Socket_1"}
    ]
  }],
  "materials":[{
    "name":"Closure Zone Material",
    "cycles_sync":{"shader_index":4},
    "node_tree":{
      "name":"Closure Zone Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,
      "displacement_root":null,
      "links":[
        {"from_node":"Closure In","from_socket":"Item_0","to_node":"Offset","to_socket":"Vector"},
        {"from_node":"Offset","from_socket":"Vector","to_node":"Closure Out","to_socket":"Item_0"},
        {"from_node":"Closure Out","from_socket":"Closure","to_node":"EvalGroup","to_socket":"Socket_0"},
        {"from_node":"EvalGroup","from_socket":"Socket_1","to_node":"Emission","to_socket":"Color"}
      ],
      "nodes":[
        {
          "name":"Closure In","type":"NodeClosureInput","mute":false,"internal_links":[],
          "inputs":[],
          "outputs":[{"identifier":"Item_0","name":"Vector","type":"NodeSocketVector","linked":true,"default":[0.0,0.0,0.0]}],
          "properties":{},"special":{}
        },
        {
          "name":"Offset","type":"VECT_MATH","mute":false,"internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector","type":"NodeSocketVector","linked":true,"default":[0.0,0.0,0.0]},
            {"identifier":"Vector_001","name":"Vector","type":"NodeSocketVector","linked":false,"default":[0.1,0.2,0.3]}
          ],
          "outputs":[{"identifier":"Vector","name":"Vector","type":"NodeSocketVector","linked":true,"default":[0.0,0.0,0.0]}],
          "properties":{"operation":"ADD"},"special":{}
        },
        {
          "name":"Closure Out","type":"NodeClosureOutput","mute":false,"internal_links":[],
          "inputs":[{"identifier":"Item_0","name":"Color","type":"NodeSocketColor","linked":true,"default":[0.0,0.0,0.0,1.0]}],
          "outputs":[{"identifier":"Closure","name":"Closure","type":"NodeSocketClosure","linked":true}],
          "properties":{},"special":{}
        },
        {
          "name":"EvalGroup","type":"GROUP","node_tree":"EvalClosureGroup","mute":false,"internal_links":[],
          "inputs":[{"identifier":"Socket_0","name":"Fn","type":"NodeSocketClosure","linked":true}],
          "outputs":[{"identifier":"Socket_1","name":"Out","type":"NodeSocketColor","linked":true,"default":[0.0,0.0,0.0,1.0]}],
          "properties":{},"special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,"internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color","type":"NodeSocketColor","linked":true,"default":[1.0,1.0,1.0,1.0]},
            {"identifier":"Strength","name":"Strength","type":"NodeSocketFloat","linked":false,"default":1.0}
          ],
          "outputs":[{"identifier":"Emission","name":"Emission","type":"NodeSocketShader","linked":true}],
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

void test_closure_zone_import() {
  TemporaryDirectory temporary;
  write_closure_zone_bundle(temporary.path());

  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Closure Zone scene did not import");
  require(imported.diagnostics.empty(),
          "Closure Zone import emitted unexpected diagnostics");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Closure Zone Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Closure Zone material is missing");

  std::size_t vec_math_count = 0u;
  for (const auto &node : material->shader.nodes()) {
    if (node.type == psycles::compiler::node_type::vector_math) {
      ++vec_math_count;
    }
  }
  require(vec_math_count == 3u,
          "Expected two lowered closure Offset nodes plus outer Sum node");

  psycles::compiler::ShaderCompiler compiler{
      psycles::compiler::make_core_node_registry()};
  const auto shader = compiler.compile(material->shader);
  require(shader.ok(), "Closure Zone graph did not validate");

  namespace abi = psycles::compiler::cycles_svm;
  const std::array units{
      abi::ShaderTableCompileUnit{.shader_index = 0u,
                                  .shader = shader.program.get()}};
  const auto table = abi::compile_shader_table(units);
  require(table.table.valid, "Closure Zone SVM shader table failed to compile");
}

} // namespace

int main() {
  try {
    test_ambient_occlusion_import();
    test_closure_zone_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

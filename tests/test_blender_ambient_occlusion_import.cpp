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

using psycles::adapter::load_blender_scene_bundle;
using psycles::compiler::ValueOperation;
using psycles::contract::feature_bit;
using psycles::contract::ShaderFeature;
using psycles::contract::SocketType;

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

  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  require(surface.ok(), "Ambient Occlusion graph did not lower");
  const psycles::compiler::ValueInstruction *ao_instruction = nullptr;
  auto ao_instruction_count = 0u;
  auto multiply_instruction_count = 0u;
  for (const auto &instruction : surface.program->value_instructions()) {
    if (instruction.operation == ValueOperation::ambient_occlusion) {
      ++ao_instruction_count;
      ao_instruction = &instruction;
    }
    multiply_instruction_count +=
        instruction.operation == ValueOperation::multiply_color;
  }
  require(ao_instruction_count == 1u && ao_instruction != nullptr,
          "AO outputs duplicated scene traversal in the value IR");
  require(multiply_instruction_count == 1u,
          "AO Color multiplication was not lowered exactly once");
  require(ao_instruction->result_type == SocketType::floating &&
              ao_instruction->operands.size() ==
                  psycles::compiler::value_operand::ambient_occlusion::count,
          "AO value instruction lost its typed operand contract");
  require(ao_instruction->static_u0 ==
              (psycles::compiler::ambient_occlusion_only_local |
               psycles::compiler::ambient_occlusion_inside |
               psycles::compiler::ambient_occlusion_global_radius |
               psycles::compiler::ambient_occlusion_normal_linked),
          "AO value instruction lost immutable Cycles flags");

  const auto samples = ao_instruction->operand(
      psycles::compiler::value_operand::ambient_occlusion::samples);
  require(samples.valid() &&
              samples.value < surface.program->value_instructions().size(),
          "AO Samples operand is invalid");
  const auto &samples_value =
      surface.program->value_instructions()[samples.value];
  require(samples_value.operation == ValueOperation::parameter &&
              samples_value.parameter.valid(),
          "AO Samples was embedded in shader shape instead of parameter data");
  const auto &samples_parameter =
      surface.program->parameters()[samples_value.parameter.value];
  require(samples_parameter.source ==
                  psycles::compiler::ParameterSource::property &&
              samples_parameter.socket == "Samples" &&
              samples_parameter.type == SocketType::unsigned_integer &&
              std::get<std::uint64_t>(samples_parameter.default_value.value) ==
                  17u,
          "AO Samples parameter contract is not exact");
}

} // namespace

int main() {
  try {
    test_ambient_occlusion_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

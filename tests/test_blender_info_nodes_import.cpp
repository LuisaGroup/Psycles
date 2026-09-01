#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

struct RawSocket {
  std::string_view name;
  std::string_view type;
};

struct Family {
  std::string_view name;
  std::string_view raw_type;
  std::string_view semantic_type;
  ShaderNodeType direct_opcode;
  std::uint32_t maximum_info_type;
  std::span<const RawSocket> outputs;
  std::span<const std::uint32_t> direct_values;
  std::span<const std::uint32_t> attribute_values;
};

constexpr auto object_sockets = std::array{
    RawSocket{"Location", "NodeSocketVector"},
    RawSocket{"Color", "NodeSocketColor"},
    RawSocket{"Alpha", "NodeSocketFloat"},
    RawSocket{"Object Index", "NodeSocketFloat"},
    RawSocket{"Material Index", "NodeSocketFloat"},
    RawSocket{"Random", "NodeSocketFloat"},
};
constexpr auto object_direct = std::array{
    std::uint32_t{NODE_INFO_OB_LOCATION}, std::uint32_t{NODE_INFO_OB_COLOR},
    std::uint32_t{NODE_INFO_OB_ALPHA}, std::uint32_t{NODE_INFO_OB_INDEX},
    std::uint32_t{NODE_INFO_MAT_INDEX}, std::uint32_t{NODE_INFO_OB_RANDOM}};

constexpr auto particle_sockets = std::array{
    RawSocket{"Index", "NodeSocketFloat"},
    RawSocket{"Random", "NodeSocketFloat"},
    RawSocket{"Age", "NodeSocketFloat"},
    RawSocket{"Lifetime", "NodeSocketFloat"},
    RawSocket{"Location", "NodeSocketVector"},
    RawSocket{"Size", "NodeSocketFloat"},
    RawSocket{"Velocity", "NodeSocketVector"},
    RawSocket{"Angular Velocity", "NodeSocketVector"},
};
constexpr auto particle_direct = std::array{
    std::uint32_t{NODE_INFO_PAR_INDEX},
    std::uint32_t{NODE_INFO_PAR_RANDOM},
    std::uint32_t{NODE_INFO_PAR_AGE},
    std::uint32_t{NODE_INFO_PAR_LIFETIME},
    std::uint32_t{NODE_INFO_PAR_LOCATION},
    std::uint32_t{NODE_INFO_PAR_SIZE},
    std::uint32_t{NODE_INFO_PAR_VELOCITY},
    std::uint32_t{NODE_INFO_PAR_ANGULAR_VELOCITY},
};

constexpr auto hair_sockets = std::array{
    RawSocket{"Is Strand", "NodeSocketFloat"},
    RawSocket{"Intercept", "NodeSocketFloat"},
    RawSocket{"Length", "NodeSocketFloat"},
    RawSocket{"Thickness", "NodeSocketFloat"},
    RawSocket{"Tangent Normal", "NodeSocketVector"},
    RawSocket{"Random", "NodeSocketFloat"},
};
constexpr auto hair_direct = std::array{
    std::uint32_t{NODE_INFO_CURVE_IS_STRAND},
    std::uint32_t{NODE_INFO_CURVE_THICKNESS},
    std::uint32_t{NODE_INFO_CURVE_TANGENT_NORMAL},
};
constexpr auto hair_attributes = std::array{
    std::uint32_t{ATTR_STD_CURVE_INTERCEPT},
    std::uint32_t{ATTR_STD_CURVE_LENGTH},
    std::uint32_t{ATTR_STD_CURVE_RANDOM},
};

constexpr auto point_sockets = std::array{
    RawSocket{"Position", "NodeSocketVector"},
    RawSocket{"Radius", "NodeSocketFloat"},
    RawSocket{"Random", "NodeSocketFloat"},
};
constexpr auto point_direct = std::array{
    std::uint32_t{NODE_INFO_POINT_POSITION},
    std::uint32_t{NODE_INFO_POINT_RADIUS},
};
constexpr auto point_attributes =
    std::array{std::uint32_t{ATTR_STD_POINT_RANDOM}};

constexpr std::array<std::uint32_t, 0u> no_attributes{};

constexpr auto families = std::array{
    Family{"Object Info", "OBJECT_INFO", node_type::object_info,
           NODE_OBJECT_INFO, NODE_INFO_OB_RANDOM, object_sockets,
           object_direct, no_attributes},
    Family{"Particle Info", "PARTICLE_INFO", node_type::particle_info,
           NODE_PARTICLE_INFO, NODE_INFO_PAR_ANGULAR_VELOCITY,
           particle_sockets, particle_direct, no_attributes},
    Family{"Hair Info", "HAIR_INFO", node_type::hair_info, NODE_HAIR_INFO,
           NODE_INFO_CURVE_RANDOM, hair_sockets, hair_direct,
           hair_attributes},
    Family{"Point Info", "POINT_INFO", node_type::point_info,
           NODE_POINT_INFO, NODE_INFO_POINT_RANDOM, point_sockets,
           point_direct, point_attributes},
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
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
            ("psycles-info-import-" + std::to_string(nonce));
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

void write_info_node(std::ostream &scene, const Family &family) {
  scene << "{\"name\":\"Info\",\"type\":\"" << family.raw_type
        << "\",\"mute\":false,\"internal_links\":[],"
           "\"properties\":{},\"special\":{},\"inputs\":[],\"outputs\":[";
  for (auto index = std::size_t{}; index < family.outputs.size(); ++index) {
    const auto &socket = family.outputs[index];
    if (index != 0u) {
      scene << ',';
    }
    scene << "{\"identifier\":\"" << socket.name << "\",\"name\":\""
          << socket.name << "\",\"type\":\"" << socket.type
          << "\",\"linked\":true,\"default\":"
          << (socket.type == "NodeSocketFloat" ? "0" : "[0,0,0]") << '}';
  }
  scene << "]}";
}

void write_emission_node(std::ostream &scene, std::size_t index,
                         bool scalar) {
  scene << "{\"name\":\"Emission " << index
        << "\",\"type\":\"EMISSION\",\"mute\":false,"
           "\"internal_links\":[],\"properties\":{},\"special\":{},"
           "\"inputs\":["
        << "{\"identifier\":\"Color\",\"name\":\"Color\","
           "\"type\":\"NodeSocketColor\",\"linked\":"
        << (scalar ? "false" : "true") << ",\"default\":[1,1,1,1]},"
        << "{\"identifier\":\"Strength\",\"name\":\"Strength\","
           "\"type\":\"NodeSocketFloat\",\"linked\":"
        << (scalar ? "true" : "false") << ",\"default\":1}],"
        << "\"outputs\":[{\"identifier\":\"Emission\","
           "\"name\":\"Emission\",\"type\":\"NodeSocketShader\","
           "\"linked\":true}]}";
}

void write_add_node(std::ostream &scene, std::size_t index) {
  scene << "{\"name\":\"Add " << index
        << "\",\"type\":\"ADD_SHADER\",\"mute\":false,"
           "\"internal_links\":[],\"properties\":{},\"special\":{},"
           "\"inputs\":["
           "{\"identifier\":\"Shader\",\"name\":\"Shader\","
           "\"type\":\"NodeSocketShader\",\"linked\":true},"
           "{\"identifier\":\"Shader_001\",\"name\":\"Shader\","
           "\"type\":\"NodeSocketShader\",\"linked\":true}],"
           "\"outputs\":[{\"identifier\":\"Shader\","
           "\"name\":\"Shader\",\"type\":\"NodeSocketShader\","
           "\"linked\":true}]}";
}

void write_material(std::ostream &scene, const Family &family,
                    bool first_material) {
  if (!first_material) {
    scene << ',';
  }
  const auto root_name = family.outputs.size() == 1u
                             ? std::string{"Emission 0"}
                             : "Add " +
                                   std::to_string(family.outputs.size() - 2u);
  const auto root_socket =
      family.outputs.size() == 1u ? "Emission" : "Shader";
  scene << "{\"name\":\"" << family.name
        << " Material\",\"cycles_sync\":{\"shader_index\":7},"
           "\"node_tree\":{\"name\":\""
        << family.name << " Material\",\"surface_root\":{\"node\":\""
        << root_name << "\",\"socket\":\"" << root_socket
        << "\"},\"volume_root\":null,\"displacement_root\":null,"
           "\"links\":[";

  auto first_link = true;
  auto link = [&](std::string_view from_node,
                  std::string_view from_socket,
                  std::string_view to_node,
                  std::string_view to_socket) {
    if (!first_link) {
      scene << ',';
    }
    first_link = false;
    scene << "{\"from_node\":\"" << from_node
          << "\",\"from_socket\":\"" << from_socket
          << "\",\"to_node\":\"" << to_node
          << "\",\"to_socket\":\"" << to_socket << "\"}";
  };
  for (auto index = std::size_t{}; index < family.outputs.size(); ++index) {
    const auto emission = "Emission " + std::to_string(index);
    link("Info", family.outputs[index].name, emission,
         family.outputs[index].type == "NodeSocketFloat" ? "Strength"
                                                           : "Color");
  }
  if (family.outputs.size() > 1u) {
    link("Emission 0", "Emission", "Add 0", "Shader");
    link("Emission 1", "Emission", "Add 0", "Shader_001");
    for (auto index = std::size_t{2u}; index < family.outputs.size(); ++index) {
      const auto prior = "Add " + std::to_string(index - 2u);
      const auto current = "Add " + std::to_string(index - 1u);
      const auto emission = "Emission " + std::to_string(index);
      link(prior, "Shader", current, "Shader");
      link(emission, "Emission", current, "Shader_001");
    }
  }
  scene << "],\"nodes\":[";
  write_info_node(scene, family);
  for (auto index = std::size_t{}; index < family.outputs.size(); ++index) {
    scene << ',';
    write_emission_node(scene, index,
                        family.outputs[index].type == "NodeSocketFloat");
  }
  for (auto index = std::size_t{}; index + 1u < family.outputs.size();
       ++index) {
    scene << ',';
    write_add_node(scene, index);
  }
  scene << "]}}";
}

void write_scene_bundle(const std::filesystem::path &directory) {
  {
    std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  std::ofstream scene{directory / "scene.json"};
  scene << "{\"schema\":\"psycles.blender-scene.v1\","
           "\"images\":[],\"node_groups\":[],\"materials\":[";
  for (auto index = std::size_t{}; index < families.size(); ++index) {
    write_material(scene, families[index], index == 0u);
  }
  scene << "],\"render\":{\"width\":16,\"height\":16,"
           "\"percentage\":100,\"cycles\":{}},"
           "\"camera\":{\"name\":\"Camera\",\"type\":\"PERSP\","
           "\"transform\":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"
           "\"clip_start\":0.01,\"clip_end\":100},"
           "\"geometries\":[],\"curve_geometries\":[],"
           "\"instances\":[],\"lights\":[],\"world\":null,"
           "\"world_environment\":null}";
}

[[nodiscard]] std::vector<std::uint32_t> direct_records(
    const ShaderImage &image, ShaderNodeType opcode,
    std::uint32_t maximum_info_type) {
  std::vector<std::uint32_t> result;
  for (auto index = std::size_t{}; index + 2u < image.words.size(); ++index) {
    if (image.words[index] == static_cast<std::uint32_t>(opcode) &&
        image.words[index + 1u] <= maximum_info_type &&
        (image.words[index + 2u] & 0xffffff00u) == 0u) {
      result.emplace_back(image.words[index + 1u]);
      index += 2u;
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::uint32_t>
attribute_records(const ShaderImage &image) {
  std::vector<std::uint32_t> result;
  for (auto index = std::size_t{}; index + 3u < image.words.size(); ++index) {
    if (image.words[index] != static_cast<std::uint32_t>(NODE_ATTR)) {
      continue;
    }
    const auto packed = image.words[index + 2u];
    if (((packed >> 8u) & 0xffu) == NODE_ATTR_OUTPUT_FLOAT &&
        ((packed >> 16u) & 0xffffu) == 0u && image.words[index + 3u] == 0u) {
      result.emplace_back(image.words[index + 1u]);
      index += 3u;
    }
  }
  return result;
}

void test_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Info scene did not import");

  for (const auto &family : families) {
    const auto material_name = std::string{family.name} + " Material";
    const MaterialDesc *material = nullptr;
    for (const auto &[id, candidate] : imported.scene->materials) {
      static_cast<void>(id);
      if (candidate.name == material_name) {
        material = &candidate;
        break;
      }
    }
    require(material != nullptr, "Info material is missing");
    const auto semantic_count = std::ranges::count_if(
        material->shader.nodes(), [&](const ShaderNode &node) noexcept {
          return node.type == family.semantic_type;
        });
    require(semantic_count == 1,
            "raw Info outputs did not share one semantic source node");

    const ShaderCompiler frontend{make_core_node_registry()};
    const auto shader = frontend.compile(material->shader);
    require(shader.ok(), "imported Info graph did not validate");
    AttributeIDMap attributes;
    ImageIDMap images;
    const auto image = compile_shader(
        *shader.program, attributes, images,
        ShaderCompileContext{.background = false});
    require(image.valid, image.diagnostic);
    require(direct_records(image, family.direct_opcode,
                           family.maximum_info_type) ==
                std::vector<std::uint32_t>{family.direct_values.begin(),
                                           family.direct_values.end()} &&
                attribute_records(image) ==
                    std::vector<std::uint32_t>{family.attribute_values.begin(),
                                               family.attribute_values.end()},
            "imported Info node emitted the wrong Cycles records");
  }
}

} // namespace

int main() {
  try {
    test_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

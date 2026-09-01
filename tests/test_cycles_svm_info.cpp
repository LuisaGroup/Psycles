#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

enum class Encoding : std::uint8_t { direct, attribute };

struct InfoOutput {
  std::string_view name;
  SocketType type;
  Encoding encoding;
  std::uint32_t value;
};

struct DirectPayload {
  std::uint32_t info_type;
  std::uint32_t packed_output;

  auto operator<=>(const DirectPayload &) const = default;
};

struct AttributePayload {
  std::uint32_t attribute;
  std::uint32_t packed_output;
  std::uint32_t bump_filter_width;

  auto operator<=>(const AttributePayload &) const = default;
};

constexpr auto object_outputs = std::array{
    InfoOutput{"Location", SocketType::vector, Encoding::direct,
               NODE_INFO_OB_LOCATION},
    InfoOutput{"Color", SocketType::color, Encoding::direct,
               NODE_INFO_OB_COLOR},
    InfoOutput{"Alpha", SocketType::floating, Encoding::direct,
               NODE_INFO_OB_ALPHA},
    InfoOutput{"ObjectIndex", SocketType::floating, Encoding::direct,
               NODE_INFO_OB_INDEX},
    InfoOutput{"MaterialIndex", SocketType::floating, Encoding::direct,
               NODE_INFO_MAT_INDEX},
    InfoOutput{"Random", SocketType::floating, Encoding::direct,
               NODE_INFO_OB_RANDOM},
};

constexpr auto particle_outputs = std::array{
    InfoOutput{"Index", SocketType::floating, Encoding::direct,
               NODE_INFO_PAR_INDEX},
    InfoOutput{"Random", SocketType::floating, Encoding::direct,
               NODE_INFO_PAR_RANDOM},
    InfoOutput{"Age", SocketType::floating, Encoding::direct,
               NODE_INFO_PAR_AGE},
    InfoOutput{"Lifetime", SocketType::floating, Encoding::direct,
               NODE_INFO_PAR_LIFETIME},
    InfoOutput{"Location", SocketType::point, Encoding::direct,
               NODE_INFO_PAR_LOCATION},
    InfoOutput{"Size", SocketType::floating, Encoding::direct,
               NODE_INFO_PAR_SIZE},
    InfoOutput{"Velocity", SocketType::vector, Encoding::direct,
               NODE_INFO_PAR_VELOCITY},
    InfoOutput{"AngularVelocity", SocketType::vector, Encoding::direct,
               NODE_INFO_PAR_ANGULAR_VELOCITY},
};

constexpr auto hair_outputs = std::array{
    InfoOutput{"IsStrand", SocketType::floating, Encoding::direct,
               NODE_INFO_CURVE_IS_STRAND},
    InfoOutput{"Intercept", SocketType::floating, Encoding::attribute,
               ATTR_STD_CURVE_INTERCEPT},
    InfoOutput{"Length", SocketType::floating, Encoding::attribute,
               ATTR_STD_CURVE_LENGTH},
    InfoOutput{"Thickness", SocketType::floating, Encoding::direct,
               NODE_INFO_CURVE_THICKNESS},
    InfoOutput{"TangentNormal", SocketType::normal, Encoding::direct,
               NODE_INFO_CURVE_TANGENT_NORMAL},
    InfoOutput{"Random", SocketType::floating, Encoding::attribute,
               ATTR_STD_CURVE_RANDOM},
};

constexpr auto point_outputs = std::array{
    InfoOutput{"Position", SocketType::point, Encoding::direct,
               NODE_INFO_POINT_POSITION},
    InfoOutput{"Radius", SocketType::floating, Encoding::direct,
               NODE_INFO_POINT_RADIUS},
    InfoOutput{"Random", SocketType::floating, Encoding::attribute,
               ATTR_STD_POINT_RANDOM},
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] bool connect_output_to_emission(
    ShaderGraph &graph, OutputRef source, SocketType type,
    NodeId emission) {
  if (type == SocketType::floating) {
    return graph.connect(std::move(source), emission, "Strength");
  }
  if (type == SocketType::color) {
    return graph.connect(std::move(source), emission, "Color");
  }

  auto vector_source = std::move(source);
  if (type == SocketType::point) {
    const auto convert =
        graph.add_node(node_type::point_to_vector, "Point to Vector");
    if (!graph.connect(std::move(vector_source), convert, "Point")) {
      return false;
    }
    vector_source = {.node = convert, .socket = "Vector"};
  } else if (type == SocketType::normal) {
    const auto convert =
        graph.add_node(node_type::normal_to_vector, "Normal to Vector");
    if (!graph.connect(std::move(vector_source), convert, "Normal")) {
      return false;
    }
    vector_source = {.node = convert, .socket = "Vector"};
  }
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  return graph.connect(std::move(vector_source), to_color, "Vector") &&
         graph.connect({to_color, "Color"}, emission, "Color");
}

[[nodiscard]] ShaderGraph make_graph(
    std::string_view node_type_name, std::span<const InfoOutput> outputs,
    std::span<const std::size_t> connection_order) {
  ShaderGraph graph;
  const auto info = graph.add_node(std::string{node_type_name}, "Info");
  std::optional<OutputRef> root;
  for (const auto output_index : connection_order) {
    require(output_index < outputs.size(), "Info output index is invalid");
    const auto &output = outputs[output_index];
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(connect_output_to_emission(
                graph,
                OutputRef{.node = info, .socket = std::string{output.name}},
                output.type, emission),
            "failed to connect an Info output");
    OutputRef closure{.node = emission, .socket = "Closure"};
    if (root) {
      const auto add = graph.add_node(node_type::add_closure, "Add Closure");
      require(graph.connect(std::move(*root), add, "A") &&
                  graph.connect(std::move(closure), add, "B"),
              "failed to keep all Info outputs live");
      root = OutputRef{.node = add, .socket = "Closure"};
    } else {
      root = std::move(closure);
    }
  }
  require(root.has_value(), "Info graph has no live outputs");
  graph.set_root(ShaderDomain::surface, std::move(*root));
  return graph;
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "Info graph failed frontend validation");
  AttributeIDMap attributes;
  ImageIDMap images;
  auto image = compile_shader(*shader.program, attributes, images,
                              ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);
  return image;
}

[[nodiscard]] std::vector<std::uint32_t> direct_records(
    const ShaderImage &image, ShaderNodeType opcode,
    std::uint32_t maximum_info_type) {
  std::vector<std::uint32_t> result;
  for (auto index = std::size_t{}; index + 2u < image.words.size(); ++index) {
    if (image.words[index] != static_cast<std::uint32_t>(opcode) ||
        image.words[index + 1u] > maximum_info_type ||
        (image.words[index + 2u] & 0xffffff00u) != 0u) {
      continue;
    }
    result.emplace_back(image.words[index + 1u]);
    index += 2u;
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
    if (((packed >> 8u) & 0xffu) != NODE_ATTR_OUTPUT_FLOAT ||
        ((packed >> 16u) & 0xffffu) != 0u || image.words[index + 3u] != 0u) {
      continue;
    }
    result.emplace_back(image.words[index + 1u]);
    index += 3u;
  }
  return result;
}

[[nodiscard]] std::vector<DirectPayload> exact_direct_payloads(
    const ShaderImage &image, ShaderNodeType opcode,
    std::uint32_t maximum_info_type) {
  std::vector<DirectPayload> result;
  for (auto index = std::size_t{}; index + 2u < image.words.size(); ++index) {
    if (image.words[index] == static_cast<std::uint32_t>(opcode) &&
        image.words[index + 1u] <= maximum_info_type &&
        (image.words[index + 2u] & 0xffffff00u) == 0u) {
      result.emplace_back(DirectPayload{image.words[index + 1u],
                                        image.words[index + 2u]});
      index += 2u;
    }
  }
  return result;
}

[[nodiscard]] std::vector<AttributePayload>
exact_attribute_payloads(const ShaderImage &image) {
  std::vector<AttributePayload> result;
  for (auto index = std::size_t{}; index + 3u < image.words.size(); ++index) {
    if (image.words[index] != static_cast<std::uint32_t>(NODE_ATTR)) {
      continue;
    }
    const auto packed = image.words[index + 2u];
    if (((packed >> 8u) & 0xffu) == NODE_ATTR_OUTPUT_FLOAT &&
        ((packed >> 16u) & 0xffffu) == 0u) {
      result.emplace_back(AttributePayload{image.words[index + 1u], packed,
                                            image.words[index + 3u]});
      index += 3u;
    }
  }
  return result;
}

[[nodiscard]] ShaderImage compile_all(
    std::string_view type, std::span<const InfoOutput> outputs) {
  std::vector<std::size_t> reverse_order(outputs.size());
  for (auto index = std::size_t{}; index < outputs.size(); ++index) {
    reverse_order[index] = outputs.size() - index - 1u;
  }
  auto graph = make_graph(type, outputs, reverse_order);
  return compile_graph(graph);
}

template<std::size_t N>
[[nodiscard]] bool contains_exact_record_sequence(
    const ShaderImage &image,
    const std::array<std::uint32_t, N> &expected) {
  return std::search(image.words.begin(), image.words.end(), expected.begin(),
                     expected.end()) != image.words.end();
}

void test_external_cycles_all_output_payloads() {
  // Frozen from the Cycles 5.2.1 diagnostic stream for the canonical
  // info_nodes_matrix probe. These are complete payload words, including
  // Cycles' float3 stack widths and the Hair/Point attribute split.
  static constexpr auto object_expected = std::array{
      DirectPayload{NODE_INFO_OB_LOCATION, 0u},
      DirectPayload{NODE_INFO_OB_COLOR, 3u},
      DirectPayload{NODE_INFO_OB_ALPHA, 6u},
      DirectPayload{NODE_INFO_OB_INDEX, 7u},
      DirectPayload{NODE_INFO_MAT_INDEX, 8u},
      DirectPayload{NODE_INFO_OB_RANDOM, 9u},
  };
  static constexpr auto object_record_sequence =
      std::array<std::uint32_t, 18u>{
          NODE_OBJECT_INFO, NODE_INFO_OB_LOCATION, 0u,
          NODE_OBJECT_INFO, NODE_INFO_OB_COLOR, 3u,
          NODE_OBJECT_INFO, NODE_INFO_OB_ALPHA, 6u,
          NODE_OBJECT_INFO, NODE_INFO_OB_INDEX, 7u,
          NODE_OBJECT_INFO, NODE_INFO_MAT_INDEX, 8u,
          NODE_OBJECT_INFO, NODE_INFO_OB_RANDOM, 9u,
      };
  auto image = compile_all(node_type::object_info, object_outputs);
  require(std::ranges::equal(
              exact_direct_payloads(image, NODE_OBJECT_INFO,
                                    NODE_INFO_OB_RANDOM),
              object_expected) &&
              contains_exact_record_sequence(image,
                                             object_record_sequence),
          "Object Info payload words differ from Cycles 5.2.1");

  static constexpr auto particle_expected = std::array{
      DirectPayload{NODE_INFO_PAR_INDEX, 0u},
      DirectPayload{NODE_INFO_PAR_RANDOM, 1u},
      DirectPayload{NODE_INFO_PAR_AGE, 2u},
      DirectPayload{NODE_INFO_PAR_LIFETIME, 3u},
      DirectPayload{NODE_INFO_PAR_LOCATION, 4u},
      DirectPayload{NODE_INFO_PAR_SIZE, 7u},
      DirectPayload{NODE_INFO_PAR_VELOCITY, 8u},
      DirectPayload{NODE_INFO_PAR_ANGULAR_VELOCITY, 11u},
  };
  static constexpr auto particle_record_sequence =
      std::array<std::uint32_t, 24u>{
          NODE_PARTICLE_INFO, NODE_INFO_PAR_INDEX, 0u,
          NODE_PARTICLE_INFO, NODE_INFO_PAR_RANDOM, 1u,
          NODE_PARTICLE_INFO, NODE_INFO_PAR_AGE, 2u,
          NODE_PARTICLE_INFO, NODE_INFO_PAR_LIFETIME, 3u,
          NODE_PARTICLE_INFO, NODE_INFO_PAR_LOCATION, 4u,
          NODE_PARTICLE_INFO, NODE_INFO_PAR_SIZE, 7u,
          NODE_PARTICLE_INFO, NODE_INFO_PAR_VELOCITY, 8u,
          NODE_PARTICLE_INFO, NODE_INFO_PAR_ANGULAR_VELOCITY, 11u,
      };
  image = compile_all(node_type::particle_info, particle_outputs);
  require(std::ranges::equal(
              exact_direct_payloads(image, NODE_PARTICLE_INFO,
                                    NODE_INFO_PAR_ANGULAR_VELOCITY),
              particle_expected) &&
              contains_exact_record_sequence(image,
                                             particle_record_sequence),
          "Particle Info payload words differ from Cycles 5.2.1");

  static constexpr auto hair_direct_expected = std::array{
      DirectPayload{NODE_INFO_CURVE_IS_STRAND, 0u},
      DirectPayload{NODE_INFO_CURVE_THICKNESS, 3u},
      DirectPayload{NODE_INFO_CURVE_TANGENT_NORMAL, 4u},
  };
  static constexpr auto hair_attribute_expected = std::array{
      AttributePayload{ATTR_STD_CURVE_INTERCEPT, 0x00000101u, 0u},
      AttributePayload{ATTR_STD_CURVE_LENGTH, 0x00000102u, 0u},
      AttributePayload{ATTR_STD_CURVE_RANDOM, 0x00000107u, 0u},
  };
  static constexpr auto hair_record_sequence =
      std::array<std::uint32_t, 21u>{
          NODE_HAIR_INFO, NODE_INFO_CURVE_IS_STRAND, 0u,
          NODE_ATTR, ATTR_STD_CURVE_INTERCEPT, 0x00000101u, 0u,
          NODE_ATTR, ATTR_STD_CURVE_LENGTH, 0x00000102u, 0u,
          NODE_HAIR_INFO, NODE_INFO_CURVE_THICKNESS, 3u,
          NODE_HAIR_INFO, NODE_INFO_CURVE_TANGENT_NORMAL, 4u,
          NODE_ATTR, ATTR_STD_CURVE_RANDOM, 0x00000107u, 0u,
      };
  image = compile_all(node_type::hair_info, hair_outputs);
  require(std::ranges::equal(
              exact_direct_payloads(image, NODE_HAIR_INFO,
                                    NODE_INFO_CURVE_RANDOM),
              hair_direct_expected) &&
              std::ranges::equal(exact_attribute_payloads(image),
                                 hair_attribute_expected) &&
              contains_exact_record_sequence(image,
                                             hair_record_sequence),
          "Hair Info payload words differ from Cycles 5.2.1");

  static constexpr auto point_direct_expected = std::array{
      DirectPayload{NODE_INFO_POINT_POSITION, 0u},
      DirectPayload{NODE_INFO_POINT_RADIUS, 3u},
  };
  static constexpr auto point_attribute_expected = std::array{
      AttributePayload{ATTR_STD_POINT_RANDOM, 0x00000104u, 0u},
  };
  static constexpr auto point_record_sequence =
      std::array<std::uint32_t, 10u>{
          NODE_POINT_INFO, NODE_INFO_POINT_POSITION, 0u,
          NODE_POINT_INFO, NODE_INFO_POINT_RADIUS, 3u,
          NODE_ATTR, ATTR_STD_POINT_RANDOM, 0x00000104u, 0u,
      };
  image = compile_all(node_type::point_info, point_outputs);
  require(std::ranges::equal(
              exact_direct_payloads(image, NODE_POINT_INFO,
                                    NODE_INFO_POINT_RANDOM),
              point_direct_expected) &&
              std::ranges::equal(exact_attribute_payloads(image),
                                 point_attribute_expected) &&
              contains_exact_record_sequence(image,
                                             point_record_sequence),
          "Point Info payload words differ from Cycles 5.2.1");
}

void test_schema(std::string_view type,
                 std::span<const InfoOutput> outputs) {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(type);
  require(schema != nullptr && schema->inputs.empty() &&
              schema->outputs.size() == outputs.size(),
          "Info typed schema size changed");
  for (auto index = std::size_t{}; index < outputs.size(); ++index) {
    require(schema->outputs[index].name == outputs[index].name &&
                schema->outputs[index].type == outputs[index].type,
            "Info typed output order changed");
  }
}

void test_family(std::string_view type, ShaderNodeType direct_opcode,
                 std::uint32_t maximum_info_type,
                 std::span<const InfoOutput> outputs) {
  test_schema(type, outputs);
  for (auto index = std::size_t{}; index < outputs.size(); ++index) {
    const std::array live{index};
    auto graph = make_graph(type, outputs, live);
    const auto image = compile_graph(graph);
    const auto direct = direct_records(image, direct_opcode, maximum_info_type);
    const auto attributes = attribute_records(image);
    if (outputs[index].encoding == Encoding::direct) {
      require(direct.size() == 1u && direct.front() == outputs[index].value &&
                  attributes.empty() && image.node_types_used[direct_opcode],
              "Info output did not emit its exact direct Cycles payload");
    } else {
      require(direct.empty() && attributes.size() == 1u &&
                  attributes.front() == outputs[index].value &&
                  image.node_types_used[NODE_ATTR],
              "Info output did not emit its exact Cycles attribute payload");
    }
  }

  std::vector<std::size_t> reverse_order(outputs.size());
  for (auto index = std::size_t{}; index < outputs.size(); ++index) {
    reverse_order[index] = outputs.size() - index - 1u;
  }
  auto graph = make_graph(type, outputs, reverse_order);
  const auto image = compile_graph(graph);
  const auto direct = direct_records(image, direct_opcode, maximum_info_type);
  const auto attributes = attribute_records(image);
  std::vector<std::uint32_t> expected_direct;
  std::vector<std::uint32_t> expected_attributes;
  for (const auto &output : outputs) {
    (output.encoding == Encoding::direct ? expected_direct
                                         : expected_attributes)
        .emplace_back(output.value);
  }
  require(direct == expected_direct && attributes == expected_attributes,
          "Info records do not follow Cycles' fixed compile order");
}

} // namespace

int main() {
  test_external_cycles_all_output_payloads();
  test_family(node_type::object_info, NODE_OBJECT_INFO,
              NODE_INFO_OB_RANDOM, object_outputs);
  test_family(node_type::particle_info, NODE_PARTICLE_INFO,
              NODE_INFO_PAR_ANGULAR_VELOCITY, particle_outputs);
  test_family(node_type::hair_info, NODE_HAIR_INFO,
              NODE_INFO_CURVE_RANDOM, hair_outputs);
  test_family(node_type::point_info, NODE_POINT_INFO,
              NODE_INFO_POINT_RANDOM, point_outputs);
  return EXIT_SUCCESS;
}

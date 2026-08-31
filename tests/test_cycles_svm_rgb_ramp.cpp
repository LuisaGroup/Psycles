#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-5f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] std::string table_string(const packed_float4 &a,
                                       const packed_float4 &b) {
  std::ostringstream stream;
  stream << std::setprecision(9) << "0," << a.x << ',' << a.y << ',' << a.z
         << ',' << a.w << ";1," << b.x << ',' << b.y << ',' << b.z << ','
         << b.w;
  return stream.str();
}

[[nodiscard]] ShaderGraph make_constant_graph(const packed_float4 &a,
                                              const packed_float4 &b,
                                              float factor, bool interpolate,
                                              bool alpha) {
  ShaderGraph graph;
  const auto ramp = graph.add_node(node_type::color_ramp, "RGB Ramp");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid =
      graph.set_input(ramp, "Factor", SocketValue::floating(factor)) &&
      graph.set_property(ramp, "Sampled", SocketValue::boolean(true)) &&
      graph.set_property(
          ramp, "Interpolation",
          SocketValue::string(interpolate ? "LINEAR" : "CONSTANT")) &&
      graph.set_property(ramp, "Table",
                         SocketValue::string(table_string(a, b)));
  if (alpha) {
    const auto scalar_to_color =
        graph.add_node(node_type::scalar_to_color, "Alpha to Color");
    valid = valid && graph.connect({ramp, "Alpha"}, scalar_to_color, "Value") &&
            graph.connect({scalar_to_color, "Color"}, emission, "Color");
  } else {
    valid = valid && graph.connect({ramp, "Color"}, emission, "Color");
  }
  require(valid, "failed to construct constant RGB Ramp graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph
make_dynamic_graph(bool color, bool sampled = true,
                   std::string interpolation = "LINEAR") {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Generated Coordinates");
  const auto point_to_vector =
      graph.add_node(node_type::point_to_vector, "Point to Vector");
  const auto gradient =
      graph.add_node(node_type::gradient_texture, "Linear Gradient");
  const auto ramp = graph.add_node(node_type::color_ramp, "RGB Ramp");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const packed_float4 a{0.1f, 0.2f, 0.3f, 0.4f};
  const packed_float4 b{0.9f, 0.8f, 0.7f, 0.6f};
  auto valid =
      graph.connect({coordinates, "Generated"}, point_to_vector, "Point") &&
      graph.connect({point_to_vector, "Vector"}, gradient, "Vector") &&
      graph.connect({gradient, "Factor"}, ramp, "Factor") &&
      graph.set_property(gradient, "GradientType",
                         SocketValue::string("LINEAR")) &&
      graph.set_property(ramp, "Sampled", SocketValue::boolean(sampled)) &&
      graph.set_property(ramp, "Interpolation",
                         SocketValue::string(std::move(interpolation))) &&
      graph.set_property(ramp, "Table",
                         SocketValue::string(table_string(a, b)));
  if (color) {
    valid = valid && graph.connect({ramp, "Color"}, emission, "Color");
  } else {
    const auto scalar_to_color =
        graph.add_node(node_type::scalar_to_color, "Alpha to Color");
    valid = valid && graph.connect({ramp, "Alpha"}, scalar_to_color, "Value") &&
            graph.connect({scalar_to_color, "Color"}, emission, "Color");
  }
  require(valid, "failed to construct dynamic RGB Ramp graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "RGB Ramp graph failed frontend validation");
  AttributeIDMap attributes;
  ImageIDMap images;
  return compile_shader(*shader.program, attributes, images,
                        ShaderCompileContext{.background = false});
}

struct ExternalCase {
  bool interpolate;
  float factor;
  packed_float4 a;
  packed_float4 b;
  std::array<std::uint32_t, 3u> expected;
  bool alpha;
};

// Adjacent table entries and final constants captured from the external
// Cycles 5.2.1 color_ramp_modes and color_ramp_alpha_modes probes. Blender's
// five interpolation families and HSV/HSL modes are already represented by
// the sampled table; NODE_RGB_RAMP itself sees only this pair and the exact
// fractional table coordinate.
constexpr std::array external_cases{
    ExternalCase{true,
                 f32(0x3e937480u),
                 {f32(0x3f7721cbu), f32(0x3da149b6u), f32(0x3e2cd605u),
                  f32(0x3e370416u)},
                 {f32(0x3f770e98u), f32(0x3da83ce9u), f32(0x3e2c1c6bu),
                  f32(0x3e3a2a7du)},
                 {0x3f771c44u, 0x3da34a1fu, 0x3e2ca091u},
                 false},
    ExternalCase{false,
                 0.0f,
                 {f32(0x3f7851ecu), f32(0x3ccccccdu), f32(0x3e3851ecu),
                  f32(0x3e051eb8u)},
                 {f32(0x3f7851ecu), f32(0x3ccccccdu), f32(0x3e3851ecu),
                  f32(0x3e051eb8u)},
                 {0x3f7851ecu, 0x3ccccccdu, 0x3e3851ecu},
                 false},
    ExternalCase{true,
                 f32(0x3eed9200u),
                 {1.0f, f32(0x3f0c3051u), 0.0f, f32(0x3f16785bu)},
                 {1.0f, f32(0x3f0d307bu), 0.0f, f32(0x3f175e8fu)},
                 {0x3f800000u, 0x3f0ca767u, 0x00000000u},
                 false},
    ExternalCase{true,
                 f32(0x3f73b680u),
                 {f32(0x3e7c21bdu), f32(0x3ce815dbu), f32(0x3f7374ffu),
                  f32(0x3ea8aed8u)},
                 {f32(0x3e64cc06u), f32(0x3ce87f24u), f32(0x3f7361cbu),
                  f32(0x3eaa420bu)},
                 {0x3e65eab0u, 0x3ce879f6u, 0x3f7362b7u},
                 false},
    ExternalCase{true,
                 f32(0x3e937480u),
                 {f32(0x3f7721cbu), f32(0x3da149b6u), f32(0x3e2cd605u),
                  f32(0x3e370416u)},
                 {f32(0x3f770e98u), f32(0x3da83ce9u), f32(0x3e2c1c6bu),
                  f32(0x3e3a2a7du)},
                 {0x3e37ec55u, 0u, 0u},
                 true},
    ExternalCase{true,
                 f32(0x3eed9200u),
                 {1.0f, f32(0x3f0c3051u), 0.0f, f32(0x3f16785bu)},
                 {1.0f, f32(0x3f0d307bu), 0.0f, f32(0x3f175e8fu)},
                 {0x3f16e357u, 0u, 0u},
                 true}};

void test_external_constant_folds() {
  for (auto case_index = std::size_t{}; case_index < external_cases.size();
       ++case_index) {
    const auto &test = external_cases[case_index];
    auto graph = make_constant_graph(test.a, test.b, test.factor,
                                     test.interpolate, test.alpha);
    const auto image = compile_graph(graph);
    require(image.valid, image.diagnostic.c_str());
    const auto x = test.expected[0u];
    const auto y = test.alpha ? x : test.expected[1u];
    const auto z = test.alpha ? x : test.expected[2u];
    static constexpr std::array structure{
        0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu, 0x00000005u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000003u, 0x000000ffu,
        0x00000000u, 0x00000000u, 0x00000000u};
    require(image.words.size() == structure.size(),
            "RGB Ramp constant-fold stream has the wrong size");
    for (auto index = std::size_t{}; index < structure.size(); ++index) {
      if (index < 5u || index > 7u) {
        require(image.words[index] == structure[index],
                "RGB Ramp constant-fold stream structure differs from Cycles");
      }
    }
    if (!near(f32(image.words[5u]), f32(x)) ||
        !near(f32(image.words[6u]), f32(y)) ||
        !near(f32(image.words[7u]), f32(z))) {
      std::cerr << "RGB Ramp external case " << case_index
                << " differs materially from Cycles: got ("
                << f32(image.words[5u]) << ", " << f32(image.words[6u]) << ", "
                << f32(image.words[7u]) << "), expected (" << f32(x) << ", "
                << f32(y) << ", " << f32(z) << ")\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

void test_dynamic_payload_and_output_liveness() {
  for (const auto color_live : {true, false}) {
    auto graph = make_dynamic_graph(color_live);
    const auto image = compile_graph(graph);
    require(image.valid, image.diagnostic.c_str());
    const auto opcode = std::find(image.words.begin(), image.words.end(),
                                  static_cast<std::uint32_t>(NODE_RGB_RAMP));
    require(opcode != image.words.end(), "RGB Ramp opcode is missing");
    const auto begin = static_cast<std::size_t>(opcode - image.words.begin());
    require(begin + 12u <= image.words.size(),
            "RGB Ramp payload or table is truncated");
    require(image.words[begin + 1u] == 2u,
            "RGB Ramp did not preserve the table size");
    require((image.words[begin + 2u] >> 8u) ==
                (SVM_INPUT_STACK_OFFSET_MASK >> 8u),
            "RGB Ramp factor is not a typed stack input");
    const auto packed = image.words[begin + 3u];
    require((packed & 0xffu) == 1u, "RGB Ramp lost the interpolation byte");
    const auto color_offset = (packed >> 8u) & 0xffu;
    const auto alpha_offset = (packed >> 16u) & 0xffu;
    require((color_live && color_offset != SVM_STACK_INVALID &&
             alpha_offset == SVM_STACK_INVALID) ||
                (!color_live && color_offset == SVM_STACK_INVALID &&
                 alpha_offset != SVM_STACK_INVALID),
            "RGB Ramp output liveness differs from Cycles");
    static constexpr std::array table_words{
        0x3dcccccdu, 0x3e4ccccdu, 0x3e99999au, 0x3ecccccdu,
        0x3f666666u, 0x3f4ccccdu, 0x3f333333u, 0x3f19999au};
    require(std::equal(table_words.begin(), table_words.end(),
                       image.words.begin() +
                           static_cast<std::ptrdiff_t>(begin + 4u)),
            "RGB Ramp table words differ from the typed Cycles payload");
  }
}

void test_schema_and_invalid_tables() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::color_ramp);
  require(schema != nullptr && schema->outputs.size() == 2u &&
              schema->outputs[0u].name == "Color" &&
              schema->outputs[0u].type == SocketType::color &&
              schema->outputs[1u].name == "Alpha" &&
              schema->outputs[1u].type == SocketType::floating,
          "RGB Ramp schema lost its typed Color and Alpha outputs");

  auto unsampled = make_dynamic_graph(true, false);
  require(!compile_graph(unsampled).valid,
          "unsampled control-point RGB Ramp was accepted as Cycles bytecode");
  auto invalid_interpolation =
      make_dynamic_graph(true, true, "NOT_AN_INTERPOLATION");
  require(!compile_graph(invalid_interpolation).valid,
          "invalid RGB Ramp interpolation was accepted");
}

} // namespace

int main() {
  test_external_constant_folds();
  test_dynamic_payload_and_output_liveness();
  test_schema_and_invalid_tables();
  return EXIT_SUCCESS;
}

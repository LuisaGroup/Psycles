#include "cycles_svm_imported_word_fixture.h"

#include <psycles/compiler/cycles_svm_node_types.h>

namespace {
using namespace psycles::compiler::cycles_svm;
using psycles::test_support::require;

// Only the two explicit RGB float payloads in a constant diffuse image admit
// four ULPs. The grammar is derived from the original typed structs; no opcode,
// jump, padding, stack address, dynamic input or general word is relaxed.
bool constant_diffuse_literals(std::span<const std::uint32_t> actual,
                               std::span<const std::uint32_t> expected) {
  constexpr auto weight = 4 + 1 + sizeof(SVMNodeGeometry) / 4;
  constexpr auto bsdf = weight + 1 + sizeof(SVMNodeClosureSetWeight) / 4;
  constexpr auto color = bsdf + 1 + sizeof(SVMNodeClosureBsdf) / 4;
  constexpr auto end = color + sizeof(SVMNodeDiffuseBsdfData) / 4;
  if (actual.size() != end + 3 || expected.size() != end + 3 ||
      expected[0] != NODE_SHADER_JUMP || expected[1] != 4 ||
      expected[2] != end + 1 || expected[3] != end + 2 ||
      expected[4] != NODE_GEOMETRY || expected[weight] != NODE_CLOSURE_SET_WEIGHT ||
      expected[bsdf] != NODE_CLOSURE_BSDF || expected[bsdf + 1] != CLOSURE_BSDF_DIFFUSE_ID ||
      expected[end] != NODE_END || expected[end + 1] != NODE_END || expected[end + 2] != NODE_END) {
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto rgb = (i > weight && i < weight + 4) || (i >= color && i < color + 3);
    const auto a = std::bit_cast<float>(actual[i]);
    const auto b = std::bit_cast<float>(expected[i]);
    if (rgb && (!std::isfinite(a) || !std::isfinite(b))) { return false; }
    if (actual[i] == expected[i]) { continue; }
    const auto distance = actual[i] > expected[i] ? actual[i] - expected[i] : expected[i] - actual[i];
    if (!rgb || !std::isfinite(a) || !std::isfinite(b) || distance > 4) { return false; }
  }
  return true;
}

void check_literal_comparison_and_origin_hash() {
  using namespace psycles;
  contract::ShaderGraph graph;
  const auto color = graph.add_node(compiler::node_type::constant_color);
  const auto diffuse = graph.add_node(compiler::node_type::diffuse_bsdf);
  require(graph.set_input(color, "Color", contract::SocketValue::color({0.25f, 0.5f, 0.75f})), "color input");
  require(graph.connect({color, "Color"}, diffuse, "Color"), "color link");
  graph.set_root(contract::ShaderDomain::surface, contract::OutputRef{diffuse, "Closure"});
  compiler::ShaderCompiler compiler{compiler::make_core_node_registry()};
  const auto authored = compiler.compile(graph);
  require(authored.ok(), "authored graph");
  graph.find(color)->origin = contract::ShaderNodeOrigin::blender_input_value;
  const auto primitive = compiler.compile(graph);
  require(primitive.ok() && authored.program->analysis().structure_signature !=
          primitive.program->analysis().structure_signature, "source origin must enter structural hash");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto original = compile_shader(*authored.program, attributes, images, ShaderCompileContext{});
  require(original.valid && constant_diffuse_literals(original.words, original.words), "typed constant grammar");
  constexpr auto weight = 4 + 1 + sizeof(SVMNodeGeometry) / 4;
  for (std::size_t i = 0; i < original.words.size(); ++i) {
    auto changed = original.words;
    // More than the admitted rounding in float lanes, and a structural
    // mutation everywhere else. Neither must be accepted.
    changed[i] += 5;
    require(!constant_diffuse_literals(changed, original.words), "literal comparison hid a changed word");
  }
  auto rounded = original.words;
  rounded[weight + 1] += 4;
  require(constant_diffuse_literals(rounded, original.words), "four-ULP literal comparison");
  auto nonfinite = original.words;
  nonfinite[weight + 1] = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity());
  require(!constant_diffuse_literals(nonfinite, nonfinite), "non-finite fields are not literal tolerance");
  graph.find(color)->origin = contract::ShaderNodeOrigin::blender_implicit_conversion;
  const auto invalid = compiler.compile(graph);
  require(invalid.ok() && !compile_shader(*invalid.program, attributes, images, ShaderCompileContext{}).valid,
          "source metadata must not turn an arbitrary node into a forwarding conversion");
}

void check_color_space_import() {
  psycles::test_support::Bundle bundle{"cycles_group_forward"};
  const auto before = psycles::adapter::load_blender_scene_bundle(bundle.path);
  require(before.ok(), "source color-space fixture");
  const auto scene_file = bundle.path / "scene.json";
  std::ifstream input{scene_file};
  std::string json{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  const auto first = json.find("\"blender_luma\":[");
  require(first != std::string::npos, "original exporter must retain Blender luma coefficients");
  const auto last = json.find(']', first);
  require(last != std::string::npos, "luma input array");
  json.replace(first, last - first + 1, "\"blender_luma\":[0.5,0.25,0.25]");
  input.close();
  { std::ofstream output{scene_file}; output << json; require(output.good(), "color-space test input"); }
  const auto after = psycles::adapter::load_blender_scene_bundle(bundle.path);
  require(after.ok() && after.scene->shader_color_space.blender_luma == psycles::Vec3f{0.5f, 0.25f, 0.25f},
          "Blender luma must come from source metadata, not a fixed shader coefficient");
  require(after.scene->shader_color_space.xyz_to_r == before.scene->shader_color_space.xyz_to_r &&
          after.scene->shader_color_space.xyz_to_g == before.scene->shader_color_space.xyz_to_g &&
          after.scene->shader_color_space.xyz_to_b == before.scene->shader_color_space.xyz_to_b,
          "Blender primitive luma must not replace the native Cycles XYZ transform");
}
} // namespace

int main() {
  try {
    check_literal_comparison_and_origin_hash();
    check_color_space_import();
    psycles::test_support::check_imported_words("cycles_group_forward", 34, false, constant_diffuse_literals);
    std::cout << "34 group-forwarding graphs match Cycles (typed literal roundoff only)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

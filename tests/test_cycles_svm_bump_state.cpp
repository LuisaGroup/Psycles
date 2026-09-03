#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

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

namespace {

using psycles::adapter::load_blender_scene_bundle;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

class TemporaryDirectory final {
private:
  std::filesystem::path _path;

public:
  TemporaryDirectory() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
            ("psycles-bump-state-" + std::to_string(nonce));
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

// Exact material projection exported by Blender Cycles 5.2.1 (9e2066aef7ef).
// The probe deliberately retains the linked socket defaults because Cycles'
// graph importer and constant folding inspect the same typed payloads.
constexpr std::string_view scene_json = R"JSON({"schema":"psycles.blender-scene.v2","images":[],"node_groups":[],"materials":[{"cycles_sync":{"pass_id":0,"shader_index":8},"displacement_method":"BOTH","emission_sampling":"AUTO","name":"Both Displacement","node_tree":{"displacement_root":{"node":"Both Displacement Displacement","socket":"Displacement"},"links":[{"from_node":"Both Displacement Sine Height","from_socket":"Value","to_node":"Both Displacement Displacement","to_socket":"Height"},{"from_node":"Both Displacement Separate UV","from_socket":"X","to_node":"Both Displacement Frequency","to_socket":"Value"},{"from_node":"Both Displacement Coordinates","from_socket":"UV","to_node":"Both Displacement Separate UV","to_socket":"Vector"},{"from_node":"Both Displacement Frequency","from_socket":"Value","to_node":"Both Displacement Sine Height","to_socket":"Value"},{"from_node":"Both Displacement Displacement","from_socket":"Displacement","to_node":"Material Output","to_socket":"Displacement"},{"from_node":"Both Displacement Diffuse","from_socket":"BSDF","to_node":"Material Output","to_socket":"Surface"}],"name":"Shader Nodetree","nodes":[{"bl_idname":"ShaderNodeTexCoord","image":null,"inputs":[],"internal_links":[],"label":"","mute":false,"name":"Both Displacement Coordinates","node_tree":null,"outputs":[{"default":[0.0,0.0,0.0],"identifier":"Generated","linked":false,"name":"Generated","type":"NodeSocketVector"},{"default":[0.0,0.0,0.0],"identifier":"Normal","linked":false,"name":"Normal","type":"NodeSocketVector"},{"default":[0.0,0.0,0.0],"identifier":"UV","linked":true,"name":"UV","type":"NodeSocketVector"},{"default":[0.0,0.0,0.0],"identifier":"Object","linked":false,"name":"Object","type":"NodeSocketVector"},{"default":[0.0,0.0,0.0],"identifier":"Camera","linked":false,"name":"Camera","type":"NodeSocketVector"},{"default":[0.0,0.0,0.0],"identifier":"Window","linked":false,"name":"Window","type":"NodeSocketVector"},{"default":[0.0,0.0,0.0],"identifier":"Reflection","linked":false,"name":"Reflection","type":"NodeSocketVector"}],"properties":{"from_instancer":false},"special":{},"type":"TEX_COORD"},{"bl_idname":"ShaderNodeBsdfDiffuse","image":null,"inputs":[{"default":[0.07999999821186066,0.20000000298023224,0.6200000047683716,1.0],"identifier":"Color","linked":false,"name":"Color","type":"NodeSocketColor"},{"default":0.3499999940395355,"identifier":"Roughness","linked":false,"name":"Roughness","type":"NodeSocketFloatFactor"},{"default":[0.0,0.0,0.0],"identifier":"Normal","linked":false,"name":"Normal","type":"NodeSocketVector"},{"default":0.0,"identifier":"Weight","linked":false,"name":"Weight","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"Both Displacement Diffuse","node_tree":null,"outputs":[{"identifier":"BSDF","linked":true,"name":"BSDF","type":"NodeSocketShader"}],"properties":{},"special":{},"type":"BSDF_DIFFUSE"},{"bl_idname":"ShaderNodeDisplacement","image":null,"inputs":[{"default":0.0,"identifier":"Height","linked":true,"name":"Height","type":"NodeSocketFloat"},{"default":0.0,"identifier":"Midlevel","linked":false,"name":"Midlevel","type":"NodeSocketFloat"},{"default":0.2199999988079071,"identifier":"Scale","linked":false,"name":"Scale","type":"NodeSocketFloat"},{"default":[0.0,0.0,0.0],"identifier":"Normal","linked":false,"name":"Normal","type":"NodeSocketVector"}],"internal_links":[{"from_socket":"Normal","to_socket":"Displacement"}],"label":"","mute":false,"name":"Both Displacement Displacement","node_tree":null,"outputs":[{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"}],"properties":{"space":"OBJECT"},"special":{},"type":"DISPLACEMENT"},{"bl_idname":"ShaderNodeMath","image":null,"inputs":[{"default":11.0,"identifier":"Value","linked":true,"name":"Value","type":"NodeSocketFloat"},{"default":0.5,"identifier":"Value_001","linked":false,"name":"Value","type":"NodeSocketFloat"},{"default":0.5,"identifier":"Value_002","linked":false,"name":"Value","type":"NodeSocketFloat"}],"internal_links":[{"from_socket":"Value","to_socket":"Value"}],"label":"","mute":false,"name":"Both Displacement Frequency","node_tree":null,"outputs":[{"default":0.0,"identifier":"Value","linked":true,"name":"Value","type":"NodeSocketFloat"}],"properties":{"operation":"MULTIPLY","use_clamp":false},"special":{},"type":"MATH"},{"bl_idname":"ShaderNodeSeparateXYZ","image":null,"inputs":[{"default":[0.0,0.0,0.0],"identifier":"Vector","linked":true,"name":"Vector","type":"NodeSocketVector"}],"internal_links":[{"from_socket":"Vector","to_socket":"X"},{"from_socket":"Vector","to_socket":"Y"},{"from_socket":"Vector","to_socket":"Z"}],"label":"","mute":false,"name":"Both Displacement Separate UV","node_tree":null,"outputs":[{"default":0.0,"identifier":"X","linked":true,"name":"X","type":"NodeSocketFloat"},{"default":0.0,"identifier":"Y","linked":false,"name":"Y","type":"NodeSocketFloat"},{"default":0.0,"identifier":"Z","linked":false,"name":"Z","type":"NodeSocketFloat"}],"properties":{},"special":{},"type":"SEPXYZ"},{"bl_idname":"ShaderNodeMath","image":null,"inputs":[{"default":0.5,"identifier":"Value","linked":true,"name":"Value","type":"NodeSocketFloat"},{"default":0.5,"identifier":"Value_001","linked":false,"name":"Value","type":"NodeSocketFloat"},{"default":0.5,"identifier":"Value_002","linked":false,"name":"Value","type":"NodeSocketFloat"}],"internal_links":[{"from_socket":"Value","to_socket":"Value"}],"label":"","mute":false,"name":"Both Displacement Sine Height","node_tree":null,"outputs":[{"default":0.0,"identifier":"Value","linked":true,"name":"Value","type":"NodeSocketFloat"}],"properties":{"operation":"SINE","use_clamp":false},"special":{},"type":"MATH"},{"bl_idname":"ShaderNodeOutputMaterial","image":null,"inputs":[{"identifier":"Surface","linked":true,"name":"Surface","type":"NodeSocketShader"},{"identifier":"Volume","linked":false,"name":"Volume","type":"NodeSocketShader"},{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"},{"default":0.0,"identifier":"Thickness","linked":false,"name":"Thickness","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"Material Output","node_tree":null,"outputs":[],"properties":{"is_active_output":true,"target":"ALL"},"special":{},"type":"OUTPUT_MATERIAL"}],"surface_root":{"node":"Both Displacement Diffuse","socket":"BSDF"},"volume_root":null},"surface_render_method":"DITHERED","use_bump_map_correction":true,"use_transparent_shadow":true,"volume_interpolation":"LINEAR","volume_sampling":"MULTIPLE_IMPORTANCE"}],"render":{"width":16,"height":16,"percentage":100,"cycles":{}},"camera":{"name":"Camera","type":"PERSP","transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"clip_start":0.01,"clip_end":100},"geometries":[],"curve_geometries":[],"instances":[],"lights":[],"world":null,"world_environment":null})JSON";

void write_scene_bundle(const std::filesystem::path &directory) {
  {
    std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
  }
  std::ofstream scene{directory / "scene.json"};
  scene << scene_json;
}

void require_words(std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected) {
  if (actual.size() != expected.size()) {
    std::cerr << "Both Displacement has " << actual.size()
              << " words, expected " << expected.size() << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << "  [" << std::dec << index << "] = 0x" << std::hex
                << actual[index] << '\n';
    }
    throw std::runtime_error{
        "Both Displacement word count differs from Cycles 5.2.1"};
  }
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << "Both Displacement differs at word " << index
                << ": got 0x" << std::hex << actual[index]
                << ", expected 0x" << expected[index] << std::dec << '\n';
      for (auto dump_index = std::size_t{}; dump_index < actual.size();
           ++dump_index) {
        std::cerr << "  [" << std::dec << dump_index << "] = 0x" << std::hex
                  << actual[dump_index] << '\n';
      }
      throw std::runtime_error{
          "Both Displacement word stream differs from Cycles 5.2.1"};
    }
  }
}

void test_bump_state_word_image() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  if (!imported.ok()) {
    for (const auto &diagnostic : imported.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(imported.ok(), "Both Displacement scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Both Displacement") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Both Displacement material is missing");
  require(material->displacement_method == DisplacementMethod::both,
          "BOTH displacement policy changed during import");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "Both Displacement graph did not normalize");

  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(
      *shader.program, attributes, images,
      ShaderCompileContext{.background = false,
                           .displacement_method = material->displacement_method});
  require(image.valid, image.diagnostic);

  // Local extraction of global shader 8 from the Cycles oracle: its jump is
  // rebased from (1, 400, 590, 591) to (1, 4, 194, 195).
  static constexpr std::array<std::uint32_t, 238u> expected{
      0x00000001u, 0x00000004u, 0x000000c2u, 0x000000c3u, 0x00000023u, 0x00000000u,
      0x00000016u, 0x00000005u, 0x0000000au, 0x3dcccccdu, 0x00000054u, 0x7fc0000au,
      0x00000000u, 0x00000000u, 0x00000d00u, 0x00000054u, 0x7fc0000au, 0x00000000u,
      0x00000000u, 0x0000ff01u, 0x00000054u, 0x7fc0000au, 0x00000000u, 0x00000000u,
      0x0000ff02u, 0x0000002cu, 0x00000002u, 0x7fc0000du, 0x3f000000u, 0x00000000u,
      0x0000000au, 0x0000002cu, 0x00000004u, 0x7fc0000au, 0x3f000000u, 0x00000000u,
      0x0000000bu, 0x0000000bu, 0x0c000001u, 0x3dcccccdu, 0x0000001au, 0x00000001u,
      0x7fc0000bu, 0x00000000u, 0x3e6147aeu, 0x00000f0cu, 0x0000000bu, 0x0a000001u,
      0x00000000u, 0x0000002du, 0x00000007u, 0x7fc0000fu, 0x00000000u, 0x00000000u,
      0x7fc0000au, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x0000ff0du, 0x00000016u, 0x00000005u, 0x0001000eu, 0x3dcccccdu,
      0x00000054u, 0x7fc0000eu, 0x00000000u, 0x00000000u, 0x00001100u, 0x00000054u,
      0x7fc0000eu, 0x00000000u, 0x00000000u, 0x0000ff01u, 0x00000054u, 0x7fc0000eu,
      0x00000000u, 0x00000000u, 0x0000ff02u, 0x0000002cu, 0x00000002u, 0x7fc00011u,
      0x3f000000u, 0x00000000u, 0x0000000eu, 0x0000002cu, 0x00000004u, 0x7fc0000eu,
      0x3f000000u, 0x00000000u, 0x0000000fu, 0x0000000bu, 0x10000001u, 0x3dcccccdu,
      0x0000001au, 0x00000001u, 0x7fc0000fu, 0x00000000u, 0x3e6147aeu, 0x00001310u,
      0x0000002du, 0x00000007u, 0x7fc00013u, 0x00000000u, 0x00000000u, 0x7fc0000au,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x0000ff0eu, 0x00000016u, 0x00000005u, 0x0002000fu, 0x3dcccccdu, 0x00000054u,
      0x7fc0000fu, 0x00000000u, 0x00000000u, 0x00001200u, 0x00000054u, 0x7fc0000fu,
      0x00000000u, 0x00000000u, 0x0000ff01u, 0x00000054u, 0x7fc0000fu, 0x00000000u,
      0x00000000u, 0x0000ff02u, 0x0000002cu, 0x00000002u, 0x7fc00012u, 0x3f000000u,
      0x00000000u, 0x0000000fu, 0x0000002cu, 0x00000004u, 0x7fc0000fu, 0x3f000000u,
      0x00000000u, 0x00000010u, 0x0000000bu, 0x11000001u, 0x3dcccccdu, 0x0000001au,
      0x00000001u, 0x7fc00010u, 0x00000000u, 0x3e6147aeu, 0x00001411u, 0x0000002du,
      0x00000007u, 0x7fc00014u, 0x00000000u, 0x00000000u, 0x7fc0000au, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x3f800000u, 0x0000ff0fu,
      0x00000021u, 0x3f800000u, 0x3f800000u, 0x3dcccccdu, 0x0d01000au, 0x00100f0eu,
      0x00000022u, 0x00000a10u, 0x00000024u, 0x00000000u, 0x0000000bu, 0x00000001u,
      0x00000000u, 0x00000005u, 0x3da3d70au, 0x3e4ccccdu, 0x3f1eb852u, 0x00000002u,
      0x00000002u, 0x000000ffu, 0x3da3d70au, 0x3e4ccccdu, 0x3f1eb852u, 0x3eb33333u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000015u, 0x00000005u, 0x00000000u,
      0x00000000u, 0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u, 0x00000300u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u, 0x0000ff01u, 0x00000054u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x0000ff02u, 0x0000002cu, 0x00000002u,
      0x7fc00003u, 0x3f000000u, 0x00000000u, 0x00000000u, 0x0000002cu, 0x00000004u,
      0x7fc00000u, 0x3f000000u, 0x00000000u, 0x00000001u, 0x0000000bu, 0x02000001u,
      0x00000000u, 0x0000001au, 0x00000001u, 0x7fc00001u, 0x00000000u, 0x3e6147aeu,
      0x00000502u, 0x00000019u, 0x00000005u, 0x00000000u};

  require_words(image.words, expected);
  require(image.words[4u] == static_cast<std::uint32_t>(NODE_ENTER_BUMP_EVAL) &&
              image.words[168u] == static_cast<std::uint32_t>(NODE_SET_BUMP) &&
              image.words[174u] ==
                  static_cast<std::uint32_t>(NODE_CLOSURE_SET_NORMAL) &&
              image.words[176u] == static_cast<std::uint32_t>(NODE_LEAVE_BUMP_EVAL) &&
              image.words[194u] == static_cast<std::uint32_t>(NODE_END),
          "bump-state node ordering differs from Cycles 5.2.1");
}

} // namespace

int main() {
  try {
    test_bump_state_word_image();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "Cycles SVM bump-state compiler test passed\n";
  return EXIT_SUCCESS;
}

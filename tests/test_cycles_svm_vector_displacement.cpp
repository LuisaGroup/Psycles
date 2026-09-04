#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/material_library.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
            ("psycles-vector-displacement-" + std::to_string(nonce));
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

// Exact unbaked material projection exported by Blender Cycles 5.2.1
// (9e2066aef7ef). The three materials differ only in Vector Displacement
// space, so any stream difference beyond the payload is structural.
constexpr std::string_view scene_json = R"JSON({"schema":"psycles.blender-scene.v2","images":[],"node_groups":[],"materials":[{"cycles_sync":{"pass_id":0,"shader_index":6},"displacement_method":"DISPLACEMENT","emission_sampling":"AUTO","name":"SVM Vector Displacement Object","node_tree":{"displacement_root":{"node":"SVM Vector Displacement Object Vector Displacement","socket":"Displacement"},"links":[{"from_node":"SVM Vector Displacement Object Vector Displacement","from_socket":"Displacement","to_node":"Material Output","to_socket":"Displacement"},{"from_node":"SVM Vector Displacement Object Emission","from_socket":"Emission","to_node":"Material Output","to_socket":"Surface"}],"name":"Shader Nodetree","nodes":[{"bl_idname":"ShaderNodeOutputMaterial","image":null,"inputs":[{"identifier":"Surface","linked":true,"name":"Surface","type":"NodeSocketShader"},{"identifier":"Volume","linked":false,"name":"Volume","type":"NodeSocketShader"},{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"},{"default":0.0,"identifier":"Thickness","linked":false,"name":"Thickness","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"Material Output","node_tree":null,"outputs":[],"properties":{"is_active_output":true,"target":"ALL"},"special":{},"type":"OUTPUT_MATERIAL"},{"bl_idname":"ShaderNodeEmission","image":null,"inputs":[{"default":[0.17000000178813934,0.4300000071525574,0.7900000214576721,1.0],"identifier":"Color","linked":false,"name":"Color","type":"NodeSocketColor"},{"default":1.0,"identifier":"Strength","linked":false,"name":"Strength","type":"NodeSocketFloat"},{"default":0.0,"identifier":"Weight","linked":false,"name":"Weight","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"SVM Vector Displacement Object Emission","node_tree":null,"outputs":[{"identifier":"Emission","linked":true,"name":"Emission","type":"NodeSocketShader"}],"properties":{},"special":{},"type":"EMISSION"},{"bl_idname":"ShaderNodeVectorDisplacement","image":null,"inputs":[{"default":[0.7300000190734863,0.28999999165534973,0.6100000143051147,1.0],"identifier":"Vector","linked":false,"name":"Vector","type":"NodeSocketColor"},{"default":0.3700000047683716,"identifier":"Midlevel","linked":false,"name":"Midlevel","type":"NodeSocketFloat"},{"default":0.4099999964237213,"identifier":"Scale","linked":false,"name":"Scale","type":"NodeSocketFloat"}],"internal_links":[{"from_socket":"Midlevel","to_socket":"Displacement"}],"label":"","mute":false,"name":"SVM Vector Displacement Object Vector Displacement","node_tree":null,"outputs":[{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"}],"properties":{"space":"OBJECT"},"special":{},"type":"VECTOR_DISPLACEMENT"}],"surface_root":{"node":"SVM Vector Displacement Object Emission","socket":"Emission"},"volume_root":null},"surface_render_method":"DITHERED","use_bump_map_correction":true,"use_transparent_shadow":true,"volume_interpolation":"LINEAR","volume_sampling":"MULTIPLE_IMPORTANCE"},{"cycles_sync":{"pass_id":0,"shader_index":5},"displacement_method":"DISPLACEMENT","emission_sampling":"AUTO","name":"SVM Vector Displacement Tangent","node_tree":{"displacement_root":{"node":"SVM Vector Displacement Tangent Vector Displacement","socket":"Displacement"},"links":[{"from_node":"SVM Vector Displacement Tangent Vector Displacement","from_socket":"Displacement","to_node":"Material Output","to_socket":"Displacement"},{"from_node":"SVM Vector Displacement Tangent Emission","from_socket":"Emission","to_node":"Material Output","to_socket":"Surface"}],"name":"Shader Nodetree","nodes":[{"bl_idname":"ShaderNodeOutputMaterial","image":null,"inputs":[{"identifier":"Surface","linked":true,"name":"Surface","type":"NodeSocketShader"},{"identifier":"Volume","linked":false,"name":"Volume","type":"NodeSocketShader"},{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"},{"default":0.0,"identifier":"Thickness","linked":false,"name":"Thickness","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"Material Output","node_tree":null,"outputs":[],"properties":{"is_active_output":true,"target":"ALL"},"special":{},"type":"OUTPUT_MATERIAL"},{"bl_idname":"ShaderNodeEmission","image":null,"inputs":[{"default":[0.17000000178813934,0.4300000071525574,0.7900000214576721,1.0],"identifier":"Color","linked":false,"name":"Color","type":"NodeSocketColor"},{"default":1.0,"identifier":"Strength","linked":false,"name":"Strength","type":"NodeSocketFloat"},{"default":0.0,"identifier":"Weight","linked":false,"name":"Weight","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"SVM Vector Displacement Tangent Emission","node_tree":null,"outputs":[{"identifier":"Emission","linked":true,"name":"Emission","type":"NodeSocketShader"}],"properties":{},"special":{},"type":"EMISSION"},{"bl_idname":"ShaderNodeVectorDisplacement","image":null,"inputs":[{"default":[0.7300000190734863,0.28999999165534973,0.6100000143051147,1.0],"identifier":"Vector","linked":false,"name":"Vector","type":"NodeSocketColor"},{"default":0.3700000047683716,"identifier":"Midlevel","linked":false,"name":"Midlevel","type":"NodeSocketFloat"},{"default":0.4099999964237213,"identifier":"Scale","linked":false,"name":"Scale","type":"NodeSocketFloat"}],"internal_links":[{"from_socket":"Midlevel","to_socket":"Displacement"}],"label":"","mute":false,"name":"SVM Vector Displacement Tangent Vector Displacement","node_tree":null,"outputs":[{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"}],"properties":{"space":"TANGENT"},"special":{},"type":"VECTOR_DISPLACEMENT"}],"surface_root":{"node":"SVM Vector Displacement Tangent Emission","socket":"Emission"},"volume_root":null},"surface_render_method":"DITHERED","use_bump_map_correction":true,"use_transparent_shadow":true,"volume_interpolation":"LINEAR","volume_sampling":"MULTIPLE_IMPORTANCE"},{"cycles_sync":{"pass_id":0,"shader_index":7},"displacement_method":"DISPLACEMENT","emission_sampling":"AUTO","name":"SVM Vector Displacement World","node_tree":{"displacement_root":{"node":"SVM Vector Displacement World Vector Displacement","socket":"Displacement"},"links":[{"from_node":"SVM Vector Displacement World Vector Displacement","from_socket":"Displacement","to_node":"Material Output","to_socket":"Displacement"},{"from_node":"SVM Vector Displacement World Emission","from_socket":"Emission","to_node":"Material Output","to_socket":"Surface"}],"name":"Shader Nodetree","nodes":[{"bl_idname":"ShaderNodeOutputMaterial","image":null,"inputs":[{"identifier":"Surface","linked":true,"name":"Surface","type":"NodeSocketShader"},{"identifier":"Volume","linked":false,"name":"Volume","type":"NodeSocketShader"},{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"},{"default":0.0,"identifier":"Thickness","linked":false,"name":"Thickness","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"Material Output","node_tree":null,"outputs":[],"properties":{"is_active_output":true,"target":"ALL"},"special":{},"type":"OUTPUT_MATERIAL"},{"bl_idname":"ShaderNodeEmission","image":null,"inputs":[{"default":[0.17000000178813934,0.4300000071525574,0.7900000214576721,1.0],"identifier":"Color","linked":false,"name":"Color","type":"NodeSocketColor"},{"default":1.0,"identifier":"Strength","linked":false,"name":"Strength","type":"NodeSocketFloat"},{"default":0.0,"identifier":"Weight","linked":false,"name":"Weight","type":"NodeSocketFloat"}],"internal_links":[],"label":"","mute":false,"name":"SVM Vector Displacement World Emission","node_tree":null,"outputs":[{"identifier":"Emission","linked":true,"name":"Emission","type":"NodeSocketShader"}],"properties":{},"special":{},"type":"EMISSION"},{"bl_idname":"ShaderNodeVectorDisplacement","image":null,"inputs":[{"default":[0.7300000190734863,0.28999999165534973,0.6100000143051147,1.0],"identifier":"Vector","linked":false,"name":"Vector","type":"NodeSocketColor"},{"default":0.3700000047683716,"identifier":"Midlevel","linked":false,"name":"Midlevel","type":"NodeSocketFloat"},{"default":0.4099999964237213,"identifier":"Scale","linked":false,"name":"Scale","type":"NodeSocketFloat"}],"internal_links":[{"from_socket":"Midlevel","to_socket":"Displacement"}],"label":"","mute":false,"name":"SVM Vector Displacement World Vector Displacement","node_tree":null,"outputs":[{"default":[0.0,0.0,0.0],"identifier":"Displacement","linked":true,"name":"Displacement","type":"NodeSocketVector"}],"properties":{"space":"WORLD"},"special":{},"type":"VECTOR_DISPLACEMENT"}],"surface_root":{"node":"SVM Vector Displacement World Emission","socket":"Emission"},"volume_root":null},"surface_render_method":"DITHERED","use_bump_map_correction":true,"use_transparent_shadow":true,"volume_interpolation":"LINEAR","volume_sampling":"MULTIPLE_IMPORTANCE"}],"render":{"color_management":{"display_device":"sRGB","exposure":0.0,"gamma":1.0,"look":"None","sequencer_color_space":"sRGB","shader_transforms":{"rec709_to_rgb":[[1.0001587867736816,-0.000001341104507446289,-7.450580596923828E-7],[-8.597271516919136E-8,0.9999764561653137,-3.427267074584961E-7],[1.30385160446167E-8,1.9371509552001953E-7,0.9997596144676208]],"xyz_to_rgb":[[3.2409698963165283,-1.5373847484588623,-0.4986114501953125],[-0.9692434668540955,1.8759669065475464,0.04155469685792923],[0.055629879236221313,-0.20397651195526123,1.0569710731506348]]},"use_curve_mapping":false,"view_transform":"AgX"},"cycles":{"adaptive_min_samples":0,"adaptive_threshold":0.009999999776482582,"ao_bounces":1,"ao_bounces_render":1,"ao_distance":10.0,"ao_factor":1.0,"blur_glossy":1.0,"caustics_reflective":true,"caustics_refractive":true,"diffuse_bounces":4,"direct_light_sampling_type":"MULTIPLE_IMPORTANCE_SAMPLING","effective_seed":20903,"fast_gi_method":"REPLACE","film_exposure":1.0,"glossy_bounces":4,"light_sampling_threshold":0.009999999776482582,"max_bounces":0,"min_light_bounces":0,"min_transparent_bounces":0,"sample_clamp_direct":0.0,"sample_clamp_indirect":10.0,"samples":256,"seed":20903,"seed_frame":1,"seed_subframe":0.0,"transmission_bounces":8,"transparent_max_bounces":8,"use_adaptive_sampling":false,"use_animated_seed":false,"use_denoising":false,"use_fast_gi":false,"use_light_tree":false,"volume_bounces":0},"filter_width":1.0,"height":64,"pass_alpha_threshold":0.5,"percentage":100,"pixel_filter_type":"BOX","transparent":false,"width":64},"camera":{"angle":0.6911112070083618,"angle_x":0.6911112070083618,"angle_y":0.4710899591445923,"clip_end":1000.0,"clip_start":0.10000000149011612,"dof":{"blades":0,"enabled":false,"focus_distance":10.0,"fstop":2.799999952316284,"ratio":1.0,"rotation":0.0},"lens":50.0,"name":"Probe Camera","ortho_scale":2.200000047683716,"sensor_fit":"AUTO","shift_x":0.0,"shift_y":0.0,"transform":[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,3.0,1.0],"type":"ORTHO"},"geometries":[],"curve_geometries":[],"instances":[],"lights":[],"world":null,"world_environment":null})JSON";

void write_scene_bundle(const std::filesystem::path &directory,
                        std::string_view payload = scene_json) {
  {
    std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
  }
  std::ofstream scene{directory / "scene.json"};
  scene << payload;
}

[[nodiscard]] std::string both_tangent_scene_json() {
  auto result = std::string{scene_json};
  constexpr auto exported_material =
      "\"displacement_method\":\"DISPLACEMENT\","
      "\"emission_sampling\":\"AUTO\","
      "\"name\":\"SVM Vector Displacement Tangent\"";
  constexpr auto both_material =
      "\"displacement_method\":\"BOTH\","
      "\"emission_sampling\":\"AUTO\","
      "\"name\":\"SVM Vector Displacement Tangent\"";
  const auto position = result.find(exported_material);
  require(position != std::string::npos,
          "Vector Displacement tangent fixture marker is missing");
  result.replace(position, std::char_traits<char>::length(exported_material),
                 both_material);
  return result;
}

[[nodiscard]] std::string nested_both_tangent_scene_json() {
  auto result = both_tangent_scene_json();
  const auto tangent_material = result.find(
      "\"name\":\"SVM Vector Displacement Tangent\",\"node_tree\"");
  require(tangent_material != std::string::npos,
          "nested Vector Displacement tangent material is missing");

  const auto replace_after = [&](std::string_view source,
                                 std::string_view replacement) {
    const auto position = result.find(source, tangent_material);
    require(position != std::string::npos,
            "nested Vector Displacement fixture marker is missing");
    result.replace(position, source.size(), replacement);
  };
  replace_after(
      "\"displacement_root\":{\"node\":\"SVM Vector Displacement Tangent "
      "Vector Displacement\",\"socket\":\"Displacement\"}",
      "\"displacement_root\":{\"node\":\"SVM Vector Displacement Tangent "
      "Post Scale\",\"socket\":\"Vector\"}");
  replace_after(
      "{\"from_node\":\"SVM Vector Displacement Tangent Vector "
      "Displacement\",\"from_socket\":\"Displacement\",\"to_node\":\"Material "
      "Output\",\"to_socket\":\"Displacement\"}",
      "{\"from_node\":\"SVM Vector Displacement Tangent Post "
      "Scale\",\"from_socket\":\"Vector\",\"to_node\":\"Material "
      "Output\",\"to_socket\":\"Displacement\"},{\"from_node\":\"SVM Vector "
      "Displacement Tangent Vector Displacement\",\"from_socket\":\"Displacement\","
      "\"to_node\":\"SVM Vector Displacement Tangent Post "
      "Scale\",\"to_socket\":\"Vector\"}");

  const auto vector_displacement = result.find(
      "{\"bl_idname\":\"ShaderNodeVectorDisplacement\"", tangent_material);
  require(vector_displacement != std::string::npos,
          "nested Vector Displacement source node is missing");
  constexpr std::string_view scale_node =
      "{\"bl_idname\":\"ShaderNodeVectorMath\",\"image\":null,\"inputs\":[{"
      "\"default\":[0.0,0.0,0.0],\"identifier\":\"Vector\",\"linked\":true,"
      "\"name\":\"Vector\",\"type\":\"NodeSocketVector\"},{\"default\":[0.0,0.0,"
      "0.0],\"identifier\":\"Vector_001\",\"linked\":false,\"name\":\"Vector\","
      "\"type\":\"NodeSocketVector\"},{\"default\":[0.0,0.0,0.0],\"identifier\":"
      "\"Vector_002\",\"linked\":false,\"name\":\"Vector\",\"type\":"
      "\"NodeSocketVector\"},{\"default\":0.7300000190734863,\"identifier\":"
      "\"Scale\",\"linked\":false,\"name\":\"Scale\",\"type\":"
      "\"NodeSocketFloat\"}],\"internal_links\":[{\"from_socket\":\"Vector\","
      "\"to_socket\":\"Vector\"}],\"label\":\"Nested displacement root: SCALE "
      "0.73\",\"mute\":false,\"name\":\"SVM Vector Displacement Tangent Post "
      "Scale\",\"node_tree\":null,\"outputs\":[{\"default\":[0.0,0.0,0.0],"
      "\"identifier\":\"Vector\",\"linked\":true,\"name\":\"Vector\",\"type\":"
      "\"NodeSocketVector\"},{\"default\":0.0,\"identifier\":\"Value\","
      "\"linked\":false,\"name\":\"Value\",\"type\":\"NodeSocketFloat\"}],"
      "\"properties\":{\"operation\":\"SCALE\"},\"special\":{},\"type\":"
      "\"VECT_MATH\"},";
  result.insert(vector_displacement, scale_node);
  return result;
}

void require_words(std::string_view name,
                   std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected) {
  if (actual.size() == expected.size()) {
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      if (actual[index] == expected[index]) {
        continue;
      }
      std::cerr << name << " differs at word " << index << ": got 0x"
                << std::hex << actual[index] << ", expected 0x"
                << expected[index] << std::dec << '\n';
      break;
    }
  } else {
    std::cerr << name << " has " << actual.size() << " words, expected "
              << expected.size() << '\n';
  }
  if (actual.size() != expected.size() ||
      !std::ranges::equal(actual, expected)) {
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << "  [" << std::dec << index << "] = 0x" << std::hex
                << actual[index] << '\n';
    }
    throw std::runtime_error{
        std::string{name} + " word stream differs from Cycles 5.2.1"};
  }
}

void test_vector_displacement_word_images() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  if (!imported.ok()) {
    for (const auto &diagnostic : imported.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(imported.ok(), "Vector Displacement scene did not import");

  const ShaderCompiler frontend{make_core_node_registry()};
  MaterialLibrary materials;
  const auto material_update = materials.update(*imported.scene, frontend);
  if (!material_update.committed) {
    for (const auto &diagnostic : material_update.diagnostics) {
      std::cerr << "Material " << diagnostic.material.value << ": "
                << diagnostic.message << '\n';
    }
  }
  require(material_update.committed,
          "Vector Displacement scene did not survive material compilation");

  struct Oracle {
    std::string_view name;
    std::uint32_t space;
    std::uint32_t attr;
    std::uint32_t attr_sign;
  };
  static constexpr std::array oracles{
      Oracle{"SVM Vector Displacement Tangent", 0u, 8u, 9u},
      Oracle{"SVM Vector Displacement Object", 1u, 0u, 0u},
      Oracle{"SVM Vector Displacement World", 2u, 0u, 0u},
  };

  for (const auto &oracle : oracles) {
    const MaterialDesc *material = nullptr;
    for (const auto &[id, candidate] : imported.scene->materials) {
      static_cast<void>(id);
      if (candidate.name == oracle.name) {
        material = &candidate;
        break;
      }
    }
    require(material != nullptr, "Vector Displacement material is missing");
    require(material->displacement_method ==
                DisplacementMethod::displacement,
            "true-displacement policy changed during import");

    const auto shader = frontend.compile(material->shader);
    if (!shader.ok()) {
      for (const auto &diagnostic : shader.diagnostics) {
        std::cerr << diagnostic.message << '\n';
      }
    }
    require(shader.ok(), "Vector Displacement graph did not normalize");

    AttributeIDMap attributes;
    ImageIDMap images;
    const auto image = compile_shader(
        *shader.program, attributes, images,
        ShaderCompileContext{
            .background = false,
            .displacement_method = material->displacement_method});
    require(image.valid, image.diagnostic);

    // Local extraction of shaders 5--7 from the Cycles 5.2.1 diagnostic
    // dump. Global jumps (97,104,105), (118,125,126), and (139,146,147)
    // all rebase to the same local jump (1,4,11,12).
    const std::array<std::uint32_t, 25u> expected{
        0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
        0x00000005u, 0x3e2e147bu, 0x3edc28f6u, 0x3f4a3d71u,
        0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
        0x0000001bu, oracle.space, 0x3f3ae148u, 0x3e947ae1u,
        0x3f1c28f6u, 0x3ebd70a4u, 0x3ed1eb85u, oracle.attr,
        oracle.attr_sign, 0x00000000u, 0x00000019u, 0x00000000u,
        0x00000000u};
    require_words(oracle.name, image.words, expected);
    require(image.node_types_used[NODE_VECTOR_DISPLACEMENT] &&
                image.node_types_used[NODE_SET_DISPLACEMENT] &&
                image.node_types_used[NODE_END],
            "Vector Displacement opcode usage differs from Cycles 5.2.1");

    if (oracle.space == 0u) {
      const auto requests = image.attribute_requests;
      require(requests.size() == 3u &&
                  requests[0u].standard == ATTR_STD_UV &&
                  requests[1u].standard ==
                      ATTR_STD_UV_TANGENT_UNDISPLACED &&
                  requests[2u].standard ==
                      ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED,
              "tangent Vector Displacement attribute requests differ");
    } else {
      require(image.attribute_requests.empty(),
              "non-tangent Vector Displacement requested attributes");
    }
  }
}

void test_both_vector_displacement_word_image() {
  TemporaryDirectory temporary;
  const auto json = both_tangent_scene_json();
  write_scene_bundle(temporary.path(), json);
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "BOTH Vector Displacement scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "SVM Vector Displacement Tangent") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr,
          "BOTH Vector Displacement material is missing");
  require(material->displacement_method == DisplacementMethod::both,
          "BOTH Vector Displacement policy changed during import");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "BOTH Vector Displacement graph did not normalize");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(
      *shader.program, attributes, images,
      ShaderCompileContext{
          .background = false,
          .displacement_method = DisplacementMethod::both});
  require(image.valid, image.diagnostic);

  // Exact local extraction of shader 5 from the Cycles 5.2.1 diagnostic
  // oracle /tmp/psycles-svm-vector-displacement-both-oracle-2.svm52.
  // Global jump (1,97,188,189) rebases to (1,4,95,96); words 97--201
  // are copied without normalization or synthesized expectations.
  static constexpr std::array<std::uint32_t, 109u> expected{
      0x00000001u, 0x00000004u, 0x0000005fu, 0x00000060u,
      0x00000023u, 0x00000000u, 0x0000000bu, 0x0a000001u,
      0x00000000u, 0x0000001bu, 0x00000000u, 0x3f3ae148u,
      0x3e947ae1u, 0x3f1c28f6u, 0x3ebd70a4u, 0x3ed1eb85u,
      0x00000008u, 0x00000009u, 0x0000000du, 0x0000002du,
      0x00000007u, 0x7fc0000du, 0x00000000u, 0x00000000u,
      0x7fc0000au, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x0000ff10u,
      0x0000001bu, 0x00000000u, 0x3f3ae148u, 0x3e947ae1u,
      0x3f1c28f6u, 0x3ebd70a4u, 0x3ed1eb85u, 0x00000008u,
      0x00000009u, 0x0000000du, 0x0000002du, 0x00000007u,
      0x7fc0000du, 0x00000000u, 0x00000000u, 0x7fc0000au,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x0000ff11u, 0x0000001bu,
      0x00000000u, 0x3f3ae148u, 0x3e947ae1u, 0x3f1c28f6u,
      0x3ebd70a4u, 0x3ed1eb85u, 0x00000008u, 0x00000009u,
      0x0000000du, 0x0000002du, 0x00000007u, 0x7fc0000du,
      0x00000000u, 0x00000000u, 0x7fc0000au, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x0000ff12u, 0x00000021u, 0x3f800000u,
      0x3f800000u, 0x3dcccccdu, 0x1001000au, 0x000d1211u,
      0x00000022u, 0x00000a0du, 0x00000024u, 0x00000000u,
      0x00000005u, 0x3e2e147bu, 0x3edc28f6u, 0x3f4a3d71u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x0000001bu, 0x00000000u, 0x3f3ae148u, 0x3e947ae1u,
      0x3f1c28f6u, 0x3ebd70a4u, 0x3ed1eb85u, 0x00000008u,
      0x00000009u, 0x00000000u, 0x00000019u, 0x00000000u,
      0x00000000u};
  require_words("SVM Vector Displacement Tangent BOTH", image.words,
                expected);
}

void test_nested_both_vector_displacement_word_image() {
  TemporaryDirectory temporary;
  const auto json = nested_both_tangent_scene_json();
  write_scene_bundle(temporary.path(), json);
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(),
          "nested BOTH Vector Displacement scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "SVM Vector Displacement Tangent") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr,
          "nested BOTH Vector Displacement material is missing");
  require(material->displacement_method == DisplacementMethod::both,
          "nested BOTH Vector Displacement policy changed during import");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(),
          "nested BOTH Vector Displacement graph did not normalize");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(
      *shader.program, attributes, images,
      ShaderCompileContext{.background = false,
                           .displacement_method = DisplacementMethod::both});
  require(image.valid, image.diagnostic);

  // Exact local extraction of shader 5 from the Cycles 5.2.1 diagnostic
  // oracle /tmp/psycles-svm-vector-displacement-nested-both.svm52.
  // Global jump (1,97,227,228) rebases to (1,4,134,135); words 97--253
  // are copied without normalization or synthesized expectations.
  static constexpr std::array<std::uint32_t, 161u> expected{
      0x00000001u, 0x00000004u, 0x00000086u, 0x00000087u, 0x00000023u, 0x00000000u,
      0x0000001bu, 0x00000000u, 0x3f3ae148u, 0x3e947ae1u, 0x3f1c28f6u, 0x3ebd70a4u,
      0x3ed1eb85u, 0x00000008u, 0x00000009u, 0x0000000au, 0x0000002du, 0x0000000au,
      0x7fc0000au, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x3f3ae148u, 0x00000dffu, 0x0000000bu,
      0x0a000001u, 0x00000000u, 0x0000002du, 0x00000007u, 0x7fc0000du, 0x00000000u,
      0x00000000u, 0x7fc0000au, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x0000ff10u, 0x0000001bu, 0x00000000u, 0x3f3ae148u,
      0x3e947ae1u, 0x3f1c28f6u, 0x3ebd70a4u, 0x3ed1eb85u, 0x00000008u, 0x00000009u,
      0x0000000du, 0x0000002du, 0x0000000au, 0x7fc0000du, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x3f3ae148u, 0x000011ffu, 0x0000002du, 0x00000007u, 0x7fc00011u, 0x00000000u,
      0x00000000u, 0x7fc0000au, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x0000ff0du, 0x0000001bu, 0x00000000u, 0x3f3ae148u,
      0x3e947ae1u, 0x3f1c28f6u, 0x3ebd70a4u, 0x3ed1eb85u, 0x00000008u, 0x00000009u,
      0x00000011u, 0x0000002du, 0x0000000au, 0x7fc00011u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x3f3ae148u, 0x000014ffu, 0x0000002du, 0x00000007u, 0x7fc00014u, 0x00000000u,
      0x00000000u, 0x7fc0000au, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x0000ff0eu, 0x00000021u, 0x3f800000u, 0x3f800000u,
      0x3dcccccdu, 0x1001000au, 0x00110e0du, 0x00000022u, 0x00000a11u, 0x00000024u,
      0x00000000u, 0x00000005u, 0x3e2e147bu, 0x3edc28f6u, 0x3f4a3d71u, 0x00000003u,
      0x000000ffu, 0x00000000u, 0x00000000u, 0x0000001bu, 0x00000000u, 0x3f3ae148u,
      0x3e947ae1u, 0x3f1c28f6u, 0x3ebd70a4u, 0x3ed1eb85u, 0x00000008u, 0x00000009u,
      0x00000000u, 0x0000002du, 0x0000000au, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x3f3ae148u, 0x000003ffu, 0x00000019u, 0x00000003u, 0x00000000u};
  require_words("SVM Vector Displacement nested BOTH", image.words,
                expected);
}

} // namespace

int main() {
  try {
    test_vector_displacement_word_images();
    test_both_vector_displacement_word_image();
    test_nested_both_vector_displacement_word_image();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "Cycles SVM vector-displacement compiler test passed\n";
  return EXIT_SUCCESS;
}

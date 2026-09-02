#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler::cycles_svm;
using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

// Complete global svm_nodes image dumped after
// SVMShaderManager::device_update_specific by Blender Cycles 5.2.1 commit
// 9e2066aef7ef7e20c142ad7bd3303138a4304c93. The six shader ids are
// default_surface, default_volume, default_light, default_background,
// default_empty, and Diffuse Probe. This is deliberately the final global
// stream rather than a Psycles-generated expected value.
constexpr std::array<std::uint32_t, 113u> cycles_5_2_1_global_oracle{
    0x00000001u, 0x00000018u, 0x0000004bu, 0x0000004cu, 0x00000001u,
    0x0000004du, 0x0000004eu, 0x0000004fu, 0x00000001u, 0x00000050u,
    0x00000051u, 0x00000052u, 0x00000001u, 0x00000053u, 0x0000005au,
    0x0000005bu, 0x00000001u, 0x0000005cu, 0x0000005du, 0x0000005eu,
    0x00000001u, 0x0000005fu, 0x0000006fu, 0x00000070u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000002u, 0x0000002bu, 0x000000ffu,
    0x0000001au, 0x3fc00000u, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x3f4ccccdu, 0x3f4ccccdu,
    0x3f4ccccdu, 0x3f800000u, 0x00000000u, 0x0000ff00u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x00000000u, 0x00000000u,
    0x3f800000u, 0x3f800000u, 0x3f800000u, 0x00000000u, 0x3f800000u,
    0x3f800000u, 0x3f800000u, 0x3f000000u, 0x3f800000u, 0x3f800000u,
    0x3f800000u, 0x3cf5c28fu, 0x3fc00000u, 0x00000020u, 0x3f800000u,
    0x3e4ccccdu, 0x3dcccccdu, 0x3ba3d70au, 0x3fb33333u, 0x00000000u,
    0x00000000u, 0x3faa3d71u, 0x00000000u, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000005u, 0x3eac0831u,
    0x3ed4fdf3u, 0x3f051eb8u, 0x00000004u, 0x000000ffu, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x0000000bu, 0x00000001u, 0x00000000u, 0x00000005u, 0x3f2e147bu,
    0x3e75c28fu, 0x3db851ecu, 0x00000002u, 0x00000002u, 0x000000ffu,
    0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu, 0x3edc28f6u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u};

std::vector<ShaderImage> split_external_global_oracle() {
  constexpr auto shader_count = 6u;
  constexpr auto jump_words = 4u;
  std::vector<ShaderImage> shaders(shader_count);
  for (auto shader = std::size_t{}; shader < shaders.size(); ++shader) {
    const auto jump_base = shader * jump_words;
    const auto tail_begin =
        static_cast<std::size_t>(cycles_5_2_1_global_oracle[jump_base + 1u]);
    const auto tail_end =
        shader + 1u < shaders.size()
            ? static_cast<std::size_t>(
                  cycles_5_2_1_global_oracle[(shader + 1u) * jump_words + 1u])
            : cycles_5_2_1_global_oracle.size();
    require(tail_begin >= shader_count * jump_words && tail_begin < tail_end &&
                tail_end <= cycles_5_2_1_global_oracle.size(),
            "external Cycles shader tail partition is invalid");

    auto &local = shaders[shader];
    local.valid = true;
    local.node_types_used[NODE_SHADER_JUMP] = true;
    local.words.resize(jump_words + tail_end - tail_begin);
    local.words[0u] = NODE_SHADER_JUMP;
    for (auto entry = std::size_t{1u}; entry < jump_words; ++entry) {
      const auto global = static_cast<std::size_t>(
          cycles_5_2_1_global_oracle[jump_base + entry]);
      require(global >= tail_begin && global < tail_end,
              "external Cycles jump entry escapes its shader tail");
      local.words[entry] =
          static_cast<std::uint32_t>(global - tail_begin + jump_words);
    }
    std::copy(cycles_5_2_1_global_oracle.begin() + tail_begin,
              cycles_5_2_1_global_oracle.begin() + tail_end,
              local.words.begin() + jump_words);
  }
  return shaders;
}

void test_global_link_matches_cycles_5_2_1() {
  auto local = split_external_global_oracle();
  local[0u].peak_stack_usage = 17u;
  local[5u].peak_stack_usage = 3u;
  local[5u].node_types_used[NODE_CLOSURE_BSDF] = true;

  const auto linked = link_shader_table(local);
  require(linked.valid, linked.diagnostic);
  require(linked.shader_count == local.size(),
          "linked shader count differs from Cycles");
  require(linked.words.size() == cycles_5_2_1_global_oracle.size(),
          "linked word count differs from Cycles");
  require(std::equal(linked.words.begin(), linked.words.end(),
                     cycles_5_2_1_global_oracle.begin()),
          "linked global word stream differs from Cycles 5.2.1");
  require(linked.peak_stack_usage == 17u,
          "linked peak stack usage is not the shader maximum");
  require(linked.node_types_used[NODE_SHADER_JUMP] &&
              linked.node_types_used[NODE_CLOSURE_BSDF],
          "linked node-use domain is not the shader union");
}

void test_malformed_local_images_are_rejected() {
  auto local = split_external_global_oracle();

  auto missing_tail = local.front();
  missing_tail.words.resize(4u);
  require(!link_shader_table(std::span{&missing_tail, 1u}).valid,
          "linker accepted a local shader with no tail");

  auto wrong_opcode = local.front();
  wrong_opcode.words[0u] = NODE_END;
  require(!link_shader_table(std::span{&wrong_opcode, 1u}).valid,
          "linker accepted a non-ShaderJump prefix");

  auto negative_entry = local.front();
  negative_entry.words[1u] = 0xffffffffu;
  require(!link_shader_table(std::span{&negative_entry, 1u}).valid,
          "linker accepted a negative ShaderJump entry");

  auto escaping_entry = local.front();
  escaping_entry.words[2u] =
      static_cast<std::uint32_t>(escaping_entry.words.size());
  require(!link_shader_table(std::span{&escaping_entry, 1u}).valid,
          "linker accepted a ShaderJump entry outside the tail");

  auto missing_usage = local.front();
  missing_usage.node_types_used[NODE_SHADER_JUMP] = false;
  require(!link_shader_table(std::span{&missing_usage, 1u}).valid,
          "linker accepted incomplete node-use metadata");

  auto stack_overflow = local.front();
  stack_overflow.peak_stack_usage = SVM_STACK_SIZE + 1u;
  require(!link_shader_table(std::span{&stack_overflow, 1u}).valid,
          "linker accepted a shader exceeding the Cycles stack");
}

std::shared_ptr<const ShaderProgram> compile_diffuse(float roughness) {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Diffuse");
  require(
      graph.set_input(diffuse, "Color", SocketValue::color({0.3f, 0.5f, 0.7f})),
      "failed to set scene-test diffuse color");
  require(
      graph.set_input(diffuse, "Roughness", SocketValue::floating(roughness)),
      "failed to set scene-test diffuse roughness");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});
  const ShaderCompiler compiler{make_core_node_registry()};
  auto compiled = compiler.compile(graph);
  require(compiled.ok(), "scene-test diffuse graph did not compile");
  return std::move(compiled.program);
}

std::shared_ptr<const ShaderProgram> compile_particle_index() {
  ShaderGraph graph;
  const auto particle =
      graph.add_node(node_type::particle_info, "Particle Info");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({particle, "Index"}, emission, "Strength"),
          "failed to keep Particle Info live");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const ShaderCompiler compiler{make_core_node_registry()};
  auto compiled = compiler.compile(graph);
  require(compiled.ok(), "scene-test Particle Info graph did not compile");
  return std::move(compiled.program);
}

std::shared_ptr<const ShaderProgram>
compile_graph(ShaderGraph graph, std::string_view diagnostic) {
  const ShaderCompiler compiler{make_core_node_registry()};
  auto compiled = compiler.compile(graph);
  require(compiled.ok(), diagnostic);
  return std::move(compiled.program);
}

[[nodiscard]] std::uint32_t shader_flags(const KernelShader &shader) {
  return std::bit_cast<std::uint32_t>(shader.flags);
}

void require_shader_flags(const KernelShader &shader, std::uint32_t expected,
                          std::string_view message) {
  const auto actual = shader_flags(shader);
  if (actual != expected) {
    std::cerr << message << ": actual=0x" << std::hex << actual
              << ", expected=0x" << expected << std::dec << '\n';
    std::exit(1);
  }
}

void test_kernel_shader_image_matches_cycles_metadata() {
  ShaderGraph graph;
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(emission, "Color",
                          SocketValue::color({0.25f, 0.5f, 1.0f})) &&
              graph.set_input(emission, "Strength",
                              SocketValue::floating(4.0f)),
          "failed to configure KernelShader emission probe");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const auto shader =
      compile_graph(std::move(graph), "KernelShader emission graph failed");
  const std::array units{ShaderTableCompileUnit{
      .shader_index = 2u,
      .shader = shader.get(),
      .kernel = {.name = "Cube",
                 .use_transparent_shadow = false,
                 .use_bump_map_correction = false,
                 .emission_sampling = EmissionSampling::front,
                 .volume_sampling = VolumeSampling::distance,
                 .volume_interpolation = VolumeInterpolation::linear,
                 .pass_id = 17}}};
  const auto compiled = compile_shader_table(units);
  require(compiled.table.valid, compiled.table.diagnostic);
  require(compiled.kernel_shaders.size() == 3u,
          "KernelShader image does not share the dense shader domain");

  constexpr std::array<std::uint32_t, 8u> zero_words{};
  require(std::bit_cast<std::array<std::uint32_t, 8u>>(
              compiled.kernel_shaders[0u]) == zero_words &&
              std::bit_cast<std::array<std::uint32_t, 8u>>(
                  compiled.kernel_shaders[1u]) == zero_words,
          "unrepresented KernelShader holes are not byte-zero");

  const auto &kernel = compiled.kernel_shaders[2u];
  require(kernel.constant_emission.x == 1.0f &&
              kernel.constant_emission.y == 2.0f &&
              kernel.constant_emission.z == 4.0f,
          "constant emission estimate differs from Cycles");
  require(std::bit_cast<std::uint32_t>(kernel.cryptomatte_id) ==
              0xa8fce865u,
          "KernelShader name hash differs from the Cycles Cube oracle");
  constexpr auto expected_flags =
      static_cast<std::uint32_t>(SD_MIS_FRONT | SD_HAS_CONSTANT_EMISSION |
                                 SD_HAS_EMISSION);
  require_shader_flags(kernel, expected_flags,
                       "constant-emission KernelShader flags differ from Cycles");
  require(kernel.pass_id == 17 && kernel.pad2 == 0 && kernel.pad3 == 0,
          "constant-emission KernelShader scalar fields differ from Cycles");
}

void test_kernel_shader_transparency_and_duplicate_contract() {
  ShaderGraph graph;
  const auto transparent =
      graph.add_node(node_type::transparent_bsdf, "Transparent");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = transparent, .socket = "Closure"});
  const auto shader =
      compile_graph(std::move(graph), "transparent metadata graph failed");

  const std::array enabled{ShaderTableCompileUnit{
      .shader_index = 0u,
      .shader = shader.get(),
      .kernel = {.name = "Transparent",
                 .use_transparent_shadow = true,
                 .use_bump_map_correction = false,
                 .volume_sampling = VolumeSampling::distance}}};
  const auto compiled = compile_shader_table(enabled);
  require(compiled.table.valid, compiled.table.diagnostic);
  require_shader_flags(
      compiled.kernel_shaders[0u],
      static_cast<std::uint32_t>(SD_HAS_TRANSPARENT_SHADOW |
                                 SD_HAS_CONSTANT_EMISSION),
      "transparent-shadow policy was not projected into KernelShader");

  const std::array disabled{ShaderTableCompileUnit{
      .shader_index = 0u,
      .shader = shader.get(),
      .kernel = {.name = "Transparent",
                 .use_transparent_shadow = false,
                 .use_bump_map_correction = false,
                 .volume_sampling = VolumeSampling::distance}}};
  const auto no_shadow = compile_shader_table(disabled);
  require(no_shadow.table.valid &&
              shader_flags(no_shadow.kernel_shaders[0u]) ==
                  static_cast<std::uint32_t>(SD_HAS_CONSTANT_EMISSION),
          "disabled transparent shadows retained the scheduling flag");

  const std::array ambiguous{
      ShaderTableCompileUnit{.shader_index = 0u,
                             .shader = shader.get(),
                             .kernel = {.name = "Transparent",
                                        .use_transparent_shadow = true}},
      ShaderTableCompileUnit{.shader_index = 0u,
                             .shader = shader.get(),
                             .kernel = {.name = "Transparent",
                                        .use_transparent_shadow = false}}};
  require(!compile_shader_table(ambiguous).table.valid,
          "duplicate bytecode with distinct KernelShader metadata was merged");
}

void test_kernel_shader_volume_and_light_path_flags() {
  ShaderGraph volume_graph;
  const auto attribute =
      volume_graph.add_node(node_type::attribute, "Density");
  const auto volume =
      volume_graph.add_node(node_type::volume_absorption, "Volume");
  require(volume_graph.set_property(attribute, "Attribute",
                                    SocketValue::string("density")) &&
              volume_graph.set_property(
                  attribute, "AttributeId",
                  SocketValue::unsigned_integer(attribute_id("density"))) &&
              volume_graph.connect({attribute, "Fac"}, volume, "Density"),
          "failed to configure KernelShader volume probe");
  volume_graph.set_root(ShaderDomain::volume,
                        OutputRef{.node = volume, .socket = "Volume"});
  const auto volume_shader = compile_graph(
      std::move(volume_graph), "volume metadata graph failed");
  const std::array volume_units{ShaderTableCompileUnit{
      .shader_index = 0u,
      .shader = volume_shader.get(),
      .kernel = {.name = "Volume",
                 .use_bump_map_correction = false,
                 .volume_sampling = VolumeSampling::equiangular,
                 .volume_interpolation = VolumeInterpolation::cubic}}};
  const auto volume_compiled = compile_shader_table(volume_units);
  require(volume_compiled.table.valid, volume_compiled.table.diagnostic);
  constexpr auto expected_volume_flags = static_cast<std::uint32_t>(
      SD_HAS_TRANSPARENT_SHADOW | SD_HAS_VOLUME | SD_HAS_ONLY_VOLUME |
      SD_HETEROGENEOUS_VOLUME | SD_NEED_VOLUME_ATTRIBUTES |
      SD_VOLUME_EQUIANGULAR | SD_VOLUME_CUBIC | SD_HAS_CONSTANT_EMISSION);
  require_shader_flags(volume_compiled.kernel_shaders[0u],
                       expected_volume_flags,
                       "volume scheduling flags differ from Cycles");

  ShaderGraph path_graph;
  const auto path = path_graph.add_node(node_type::light_path, "Light Path");
  const auto path_emission =
      path_graph.add_node(node_type::emission, "Emission");
  require(path_graph.connect({path, "RayDepth"}, path_emission, "Strength"),
          "failed to configure KernelShader Light Path probe");
  path_graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = path_emission, .socket = "Closure"});
  const auto path_shader =
      compile_graph(std::move(path_graph), "Light Path metadata graph failed");
  const std::array path_units{ShaderTableCompileUnit{
      .shader_index = 0u,
      .shader = path_shader.get(),
      .kernel = {.name = "Light Path",
                 .use_bump_map_correction = false,
                 .volume_sampling = VolumeSampling::distance}}};
  const auto path_compiled = compile_shader_table(path_units);
  require(path_compiled.table.valid, path_compiled.table.diagnostic);
  constexpr auto expected_path_flags = static_cast<std::uint32_t>(
      SD_MIS_FRONT | SD_MIS_BACK | SD_HAS_LIGHT_PATH_NODE | SD_HAS_EMISSION);
  require_shader_flags(path_compiled.kernel_shaders[0u], expected_path_flags,
                       "Light Path/emission scheduling flags differ from Cycles");
}

void test_kernel_shader_bump_flag_implications() {
  ShaderGraph surface_graph;
  const auto wireframe =
      surface_graph.add_node(node_type::wireframe, "Bump Height");
  const auto bump = surface_graph.add_node(node_type::bump, "Bump");
  const auto bssrdf = surface_graph.add_node(
      node_type::subsurface_scattering, "Bumped BSSRDF");
  require(surface_graph.set_property(wireframe, "Use Pixel Size",
                                     SocketValue::boolean(false)) &&
              surface_graph.set_property(bump, "Invert",
                                         SocketValue::boolean(false)) &&
              surface_graph.set_property(bump, "UseObjectSpace",
                                         SocketValue::boolean(false)) &&
              surface_graph.connect({wireframe, "Fac"}, bump, "Height") &&
              surface_graph.connect({bump, "Normal"}, bssrdf, "Normal"),
          "failed to configure surface-bump metadata probe");
  surface_graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = bssrdf, .socket = "Closure"});
  const auto surface_shader = compile_graph(
      std::move(surface_graph), "surface-bump metadata graph failed");
  const std::array surface_units{ShaderTableCompileUnit{
      .shader_index = 0u,
      .shader = surface_shader.get(),
      .kernel = {.name = "Surface Bump",
                 .use_bump_map_correction = false,
                 .volume_sampling = VolumeSampling::distance}}};
  const auto surface_compiled = compile_shader_table(surface_units);
  require(surface_compiled.table.valid, surface_compiled.table.diagnostic);
  require_shader_flags(
      surface_compiled.kernel_shaders[0u],
      static_cast<std::uint32_t>(SD_HAS_BUMP_FROM_SURFACE |
                                 SD_HAS_BSSRDF_BUMP |
                                 SD_HAS_CONSTANT_EMISSION),
      "surface BSSRDF bump flags differ from Cycles");

  ShaderGraph displacement_graph;
  const auto surface =
      displacement_graph.add_node(node_type::diffuse_bsdf, "Surface");
  const auto displacement_height =
      displacement_graph.add_node(node_type::wireframe, "Displacement Height");
  const auto displacement_bump =
      displacement_graph.add_node(node_type::bump, "Displacement Bump");
  const auto geometry =
      displacement_graph.add_node(node_type::geometry, "Geometry");
  const auto displacement = displacement_graph.add_node(
      node_type::normal_to_vector, "True Displacement");
  require(displacement_graph.set_property(
              displacement_height, "Use Pixel Size",
              SocketValue::boolean(false)) &&
              displacement_graph.set_property(
                  displacement_bump, "Invert", SocketValue::boolean(false)) &&
              displacement_graph.set_property(
                  displacement_bump, "UseObjectSpace",
                  SocketValue::boolean(true)) &&
              displacement_graph.connect({displacement_height, "Fac"},
                                         displacement_bump, "Height") &&
              displacement_graph.connect({geometry, "Normal"}, displacement,
                                         "Normal"),
          "failed to configure displacement-bump metadata probe");
  displacement_graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = surface, .socket = "Closure"});
  displacement_graph.set_root(
      ShaderDomain::surface_normal,
      OutputRef{.node = displacement_bump, .socket = "Normal"});
  displacement_graph.set_root(
      ShaderDomain::displacement,
      OutputRef{.node = displacement, .socket = "Vector"});
  const auto displacement_shader = compile_graph(
      std::move(displacement_graph),
      "displacement-bump metadata graph failed");
  const std::array displacement_units{ShaderTableCompileUnit{
      .shader_index = 0u,
      .shader = displacement_shader.get(),
      .context = {.displacement_method = DisplacementMethod::both},
      .kernel = {.name = "Displacement Bump",
                 .use_bump_map_correction = false,
                 .volume_sampling = VolumeSampling::distance}}};
  const auto displacement_compiled =
      compile_shader_table(displacement_units);
  require(displacement_compiled.table.valid,
          displacement_compiled.table.diagnostic);
  require_shader_flags(
      displacement_compiled.kernel_shaders[0u],
      static_cast<std::uint32_t>(SD_HAS_BUMP_FROM_DISPLACEMENT |
                                 SD_HAS_BSSRDF_BUMP |
                                 SD_HAS_DISPLACEMENT |
                                 SD_HAS_CONSTANT_EMISSION),
      "automatic/true displacement flags differ from Cycles");
}

void test_scene_compile_transaction_preserves_shader_identity() {
  const auto diffuse = compile_diffuse(0.0f);
  const auto particle = compile_particle_index();
  const std::array units{
      ShaderTableCompileUnit{.shader_index = 2u, .shader = diffuse.get()},
      ShaderTableCompileUnit{.shader_index = 5u, .shader = particle.get()},
      ShaderTableCompileUnit{.shader_index = 5u, .shader = particle.get()}};
  const auto compiled = compile_shader_table(units);
  require(compiled.table.valid, compiled.table.diagnostic);
  require(compiled.table.shader_count == 6u,
          "scene compiler did not preserve the maximum Cycles shader id");
  require(compiled.table.node_types_used[NODE_CLOSURE_BSDF],
          "scene compiler lost a live shader opcode");
  require(compiled.table.node_types_used[NODE_PARTICLE_INFO] &&
              compiled.shader_node_types_used.size() == 6u &&
              !compiled.shader_node_types_used[2u][NODE_PARTICLE_INFO] &&
              compiled.shader_node_types_used[5u][NODE_PARTICLE_INFO] &&
              !compiled.shader_node_types_used[0u][NODE_PARTICLE_INFO],
          "per-shader Particle Info demand collapsed into the global union");
  require(compiled.shader_attribute_ids_used.size() == 6u &&
              compiled.shader_attribute_ids_used[2u].empty() &&
              std::ranges::binary_search(
                  compiled.shader_attribute_ids_used[5u],
                  static_cast<std::uint64_t>(ATTR_STD_PARTICLE)) &&
              compiled.shader_attribute_ids_used[0u].empty(),
          "per-shader Cycles attribute requests were not preserved");
  require(compiled.named_attributes.empty() && compiled.images.empty() &&
              compiled.ies.empty(),
          "resource-free scene unexpectedly allocated SVM resources");

  for (const auto hole : {0u, 1u, 3u, 4u}) {
    const auto jump = static_cast<std::size_t>(hole) * 4u;
    const auto surface = compiled.table.words[jump + 1u];
    const auto volume = compiled.table.words[jump + 2u];
    const auto displacement = compiled.table.words[jump + 3u];
    require(volume == surface + 1u && displacement == surface + 2u,
            "inert shader does not contain three adjacent routines");
    require(compiled.table.words[surface] == NODE_END &&
                compiled.table.words[volume] == NODE_END &&
                compiled.table.words[displacement] == NODE_END,
            "inert shader routine is not an END node");
  }

  const auto distinct = compile_diffuse(0.47f);
  const std::array collision{
      ShaderTableCompileUnit{.shader_index = 7u, .shader = diffuse.get()},
      ShaderTableCompileUnit{.shader_index = 7u, .shader = distinct.get()}};
  require(!compile_shader_table(collision).table.valid,
          "scene compiler merged distinct shaders with the same source id");

  const std::array missing{
      ShaderTableCompileUnit{.shader_index = 0u, .shader = nullptr}};
  require(!compile_shader_table(missing).table.valid,
          "scene compiler accepted an absent ShaderProgram");
}

} // namespace

int main() {
  test_global_link_matches_cycles_5_2_1();
  test_malformed_local_images_are_rejected();
  test_kernel_shader_image_matches_cycles_metadata();
  test_kernel_shader_transparency_and_duplicate_contract();
  test_kernel_shader_volume_and_light_path_flags();
  test_kernel_shader_bump_flag_implications();
  test_scene_compile_transaction_preserves_shader_identity();
  std::cout << "Cycles SVM scene linker tests passed\n";
  return 0;
}

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include <algorithm>
#include <array>
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

void test_scene_compile_transaction_preserves_shader_identity() {
  const auto diffuse = compile_diffuse(0.0f);
  const std::array units{
      ShaderTableCompileUnit{.shader_index = 2u, .shader = diffuse.get()},
      ShaderTableCompileUnit{.shader_index = 5u, .shader = diffuse.get()},
      ShaderTableCompileUnit{.shader_index = 5u, .shader = diffuse.get()}};
  const auto compiled = compile_shader_table(units);
  require(compiled.table.valid, compiled.table.diagnostic);
  require(compiled.table.shader_count == 6u,
          "scene compiler did not preserve the maximum Cycles shader id");
  require(compiled.table.node_types_used[NODE_CLOSURE_BSDF],
          "scene compiler lost a live shader opcode");
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
  test_scene_compile_transaction_preserves_shader_identity();
  std::cout << "Cycles SVM scene linker tests passed\n";
  return 0;
}

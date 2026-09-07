#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace psycles::contract;
using namespace psycles::compiler;
namespace svm = psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

std::shared_ptr<const ShaderProgram> compile(const ShaderGraph &graph) {
  auto result = ShaderCompiler{make_core_node_registry()}.compile(graph);
  for (const auto &diagnostic : result.diagnostics) {
    require(result.ok(), diagnostic.message);
  }
  require(result.ok(), "invalid closure-budget source graph");
  return std::move(result.program);
}

svm::CompiledShaderTable table(const ShaderGraph &graph) {
  const auto shader = compile(graph);
  const std::array units{svm::ShaderTableCompileUnit{
      .shader_index = 3u, .shader = shader.get()}};
  auto result = svm::compile_shader_table(units);
  require(result.table.valid, result.table.diagnostic);
  for (auto index = 0u; index < 3u; ++index) {
    require(result.shader_metadata[index].num_closures == 0u,
            "inert shader holes allocate closures");
  }
  return result;
}

const auto &oracle() {
  static const auto values = [] {
    std::ifstream input{PSYCLES_CLOSURE_BUDGET_ORACLE};
    require(bool(input), "missing original Cycles closure-budget observations");
    std::map<std::string, std::array<std::uint32_t, 2u>, std::less<>> rows;
    for (std::string line; std::getline(input, line);) {
      if (line.empty() || line.front() == '#') { continue; }
      std::istringstream fields{line};
      std::string name, extra;
      std::array<std::uint32_t, 2u> row{};
      require(bool(fields >> name >> row[0u] >> row[1u]) && !(fields >> extra),
              "invalid original Cycles closure-budget observation");
      require(rows.emplace(name, row).second, "duplicate Cycles observation");
    }
    require(input.eof() && rows.size() == 20u, "incomplete Cycles observations");
    return rows;
  }();
  return values;
}

void check(const ShaderGraph &graph, std::string_view label) {
  const auto expected = oracle().find(label);
  require(expected != oracle().end(), "missing Cycles observation for fixture");
  const auto [count, budget] = expected->second;
  const auto result = table(graph);
  const auto actual = result.shader_metadata[3u].num_closures;
  require(actual == count,
          std::string{label} + ": graph count " + std::to_string(actual) +
              ", Cycles count " + std::to_string(count));
  require(result.max_closures == budget,
          std::string{label} + ": scene budget " +
              std::to_string(result.max_closures) + ", Cycles budget " +
              std::to_string(budget));
}

ShaderGraph leaf(std::string_view type, ShaderDomain domain = ShaderDomain::surface) {
  ShaderGraph graph;
  const auto node = graph.add_node(std::string{type});
  graph.set_root(domain, OutputRef{node, domain == ShaderDomain::volume
                                           ? "Volume" : "Closure"});
  return graph;
}

void test_node_virtuals() {
  // Oracle: Cycles 5.2.1 scene/shader_graph.cpp::get_num_closures,
  // scene/shader_nodes.h::get_closure_type, and kernel/types.h. In particular
  // Emission and Background inherit CLOSURE_NONE_ID; an emission opcode is
  // not evidence of a ShaderClosure allocation. No renderer evaluates these
  // fixtures on the CPU: this test observes only the host graph metadata.
  // ShaderManager::add_default permanently references a Principled graph,
  // so the full scene budget is at least 12 even for an emission-only scene.
  check(leaf(node_type::diffuse_bsdf), "diffuse");
  check(leaf(node_type::emission), "emission");
  check(leaf(node_type::background), "background");
  check(leaf(node_type::principled_bsdf), "principled");
  check(leaf(node_type::subsurface_scattering), "bssrdf");
  check(leaf(node_type::metallic_bsdf), "conductor");
  check(leaf(node_type::hair_bsdf), "hair");
  check(leaf(node_type::volume_absorption, ShaderDomain::volume),
        "absorption");
  check(leaf(node_type::volume_scatter, ShaderDomain::volume),
        "scatter");

  for (const auto distribution : {"GGX", "MULTI_GGX", "BECKMANN", "ASHIKHMIN_SHIRLEY"}) {
    auto graph = leaf(node_type::glossy_bsdf);
    require(graph.set_property(graph.nodes().front().id, "Distribution",
                               SocketValue::string(distribution)),
            "cannot set Glossy distribution");
    check(graph, std::string{"glossy-"} + distribution);
  }
  for (const auto distribution : {"GGX", "MULTI_GGX", "BECKMANN"}) {
    auto graph = leaf(node_type::glass_bsdf);
    require(graph.set_property(graph.nodes().front().id, "Distribution",
                               SocketValue::string(distribution)),
            "cannot set Glass distribution");
    check(graph, std::string{"glass-"} + distribution);
  }
}

void test_finalized_graph_and_scene_domain() {
  auto graph = leaf(node_type::diffuse_bsdf);
  const auto diffuse = graph.nodes().front().id;
  const auto emission = graph.add_node(node_type::emission);
  const auto add = graph.add_node(node_type::add_closure);
  require(graph.connect({diffuse, "Closure"}, add, "A") &&
              graph.connect({emission, "Closure"}, add, "B"),
          "cannot connect mixed emission fixture");
  graph.set_root(ShaderDomain::surface, OutputRef{add, "Closure"});
  check(graph, "diffuse-emission");

  const auto principled = graph.add_node(node_type::principled_bsdf);
  const auto mix = graph.add_node(node_type::mix_closure);
  require(graph.connect({add, "Closure"}, mix, "A") &&
              graph.connect({principled, "Closure"}, mix, "B") &&
              graph.set_input(mix, "Factor", SocketValue::floating(0.0f)),
          "cannot connect folded Mix fixture");
  graph.set_root(ShaderDomain::surface, OutputRef{mix, "Closure"});
  check(graph, "folded-principled");

  // A shared graph node is counted once, even when the closure-control
  // traversal reaches it more than once. This cannot be recovered by counting
  // NODE_CLOSURE_BSDF instructions in the resulting word stream.
  auto shared = leaf(node_type::diffuse_bsdf);
  const auto shared_diffuse = shared.nodes().front().id;
  const auto twice = shared.add_node(node_type::add_closure);
  require(shared.connect({shared_diffuse, "Closure"}, twice, "A") &&
              shared.connect({shared_diffuse, "Closure"}, twice, "B"),
          "cannot connect shared closure fixture");
  shared.set_root(ShaderDomain::surface, OutputRef{twice, "Closure"});
  check(shared, "shared-diffuse");

  auto volume = leaf(node_type::volume_absorption, ShaderDomain::volume);
  auto tail = OutputRef{volume.nodes().front().id, "Volume"};
  for (auto index = 1u; index < 3u; ++index) {
    const auto scatter = volume.add_node(node_type::volume_scatter);
    // Different colors keep genuine distinct nodes across graph deduplication.
    require(volume.set_input(scatter, "Color", SocketValue::color(
                {static_cast<float>(index) * 0.25f, 0.5f, 0.75f})),
            "cannot configure phase color");
    const auto sum = volume.add_node(node_type::add_volume);
    require(volume.connect(tail, sum, "A") &&
                volume.connect({scatter, "Volume"}, sum, "B"),
            "cannot connect multiple-volume fixture");
    tail = {sum, "Volume"};
  }
  volume.set_root(ShaderDomain::volume, tail);
  check(volume, "volume-cap");

  const auto diffuse_shader = compile(graph);
  const auto volume_shader = compile(volume);
  const std::array units{
      svm::ShaderTableCompileUnit{.shader_index = 0u,
                                  .shader = diffuse_shader.get()},
      svm::ShaderTableCompileUnit{.shader_index = 8u,
                                  .shader = volume_shader.get()},
      svm::ShaderTableCompileUnit{.shader_index = 8u,
                                  .shader = volume_shader.get()}};
  const auto result = svm::compile_shader_table(units);
  require(result.table.valid, result.table.diagnostic);
  require(result.max_closures == 64u &&
              result.shader_metadata[0u].num_closures == 1u &&
              result.shader_metadata[8u].num_closures == 96u,
          "scene maximum does not preserve the dense used-shader domain");
  require(svm::compile_shader_table({}).max_closures == 12u,
          "empty scene lost Cycles' always-referenced default shader budget");
}
} // namespace

int main() {
  try {
    test_node_virtuals();
    test_finalized_graph_and_scene_domain();
    std::cout << "Cycles SVM closure budget tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

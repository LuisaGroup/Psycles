#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;
namespace svm = psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void test_graph_features_are_independent_of_evaluation_domain() {
  // Cycles 5.2.1 ShaderManager::get_graph_kernel_features() in
  // scene/shader.cpp queries every graph node. This is deliberately distinct
  // from SVMCompiler's surface-only accumulation of KernelShader flags.
  // A surface closure on a live Volume branch is skipped during evaluation,
  // but its graph still requests the same feature as a Surface connection.
  // Confirmed with Cycles 5.2.1 9e2066aef7ef, HIP, 16x16/1 sample:
  // Transparent + Emission -> Volume: Use Transparent True, Subsurface False.
  // BSSRDF + Emission -> Volume: Use Transparent False, Subsurface True.
  // Both report Use Volume True. No surface output is connected.
  for (const auto &[type, feature] : {
           std::pair{node_type::transparent_bsdf, svm::kernel_feature_transparent},
           std::pair{node_type::subsurface_scattering, svm::kernel_feature_subsurface}}) {
    for (const auto domain : {ShaderDomain::surface, ShaderDomain::volume}) {
      ShaderGraph graph;
      const auto closure = graph.add_node(type, "Graph feature probe");
      auto root = OutputRef{.node = closure, .socket = "Closure"};
      if (domain == ShaderDomain::volume) {
        // optimize_volume_output removes a purely surface-only Volume tree.
        // Emission has volume support and keeps this mixed graph live in
        // Cycles, without making the surface closure run as a surface shader.
        const auto emission = graph.add_node(node_type::emission, "Volume emission");
        const auto add = graph.add_node(node_type::add_closure, "Mixed volume branch");
        require(graph.connect(root, add, "A") &&
                    graph.connect({emission, "Closure"}, add, "B"),
                "failed to keep the mixed Volume graph live");
        root = OutputRef{.node = add, .socket = "Closure"};
      }
      graph.set_root(domain, root);
      const ShaderCompiler compiler{make_core_node_registry()};
      const auto compiled = compiler.compile(graph);
      require(compiled.ok(), "feature probe graph failed to compile");
      const std::array units{svm::ShaderTableCompileUnit{
          .shader_index = 0u, .shader = compiled.program.get()}};
      const auto table = svm::compile_shader_table(units);
      require(table.table.valid, table.table.diagnostic);
      const auto expected = svm::kernel_feature_node_bsdf |
                            svm::kernel_feature_node_emission | feature |
                            (domain == ShaderDomain::volume
                                 ? svm::kernel_feature_volume : 0u);
      if (table.kernel_features != expected) {
        std::cerr << "graph kernel features incorrectly depend on evaluation "
                     "domain; type=" << type << ", volume="
                  << (domain == ShaderDomain::volume) << ", actual=0x"
                  << std::hex << table.kernel_features << ", expected=0x"
                  << expected << '\n';
        std::exit(EXIT_FAILURE);
      }
    }
  }
}

} // namespace

int main() {
  test_graph_features_are_independent_of_evaluation_domain();
  std::cout << "Cycles SVM kernel feature tests passed\n";
}

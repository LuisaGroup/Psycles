#include "cycles_svm_imported_word_fixture.h"
#include "cycles_svm_graph.h"

namespace {
void check_conversion_identity() {
  using namespace psycles::compiler::cycles_svm;
  const auto make = [](GraphSocketType target) {
    auto node = make_graph_node(cycles_synthetic_float3_autoconvert);
    node->type = cycles_synthetic_float3_autoconvert;
    node->inputs.push_back({.name = "value_float", .type = GraphSocketType::floating});
    node->outputs.push_back({.name = "value", .type = target});
    return node;
  };
  auto point = make(GraphSocketType::point);
  auto color = make(GraphSocketType::color);
  auto same_point = make(GraphSocketType::point);
  psycles::test_support::require(point->equals(*same_point) && !point->equals(*color),
      "ConvertNode identity must include both Cycles socket types");
}
}

int main() {
  try {
    check_conversion_identity();
    psycles::test_support::check_imported_words("cycles_socket_types", 27);
    std::cout << "27 typed texture-input graphs match Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

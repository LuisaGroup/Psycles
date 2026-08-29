#include <psycles/sampling/light_tree.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] psycles::sampling::LightTreeEmitter
emitter(std::uint32_t identity, psycles::Vec3f position, float energy,
        bool distant = false) {
  using namespace psycles::sampling;
  return {.measure = {.bounds = {.minimum = position,
                                 .maximum = position,
                                 .empty = distant},
                      .orientation =
                          {.axis = {0.0f, 0.0f, 1.0f},
                           .theta_o = distant ? 0.0f : 3.14159265358979323846f,
                           .theta_e = distant ? 0.1f : 1.57079632679489661923f,
                           .empty = false},
                      .energy = energy},
          .centroid = position,
          .emitter_id = identity,
          .distant = distant};
}

} // namespace

int main() {
  using namespace psycles::sampling;

  const LightTreeBounds a{.minimum = {-1.0f, 0.0f, 2.0f},
                          .maximum = {0.0f, 1.0f, 3.0f},
                          .empty = false};
  const LightTreeBounds b{.minimum = {2.0f, -2.0f, 1.0f},
                          .maximum = {4.0f, 0.5f, 5.0f},
                          .empty = false};
  const auto bounds = merge_light_tree_bounds(a, b);
  require(bounds.minimum == psycles::Vec3f{-1.0f, -2.0f, 1.0f} &&
              bounds.maximum == psycles::Vec3f{4.0f, 1.0f, 5.0f},
          "bounding union mismatch");

  const LightTreeOrientationBounds cone_x{.axis = {1.0f, 0.0f, 0.0f},
                                          .theta_o = 0.0f,
                                          .theta_e = 0.25f,
                                          .empty = false};
  const LightTreeOrientationBounds cone_y{.axis = {0.0f, 1.0f, 0.0f},
                                          .theta_o = 0.0f,
                                          .theta_e = 0.5f,
                                          .empty = false};
  const auto cone = merge_light_tree_orientation_bounds(cone_x, cone_y);
  require(std::abs(cone.theta_o - 0.78539816339f) < 1.0e-6f,
          "orientation half-angle mismatch");
  require(std::abs(cone.axis.x - 0.70710677f) < 1.0e-6f &&
              std::abs(cone.axis.y - 0.70710677f) < 1.0e-6f,
          "orientation axis mismatch");
  require(cone.theta_e == 0.5f, "emission cone was not enclosed");

  std::vector<LightTreeEmitter> inputs;
  for (std::uint32_t index = 0u; index < 10u; ++index) {
    inputs.emplace_back(emitter(index,
                                {static_cast<float>(index),
                                 static_cast<float>((index * 7u) % 3u), 0.0f},
                                static_cast<float>(index + 1u)));
  }
  inputs.emplace_back(emitter(10u, {0.0f, 0.0f, 1.0f}, 4.0f, true));
  inputs.emplace_back(emitter(11u, {0.0f, 0.0f, -1.0f}, 2.0f, true));

  const auto tree = build_cycles_light_tree(inputs, 3u);
  require(tree.usable(), "non-empty light tree is unusable");
  require(tree.root == 0u, "root index is not stable");
  const auto &root = tree.nodes[tree.root];
  require(root.kind == LightTreeNodeKind::inner, "root is not inner");
  require(tree.nodes[root.right].kind == LightTreeNodeKind::distant,
          "distant emitters are not isolated at the root");
  require(tree.nodes[root.right].emitter_count == 2u,
          "distant cluster population mismatch");
  require(std::abs(root.measure.energy - 61.0f) < 1.0e-6f,
          "root energy mismatch");

  std::vector<bool> seen(inputs.size(), false);
  for (std::size_t tree_index = 0u; tree_index < tree.emitters.size();
       ++tree_index) {
    const auto identity = tree.emitters[tree_index].emitter_id;
    require(identity < seen.size() && !seen[identity],
            "emitter permutation is not bijective");
    seen[identity] = true;
    require(tree.emitter_to_tree[identity] == tree_index,
            "reverse emitter index mismatch");
    const auto leaf = tree.emitter_to_leaf[identity];
    require(leaf < tree.nodes.size(), "reverse leaf index is invalid");
    const auto &node = tree.nodes[leaf];
    require(node.kind == LightTreeNodeKind::distant || node.emitter_count <= 3u,
            "local leaf exceeds configured capacity");
    require(tree_index >= node.first_emitter &&
                tree_index < node.first_emitter + node.emitter_count,
            "reverse leaf does not contain emitter");
  }

  const auto zero =
      build_cycles_light_tree(std::vector{emitter(0u, {}, 0.0f)}, 8u);
  require(!zero.usable(), "zero-energy tree unexpectedly became usable");

  bool rejected_sparse_identity = false;
  try {
    static_cast<void>(
        build_cycles_light_tree(std::vector{emitter(7u, {}, 1.0f)}, 8u));
  } catch (const std::invalid_argument &) {
    rejected_sparse_identity = true;
  }
  require(rejected_sparse_identity, "sparse emitter identity was accepted");

  std::vector<LightTreeEmitter> subtree_inputs;
  for (std::uint32_t index = 0u; index < 5u; ++index) {
    subtree_inputs.emplace_back(
        emitter(index, {static_cast<float>(index), 0.0f, 0.0f}, 1.0f));
  }
  const auto subtree = build_cycles_light_subtree(subtree_inputs, 2u);
  require(subtree.valid() && subtree.usable(),
          "mesh-local subtree is invalid");
  require(subtree.nodes[subtree.root].parent == invalid_light_tree_index,
          "mesh-local subtree root unexpectedly has a parent");
  require(std::none_of(subtree.nodes.begin(), subtree.nodes.end(),
                       [](const LightTreeNode &node) noexcept {
                         return node.kind == LightTreeNodeKind::distant;
                       }),
          "mesh-local subtree contains a top-level distant fork");

  bool rejected_distant_subtree = false;
  try {
    static_cast<void>(build_cycles_light_subtree(
        std::vector{emitter(0u, {}, 1.0f, true)}, 8u));
  } catch (const std::invalid_argument &) {
    rejected_distant_subtree = true;
  }
  require(rejected_distant_subtree,
          "mesh-local subtree accepted a distant emitter");

  std::cout << "Cycles light-tree construction tests passed.\n";
  return EXIT_SUCCESS;
}

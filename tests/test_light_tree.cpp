#include <psycles/sampling/light_tree.h>

#include <algorithm>
#include <array>
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

[[nodiscard]] psycles::sampling::LightTreeEmitter oriented_emitter(
    std::uint32_t identity, psycles::Vec3f position, psycles::Vec3f axis,
    float theta_o, float theta_e, float energy = 1.0f) {
  using namespace psycles::sampling;
  return {.measure = {.bounds = {.minimum = position,
                                 .maximum = position,
                                 .empty = false},
                      .orientation = {.axis = axis,
                                      .theta_o = theta_o,
                                      .theta_e = theta_e,
                                      .empty = false},
                      .energy = energy},
          .centroid = position,
          .emitter_id = identity,
          .distant = false};
}

[[nodiscard]] bool close(float a, float b, float tolerance = 1.0e-6f) {
  return std::abs(a - b) <= tolerance;
}

[[nodiscard]] bool close(psycles::Vec3f a, psycles::Vec3f b,
                         float tolerance = 1.0e-6f) {
  return close(a.x, b.x, tolerance) && close(a.y, b.y, tolerance) &&
         close(a.z, b.z, tolerance);
}

[[nodiscard]] bool close(
    const psycles::sampling::LightTreeMeasure &a,
    const psycles::sampling::LightTreeMeasure &b,
    float tolerance = 1.0e-6f) {
  return a.bounds.empty == b.bounds.empty &&
         (a.bounds.empty ||
          (close(a.bounds.minimum, b.bounds.minimum, tolerance) &&
           close(a.bounds.maximum, b.bounds.maximum, tolerance))) &&
         a.orientation.empty == b.orientation.empty &&
         (a.orientation.empty ||
          (close(a.orientation.axis, b.orientation.axis, tolerance) &&
           close(a.orientation.theta_o, b.orientation.theta_o, tolerance) &&
           close(a.orientation.theta_e, b.orientation.theta_e, tolerance))) &&
         close(a.energy, b.energy, tolerance);
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

  // Captured directly from Cycles' make_orthonormals branch for opposed
  // orientation bounds. This branch chooses the orthogonal axis itself; it
  // does not rotate the wide axis by theta_o - wide.theta_o.
  const LightTreeOrientationBounds opposed_x{
      .axis = {-1.0f, 0.0f, 0.0f},
      .theta_o = 0.0f,
      .theta_e = 0.125f,
      .empty = false};
  const auto opposed =
      merge_light_tree_orientation_bounds(cone_x, opposed_x);
  require(close(opposed.axis,
                {0.0f, 0.707106769f, -0.707106769f}) &&
              close(opposed.theta_o, 1.570796371f) &&
              close(opposed.theta_e, 0.25f),
          "opposed orientation merge diverges from Cycles");

  // Cycles defines every recursive node measure through the dimension-zero
  // bucket reduction, not through the incidental input order. The relation is
  // observable because orientation-cone merge is intentionally non-associative.
  std::array<LightTreeEmitter, 4u> bucket_inputs{
      oriented_emitter(0u, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 0.05f,
                       0.2f),
      oriented_emitter(1u, {30.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.15f,
                       0.3f),
      oriented_emitter(2u, {20.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.25f,
                       0.4f),
      oriented_emitter(3u, {10.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, 0.35f,
                       0.5f)};
  LightTreeMeasure bucket_order_measure;
  for (const auto index : {0u, 3u, 2u, 1u}) {
    bucket_order_measure = merge_light_tree_measures(
        bucket_order_measure, bucket_inputs[index].measure);
  }
  LightTreeMeasure input_order_measure;
  for (const auto &input : bucket_inputs) {
    input_order_measure =
        merge_light_tree_measures(input_order_measure, input.measure);
  }
  require(!close(bucket_order_measure, input_order_measure, 1.0e-4f),
          "bucket-order regression fixture is accidentally associative");
  const auto bucket_tree = build_cycles_light_subtree(bucket_inputs, 8u);
  require(close(bucket_tree.nodes[bucket_tree.root].measure,
                bucket_order_measure),
          "node measure does not follow Cycles' fixed bucket reduction");

  // Emitter IDs are a reverse mapping label, not a spatial construction key.
  // Relabeling the same population by any dense permutation must leave every
  // node and measure unchanged, including equal-centroid partitions.
  constexpr std::array<psycles::Vec3f, 12u> rename_axes{
      psycles::Vec3f{1.0f, 0.0f, 0.0f},
      psycles::Vec3f{0.0f, 1.0f, 0.0f},
      psycles::Vec3f{0.0f, 0.0f, 1.0f},
      psycles::Vec3f{-1.0f, 0.0f, 0.0f},
      psycles::Vec3f{0.0f, -1.0f, 0.0f},
      psycles::Vec3f{0.0f, 0.0f, -1.0f},
      psycles::Vec3f{1.0f, 1.0f, 0.0f},
      psycles::Vec3f{-1.0f, 1.0f, 0.0f},
      psycles::Vec3f{1.0f, 0.0f, 1.0f},
      psycles::Vec3f{-1.0f, 0.0f, 1.0f},
      psycles::Vec3f{0.0f, 1.0f, 1.0f},
      psycles::Vec3f{0.0f, -1.0f, 1.0f}};
  std::vector<LightTreeEmitter> rename_inputs;
  rename_inputs.reserve(rename_axes.size());
  for (std::uint32_t index = 0u; index < rename_axes.size(); ++index) {
    rename_inputs.emplace_back(oriented_emitter(
        index,
        {index < 8u ? 0.0f : 100.0f, 0.0f, 0.0f},
        rename_axes[index],
        0.025f * static_cast<float>(index % 5u),
        0.1f + 0.02f * static_cast<float>(index % 4u),
        1.0f + static_cast<float>(index)));
  }
  auto renamed_inputs = rename_inputs;
  for (std::uint32_t index = 0u; index < renamed_inputs.size(); ++index) {
    renamed_inputs[index].emitter_id =
        static_cast<std::uint32_t>(renamed_inputs.size() - 1u - index);
  }
  const auto rename_tree = build_cycles_light_subtree(rename_inputs, 3u);
  const auto renamed_tree = build_cycles_light_subtree(renamed_inputs, 3u);
  require(rename_tree.nodes.size() == renamed_tree.nodes.size(),
          "emitter relabeling changed light-tree topology");
  for (std::size_t index = 0u; index < rename_tree.nodes.size(); ++index) {
    const auto &a_node = rename_tree.nodes[index];
    const auto &b_node = renamed_tree.nodes[index];
    require(a_node.kind == b_node.kind && a_node.parent == b_node.parent &&
                a_node.left == b_node.left && a_node.right == b_node.right &&
                a_node.first_emitter == b_node.first_emitter &&
                a_node.emitter_count == b_node.emitter_count &&
                close(a_node.measure, b_node.measure),
            "emitter identity leaked into spatial construction");
  }

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

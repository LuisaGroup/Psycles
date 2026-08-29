#pragma once

#include <psycles/core/math.h>

#include <cstdint>
#include <span>
#include <vector>

namespace psycles::sampling {

inline constexpr std::uint32_t invalid_light_tree_index = ~std::uint32_t{0u};

struct LightTreeBounds {
  Vec3f minimum{};
  Vec3f maximum{};
  bool empty{true};
};

struct LightTreeOrientationBounds {
  Vec3f axis{};
  float theta_o{};
  float theta_e{};
  bool empty{true};
};

struct LightTreeMeasure {
  LightTreeBounds bounds;
  LightTreeOrientationBounds orientation;
  float energy{};
};

struct LightTreeEmitter {
  LightTreeMeasure measure;
  Vec3f centroid{};
  // Index in the renderer's complete emitter population. This identity is
  // stable even though construction reorders emitters spatially.
  std::uint32_t emitter_id{};
  bool distant{};
};

enum class LightTreeNodeKind : std::uint32_t { inner, leaf, distant };

struct LightTreeNode {
  LightTreeMeasure measure;
  std::uint32_t parent{invalid_light_tree_index};
  std::uint32_t left{invalid_light_tree_index};
  std::uint32_t right{invalid_light_tree_index};
  std::uint32_t first_emitter{};
  std::uint32_t emitter_count{};
  LightTreeNodeKind kind{LightTreeNodeKind::leaf};
};

struct CyclesLightTree {
  std::vector<LightTreeEmitter> emitters;
  std::vector<LightTreeNode> nodes;
  // Indexed by emitter_id. The two mappings make selection and reverse-PDF
  // reconstruction use exactly the same reordered population.
  std::vector<std::uint32_t> emitter_to_tree;
  std::vector<std::uint32_t> emitter_to_leaf;
  std::uint32_t root{invalid_light_tree_index};

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool usable() const noexcept;
};

[[nodiscard]] LightTreeBounds
merge_light_tree_bounds(const LightTreeBounds &a,
                        const LightTreeBounds &b) noexcept;

[[nodiscard]] LightTreeOrientationBounds merge_light_tree_orientation_bounds(
    const LightTreeOrientationBounds &a,
    const LightTreeOrientationBounds &b) noexcept;

[[nodiscard]] LightTreeMeasure
merge_light_tree_measures(const LightTreeMeasure &a,
                          const LightTreeMeasure &b) noexcept;

[[nodiscard]] float
light_tree_measure_cost(const LightTreeMeasure &measure) noexcept;

// Builds the spatial hierarchy used by Cycles' many-light proposal. Local
// emitters use the 12-bucket surface-area-orientation heuristic; distant
// emitters occupy the dedicated root branch and are sampled as one cluster.
[[nodiscard]] CyclesLightTree
build_cycles_light_tree(std::span<const LightTreeEmitter> emitters,
                        std::uint32_t max_emitters_per_leaf = 8u);

// Builds one mesh-local subtree. Unlike the top-level hierarchy this has no
// synthetic local/distant fork: every input must be local, and root is the
// actual recursive-build root used after a mesh-instance transition.
[[nodiscard]] CyclesLightTree
build_cycles_light_subtree(std::span<const LightTreeEmitter> emitters,
                           std::uint32_t max_emitters_per_leaf = 8u);

} // namespace psycles::sampling

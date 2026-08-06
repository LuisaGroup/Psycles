#include <psycles/sampling/light_tree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace psycles::sampling {
namespace {

constexpr auto pi = 3.14159265358979323846f;
constexpr auto half_pi = 0.5f * pi;
constexpr auto two_pi = 2.0f * pi;
constexpr std::size_t bucket_count = 12u;

[[nodiscard]] Vec3f add(Vec3f a, Vec3f b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3f subtract(Vec3f a, Vec3f b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3f multiply(Vec3f value, float scale) noexcept {
  return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] float dot_product(Vec3f a, Vec3f b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] float length(Vec3f value) noexcept {
  return std::sqrt(std::max(dot_product(value, value), 0.0f));
}

[[nodiscard]] Vec3f normalize_or(Vec3f value, Vec3f fallback) noexcept {
  const auto magnitude = length(value);
  return magnitude > 1.0e-20f ? multiply(value, 1.0f / magnitude) : fallback;
}

[[nodiscard]] Vec3f component_min(Vec3f a, Vec3f b) noexcept {
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

[[nodiscard]] Vec3f component_max(Vec3f a, Vec3f b) noexcept {
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

[[nodiscard]] float component(Vec3f value, std::size_t axis) noexcept {
  return axis == 0u ? value.x : axis == 1u ? value.y : value.z;
}

[[nodiscard]] bool finite(Vec3f value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] LightTreeMeasure
valid_measure(const LightTreeMeasure &measure) noexcept {
  if (!std::isfinite(measure.energy) || measure.energy <= 0.0f) {
    return {};
  }
  auto result = measure;
  if (!result.bounds.empty &&
      (!finite(result.bounds.minimum) || !finite(result.bounds.maximum))) {
    result.bounds = {};
  }
  if (!result.orientation.empty) {
    result.orientation.axis =
        normalize_or(result.orientation.axis, {0.0f, 0.0f, 1.0f});
    result.orientation.theta_o =
        std::clamp(result.orientation.theta_o, 0.0f, pi);
    result.orientation.theta_e =
        std::clamp(result.orientation.theta_e, 0.0f, pi);
  }
  return result;
}

struct Bucket {
  LightTreeMeasure measure;
  std::uint32_t count{};
};

[[nodiscard]] Bucket merge_buckets(const Bucket &a, const Bucket &b) noexcept {
  return {.measure = merge_light_tree_measures(a.measure, b.measure),
          .count = a.count + b.count};
}

class Builder {

private:
  CyclesLightTree _tree;
  std::uint32_t _maximum_leaf_size;

  [[nodiscard]] std::uint32_t append_node(std::uint32_t parent) {
    if (_tree.nodes.size() >=
        static_cast<std::size_t>(invalid_light_tree_index)) {
      throw std::length_error("light tree exceeds 32-bit node addressing");
    }
    const auto index = static_cast<std::uint32_t>(_tree.nodes.size());
    LightTreeNode node;
    node.parent = parent;
    _tree.nodes.emplace_back(node);
    return index;
  }

  [[nodiscard]] std::uint32_t
  build_local(std::uint32_t begin, std::uint32_t end, std::uint32_t parent) {
    const auto node_index = append_node(parent);
    auto &node = _tree.nodes[node_index];
    const auto count = end - begin;
    if (count == 0u) {
      node.kind = LightTreeNodeKind::leaf;
      return node_index;
    }

    LightTreeBounds centroid_bounds;
    for (auto index = begin; index < end; ++index) {
      const auto &emitter = _tree.emitters[index];
      node.measure = merge_light_tree_measures(node.measure, emitter.measure);
      const LightTreeBounds point{.minimum = emitter.centroid,
                                  .maximum = emitter.centroid,
                                  .empty = false};
      centroid_bounds = merge_light_tree_bounds(centroid_bounds, point);
    }

    auto split_axis = std::numeric_limits<std::size_t>::max();
    auto split_middle = begin + count / 2u;
    auto best_cost = std::numeric_limits<float>::infinity();
    const auto parent_cost = light_tree_measure_cost(node.measure);
    const auto extent =
        subtract(centroid_bounds.maximum, centroid_bounds.minimum);
    const auto maximum_extent = std::max({extent.x, extent.y, extent.z});

    if (count > 1u && parent_cost > 0.0f && maximum_extent > 0.0f) {
      for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const auto axis_extent = component(extent, axis);
        if (!(axis_extent > 0.0f)) {
          continue;
        }
        std::array<Bucket, bucket_count> buckets{};
        for (auto index = begin; index < end; ++index) {
          const auto relative =
              (component(_tree.emitters[index].centroid, axis) -
               component(centroid_bounds.minimum, axis)) /
              axis_extent;
          const auto bucket = std::min<std::size_t>(
              static_cast<std::size_t>(std::max(relative, 0.0f) *
                                       static_cast<float>(bucket_count)),
              bucket_count - 1u);
          buckets[bucket].measure = merge_light_tree_measures(
              buckets[bucket].measure, _tree.emitters[index].measure);
          ++buckets[bucket].count;
        }

        std::array<Bucket, bucket_count - 1u> left{};
        std::array<Bucket, bucket_count - 1u> right{};
        left.front() = buckets.front();
        for (std::size_t bucket = 1u; bucket + 1u < bucket_count; ++bucket) {
          left[bucket] = merge_buckets(left[bucket - 1u], buckets[bucket]);
        }
        right.back() = buckets.back();
        for (std::size_t bucket = bucket_count - 2u; bucket-- > 0u;) {
          right[bucket] =
              merge_buckets(right[bucket + 1u], buckets[bucket + 1u]);
        }

        const auto regularization = maximum_extent / axis_extent;
        for (std::size_t split = 0u; split + 1u < bucket_count; ++split) {
          if (left[split].count == 0u || right[split].count == 0u) {
            continue;
          }
          const auto cost =
              regularization * (light_tree_measure_cost(left[split].measure) +
                                light_tree_measure_cost(right[split].measure));
          if (cost < parent_cost && cost < best_cost) {
            best_cost = cost;
            split_axis = axis;
            split_middle = begin + left[split].count;
          }
        }
      }
    }

    const auto should_split =
        count > _maximum_leaf_size ||
        split_axis != std::numeric_limits<std::size_t>::max();
    if (!should_split) {
      node.kind = LightTreeNodeKind::leaf;
      node.first_emitter = begin;
      node.emitter_count = count;
      for (auto index = begin; index < end; ++index) {
        _tree.emitter_to_tree[_tree.emitters[index].emitter_id] = index;
        _tree.emitter_to_leaf[_tree.emitters[index].emitter_id] = node_index;
      }
      return node_index;
    }

    if (split_axis != std::numeric_limits<std::size_t>::max()) {
      std::nth_element(
          _tree.emitters.begin() + begin, _tree.emitters.begin() + split_middle,
          _tree.emitters.begin() + end,
          [split_axis](const LightTreeEmitter &a,
                       const LightTreeEmitter &b) noexcept {
            const auto av = component(a.centroid, split_axis);
            const auto bv = component(b.centroid, split_axis);
            return av < bv || (av == bv && a.emitter_id < b.emitter_id);
          });
    }
    _tree.nodes[node_index].kind = LightTreeNodeKind::inner;
    const auto left = build_local(begin, split_middle, node_index);
    const auto right = build_local(split_middle, end, node_index);
    _tree.nodes[node_index].left = left;
    _tree.nodes[node_index].right = right;
    return node_index;
  }

public:
  Builder(std::span<const LightTreeEmitter> emitters,
          std::uint32_t maximum_leaf_size)
      : _maximum_leaf_size{std::max(maximum_leaf_size, 1u)} {
    if (emitters.size() >= static_cast<std::size_t>(invalid_light_tree_index)) {
      throw std::length_error("light tree exceeds 32-bit emitter addressing");
    }
    _tree.emitters.reserve(emitters.size());
    for (const auto &source : emitters) {
      auto emitter = source;
      emitter.measure = valid_measure(source.measure);
      if (!finite(emitter.centroid)) {
        emitter.centroid = {};
      }
      _tree.emitters.emplace_back(emitter);
    }
  }

  [[nodiscard]] CyclesLightTree build() {
    if (_tree.emitters.empty()) {
      return std::move(_tree);
    }
    std::vector<bool> identities(_tree.emitters.size(), false);
    for (const auto &emitter : _tree.emitters) {
      if (emitter.emitter_id >= identities.size() ||
          identities[emitter.emitter_id]) {
        throw std::invalid_argument(
            "light-tree emitter identities must be a dense permutation");
      }
      identities[emitter.emitter_id] = true;
    }
    _tree.emitter_to_tree.assign(_tree.emitters.size(),
                                 invalid_light_tree_index);
    _tree.emitter_to_leaf.assign(_tree.emitters.size(),
                                 invalid_light_tree_index);

    const auto distant_begin = static_cast<std::uint32_t>(
        std::stable_partition(_tree.emitters.begin(), _tree.emitters.end(),
                              [](const LightTreeEmitter &emitter) noexcept {
                                return !emitter.distant;
                              }) -
        _tree.emitters.begin());
    const auto emitter_count =
        static_cast<std::uint32_t>(_tree.emitters.size());

    _tree.root = append_node(invalid_light_tree_index);
    _tree.nodes[_tree.root].kind = LightTreeNodeKind::inner;
    const auto local = build_local(0u, distant_begin, _tree.root);
    const auto distant = append_node(_tree.root);
    auto &distant_node = _tree.nodes[distant];
    distant_node.kind = LightTreeNodeKind::distant;
    distant_node.first_emitter = distant_begin;
    distant_node.emitter_count = emitter_count - distant_begin;
    for (auto index = distant_begin; index < emitter_count; ++index) {
      distant_node.measure = merge_light_tree_measures(
          distant_node.measure, _tree.emitters[index].measure);
      _tree.emitter_to_tree[_tree.emitters[index].emitter_id] = index;
      _tree.emitter_to_leaf[_tree.emitters[index].emitter_id] = distant;
    }
    _tree.nodes[_tree.root].left = local;
    _tree.nodes[_tree.root].right = distant;
    _tree.nodes[_tree.root].measure = merge_light_tree_measures(
        _tree.nodes[local].measure, distant_node.measure);
    return std::move(_tree);
  }
};

} // namespace

bool CyclesLightTree::usable() const noexcept {
  return root < nodes.size() && !emitters.empty() &&
         emitter_to_tree.size() == emitters.size() &&
         emitter_to_leaf.size() == emitters.size() &&
         nodes[root].measure.energy > 0.0f;
}

LightTreeBounds merge_light_tree_bounds(const LightTreeBounds &a,
                                        const LightTreeBounds &b) noexcept {
  if (a.empty) {
    return b;
  }
  if (b.empty) {
    return a;
  }
  return {.minimum = component_min(a.minimum, b.minimum),
          .maximum = component_max(a.maximum, b.maximum),
          .empty = false};
}

LightTreeOrientationBounds merge_light_tree_orientation_bounds(
    const LightTreeOrientationBounds &a,
    const LightTreeOrientationBounds &b) noexcept {
  if (a.empty) {
    return b;
  }
  if (b.empty) {
    return a;
  }
  const auto *wide = &a;
  const auto *narrow = &b;
  if (b.theta_o > a.theta_o) {
    std::swap(wide, narrow);
  }
  const auto cosine =
      std::clamp(dot_product(wide->axis, narrow->axis), -1.0f, 1.0f);
  const auto axis_angle = std::acos(cosine);
  const auto emission_angle = std::max(a.theta_e, b.theta_e);
  if (wide->theta_o + 5.0e-4f >= std::min(pi, axis_angle + narrow->theta_o)) {
    return {.axis = wide->axis,
            .theta_o = wide->theta_o,
            .theta_e = emission_angle,
            .empty = false};
  }
  const auto theta_o = 0.5f * (axis_angle + wide->theta_o + narrow->theta_o);
  if (theta_o >= pi) {
    return {.axis = wide->axis,
            .theta_o = pi,
            .theta_e = emission_angle,
            .empty = false};
  }
  const auto rotation = theta_o - wide->theta_o;
  auto orthogonal = subtract(narrow->axis, multiply(wide->axis, cosine));
  if (cosine < -0.9995f) {
    const auto candidate = std::abs(wide->axis.x) < 0.9f
                               ? Vec3f{1.0f, 0.0f, 0.0f}
                               : Vec3f{0.0f, 1.0f, 0.0f};
    orthogonal =
        normalize_or({wide->axis.y * candidate.z - wide->axis.z * candidate.y,
                      wide->axis.z * candidate.x - wide->axis.x * candidate.z,
                      wide->axis.x * candidate.y - wide->axis.y * candidate.x},
                     {0.0f, 0.0f, 1.0f});
  } else {
    orthogonal = normalize_or(orthogonal, {0.0f, 0.0f, 1.0f});
  }
  return {.axis = normalize_or(add(multiply(wide->axis, std::cos(rotation)),
                                   multiply(orthogonal, std::sin(rotation))),
                               wide->axis),
          .theta_o = theta_o,
          .theta_e = emission_angle,
          .empty = false};
}

LightTreeMeasure merge_light_tree_measures(const LightTreeMeasure &a,
                                           const LightTreeMeasure &b) noexcept {
  if (!(a.energy > 0.0f)) {
    return b;
  }
  if (!(b.energy > 0.0f)) {
    return a;
  }
  return {.bounds = merge_light_tree_bounds(a.bounds, b.bounds),
          .orientation =
              merge_light_tree_orientation_bounds(a.orientation, b.orientation),
          .energy = a.energy + b.energy};
}

float light_tree_measure_cost(const LightTreeMeasure &measure) noexcept {
  if (!(measure.energy > 0.0f) || measure.orientation.empty) {
    return 0.0f;
  }
  float spatial_measure = 0.0f;
  if (!measure.bounds.empty) {
    const auto extent =
        subtract(measure.bounds.maximum, measure.bounds.minimum);
    spatial_measure = 2.0f * (extent.x * extent.y + extent.x * extent.z +
                              extent.y * extent.z);
    if (spatial_measure == 0.0f) {
      spatial_measure = length(extent);
    }
  }
  const auto theta_w =
      std::min(pi, measure.orientation.theta_o + measure.orientation.theta_e);
  const auto cosine_o = std::cos(measure.orientation.theta_o);
  const auto sine_o = std::sin(measure.orientation.theta_o);
  const auto orientation_measure =
      two_pi * (1.0f - cosine_o) +
      half_pi * (2.0f * theta_w * sine_o -
                 std::cos(measure.orientation.theta_o - 2.0f * theta_w) -
                 2.0f * measure.orientation.theta_o * sine_o + cosine_o);
  return measure.energy * spatial_measure * orientation_measure;
}

CyclesLightTree
build_cycles_light_tree(std::span<const LightTreeEmitter> emitters,
                        std::uint32_t max_emitters_per_leaf) {
  return Builder{emitters, max_emitters_per_leaf}.build();
}

} // namespace psycles::sampling

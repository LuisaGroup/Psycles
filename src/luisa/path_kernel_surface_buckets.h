#pragma once

#include "path_kernel_builder.h"

#include <psycles/compiler/cycles_svm_compiler.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace psycles::luisa_backend::detail {

struct PathSurfaceBucketSpec {
  const char *continuation_name{nullptr};
  std::vector<std::uint8_t> active_shader_mask;
  compiler::cycles_svm::ShaderEntryUsage entry_usage{};
  std::vector<std::uint32_t> membership_words;
  std::vector<std::uint32_t> prefix_popcounts;
  std::uint32_t key_count{1u};
  std::uint32_t active_nonempty_shaders{0u};
  std::uint32_t unique_routine_groups{0u};
  std::uint32_t max_surface_closures{1u};
  bool may_have_emission{true};
  bool may_have_holdout{true};
  bool may_have_bssrdf{true};
  bool may_have_transparent{true};
  bool may_have_ray_portal{false};
};

struct PathSurfaceBucketPlan {
  std::vector<PathSurfaceBucketSpec> buckets;
  std::uint32_t shader_count{0u};
  bool native_specialized{false};
  bool post_closure_substage{false};

  [[nodiscard]] bool multi_bucket() const noexcept {
    return buckets.size() > 1u;
  }

  [[nodiscard]] bool any_may_have_bssrdf() const noexcept {
    for (const auto &b : buckets) {
      if (b.may_have_bssrdf) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::vector<std::string_view>
  producer_continuations() const;

  [[nodiscard]] UInt
  emit_bucket_index(Expr<std::uint32_t> shader_id) const noexcept;

  [[nodiscard]] UInt emit_dense_sort_key(
      std::size_t bucket_index, Expr<std::uint32_t> shader_id) const noexcept;
};

[[nodiscard]] PathSurfaceBucketPlan make_path_surface_bucket_plan(
    const LuisaSceneData &scene, bool staged_surface_sorting,
    bool staged_surface_sort_annotation, bool path_trace_enabled = false,
    bool has_subsurface = false) noexcept;

[[nodiscard]] PathSurfaceBucketPlan
make_path_surface_bucket_plan(const PathKernelConfig &config) noexcept;

} // namespace psycles::luisa_backend::detail

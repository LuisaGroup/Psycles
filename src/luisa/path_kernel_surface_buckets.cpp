#include "path_kernel_surface_buckets.h"

#include "path_kernel_surface_queue.h"
#include "path_kernel_transitions.h"
#include "path_tracer_cycles_svm_kernel_globals.h"

#include <psycles/compiler/cycles_svm_ir.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/cycles_svm_translator.h>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string_view>
#include <tuple>

#include <luisa/core/logging.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace abi = compiler::cycles_svm;
namespace svm = cycles_svm;

enum class SurfaceBucketMode : std::uint8_t {
  auto_select = 0u,
  off = 1u,
  large_only = 2u,
  split_closures = 3u,
};

[[nodiscard]] SurfaceBucketMode configured_bucket_mode() noexcept {
  const char *raw = std::getenv("PSYCLES_SURFACE_BUCKET_MODE");
  if (raw == nullptr || *raw == '\0') {
    return SurfaceBucketMode::auto_select;
  }
  const std::string_view v{raw};
  if (v == "0" || v == "off" || v == "none") {
    return SurfaceBucketMode::off;
  }
  if (v == "large" || v == "large_only") {
    return SurfaceBucketMode::large_only;
  }
  if (v == "split" || v == "split_closures" || v == "aggressive") {
    return SurfaceBucketMode::split_closures;
  }
  return SurfaceBucketMode::auto_select;
}

[[nodiscard]] std::size_t configured_bucket_max_shaders() noexcept {
  const char *raw = std::getenv("PSYCLES_SURFACE_BUCKET_MAX_SHADERS");
  if (raw != nullptr && *raw != '\0') {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end != raw && *end == '\0' && parsed >= 8ul) {
      return static_cast<std::size_t>(parsed);
    }
  }
  return 44u;
}

[[nodiscard]] bool configured_post_closure_substage(
    const LuisaSceneData &scene, bool staged_surface_sort_annotation,
    bool path_trace_enabled) noexcept {
  if (!staged_surface_sort_annotation ||
      path_trace_enabled || scene.cycles_svm == nullptr) {
    return false;
  }
  const auto features = scene.cycles_svm->kernel_features;
  if ((features & svm::kernel_feature_node_portal) != 0u) {
    return false;
  }
  const char *raw = std::getenv("PSYCLES_SURFACE_POST_SUBSTAGE");
  if (raw == nullptr || *raw == '\0') {
    return false;
  }
  const std::string_view v{raw};
  return v == "1" || v == "true" || v == "on";
}

[[nodiscard]] std::size_t count_ir_ops(const abi::SvmBlockIr &block) noexcept {
  std::size_t total = 0u;
  for (const auto &op : block) {
    if (op.is_conditional) {
      total += 1u + count_ir_ops(op.children);
    } else {
      ++total;
    }
  }
  return total;
}

struct RoutineGroup {
  std::vector<std::uint32_t> shader_ids;
  const abi::SvmBlockIr *routine{nullptr};
  std::uint32_t max_closures{0u};
  std::size_t ir_ops{0u};
  std::size_t jit_weight{0u};
  bool has_bssrdf{false};
  bool has_emission{false};
  bool has_holdout{false};
};

void populate_bucket_bitmasks(PathSurfaceBucketSpec &spec,
                              std::uint32_t shader_count,
                              const compiler::cycles_svm::ShaderTableImage *table = nullptr) noexcept {
  const std::size_t word_count =
      (static_cast<std::size_t>(shader_count) + 31u) / 32u;
  const bool has_ir =
      table != nullptr && table->ir.valid &&
      table->ir.shaders.size() == static_cast<std::size_t>(shader_count);
  spec.membership_words.assign(word_count, 0u);
  spec.prefix_popcounts.assign(word_count, 0u);
  spec.sort_key_words.assign(word_count, 0u);
  spec.sort_key_prefix_popcounts.assign(word_count, 0u);
  for (std::uint32_t i = 0u; i < shader_count; ++i) {
    if (i < spec.active_shader_mask.size() && spec.active_shader_mask[i] != 0u) {
      spec.membership_words[i >> 5u] |= (1u << (i & 31u));
      if (!has_ir ||
          !table->ir.shaders[i].routine_for(abi::SHADER_TYPE_SURFACE).empty()) {
        spec.sort_key_words[i >> 5u] |= (1u << (i & 31u));
      }
    }
  }
  std::uint32_t membership_running = 0u;
  std::uint32_t sort_key_running = 0u;
  for (std::size_t w = 0u; w < word_count; ++w) {
    spec.prefix_popcounts[w] = membership_running;
    membership_running += static_cast<std::uint32_t>(
        std::popcount(spec.membership_words[w]));
    spec.sort_key_prefix_popcounts[w] = sort_key_running;
    sort_key_running += static_cast<std::uint32_t>(
        std::popcount(spec.sort_key_words[w]));
  }
  const bool has_collapsed_empty = membership_running > sort_key_running;
  spec.key_count = std::max<std::uint32_t>(
      sort_key_running + (has_collapsed_empty ? 1u : 0u), 1u);
}

void finalize_bucket_metadata(const LuisaSceneData &scene,
                              PathSurfaceBucketSpec &spec) noexcept {
  const auto &compilation = scene.cycles_svm->compilation;
  const auto count = compilation.table.shader_count;
  spec.entry_usage =
      cycles_svm_masked_entry_usage(scene, spec.active_shader_mask);
  std::uint32_t max_closures = 1u;
  bool may_emit = false;
  bool may_holdout = spec.entry_usage.node_types_used[abi::NODE_CLOSURE_HOLDOUT];
  bool may_bssrdf = false;
  bool may_transparent = false;
  bool may_portal =
      (scene.cycles_svm->kernel_features & svm::kernel_feature_node_portal) != 0u;
  std::uint32_t active_nonempty = 0u;

  const bool has_ir =
      compilation.table.ir.valid &&
      compilation.table.ir.shaders.size() == static_cast<std::size_t>(count);

  for (std::uint32_t i = 0u; i < count; ++i) {
    if (i >= spec.active_shader_mask.size() ||
        spec.active_shader_mask[i] == 0u) {
      continue;
    }
    const auto &meta = compilation.shader_metadata[i];
    const auto &entry = compilation.shader_surface_entry_usage[i];
    const auto flags =
        static_cast<std::uint32_t>(compilation.kernel_shaders[i].flags);
    max_closures = std::max(max_closures, meta.surface_num_closures);
    may_bssrdf = may_bssrdf || meta.has_surface_bssrdf;
    may_emit =
        may_emit ||
        (flags & static_cast<std::uint32_t>(abi::SD_HAS_EMISSION)) != 0u ||
        entry.node_types_used[abi::NODE_CLOSURE_EMISSION];
    may_holdout =
        may_holdout || entry.node_types_used[abi::NODE_CLOSURE_HOLDOUT];
    may_transparent =
        may_transparent ||
        (meta.surface_closure_usage.surface_closure_mask &
         abi::ClosureUsage::bit(cycles_closure::type_transparent)) != 0ull ||
        (flags & static_cast<std::uint32_t>(abi::SD_HAS_TRANSPARENT_SHADOW)) !=
            0u;
    if (has_ir &&
        !compilation.table.ir.shaders[i]
             .routine_for(abi::SHADER_TYPE_SURFACE)
             .empty()) {
      ++active_nonempty;
    }
  }
  spec.max_surface_closures = std::clamp<std::uint32_t>(
      max_closures, 1u, static_cast<std::uint32_t>(svm::maximum_closure_capacity));
  spec.may_have_emission = may_emit;
  spec.may_have_holdout = may_holdout;
  spec.may_have_bssrdf =
      may_bssrdf &&
      ((scene.cycles_svm->kernel_features & svm::kernel_feature_subsurface) != 0u);
  spec.may_have_transparent = may_transparent;
  spec.may_have_ray_portal = may_portal;
  spec.active_nonempty_shaders = active_nonempty;
  populate_bucket_bitmasks(spec, count, &compilation.table);
}

[[nodiscard]] UInt select_constant_word_range(
    std::span<const std::uint32_t> words, std::size_t lo, std::size_t hi,
    Expr<std::uint32_t> word_idx) noexcept {
  using namespace luisa::compute;
  const std::uint32_t first = words[lo];
  bool all_equal = true;
  for (std::size_t i = lo + 1u; i < hi; ++i) {
    if (words[i] != first) {
      all_equal = false;
      break;
    }
  }
  if (all_equal) {
    return first;
  }
  const std::size_t mid = lo + (hi - lo) / 2u;
  const UInt left = select_constant_word_range(words, lo, mid, word_idx);
  const UInt right = select_constant_word_range(words, mid, hi, word_idx);
  return select(left, right, word_idx >= static_cast<std::uint32_t>(mid));
}

[[nodiscard]] UInt select_constant_word(
    std::span<const std::uint32_t> words, Expr<std::uint32_t> word_idx) noexcept {
  using namespace luisa::compute;
  if (words.empty()) {
    return 0u;
  }
  return select_constant_word_range(words, 0u, words.size(), word_idx);
}

} // namespace

std::vector<std::string_view>
PathSurfaceBucketPlan::producer_continuations() const {
  std::vector<std::string_view> names;
  if (post_closure_substage) {
    names.emplace_back(path_transition::shade_surface_post);
    return names;
  }
  names.reserve(buckets.size());
  for (const auto &b : buckets) {
    names.emplace_back(b.continuation_name);
  }
  return names;
}

UInt PathSurfaceBucketPlan::emit_bucket_index(
    Expr<std::uint32_t> shader_id) const noexcept {
  using namespace luisa::compute;
  if (buckets.size() <= 1u || shader_count == 0u) {
    return 0u;
  }
  const UInt s = min(UInt{shader_id}, UInt{shader_count - 1u});
  const UInt word_idx = s >> 5u;
  const UInt bit_mask = 1u << (s & 31u);
  UInt bucket = 0u;
  for (std::size_t b = 1u; b < buckets.size(); ++b) {
    const UInt w = select_constant_word(buckets[b].membership_words, word_idx);
    bucket = select(bucket, UInt{static_cast<std::uint32_t>(b)},
                    (w & bit_mask) != 0u);
  }
  return bucket;
}

UInt PathSurfaceBucketPlan::emit_dense_sort_key(
    std::size_t bucket_index, Expr<std::uint32_t> shader_id) const noexcept {
  using namespace luisa::compute;
  LUISA_ASSERT(bucket_index < buckets.size(),
               "Surface bucket index {} out of range {}.", bucket_index,
               buckets.size());
  const auto &spec = buckets[bucket_index];
  if (!native_specialized || buckets.size() <= 1u || shader_count == 0u) {
    return shader_id;
  }
  if (spec.key_count <= 1u) {
    return 0u;
  }
  const UInt s = min(UInt{shader_id}, UInt{shader_count - 1u});
  const UInt word_idx = s >> 5u;
  const UInt bit_mask = 1u << (s & 31u);
  const UInt low_mask = bit_mask - 1u;
  const UInt w = select_constant_word(spec.sort_key_words, word_idx);
  const UInt base = select_constant_word(spec.sort_key_prefix_popcounts, word_idx);
  const UInt dense = min(base + popcount(w & low_mask), UInt{spec.key_count - 1u});
  return select(UInt{spec.key_count - 1u}, dense, (w & bit_mask) != 0u);
}

PathSurfaceBucketPlan make_path_surface_bucket_plan(
    const LuisaSceneData &scene, bool staged_surface_sorting,
    bool staged_surface_sort_annotation, bool path_trace_enabled,
    bool has_subsurface, bool force_large_bucket_split) noexcept {
  PathSurfaceBucketPlan plan{};
  plan.post_closure_substage = configured_post_closure_substage(
      scene, staged_surface_sort_annotation, path_trace_enabled);

  const bool can_specialize_native =
      !path_trace_enabled &&
      cycles_svm_has_domain_metadata(scene) &&
      scene.cycles_svm->compilation.table.ir.valid &&
      scene.cycles_svm->compilation.table.ir.shaders.size() ==
          static_cast<std::size_t>(
              scene.cycles_svm->compilation.table.shader_count);

  if (!can_specialize_native) {
    PathSurfaceBucketSpec single{};
    single.continuation_name = path_transition::shade_surface;
    single.key_count = std::max(surface_queue_key_range(scene), 1u);
    single.max_surface_closures =
        scene.cycles_svm
            ? std::clamp<std::uint32_t>(
                  static_cast<std::uint32_t>(
                      scene.cycles_svm->compilation.max_surface_closures),
                  1u, static_cast<std::uint32_t>(svm::maximum_closure_capacity))
            : static_cast<std::uint32_t>(svm::maximum_closure_capacity);
    single.may_have_bssrdf = has_subsurface;
    plan.buckets.push_back(std::move(single));
    return plan;
  }

  const auto &compilation = scene.cycles_svm->compilation;
  const auto count = compilation.table.shader_count;
  plan.shader_count = count;
  plan.native_specialized = true;

  const auto domain_mask = cycles_svm_domain_shader_mask(
      scene, PathCyclesSvmShaderDomain::surface_only);

  const auto mode = configured_bucket_mode();
  const auto backend_name =
      static_cast<bool>(scene.device)
          ? std::string_view{scene.device.backend_name()}
          : std::string_view{};
  std::size_t domain_active_nonempty = 0u;
  std::size_t domain_unique_jit_weight = 0u;
  const auto host_words = std::span<const std::uint32_t>{compilation.table.words};
  luisa::vector<const abi::SvmBlockIr *> domain_unique_routines;
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (i < domain_mask.size() && domain_mask[i] == 0u) {
      continue;
    }
    const auto &routine =
        compilation.table.ir.shaders[i].routine_for(abi::SHADER_TYPE_SURFACE);
    if (!routine.empty()) {
      ++domain_active_nonempty;
      bool merged = false;
      for (const abi::SvmBlockIr *existing : domain_unique_routines) {
        if (abi::same_linked_svm_block(host_words, *existing, routine)) {
          merged = true;
          break;
        }
      }
      if (!merged) {
        domain_unique_routines.push_back(&routine);
        domain_unique_jit_weight += svm::svm_block_jit_weight(
            routine, host_words, abi::SHADER_TYPE_SURFACE,
            svm::kernel_feature_node_mask_surface, false);
      }
    }
  }
  const std::size_t jit_weight_cap =
      std::max<std::size_t>(svm::direct_jit_max_weight(), 1u);
  const bool explicit_weight_env = []() noexcept {
    const char *raw = std::getenv("PSYCLES_SVM_JIT_MAX_WEIGHT");
    return raw != nullptr && *raw != '\0';
  }();
  std::size_t bucket_weight_cap = jit_weight_cap;
  if (!explicit_weight_env) {
    if (backend_name == "hip") {
      bucket_weight_cap = std::min<std::size_t>(jit_weight_cap, 9000u);
    } else if (backend_name == "dx") {
      bucket_weight_cap = std::min<std::size_t>(jit_weight_cap, 7000u);
    } else if (backend_name == "vk") {
      bucket_weight_cap = std::min<std::size_t>(jit_weight_cap, 5000u);
    }
  }
  const SurfaceBucketMode effective_mode = [&]() noexcept {
    if (mode != SurfaceBucketMode::auto_select) {
      return mode;
    }
    if (force_large_bucket_split) {
      return SurfaceBucketMode::large_only;
    }
    // Multi-bucket splitting reduces shader I-cache/register pressure on
    // CUDA and Metal, while on DX and Vulkan single-bucket dispatch avoids
    // extra per-bucket surface-sort wave launches when active surface
    // shaders fit within DirectJitTranslator's 80-shader budget. Beyond 80
    // shaders (e.g. Barbershop), falling back to a single BytecodeVM bucket
    // severely degrades DX/VK throughput, so split into <=80-shader buckets.
    if (backend_name == "dx" &&
        domain_active_nonempty <= svm::direct_jit_max_shaders() &&
        domain_unique_jit_weight <= bucket_weight_cap) {
      return SurfaceBucketMode::off;
    }
    if (backend_name == "vk" &&
        domain_active_nonempty <= svm::direct_jit_max_shaders()) {
      return SurfaceBucketMode::off;
    }
    return SurfaceBucketMode::large_only;
  }();
  const bool allow_multi_bucket =
      staged_surface_sort_annotation && staged_surface_sorting &&
      effective_mode != SurfaceBucketMode::off && count > 1u;

  if (!allow_multi_bucket) {
    PathSurfaceBucketSpec single{};
    single.continuation_name = path_transition::shade_surface;
    single.active_shader_mask = domain_mask;
    finalize_bucket_metadata(scene, single);
    single.key_count = std::max(count, 1u);
    plan.buckets.push_back(std::move(single));
    return plan;
  }

  std::vector<RoutineGroup> groups;
  std::vector<std::uint32_t> inactive_or_empty_shaders;

  for (std::uint32_t i = 0u; i < count; ++i) {
    if (i < domain_mask.size() && domain_mask[i] == 0u) {
      inactive_or_empty_shaders.push_back(i);
      continue;
    }
    const auto &routine =
        compilation.table.ir.shaders[i].routine_for(abi::SHADER_TYPE_SURFACE);
    if (routine.empty()) {
      inactive_or_empty_shaders.push_back(i);
      continue;
    }
    const auto &meta = compilation.shader_metadata[i];
    const auto &entry = compilation.shader_surface_entry_usage[i];
    const auto flags =
        static_cast<std::uint32_t>(compilation.kernel_shaders[i].flags);
    const bool has_emit =
        (flags & static_cast<std::uint32_t>(abi::SD_HAS_EMISSION)) != 0u ||
        entry.node_types_used[abi::NODE_CLOSURE_EMISSION];
    const bool has_hold = entry.node_types_used[abi::NODE_CLOSURE_HOLDOUT];

    bool merged = false;
    for (auto &g : groups) {
      if (abi::same_linked_svm_block(host_words, *g.routine, routine)) {
        g.shader_ids.push_back(i);
        g.max_closures = std::max(g.max_closures, meta.surface_num_closures);
        g.has_bssrdf = g.has_bssrdf || meta.has_surface_bssrdf;
        g.has_emission = g.has_emission || has_emit;
        g.has_holdout = g.has_holdout || has_hold;
        merged = true;
        break;
      }
    }
    if (!merged) {
      RoutineGroup g{};
      g.shader_ids.push_back(i);
      g.routine = &routine;
      g.max_closures = meta.surface_num_closures;
      g.ir_ops = count_ir_ops(routine);
      g.jit_weight = std::max<std::size_t>(
          svm::svm_block_jit_weight(routine, host_words,
                                    abi::SHADER_TYPE_SURFACE,
                                    svm::kernel_feature_node_mask_surface,
                                    false),
          1u);
      g.has_bssrdf = meta.has_surface_bssrdf;
      g.has_emission = has_emit;
      g.has_holdout = has_hold;
      groups.push_back(std::move(g));
    }
  }

  std::sort(groups.begin(), groups.end(),
            [](const RoutineGroup &a, const RoutineGroup &b) noexcept {
              const bool a_multi = a.max_closures > 1u;
              const bool b_multi = b.max_closures > 1u;
              const bool a_special = a.has_emission || a.has_holdout;
              const bool b_special = b.has_emission || b.has_holdout;
              return std::tuple{a_multi, a.has_bssrdf, a_special,
                                a.max_closures, a.ir_ops,
                                a.shader_ids.front()} <
                     std::tuple{b_multi, b.has_bssrdf, b_special,
                                b.max_closures, b.ir_ops,
                                b.shader_ids.front()};
            });

  std::size_t total_active_nonempty = 0u;
  std::size_t total_jit_weight = 0u;
  std::size_t pure_simple_shaders = 0u;
  std::size_t pure_simple_weight = 0u;
  std::size_t pure_simple_groups = 0u;
  std::size_t simple_single_closure_shaders = 0u;
  std::size_t simple_single_closure_weight = 0u;
  std::size_t simple_single_closure_groups = 0u;
  for (const auto &g : groups) {
    total_active_nonempty += g.shader_ids.size();
    total_jit_weight += g.jit_weight;
    if (g.max_closures <= 1u && !g.has_bssrdf) {
      simple_single_closure_shaders += g.shader_ids.size();
      simple_single_closure_weight += g.jit_weight;
      ++simple_single_closure_groups;
      if (!g.has_emission && !g.has_holdout) {
        pure_simple_shaders += g.shader_ids.size();
        pure_simple_weight += g.jit_weight;
        ++pure_simple_groups;
      }
    }
  }

  const std::size_t max_per_bucket = configured_bucket_max_shaders();
  const std::size_t jit_shader_cap =
      std::max<std::size_t>(svm::direct_jit_max_shaders(), 1u);

  std::vector<std::size_t> suffix_min_buckets(groups.size() + 1u, 0u);
  for (std::size_t i = groups.size(); i-- > 0u;) {
    std::size_t s = groups[i].shader_ids.size();
    std::size_t w = groups[i].jit_weight;
    std::size_t j = i + 1u;
    while (j < groups.size() &&
           s + groups[j].shader_ids.size() <= jit_shader_cap &&
           w + groups[j].jit_weight <= bucket_weight_cap) {
      s += groups[j].shader_ids.size();
      w += groups[j].jit_weight;
      ++j;
    }
    suffix_min_buckets[i] = 1u + suffix_min_buckets[j];
  }
  const std::size_t min_buckets_for_jit = suffix_min_buckets[0u];

  const std::size_t max_buckets_allowed = std::min<std::size_t>(
      path_transition::max_surface_buckets, groups.size());
  std::size_t bucket_count = 1u;
  if ((total_active_nonempty > max_per_bucket || min_buckets_for_jit > 1u) &&
      groups.size() >= 2u) {
    const std::size_t shader_buckets = std::clamp<std::size_t>(
        (total_active_nonempty + max_per_bucket - 1u) / max_per_bucket,
        2u, 4u);
    bucket_count = std::clamp<std::size_t>(
        std::max(shader_buckets, min_buckets_for_jit), 2u,
        max_buckets_allowed);
  } else if (effective_mode == SurfaceBucketMode::split_closures &&
             simple_single_closure_groups >= 3u &&
             simple_single_closure_groups < groups.size() &&
             (groups.size() - simple_single_closure_groups) >= 2u) {
    bucket_count = 2u;
  }

  if (bucket_count <= 1u) {
    PathSurfaceBucketSpec single{};
    single.continuation_name = path_transition::shade_surface;
    single.active_shader_mask = domain_mask;
    single.unique_routine_groups = static_cast<std::uint32_t>(groups.size());
    finalize_bucket_metadata(scene, single);
    single.key_count = std::max(count, 1u);
    plan.buckets.push_back(std::move(single));
    return plan;
  }

  // Partition sorted routine groups into `bucket_count` contiguous slices so
  // every bucket fits within DirectJitTranslator's shader and weight budgets
  // whenever feasible (`min_buckets_for_jit <= bucket_count`), while snapping
  // Bucket 0 to the single-closure boundary when compatible.
  std::vector<std::size_t> group_cuts;
  if (bucket_count == 2u) {
    std::size_t cut = groups.size() / 2u;
    const auto fits_slice = [&](std::size_t shaders,
                                std::size_t weight) noexcept {
      return shaders <= jit_shader_cap && weight <= bucket_weight_cap;
    };
    if (pure_simple_groups >= 1u &&
        pure_simple_groups < groups.size() &&
        fits_slice(pure_simple_shaders, pure_simple_weight) &&
        fits_slice(total_active_nonempty - pure_simple_shaders,
                   total_jit_weight - pure_simple_weight)) {
      cut = pure_simple_groups;
    } else if (simple_single_closure_groups >= 1u &&
               simple_single_closure_groups < groups.size() &&
               fits_slice(simple_single_closure_shaders,
                          simple_single_closure_weight) &&
               fits_slice(total_active_nonempty - simple_single_closure_shaders,
                          total_jit_weight - simple_single_closure_weight)) {
      cut = simple_single_closure_groups;
    } else {
      std::size_t acc_s = 0u;
      std::size_t acc_w = 0u;
      std::size_t best_cut = 1u;
      std::size_t best_diff = std::numeric_limits<std::size_t>::max();
      bool found_feasible = false;
      for (std::size_t gi = 0u; gi + 1u < groups.size(); ++gi) {
        acc_s += groups[gi].shader_ids.size();
        acc_w += groups[gi].jit_weight;
        const std::size_t rem_s = total_active_nonempty - acc_s;
        const std::size_t rem_w = total_jit_weight - acc_w;
        const bool feasible =
            fits_slice(acc_s, acc_w) && fits_slice(rem_s, rem_w);
        const std::size_t diff =
            acc_w > rem_w ? (acc_w - rem_w) : (rem_w - acc_w);
        if (feasible && (!found_feasible || diff < best_diff)) {
          found_feasible = true;
          best_diff = diff;
          best_cut = gi + 1u;
        } else if (!found_feasible && diff < best_diff) {
          best_diff = diff;
          best_cut = gi + 1u;
        }
      }
      cut = best_cut;
    }
    group_cuts = {0u, cut, groups.size()};
  } else {
    group_cuts.push_back(0u);
    std::size_t start = 0u;
    std::size_t remaining_weight = total_jit_weight;
    std::size_t remaining_shaders = total_active_nonempty;
    const bool jit_feasible = min_buckets_for_jit <= bucket_count;
    for (std::size_t b = 0u; b + 1u < bucket_count; ++b) {
      const std::size_t buckets_left = bucket_count - b;
      const std::size_t rem_buckets = buckets_left - 1u;
      const std::size_t target_weight =
          (remaining_weight + buckets_left - 1u) / buckets_left;
      const std::size_t max_end = groups.size() - rem_buckets;
      std::size_t end = start + 1u;
      std::size_t acc_weight = groups[start].jit_weight;
      std::size_t acc_shaders = groups[start].shader_ids.size();
      if (jit_feasible) {
        while (end < max_end && suffix_min_buckets[end] > rem_buckets) {
          acc_weight += groups[end].jit_weight;
          acc_shaders += groups[end].shader_ids.size();
          ++end;
        }
        while (end < max_end) {
          if (b == 0u && groups[end - 1u].max_closures <= 1u &&
              groups[end].max_closures > 1u &&
              suffix_min_buckets[end] <= rem_buckets) {
            break;
          }
          const std::size_t next_s = groups[end].shader_ids.size();
          const std::size_t next_w = groups[end].jit_weight;
          if (acc_weight >= target_weight ||
              acc_shaders + next_s > jit_shader_cap ||
              acc_weight + next_w > bucket_weight_cap) {
            break;
          }
          acc_weight += next_w;
          acc_shaders += next_s;
          ++end;
        }
      } else {
        const std::size_t target_shaders =
            (remaining_shaders + buckets_left - 1u) / buckets_left;
        while (end < max_end && acc_weight < target_weight &&
               acc_shaders < target_shaders) {
          acc_weight += groups[end].jit_weight;
          acc_shaders += groups[end].shader_ids.size();
          ++end;
        }
      }
      remaining_weight -= acc_weight;
      remaining_shaders -= acc_shaders;
      group_cuts.push_back(end);
      start = end;
    }
    group_cuts.push_back(groups.size());
  }

  plan.buckets.resize(bucket_count);
  for (std::size_t b = 0u; b < bucket_count; ++b) {
    auto &spec = plan.buckets[b];
    spec.continuation_name =
        path_transition::shade_surface_bucket_name(static_cast<unsigned>(b));
    spec.active_shader_mask.assign(count, 0u);
    spec.unique_routine_groups =
        static_cast<std::uint32_t>(group_cuts[b + 1u] - group_cuts[b]);
    for (std::size_t gi = group_cuts[b]; gi < group_cuts[b + 1u]; ++gi) {
      for (const std::uint32_t sid : groups[gi].shader_ids) {
        if (sid < count) {
          spec.active_shader_mask[sid] = 1u;
        }
      }
    }
    if (b == 0u) {
      // Route empty/background-only shader indices to Bucket 0 ($default no-op)
      // so every shader ID in [0, count) belongs to exactly one bucket.
      for (const std::uint32_t sid : inactive_or_empty_shaders) {
        if (sid < count) {
          spec.active_shader_mask[sid] = 1u;
        }
      }
    }
    std::size_t bucket_ir_ops = 0u;
    std::size_t bucket_jit_weight = 0u;
    for (std::size_t gi = group_cuts[b]; gi < group_cuts[b + 1u]; ++gi) {
      bucket_ir_ops += groups[gi].ir_ops;
      bucket_jit_weight += groups[gi].jit_weight;
    }
    finalize_bucket_metadata(scene, spec);
    LUISA_INFO("Psycles surface bucket[{}] '{}': active_nonempty={} "
               "unique_groups={} ir_ops={} jit_weight={} sort_keys={} "
               "max_closures={} emit={} holdout={} bssrdf={}.",
               b, spec.continuation_name, spec.active_nonempty_shaders,
               spec.unique_routine_groups, bucket_ir_ops, bucket_jit_weight,
               spec.key_count, spec.max_surface_closures,
               spec.may_have_emission, spec.may_have_holdout,
               spec.may_have_bssrdf);
  }
  return plan;
}

PathSurfaceBucketPlan
make_path_surface_bucket_plan(const PathKernelConfig &config) noexcept {
  return make_path_surface_bucket_plan(
      *config.scene, config.staged_surface_sorting,
      config.staged_surface_sort_annotation, config.path_trace_enabled,
      config.has_subsurface);
}

} // namespace psycles::luisa_backend::detail

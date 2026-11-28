#include "path_kernel_surface_queue.h"

#include "cycles_shader_identity.h"
#include "cycles_surface_sort.h"
#include "path_kernel_primitive_material.h"

#include <psycles/core/parallel.h>
#include <psycles/luisa/cycles_svm.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <luisa/coro/radix_sort.h>

namespace psycles::luisa_backend::detail {
namespace {

constexpr auto surface_sort_schema = "org.psycles.cycles.surface_sort";

struct FusedSurfaceSortShaders {
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint,
                           luisa::uint, luisa::uint, luisa::uint>
      single_block_sort;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint,
                           luisa::uint, luisa::uint, luisa::uint,
                           luisa::uint, luisa::uint>
      bucket;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>, luisa::uint>
      prefix;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint,
                           luisa::uint, luisa::uint, luisa::uint,
                           luisa::uint, luisa::uint>
      scatter;
};

struct SurfaceSortShaderCacheKey {
  std::vector<std::uint64_t> words;
  luisa::string shader_name;
  luisa::string native_include;
  bool operator==(const SurfaceSortShaderCacheKey &) const noexcept = default;
};

struct SurfaceSortShaderCache {
  std::mutex mtx;
  std::vector<std::pair<SurfaceSortShaderCacheKey,
                        std::shared_ptr<const FusedSurfaceSortShaders>>>
      entries;
};

[[nodiscard]] std::optional<SurfaceSortShaderCacheKey>
make_fused_shader_cache_key(
    const luisa::compute::coro::WavefrontCoroExtensionPrepareContext &context,
    const luisa::compute::coro::CoroSlotAccess &binding,
    luisa::span<const std::size_t> fields) noexcept {
  if (binding.pieces().empty()) {
    return std::nullopt;
  }
  SurfaceSortShaderCacheKey key;
  key.shader_name = context.shader_option.name;
  key.native_include = context.shader_option.native_include;
  auto &w = key.words;
  w.reserve(64u);
  w.push_back(reinterpret_cast<std::uintptr_t>(&context.device));
  const auto &opt = context.shader_option;
  const std::uint64_t opt_bits =
      (static_cast<std::uint64_t>(opt.enable_cache) << 0u) |
      (static_cast<std::uint64_t>(opt.enable_fast_math) << 1u) |
      (static_cast<std::uint64_t>(opt.enable_debug_info) << 2u) |
      (static_cast<std::uint64_t>(opt.compile_only) << 3u) |
      (static_cast<std::uint64_t>(opt.time_trace) << 4u) |
      (static_cast<std::uint64_t>(opt.enable_extended_accel_limits) << 5u) |
      (static_cast<std::uint64_t>(opt.enable_scalarizer) << 6u) |
      (static_cast<std::uint64_t>(opt.enable_ray_query_pipeline) << 7u) |
      (static_cast<std::uint64_t>(opt.force_ray_query_pipeline) << 8u) |
      (static_cast<std::uint64_t>(opt.assume_no_packed_textures) << 9u) |
      (static_cast<std::uint64_t>(opt.enable_driver_optimization) << 10u);
  w.push_back(opt_bits);
  w.push_back(opt.max_registers);
  const auto &layout = context.frame_layout;
  const bool runtime_cap = layout.has_runtime_capacity();
  w.push_back(static_cast<std::uint64_t>(context.global_memory_soa));
  w.push_back(layout.frame_stride);
  w.push_back(static_cast<std::uint64_t>(
      layout.size_bytes <= std::numeric_limits<luisa::uint>::max()));
  w.push_back(static_cast<std::uint64_t>(runtime_cap));
  auto append_field = [&](std::size_t f) noexcept {
    w.push_back(f);
    w.push_back(reinterpret_cast<std::uintptr_t>(
        f < context.frame_desc.frame_field_count()
            ? context.frame_desc.frame_field_type(f)
            : nullptr));
    w.push_back(f < layout.field_offsets.size() ? layout.field_offsets[f] : 0u);
    w.push_back(f < layout.field_strides.size() ? layout.field_strides[f] : 0u);
    w.push_back(runtime_cap && f < layout.field_capacity_strides.size()
                    ? layout.field_capacity_strides[f]
                    : 0u);
  };
  append_field(6u);
  w.push_back(fields.size());
  for (const auto f : fields) {
    append_field(f);
  }
  std::size_t matching_chunks = 0u;
  for (const auto &chunk : layout.packed_chunks) {
    bool touches = false;
    for (std::uint8_t lane = 0u; lane < chunk.width; ++lane) {
      if (std::find(fields.begin(), fields.end(), chunk.fields[lane]) !=
          fields.end()) {
        touches = true;
        break;
      }
    }
    if (!touches) {
      continue;
    }
    ++matching_chunks;
    w.push_back(chunk.capacity_stride);
    w.push_back(chunk.width);
    for (const auto cf : chunk.fields) {
      w.push_back(cf);
    }
  }
  w.push_back(matching_chunks);
  w.push_back(reinterpret_cast<std::uintptr_t>(binding.type()));
  w.push_back(static_cast<std::uint64_t>(binding.access()));
  w.push_back(static_cast<std::uint64_t>(binding.lifetime()));
  w.push_back(binding.pieces().size());
  for (const auto &piece : binding.pieces()) {
    w.push_back(piece.frame_value_index);
    w.push_back(piece.field_index);
    w.push_back(reinterpret_cast<std::uintptr_t>(piece.logical_type));
    w.push_back(reinterpret_cast<std::uintptr_t>(piece.physical_type));
    w.push_back(static_cast<std::uint64_t>(piece.bit_offset.has_value()));
    w.push_back(piece.bit_offset.value_or(0u));
    w.push_back(piece.access_chain.size());
    for (const auto idx : piece.access_chain) {
      w.push_back(idx);
    }
  }
  return key;
}

class SurfaceSortHandler final
    : public luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler {
  using Prepare = luisa::compute::coro::WavefrontCoroExtensionPrepareContext;
  using Stage = luisa::compute::coro::WavefrontCoroExtensionStage;
  using Dispatch = luisa::compute::coro::WavefrontCoroExtensionDispatchContext;
  static constexpr unsigned max_fused_shader_range = 256u;
  static constexpr unsigned max_fused_composite_range = 16384u;
  static constexpr unsigned single_block_max_scan = 4096u;
  luisa::compute::Buffer<luisa::uint> _keys[2u];
  luisa::compute::Buffer<luisa::uint> _indices;
  luisa::compute::Buffer<luisa::uint> _bucket_hist;
  luisa::compute::Buffer<luisa::uint> _bucket_offsets;
  std::shared_ptr<const FusedSurfaceSortShaders> _fused_shaders;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>, luisa::uint>
      _bucket_clear_shader;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint,
                           luisa::uint, luisa::uint, luisa::uint>
      _bucket_hist_shader;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint>
      _bucket_scatter_shader;
  luisa::compute::coro::radix_sort::temp_storage _temporary;
  unsigned _range;
  unsigned _partition;
  unsigned _composite_range;
  unsigned _subblocks_per_partition{1u};
  unsigned _max_total_blocks{1u};
  bool _use_fused_partition_sort{false};
  bool _warp32{false};
  bool _fused_buffers_allocated{false};
  bool _fallback_shaders_compiled{false};
  unsigned _min_sort_frames{512u};
  luisa::compute::Device *_device{nullptr};
  unsigned _frame_capacity{0u};
  luisa::compute::ShaderOption _shader_option{};
  std::optional<luisa::compute::Callable<luisa::uint(
      luisa::uint, luisa::compute::Buffer<luisa::uint>,
      luisa::compute::ByteBuffer, luisa::uint, luisa::uint, luisa::uint)>>
      _key_callable;
  luisa::compute::coro::radix_sort::instance<luisa::compute::Buffer<luisa::uint>,
                                             luisa::compute::ByteBuffer, luisa::uint,
                                             luisa::uint, luisa::uint>
      _sort;

public:
  SurfaceSortHandler(Prepare &context, const Stage &stage, unsigned range,
                     unsigned partition, unsigned composite_range,
                     std::shared_ptr<SurfaceSortShaderCache> shader_cache = nullptr)
      : _range{std::max(range, 1u)}, _partition{partition},
        _composite_range{std::max(composite_range, 1u)},
        _warp32{context.device.compute_warp_size() == 32u},
        _device{&context.device},
        _frame_capacity{context.frame_capacity},
        _shader_option{context.shader_option} {
    using namespace luisa::compute;
    using namespace luisa::compute::coro;
    _use_fused_partition_sort =
        _warp32 && _range <= max_fused_shader_range &&
        _composite_range <= max_fused_composite_range;
    if (const char *raw = std::getenv("PSYCLES_SURFACE_SORT_MIN_FRAMES");
        raw != nullptr && *raw != '\0') {
      _min_sort_frames = static_cast<unsigned>(std::strtoul(raw, nullptr, 10));
    }
    if (!_use_fused_partition_sort) {
      _indices = context.device.create_buffer<luisa::uint>(context.frame_capacity);
    }
    const auto *binding = &stage.binding("shader");
    const auto *desc = &context.frame_desc;
    auto fields = binding->reconstruct_slots();
    Callable<luisa::uint(luisa::uint, ByteBuffer, luisa::uint)> read_shader =
        [binding, desc, fields, layout = context.frame_layout,
         soa = context.global_memory_soa](UInt index, ByteBufferVar storage,
                                          UInt capacity) {
          auto frame = CoroFrame::create(desc);
          coro_frame_load_into(frame, storage, index, capacity, layout, soa,
                               fields, false, false);
          return binding->read<luisa::uint>(frame);
        };
    _key_callable.emplace([read_shader](UInt slot, BufferUInt indices,
                                          ByteBufferVar storage, UInt capacity,
                                          UInt range_arg, UInt partition_arg) {
      auto index = indices.read(slot);
      auto shader = read_shader(index, storage, capacity);
      UInt locality = 0u;
      $if(partition_arg != 0u) { locality = index / partition_arg; };
      return shader + locality * range_arg;
    });

    if (_use_fused_partition_sort) {
      const auto max_partitions =
          _partition == 0u
              ? 1u
              : std::max(1u, (context.frame_capacity + _partition - 1u) /
                                 _partition);
      const unsigned target_blocks =
          context.frame_capacity >= 65536u
              ? 96u
              : (context.frame_capacity >= 4096u ? 24u : 1u);
      _subblocks_per_partition = std::clamp(
          (target_blocks + max_partitions - 1u) / max_partitions, 1u, 64u);
      _max_total_blocks = max_partitions * _subblocks_per_partition;

      std::optional<SurfaceSortShaderCacheKey> cache_key;
      if (shader_cache != nullptr) {
        cache_key = make_fused_shader_cache_key(context, *binding, fields);
        if (cache_key.has_value()) {
          std::lock_guard lock{shader_cache->mtx};
          for (const auto &[k, cached] : shader_cache->entries) {
            if (k == *cache_key) {
              _fused_shaders = cached;
              break;
            }
          }
        }
      }

      if (_fused_shaders == nullptr) {
        auto compiled = std::make_shared<FusedSurfaceSortShaders>();
        Kernel1D fused_single_block_sort_kernel =
            [read_shader, layout = context.frame_layout,
             soa = context.global_memory_soa](
                ByteBufferVar storage, BufferUInt indices_out, UInt capacity,
                UInt source_queue, UInt scan_count,
                UInt shader_range) noexcept {
              set_block_size(256u);
              set_warp_size(32u);
              Shared<uint> hist{max_fused_shader_range};
              Shared<uint> warp_sums{8u};
              const auto tid = thread_x();
              const auto warp_id = tid >> 5u;
              const auto lane_id = tid & 31u;
              $if(tid < shader_range) { hist[tid] = 0u; };
              sync_block();

              UInt active_mask = 0u;
              ArrayUInt<16u> cached_keys;
              UInt iter = 0u;
              UInt base_pos = 0u;
              $while(base_pos < scan_count) {
                const auto pos = base_pos + tid;
                $if(pos < scan_count) {
                  const auto tok = coro_frame_read_field<uint>(
                      storage, pos, capacity, layout, soa, 6u);
                  $if(tok == source_queue) {
                    const auto s_key = min(read_shader(pos, storage, capacity),
                                           shader_range - 1u);
                    active_mask |= (1u << iter);
                    cached_keys[iter] = s_key;
                    $if(warp_active_all_equal(s_key)) {
                      const auto total = warp_active_count_bits(true);
                      $if(warp_prefix_count_bits(true) == 0u) {
                        hist.atomic(s_key).fetch_add(total);
                      };
                    }
                    $else {
                      hist.atomic(s_key).fetch_add(1u);
                    };
                  };
                };
                base_pos += 256u;
                iter += 1u;
              };
              sync_block();

              UInt val = 0u;
              $if(tid < shader_range) { val = hist[tid]; };
              const auto w_prefix = warp_prefix_sum(val);
              const auto w_total = warp_active_sum(val);
              $if(lane_id == 0u) { warp_sums[warp_id] = w_total; };
              sync_block();
              $if(warp_id == 0u) {
                UInt ws = 0u;
                $if(lane_id < 8u) { ws = warp_sums[lane_id]; };
                const auto wp = warp_prefix_sum(ws);
                $if(lane_id < 8u) { warp_sums[lane_id] = wp; };
              };
              sync_block();
              $if(tid < shader_range) {
                hist[tid] = warp_sums[warp_id] + w_prefix;
              };
              sync_block();

              iter = 0u;
              base_pos = 0u;
              $while(base_pos < scan_count) {
                $if((active_mask & (1u << iter)) != 0u) {
                  const auto s_key = cached_keys[iter];
                  UInt dst = 0u;
                  $if(warp_active_all_equal(s_key)) {
                    const auto total = warp_active_count_bits(true);
                    const auto prefix = warp_prefix_count_bits(true);
                    UInt base_dst = 0u;
                    $if(prefix == 0u) {
                      base_dst = hist.atomic(s_key).fetch_add(total);
                    };
                    dst = warp_read_first_active_lane(base_dst) + prefix;
                  }
                  $else {
                    dst = hist.atomic(s_key).fetch_add(1u);
                  };
                  indices_out.write(dst, base_pos + tid);
                };
                base_pos += 256u;
                iter += 1u;
              };
            };

        Kernel1D fused_bucket_kernel =
            [read_shader, layout = context.frame_layout,
             soa = context.global_memory_soa](
                ByteBufferVar storage, BufferUInt block_hist, UInt capacity,
                UInt source_queue, UInt scan_count, UInt partition_eff,
                UInt subblocks_per_part, UInt shader_range) noexcept {
              set_block_size(256u);
              set_warp_size(32u);
              Shared<uint> local_hist{max_fused_shader_range};
              const auto tid = thread_x();
              $if(tid < shader_range) { local_hist[tid] = 0u; };
              sync_block();

              const auto b = block_x();
              const auto p = b / subblocks_per_part;
              const auto s = b - p * subblocks_per_part;
              const auto part_begin = min(p * partition_eff, scan_count);
              const auto part_end = min(part_begin + partition_eff, scan_count);
              const auto sub_size =
                  (partition_eff + subblocks_per_part - 1u) / subblocks_per_part;
              const auto sub_begin = min(part_begin + s * sub_size, part_end);
              const auto sub_end = min(sub_begin + sub_size, part_end);

              UInt base_pos = sub_begin;
              $while(base_pos < sub_end) {
                const auto pos0 = base_pos + tid;
                const auto pos1 = pos0 + 256u;
                const auto pos2 = pos0 + 512u;
                const auto pos3 = pos0 + 768u;
                const UInt miss_tok = ~source_queue;
                UInt tok0 = miss_tok;
                UInt tok1 = miss_tok;
                UInt tok2 = miss_tok;
                UInt tok3 = miss_tok;
                $if(pos0 < sub_end) {
                  tok0 = coro_frame_read_field<uint>(storage, pos0, capacity,
                                                     layout, soa, 6u);
                };
                $if(pos1 < sub_end) {
                  tok1 = coro_frame_read_field<uint>(storage, pos1, capacity,
                                                     layout, soa, 6u);
                };
                $if(pos2 < sub_end) {
                  tok2 = coro_frame_read_field<uint>(storage, pos2, capacity,
                                                     layout, soa, 6u);
                };
                $if(pos3 < sub_end) {
                  tok3 = coro_frame_read_field<uint>(storage, pos3, capacity,
                                                     layout, soa, 6u);
                };
                const auto count_match = [&](UInt pos, UInt tok) noexcept {
                  $if(tok == source_queue) {
                    const auto s_key = min(read_shader(pos, storage, capacity),
                                           shader_range - 1u);
                    $if(warp_active_all_equal(s_key)) {
                      const auto total = warp_active_count_bits(true);
                      $if(warp_prefix_count_bits(true) == 0u) {
                        local_hist.atomic(s_key).fetch_add(total);
                      };
                    }
                    $else { local_hist.atomic(s_key).fetch_add(1u); };
                  };
                };
                count_match(pos0, tok0);
                count_match(pos1, tok1);
                count_match(pos2, tok2);
                count_match(pos3, tok3);
                base_pos += 1024u;
              };
              sync_block();

              $if(tid < shader_range) {
                const auto hist_idx =
                    (p * shader_range + tid) * subblocks_per_part + s;
                block_hist.write(hist_idx, local_hist[tid]);
              };
            };

        Kernel1D fused_prefix_kernel = [](BufferUInt hist,
                                          UInt total_buckets) noexcept {
          set_block_size(256u);
          set_warp_size(32u);
          Shared<uint> warp_sums{8u};
          Shared<uint> running_base{1u};
          const auto tid = thread_x();
          const auto warp_id = tid >> 5u;
          const auto lane_id = tid & 31u;
          $if(tid == 0u) { running_base[0u] = 0u; };
          sync_block();

          UInt base = 0u;
          $while(base < total_buckets) {
            const auto idx0 = base + tid * 4u;
            const auto idx1 = idx0 + 1u;
            const auto idx2 = idx0 + 2u;
            const auto idx3 = idx0 + 3u;
            UInt v0 = 0u;
            UInt v1 = 0u;
            UInt v2 = 0u;
            UInt v3 = 0u;
            $if(idx0 < total_buckets) { v0 = hist.read(idx0); };
            $if(idx1 < total_buckets) { v1 = hist.read(idx1); };
            $if(idx2 < total_buckets) { v2 = hist.read(idx2); };
            $if(idx3 < total_buckets) { v3 = hist.read(idx3); };
            const auto s1 = v0;
            const auto s2 = s1 + v1;
            const auto s3 = s2 + v2;
            const auto t_sum = s3 + v3;
            const auto w_prefix = warp_prefix_sum(t_sum);
            const auto w_total = warp_active_sum(t_sum);
            $if(lane_id == 0u) { warp_sums[warp_id] = w_total; };
            sync_block();
            $if(warp_id == 0u) {
              UInt ws = 0u;
              $if(lane_id < 8u) { ws = warp_sums[lane_id]; };
              const auto wp = warp_prefix_sum(ws);
              const auto wt = warp_active_sum(ws);
              const auto prev_base = running_base[0u];
              $if(lane_id < 8u) { warp_sums[lane_id] = prev_base + wp; };
              $if(lane_id == 0u) { running_base[0u] = prev_base + wt; };
            };
            sync_block();
            const auto t_base = warp_sums[warp_id] + w_prefix;
            $if(idx0 < total_buckets) { hist.write(idx0, t_base); };
            $if(idx1 < total_buckets) { hist.write(idx1, t_base + s1); };
            $if(idx2 < total_buckets) { hist.write(idx2, t_base + s2); };
            $if(idx3 < total_buckets) { hist.write(idx3, t_base + s3); };
            sync_block();
            base += 1024u;
          };
        };

        Kernel1D fused_scatter_kernel =
            [read_shader, layout = context.frame_layout,
             soa = context.global_memory_soa](
                ByteBufferVar storage, BufferUInt block_offsets,
                BufferUInt indices_out, UInt capacity, UInt source_queue,
                UInt scan_count, UInt partition_eff,
                UInt subblocks_per_part, UInt shader_range) noexcept {
              set_block_size(256u);
              set_warp_size(32u);
              Shared<uint> local_offsets{max_fused_shader_range};
              const auto tid = thread_x();
              const auto b = block_x();
              const auto p = b / subblocks_per_part;
              const auto s = b - p * subblocks_per_part;
              $if(tid < shader_range) {
                const auto hist_idx =
                    (p * shader_range + tid) * subblocks_per_part + s;
                local_offsets[tid] = block_offsets.read(hist_idx);
              };
              sync_block();

              const auto part_begin = min(p * partition_eff, scan_count);
              const auto part_end = min(part_begin + partition_eff, scan_count);
              const auto sub_size =
                  (partition_eff + subblocks_per_part - 1u) / subblocks_per_part;
              const auto sub_begin = min(part_begin + s * sub_size, part_end);
              const auto sub_end = min(sub_begin + sub_size, part_end);

              UInt base_pos = sub_begin;
              $while(base_pos < sub_end) {
                const auto pos0 = base_pos + tid;
                const auto pos1 = pos0 + 256u;
                const auto pos2 = pos0 + 512u;
                const auto pos3 = pos0 + 768u;
                const UInt miss_tok = ~source_queue;
                UInt tok0 = miss_tok;
                UInt tok1 = miss_tok;
                UInt tok2 = miss_tok;
                UInt tok3 = miss_tok;
                $if(pos0 < sub_end) {
                  tok0 = coro_frame_read_field<uint>(storage, pos0, capacity,
                                                     layout, soa, 6u);
                };
                $if(pos1 < sub_end) {
                  tok1 = coro_frame_read_field<uint>(storage, pos1, capacity,
                                                     layout, soa, 6u);
                };
                $if(pos2 < sub_end) {
                  tok2 = coro_frame_read_field<uint>(storage, pos2, capacity,
                                                     layout, soa, 6u);
                };
                $if(pos3 < sub_end) {
                  tok3 = coro_frame_read_field<uint>(storage, pos3, capacity,
                                                     layout, soa, 6u);
                };
                const auto scatter_match = [&](UInt pos, UInt tok) noexcept {
                  $if(tok == source_queue) {
                    const auto s_key = min(read_shader(pos, storage, capacity),
                                           shader_range - 1u);
                    UInt dst = 0u;
                    $if(warp_active_all_equal(s_key)) {
                      const auto total = warp_active_count_bits(true);
                      const auto prefix = warp_prefix_count_bits(true);
                      UInt base_dst = 0u;
                      $if(prefix == 0u) {
                        base_dst = local_offsets.atomic(s_key).fetch_add(total);
                      };
                      dst = warp_read_first_active_lane(base_dst) + prefix;
                    }
                    $else { dst = local_offsets.atomic(s_key).fetch_add(1u); };
                    indices_out.write(dst, pos);
                  };
                };
                scatter_match(pos0, tok0);
                scatter_match(pos1, tok1);
                scatter_match(pos2, tok2);
                scatter_match(pos3, tok3);
                base_pos += 1024u;
              };
            };

        psycles::parallel_for(4u, [&](std::size_t idx) {
          switch (idx) {
            case 0u:
              compiled->bucket =
                  context.device.compile(fused_bucket_kernel, context.shader_option);
              break;
            case 1u:
              compiled->prefix =
                  context.device.compile(fused_prefix_kernel, context.shader_option);
              break;
            case 2u:
              compiled->scatter =
                  context.device.compile(fused_scatter_kernel, context.shader_option);
              break;
            default:
              compiled->single_block_sort = context.device.compile(
                  fused_single_block_sort_kernel, context.shader_option);
              break;
          }
        });
        compiled->single_block_sort.set_name(
            "psycles_fused_surface_single_block_sort");
        compiled->bucket.set_name("psycles_fused_surface_bucket");
        compiled->prefix.set_name("psycles_fused_surface_prefix");
        compiled->scatter.set_name("psycles_fused_surface_scatter");
        _fused_shaders = compiled;
        if (shader_cache != nullptr && cache_key.has_value()) {
          std::lock_guard lock{shader_cache->mtx};
          shader_cache->entries.emplace_back(std::move(*cache_key),
                                             std::move(compiled));
        }
      }
    } else {
      _keys[0] =
          context.device.create_buffer<luisa::uint>(context.frame_capacity);
      auto high_bit = 0u;
      while (_composite_range > (1u << (high_bit + 1u))) {
        ++high_bit;
      }
      auto use_bucket = _composite_range <= radix_sort::hist_block_size;
      const auto max_shared = context.device.compute_max_shared_memory_size();
      const auto max_onesweep_warps =
          radix_sort::onesweep_block_size / radix_sort::warp_size;
      const auto max_single_pass_digit =
          max_shared >= 4096u
              ? static_cast<unsigned>((max_shared / sizeof(luisa::uint) - 1u) /
                                      (max_onesweep_warps + 1u))
              : 512u;
      const auto single_pass_onesweep =
          !use_bucket && _composite_range <= max_single_pass_digit;
      auto digit = use_bucket
                       ? _composite_range
                       : (single_pass_onesweep ? _composite_range
                                               : radix_sort::hist_block_size);
      if (!use_bucket && !single_pass_onesweep) {
        _keys[1] =
            context.device.create_buffer<luisa::uint>(context.frame_capacity);
      }
      _temporary =
          radix_sort::temp_storage{context.device, context.frame_capacity, digit};
      Callable<luisa::uint(luisa::uint, Buffer<luisa::uint>, ByteBuffer,
                           luisa::uint, luisa::uint, luisa::uint)>
          index = [](UInt slot, BufferUInt indices, ByteBufferVar, UInt, UInt,
                     UInt) { return indices.read(slot); };
      _sort = radix_sort::instance<Buffer<luisa::uint>, ByteBuffer, luisa::uint,
                                   luisa::uint, luisa::uint>{
          context.device,
          context.frame_capacity,
          _temporary,
          &*_key_callable,
          &index,
          &*_key_callable,
          use_bucket ? 1u : 0u,
          digit,
          0u,
          high_bit,
          "psycles_surface_sort",
          false};
    }
    LUISA_INFO("Psycles surface-sort Handler: boundary={} shader_keys={} "
               "partition_size={} composite_keys={} fused_partition={}.",
               stage.boundary->index, range, partition, composite_range,
               _use_fused_partition_sort);
  }
  void _ensure_fused_buffers() noexcept {
    if (_fused_buffers_allocated) {
      return;
    }
    _indices = _device->create_buffer<luisa::uint>(_frame_capacity);
    const auto max_total_buckets =
        std::max(_max_total_blocks * _range, _composite_range);
    _bucket_hist = _device->create_buffer<luisa::uint>(max_total_buckets);
    _bucket_offsets = _device->create_buffer<luisa::uint>(_composite_range);
    _fused_buffers_allocated = true;
  }

  void _ensure_fallback_shaders() noexcept {
    if (_fallback_shaders_compiled) {
      return;
    }
    using namespace luisa::compute;
    const auto buckets = _composite_range;
    Kernel1D clear_kernel = [](BufferUInt hist, UInt n) noexcept {
      $if(dispatch_x() < n) { hist.write(dispatch_x(), 0u); };
    };
    Kernel1D hist_kernel =
        [key = *_key_callable, buckets](
            BufferUInt indices, ByteBufferVar storage, BufferUInt keys_out,
            BufferUInt global_hist, UInt count, UInt capacity, UInt range_arg,
            UInt partition_arg) noexcept {
          set_block_size(128u);
          set_warp_size(32u);
          const auto slot = dispatch_x();
          $if(slot >= count) { $return(); };
          auto k =
              key(slot, indices, storage, capacity, range_arg, partition_arg);
          k = min(k, buckets - 1u);
          keys_out.write(slot, k);
          global_hist.atomic(k).fetch_add(1u);
        };
    Kernel1D scatter_kernel =
        [](BufferUInt indices_in, BufferUInt keys_in, BufferUInt indices_out,
           BufferUInt global_offsets, UInt count) noexcept {
          set_block_size(128u);
          set_warp_size(32u);
          const auto slot = dispatch_x();
          $if(slot >= count) { $return(); };
          const auto k = keys_in.read(slot);
          const auto rank = global_offsets.atomic(k).fetch_add(1u);
          indices_out.write(rank, indices_in.read(slot));
        };
    psycles::parallel_for(3u, [&](std::size_t idx) {
      switch (idx) {
        case 0u:
          _bucket_clear_shader = _device->compile(clear_kernel, _shader_option);
          break;
        case 1u:
          _bucket_hist_shader = _device->compile(hist_kernel, _shader_option);
          break;
        default:
          _bucket_scatter_shader =
              _device->compile(scatter_kernel, _shader_option);
          break;
      }
    });
    _bucket_clear_shader.set_name("psycles_surface_bucket_clear");
    _bucket_hist_shader.set_name("psycles_surface_bucket_hist");
    _bucket_scatter_shader.set_name("psycles_surface_bucket_scatter");
    _fallback_shaders_compiled = true;
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return surface_sort_schema;
  }
  [[nodiscard]] luisa::compute::coro::WavefrontCoroExtensionExecution
  execution() const noexcept override {
    return luisa::compute::coro::WavefrontCoroExtensionExecution::before_resume;
  }
  [[nodiscard]] luisa::string_view batching_identity() const noexcept override {
    // Key policy is determined entirely by the normalized annotation and the
    // scheduler's shared capacity/layout. Any compatible instance can sort
    // the union: neither predecessor identity nor per-boundary calls matter.
    // Luisa still proves identical typed bindings and resident certificates.
    return "org.psycles.cycles.surface_sort.batch.v1";
  }
  [[nodiscard]] bool supports_command_list() const noexcept override {
    return true;
  }
  [[nodiscard]] bool gathers_selected_frames() const noexcept override {
    return _use_fused_partition_sort;
  }
  [[nodiscard]] bool gathers_selected_frames_for(
      luisa::uint frame_count, luisa::uint scan_count) const noexcept override {
    if (!_use_fused_partition_sort) {
      return false;
    }
    if (frame_count <= _min_sort_frames && scan_count > single_block_max_scan) {
      return false;
    }
    return true;
  }
  [[nodiscard]] luisa::compute::BufferView<luisa::uint>
  dispatch_queue(const Dispatch &context) noexcept override {
    if (!context.ungathered_scan &&
        context.scan_count > single_block_max_scan &&
        context.frame_count <= _min_sort_frames) {
      return context.frame_indices;
    }
    if (_use_fused_partition_sort) {
      if (context.ungathered_scan) {
        if (context.frame_count == 0u || context.scan_count == 0u) {
          return luisa::compute::BufferView<luisa::uint>{};
        }
        _ensure_fused_buffers();
        auto out_indices = _indices.view().subview(0u, context.frame_count);
        const auto scan_count = context.scan_count;
        const auto partition_eff =
            _partition == 0u ? scan_count : _partition;
        const auto active_partitions =
            std::max(1u, (scan_count + partition_eff - 1u) / partition_eff);
        if (active_partitions == 1u && scan_count <= single_block_max_scan) {
          auto encode_single = [&](luisa::compute::CommandList &cmds) noexcept {
            cmds << _fused_shaders->single_block_sort(
                        context.frame_buffer, out_indices,
                        context.frame_capacity, context.source_queue_token,
                        scan_count, _range)
                        .dispatch(256u);
          };
          if (context.command_list != nullptr) {
            encode_single(*context.command_list);
          } else {
            auto cmds = luisa::compute::CommandList::create(1u);
            encode_single(cmds);
            context.stream << cmds.commit();
          }
          return out_indices;
        }
        const auto max_subblocks_by_buf =
            std::max(1u, _max_total_blocks / active_partitions);
        const auto max_subblocks_by_work =
            std::max(1u, scan_count / (active_partitions * 2048u));
        const auto subblocks = std::clamp(
            std::min(max_subblocks_by_buf, max_subblocks_by_work), 1u, 64u);
        const auto total_blocks = active_partitions * subblocks;
        const auto total_buckets = active_partitions * _range * subblocks;
        auto encode = [&](luisa::compute::CommandList &cmds) noexcept {
          cmds << _fused_shaders->bucket(
                      context.frame_buffer, _bucket_hist.view(),
                      context.frame_capacity, context.source_queue_token,
                      scan_count, partition_eff, subblocks, _range)
                      .dispatch(total_blocks * 256u)
               << _fused_shaders->prefix(_bucket_hist.view(), total_buckets)
                      .dispatch(256u)
               << _fused_shaders->scatter(
                      context.frame_buffer, _bucket_hist.view(), out_indices,
                      context.frame_capacity, context.source_queue_token,
                      scan_count, partition_eff, subblocks, _range)
                      .dispatch(total_blocks * 256u);
        };
        if (context.command_list != nullptr) {
          encode(*context.command_list);
        } else {
          auto cmds = luisa::compute::CommandList::create(3u);
          encode(cmds);
          context.stream << cmds.commit();
        }
        return out_indices;
      }
      if (context.frame_count <= 1u || _composite_range <= 1u) {
        return context.frame_indices;
      }
      _ensure_fused_buffers();
      _ensure_fallback_shaders();
      if (!_keys[0]) {
        _keys[0] = _device->create_buffer<luisa::uint>(_frame_capacity);
      }
      auto out_indices = _indices.view().subview(0u, context.frame_count);
      auto hist_view = _bucket_offsets.view().subview(0u, _composite_range);
      auto encode_fallback = [&](luisa::compute::CommandList &cmds) noexcept {
        cmds << _bucket_clear_shader(hist_view, _composite_range)
                    .dispatch(_composite_range)
             << _bucket_hist_shader(context.frame_indices, context.frame_buffer,
                                    _keys[0].view(), hist_view,
                                    context.frame_count, context.frame_capacity,
                                    _range, _partition)
                    .dispatch(context.frame_count)
             << _fused_shaders->prefix(hist_view, _composite_range).dispatch(256u)
             << _bucket_scatter_shader(context.frame_indices, _keys[0].view(),
                                       out_indices, hist_view,
                                       context.frame_count)
                    .dispatch(context.frame_count);
      };
      if (context.command_list != nullptr) {
        encode_fallback(*context.command_list);
      } else {
        auto cmds = luisa::compute::CommandList::create(4u);
        encode_fallback(cmds);
        context.stream << cmds.commit();
      }
      return out_indices;
    }
    if (context.frame_count <= 1u || _composite_range <= 1u) {
      return context.frame_indices;
    }
    luisa::compute::BufferView<luisa::uint> keys[]{
        _keys[0].view(), _keys[1] ? _keys[1].view() : _keys[0].view()};
    luisa::compute::BufferView<luisa::uint> indices[]{
        context.frame_indices, _indices.view().subview(0u, context.frame_count)};
    auto result = context.command_list != nullptr
                      ? _sort.encode_sort_switch(
                            *context.command_list, keys, indices, context.frame_count,
                            context.frame_indices, context.frame_buffer,
                            context.frame_capacity, _range, _partition)
                      : _sort.sort_switch(
                            context.stream, keys, indices, context.frame_count,
                            context.frame_indices, context.frame_buffer,
                            context.frame_capacity, _range, _partition);
    return indices[result];
  }
};

class SurfaceQueueKeyStageImpl final : public SurfaceQueueKeyStage {

private:
  ScenePrimitiveStagePlan _plan;
  std::shared_ptr<const PrimitiveMaterialComponent> _material{
      make_primitive_material_component()};

public:
  explicit SurfaceQueueKeyStageImpl(ScenePrimitiveStagePlan plan) noexcept
      : _plan{plan} {}

  [[nodiscard]] UInt
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       const Var<luisa::compute::CommittedHit> &hit) const noexcept override {
    const UInt instance_id = hit->inst;
    Var<InstanceGpu> instance = scene->instance_buffer->read(instance_id);

    // Cycles intersection_get_shader_from_isect_prim(): the native
    // interpreter's identity is the shader-table index, not the deduped
    // graph topology. Material overrides are already resolved in this
    // geometry image.
    LUISA_ASSERT(scene->cycles_svm && scene->cycles_svm->geometry,
                 "Native surface sorting requires finalized Cycles geometry.");
    const auto &native = *scene->cycles_svm->geometry;
    UInt shader = 0u;
    const auto triangle_shader = [&] noexcept {
      shader = native.triangle_shader_buffer->read(
          instance.cycles_primitive_offset + hit->prim);
    };
    const auto curve_shader = [&] noexcept {
      Var<GeometryGpu> geometry =
          scene->geometry_buffer->read(instance.geometry_index);
      const auto segment =
          scene->heap->buffer<CurveSegmentGpu>(geometry.bindless_base).read(hit->prim);
      shader = native.curve_buffer->read(segment.cycles_curve_index)
                   .shader_id.cast<std::uint32_t>();
    };
    if (_plan.mixed()) {
      $if(hit->is_procedural()) { curve_shader(); }
      $else { triangle_shader(); };
    } else if (_plan.curves) {
      curve_shader();
    } else {
      triangle_shader();
    }
    return shader & cycles_shader_identity::shader_mask;
  }
};

[[nodiscard]] luisa::unique_ptr<
    luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler>
make_surface_sort_handler_with_cache(
    luisa::compute::coro::WavefrontCoroExtensionPrepareContext &context,
    const luisa::compute::coro::WavefrontCoroExtensionStage &stage,
    std::shared_ptr<SurfaceSortShaderCache> shader_cache) noexcept {
  using namespace luisa::compute;
  namespace sort = luisa::compute::coro::radix_sort;
  if (stage.extension->schema() != surface_sort_schema) {
    return nullptr;
  }
  LUISA_ASSERT(stage.extension->version() == 1u,
               "Unsupported Psycles surface-sort schema version.");
  std::uint64_t range = 0u;
  bool native = false, has_native = false;
  for (const auto &attribute : stage.extension->attributes()) {
    if (attribute.name == "shader_count") {
      range = luisa::get<std::uint64_t>(attribute.value);
    }
    if (attribute.name == "native_cycles") {
      native = luisa::get<bool>(attribute.value);
      has_native = true;
    }
  }
  LUISA_ASSERT(range != 0u && range <= std::numeric_limits<unsigned>::max() &&
                   has_native &&
                   stage.binding("shader").type() == Type::of<luisa::uint>() &&
                   stage.binding("shader").readable(),
               "Invalid Psycles surface-sort typed binding or attributes.");
  auto shader_count = static_cast<unsigned>(range);
  auto partition =
      native ? cycles_surface_sort_partition_size(context.frame_capacity, shader_count)
             : 0u;
  auto partitions = partition == 0u
                        ? 1u
                        : context.frame_capacity / partition +
                              unsigned(context.frame_capacity % partition != 0u);
  auto composite = range * partitions;
  LUISA_ASSERT(composite <= std::numeric_limits<unsigned>::max(),
               "Surface-sort composite key overflows uint.");
  if (composite > sort::hist_block_size &&
      context.device.compute_warp_size() != sort::warp_size) {
    LUISA_WARNING(
        "Psycles surface-sort radix path needs {}-lane subgroups; annotation ignored.",
        sort::warp_size);
    return nullptr;
  }
  return luisa::make_unique<SurfaceSortHandler>(
      context, stage, shader_count, partition, static_cast<unsigned>(composite),
      std::move(shader_cache));
}

} // namespace

std::unique_ptr<SurfaceQueueKeyStage>
make_surface_queue_key_stage(ScenePrimitiveStagePlan plan) {
  return std::make_unique<SurfaceQueueKeyStageImpl>(plan);
}

std::uint32_t surface_queue_key_range(const LuisaSceneData &scene) noexcept {
  LUISA_ASSERT(scene.cycles_svm,
               "Native surface sorting requires a Cycles shader table.");
  const auto count = scene.cycles_svm->compilation.kernel_shaders.size();
  LUISA_ASSERT(count <= std::numeric_limits<std::uint32_t>::max(),
               "Surface queue key range {} exceeds the uint32 scheduler ABI.", count);
  return static_cast<std::uint32_t>(count);
}

luisa::compute::CoroSuspendExtensionPtr
make_surface_sort_annotation(UInt shader, std::uint32_t shader_count,
                             bool native_cycles) noexcept {
  auto annotation = luisa::compute::coro_annotation(surface_sort_schema, 1u);
  annotation.read("shader", shader)
      .attribute("shader_count", shader_count)
      .attribute("native_cycles", native_cycles);
  return std::move(annotation).build();
}

luisa::unique_ptr<luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler>
make_surface_sort_handler(
    luisa::compute::coro::WavefrontCoroExtensionPrepareContext &context,
    const luisa::compute::coro::WavefrontCoroExtensionStage &stage) noexcept {
  return make_surface_sort_handler_with_cache(context, stage, nullptr);
}

std::function<luisa::unique_ptr<
    luisa::compute::coro::WavefrontCoroSchedulerExtensionHandler>(
    luisa::compute::coro::WavefrontCoroExtensionPrepareContext &,
    const luisa::compute::coro::WavefrontCoroExtensionStage &)>
make_surface_sort_handler_factory() noexcept {
  auto shader_cache = std::make_shared<SurfaceSortShaderCache>();
  return [shader_cache = std::move(shader_cache)](
             luisa::compute::coro::WavefrontCoroExtensionPrepareContext &context,
             const luisa::compute::coro::WavefrontCoroExtensionStage &stage) noexcept {
    return make_surface_sort_handler_with_cache(context, stage, shader_cache);
  };
}

} // namespace psycles::luisa_backend::detail

#include "path_kernel_surface_queue.h"

#include "cycles_shader_identity.h"
#include "cycles_surface_sort.h"
#include "path_kernel_primitive_material.h"

#include <psycles/core/parallel.h>
#include <psycles/luisa/cycles_svm.h>

#include <limits>
#include <utility>

#include <luisa/coro/radix_sort.h>

namespace psycles::luisa_backend::detail {
namespace {

constexpr auto surface_sort_schema = "org.psycles.cycles.surface_sort";

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
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint,
                           luisa::uint, luisa::uint>
      _fused_single_block_sort_shader;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint,
                           luisa::uint, luisa::uint, luisa::uint, luisa::uint>
      _fused_bucket_shader;
  luisa::compute::Shader1D<luisa::compute::Buffer<luisa::uint>, luisa::uint>
      _fused_prefix_shader;
  luisa::compute::Shader1D<luisa::compute::ByteBuffer,
                           luisa::compute::Buffer<luisa::uint>,
                           luisa::compute::Buffer<luisa::uint>, luisa::uint,
                           luisa::uint, luisa::uint, luisa::uint, luisa::uint>
      _fused_scatter_shader;
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
  bool _use_fused_partition_sort{false};
  bool _warp32{false};
  unsigned _min_sort_frames{512u};
  luisa::compute::Device *_device{nullptr};
  unsigned _frame_capacity{0u};
  luisa::compute::coro::radix_sort::instance<luisa::compute::Buffer<luisa::uint>,
                                             luisa::compute::ByteBuffer, luisa::uint,
                                             luisa::uint, luisa::uint>
      _sort;

public:
  SurfaceSortHandler(Prepare &context, const Stage &stage, unsigned range,
                     unsigned partition, unsigned composite_range)
      : _range{std::max(range, 1u)}, _partition{partition},
        _composite_range{std::max(composite_range, 1u)},
        _warp32{context.device.compute_warp_size() == 32u},
        _device{&context.device},
        _frame_capacity{context.frame_capacity} {
    using namespace luisa::compute;
    using namespace luisa::compute::coro;
    _use_fused_partition_sort =
        _warp32 && _range <= max_fused_shader_range &&
        _composite_range <= max_fused_composite_range;
    if (const char *raw = std::getenv("PSYCLES_SURFACE_SORT_MIN_FRAMES");
        raw != nullptr && *raw != '\0') {
      _min_sort_frames = static_cast<unsigned>(std::strtoul(raw, nullptr, 10));
    }
    _indices = context.device.create_buffer<luisa::uint>(context.frame_capacity);
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
    Callable<luisa::uint(luisa::uint, Buffer<luisa::uint>, ByteBuffer, luisa::uint,
                         luisa::uint, luisa::uint)>
        key = [read_shader](UInt slot, BufferUInt indices,
                            ByteBufferVar storage, UInt capacity,
                            UInt range_arg, UInt partition_arg) {
          auto index = indices.read(slot);
          auto shader = read_shader(index, storage, capacity);
          UInt locality = 0u;
          $if(partition_arg != 0u) { locality = index / partition_arg; };
          return shader + locality * range_arg;
        };

    if (_use_fused_partition_sort) {
      const auto max_partitions =
          _partition == 0u
              ? 1u
              : std::max(1u, (context.frame_capacity + _partition - 1u) /
                                 _partition);
      const unsigned target_blocks =
          context.frame_capacity >= 65536u
              ? 64u
              : (context.frame_capacity >= 4096u ? 16u : 1u);
      _subblocks_per_partition =
          std::clamp(target_blocks / max_partitions, 1u, 64u);
      const auto max_total_buckets =
          std::max(max_partitions * _range * _subblocks_per_partition,
                   _composite_range);
      _bucket_hist =
          context.device.create_buffer<luisa::uint>(max_total_buckets);
      _bucket_offsets =
          context.device.create_buffer<luisa::uint>(_composite_range);

      const auto shader_range = _range;
      Kernel1D fused_single_block_sort_kernel =
          [read_shader, shader_range, layout = context.frame_layout,
           soa = context.global_memory_soa](
              ByteBufferVar storage, BufferUInt indices_out, UInt capacity,
              UInt source_queue, UInt scan_count) noexcept {
            set_block_size(256u);
            set_warp_size(32u);
            Shared<uint> hist{shader_range};
            Shared<uint> warp_sums{8u};
            const auto tid = thread_x();
            const auto warp_id = tid >> 5u;
            const auto lane_id = tid & 31u;
            if (shader_range <= 256u) {
              $if(tid < shader_range) { hist[tid] = 0u; };
            }
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
                  const auto s_key =
                      min(read_shader(pos, storage, capacity), shader_range - 1u);
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
            $if(tid < shader_range) { hist[tid] = warp_sums[warp_id] + w_prefix; };
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
          [read_shader, shader_range, layout = context.frame_layout,
           soa = context.global_memory_soa](
              ByteBufferVar storage, BufferUInt block_hist, UInt capacity,
              UInt source_queue, UInt scan_count, UInt partition_eff,
              UInt subblocks_per_part) noexcept {
            set_block_size(256u);
            set_warp_size(32u);
            Shared<uint> local_hist{shader_range};
            const auto tid = thread_x();
            if (shader_range <= 256u) {
              $if(tid < shader_range) { local_hist[tid] = 0u; };
            }
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
              const auto pos = base_pos + tid;
              $if(pos < sub_end) {
                const auto tok = coro_frame_read_field<uint>(
                    storage, pos, capacity, layout, soa, 6u);
                $if(tok == source_queue) {
                  const auto s_key =
                      min(read_shader(pos, storage, capacity), shader_range - 1u);
                  $if(warp_active_all_equal(s_key)) {
                    const auto total = warp_active_count_bits(true);
                    $if(warp_prefix_count_bits(true) == 0u) {
                      local_hist.atomic(s_key).fetch_add(total);
                    };
                  }
                  $else {
                    local_hist.atomic(s_key).fetch_add(1u);
                  };
                };
              };
              base_pos += 256u;
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
          const auto idx = base + tid;
          UInt val = 0u;
          $if(idx < total_buckets) { val = hist.read(idx); };
          const auto w_prefix = warp_prefix_sum(val);
          const auto w_total = warp_active_sum(val);
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
          $if(idx < total_buckets) {
            hist.write(idx, warp_sums[warp_id] + w_prefix);
          };
          sync_block();
          base += 256u;
        };
      };

      Kernel1D fused_scatter_kernel =
          [read_shader, shader_range, layout = context.frame_layout,
           soa = context.global_memory_soa](
              ByteBufferVar storage, BufferUInt block_offsets,
              BufferUInt indices_out, UInt capacity, UInt source_queue,
              UInt scan_count, UInt partition_eff,
              UInt subblocks_per_part) noexcept {
            set_block_size(256u);
            set_warp_size(32u);
            Shared<uint> local_offsets{shader_range};
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
              const auto pos = base_pos + tid;
              $if(pos < sub_end) {
                const auto tok = coro_frame_read_field<uint>(
                    storage, pos, capacity, layout, soa, 6u);
                $if(tok == source_queue) {
                  const auto s_key =
                      min(read_shader(pos, storage, capacity), shader_range - 1u);
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
                  $else {
                    dst = local_offsets.atomic(s_key).fetch_add(1u);
                  };
                  indices_out.write(dst, pos);
                };
              };
              base_pos += 256u;
            };
          };

      // Gathered fallback shaders for non-incremental or multi-member queues.
      const auto buckets = _composite_range;
      Kernel1D clear_kernel = [](BufferUInt hist, UInt n) noexcept {
        $if(dispatch_x() < n) { hist.write(dispatch_x(), 0u); };
      };
      Kernel1D hist_kernel =
          [key, buckets](BufferUInt indices, ByteBufferVar storage,
                         BufferUInt keys_out, BufferUInt global_hist,
                         UInt count, UInt capacity, UInt range_arg,
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

      psycles::parallel_for(7u, [&](std::size_t idx) {
        switch (idx) {
          case 0u:
            _fused_bucket_shader =
                context.device.compile(fused_bucket_kernel, context.shader_option);
            break;
          case 1u:
            _fused_prefix_shader =
                context.device.compile(fused_prefix_kernel, context.shader_option);
            break;
          case 2u:
            _fused_scatter_shader =
                context.device.compile(fused_scatter_kernel, context.shader_option);
            break;
          case 3u:
            _bucket_clear_shader =
                context.device.compile(clear_kernel, context.shader_option);
            break;
          case 4u:
            _bucket_hist_shader =
                context.device.compile(hist_kernel, context.shader_option);
            break;
          case 5u:
            _bucket_scatter_shader =
                context.device.compile(scatter_kernel, context.shader_option);
            break;
          default:
            _fused_single_block_sort_shader = context.device.compile(
                fused_single_block_sort_kernel, context.shader_option);
            break;
        }
      });
      _fused_single_block_sort_shader.set_name(
          "psycles_fused_surface_single_block_sort");
      _fused_bucket_shader.set_name("psycles_fused_surface_bucket");
      _fused_prefix_shader.set_name("psycles_fused_surface_prefix");
      _fused_scatter_shader.set_name("psycles_fused_surface_scatter");
      _bucket_clear_shader.set_name("psycles_surface_bucket_clear");
      _bucket_hist_shader.set_name("psycles_surface_bucket_hist");
      _bucket_scatter_shader.set_name("psycles_surface_bucket_scatter");
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
          &key,
          &index,
          &key,
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
          return _indices.view().subview(0u, 0u);
        }
        auto out_indices = _indices.view().subview(0u, context.frame_count);
        const auto scan_count = context.scan_count;
        const auto partition_eff =
            _partition == 0u ? scan_count : _partition;
        const auto active_partitions =
            std::max(1u, (scan_count + partition_eff - 1u) / partition_eff);
        if (active_partitions == 1u && scan_count <= single_block_max_scan) {
          auto encode_single = [&](luisa::compute::CommandList &cmds) noexcept {
            cmds << _fused_single_block_sort_shader(
                        context.frame_buffer, out_indices,
                        context.frame_capacity, context.source_queue_token,
                        scan_count)
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
        const auto subblocks = _subblocks_per_partition;
        const auto total_blocks = active_partitions * subblocks;
        const auto total_buckets = active_partitions * _range * subblocks;
        auto encode = [&](luisa::compute::CommandList &cmds) noexcept {
          cmds << _fused_bucket_shader(
                      context.frame_buffer, _bucket_hist.view(),
                      context.frame_capacity, context.source_queue_token,
                      scan_count, partition_eff, subblocks)
                      .dispatch(total_blocks * 256u)
               << _fused_prefix_shader(_bucket_hist.view(), total_buckets)
                      .dispatch(256u)
               << _fused_scatter_shader(
                      context.frame_buffer, _bucket_hist.view(), out_indices,
                      context.frame_capacity, context.source_queue_token,
                      scan_count, partition_eff, subblocks)
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
             << _fused_prefix_shader(hist_view, _composite_range).dispatch(256u)
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
  return luisa::make_unique<SurfaceSortHandler>(context, stage, shader_count, partition,
                                                static_cast<unsigned>(composite));
}

} // namespace psycles::luisa_backend::detail

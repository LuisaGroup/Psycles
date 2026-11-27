#include "path_kernel_direct_light_queue.h"

#include "cycles_wavefront_policy.h"
#include "path_kernel_film.h"
#include "path_kernel_shadow_storage.h"
#include "path_kernel_surface_buckets.h"
#include "path_kernel_transitions.h"
#include "virtual_texture_cache.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/core/logging.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/soa.h>

namespace psycles::luisa_backend::detail {
namespace {

using DirectLightConsumerShader = luisa::compute::Shader1D<
    luisa::compute::Buffer<luisa::float4>, luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::float4>, RenderKernelParameters, std::uint32_t,
    std::uint32_t>;
using DirectLightConsumerKernel = luisa::compute::Kernel1D<
    luisa::compute::Buffer<luisa::float4>, luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::float4>, RenderKernelParameters, std::uint32_t,
    std::uint32_t>;

class DirectLightTaskQueue final : public RenderAuxiliaryWork,
                                   public DirectLightTaskSink {

private:
  luisa::compute::ByteBuffer _task_buffer;
  std::shared_ptr<const ShadowIntersectionBatchStorage> _shadow_storage;
  luisa::compute::Buffer<luisa::uint> _tokens;
  luisa::compute::Buffer<luisa::uint> _indices;
  std::array<luisa::compute::Buffer<luisa::uint>, 2u> _chained_indices;
  luisa::compute::Buffer<luisa::uint> _scratch_count;
  luisa::compute::Buffer<luisa::uint> _relocation_counts;
  // Counts 0..2 are live stages; count 3 is the append allocation extent;
  // count 4 signals pending virtual texture cache misses from stage 0 or 2.
  luisa::compute::Buffer<luisa::uint> _count;
  std::array<DirectLightConsumerShader, 3u> _consumers;
  luisa::compute::Shader1D<luisa::uint> _initialize;
  luisa::compute::Shader1D<luisa::uint> _clear_count_slot;
  luisa::compute::Shader1D<luisa::uint, luisa::uint, luisa::uint, luisa::uint> _gather;
  luisa::compute::Shader1D<luisa::uint, luisa::uint, luisa::uint>
      _set_extent_and_clear_reloc;
  luisa::compute::Shader1D<luisa::uint, luisa::uint> _collect_relocations;
  luisa::compute::Shader1D<luisa::uint, luisa::uint> _compact;
  luisa::vector<luisa::compute::coro::WavefrontCoroAuxiliaryProducer> _producers;
  std::shared_ptr<VirtualTextureCache> _vt_cache;
  std::uint32_t _capacity{};
  std::uint32_t _execution_block_size{};
  std::uint32_t _warp_size{};
  std::uint32_t _scratch_slot{0u};
  std::uint32_t _chain_count_slot{0u};
  std::uint32_t _chain_buf_slot{0u};
  bool _fuse_unoccluded_shadow{false};
  bool _may_have_shade_shadow{true};
  bool _may_continue_shadow{true};
  bool _has_fallback_light_shader{false};
  bool _spectral{false};
  std::array<std::uint32_t, 5u> _host_counts{};

  static constexpr std::uint32_t k_stride_ray_origin_tmin = 0u;
  static constexpr std::uint32_t k_stride_ray_dir_tmax = 16u;
  static constexpr std::uint32_t k_stride_ray_ids = 32u;
  static constexpr std::uint32_t k_stride_throughput_and_term = 48u;
  static constexpr std::uint32_t k_stride_diffuse_glossy_0 = 64u;
  static constexpr std::uint32_t k_stride_film_meta = 80u;
  static constexpr std::uint32_t k_stride_diff_time_flags = 96u;
  static constexpr std::uint32_t k_stride_rng_and_depths = 112u;
  static constexpr std::uint32_t k_stride_nee_throughput = 128u;
  static constexpr std::uint32_t k_stride_fallback_light = 144u;

  [[nodiscard]] std::uint32_t stride_after_16() const noexcept {
    return _has_fallback_light_shader ? 160u : 144u;
  }
  [[nodiscard]] std::uint32_t stride_glossy_yz() const noexcept {
    return stride_after_16();
  }
  [[nodiscard]] std::uint32_t stride_transp_and_vol() const noexcept {
    return stride_after_16() + 8u;
  }
  [[nodiscard]] std::uint32_t stride_transm_and_portal() const noexcept {
    return stride_after_16() + 12u;
  }
  [[nodiscard]] std::uint32_t stride_spectral() const noexcept {
    return stride_after_16() + 16u;
  }
  [[nodiscard]] std::uint32_t total_slot_stride() const noexcept {
    return stride_after_16() + (_spectral ? 32u : 16u);
  }

  [[nodiscard]] static constexpr std::uint32_t
  pack_chain_mode(std::uint32_t in_kind, std::uint32_t out_kind) noexcept {
    return in_kind | (out_kind << 8u);
  }

  template <typename T>
  [[nodiscard]] static UInt slot_byte_offset(Expr<luisa::uint> capacity,
                                             std::uint32_t prefix_stride,
                                             Expr<luisa::uint> slot) noexcept {
    return capacity * prefix_stride + slot * static_cast<luisa::uint>(sizeof(T));
  }

public:
  [[nodiscard]] std::size_t task_storage_bytes() const noexcept {
    return static_cast<std::size_t>(total_slot_stride()) *
           static_cast<std::size_t>(_capacity);
  }

  [[nodiscard]] std::size_t shadow_storage_bytes() const noexcept {
    return _shadow_storage ? _shadow_storage->size_bytes() : 0u;
  }

  DirectLightTaskQueue(
      luisa::compute::Device &device, DirectLightTaskEvaluator evaluator,
      std::uint32_t capacity, std::uint32_t execution_block_size,
      const luisa::compute::ShaderOption &shader_option,
      bool fuse_unoccluded_shadow = false,
      std::shared_ptr<const ShadowIntersectionBatchStorage> shadow_storage = {},
      bool may_have_shade_shadow = true,
      bool may_continue_shadow = true,
      std::span<const std::string_view> producer_continuations = {},
      std::shared_ptr<VirtualTextureCache> vt_cache = {}) noexcept
      : _task_buffer{device.create_byte_buffer(
            static_cast<std::size_t>(
                (evaluator.light_emission == nullptr ? 160u : 144u) +
                (evaluator.spectral_tables != nullptr ? 32u : 16u)) *
            static_cast<std::size_t>(capacity))},
        _shadow_storage{shadow_storage
                            ? std::move(shadow_storage)
                            : std::make_shared<ShadowIntersectionBatchStorage>(
                                  device, may_have_shade_shadow ? capacity : 1u)},
        _tokens{device.create_buffer<luisa::uint>(capacity)},
        _indices{device.create_buffer<luisa::uint>(capacity)},
        _chained_indices{
            fuse_unoccluded_shadow
                ? device.create_buffer<luisa::uint>(capacity)
                : luisa::compute::Buffer<luisa::uint>{},
            (fuse_unoccluded_shadow && may_have_shade_shadow)
                ? device.create_buffer<luisa::uint>(capacity)
                : luisa::compute::Buffer<luisa::uint>{}},
        _scratch_count{device.create_buffer<luisa::uint>(5u)},
        _relocation_counts{device.create_buffer<luisa::uint>(2u)},
        _count{device.create_buffer<luisa::uint>(5u)},
        _producers{},
        _vt_cache{std::move(vt_cache)},
        _capacity{capacity}, _execution_block_size{execution_block_size},
        _warp_size{static_cast<std::uint32_t>(device.compute_warp_size())},
        _fuse_unoccluded_shadow{fuse_unoccluded_shadow},
        _may_have_shade_shadow{may_have_shade_shadow},
        _may_continue_shadow{may_continue_shadow},
        _has_fallback_light_shader{evaluator.light_emission == nullptr},
        _spectral{evaluator.spectral_tables != nullptr} {
    if (producer_continuations.empty()) {
      _producers.push_back(
          {.continuation = path_transition::shade_surface,
           .max_emitted_per_invocation = 1u});
    } else {
      _producers.reserve(producer_continuations.size());
      for (const auto name : producer_continuations) {
        _producers.push_back(
            {.continuation = luisa::string{name},
             .max_emitted_per_invocation = 1u});
      }
    }
    LUISA_ASSERT(execution_block_size != 0u,
                 "Direct-light consumer block size must be positive.");
    luisa::compute::Kernel1D<luisa::uint> initialize = [this](UInt capacity) {
      const auto x = dispatch_x();
      $if(x < capacity) { _tokens->write(x, 0u); };
      $if(x < 5u) {
        _scratch_count->write(x, 0u);
        _count->write(x, 0u);
      };
      $if(x < 2u) {
        _relocation_counts->write(x, 0u);
      };
    };
    luisa::compute::Kernel1D<luisa::uint> clear_count_slot =
        [this](UInt slot) { _count->write(slot, 0u); };
    _clear_count_slot = device.compile(clear_count_slot, shader_option);
    luisa::compute::Kernel1D<luisa::uint, luisa::uint, luisa::uint>
        set_extent_and_clear_reloc =
            [this](UInt new_extent, UInt write_extent, UInt clear_reloc) {
              const auto x = dispatch_x();
              $if((x == 0u) & (write_extent != 0u)) {
                _count->write(3u, new_extent);
              };
              $if((x < 2u) & (clear_reloc != 0u)) {
                _relocation_counts->write(x, 0u);
              };
            };
    // Enumerate exactly the selected stage. Consumers dispatch the queue
    // count, not the whole pool.
    std::optional<luisa::compute::Kernel1D<luisa::uint, luisa::uint, luisa::uint,
                                           luisa::uint>>
        gather_kernel;
    if (device.compute_warp_size() == 32u) {
      gather_kernel.emplace([this](UInt token, UInt extent, UInt scratch_slot,
                                   UInt clear_stage_count) {
        luisa::compute::set_block_size(256u);
        luisa::compute::set_warp_size(32u);
        luisa::compute::Shared<luisa::uint> warp_totals{8u};
        luisa::compute::Shared<luisa::uint> block_base{1u};
        const auto slot = dispatch_x();
        $if(slot == 0u) {
          _scratch_count->write(scratch_slot ^ 1u, 0u);
          $if(clear_stage_count != 0u) {
            _count->write(token - 1u, 0u);
          };
        };
        const auto tid = luisa::compute::thread_x();
        const auto warp_id = tid >> 5u;
        const auto lane_id = tid & 31u;
        Bool active = false;
        $if(slot < extent) {
          active = _tokens->read(slot) == token;
        };
        const auto lane_prefix = luisa::compute::warp_prefix_count_bits(active);
        const auto warp_count = luisa::compute::warp_active_count_bits(active);
        $if(lane_id == 0u) {
          warp_totals[warp_id] = warp_count;
        };
        luisa::compute::sync_block();
        $if(tid == 0u) {
          UInt total = 0u;
          for (unsigned w = 0u; w < 8u; ++w) {
            const auto c = warp_totals[w];
            warp_totals[w] = total;
            total += c;
          }
          $if(total != 0u) {
            block_base[0u] =
                _scratch_count->atomic(scratch_slot).fetch_add(total);
          };
        };
        luisa::compute::sync_block();
        $if(active) {
          const auto index = block_base[0u] + warp_totals[warp_id] + lane_prefix;
          _indices->write(index, slot);
        };
      });
    } else {
      gather_kernel.emplace([this](UInt token, UInt extent, UInt scratch_slot,
                                   UInt clear_stage_count) {
        const auto slot = dispatch_x();
        $if(slot == 0u) {
          _scratch_count->write(scratch_slot ^ 1u, 0u);
          $if(clear_stage_count != 0u) {
            _count->write(token - 1u, 0u);
          };
        };
        $if(slot < extent) {
          $if(_tokens->read(slot) == token) {
            const auto index =
                _scratch_count->atomic(scratch_slot).fetch_add(1u);
            _indices->write(index, slot);
          };
        };
      });
    }
    // Cycles' compact_paths stores source indices at 0 and hole indices at
    // live. If H slots below live are empty, exactly H live sources remain
    // above it. Thus H <= min(live, extent-live), and live+H <= extent:
    // both lists fit in the existing index allocation without overlapping.
    std::optional<luisa::compute::Kernel1D<luisa::uint, luisa::uint>>
        collect_kernel;
    if (device.compute_warp_size() == 32u) {
      collect_kernel.emplace([this](UInt live, UInt extent) {
        luisa::compute::set_block_size(256u);
        luisa::compute::set_warp_size(32u);
        const auto slot = dispatch_x();
        const auto lane_id = luisa::compute::thread_x() & 31u;
        Bool is_hole = false;
        Bool is_src = false;
        $if(slot < extent) {
          const auto token = _tokens->read(slot);
          is_hole = (slot < live) & (token == 0u);
          is_src = (slot >= live) & (token != 0u);
        };
        const auto hole_prefix = luisa::compute::warp_prefix_count_bits(is_hole);
        const auto hole_count = luisa::compute::warp_active_count_bits(is_hole);
        UInt hole_base = 0u;
        $if((lane_id == 0u) & (hole_count != 0u)) {
          hole_base = _relocation_counts->atomic(0u).fetch_add(hole_count);
        };
        hole_base = luisa::compute::warp_read_lane(hole_base, 0u);
        $if(is_hole) {
          _indices->write(live + hole_base + hole_prefix, slot);
        };
        const auto src_prefix = luisa::compute::warp_prefix_count_bits(is_src);
        const auto src_count = luisa::compute::warp_active_count_bits(is_src);
        UInt src_base = 0u;
        $if((lane_id == 0u) & (src_count != 0u)) {
          src_base = _relocation_counts->atomic(1u).fetch_add(src_count);
        };
        src_base = luisa::compute::warp_read_lane(src_base, 0u);
        $if(is_src) {
          _indices->write(src_base + src_prefix, slot);
        };
      });
    } else {
      collect_kernel.emplace([this](UInt live, UInt extent) {
        const auto slot = dispatch_x();
        $if(slot < extent) {
          const auto token = _tokens->read(slot);
          $if(slot < live) {
            $if(token == 0u) {
              const auto index = _relocation_counts->atomic(0u).fetch_add(1u);
              _indices->write(live + index, slot);
            };
          }
          $elif(token != 0u) {
            const auto index = _relocation_counts->atomic(1u).fetch_add(1u);
            _indices->write(index, slot);
          };
        };
      });
    }
    LUISA_INFO("Psycles shadow relocation: name='collect' kernel_{:016x}.",
               (*collect_kernel).function()->function().hash());
    // Only the dense relocation list reaches the payload-copy kernel. The
    // sources [live, extent) and holes [0, live) are disjoint, so the move is
    // in-place without a second payload allocation or inter-thread barriers.
    luisa::compute::Kernel1D<luisa::uint, luisa::uint> compact =
        [this](UInt capacity, UInt live) {
          const auto index = dispatch_x();
          $if(index < _relocation_counts->read(1u)) {
            const auto source = _indices->read(index);
            const auto destination = _indices->read(live + index);
            const auto token = _tokens->read(source);
            const auto buf = Expr<luisa::compute::ByteBuffer>{_task_buffer};
            const auto copy_chunk = [&]<typename T>(std::uint32_t prefix_stride) {
              buf.write(slot_byte_offset<T>(capacity, prefix_stride, destination),
                        buf.read<T>(slot_byte_offset<T>(capacity, prefix_stride, source)));
            };
            copy_chunk.operator()<luisa::float4>(k_stride_ray_origin_tmin);
            copy_chunk.operator()<luisa::float4>(k_stride_ray_dir_tmax);
            copy_chunk.operator()<luisa::uint4>(k_stride_ray_ids);
            copy_chunk.operator()<luisa::float4>(k_stride_throughput_and_term);
            copy_chunk.operator()<luisa::float4>(k_stride_diffuse_glossy_0);
            copy_chunk.operator()<luisa::uint4>(k_stride_film_meta);
            copy_chunk.operator()<luisa::float2>(stride_glossy_yz());
            copy_chunk.operator()<luisa::uint>(stride_transp_and_vol());
            const auto copy_shader_context = [&] {
              copy_chunk.operator()<luisa::float4>(k_stride_diff_time_flags);
              copy_chunk.operator()<luisa::uint4>(k_stride_rng_and_depths);
              copy_chunk.operator()<luisa::uint>(stride_transm_and_portal());
            };
            if (_may_have_shade_shadow) {
              copy_shader_context();
            }
            $if(token == 1u) {
              if (!_may_have_shade_shadow) {
                copy_shader_context();
              }
              copy_chunk.operator()<luisa::float4>(k_stride_nee_throughput);
              if (_has_fallback_light_shader) {
                copy_chunk.operator()<luisa::float4>(k_stride_fallback_light);
              }
            };
            if (_spectral) {
              copy_chunk.operator()<luisa::float4>(stride_spectral());
            }
            // Only this edge owns a traversal batch. Never read the scratch
            // fields of a task at NEE or INTERSECT_SHADOW.
            if (_may_have_shade_shadow) {
              $if(token == 3u) {
                _shadow_storage->relocate(source, destination, capacity);
              };
            }
            _tokens->write(destination, token);
            _tokens->write(source, 0u);
          };
        };
    LUISA_INFO("Psycles shadow relocation: name='copy' kernel_{:016x}.",
               compact.function()->function().hash());
    const auto active_stages = _may_have_shade_shadow ? 3u : 2u;
    const bool vt_enabled = _vt_cache != nullptr;
    std::array<std::optional<DirectLightConsumerKernel>, 3u> consumer_kernels;
    std::array<luisa::compute::ShaderOption, 3u> consumer_options;
    for (auto stage = 0u; stage < active_stages; ++stage) {
      const bool stage_may_have_shade_shadow = _may_have_shade_shadow;
      const bool stage_may_continue_shadow = _may_continue_shadow;
      const bool can_chain_out =
          fuse_unoccluded_shadow &&
          ((stage == 0u) ||
           (stage == 1u && stage_may_have_shade_shadow) ||
           (stage == 2u && stage_may_continue_shadow && !vt_enabled));
      consumer_kernels[stage].emplace([this, evaluator, execution_block_size,
                                       stage, fuse_unoccluded_shadow,
                                       may_have_shade_shadow = stage_may_have_shade_shadow,
                                       can_chain_out,
                                       vt_enabled](
                                          BufferFloat4 combined,
                                          BufferFloat4 light_passes,
                                          BufferFloat4 volume_guiding_raw,
                                          Var<RenderKernelParameters> parameters,
                                          UInt count,
                                          UInt chain_mode) noexcept {
        set_block_size(execution_block_size);
        if (_warp_size == 32u) {
          luisa::compute::set_warp_size(32u);
        }
        const auto x = dispatch_x();
        const auto in_kind = chain_mode & 0xffu;
        const auto out_kind = chain_mode >> 8u;
        UInt active_count = count;
        if (stage != 0u && fuse_unoccluded_shadow) {
          if (may_have_shade_shadow) {
            $if((in_kind >= 1u) & (in_kind <= 3u)) {
              active_count = _scratch_count->read(in_kind + 1u);
            }
            $elif((in_kind >= 4u) & (in_kind <= 6u)) {
              active_count = _scratch_count->read(in_kind - 2u);
            }
            $elif(in_kind >= 7u) {
              active_count = _scratch_count->read(in_kind - 7u);
            };
          } else {
            $if((in_kind >= 1u) & (in_kind <= 3u)) {
              active_count = _scratch_count->read(in_kind + 1u);
            }
            $elif(in_kind >= 7u) {
              active_count = _scratch_count->read(in_kind - 7u);
            };
          }
        }
        UInt slot = 0u;
        UInt next = 0u;
        $if(x < active_count) {
          if (stage == 0u || !fuse_unoccluded_shadow) {
            slot = _indices->read(x);
          } else if (may_have_shade_shadow) {
            $if((in_kind >= 1u) & (in_kind <= 3u)) {
              slot = _chained_indices[0u]->read(x);
            }
            $elif((in_kind >= 4u) & (in_kind <= 6u)) {
              slot = _chained_indices[1u]->read(x);
            }
            $else {
              slot = _indices->read(x);
            };
          } else {
            $if((in_kind >= 1u) & (in_kind <= 3u)) {
              slot = _chained_indices[0u]->read(x);
            }
            $else {
              slot = _indices->read(x);
            };
          }
          const auto cap = parameters.wavefront_frame_capacity;
          const auto buf = Expr<luisa::compute::ByteBuffer>{_task_buffer};
          const auto read_chunk = [&]<typename T>(std::uint32_t prefix_stride) {
            return buf.read<T>(slot_byte_offset<T>(cap, prefix_stride, slot));
          };
          const auto write_chunk = [&]<typename T>(std::uint32_t prefix_stride,
                                                   const Var<T> &val) {
            buf.write(slot_byte_offset<T>(cap, prefix_stride, slot), val);
          };
          if (stage == 0u) {
            Var<DirectLightTaskCall> task;
            const auto orig_tmin =
                read_chunk.operator()<luisa::float4>(k_stride_ray_origin_tmin);
            const auto dir_tmax =
                read_chunk.operator()<luisa::float4>(k_stride_ray_dir_tmax);
            const auto ids =
                read_chunk.operator()<luisa::uint4>(k_stride_ray_ids);
            const auto tv =
                read_chunk.operator()<luisa::uint>(stride_transp_and_vol());
            const auto tt = read_chunk.operator()<luisa::float4>(
                k_stride_throughput_and_term);
            const auto film =
                read_chunk.operator()<luisa::uint4>(k_stride_film_meta);
            const auto dtf =
                read_chunk.operator()<luisa::float4>(k_stride_diff_time_flags);
            const auto rng_d =
                read_chunk.operator()<luisa::uint4>(k_stride_rng_and_depths);
            const auto tp =
                read_chunk.operator()<luisa::uint>(stride_transm_and_portal());
            task.ray_origin = orig_tmin.xyz();
            task.ray_minimum = orig_tmin.w;
            task.ray_direction = dir_tmax.xyz();
            task.ray_maximum = dir_tmax.w;
            task.source_object = ids.x;
            task.source_primitive = ids.y;
            task.light_object = ids.z;
            task.light_primitive = ids.w;
            task.transparent_depth = (tv & 0xffffu).cast<std::uint16_t>();
            task.volume_bounds_bounce = tv >> 16u;
            task.unshadowed_contribution = tt.xyz();
            task.light_terminate_sample = tt.w;
            task.path_depth = film.y;
            task.path_flags = film.z;
            task.ray_dP = dtf.x;
            task.ray_dD = dtf.y;
            task.ray_time = dtf.z;
            task.shader_flags = dtf.w.bitcast<luisa::uint>();
            task.sample_index = rng_d.x;
            task.rng_hash = rng_d.y;
            task.rng_offset = rng_d.z;
            task.diffuse_depth = rng_d.w & 0xffffu;
            task.glossy_depth = rng_d.w >> 16u;
            task.transmission_depth = tp & 0xffffu;
            task.portal_depth = (tp >> 16u).cast<std::uint16_t>();
            const auto nee_chunk =
                read_chunk.operator()<luisa::float4>(k_stride_nee_throughput);
            task.nee_path_throughput = nee_chunk.xyz();
            Float fallback_light_w = 0.0f;
            if (_has_fallback_light_shader) {
              const auto fallback_chunk =
                  read_chunk.operator()<luisa::float4>(k_stride_fallback_light);
              task.light_shader = fallback_chunk.xyz();
              fallback_light_w = fallback_chunk.w;
            }
            std::optional<DirectLightSpectralExtension> spec;
            if (_spectral) {
              const auto spec_chunk =
                  read_chunk.operator()<luisa::float4>(stride_spectral());
              spec.emplace(DirectLightSpectralExtension{
                  .unshadowed_contribution_w = spec_chunk.x,
                  .nee_path_throughput_w = nee_chunk.w,
                  .light_shader_w = fallback_light_w,
                  .shadow_throughput_w = spec_chunk.x,
                  .u_lambda = spec_chunk.y});
            }
            UInt vt_miss = vt_invalid_page;
            $if(evaluator.shade_light_nee(
                    task, parameters, vt_enabled ? &vt_miss : nullptr,
                    spec ? &*spec : nullptr)) {
              write_chunk.operator()<luisa::float4>(
                  k_stride_throughput_and_term,
                  def(luisa::compute::make_float4(task.shadow_throughput, 0.0f)));
              if (_spectral) {
                write_chunk.operator()<luisa::float4>(
                    stride_spectral(),
                    def(luisa::compute::make_float4(
                        spec->shadow_throughput_w, spec->u_lambda, 0.0f, 0.0f)));
              }
              next = 2u;
            };
            if (vt_enabled) {
              $if(vt_miss != vt_invalid_page) {
                next = 1u;
                _vt_cache->record_aux_miss(vt_miss);
                $if(_count->read(4u) == 0u) {
                  _count->atomic(4u).fetch_add(1u);
                };
              };
            }
          } else if (stage == 1u) {
            Var<DirectLightTaskCall> task;
            const auto orig_tmin =
                read_chunk.operator()<luisa::float4>(k_stride_ray_origin_tmin);
            const auto dir_tmax =
                read_chunk.operator()<luisa::float4>(k_stride_ray_dir_tmax);
            const auto ids =
                read_chunk.operator()<luisa::uint4>(k_stride_ray_ids);
            const auto tv =
                read_chunk.operator()<luisa::uint>(stride_transp_and_vol());
            task.ray_origin = orig_tmin.xyz();
            task.ray_minimum = orig_tmin.w;
            task.ray_direction = dir_tmax.xyz();
            task.ray_maximum = dir_tmax.w;
            task.source_object = ids.x;
            task.source_primitive = ids.y;
            task.light_object = ids.z;
            task.light_primitive = ids.w;
            task.transparent_depth = (tv & 0xffffu).cast<std::uint16_t>();
            task.volume_bounds_bounce = tv >> 16u;
            const auto use_summary =
                evaluator.intersect_shadow != nullptr &&
                evaluator.intersect_shadow->has_summary_callable();
            UInt batch_blocked = 0u;
            UInt batch_count = 0u;
            UInt batch_total = 0u;
            if (use_summary) {
              const auto ray = luisa::compute::make_ray(
                  task.ray_origin, task.ray_direction, task.ray_minimum,
                  task.ray_maximum);
              const auto remaining =
                  parameters.transparent_max_bounces -
                  luisa::compute::min(task.transparent_depth.cast<unsigned>(),
                                      parameters.transparent_max_bounces);
              const auto summary =
                  evaluator.intersect_shadow->summary_callable()(
                      ray, task.source_object, task.source_primitive,
                      task.light_object, task.light_primitive, remaining, slot,
                      parameters.wavefront_frame_capacity);
              batch_blocked = summary.blocked;
              batch_count = summary.count;
              batch_total = summary.total;
            } else {
              const auto batch = evaluator.intersect(task, parameters);
              batch_blocked = batch.blocked;
              batch_count = batch.count;
              batch_total = batch.total;
              $if(batch_blocked == 0u) {
                if (fuse_unoccluded_shadow) {
                  $if(batch_count != 0u) {
                    _shadow_storage->write_batch(
                        slot, batch, parameters.wavefront_frame_capacity);
                  };
                } else {
                  _shadow_storage->write_batch(
                      slot, batch, parameters.wavefront_frame_capacity);
                }
              };
            }
            $if(batch_blocked == 0u) {
              if (fuse_unoccluded_shadow) {
                $if(batch_count == 0u) {
                  $if(task.volume_bounds_bounce <= shadow_volume_bounds_max) {
                    const auto dg0 = read_chunk.operator()<luisa::float4>(
                        k_stride_diffuse_glossy_0);
                    const auto gyz =
                        read_chunk.operator()<luisa::float2>(stride_glossy_yz());
                    const auto film =
                        read_chunk.operator()<luisa::uint4>(k_stride_film_meta);
                    task.shadow_throughput =
                        read_chunk.operator()<luisa::float4>(
                            k_stride_throughput_and_term)
                            .xyz();
                    task.diffuse_weight = dg0.xyz();
                    task.glossy_weight =
                        luisa::compute::make_float3(dg0.w, gyz.x, gyz.y);
                    task.pixel = film.x;
                    task.path_depth = film.y;
                    task.path_flags = film.z;
                    task.path_visibility = film.w;
                    std::optional<DirectLightSpectralExtension> spec;
                    if (_spectral) {
                      const auto spec_chunk =
                          read_chunk.operator()<luisa::float4>(stride_spectral());
                      spec.emplace(DirectLightSpectralExtension{
                          .shadow_throughput_w = spec_chunk.x,
                          .u_lambda = spec_chunk.y});
                    }
                    const auto value = evaluator.contribution(
                        task, task.shadow_throughput, parameters,
                        spec ? &*spec : nullptr);
                    atomic_accumulate_radiance(combined, volume_guiding_raw, task.pixel,
                                               task.pixel * volume_guiding::raw_pixel_stride,
                                               evaluator.volume_guiding, task.path_flags,
                                               task.path_visibility, task.path_depth, value);
                    evaluator.accumulate_passes_atomic(light_passes,
                                                       task.pixel * light_pass_buffer_count,
                                                       task, value);
                  };
                }
                $else {
                  if (use_summary) {
                    _shadow_storage->write_summary(
                        slot, batch_count, batch_total, 0u,
                        parameters.wavefront_frame_capacity);
                  }
                  next = 3u;
                };
              } else {
                if (use_summary) {
                  _shadow_storage->write_summary(
                      slot, batch_count, batch_total, 0u,
                      parameters.wavefront_frame_capacity);
                }
                next = 3u;
              }
            };
          } else {
            const auto saved_summary = _shadow_storage->read_summary(
                slot, parameters.wavefront_frame_capacity);
            const auto batch_blocked = saved_summary->blocked;
            const auto batch_count = saved_summary->count;
            $if(batch_blocked == 0u) {
              $if(batch_count == 0u) {
                Var<DirectLightTaskCall> task;
                const auto tv =
                    read_chunk.operator()<luisa::uint>(stride_transp_and_vol());
                task.volume_bounds_bounce = tv >> 16u;
                $if(task.volume_bounds_bounce <= shadow_volume_bounds_max) {
                  const auto dg0 = read_chunk.operator()<luisa::float4>(
                      k_stride_diffuse_glossy_0);
                  const auto gyz =
                      read_chunk.operator()<luisa::float2>(stride_glossy_yz());
                  const auto film =
                      read_chunk.operator()<luisa::uint4>(k_stride_film_meta);
                  task.shadow_throughput =
                      read_chunk.operator()<luisa::float4>(
                          k_stride_throughput_and_term)
                          .xyz();
                  task.diffuse_weight = dg0.xyz();
                  task.glossy_weight =
                      luisa::compute::make_float3(dg0.w, gyz.x, gyz.y);
                  task.pixel = film.x;
                  task.path_depth = film.y;
                  task.path_flags = film.z;
                  task.path_visibility = film.w;
                  std::optional<DirectLightSpectralExtension> spec;
                  if (_spectral) {
                    const auto spec_chunk =
                        read_chunk.operator()<luisa::float4>(stride_spectral());
                    spec.emplace(DirectLightSpectralExtension{
                        .shadow_throughput_w = spec_chunk.x,
                        .u_lambda = spec_chunk.y});
                  }
                  const auto value = evaluator.contribution(
                      task, task.shadow_throughput, parameters,
                      spec ? &*spec : nullptr);
                  atomic_accumulate_radiance(combined, volume_guiding_raw, task.pixel,
                                             task.pixel * volume_guiding::raw_pixel_stride,
                                             evaluator.volume_guiding, task.path_flags,
                                             task.path_visibility, task.path_depth, value);
                  evaluator.accumulate_passes_atomic(light_passes,
                                                     task.pixel * light_pass_buffer_count,
                                                     task, value);
                };
              }
              $else {
                Var<DirectLightTaskCall> task;
                const auto orig_tmin = read_chunk.operator()<luisa::float4>(
                    k_stride_ray_origin_tmin);
                const auto dir_tmax =
                    read_chunk.operator()<luisa::float4>(k_stride_ray_dir_tmax);
                const auto tv =
                    read_chunk.operator()<luisa::uint>(stride_transp_and_vol());
                const auto dg0 = read_chunk.operator()<luisa::float4>(
                    k_stride_diffuse_glossy_0);
                const auto gyz =
                    read_chunk.operator()<luisa::float2>(stride_glossy_yz());
                const auto film =
                    read_chunk.operator()<luisa::uint4>(k_stride_film_meta);
                const auto dtf = read_chunk.operator()<luisa::float4>(
                    k_stride_diff_time_flags);
                const auto rng_d =
                    read_chunk.operator()<luisa::uint4>(k_stride_rng_and_depths);
                const auto tp = read_chunk.operator()<luisa::uint>(
                    stride_transm_and_portal());
                task.ray_origin = orig_tmin.xyz();
                task.ray_minimum = orig_tmin.w;
                task.ray_direction = dir_tmax.xyz();
                task.ray_maximum = dir_tmax.w;
                task.transparent_depth = (tv & 0xffffu).cast<std::uint16_t>();
                task.volume_bounds_bounce = tv >> 16u;
                task.shadow_throughput =
                    read_chunk.operator()<luisa::float4>(
                        k_stride_throughput_and_term)
                        .xyz();
                task.diffuse_weight = dg0.xyz();
                task.glossy_weight =
                    luisa::compute::make_float3(dg0.w, gyz.x, gyz.y);
                task.pixel = film.x;
                task.path_depth = film.y;
                task.path_flags = film.z;
                task.path_visibility = film.w;
                task.ray_dP = dtf.x;
                task.ray_dD = dtf.y;
                task.ray_time = dtf.z;
                task.shader_flags = dtf.w.bitcast<luisa::uint>();
                task.sample_index = rng_d.x;
                task.rng_hash = rng_d.y;
                task.rng_offset = rng_d.z;
                task.diffuse_depth = rng_d.w & 0xffffu;
                task.glossy_depth = rng_d.w >> 16u;
                task.transmission_depth = tp & 0xffffu;
                task.portal_depth = (tp >> 16u).cast<std::uint16_t>();
                Var<ShadowIntersectionBatchCall> batch;
                if (evaluator.intersect_shadow != nullptr &&
                    evaluator.intersect_shadow->has_summary_callable()) {
                  batch = _shadow_storage->materialize(
                      slot, saved_summary, task.ray_maximum,
                      parameters.wavefront_frame_capacity);
                } else {
                  batch = _shadow_storage->read_batch(
                      slot, parameters.wavefront_frame_capacity);
                }
                UInt vt_miss = vt_invalid_page;
                std::optional<DirectLightSpectralExtension> spec;
                if (_spectral) {
                  const auto spec_chunk =
                      read_chunk.operator()<luisa::float4>(stride_spectral());
                  spec.emplace(DirectLightSpectralExtension{
                      .shadow_throughput_w = spec_chunk.x,
                      .u_lambda = spec_chunk.y});
                }
                const auto step = evaluator.shade_shadow(
                    task, batch, parameters, vt_enabled ? &vt_miss : nullptr,
                    spec ? &*spec : nullptr);
                Bool row_vt_missed = false;
                if (vt_enabled) {
                  row_vt_missed = (vt_miss != vt_invalid_page);
                  $if(row_vt_missed) {
                    next = 3u;
                    _vt_cache->record_aux_miss(vt_miss);
                    $if(_count->read(4u) == 0u) {
                      _count->atomic(4u).fetch_add(1u);
                    };
                  };
                }
                $if(!row_vt_missed) {
                  if (_may_continue_shadow) {
                    $if(step.continue_shadow) {
                      write_chunk.operator()<luisa::float4>(
                          k_stride_ray_origin_tmin,
                          def(luisa::compute::make_float4(task.ray_origin,
                                                          task.ray_minimum)));
                      write_chunk.operator()<luisa::float4>(
                          k_stride_throughput_and_term,
                          def(luisa::compute::make_float4(task.shadow_throughput,
                                                          0.0f)));
                      if (_spectral) {
                        write_chunk.operator()<luisa::float4>(
                            stride_spectral(),
                            def(luisa::compute::make_float4(
                                spec->shadow_throughput_w, spec->u_lambda,
                                0.0f, 0.0f)));
                      }
                      write_chunk.operator()<luisa::uint>(
                          stride_transp_and_vol(),
                          def((task.transparent_depth.cast<luisa::uint>() &
                               0xffffu) |
                              (luisa::compute::min(task.volume_bounds_bounce,
                                                   0xffffu)
                               << 16u)));
                      write_chunk.operator()<luisa::uint4>(
                          k_stride_rng_and_depths,
                          def(luisa::compute::make_uint4(
                              task.sample_index, task.rng_hash, task.rng_offset,
                              rng_d.w)));
                      next = 2u;
                    }
                    $elif(step.visible) {
                      const auto value = evaluator.contribution(
                          task, task.shadow_throughput, parameters,
                          spec ? &*spec : nullptr);
                      atomic_accumulate_radiance(
                          combined, volume_guiding_raw, task.pixel,
                          task.pixel * volume_guiding::raw_pixel_stride,
                          evaluator.volume_guiding, task.path_flags,
                          task.path_visibility, task.path_depth, value);
                      evaluator.accumulate_passes_atomic(
                          light_passes, task.pixel * light_pass_buffer_count,
                          task, value);
                    };
                  } else {
                    $if(step.visible) {
                      const auto value = evaluator.contribution(
                          task, task.shadow_throughput, parameters,
                          spec ? &*spec : nullptr);
                      atomic_accumulate_radiance(
                          combined, volume_guiding_raw, task.pixel,
                          task.pixel * volume_guiding::raw_pixel_stride,
                          evaluator.volume_guiding, task.path_flags,
                          task.path_visibility, task.path_depth, value);
                      evaluator.accumulate_passes_atomic(
                          light_passes, task.pixel * light_pass_buffer_count,
                          task, value);
                    };
                  }
                };
              };
            };
          }
        };
        if (can_chain_out) {
          const auto target_next =
              static_cast<luisa::uint>(stage == 1u ? 3u : 2u);
          $if(out_kind != 0u) {
            UInt out_count_slot = out_kind + 1u;
            if (stage != 0u && may_have_shade_shadow) {
              out_count_slot =
                  select(out_kind - 2u, out_kind + 1u, out_kind <= 3u);
            }
            const auto clear_count_slot =
                2u + (((out_count_slot - 2u) + 1u) % 3u);
            $if(x == 0u) {
              if (stage == 0u) {
                _count->atomic(0u).fetch_sub(active_count);
              } else {
                $if(in_kind == 0u) {
                  _count->atomic(stage).fetch_sub(active_count);
                };
              }
              _scratch_count->write(clear_count_slot, 0u);
            };
            const auto survived = (x < active_count) & (next == target_next);
            $if((x < active_count) & (next == 0u)) {
              _tokens->write(slot, 0u);
            };
            if (vt_enabled && stage != 1u) {
              const auto retry_tok = static_cast<luisa::uint>(stage + 1u);
              const auto vt_retry = (x < active_count) & (next == retry_tok);
              $if(vt_retry) {
                _tokens->write(slot, retry_tok);
              };
              if (_warp_size == 32u) {
                const auto miss_lanes =
                    luisa::compute::warp_active_count_bits(vt_retry);
                $if(vt_retry &
                    (luisa::compute::warp_prefix_count_bits(vt_retry) == 0u)) {
                  _count->atomic(stage).fetch_add(miss_lanes);
                };
              } else {
                $if(vt_retry) {
                  _count->atomic(stage).fetch_add(1u);
                };
              }
            }
            const auto append_chained =
                [&](const luisa::compute::Buffer<luisa::uint> &dst_indices) noexcept {
                  if (_warp_size == 32u) {
                    $if(survived) {
                      const auto lane_prefix =
                          luisa::compute::warp_prefix_count_bits(true);
                      const auto warp_total =
                          luisa::compute::warp_active_count_bits(true);
                      UInt warp_base = 0u;
                      $if(lane_prefix == 0u) {
                        warp_base = _scratch_count->atomic(out_count_slot)
                                        .fetch_add(warp_total);
                      };
                      warp_base =
                          luisa::compute::warp_read_first_active_lane(warp_base);
                      dst_indices->write(warp_base + lane_prefix, slot);
                    };
                  } else {
                    $if(survived) {
                      const auto out_idx =
                          _scratch_count->atomic(out_count_slot).fetch_add(1u);
                      dst_indices->write(out_idx, slot);
                    };
                  }
                };
            if (stage == 0u || !may_have_shade_shadow) {
              append_chained(_chained_indices[0u]);
            } else {
              $if(out_kind <= 3u) {
                append_chained(_chained_indices[0u]);
              }
              $else {
                append_chained(_chained_indices[1u]);
              };
            }
          }
          $else {
            $if(x < active_count) {
              _tokens->write(slot, next);
              if (stage == 0u) {
                $if(x == 0u) { _count->atomic(0u).fetch_sub(active_count); };
              } else {
                $if((x == 0u) & (in_kind == 0u)) {
                  _count->atomic(stage).fetch_sub(active_count);
                };
              }
              if (_warp_size == 32u) {
                $if(luisa::compute::warp_active_all_equal(next)) {
                  const auto total =
                      luisa::compute::warp_active_count_bits(true);
                  $if((next != 0u) &
                      (luisa::compute::warp_prefix_count_bits(true) == 0u)) {
                    _count->atomic(next - 1u).fetch_add(total);
                  };
                }
                $else {
                  $if(next != 0u) {
                    _count->atomic(next - 1u).fetch_add(1u);
                  };
                };
              } else {
                $if(next != 0u) {
                  _count->atomic(next - 1u).fetch_add(1u);
                };
              }
            };
          };
        } else {
          $if(x < active_count) {
            _tokens->write(slot, next);
            $if((x == 0u) & (in_kind == 0u)) {
              _count->atomic(stage).fetch_sub(active_count);
            };
            if (!(stage == 2u && !_may_continue_shadow && !vt_enabled)) {
              if (_warp_size == 32u) {
                $if(luisa::compute::warp_active_all_equal(next)) {
                  const auto total =
                      luisa::compute::warp_active_count_bits(true);
                  $if((next != 0u) &
                      (luisa::compute::warp_prefix_count_bits(true) == 0u)) {
                    _count->atomic(next - 1u).fetch_add(total);
                  };
                }
                $else {
                  $if(next != 0u) {
                    _count->atomic(next - 1u).fetch_add(1u);
                  };
                };
              } else {
                $if(next != 0u) {
                  _count->atomic(next - 1u).fetch_add(1u);
                };
              }
            }
          };
        }
      });
      const auto label =
          luisa::string{"wavefront_aux_"} + luisa::string{stage_name(stage)};
      LUISA_INFO("Psycles shadow stage: name='{}' kernel_{:016x} block_size={}.",
                 stage_name(stage),
                 (*consumer_kernels[stage]).function()->function().hash(),
                 execution_block_size);
      consumer_options[stage] =
          luisa::compute::coro::detail::coro_scheduler_shader_option(
              shader_option, label);
    }
    psycles::parallel_for(5u + active_stages, [&](std::size_t idx) {
      switch (idx) {
        case 0u:
          _initialize = device.compile(initialize, shader_option);
          break;
        case 1u:
          _set_extent_and_clear_reloc =
              device.compile(set_extent_and_clear_reloc, shader_option);
          break;
        case 2u:
          _gather = device.compile(*gather_kernel, shader_option);
          break;
        case 3u:
          _collect_relocations = device.compile(*collect_kernel, shader_option);
          break;
        case 4u:
          _compact = device.compile(compact, shader_option);
          break;
        default: {
          const auto stage = idx - 5u;
          _consumers[stage] = device.compile(
              *consumer_kernels[stage], consumer_options[stage]);
          break;
        }
      }
    });
    _clear_count_slot.set_name("aux_shadow_clear_count");
    _initialize.set_name("aux_shadow_init");
    _set_extent_and_clear_reloc.set_name("aux_shadow_set_extent");
    _gather.set_name("aux_shadow_gather");
    _collect_relocations.set_name("aux_shadow_collect_reloc");
    _compact.set_name("aux_shadow_compact");
    for (auto stage = 0u; stage < active_stages; ++stage) {
      _consumers[stage].set_name(
          luisa::string{"aux_shadow_stage_"} + luisa::string{stage_name(stage)});
    }
    if (device.backend_name() == "vk" &&
        device.impl()->query("vk_timeline_active") == "1") {
      auto lbl = [&](const char *name, std::uint64_t handle) noexcept {
        if (handle == 0ull || handle == luisa::compute::invalid_resource_handle) {
          return;
        }
        char qbuf[192];
        std::snprintf(qbuf, sizeof(qbuf), "vk_timeline_label:shader:%llu:%s",
                      static_cast<unsigned long long>(handle), name);
        static_cast<void>(device.impl()->query(qbuf));
      };
      lbl("aux_shadow_clear_count", _clear_count_slot.handle());
      lbl("aux_shadow_init", _initialize.handle());
      lbl("aux_shadow_set_extent", _set_extent_and_clear_reloc.handle());
      lbl("aux_shadow_gather", _gather.handle());
      lbl("aux_shadow_collect_reloc", _collect_relocations.handle());
      lbl("aux_shadow_compact", _compact.handle());
      for (auto stage = 0u; stage < active_stages; ++stage) {
        char cname[64];
        std::snprintf(cname, sizeof(cname), "aux_shadow_consumer_%u", stage);
        lbl(cname, _consumers[stage].handle());
      }
    }
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return "direct_light_shadow";
  }

  [[nodiscard]] std::uint32_t capacity() const noexcept override { return _capacity; }

  [[nodiscard]] luisa::span<const luisa::compute::coro::WavefrontCoroAuxiliaryProducer>
  producers() const noexcept override {
    return _producers;
  }

  void emit(Var<DirectLightTaskCall> task,
            Expr<std::uint32_t> runtime_capacity,
            const DirectLightSpectralExtension *spectral = nullptr)
      const noexcept override {
    if (_spectral) {
      LUISA_ASSERT(spectral != nullptr,
                   "Spectral direct-light queue requires spectral extension.");
    }
    const auto buf = Expr<luisa::compute::ByteBuffer>{_task_buffer};
    const auto tokens =
        Expr<luisa::compute::Buffer<luisa::uint>>{_tokens};
    const auto queue_count = Expr<luisa::compute::Buffer<luisa::uint>>{_count};

    UInt slot = 0u;
    if (_warp_size == 32u) {
      const auto active_lanes = luisa::compute::warp_active_count_bits(true);
      const auto lane_prefix = luisa::compute::warp_prefix_count_bits(true);
      UInt base_slot = 0u;
      $if(lane_prefix == 0u) {
        base_slot = queue_count.atomic(3u).fetch_add(active_lanes);
      };
      slot = luisa::compute::warp_read_first_active_lane(base_slot) + lane_prefix;
    } else {
      slot = queue_count.atomic(3u).fetch_add(1u);
    }

    const auto valid_slot = slot < runtime_capacity;
    const auto is_const_light = task.constant_light_shader != 0u;
    const auto token = select(1u, 2u, is_const_light);
    // Admission control proves slot < capacity. Keep the guard as a local
    // memory-safety invariant: if a producer ever violates its declared
    // bound, the scheduler observes count > capacity and diagnoses it
    // without permitting an out-of-bounds payload write.
    $if(valid_slot) {
      buf.write(slot_byte_offset<luisa::float4>(runtime_capacity,
                                                k_stride_ray_origin_tmin, slot),
                luisa::compute::make_float4(task.ray_origin, task.ray_minimum));
      buf.write(slot_byte_offset<luisa::float4>(runtime_capacity,
                                                k_stride_ray_dir_tmax, slot),
                luisa::compute::make_float4(task.ray_direction, task.ray_maximum));
      buf.write(slot_byte_offset<luisa::uint4>(runtime_capacity,
                                               k_stride_ray_ids, slot),
                luisa::compute::make_uint4(task.source_object,
                                           task.source_primitive,
                                           task.light_object,
                                           task.light_primitive));
      buf.write(
          slot_byte_offset<luisa::uint>(runtime_capacity,
                                        stride_transp_and_vol(), slot),
          (task.transparent_depth.cast<luisa::uint>() & 0xffffu) |
              (luisa::compute::min(task.volume_bounds_bounce, 0xffffu) << 16u));
      const auto initial_throughput = select(
          task.unshadowed_contribution, task.shadow_throughput, is_const_light);
      buf.write(slot_byte_offset<luisa::float4>(
                    runtime_capacity, k_stride_throughput_and_term, slot),
                luisa::compute::make_float4(initial_throughput,
                                            task.light_terminate_sample));
      buf.write(slot_byte_offset<luisa::float4>(
                    runtime_capacity, k_stride_diffuse_glossy_0, slot),
                luisa::compute::make_float4(task.diffuse_weight,
                                            task.glossy_weight.x));
      buf.write(slot_byte_offset<luisa::float2>(runtime_capacity,
                                                stride_glossy_yz(), slot),
                luisa::compute::make_float2(task.glossy_weight.y,
                                            task.glossy_weight.z));
      buf.write(slot_byte_offset<luisa::uint4>(runtime_capacity,
                                               k_stride_film_meta, slot),
                luisa::compute::make_uint4(task.pixel, task.path_depth,
                                           task.path_flags,
                                           task.path_visibility));
      const auto write_shader_context = [&] {
        buf.write(slot_byte_offset<luisa::float4>(
                      runtime_capacity, k_stride_diff_time_flags, slot),
                  luisa::compute::make_float4(
                      task.ray_dP, task.ray_dD, task.ray_time,
                      task.shader_flags.bitcast<float>()));
        buf.write(
            slot_byte_offset<luisa::uint4>(runtime_capacity,
                                           k_stride_rng_and_depths, slot),
            luisa::compute::make_uint4(
                task.sample_index, task.rng_hash, task.rng_offset,
                luisa::compute::min(task.diffuse_depth, 0xffffu) |
                    (luisa::compute::min(task.glossy_depth, 0xffffu) << 16u)));
        buf.write(
            slot_byte_offset<luisa::uint>(runtime_capacity,
                                          stride_transm_and_portal(), slot),
            luisa::compute::min(task.transmission_depth, 0xffffu) |
                (task.portal_depth.cast<luisa::uint>() << 16u));
      };
      if (_may_have_shade_shadow) {
        write_shader_context();
      }
      $if(!is_const_light) {
        if (!_may_have_shade_shadow) {
          write_shader_context();
        }
        buf.write(slot_byte_offset<luisa::float4>(
                      runtime_capacity, k_stride_nee_throughput, slot),
                  luisa::compute::make_float4(
                      task.nee_path_throughput,
                      spectral != nullptr ? spectral->nee_path_throughput_w
                                          : Float{0.0f}));
        if (_has_fallback_light_shader) {
          buf.write(slot_byte_offset<luisa::float4>(
                        runtime_capacity, k_stride_fallback_light, slot),
                    luisa::compute::make_float4(
                        task.light_shader,
                        spectral != nullptr ? spectral->light_shader_w
                                            : Float{0.0f}));
        }
      };
      if (_spectral) {
        const auto initial_w = select(spectral->unshadowed_contribution_w,
                                      spectral->shadow_throughput_w,
                                      is_const_light);
        buf.write(slot_byte_offset<luisa::float4>(
                      runtime_capacity, stride_spectral(), slot),
                  luisa::compute::make_float4(initial_w, spectral->u_lambda,
                                              0.0f, 0.0f));
      }
      tokens.write(slot, token);
      if (_warp_size == 32u) {
        $if(luisa::compute::warp_active_all_equal(token)) {
          const auto total = luisa::compute::warp_active_count_bits(true);
          $if(luisa::compute::warp_prefix_count_bits(true) == 0u) {
            queue_count.atomic(token - 1u).fetch_add(total);
          };
        }
        $else { queue_count.atomic(token - 1u).fetch_add(1u); };
      } else {
        queue_count.atomic(token - 1u).fetch_add(1u);
      }
    };
  }

  void reset(Stream &stream) noexcept override {
    _host_counts = {};
    _scratch_slot = 0u;
    _chain_count_slot = 0u;
    _chain_buf_slot = 0u;
    stream << _initialize(_capacity).dispatch(std::max(_capacity, 5u));
  }

  void encode_reset(Stream &, luisa::compute::CommandList &commands) noexcept override {
    _host_counts = {};
    _scratch_slot = 0u;
    _chain_count_slot = 0u;
    _chain_buf_slot = 0u;
    commands << _initialize(_capacity).dispatch(std::max(_capacity, 5u));
  }

  void enqueue_count_readback(Stream &stream) noexcept override {
    stream << _count.copy_to(luisa::span{_host_counts});
  }

  void encode_count_readback(Stream &,
                             luisa::compute::CommandList &commands) noexcept override {
    commands << _count.copy_to(luisa::span{_host_counts});
  }

  [[nodiscard]] std::uint32_t host_count() const noexcept override {
    LUISA_ASSERT(_host_counts[3] <= _capacity,
                 "Shadow append extent {} exceeds capacity {}.", _host_counts[3],
                 _capacity);
    const auto live = uint64_t{_host_counts[0]} + _host_counts[1] + _host_counts[2];
    LUISA_ASSERT(live <= _host_counts[3],
                 "Shadow live count exceeds its allocated extent.");
    return static_cast<unsigned>(live);
  }

  [[nodiscard]] unsigned stage_count() const noexcept override { return 3u; }
  [[nodiscard]] luisa::string_view stage_name(unsigned stage) const noexcept override {
    constexpr luisa::string_view names[]{path_transition::shade_light_nee,
                                         path_transition::intersect_shadow,
                                         path_transition::shade_shadow};
    LUISA_ASSERT(stage < 3u, "Invalid shadow stage {}.", stage);
    return names[stage];
  }
  [[nodiscard]] unsigned stage_host_count(unsigned stage) const noexcept override {
    LUISA_ASSERT(stage < 3u, "Invalid shadow stage {}.", stage);
    return _host_counts[stage];
  }

  void prepare_for_admission(Stream &stream) noexcept override {
    auto commands = luisa::compute::CommandList::create(4u);
    encode_prepare_for_admission(stream, commands);
    if (!commands.empty()) {
      stream << commands.commit();
    }
  }

  void encode_prepare_for_admission(
      Stream &, luisa::compute::CommandList &commands) noexcept override {
    const auto live = host_count();
    const auto extent = _host_counts[3];
    const auto plan = cycles_shadow_compaction(extent, live);
    // Cycles decides whether relocation is worthwhile before testing append
    // admission. Unreclaimed holes are not available producer slots; when
    // the tail is insufficient, the generic scheduler drains this side pool.
    if (plan.compact) {
      // Use the proven host bound instead of reading H back or using HIP
      // indirect dispatch, both of which require an extra host synchronization.
      const auto bound = std::min(live, extent - live);
      _host_counts[3] = plan.extent;
      commands << _set_extent_and_clear_reloc(
                      plan.extent, plan.upload_extent ? 1u : 0u, 1u)
                      .dispatch(2u)
               << _collect_relocations(live, extent).dispatch(extent)
               << _compact(_capacity, live).dispatch(bound);
    } else if (plan.upload_extent) {
      _host_counts[3] = plan.extent;
      commands << _set_extent_and_clear_reloc(plan.extent, 1u, 0u).dispatch(1u);
    }
  }

  [[nodiscard]] unsigned host_available_slots() const noexcept override {
    return _capacity - _host_counts[3];
  }

  [[nodiscard]] bool try_predict_stage_transition(unsigned stage) noexcept override {
    if (_vt_cache != nullptr && (stage == 0u || stage == 2u)) {
      return false;
    }
    if (_fuse_unoccluded_shadow && !_may_have_shade_shadow) {
      if (stage == 0u) {
        _host_counts[0u] = 0u;
        _host_counts[1u] = 0u;
        return true;
      }
      if (stage == 1u) {
        _host_counts[1u] = 0u;
        return true;
      }
    }
    if (stage == 2u && !_may_continue_shadow) {
      _host_counts[2u] = 0u;
      return true;
    }
    return false;
  }

  [[nodiscard]] bool stage_blocks_async_domain(
      unsigned stage, luisa::string_view domain) const noexcept override {
    return _vt_cache != nullptr && _host_counts[4u] != 0u &&
           (stage == 0u || stage == 2u) && domain == vt_extension_schema;
  }

  void prepare_for_producer(Stream &, unsigned required) noexcept override {
    LUISA_ASSERT(required <= host_available_slots(),
                 "Shadow producer exceeds its admitted append storage.");
  }

  void encode_prepare_for_producer(Stream &, luisa::compute::CommandList &,
                                   unsigned required) noexcept override {
    LUISA_ASSERT(required <= host_available_slots(),
                 "Shadow producer exceeds its admitted append storage.");
  }

  void encode_dispatch_stage(unsigned stage, Stream &stream,
                             luisa::compute::CommandList &commands,
                             luisa::compute::BufferView<luisa::float4> combined,
                             luisa::compute::BufferView<luisa::float4>,
                             luisa::compute::BufferView<luisa::float4>,
                             luisa::compute::BufferView<luisa::float4> light_passes,
                             luisa::compute::BufferView<luisa::uint>,
                             luisa::compute::BufferView<luisa::float4> volume_guiding_raw,
                             luisa::compute::BufferView<luisa::uint>,
                             luisa::compute::BufferView<luisa::float4>,
                             const std::uint32_t &, const std::uint32_t &,
                             luisa::compute::BufferView<luisa::float4>,
                             luisa::compute::BufferView<float>,
                             const RenderKernelParameters &parameters) noexcept override {
    if (_vt_cache && _host_counts[4u] != 0u &&
        (stage == 0u || stage == 2u)) {
      if (!commands.empty()) {
        stream << commands.commit();
      }
      _vt_cache->service_aux_misses(stream);
      _host_counts[4u] = 0u;
      commands << _clear_count_slot(4u).dispatch(1u);
    }
    const auto count = stage_host_count(stage);
    LUISA_ASSERT(parameters.wavefront_frame_capacity == _capacity,
                 "Direct-light queue runtime capacity {} does not match its "
                 "allocation capacity {}.",
                 parameters.wavefront_frame_capacity, _capacity);
    LUISA_ASSERT(count != 0u && count <= _capacity,
                 "Invalid direct-light auxiliary dispatch count {} "
                 "for capacity {}.",
                 count, _capacity);
    auto consumer_parameters = parameters;
    consumer_parameters.shadow_storage_block_size = _execution_block_size;
    // The 32-lane _gather specialization uses a 256-thread workgroup with
    // 8-warp LDS reduction and sync_block(). Dispatch whole 256-thread blocks
    // so Luisa's entry bounds guard never exits trailing warps before they
    // initialize warp_totals[warp_id] and reach the workgroup barriers.
    const auto gather_threads =
        _warp_size == 32u
            ? ((std::max(_host_counts[3], 1u) + 255u) / 256u) * 256u
            : std::max(_host_counts[3], 1u);
    const auto alloc_chain_slot = [&](bool use_buf1) noexcept {
      const auto slot = _chain_count_slot;
      _chain_count_slot = (_chain_count_slot + 1u) % 3u;
      return (use_buf1 ? 4u : 1u) + slot;
    };
    if (stage == 0u && _fuse_unoccluded_shadow) {
      const auto prior_stage1 = _host_counts[1u];
      const auto scratch_slot = _scratch_slot;
      _scratch_slot ^= 1u;
      if (prior_stage1 == 0u) {
        const auto chain01 = alloc_chain_slot(false);
        if (_may_have_shade_shadow && _host_counts[2u] == 0u) {
          const auto chain12 = alloc_chain_slot(true);
          commands << _gather(1u, _host_counts[3], scratch_slot, 0u)
                          .dispatch(gather_threads)
                   << _consumers[0u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, count,
                                     pack_chain_mode(0u, chain01))
                          .dispatch(count)
                   << _consumers[1u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, count,
                                     pack_chain_mode(chain01, chain12))
                          .dispatch(count)
                   << _consumers[2u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, count,
                                     pack_chain_mode(chain12, 0u))
                          .dispatch(count);
        } else {
          commands << _gather(1u, _host_counts[3], scratch_slot, 0u)
                          .dispatch(gather_threads)
                   << _consumers[0u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, count,
                                     pack_chain_mode(0u, chain01))
                          .dispatch(count)
                   << _consumers[1u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, count,
                                     pack_chain_mode(chain01, 0u))
                          .dispatch(count);
        }
      } else {
        const auto stage1_scratch = _scratch_slot;
        _scratch_slot ^= 1u;
        const auto upper_bound = count + prior_stage1;
        const auto chain12 =
            (_may_have_shade_shadow && _host_counts[2u] == 0u) ?
                alloc_chain_slot(true) :
                0u;
        commands << _gather(1u, _host_counts[3], scratch_slot, 0u)
                        .dispatch(gather_threads)
                 << _consumers[0u](combined, light_passes, volume_guiding_raw,
                                   consumer_parameters, count,
                                   pack_chain_mode(0u, 0u))
                        .dispatch(count)
                 << _gather(2u, _host_counts[3], stage1_scratch, 1u)
                        .dispatch(gather_threads)
                 << _consumers[1u](combined, light_passes, volume_guiding_raw,
                                   consumer_parameters, upper_bound,
                                   pack_chain_mode(7u + stage1_scratch, chain12))
                        .dispatch(upper_bound);
        if (chain12 != 0u) {
          commands << _consumers[2u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, upper_bound,
                                     pack_chain_mode(chain12, 0u))
                          .dispatch(upper_bound);
        }
      }
      return;
    }
    if (stage == 1u && _fuse_unoccluded_shadow && _may_have_shade_shadow &&
        _host_counts[2u] == 0u) {
      std::uint32_t rounds = 1u;
      if (_may_continue_shadow && _vt_cache == nullptr) {
        rounds = 2u;
        if (const auto *raw = std::getenv("PSYCLES_SHADOW_CHAIN_MAX_ROUNDS");
            raw != nullptr && raw[0] != '\0') {
          char *end = nullptr;
          const auto parsed = std::strtoul(raw, &end, 10);
          if (end != raw && *end == '\0' && parsed >= 1ul && parsed <= 16ul) {
            rounds = static_cast<std::uint32_t>(parsed);
          }
        }
      }
      const auto scratch_slot = _scratch_slot;
      _scratch_slot ^= 1u;
      commands << _gather(2u, _host_counts[3], scratch_slot, 0u)
                      .dispatch(gather_threads);
      std::uint32_t cur_in = 0u;
      bool use_buf1 = false;
      for (std::uint32_t r = 0u; r < rounds; ++r) {
        const auto chain12 = alloc_chain_slot(use_buf1);
        use_buf1 = !use_buf1;
        commands << _consumers[1u](combined, light_passes, volume_guiding_raw,
                                   consumer_parameters, count,
                                   pack_chain_mode(cur_in, chain12))
                        .dispatch(count);
        if (r + 1u < rounds) {
          const auto chain21 = alloc_chain_slot(use_buf1);
          use_buf1 = !use_buf1;
          commands << _consumers[2u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, count,
                                     pack_chain_mode(chain12, chain21))
                          .dispatch(count);
          cur_in = chain21;
        } else {
          commands << _consumers[2u](combined, light_passes, volume_guiding_raw,
                                     consumer_parameters, count,
                                     pack_chain_mode(chain12, 0u))
                          .dispatch(count);
        }
      }
      return;
    }
    const auto scratch_slot = _scratch_slot;
    _scratch_slot ^= 1u;
    commands << _gather(stage + 1u, _host_counts[3], scratch_slot, 0u)
                    .dispatch(gather_threads)
             << _consumers[stage](combined, light_passes, volume_guiding_raw,
                                  consumer_parameters, count,
                                  pack_chain_mode(0u, 0u))
                    .dispatch(count);
  }

  void dispatch_stage(unsigned stage, Stream &stream,
                      luisa::compute::BufferView<luisa::float4> combined,
                      luisa::compute::BufferView<luisa::float4> normal,
                      luisa::compute::BufferView<luisa::float4> albedo,
                      luisa::compute::BufferView<luisa::float4> light_passes,
                      luisa::compute::BufferView<luisa::uint> sample_count,
                      luisa::compute::BufferView<luisa::float4> volume_guiding_raw,
                      luisa::compute::BufferView<luisa::uint> volume_guiding_count,
                      luisa::compute::BufferView<luisa::float4> path_trace,
                      const std::uint32_t &sample_first,
                      const std::uint32_t &samples,
                      luisa::compute::BufferView<luisa::float4> sobol_table,
                      luisa::compute::BufferView<float> filter_table,
                      const RenderKernelParameters &parameters) noexcept override {
    auto commands = luisa::compute::CommandList::create(3u);
    encode_dispatch_stage(stage, stream, commands, combined, normal, albedo,
                          light_passes, sample_count, volume_guiding_raw,
                          volume_guiding_count, path_trace, sample_first,
                          samples, sobol_table, filter_table, parameters);
    stream << commands.commit();
  }

  void dispatch(Stream &, luisa::compute::BufferView<luisa::float4>,
                luisa::compute::BufferView<luisa::float4>,
                luisa::compute::BufferView<luisa::float4>,
                luisa::compute::BufferView<luisa::float4>,
                luisa::compute::BufferView<luisa::uint>,
                luisa::compute::BufferView<luisa::float4>,
                luisa::compute::BufferView<luisa::uint>,
                luisa::compute::BufferView<luisa::float4>, const std::uint32_t &,
                const std::uint32_t &, luisa::compute::BufferView<luisa::float4>,
                luisa::compute::BufferView<float>,
                const RenderKernelParameters &) noexcept override {
    LUISA_ERROR("A multi-stage shadow pool requires stage-aware wavefront scheduling.");
  }
};

} // namespace

DirectLightTaskQueueBinding
make_direct_light_task_queue_impl(
    luisa::compute::Device &device, DirectLightTaskEvaluator evaluator,
    std::uint32_t capacity, std::uint32_t execution_block_size,
    const luisa::compute::ShaderOption &shader_option,
    bool fuse_unoccluded_shadow,
    std::shared_ptr<const ShadowIntersectionBatchStorage> shadow_storage = {},
    bool may_have_shade_shadow = true,
    bool may_continue_shadow = true,
    std::span<const std::string_view> producer_continuations = {},
    std::shared_ptr<VirtualTextureCache> vt_cache = {}) noexcept {
  LUISA_ASSERT(capacity != 0u, "Direct-light task queue capacity must be positive.");
  auto queue = std::make_shared<DirectLightTaskQueue>(
      device, std::move(evaluator), capacity, execution_block_size,
      shader_option, fuse_unoccluded_shadow, std::move(shadow_storage),
      may_have_shade_shadow, may_continue_shadow, producer_continuations,
      std::move(vt_cache));
  LUISA_INFO("Psycles direct-light shadow queue: capacity={} "
             "payload_fields={} payload_aos_bytes={} task_soa_bytes={} "
             "traversal_soa_bytes={} queue_bytes={} stages={} fused_unoccluded={}.",
             capacity, luisa::compute::Type::of<DirectLightTaskCall>()->members().size(),
             luisa::compute::Type::of<DirectLightTaskCall>()->size(),
             queue->task_storage_bytes(), queue->shadow_storage_bytes(),
             (2ull * capacity + 8u) * sizeof(luisa::uint),
             may_have_shade_shadow ? 3u : 2u, fuse_unoccluded_shadow);
  return {.sink = queue, .work = std::move(queue)};
}

DirectLightTaskQueueBinding
make_direct_light_task_queue(luisa::compute::Device &device,
                             DirectLightTaskEvaluator evaluator, std::uint32_t capacity,
                             std::uint32_t execution_block_size,
                             const luisa::compute::ShaderOption &shader_option) noexcept {
  return make_direct_light_task_queue_impl(
      device, std::move(evaluator), capacity, execution_block_size,
      shader_option, false);
}

DirectLightTaskQueueBinding
make_direct_light_task_queue(luisa::compute::Device &device,
                             const PathKernelConfig &config, std::uint32_t capacity,
                             std::uint32_t execution_block_size,
                             const luisa::compute::ShaderOption &shader_option) noexcept {
  LUISA_ASSERT(capacity != 0u, "Direct-light task queue capacity must be positive.");
  const bool fast_triangle_shadow_domain =
      !config.scene->geometries.empty() &&
      config.scene->curve_geometries.empty() &&
      config.scene->traversal_object_instance_buffer.has_value() &&
      config.scene->traversal_shadow_opacity_buffer.has_value();
  const bool fast_opaque_triangle_shadow =
      fast_triangle_shadow_domain &&
      !config.scene->has_transparent_shadow_materials;
  const bool may_have_shade_shadow = !fast_opaque_triangle_shadow;
  const bool may_continue_shadow =
      !fast_triangle_shadow_domain ||
      config.transparent_max_bounces >= shadow_intersection_batch_capacity;
  auto shadow_storage = std::make_shared<ShadowIntersectionBatchStorage>(
      device, may_have_shade_shadow ? capacity : 1u);
  auto evaluator = make_direct_light_task_evaluator(config);
  evaluator.intersect_shadow = make_shadow_intersection_component(
      config.scene, shadow_storage, config.transparent_max_bounces);
  const auto bucket_plan = make_path_surface_bucket_plan(config);
  const auto producers = bucket_plan.producer_continuations();
  auto vt_cache =
      (config.enable_virtual_texture_cache && config.scene &&
       config.scene->virtual_texture_cache &&
       config.scene->virtual_texture_cache->has_virtual_images())
          ? config.scene->virtual_texture_cache
          : std::shared_ptr<VirtualTextureCache>{};
  return make_direct_light_task_queue_impl(
      device, std::move(evaluator), capacity, execution_block_size,
      shader_option, true, std::move(shadow_storage), may_have_shade_shadow,
      may_continue_shadow, producers, std::move(vt_cache));
}

} // namespace psycles::luisa_backend::detail

#include "path_kernel_direct_light_queue.h"

#include "cycles_wavefront_policy.h"
#include "path_kernel_film.h"
#include "path_kernel_transitions.h"

#include <array>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/soa.h>

namespace psycles::luisa_backend::detail {
namespace {

using DirectLightConsumerShader = luisa::compute::Shader1D<
    luisa::compute::Buffer<luisa::float4>, luisa::compute::Buffer<luisa::float4>,
    luisa::compute::Buffer<luisa::float4>, RenderKernelParameters, std::uint32_t>;

class DirectLightTaskQueue final : public RenderAuxiliaryWork,
                                   public DirectLightTaskSink {

private:
  luisa::compute::SOA<DirectLightTaskCall> _tasks;
  luisa::compute::SOA<ShadowIntersectionBatchCall> _batches;
  luisa::compute::Buffer<luisa::uint> _tokens;
  luisa::compute::Buffer<luisa::uint> _indices;
  luisa::compute::Buffer<luisa::uint> _scratch_count;
  // Counts 0..2 are live stages; count 3 is the append allocation extent.
  luisa::compute::Buffer<luisa::uint> _count;
  std::array<DirectLightConsumerShader, 3u> _consumers;
  luisa::compute::Shader1D<luisa::uint> _initialize;
  luisa::compute::Shader1D<luisa::uint, luisa::uint> _gather;
  luisa::compute::Shader1D<luisa::uint, luisa::uint, luisa::uint> _compact;
  std::array<luisa::compute::coro::WavefrontCoroAuxiliaryProducer, 1u> _producers;
  std::uint32_t _capacity{};
  std::uint32_t _execution_block_size{};
  std::array<std::uint32_t, 4u> _host_counts{};
  std::array<std::uint32_t, 4u> _zeros{};
  std::uint32_t _zero{};

  [[nodiscard]] auto batch_storage(UInt capacity) const noexcept {
    return luisa::compute::Expr<luisa::compute::SOA<ShadowIntersectionBatchCall>>{
        luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>>{_batches.buffer()},
        UInt{0u}, capacity, UInt{0u}};
  }

public:
  DirectLightTaskQueue(luisa::compute::Device &device, DirectLightTaskEvaluator evaluator,
                       std::uint32_t capacity, std::uint32_t execution_block_size,
                       const luisa::compute::ShaderOption &shader_option) noexcept
      : _tasks{device.create_soa<DirectLightTaskCall>(capacity)},
        _batches{device.create_soa<ShadowIntersectionBatchCall>(capacity)},
        _tokens{device.create_buffer<luisa::uint>(capacity)},
        _indices{device.create_buffer<luisa::uint>(capacity)},
        _scratch_count{device.create_buffer<luisa::uint>(1u)},
        _count{device.create_buffer<luisa::uint>(4u)},
        _producers{{{.continuation = path_transition::shade_surface,
                     .max_emitted_per_invocation = 1u}}},
        _capacity{capacity}, _execution_block_size{execution_block_size} {
    LUISA_ASSERT(execution_block_size != 0u,
                 "Direct-light consumer block size must be positive.");
    luisa::compute::Kernel1D initialize = [this](UInt capacity) {
      $if(dispatch_x() < capacity) { _tokens->write(dispatch_x(), 0u); };
    };
    _initialize = device.compile(initialize, shader_option);
    // Enumerate exactly the selected stage (or token-zero holes for
    // compaction). Consumers dispatch the queue count, not the whole pool.
    luisa::compute::Kernel1D gather = [this](UInt token, UInt extent) {
      const auto slot = dispatch_x();
      $if(slot < extent) {
        $if(_tokens->read(slot) == token) {
          const auto index = _scratch_count->atomic(0u).fetch_add(1u);
          _indices->write(index, slot);
        };
      };
    };
    _gather = device.compile(gather, shader_option);
    // The sources [live, extent) and holes [0, live) are disjoint, so
    // compaction is in-place without a second payload allocation or barriers.
    luisa::compute::Kernel1D compact = [this](UInt capacity, UInt live, UInt extent) {
      const auto source = live + dispatch_x();
      $if(source < extent) {
        const auto token = _tokens->read(source);
        $if(token != 0u) {
          const auto index = _scratch_count->atomic(0u).fetch_add(1u);
          const auto destination = _indices->read(index);
          const auto storage = make_runtime_direct_light_task_storage(_tasks, capacity);
          storage.write(destination, storage.read(source));
          // Only this edge owns a traversal batch. Never read the scratch
          // fields of a task at NEE or INTERSECT_SHADOW.
          $if(token == 3u) {
            const auto batches = batch_storage(capacity);
            batches.write(destination, batches.read(source));
          };
          _tokens->write(destination, token);
          _tokens->write(source, 0u);
        };
      };
    };
    _compact = device.compile(compact, shader_option);
    for (auto stage = 0u; stage < 3u; ++stage) {
      luisa::compute::Kernel1D consume = [this, evaluator, execution_block_size,
                                          stage](BufferFloat4 combined,
                                                 BufferFloat4 light_passes,
                                                 BufferFloat4 volume_guiding_raw,
                                                 Var<RenderKernelParameters> parameters,
                                                 UInt count) noexcept {
        set_block_size(execution_block_size);
        const auto x = dispatch_x();
        $if(x < count) {
          const auto slot = _indices->read(x);
          const auto storage = make_runtime_direct_light_task_storage(
              _tasks, parameters.wavefront_frame_capacity);
          auto task = storage.read(slot);
          UInt next = 0u;
          if (stage == 0u) {
            $if(evaluator.shade_light_nee(task, parameters)) {
              storage.unshadowed_contribution.write(slot, task.unshadowed_contribution);
              storage.shadow_throughput.write(slot, task.shadow_throughput);
              storage.light_shader.write(slot, task.light_shader);
              next = 2u;
            };
          } else if (stage == 1u) {
            const auto batch = evaluator.intersect(task, parameters);
            $if(batch.blocked == 0u) {
              batch_storage(parameters.wavefront_frame_capacity).write(slot, batch);
              next = 3u;
            };
          } else {
            const auto batch =
                batch_storage(parameters.wavefront_frame_capacity).read(slot);
            const auto step = evaluator.shade_shadow(task, batch, parameters);
            $if(step.continue_shadow) {
              storage.ray_minimum.write(slot, task.ray_minimum);
              storage.shadow_throughput.write(slot, task.shadow_throughput);
              storage.transparent_depth.write(slot, task.transparent_depth);
              storage.rng_offset.write(slot, task.rng_offset);
              storage.volume_bounds_bounce.write(slot, task.volume_bounds_bounce);
              next = 2u;
            }
            $elif(step.visible) {
              const auto value =
                  evaluator.contribution(task, task.shadow_throughput, parameters);
              atomic_accumulate_radiance(combined, volume_guiding_raw, task.pixel,
                                         task.pixel * volume_guiding::raw_pixel_stride,
                                         evaluator.volume_guiding, task.path_flags,
                                         task.path_visibility, task.path_depth, value);
              atomic_accumulate_light_passes(light_passes,
                                             task.pixel * light_pass_buffer_count,
                                             evaluator.split(task, value));
            };
          }
          _tokens->write(slot, next);
          $if(x == 0u) { _count->atomic(stage).fetch_sub(count); };
          $if(next != 0u) { _count->atomic(next - 1u).fetch_add(1u); };
        };
      };
      const auto label =
          luisa::string{"wavefront_aux_"} + luisa::string{stage_name(stage)};
      LUISA_INFO("Psycles shadow stage: name='{}' kernel_{:016x} block_size={}.",
                 stage_name(stage), consume.function()->function().hash(),
                 execution_block_size);
      _consumers[stage] = device.compile(
          consume, luisa::compute::coro::detail::coro_scheduler_shader_option(
                       shader_option, label));
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
            Expr<std::uint32_t> runtime_capacity) const noexcept override {
    const auto task_storage =
        make_runtime_direct_light_task_storage(_tasks, runtime_capacity);
    const auto queue_count = Expr<luisa::compute::Buffer<luisa::uint>>{_count};
    const auto slot = queue_count.atomic(3u).fetch_add(1u);
    // Admission control proves slot < capacity. Keep the guard as a local
    // memory-safety invariant: if a producer ever violates its declared
    // bound, the scheduler observes count > capacity and diagnoses it
    // without permitting an out-of-bounds payload write.
    $if(slot < runtime_capacity) {
      task_storage.write(slot, task);
      const auto token = select(1u, 2u, task.constant_light_shader != 0u);
      _tokens->write(slot, token);
      queue_count.atomic(token - 1u).fetch_add(1u);
    };
  }

  void reset(Stream &stream) noexcept override {
    _host_counts = {};
    stream << _count.copy_from(luisa::span{_zeros})
           << _initialize(_capacity).dispatch(_capacity);
  }

  void enqueue_count_readback(Stream &stream) noexcept override {
    stream << _count.copy_to(luisa::span{_host_counts});
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
    const auto live = host_count();
    const auto extent = _host_counts[3];
    const auto plan = cycles_shadow_compaction(extent, live);
    // Cycles decides whether relocation is worthwhile before testing append
    // admission. Unreclaimed holes are not available producer slots; when
    // the tail is insufficient, the generic scheduler drains this side pool.
    if (plan.compact) {
      stream << _scratch_count.copy_from(luisa::span{&_zero, 1u})
             << _gather(0u, live).dispatch(live)
             << _scratch_count.copy_from(luisa::span{&_zero, 1u})
             << _compact(_capacity, live, extent).dispatch(extent - live);
    }
    if (plan.upload_extent) {
      _host_counts[3] = plan.extent;
      stream << _count.view().subview(3u, 1u).copy_from(&_host_counts[3]);
    }
  }

  [[nodiscard]] unsigned host_available_slots() const noexcept override {
    return _capacity - _host_counts[3];
  }

  void prepare_for_producer(Stream &, unsigned required) noexcept override {
    LUISA_ASSERT(required <= host_available_slots(),
                 "Shadow producer exceeds its admitted append storage.");
  }

  void dispatch_stage(unsigned stage, Stream &stream,
                      luisa::compute::BufferView<luisa::float4> combined,
                      luisa::compute::BufferView<luisa::float4>,
                      luisa::compute::BufferView<luisa::float4>,
                      luisa::compute::BufferView<luisa::float4> light_passes,
                      luisa::compute::BufferView<luisa::uint>,
                      luisa::compute::BufferView<luisa::float4> volume_guiding_raw,
                      luisa::compute::BufferView<luisa::uint>,
                      luisa::compute::BufferView<luisa::float4>, const std::uint32_t &,
                      const std::uint32_t &, luisa::compute::BufferView<luisa::float4>,
                      luisa::compute::BufferView<float>,
                      const RenderKernelParameters &parameters) noexcept override {
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
    stream << _scratch_count.copy_from(luisa::span{&_zero, 1u})
           << _gather(stage + 1u, _host_counts[3]).dispatch(_host_counts[3])
           << _consumers[stage](combined, light_passes, volume_guiding_raw,
                                consumer_parameters, count)
                  .dispatch(count);
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
make_direct_light_task_queue(luisa::compute::Device &device,
                             DirectLightTaskEvaluator evaluator, std::uint32_t capacity,
                             std::uint32_t execution_block_size,
                             const luisa::compute::ShaderOption &shader_option) noexcept {
  LUISA_ASSERT(capacity != 0u, "Direct-light task queue capacity must be positive.");
  auto queue = std::make_shared<DirectLightTaskQueue>(
      device, std::move(evaluator), capacity, execution_block_size, shader_option);
  const auto storage_words =
      luisa::compute::SOAView<DirectLightTaskCall>::compute_soa_size(capacity);
  const auto batch_words =
      luisa::compute::SOAView<ShadowIntersectionBatchCall>::compute_soa_size(capacity);
  LUISA_INFO("Psycles direct-light shadow queue: capacity={} "
             "payload_fields={} payload_aos_bytes={} task_soa_bytes={} "
             "traversal_soa_bytes={} queue_bytes={} stages=3.",
             capacity, luisa::compute::Type::of<DirectLightTaskCall>()->members().size(),
             luisa::compute::Type::of<DirectLightTaskCall>()->size(),
             storage_words * sizeof(luisa::uint), batch_words * sizeof(luisa::uint),
             (2ull * capacity + 5u) * sizeof(luisa::uint));
  return {.sink = queue, .work = std::move(queue)};
}

DirectLightTaskQueueBinding
make_direct_light_task_queue(luisa::compute::Device &device,
                             const PathKernelConfig &config, std::uint32_t capacity,
                             std::uint32_t execution_block_size,
                             const luisa::compute::ShaderOption &shader_option) noexcept {
  return make_direct_light_task_queue(device, make_direct_light_task_evaluator(config),
                                      capacity, execution_block_size, shader_option);
}

} // namespace psycles::luisa_backend::detail

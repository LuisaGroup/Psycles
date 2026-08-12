#include "path_kernel_direct_light_queue.h"

#include "path_kernel_transitions.h"

#include <array>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/soa.h>

namespace psycles::luisa_backend::detail {
namespace {

using DirectLightConsumerShader =
    luisa::compute::Shader1D<luisa::compute::Buffer<luisa::float4>,
                             luisa::compute::Buffer<luisa::float4>,
                             luisa::compute::Buffer<luisa::float4>,
                             RenderKernelParameters, std::uint32_t>;

class DirectLightTaskQueue final : public RenderAuxiliaryWork,
                                   public DirectLightTaskSink {

private:
  luisa::compute::SOA<DirectLightTaskCall> _tasks;
  luisa::compute::Buffer<luisa::uint> _count;
  DirectLightConsumerShader _consumer;
  std::array<luisa::compute::coro::WavefrontCoroAuxiliaryProducer, 1u>
      _producers;
  std::uint32_t _capacity{};
  std::uint32_t _host_count{};
  std::uint32_t _zero{};

public:
  DirectLightTaskQueue(
      luisa::compute::Device &device, const PathKernelConfig &config,
      std::uint32_t capacity,
      const luisa::compute::ShaderOption &shader_option) noexcept
      : _tasks{device.create_soa<DirectLightTaskCall>(capacity)},
        _count{device.create_buffer<luisa::uint>(1u)},
        _producers{{{.continuation = path_transition::shade_surface,
                     .max_emitted_per_invocation = 1u}}},
        _capacity{capacity} {
    auto evaluator = make_direct_light_task_evaluator(config);
    auto *tasks = &_tasks;
    luisa::compute::Kernel1D consume =
        [tasks, evaluator = std::move(evaluator)](
            BufferFloat4 combined, BufferFloat4 light_passes,
            BufferFloat4 volume_guiding_raw,
            Var<RenderKernelParameters> parameters, UInt count) noexcept {
          const auto x = dispatch_x();
          $if(x < count) {
            const auto task_storage =
                Expr<luisa::compute::SOA<DirectLightTaskCall>>{*tasks};
            evaluator.emit_atomic(task_storage.read(x),
                                  {.combined = combined,
                                   .light_passes = light_passes,
                                   .volume_guiding_raw = volume_guiding_raw},
                                  parameters);
          };
        };
    _consumer = device.compile(
        consume, luisa::compute::coro::detail::coro_scheduler_shader_option(
                     shader_option, "wavefront_aux_direct_light"));
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return "direct_light_shadow";
  }

  [[nodiscard]] std::uint32_t capacity() const noexcept override {
    return _capacity;
  }

  [[nodiscard]] luisa::span<
      const luisa::compute::coro::WavefrontCoroAuxiliaryProducer>
  producers() const noexcept override {
    return _producers;
  }

  void emit(Var<DirectLightTaskCall> task) const noexcept override {
    const auto task_storage =
        Expr<luisa::compute::SOA<DirectLightTaskCall>>{_tasks};
    const auto queue_count = Expr<luisa::compute::Buffer<luisa::uint>>{_count};
    const auto slot = queue_count.atomic(0u).fetch_add(1u);
    // Admission control proves slot < capacity. Keep the guard as a local
    // memory-safety invariant: if a producer ever violates its declared
    // bound, the scheduler observes count > capacity and diagnoses it
    // without permitting an out-of-bounds payload write.
    $if(slot < _capacity) { task_storage.write(slot, task); };
  }

  void reset(Stream &stream) noexcept override {
    _host_count = 0u;
    stream << _count.copy_from(luisa::span{&_zero, 1u});
  }

  void enqueue_count_readback(Stream &stream) noexcept override {
    stream << _count.copy_to(luisa::span{&_host_count, 1u});
  }

  [[nodiscard]] std::uint32_t host_count() const noexcept override {
    return _host_count;
  }

  void dispatch(Stream &stream,
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
    const auto count = _host_count;
    LUISA_ASSERT(count != 0u && count <= _capacity,
                 "Invalid direct-light auxiliary dispatch count {} "
                 "for capacity {}.",
                 count, _capacity);
    stream << _consumer(combined, light_passes, volume_guiding_raw, parameters,
                        count)
                  .dispatch(count)
           << _count.copy_from(luisa::span{&_zero, 1u});
    _host_count = 0u;
  }
};

} // namespace

DirectLightTaskQueueBinding make_direct_light_task_queue(
    luisa::compute::Device &device, const PathKernelConfig &config,
    std::uint32_t capacity,
    const luisa::compute::ShaderOption &shader_option) noexcept {
  LUISA_ASSERT(capacity != 0u,
               "Direct-light task queue capacity must be positive.");
  auto queue = std::make_shared<DirectLightTaskQueue>(device, config, capacity,
                                                      shader_option);
  const auto storage_words =
      luisa::compute::SOAView<DirectLightTaskCall>::compute_soa_size(capacity);
  LUISA_INFO("Psycles direct-light shadow queue: capacity={} "
             "payload_fields={} payload_aos_bytes={} storage_bytes={}.",
             capacity,
             luisa::compute::Type::of<DirectLightTaskCall>()->members().size(),
             luisa::compute::Type::of<DirectLightTaskCall>()->size(),
             storage_words * sizeof(luisa::uint));
  return {.sink = queue, .work = std::move(queue)};
}

} // namespace psycles::luisa_backend::detail

#include "path_kernel_caustics_queue.h"

#include "path_kernel_builder.h"
#include "path_kernel_surface_buckets.h"

#include <luisa/coro/schedulers/wavefront.h>

#include <algorithm>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

using luisa::compute::Buffer;
using luisa::compute::BufferFloat4;
using luisa::compute::BufferView;
using luisa::compute::Callable;
using luisa::compute::CommandList;
using luisa::compute::Device;
using luisa::compute::Kernel1D;
using luisa::compute::Shader1D;
using luisa::compute::ShaderOption;
using luisa::compute::Stream;
using luisa::compute::UInt;
using luisa::compute::Var;
using luisa::compute::def;
using luisa::compute::dispatch_x;
using luisa::compute::set_block_size;

class CausticsTaskQueue final : public RenderAuxiliaryWork,
                                public CausticsTaskSink {
private:
  CausticsTaskEvaluator _evaluator;
  std::uint32_t _capacity{};
  luisa::uint _warp_size{};
  Buffer<CausticTaskCall> _tasks;
  // _count[0]: total emitted tasks; _count[1]: overflow diagnostic counter.
  Buffer<luisa::uint> _count;
  std::uint32_t _host_count{};
  luisa::vector<luisa::compute::coro::WavefrontCoroAuxiliaryProducer> _producers;
  Shader1D<> _clear_count;
  Shader1D<luisa::compute::Buffer<luisa::float4>,
           luisa::compute::Buffer<luisa::float4>,
           luisa::compute::Buffer<luisa::float4>,
           RenderKernelParameters,
           luisa::uint>
      _consumer;

public:
  CausticsTaskQueue(Device &device, CausticsTaskEvaluator evaluator,
                    std::uint32_t capacity, std::uint32_t execution_block_size,
                    const ShaderOption &shader_option,
                    std::span<const std::string_view> producer_continuations) noexcept
      : _evaluator{std::move(evaluator)},
        _capacity{capacity},
        _warp_size{device.compute_warp_size()},
        _tasks{device.create_buffer<CausticTaskCall>(capacity)},
        _count{device.create_buffer<luisa::uint>(2u)} {
    _producers.reserve(producer_continuations.size());
    for (const auto continuation : producer_continuations) {
      _producers.push_back(
          {.continuation = luisa::string{continuation},
           .max_emitted_per_invocation = 1u});
    }
    Kernel1D clear_kernel{[&]() noexcept {
      _count->write(0u, 0u);
      _count->write(1u, 0u);
    }};
    _clear_count = device.compile(clear_kernel, shader_option);

    Kernel1D consumer_kernel{
        [&](BufferFloat4 combined, BufferFloat4 light_passes,
            BufferFloat4 volume_guiding_raw,
            Var<RenderKernelParameters> parameters,
            UInt active_count) noexcept {
          set_block_size(execution_block_size);
          const auto x = dispatch_x();
          $if(x < active_count) {
            const auto task = _tasks->read(x);
            const DirectLightTaskFilm film{
                .combined = combined,
                .light_passes = light_passes,
                .volume_guiding_raw = volume_guiding_raw};
            _evaluator.evaluate_and_accumulate_atomic(task, film, parameters);
          };
        }};
    const auto consumer_option =
        luisa::compute::coro::detail::coro_scheduler_shader_option(
            shader_option, "wavefront_aux_caustics_manifold");
    _consumer = device.compile(consumer_kernel, consumer_option);
  }

  void emit(Var<CausticTaskCall> task,
            Expr<std::uint32_t> runtime_capacity) const noexcept override {
    UInt slot = 0u;
    if (_warp_size == 32u) {
      const auto lane_prefix = luisa::compute::warp_prefix_count_bits(true);
      const auto warp_total = luisa::compute::warp_active_count_bits(true);
      UInt warp_base = 0u;
      $if(lane_prefix == 0u) {
        warp_base = _count->atomic(0u).fetch_add(warp_total);
      };
      warp_base = luisa::compute::warp_read_first_active_lane(warp_base);
      slot = warp_base + lane_prefix;
    } else {
      slot = _count->atomic(0u).fetch_add(1u);
    }
    $if(slot < runtime_capacity) {
      _tasks->write(slot, task);
    }
    $else {
      _count->atomic(1u).fetch_add(1u);
    };
  }

  [[nodiscard]] luisa::string_view name() const noexcept override {
    return "caustics_manifold";
  }

  [[nodiscard]] std::uint32_t capacity() const noexcept override {
    return _capacity;
  }

  [[nodiscard]] luisa::span<const luisa::compute::coro::WavefrontCoroAuxiliaryProducer>
  producers() const noexcept override {
    return _producers;
  }

  void reset(Stream &stream) noexcept override {
    auto commands = CommandList::create(1u);
    encode_reset(stream, commands);
    stream << commands.commit();
  }

  void encode_reset(Stream &, CommandList &commands) noexcept override {
    _host_count = 0u;
    commands << _clear_count().dispatch(1u);
  }

  void enqueue_count_readback(Stream &stream) noexcept override {
    auto commands = CommandList::create(1u);
    encode_count_readback(stream, commands);
    stream << commands.commit();
  }

  void encode_count_readback(Stream &, CommandList &commands) noexcept override {
    commands << _count.view(0u, 1u).copy_to(&_host_count);
  }

  [[nodiscard]] std::uint32_t host_count() const noexcept override {
    return std::min(_host_count, _capacity);
  }

  [[nodiscard]] unsigned stage_count() const noexcept override {
    return 1u;
  }

  [[nodiscard]] luisa::string_view
  stage_name(unsigned) const noexcept override {
    return "caustics_manifold";
  }

  [[nodiscard]] unsigned stage_host_count(unsigned) const noexcept override {
    return host_count();
  }

  [[nodiscard]] unsigned host_available_slots() const noexcept override {
    const auto active = host_count();
    return _capacity > active ? (_capacity - active) : 0u;
  }

  void encode_prepare_for_admission(Stream &,
                                    CommandList &) noexcept override {}

  void encode_prepare_for_producer(Stream &, CommandList &,
                                   unsigned) noexcept override {}

  void encode_dispatch_stage(
      unsigned, Stream &, CommandList &commands,
      BufferView<luisa::float4> combined,
      BufferView<luisa::float4> normal,
      BufferView<luisa::float4> albedo,
      BufferView<luisa::float4> light_passes,
      BufferView<luisa::uint> sample_count,
      BufferView<luisa::float4> volume_guiding_raw,
      BufferView<luisa::uint> volume_guiding_count,
      BufferView<luisa::float4> path_trace,
      const std::uint32_t &sample_first,
      const std::uint32_t &samples,
      BufferView<luisa::float4> sobol_table,
      BufferView<float> filter_table,
      const RenderKernelParameters &parameters) noexcept override {
    static_cast<void>(normal);
    static_cast<void>(albedo);
    static_cast<void>(sample_count);
    static_cast<void>(volume_guiding_count);
    static_cast<void>(path_trace);
    static_cast<void>(sample_first);
    static_cast<void>(samples);
    static_cast<void>(sobol_table);
    static_cast<void>(filter_table);
    const auto active_count = host_count();
    _host_count = 0u;
    if (active_count != 0u) {
      commands << _consumer(combined, light_passes, volume_guiding_raw,
                            parameters, active_count)
                      .dispatch(active_count);
    }
    commands << _clear_count().dispatch(1u);
  }

  void dispatch_stage(
      unsigned stage, Stream &stream,
      BufferView<luisa::float4> combined,
      BufferView<luisa::float4> normal,
      BufferView<luisa::float4> albedo,
      BufferView<luisa::float4> light_passes,
      BufferView<luisa::uint> sample_count,
      BufferView<luisa::float4> volume_guiding_raw,
      BufferView<luisa::uint> volume_guiding_count,
      BufferView<luisa::float4> path_trace,
      const std::uint32_t &sample_first,
      const std::uint32_t &samples,
      BufferView<luisa::float4> sobol_table,
      BufferView<float> filter_table,
      const RenderKernelParameters &parameters) noexcept override {
    auto commands = CommandList::create(2u);
    encode_dispatch_stage(stage, stream, commands, combined, normal, albedo,
                          light_passes, sample_count, volume_guiding_raw,
                          volume_guiding_count, path_trace, sample_first,
                          samples, sobol_table, filter_table, parameters);
    stream << commands.commit();
  }

  void dispatch(
      Stream &stream,
      BufferView<luisa::float4> combined,
      BufferView<luisa::float4> normal,
      BufferView<luisa::float4> albedo,
      BufferView<luisa::float4> light_passes,
      BufferView<luisa::uint> sample_count,
      BufferView<luisa::float4> volume_guiding_raw,
      BufferView<luisa::uint> volume_guiding_count,
      BufferView<luisa::float4> path_trace,
      const std::uint32_t &sample_first,
      const std::uint32_t &samples,
      BufferView<luisa::float4> sobol_table,
      BufferView<float> filter_table,
      const RenderKernelParameters &parameters) noexcept override {
    dispatch_stage(0u, stream, combined, normal, albedo, light_passes,
                   sample_count, volume_guiding_raw, volume_guiding_count,
                   path_trace, sample_first, samples, sobol_table,
                   filter_table, parameters);
  }
};

} // namespace

CausticsTaskQueueBinding make_caustics_task_queue(
    Device &device, const PathKernelConfig &config, std::uint32_t capacity,
    std::uint32_t execution_block_size,
    const ShaderOption &shader_option) noexcept {
  LUISA_ASSERT(capacity != 0u,
               "Caustics task queue capacity must be positive.");
  auto evaluator = make_caustics_task_evaluator(config);
  const auto bucket_plan = make_path_surface_bucket_plan(config);
  const auto producers = bucket_plan.producer_continuations();
  auto queue = std::make_shared<CausticsTaskQueue>(
      device, std::move(evaluator), capacity, execution_block_size,
      shader_option, producers);
  LUISA_INFO("Psycles caustics manifold queue: method={} unbiased={} "
             "max_trials={} capacity={} payload_fields={} "
             "payload_aos_bytes={} stages=1 spectral={}.",
             config.caustics_is_sms() ? "sms" : "mnee",
             config.sms_unbiased,
             config.sms_max_trials,
             capacity,
             luisa::compute::Type::of<CausticTaskCall>()->members().size(),
             luisa::compute::Type::of<CausticTaskCall>()->size(),
             config.spectral_rendering);
  return {.sink = queue, .work = std::move(queue)};
}

} // namespace psycles::luisa_backend::detail

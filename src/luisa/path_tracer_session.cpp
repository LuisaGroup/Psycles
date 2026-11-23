#include "path_tracer_internal.h"
#include "sample_dispatch_partition.h"
#include "virtual_texture_cache.h"

#include <luisa/core/logging.h>

#include <psycles/core/parallel.h>
#include <psycles/luisa/volume_guiding.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool film_shader_cache_enabled() noexcept {
  const auto *disabled = std::getenv("PSYCLES_DISABLE_SHADER_CACHE");
  return disabled == nullptr || std::string_view{disabled} != "1";
}

[[nodiscard]] Float3 safe_divide_even_color_device(
    const Float4 &numerator,
    const Float4 &denominator) noexcept {
  auto x = def(0.0f);
  $if (denominator.x != 0.0f) {
    x = numerator.x / denominator.x;
  };
  auto y = def(0.0f);
  $if (denominator.y != 0.0f) {
    y = numerator.y / denominator.y;
  };
  auto z = def(0.0f);
  $if (denominator.z != 0.0f) {
    z = numerator.z / denominator.z;
  };
  $if (denominator.x == 0.0f) {
    $if (denominator.y == 0.0f) {
      x = z;
      y = z;
    } $elif (denominator.z == 0.0f) {
      x = y;
      z = y;
    } $else {
      x = 0.5f * (y + z);
    };
  } $elif (denominator.y == 0.0f) {
    $if (denominator.z == 0.0f) {
      y = x;
      z = x;
    } $else {
      y = 0.5f * (x + z);
    };
  } $elif (denominator.z == 0.0f) {
    z = 0.5f * (x + y);
  };
  return make_float3(x, y, z);
}

[[nodiscard]] std::shared_ptr<const FilmPassAccessorGpu>
compile_film_pass_accessor(luisa::compute::Device &device) {
  Kernel1D verify_kernel =
      [](BufferUInt sample_count,
         BufferUInt mismatch_flag,
         UInt count,
         UInt expected_samples) noexcept {
        set_block_size(256u);
        const auto i = dispatch_x();
        $if (i >= count) {
          $return();
        };
        $if (sample_count.read(i) != expected_samples) {
          mismatch_flag.atomic(0u).fetch_or(1u);
        };
      };

  Kernel1D resolve_kernel =
      [](BufferFloat4 combined,
         BufferFloat4 normal,
         BufferFloat4 albedo,
         BufferFloat4 light_passes,
         BufferUInt sample_count,
         BufferFloat resolved_out,
         UInt out_offset,
         UInt count,
         UInt pass_kind,
         UInt channels,
         Float exposure) noexcept {
        set_block_size(256u);
        const auto i = dispatch_x();
        $if (i >= count) {
          $return();
        };
        const auto samples = sample_count.read(i);
        const auto denominator = cast<float>(max(samples, 1u));
        const auto inv_denominator = 1.0f / denominator;
        const auto exposure_scale = exposure * inv_denominator;
        const auto light_pass_base = i * light_pass_buffer_count;
        const auto read_light_pass =
            [&](LightPassBuffer kind) noexcept -> Float4 {
          return light_passes.read(light_pass_base + light_pass_index(kind));
        };
        const auto divided_light_pass =
            [&](LightPassBuffer kind, const Float4 &color) noexcept -> Float4 {
          const auto divided =
              safe_divide_even_color_device(read_light_pass(kind), color) *
              exposure;
          return make_float4(divided, 1.0f);
        };

        auto value = def(make_float4(0.0f));
        $switch (pass_kind) {
          $case (static_cast<std::uint32_t>(PassKind::combined)) {
            const auto c = combined.read(i);
            value = make_float4(
                c.xyz() * exposure_scale,
                clamp(1.0f - c.w * inv_denominator, 0.0f, 1.0f));
          };
          $case (static_cast<std::uint32_t>(PassKind::normal)) {
            value = normal.read(i) * inv_denominator;
          };
          $case (static_cast<std::uint32_t>(PassKind::denoising_normal)) {
            value = normal.read(i) * inv_denominator;
          };
          $case (static_cast<std::uint32_t>(PassKind::albedo)) {
            value = albedo.read(i) * inv_denominator;
          };
          $case (static_cast<std::uint32_t>(PassKind::denoising_albedo)) {
            value = albedo.read(i) * inv_denominator;
          };
          $case (static_cast<std::uint32_t>(PassKind::glossy_color)) {
            value = read_light_pass(LightPassBuffer::glossy_color) *
                    inv_denominator;
          };
          $case (static_cast<std::uint32_t>(PassKind::transmission_color)) {
            value = read_light_pass(LightPassBuffer::transmission_color) *
                    inv_denominator;
          };
          $case (static_cast<std::uint32_t>(PassKind::emission)) {
            value = read_light_pass(LightPassBuffer::emission) * exposure_scale;
          };
          $case (static_cast<std::uint32_t>(PassKind::environment)) {
            value =
                read_light_pass(LightPassBuffer::environment) * exposure_scale;
          };
          $case (static_cast<std::uint32_t>(PassKind::diffuse_direct)) {
            value = divided_light_pass(LightPassBuffer::diffuse_direct,
                                       albedo.read(i));
          };
          $case (static_cast<std::uint32_t>(PassKind::diffuse_indirect)) {
            value = divided_light_pass(LightPassBuffer::diffuse_indirect,
                                       albedo.read(i));
          };
          $case (static_cast<std::uint32_t>(PassKind::glossy_direct)) {
            value = divided_light_pass(
                LightPassBuffer::glossy_direct,
                read_light_pass(LightPassBuffer::glossy_color));
          };
          $case (static_cast<std::uint32_t>(PassKind::glossy_indirect)) {
            value = divided_light_pass(
                LightPassBuffer::glossy_indirect,
                read_light_pass(LightPassBuffer::glossy_color));
          };
          $case (static_cast<std::uint32_t>(PassKind::transmission_direct)) {
            value = divided_light_pass(
                LightPassBuffer::transmission_direct,
                read_light_pass(LightPassBuffer::transmission_color));
          };
          $case (static_cast<std::uint32_t>(PassKind::transmission_indirect)) {
            value = divided_light_pass(
                LightPassBuffer::transmission_indirect,
                read_light_pass(LightPassBuffer::transmission_color));
          };
          $case (static_cast<std::uint32_t>(PassKind::volume_direct)) {
            value = read_light_pass(LightPassBuffer::volume_direct) *
                    exposure_scale;
          };
          $case (static_cast<std::uint32_t>(PassKind::volume_indirect)) {
            value = read_light_pass(LightPassBuffer::volume_indirect) *
                    exposure_scale;
          };
          $case (static_cast<std::uint32_t>(PassKind::sample_count)) {
            value = make_float4(cast<float>(samples));
          };
          $default {};
        };

        const auto dst = out_offset + i * channels;
        $if (channels == 4u) {
          resolved_out.write(dst + 0u, value.x);
          resolved_out.write(dst + 1u, value.y);
          resolved_out.write(dst + 2u, value.z);
          resolved_out.write(dst + 3u, value.w);
        } $elif (channels == 3u) {
          resolved_out.write(dst + 0u, value.x);
          resolved_out.write(dst + 1u, value.y);
          resolved_out.write(dst + 2u, value.z);
        } $elif (channels == 1u) {
          resolved_out.write(dst + 0u, value.x);
        } $else {
          $for (channel, channels) {
            const auto component = select(
                select(select(value.w, value.z, channel == 2u),
                       value.y,
                       channel == 1u),
                value.x,
                channel == 0u);
            resolved_out.write(dst + channel, component);
          };
        };
      };

  Kernel1D clear_film_kernel =
      [](BufferFloat4 combined,
         BufferFloat4 normal,
         BufferFloat4 albedo,
         BufferFloat4 light_passes,
         BufferUInt sample_count,
         BufferFloat4 volume_guiding_raw,
         BufferUInt volume_guiding_denoised,
         BufferUInt volume_guiding_intermediate,
         BufferFloat4 path_trace,
         UInt pixel_count,
         UInt path_trace_count) noexcept {
        set_block_size(256u);
        const auto i = dispatch_x();
        const auto zero4 = make_float4(0.0f);
        $if (i < pixel_count) {
          combined.write(i, zero4);
          normal.write(i, zero4);
          albedo.write(i, zero4);
          sample_count.write(i, 0u);
          const auto light_base = i * light_pass_buffer_count;
          for (std::uint32_t k = 0u; k < light_pass_buffer_count; ++k) {
            light_passes.write(light_base + k, zero4);
          }
          const auto raw_base = i * volume_guiding::raw_pixel_stride;
          for (std::uint32_t k = 0u;
               k < volume_guiding::raw_pixel_stride; ++k) {
            volume_guiding_raw.write(raw_base + k, zero4);
          }
          const auto denoised_base =
              i * volume_guiding::denoised_pixel_stride;
          for (std::uint32_t k = 0u;
               k < volume_guiding::denoised_pixel_stride; ++k) {
            volume_guiding_denoised.write(denoised_base + k, 0u);
            volume_guiding_intermediate.write(denoised_base + k, 0u);
          }
        };
        $if (i < path_trace_count) {
          path_trace.write(i, zero4);
        };
      };

  const auto enable_cache = film_shader_cache_enabled();
  const luisa::compute::ShaderOption option{.enable_cache = enable_cache};
  auto accessor = std::make_shared<FilmPassAccessorGpu>();
  psycles::parallel_for(3u, [&](std::size_t stage) {
    if (stage == 0u) {
      accessor->verify_sample_count = device.compile(verify_kernel, option);
    } else if (stage == 1u) {
      accessor->resolve_pass = device.compile(resolve_kernel, option);
    } else {
      accessor->clear_film = device.compile(clear_film_kernel, option);
    }
  });
  return accessor;
}

std::mutex &film_pass_accessor_mutex() noexcept {
  static std::mutex mutex;
  return mutex;
}

} // namespace

std::shared_ptr<const FilmPassAccessorGpu>
get_or_compile_film_pass_accessor(
    luisa::compute::Device &device,
    std::shared_ptr<const FilmPassAccessorGpu> &slot) {
  std::scoped_lock lock{film_pass_accessor_mutex()};
  if (!slot) {
    slot = compile_film_pass_accessor(device);
  }
  return slot;
}

void attach_scene_film_pass_accessor(
    luisa::compute::Device &device,
    std::shared_ptr<const FilmPassAccessorGpu> &device_cache,
    std::shared_ptr<const FilmPassAccessorGpu> &scene_slot) {
  std::scoped_lock lock{film_pass_accessor_mutex()};
  if (!device_cache) {
    device_cache = compile_film_pass_accessor(device);
  }
  if (!scene_slot) {
    scene_slot = device_cache;
  }
}

PathDiagnosticBufferLayout path_diagnostic_buffer_layout(
    const LuisaPathTracerOptions &options) noexcept {
  PathDiagnosticBufferLayout layout;
  layout.path_trace_slot_count =
      options.path_trace ? path_trace_schema::slot_count : 0u;
  layout.surface_closure_count_histogram_base = layout.path_trace_slot_count;
  layout.surface_closure_count_histogram_slot_count =
      options.surface_closure_count_histogram
          ? luisa_surface_closure_count_histogram_bin_count
          : 0u;
  if (layout.surface_closure_count_histogram_base >
      std::numeric_limits<std::size_t>::max() -
          layout.surface_closure_count_histogram_slot_count) {
    std::abort();
  }
  layout.allocation_slot_count = std::max<std::size_t>(
      layout.surface_closure_count_histogram_base +
          layout.surface_closure_count_histogram_slot_count,
      1u);
  if (layout.allocation_slot_count >
      std::numeric_limits<std::uint32_t>::max()) {
    std::abort();
  }
  return layout;
}

std::size_t LuisaRenderSession::pixel_count() const noexcept {
  return static_cast<std::size_t>(_window.width) *
         static_cast<std::size_t>(_window.height);
}

void LuisaRenderSession::deliver_path_trace() {
  if (_path_trace_delivered || !_options.path_trace ||
      !_options.path_trace->sink) {
    return;
  }
  luisa::vector<luisa::float4> slots(path_trace_schema::slot_count);
  _stream << _path_trace.view(0u, path_trace_schema::slot_count)
                 .copy_to(luisa::span{slots})
          << synchronize();
  LuisaPathTrace trace{.pixel_x = _options.path_trace->pixel_x,
                       .pixel_y = _options.path_trace->pixel_y,
                       .sample = _options.path_trace->sample};
  for (auto index = std::size_t{0u}; index < slots.size(); ++index) {
    trace.slots[index] = {slots[index].x, slots[index].y, slots[index].z,
                          slots[index].w};
  }
  _options.path_trace->sink->write(trace);
  _path_trace_delivered = true;
}

void LuisaRenderSession::deliver_surface_closure_count_histogram() {
  if (!_options.surface_closure_count_histogram ||
      !_options.surface_closure_count_histogram->sink) {
    return;
  }
  luisa::vector<luisa::float4> bins(
      luisa_surface_closure_count_histogram_bin_count);
  const auto layout = path_diagnostic_buffer_layout(_options);
  _stream << _path_trace
                 .view(layout.surface_closure_count_histogram_base, bins.size())
                 .copy_to(luisa::span{bins})
          << synchronize();

  constexpr auto largest_consecutive_float_integer = 16777216.0f;
  LuisaSurfaceClosureCountHistogram histogram;
  histogram.exact = true;
  for (auto bin = std::size_t{0u}; bin < bins.size(); ++bin) {
    const std::array lanes{bins[bin].x, bins[bin].y, bins[bin].z, bins[bin].w};
    for (const auto value : lanes) {
      const auto lane_exact = std::isfinite(value) && value >= 0.0f &&
                              value < largest_consecutive_float_integer &&
                              std::trunc(value) == value;
      histogram.exact &= lane_exact;
      // Only exact lanes are converted. Besides making an invalid
      // result unusable by construction, this avoids an out-of-range
      // float-to-uint conversion if a corrupted device counter is a
      // large but finite value.
      if (lane_exact) {
        histogram.counts[bin] += static_cast<std::uint64_t>(value);
      }
    }
  }
  _options.surface_closure_count_histogram->sink->write(histogram);
}

void LuisaRenderSession::prepare_sobol_table(std::uint32_t total_samples) {
  const auto sequence_size =
      tabulated_sobol::sequence_size_for_samples(total_samples);
  if (!_sobol_table || _sobol_sequence_size != sequence_size) {
    const auto generated = tabulated_sobol::generate_table(sequence_size);
    luisa::vector<luisa::float4> table;
    table.reserve(generated.size());
    for (const auto sample : generated) {
      table.emplace_back(
          luisa::make_float4(sample.x, sample.y, sample.z, sample.w));
    }
    _sobol_table = _scene->device.create_buffer<luisa::float4>(table.size());
    _stream << _sobol_table.copy_from(luisa::span{table}) << synchronize();
    _sobol_sequence_size = sequence_size;
  }
  _kernel_parameters.sobol_sequence_size = sequence_size;
}

void LuisaRenderSession::prepare_film_pass_accessor() {
  static_cast<void>(get_or_compile_film_pass_accessor(
      _scene->device, _scene->film_pass_accessor));
  std::uint32_t max_channels = 4u;
  for (const auto &pass : _settings.passes) {
    if (supported_pass(pass.kind)) {
      max_channels = std::max(max_channels, pass_channels(pass));
    }
  }
  const auto staging_floats = std::max<std::size_t>(
      pixel_count() * static_cast<std::size_t>(max_channels), 1u);
  if (!_sample_mismatch_flag) {
    _sample_mismatch_flag = _scene->device.create_buffer<luisa::uint>(1u);
  }
  if (!_resolved_pass_staging ||
      _resolved_pass_staging.size() < staging_floats) {
    _resolved_pass_staging =
        _scene->device.create_buffer<float>(staging_floats);
  }
  _resolved_pass_host.reserve(staging_floats);
}

bool LuisaRenderSession::write_passes(contract::OutputSink &output) {
  const auto count = pixel_count();
  if (count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return false;
  }
  const auto &accessor = _scene->film_pass_accessor;
  LUISA_ASSERT(accessor != nullptr,
               "Film pass accessor must be initialized before write_passes.");

  if (count > 0u) {
    const luisa::uint zero = 0u;
    luisa::uint mismatch = 0u;
    _stream << _sample_mismatch_flag.copy_from(luisa::span{&zero, 1u})
            << accessor->verify_sample_count(
                   _sample_count,
                   _sample_mismatch_flag,
                   static_cast<std::uint32_t>(count),
                   _rendered_samples)
                   .dispatch(count)
            << _sample_mismatch_flag.copy_to(luisa::span{&mismatch, 1u})
            << synchronize();
    if (mismatch != 0u) {
      return false;
    }
  }

  output.begin(_settings);
  const auto exposure = _settings.integrator.film_exposure;
  for (const auto &pass : _settings.passes) {
    if (!supported_pass(pass.kind)) {
      continue;
    }
    const auto channels = pass_channels(pass);
    const auto element_count = count * static_cast<std::size_t>(channels);
    if (element_count > _resolved_pass_staging.size()) {
      _resolved_pass_staging = _scene->device.create_buffer<float>(
          std::max<std::size_t>(element_count, 1u));
    }
    _resolved_pass_host.resize(element_count);
    if (element_count > 0u) {
      _stream << accessor->resolve_pass(
                     _combined,
                     _normal,
                     _albedo,
                     _light_passes,
                     _sample_count,
                     _resolved_pass_staging,
                     0u,
                     static_cast<std::uint32_t>(count),
                     static_cast<std::uint32_t>(pass.kind),
                     channels,
                     exposure)
                     .dispatch(count)
              << _resolved_pass_staging.view(0u, element_count)
                     .copy_to(luisa::span{_resolved_pass_host})
              << synchronize();
    }
    output.write(
        PassTile{.pass = pass,
                 .window = _window,
                 .full_extent = _settings.full_extent,
                 .pixels = std::span<const float>{_resolved_pass_host}});
  }
  output.end(_cancelled.load());
  return true;
}

LuisaRenderSession::LuisaRenderSession(std::shared_ptr<LuisaSceneData> scene,
                                       LuisaPathTracerOptions options,
                                       const RenderSettings &settings)
    : _scene{std::move(scene)}, _options{options}, _camera{_scene->camera},
      _stream{_scene->device.create_stream()} {
  initialize(settings);
}

void LuisaRenderSession::reset(const RenderSettings &settings) {
  _cancelled.store(false);
  initialize(settings);
}

bool LuisaRenderSession::render_samples(const SampleRange &samples,
                                        contract::OutputSink &output) {
  _last_render_timings.reset();
  const auto call_begin = std::chrono::steady_clock::now();
  if (_cancelled.load() || samples.count == 0u) {
    return false;
  }
  const auto sample_begin = static_cast<std::uint64_t>(samples.first) +
                            static_cast<std::uint64_t>(samples.offset);
  const auto sample_end =
      sample_begin + static_cast<std::uint64_t>(samples.count);
  if (samples.total == 0u ||
      sample_end > static_cast<std::uint64_t>(samples.total) ||
      (_total_aa_samples != 0u && _total_aa_samples != samples.total)) {
    return false;
  }
  _total_aa_samples = samples.total;
  const auto integration_begin = std::chrono::steady_clock::now();
  prepare_sobol_table(samples.total);
  auto dispatches = _volume_guiding_filter
                        ? SampleDispatchPartition::make_volume_guided(
                              static_cast<std::uint32_t>(sample_begin),
                              samples.count, _options.max_samples_per_dispatch,
                              _rendered_samples, samples.total)
                        : SampleDispatchPartition::make(
                              static_cast<std::uint32_t>(sample_begin),
                              samples.count, _options.max_samples_per_dispatch);
  if (!dispatches) {
    return false;
  }
  std::uint32_t actual_path_dispatches = 0u;
  std::uint32_t actual_vspg_dispatches = 0u;
  std::vector<std::uint32_t> batch_counts_log;
  while (const auto batch = dispatches->next()) {
    if (_cancelled.load()) {
      return false;
    }
    auto row_dispatches = PixelRowDispatchPartition::make(
        _window.width, _window.height, batch->count,
        _options.max_pixel_samples_per_dispatch);
    if (!row_dispatches) {
      return false;
    }
    while (const auto rows = row_dispatches->next()) {
      auto parameters = _kernel_parameters;
      parameters.window_y += rows->first_row;
      const auto pixel_offset = static_cast<std::size_t>(rows->first_pixel);
      const auto pixel_count = static_cast<std::size_t>(rows->pixel_count);
      const auto light_offset = pixel_offset * light_pass_buffer_count;
      const auto light_count = pixel_count * light_pass_buffer_count;
      const auto volume_raw_offset =
          pixel_offset * volume_guiding::raw_pixel_stride;
      const auto volume_raw_count =
          pixel_count * volume_guiding::raw_pixel_stride;
      const auto volume_denoised_offset =
          pixel_offset * volume_guiding::denoised_pixel_stride;
      const auto volume_denoised_count =
          pixel_count * volume_guiding::denoised_pixel_stride;
      _render_executor.dispatch(
          _stream,
          PathKernelDispatch{
              .combined = _combined.view(pixel_offset, pixel_count),
              .normal = _normal.view(pixel_offset, pixel_count),
              .albedo = _albedo.view(pixel_offset, pixel_count),
              .light_passes = _light_passes.view(light_offset, light_count),
              .sample_count = _sample_count.view(pixel_offset, pixel_count),
              .volume_guiding_raw =
                  _volume_guiding_raw.view(volume_raw_offset, volume_raw_count),
              .volume_guiding_denoised = _volume_guiding_denoised.view(
                  volume_denoised_offset, volume_denoised_count),
              .path_trace = _path_trace.view(),
              .sample_first = batch->first,
              .samples = batch->count,
              .sobol_table = _sobol_table.view(),
              .filter_table = _pixel_filter_table.view(),
              .parameters = parameters,
              .width = _window.width,
              .height = rows->row_count,
              .pixel_count = rows->pixel_count});
      _stream << synchronize();
      ++actual_path_dispatches;
      batch_counts_log.push_back(batch->count);
    }
    _rendered_samples += batch->count;
    if (batch->filter_volume_guiding) {
      _volume_guiding_filter->dispatch(
          _stream, _volume_guiding_raw, _sample_count,
          _volume_guiding_intermediate, _volume_guiding_denoised,
          std::max(_window.width, 1u), std::max(_window.height, 1u));
      _stream << synchronize();
      ++actual_vspg_dispatches;
    }
  }
  const auto integration_end = std::chrono::steady_clock::now();
  if (const auto *raw_util = std::getenv("PSYCLES_PRINT_RENDER_UTILIZATION");
      raw_util != nullptr && std::string_view{raw_util} == "1") {
    std::printf(
        "PSYCLES_SESSION_DISPATCH_SUMMARY_JSON: "
        "{\"total_samples\":%u,\"max_samples_per_dispatch\":%u,"
        "\"max_pixel_samples_per_dispatch\":%u,\"volume_guided\":%s,"
        "\"path_dispatches\":%u,\"vspg_filter_dispatches\":%u,"
        "\"batch_sample_counts\":[",
        samples.count, _options.max_samples_per_dispatch,
        _options.max_pixel_samples_per_dispatch,
        _volume_guiding_filter ? "true" : "false", actual_path_dispatches,
        actual_vspg_dispatches);
    for (std::size_t bi = 0u; bi < batch_counts_log.size(); ++bi) {
      std::printf("%s%u", bi == 0u ? "" : ",", batch_counts_log[bi]);
    }
    std::printf("]}\n");
    std::fflush(stdout);
  }
  if (_cancelled.load()) {
    return false;
  }
  if (_options.path_trace && !_path_trace_delivered &&
      static_cast<std::uint64_t>(_options.path_trace->sample) >= sample_begin &&
      static_cast<std::uint64_t>(_options.path_trace->sample) < sample_end) {
    deliver_path_trace();
  }
  const auto pass_output_begin = std::chrono::steady_clock::now();
  if (!write_passes(output)) {
    return false;
  }
  const auto pass_output_end = std::chrono::steady_clock::now();
  deliver_surface_closure_count_histogram();
  const auto call_end = std::chrono::steady_clock::now();
  _last_render_timings = contract::RenderSampleTimings{
      .integration_seconds =
          std::chrono::duration<double>(integration_end - integration_begin)
              .count(),
      .pass_output_seconds =
          std::chrono::duration<double>(pass_output_end - pass_output_begin)
              .count(),
      .render_call_seconds =
          std::chrono::duration<double>(call_end - call_begin).count()};
  if (_options.virtual_texture_cache.report_stats &&
      _scene && _scene->virtual_texture_cache) {
    LUISA_INFO("PSYCLES_VT_CACHE_STATS_JSON={}",
               _scene->virtual_texture_cache->stats_json());
  }
  return true;
}

void LuisaRenderSession::cancel() noexcept { _cancelled.store(true); }

std::optional<contract::RenderSampleTimings>
LuisaRenderSession::last_render_timings() const noexcept {
  return _last_render_timings;
}

} // namespace psycles::luisa_backend::detail

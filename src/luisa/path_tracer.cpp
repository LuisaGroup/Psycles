#include "path_tracer_internal.h"
#include "path_tracer_backend_policy.h"

#include <charconv>
#include <cstdlib>
#include <string_view>

namespace psycles::luisa_backend {

using namespace detail;

namespace {

[[nodiscard]] std::optional<std::uint32_t>
env_max_pixel_samples_per_dispatch() noexcept {
    const auto *raw = std::getenv("PSYCLES_MAX_PIXEL_SAMPLES_PER_DISPATCH");
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    const std::string_view sv{raw};
    std::uint32_t val = 0u;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
    if (ec != std::errc{} || ptr != sv.data() + sv.size() || val == 0u) {
        return std::nullopt;
    }
    return val;
}

[[nodiscard]] std::optional<bool>
env_bool_override(const char *name) noexcept {
    const auto *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    const std::string_view sv{raw};
    if (sv == "1" || sv == "true" || sv == "on") {
        return true;
    }
    if (sv == "0" || sv == "false" || sv == "off") {
        return false;
    }
    return std::nullopt;
}

template<typename T>
[[nodiscard]] std::optional<T>
env_unsigned_override(const char *name) noexcept {
    const auto *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    const std::string_view sv{raw};
    T val{};
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
    if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
        return std::nullopt;
    }
    return val;
}

void apply_virtual_texture_env_overrides(
    LuisaVirtualTextureCacheOptions &vt) noexcept {
    if (const auto enabled = env_bool_override("PSYCLES_VIRTUAL_TEXTURE_CACHE")) {
        vt.enabled = *enabled;
    }
    if (const auto budget_mb =
            env_unsigned_override<std::uint64_t>("PSYCLES_VIRTUAL_TEXTURE_CACHE_MB");
        budget_mb && *budget_mb > 0ull) {
        vt.gpu_cache_budget_bytes = *budget_mb * 1024ull * 1024ull;
    }
    if (const auto budget_bytes =
            env_unsigned_override<std::uint64_t>("PSYCLES_VIRTUAL_TEXTURE_CACHE_BYTES");
        budget_bytes && *budget_bytes > 0ull) {
        vt.gpu_cache_budget_bytes = *budget_bytes;
    }
    if (const auto tile_size =
            env_unsigned_override<std::uint32_t>("PSYCLES_VIRTUAL_TEXTURE_TILE_SIZE");
        tile_size && valid_luisa_virtual_texture_tile_size(*tile_size)) {
        vt.physical_tile_size = *tile_size;
    }
    if (const auto host_mb =
            env_unsigned_override<std::uint64_t>("PSYCLES_VIRTUAL_TEXTURE_HOST_CACHE_MB")) {
        vt.host_decoded_cache_budget_bytes = *host_mb * 1024ull * 1024ull;
    }
    if (const auto stats = env_bool_override("PSYCLES_VIRTUAL_TEXTURE_STATS")) {
        vt.report_stats = *stats;
    }
}

[[nodiscard]] bool valid_scheduler_options(
    const LuisaPathTracerOptions &options) noexcept {
    if (options.virtual_texture_cache.enabled &&
        (!valid_luisa_virtual_texture_tile_size(
             options.virtual_texture_cache.physical_tile_size) ||
         options.virtual_texture_cache.gpu_cache_budget_bytes < 64ull * 1024ull)) {
        return false;
    }
    switch (options.scheduler) {
        case LuisaPathScheduler::megakernel:
        case LuisaPathScheduler::megakernel_per_sample:
            return true;
        case LuisaPathScheduler::wavefront:
        case LuisaPathScheduler::wavefront_staged:
            return options.wavefront_frame_capacity != 0u &&
                   valid_luisa_wavefront_execution_block_size(
                       options.wavefront_execution_block_size);
        case LuisaPathScheduler::wavefront_graph:
            return options.wavefront_frame_capacity != 0u &&
                   options.wavefront_counter_readback_batch_size != 0u &&
                   options.wavefront_counter_readback_pipeline_depth != 0u &&
                   valid_luisa_wavefront_execution_block_size(
                       options.wavefront_execution_block_size);
        case LuisaPathScheduler::persistent:
            return valid_luisa_persistent_scheduler_shape(
                options.persistent_worker_count,
                options.persistent_block_size,
                options.persistent_fetch_size);
    }
    return false;
}

}// namespace

LuisaPathTracerBackend::LuisaPathTracerBackend(
    luisa::compute::Device device,
    LuisaPathTracerOptions options) noexcept
    : _device{std::move(device)},
      _options{options} {
    if (_device) {
        const bool is_wavefront =
            _options.scheduler == LuisaPathScheduler::wavefront ||
            _options.scheduler == LuisaPathScheduler::wavefront_staged ||
            _options.scheduler == LuisaPathScheduler::wavefront_graph;
        _options.max_pixel_samples_per_dispatch =
            std::min(
                _options.max_pixel_samples_per_dispatch,
                backend_max_pixel_samples_per_dispatch(
                    _device.backend_name(), is_wavefront));
    }
    if (const auto env_limit = env_max_pixel_samples_per_dispatch()) {
        _options.max_pixel_samples_per_dispatch =
            std::min(_options.max_pixel_samples_per_dispatch, *env_limit);
    }
    apply_virtual_texture_env_overrides(_options.virtual_texture_cache);
}

std::unique_ptr<contract::RenderSession>
LuisaPathTracerBackend::create_session(
    const contract::CompiledScene &scene,
    const RenderSettings &settings) {
    const auto &compiled =
        static_cast<const LuisaCompiledScene &>(scene);
    if (settings.full_extent.width == 0u ||
        settings.full_extent.height == 0u ||
        _options.max_samples_per_dispatch == 0u ||
        _options.max_pixel_samples_per_dispatch == 0u ||
        !valid_scheduler_options(_options)) {
        return nullptr;
    }
    if (const auto &trace = _options.path_trace) {
        if (!trace->sink ||
            trace->pixel_x >= settings.full_extent.width ||
            trace->pixel_y >= settings.full_extent.height) {
            return nullptr;
        }
        const auto window = effective_window(settings);
        const auto raster_y =
            settings.full_extent.height - 1u -
            trace->pixel_y;
        if (
            trace->pixel_x < window.x ||
            trace->pixel_x >= window.x + window.width ||
            raster_y < window.y ||
            raster_y >= window.y + window.height) {
            return nullptr;
        }
    }
    if (
        !_options.next_event_estimation &&
        settings.integrator.direct_light_sampling !=
            contract::DirectLightSampling::
                forward_path_tracing) {
        return nullptr;
    }
    if (
        settings.integrator.direct_light_sampling ==
            contract::DirectLightSampling::
                forward_path_tracing &&
        compiled.data()->light_count > 0u) {
        // Analytic lights are not acceleration-structure primitives yet, so
        // a forward-only path cannot reach them.
        return nullptr;
    }
    for (const auto &pass : settings.passes) {
        if (!supported_pass(pass.kind)) {
            return nullptr;
        }
    }
    attach_scene_film_pass_accessor(
        _device, _film_pass_accessor, compiled.data()->film_pass_accessor);
    return std::make_unique<LuisaRenderSession>(
        compiled.data(), _options, settings);
}

}// namespace psycles::luisa_backend

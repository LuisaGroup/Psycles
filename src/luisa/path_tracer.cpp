#include "path_tracer_internal.h"
#include "path_tracer_backend_policy.h"

namespace psycles::luisa_backend {

using namespace detail;

namespace {

[[nodiscard]] bool valid_scheduler_options(
    const LuisaPathTracerOptions &options) noexcept {
    switch (options.scheduler) {
        case LuisaPathScheduler::megakernel:
            return true;
        case LuisaPathScheduler::wavefront:
            return options.wavefront_frame_capacity != 0u;
        case LuisaPathScheduler::persistent:
            return options.persistent_worker_count != 0u &&
                   options.persistent_block_size != 0u &&
                   options.persistent_fetch_size != 0u;
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
        _options.max_pixel_samples_per_dispatch =
            std::min(
                _options.max_pixel_samples_per_dispatch,
                backend_max_pixel_samples_per_dispatch(
                    _device.backend_name()));
    }
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
    return std::make_unique<LuisaRenderSession>(
        compiled.data(), _options, settings);
}

}// namespace psycles::luisa_backend

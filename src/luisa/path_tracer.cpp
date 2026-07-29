#include "path_tracer_internal.h"

namespace psycles::luisa_backend {

using namespace detail;

LuisaPathTracerBackend::LuisaPathTracerBackend(
    luisa::compute::Device device,
    LuisaPathTracerOptions options) noexcept
    : _device{std::move(device)},
      _options{options} {}

std::unique_ptr<contract::RenderSession>
LuisaPathTracerBackend::create_session(
    const contract::CompiledScene &scene,
    const RenderSettings &settings) {
    const auto &compiled =
        static_cast<const LuisaCompiledScene &>(scene);
    if (settings.full_extent.width == 0u ||
        settings.full_extent.height == 0u ||
        _options.max_samples_per_dispatch == 0u) {
        return nullptr;
    }
    if (settings.integrator.use_light_tree) {
        return nullptr;
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

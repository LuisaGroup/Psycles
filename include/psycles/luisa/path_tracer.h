#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/path_tracer.h> through a Psycles Luisa target."
#endif

#include <cstdint>

#include <psycles/contract/render.h>

#include <luisa/runtime/device.h>

namespace psycles::luisa_backend {

struct LuisaPathTracerOptions {
    bool next_event_estimation{true};
    // Each batch is submitted and synchronized independently. Keeping this
    // finite bounds GPU progress and error-detection latency without changing
    // the device sampler's global sample indices. Zero is invalid.
    std::uint32_t max_samples_per_dispatch{8u};
};

class LuisaPathTracerBackend final : public contract::RendererBackend {

private:
    luisa::compute::Device _device;
    LuisaPathTracerOptions _options;

public:
    explicit LuisaPathTracerBackend(
        luisa::compute::Device device,
        LuisaPathTracerOptions options = {}) noexcept;

    [[nodiscard]] contract::SceneCompilation compile_scene(
        const contract::SceneSnapshot &snapshot) override;

    [[nodiscard]] std::unique_ptr<contract::RenderSession> create_session(
        const contract::CompiledScene &scene,
        const contract::RenderSettings &settings) override;
};

}// namespace psycles::luisa_backend

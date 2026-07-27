#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/path_tracer.h> through a Psycles Luisa target."
#endif

#include <cstdint>

#include <psycles/contract/render.h>

#include <luisa/runtime/device.h>

namespace psycles::luisa_backend {

struct LuisaPathTracerOptions {
    std::uint32_t max_bounces{8u};
    std::uint32_t russian_roulette_depth{4u};
    bool next_event_estimation{true};
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

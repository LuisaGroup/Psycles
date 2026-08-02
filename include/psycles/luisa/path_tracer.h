#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/path_tracer.h> through a Psycles Luisa target."
#endif

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#include <psycles/contract/render.h>
#include <psycles/luisa/path_trace_schema.h>

#include <luisa/runtime/device.h>

namespace psycles::luisa_backend {

struct LuisaPathTrace {
    using Slot = std::array<float, 4u>;

    std::uint32_t pixel_x{};
    // Cycles film convention: zero is the lower row of the full image.
    std::uint32_t pixel_y{};
    std::uint32_t sample{};
    std::array<Slot, path_trace_schema::slot_count> slots{};
};

class LuisaPathTraceSink {

public:
    virtual ~LuisaPathTraceSink() noexcept = default;
    virtual void write(const LuisaPathTrace &trace) = 0;
};

struct LuisaPathTraceRequest {
    std::uint32_t pixel_x{};
    // Cycles film convention: zero is the lower row of the full image.
    std::uint32_t pixel_y{};
    std::uint32_t sample{};
    std::shared_ptr<LuisaPathTraceSink> sink;
};

struct LuisaPathTracerOptions {
    bool next_event_estimation{true};
    // Each batch is submitted and synchronized independently. Keeping this
    // finite bounds GPU progress and error-detection latency without changing
    // the device sampler's global sample indices. Zero is invalid.
    std::uint32_t max_samples_per_dispatch{8u};
    // Upper bound on pixel/sample work submitted in one kernel dispatch.
    // Backends without a practical watchdog use the unbounded default;
    // the Metal backend applies a conservative device-safe cap.
    std::uint32_t max_pixel_samples_per_dispatch{
        std::numeric_limits<std::uint32_t>::max()};
    // Diagnostic-only, observational trace. The kernel writes this fixed
    // schema only for the requested full-film pixel and absolute sample.
    std::optional<LuisaPathTraceRequest> path_trace;
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

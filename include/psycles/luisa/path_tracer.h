#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/path_tracer.h> through a Psycles Luisa target."
#endif

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

#include <psycles/contract/render.h>
#include <psycles/luisa/path_trace_schema.h>

#include <luisa/runtime/device.h>

namespace psycles::luisa_backend {

enum class LuisaPathScheduler : std::uint8_t {
    megakernel,
    wavefront,
    persistent,
};

[[nodiscard]] constexpr std::string_view
luisa_path_scheduler_name(
    LuisaPathScheduler scheduler) noexcept {
    switch (scheduler) {
        case LuisaPathScheduler::megakernel:
            return "megakernel";
        case LuisaPathScheduler::wavefront:
            return "wavefront";
        case LuisaPathScheduler::persistent:
            return "persistent";
    }
    return {};
}

[[nodiscard]] constexpr std::optional<LuisaPathScheduler>
parse_luisa_path_scheduler(
    std::string_view name) noexcept {
    if (name == "megakernel") {
        return LuisaPathScheduler::megakernel;
    }
    if (name == "wavefront") {
        return LuisaPathScheduler::wavefront;
    }
    if (name == "persistent") {
        return LuisaPathScheduler::persistent;
    }
    return std::nullopt;
}

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
    // All three modes record the same path program. The scheduler is a host-
    // side execution policy and is never inferred from the backend name, so a
    // GPU can use the megakernel baseline and fallback can exercise both
    // coroutine schedulers as well.
    LuisaPathScheduler scheduler{
        LuisaPathScheduler::megakernel};
    // Maximum number of live global frames in the wavefront scheduler. More
    // logical pixels are admitted in subsequent scheduler iterations.
    // GPU Coroutines' path-tracing evaluation uses 2^24 global coroutine
    // frames with SoA storage and compaction.
    std::uint32_t wavefront_frame_capacity{1u << 24u};
    // Persistent workers are independent of the logical image size. The
    // scheduler rounds this count up to a complete block.
    // The paper's persistent configuration is 2^15 workers, 128 threads per
    // block, and 16 logical instances fetched per worker acquisition.
    std::uint32_t persistent_worker_count{1u << 15u};
    // Kept backend-independent: fallback is allowed to execute the same block
    // scheduler and is not silently normalized to a scalar special case.
    std::uint32_t persistent_block_size{128u};
    std::uint32_t persistent_fetch_size{16u};
    bool persistent_shared_memory_soa{true};
    bool persistent_global_memory_extension{true};
    // Each batch is submitted and synchronized independently. Keeping this
    // finite bounds GPU progress and error-detection latency without changing
    // the device sampler's global sample indices. Zero is invalid.
    std::uint32_t max_samples_per_dispatch{4u};
    // Upper bound on pixel/sample work submitted in one kernel dispatch.
    // Backends without a practical watchdog use the unbounded default;
    // Metal and Vulkan apply a conservative device-safe cap.
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

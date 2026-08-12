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
    megakernel_per_sample,
    wavefront,
    persistent,
};

[[nodiscard]] constexpr std::string_view
luisa_path_scheduler_name(
    LuisaPathScheduler scheduler) noexcept {
    switch (scheduler) {
        case LuisaPathScheduler::megakernel:
            return "megakernel";
        case LuisaPathScheduler::megakernel_per_sample:
            return "megakernel-per-sample";
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
    if (name == "megakernel-per-sample") {
        return LuisaPathScheduler::megakernel_per_sample;
    }
    if (name == "wavefront") {
        return LuisaPathScheduler::wavefront;
    }
    if (name == "persistent") {
        return LuisaPathScheduler::persistent;
    }
    return std::nullopt;
}

[[nodiscard]] constexpr bool
valid_luisa_execution_block_size(
    std::uint32_t size) noexcept {
    // Mirrors the Luisa DSL workgroup contract, so invalid scheduler options
    // are rejected before shader AST construction reaches set_block_size().
    return size >= 32u && size <= 1024u && size % 32u == 0u;
}

[[nodiscard]] constexpr bool
valid_luisa_wavefront_execution_block_size(
    std::uint32_t size) noexcept {
    return valid_luisa_execution_block_size(size);
}

[[nodiscard]] constexpr bool
valid_luisa_persistent_scheduler_shape(
    std::uint32_t worker_count,
    std::uint32_t block_size,
    std::uint32_t fetch_size) noexcept {
    return worker_count != 0u &&
           valid_luisa_execution_block_size(block_size) &&
           fetch_size != 0u &&
           static_cast<std::uint64_t>(block_size) * fetch_size <=
               std::numeric_limits<std::uint32_t>::max();
}

// The serial megakernel gives one invocation exclusive ownership of a pixel
// and loops over its sample batch. Every other mode launches the Cartesian
// product of pixels and samples and therefore requires race-free film writes.
[[nodiscard]] constexpr bool
luisa_path_scheduler_uses_per_sample_dispatch(
    LuisaPathScheduler scheduler) noexcept {
    return scheduler != LuisaPathScheduler::megakernel;
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
    // Host/JIT choice for Luisa's thread-local wavefront generate/resume
    // kernels. Unlike frame capacity, this is part of shader structure.
    std::uint32_t wavefront_execution_block_size{32u};
    // Persistent workers are independent of the logical image size. The
    // scheduler rounds this count up to a complete block.
    // The paper's persistent configuration is 2^15 workers, 128 threads per
    // block, and 16 block-sized batches fetched per acquisition.
    std::uint32_t persistent_worker_count{1u << 15u};
    // Kept backend-independent: fallback executes the same block scheduler
    // instead of being silently normalized to a scalar special case. The
    // single-wave default is an application tuning choice; callers retain
    // control because the best occupancy depends on the coroutine frame and
    // continuation resource footprint.
    std::uint32_t persistent_block_size{32u};
    // Runtime task-allocation granularity, measured in block-sized batches.
    // It does not enter shader AST/cache identity.
    std::uint32_t persistent_fetch_size{1u};
    bool persistent_shared_memory_soa{true};
    bool persistent_global_memory_extension{true};
    // Each batch is submitted and synchronized independently. Per-sample
    // schedulers launch this as dispatch.z; the serial megakernel uses the
    // same runtime value as its loop bound. It is never part of shader AST or
    // cache identity. Setting it to one is the deterministic diagnostic path:
    // each pixel has only one atomic contributor between stream barriers, so
    // exact hashes remain meaningful. Zero is invalid.
    std::uint32_t max_samples_per_dispatch{64u};
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

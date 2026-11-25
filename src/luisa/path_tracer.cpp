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

void apply_texture_mip_env(const char *name,
                           LuisaTextureMipMode &dst) noexcept {
    if (const auto *raw = std::getenv(name);
        raw != nullptr && *raw != '\0') {
        const std::string_view sv{raw};
        if (sv == "mip0") {
            dst = LuisaTextureMipMode::mip0;
        } else if (sv == "isotropic") {
            dst = LuisaTextureMipMode::isotropic;
        }
    }
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
    if (const auto *prepack_raw = std::getenv("PSYCLES_VT_PREPACK_MODE");
        prepack_raw != nullptr && *prepack_raw != '\0') {
        const std::string_view sv{prepack_raw};
        if (sv == "off") {
            vt.prepack_mode = LuisaVirtualTexturePrepackMode::off;
        } else if (sv == "memory") {
            vt.prepack_mode = LuisaVirtualTexturePrepackMode::memory;
        } else if (sv == "disk") {
            vt.prepack_mode = LuisaVirtualTexturePrepackMode::disk;
        }
    }
    if (const auto prepack_mb = env_unsigned_override<std::uint64_t>(
            "PSYCLES_VT_PREPACK_BUDGET_MB")) {
        vt.host_prepacked_cache_budget_bytes = *prepack_mb * 1024ull * 1024ull;
    }
    if (const auto warm_init = env_bool_override("PSYCLES_VT_PREPACK_WARM_INIT")) {
        vt.prepack_warm_on_init = *warm_init;
    }
    if (const auto *disk_dir = std::getenv("PSYCLES_VT_PREPACK_DIR");
        disk_dir != nullptr && *disk_dir != '\0') {
        vt.prepack_disk_dir = disk_dir;
    }
    if (const auto async_worker = env_bool_override("PSYCLES_VT_ASYNC_WORKER")) {
        vt.enable_async_worker = *async_worker;
    }
    if (const auto async_slots =
            env_unsigned_override<std::uint32_t>("PSYCLES_VT_ASYNC_SLOTS");
        async_slots && *async_slots >= 1u && *async_slots <= 8u) {
        vt.async_batch_slots = *async_slots;
    }
    if (const auto *cq_raw = std::getenv("PSYCLES_VT_COPY_QUEUE");
        cq_raw != nullptr && *cq_raw != '\0') {
        const std::string_view sv{cq_raw};
        if (sv == "off") {
            vt.copy_queue_mode = LuisaVirtualTextureCopyQueueMode::off;
        } else if (sv == "direct") {
            vt.copy_queue_mode = LuisaVirtualTextureCopyQueueMode::direct;
        } else if (sv == "staged") {
            vt.copy_queue_mode = LuisaVirtualTextureCopyQueueMode::staged;
        } else if (sv == "staged_shader") {
            vt.copy_queue_mode = LuisaVirtualTextureCopyQueueMode::staged_shader;
        }
    }
    if (const auto *comp_raw = std::getenv("PSYCLES_VT_COMPRESSION");
        comp_raw != nullptr && *comp_raw != '\0') {
        const std::string_view sv{comp_raw};
        if (sv == "off") {
            vt.compression_mode = LuisaVirtualTextureCompressionMode::off;
        } else if (sv == "auto" || sv == "auto_hq") {
            vt.compression_mode = LuisaVirtualTextureCompressionMode::auto_hq;
        } else if (sv == "auto_compact") {
            vt.compression_mode = LuisaVirtualTextureCompressionMode::auto_compact;
        } else if (sv == "bc7") {
            vt.compression_mode = LuisaVirtualTextureCompressionMode::bc7;
        } else if (sv == "bc3") {
            vt.compression_mode = LuisaVirtualTextureCompressionMode::bc3;
        } else if (sv == "bc1") {
            vt.compression_mode = LuisaVirtualTextureCompressionMode::bc1;
        }
    }
    if (const auto *qual_raw = std::getenv("PSYCLES_VT_GPU_BC_QUALITY");
        qual_raw != nullptr && *qual_raw != '\0') {
        const std::string_view sv{qual_raw};
        if (sv == "fast" || sv == "low" || sv == "0") {
            vt.compression_quality =
                LuisaVirtualTextureCompressionQuality::fast;
        } else if (sv == "balanced" || sv == "default" || sv == "medium" ||
                   sv == "1") {
            vt.compression_quality =
                LuisaVirtualTextureCompressionQuality::balanced;
        } else if (sv == "high" || sv == "hq" || sv == "slow" || sv == "2") {
            vt.compression_quality =
                LuisaVirtualTextureCompressionQuality::high;
        }
    }
    if (const auto preflight = env_bool_override("PSYCLES_VT_PREFLIGHT")) {
        vt.enable_preflight = *preflight;
    }
    if (const auto stride =
            env_unsigned_override<std::uint32_t>("PSYCLES_VT_PREFLIGHT_STRIDE")) {
        vt.preflight_stride = std::clamp(*stride, 1u, 16u);
    }
    if (const auto bounces =
            env_unsigned_override<std::uint32_t>("PSYCLES_VT_PREFLIGHT_BOUNCES")) {
        vt.preflight_bounces = std::clamp(*bounces, 0u, 2u);
    }
    if (const auto overlap =
            env_bool_override("PSYCLES_VT_PREFLIGHT_OVERLAP")) {
        vt.preflight_overlap_pso = *overlap;
    }
    apply_texture_mip_env("PSYCLES_TEXTURE_MIP_MODE", vt.mip_mode);
    apply_texture_mip_env("PSYCLES_VT_MIP_MODE", vt.mip_mode);
}

void apply_direct_mip_env_overrides(
    LuisaTextureMipMode &direct_mip_mode) noexcept {
    apply_texture_mip_env("PSYCLES_TEXTURE_MIP_MODE", direct_mip_mode);
    apply_texture_mip_env("PSYCLES_DIRECT_MIP_MODE", direct_mip_mode);
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
    apply_direct_mip_env_overrides(_options.direct_texture_mip_mode);
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

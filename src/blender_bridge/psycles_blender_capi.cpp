#include "psycles_blender_internal.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>

namespace psycles::blender_bridge::detail {
namespace {

void set_env_var(const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

[[nodiscard]] bool is_cpu_backend(std::string_view name) noexcept {
    return name == "fallback" || name == "simd" || name == "cpu";
}

[[nodiscard]] psycles::luisa_backend::LuisaPathScheduler resolve_scheduler(
    std::uint32_t raw_scheduler,
    std::string_view backend_name) noexcept {
    using S = psycles::luisa_backend::LuisaPathScheduler;
    switch (raw_scheduler) {
        case PSYCLES_SCHEDULER_MEGAKERNEL:
            return S::megakernel;
        case PSYCLES_SCHEDULER_MEGAKERNEL_PER_SAMPLE:
            return S::megakernel_per_sample;
        case PSYCLES_SCHEDULER_WAVEFRONT:
            return S::wavefront;
        case PSYCLES_SCHEDULER_WAVEFRONT_GRAPH:
            return S::wavefront_graph;
        case PSYCLES_SCHEDULER_WAVEFRONT_STAGED:
            return S::wavefront_staged;
        case PSYCLES_SCHEDULER_PERSISTENT:
            return S::persistent;
        case PSYCLES_SCHEDULER_AUTO:
        default:
            return is_cpu_backend(backend_name) ? S::megakernel
                                                : S::wavefront_staged;
    }
}

[[nodiscard]] psycles::contract::PixelFilter resolve_pixel_filter(
    std::uint8_t value,
    psycles::contract::PixelFilter fallback) noexcept {
    using PF = psycles::contract::PixelFilter;
    switch (value) {
        case PSYCLES_PIXEL_FILTER_BOX:
            return PF::box;
        case PSYCLES_PIXEL_FILTER_GAUSSIAN:
            return PF::gaussian;
        case PSYCLES_PIXEL_FILTER_BLACKMAN_HARRIS:
            return PF::blackman_harris;
        case PSYCLES_PIXEL_FILTER_FROM_SCENE:
        default:
            return fallback;
    }
}

} // namespace

void write_cstr(char *dst, std::size_t cap, std::string_view text) noexcept {
    if (dst == nullptr || cap == 0u) {
        return;
    }
    const auto n = std::min(text.size(), cap - 1u);
    if (n > 0u) {
        std::memcpy(dst, text.data(), n);
    }
    dst[n] = '\0';
}

void BridgeOutputSink::begin(const psycles::contract::RenderSettings &settings) {
    _cancelled = false;
    _window = settings.window.width > 0u && settings.window.height > 0u
                  ? settings.window
                  : psycles::contract::PixelWindow{
                        .x = 0u,
                        .y = 0u,
                        .width = std::max(settings.full_extent.width, 1u),
                        .height = std::max(settings.full_extent.height, 1u)};
    _buffers.clear();
    _buffers.reserve(settings.passes.size());
    const auto count =
        static_cast<std::size_t>(_window.width) *
        static_cast<std::size_t>(_window.height);
    for (const auto &pass : settings.passes) {
        _buffers.push_back(PassBuffer{
            .request = pass,
            .width = _window.width,
            .height = _window.height,
            .channels = pass.channels,
            .pixels = std::vector<float>(
                count * static_cast<std::size_t>(pass.channels))});
    }
}

void BridgeOutputSink::write(const psycles::contract::PassTile &tile) {
    auto it = std::find_if(
        _buffers.begin(), _buffers.end(), [&](const PassBuffer &b) {
            return b.request.kind == tile.pass.kind &&
                   b.request.name == tile.pass.name &&
                   b.request.light_group == tile.pass.light_group;
        });
    if (it == _buffers.end() || it->channels != tile.pass.channels) {
        return;
    }
    const auto expected =
        static_cast<std::size_t>(tile.window.width) *
        static_cast<std::size_t>(tile.window.height) *
        static_cast<std::size_t>(tile.pass.channels);
    if (tile.pixels.size() != expected) {
        return;
    }
    if (tile.window.x < _window.x || tile.window.y < _window.y ||
        tile.window.x + tile.window.width > _window.x + _window.width ||
        tile.window.y + tile.window.height > _window.y + _window.height) {
        return;
    }
    const auto local_x0 = tile.window.x - _window.x;
    const auto local_y0 = tile.window.y - _window.y;
    const auto row_floats =
        static_cast<std::size_t>(tile.window.width) * it->channels;
    for (std::uint32_t y = 0u; y < tile.window.height; ++y) {
        const auto src_off = static_cast<std::size_t>(y) * row_floats;
        const auto logical_y = local_y0 + y;
        const auto dst_y =
            _flip_y ? (_window.height - 1u - logical_y) : logical_y;
        const auto dst_off =
            (static_cast<std::size_t>(dst_y) * it->width + local_x0) *
            it->channels;
        std::copy_n(
            tile.pixels.begin() + static_cast<std::ptrdiff_t>(src_off),
            row_floats,
            it->pixels.begin() + static_cast<std::ptrdiff_t>(dst_off));
    }
}

std::vector<psycles::contract::PassRequest> parse_pass_descs(
    const PsyclesPassDesc *passes,
    std::size_t pass_count) {
    std::vector<psycles::contract::PassRequest> req_passes;
    if (passes != nullptr && pass_count > 0u) {
        req_passes.reserve(pass_count);
        for (std::size_t i = 0u; i < pass_count; ++i) {
            req_passes.push_back(psycles::contract::PassRequest{
                .kind = static_cast<psycles::contract::PassKind>(
                    passes[i].kind),
                .name = passes[i].name != nullptr ? passes[i].name : "Combined",
                .light_group = passes[i].light_group != nullptr
                                   ? passes[i].light_group
                                   : "",
                .channels = passes[i].channels > 0u ? passes[i].channels : 4u});
        }
    } else {
        req_passes.push_back(psycles::contract::PassRequest{
            .kind = psycles::contract::PassKind::combined,
            .name = "Combined",
            .light_group = {},
            .channels = 4u});
    }
    return req_passes;
}

psycles::contract::RenderSettings make_render_settings(
    const psycles::adapter::BlenderSceneImport &imported,
    const PsyclesRenderParams *params,
    std::vector<psycles::contract::PassRequest> passes) {
    psycles::contract::RenderSettings settings;
    const auto width =
        params != nullptr && params->full_width > 0u ? params->full_width
                                                     : imported.width;
    const auto height =
        params != nullptr && params->full_height > 0u ? params->full_height
                                                      : imported.height;
    settings.full_extent = {.width = std::max(width, 1u),
                            .height = std::max(height, 1u)};
    if (params != nullptr && params->window_width > 0u &&
        params->window_height > 0u) {
        const auto wx =
            std::min(params->window_x, settings.full_extent.width - 1u);
        const auto wy =
            std::min(params->window_y, settings.full_extent.height - 1u);
        const auto ww =
            std::min(params->window_width, settings.full_extent.width - wx);
        const auto wh =
            std::min(params->window_height, settings.full_extent.height - wy);
        settings.window = {.x = wx, .y = wy, .width = ww, .height = wh};
    }
    settings.seed = params != nullptr && params->override_seed != 0u
                        ? params->seed
                        : imported.seed;
    settings.transparent_background =
        params != nullptr && params->override_transparent_background != 0u
            ? (params->transparent_background != 0u)
            : imported.transparent_background;
    settings.pixel_filter =
        params != nullptr
            ? resolve_pixel_filter(params->pixel_filter, imported.pixel_filter)
            : imported.pixel_filter;
    settings.filter_width =
        params != nullptr && params->filter_width > 0.0f ? params->filter_width
                                                         : imported.filter_width;
    settings.pass_alpha_threshold =
        params != nullptr && params->override_pass_alpha_threshold != 0u
            ? std::clamp(params->pass_alpha_threshold, 0.0f, 1.0f)
            : imported.pass_alpha_threshold;
    settings.integrator = imported.integrator;
    settings.passes = std::move(passes);
    return settings;
}

psycles::contract::CameraDesc convert_camera(const PsyclesCameraDesc &src) {
    psycles::contract::CameraDesc dst;
    switch (src.projection) {
        case PSYCLES_PROJECTION_ORTHOGRAPHIC:
            dst.projection = psycles::contract::CameraProjection::orthographic;
            break;
        case PSYCLES_PROJECTION_PANORAMA:
            dst.projection = psycles::contract::CameraProjection::panorama;
            break;
        case PSYCLES_PROJECTION_PERSPECTIVE:
        default:
            dst.projection = psycles::contract::CameraProjection::perspective;
            break;
    }
    switch (src.sensor_fit) {
        case PSYCLES_SENSOR_FIT_HORIZONTAL:
            dst.sensor_fit = psycles::contract::CameraSensorFit::horizontal;
            break;
        case PSYCLES_SENSOR_FIT_VERTICAL:
            dst.sensor_fit = psycles::contract::CameraSensorFit::vertical;
            break;
        case PSYCLES_SENSOR_FIT_AUTO:
        default:
            dst.sensor_fit = psycles::contract::CameraSensorFit::automatic;
            break;
    }
    for (std::size_t i = 0u; i < 16u; ++i) {
        dst.transform.elements[i] = src.transform[i];
    }
    dst.field_of_view = src.field_of_view;
    dst.horizontal_field_of_view = src.horizontal_field_of_view;
    dst.orthographic_scale = src.orthographic_scale;
    dst.lens_shift_x = src.lens_shift_x;
    dst.lens_shift_y = src.lens_shift_y;
    dst.near_clip = std::max(src.near_clip, 1.0e-6f);
    dst.far_clip = std::max(src.far_clip, dst.near_clip + 1.0e-3f);
    dst.aperture_radius = std::max(src.aperture_radius, 0.0f);
    dst.focal_distance = std::max(src.focal_distance, 0.0f);
    dst.aperture_blades = src.aperture_blades;
    dst.aperture_rotation = src.aperture_rotation;
    dst.aperture_ratio = std::max(src.aperture_ratio, 1.0e-5f);
    if (src.has_sensor != 0u && src.sensor_lens_mm > 0.0f) {
        dst.sensor = psycles::contract::CameraSensor{
            .lens_mm = std::max(src.sensor_lens_mm, 1.0e-4f),
            .width_mm = std::max(src.sensor_width_mm, 1.0e-4f),
            .height_mm = std::max(src.sensor_height_mm, 1.0e-4f)};
    }
    dst.pixel_aspect = {std::max(src.pixel_aspect_x, 1.0e-5f),
                        std::max(src.pixel_aspect_y, 1.0e-5f)};
    return dst;
}

std::shared_ptr<SceneImpl> compile_scene_impl(
    const std::shared_ptr<EngineImpl> &engine,
    std::string_view scene_json,
    std::span<const std::byte> geometry_bin,
    const std::filesystem::path &base_dir,
    PsyclesSceneInfo *out_info,
    std::string &out_error) {
    out_error.clear();
    if (!engine || !engine->renderer) {
        out_error = "engine is null";
        return nullptr;
    }
    if (scene_json.empty() || geometry_bin.empty()) {
        out_error = "scene_json and geometry_bin must be non-empty";
        return nullptr;
    }
    auto imported = psycles::adapter::load_blender_scene_memory(
        scene_json, geometry_bin, base_dir);
    if (!imported.ok()) {
        std::ostringstream oss;
        for (const auto &d : imported.diagnostics) {
            if (d.severity ==
                psycles::adapter::BlenderSceneDiagnosticSeverity::error) {
                if (oss.tellp() > 0) oss << "; ";
                oss << d.message;
            }
        }
        out_error = oss.str().empty() ? "scene import failed" : oss.str();
        return nullptr;
    }
    psycles::contract::CameraDesc default_camera{};
    if (imported.scene->active_camera) {
        if (const auto it = imported.scene->cameras.find(
                *imported.scene->active_camera);
            it != imported.scene->cameras.end()) {
            default_camera = it->second;
        }
    }
    auto compilation = engine->renderer->compile_scene(*imported.scene);
    if (!compilation.ok()) {
        std::ostringstream oss;
        for (const auto &d : compilation.diagnostics) {
            if (oss.tellp() > 0) oss << "; ";
            oss << d.message;
        }
        out_error = oss.str().empty() ? "scene compilation failed" : oss.str();
        return nullptr;
    }
    if (out_info != nullptr) {
        *out_info = PsyclesSceneInfo{
            .width = imported.width,
            .height = imported.height,
            .samples = imported.samples,
            .seed = imported.seed,
            .geometry_count =
                static_cast<std::uint32_t>(imported.scene->geometries.size()),
            .instance_count =
                static_cast<std::uint32_t>(imported.scene->instances.size()),
            .material_count =
                static_cast<std::uint32_t>(imported.scene->materials.size()),
            .light_count =
                static_cast<std::uint32_t>(imported.scene->lights.size()),
            .transparent_background = static_cast<std::uint8_t>(
                imported.transparent_background ? 1u : 0u)};
    }
    // Snapshot storage is no longer needed once compiled onto the device.
    imported.scene.reset();
    auto scene = std::make_shared<SceneImpl>();
    scene->engine = engine;
    scene->imported = std::move(imported);
    scene->default_camera = default_camera;
    scene->compiled = std::move(compilation.scene);
    return scene;
}

std::shared_ptr<SessionImpl> create_session_impl(
    const std::shared_ptr<SceneImpl> &scene,
    const PsyclesRenderParams *params,
    std::vector<psycles::contract::PassRequest> passes,
    std::string &out_error) {
    out_error.clear();
    if (!scene || !scene->engine || !scene->engine->renderer ||
        !scene->compiled) {
        out_error = "engine or scene is null";
        return nullptr;
    }
    const auto settings =
        make_render_settings(scene->imported, params, passes);
    auto session = std::make_shared<SessionImpl>();
    session->scene = scene;
    session->passes = std::move(passes);
    session->session =
        scene->engine->renderer->create_session(*scene->compiled, settings);
    if (!session->session) {
        out_error = "create_session returned null";
        return nullptr;
    }
    return session;
}

} // namespace psycles::blender_bridge::detail

using namespace psycles::blender_bridge::detail;

extern "C" {

std::uint32_t psycles_capi_version() { return PSYCLES_CAPI_VERSION; }

std::uint32_t psycles_probe_backends(
    const char *runtime_dir,
    char *out_csv,
    std::size_t out_cap) {
    if (runtime_dir == nullptr || *runtime_dir == '\0') {
        write_cstr(out_csv, out_cap, "");
        return 0u;
    }
    try {
        const auto dir = utf8_path(runtime_dir);
        if (!std::filesystem::is_directory(dir)) {
            write_cstr(out_csv, out_cap, "");
            return 0u;
        }
#if defined(_WIN32)
        constexpr std::string_view prefix = "luisa-backend-";
        constexpr std::array<std::string_view, 1> exts{".dll"};
        constexpr std::array<std::string_view, 7> priority{
            "vk", "dx", "hip", "metal", "metal4", "fallback", "simd"};
#elif defined(__APPLE__)
        constexpr std::string_view prefix = "libluisa-backend-";
        constexpr std::array<std::string_view, 2> exts{".so", ".dylib"};
        constexpr std::array<std::string_view, 7> priority{
            "metal", "metal4", "vk", "dx", "hip", "fallback", "simd"};
#else
        constexpr std::string_view prefix = "libluisa-backend-";
        constexpr std::array<std::string_view, 1> exts{".so"};
        constexpr std::array<std::string_view, 7> priority{
            "vk", "hip", "dx", "metal", "metal4", "fallback", "simd"};
#endif
        std::string csv;
        std::uint32_t count = 0u;
        for (const auto name : priority) {
            bool found = false;
            for (const auto ext : exts) {
                const auto filename =
                    std::string{prefix} + std::string{name} + std::string{ext};
                std::error_code ec;
                if (std::filesystem::is_regular_file(dir / filename, ec)) {
                    found = true;
                    break;
                }
            }
            if (found) {
                if (!csv.empty()) {
                    csv.push_back(',');
                }
                csv.append(name);
                ++count;
            }
        }
        write_cstr(out_csv, out_cap, csv);
        return count;
    } catch (...) {
        write_cstr(out_csv, out_cap, "");
        return 0u;
    }
}

PsyclesEngine *psycles_engine_create(
    const char *runtime_dir,
    const char *backend_name,
    const PsyclesEngineConfig *cfg,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    try {
        if (runtime_dir == nullptr || *runtime_dir == '\0') {
            write_cstr(err_buf, err_cap, "runtime_dir must not be empty");
            return nullptr;
        }
        if (backend_name == nullptr || *backend_name == '\0') {
            write_cstr(err_buf, err_cap, "backend_name must not be empty");
            return nullptr;
        }
        const auto rt_path = utf8_path(runtime_dir);
        if (!std::filesystem::exists(rt_path)) {
            write_cstr(err_buf, err_cap,
                       "runtime_dir does not exist: " +
                           path_to_utf8(rt_path));
            return nullptr;
        }
        std::filesystem::path data_path = rt_path;
        if (cfg != nullptr && cfg->cache_dir != nullptr &&
            *cfg->cache_dir != '\0') {
            data_path = utf8_path(cfg->cache_dir);
            std::filesystem::create_directories(data_path);
        }

        const std::string backend{backend_name};
        if (backend == "vk" || backend == "vulkan") {
            const bool require_xir =
                cfg == nullptr || cfg->vulkan_require_native_xir != 0u;
            set_env_var("LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV",
                        require_xir ? "1" : "0");
        }
        if ((backend == "hip") && cfg != nullptr &&
            cfg->hip_visible_device >= 0) {
            const auto dev_str = std::to_string(cfg->hip_visible_device);
            set_env_var("HIP_VISIBLE_DEVICES", dev_str.c_str());
        }

        auto engine = std::make_shared<EngineImpl>();
        engine->backend_name = backend;
        engine->context = std::make_unique<luisa::compute::Context>(
            rt_path.string(), data_path.string());

        const auto installed = engine->context->installed_backends();
        const bool found = std::any_of(
            installed.begin(), installed.end(),
            [&](const luisa::string &b) { return b == backend; });
        if (!found) {
            std::ostringstream oss;
            oss << "backend '" << backend << "' not available in '"
                << engine->context->runtime_directory().string()
                << "'; installed: [";
            for (std::size_t i = 0u; i < installed.size(); ++i) {
                if (i > 0u) oss << ", ";
                oss << installed[i];
            }
            oss << "]";
            write_cstr(err_buf, err_cap, oss.str());
            return nullptr;
        }

        auto device = engine->context->create_device(backend);
        psycles::luisa_backend::LuisaPathTracerOptions options;
        options.next_event_estimation = true;
        options.scheduler = resolve_scheduler(
            cfg != nullptr ? cfg->scheduler : PSYCLES_SCHEDULER_AUTO,
            backend);
        if (cfg != nullptr) {
            options.enable_fast_math = cfg->enable_fast_math != 0u;
            options.staged_surface_sorting = cfg->staged_surface_sorting != 0u;
            options.staged_direct_light_queue =
                cfg->staged_direct_light_queue != 0u;
            if (cfg->wavefront_frame_capacity > 0u) {
                options.wavefront_frame_capacity =
                    cfg->wavefront_frame_capacity;
            }
            if (cfg->wavefront_execution_block_size > 0u &&
                psycles::luisa_backend::
                    valid_luisa_wavefront_execution_block_size(
                        cfg->wavefront_execution_block_size)) {
                options.wavefront_execution_block_size =
                    cfg->wavefront_execution_block_size;
            }
            if (cfg->max_samples_per_dispatch > 0u) {
                options.max_samples_per_dispatch =
                    cfg->max_samples_per_dispatch;
            }
        }
        engine->renderer =
            std::make_unique<psycles::luisa_backend::LuisaPathTracerBackend>(
                std::move(device), options);
        return new PsyclesEngine{.impl = std::move(engine)};
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return nullptr;
    } catch (...) {
        write_cstr(err_buf, err_cap,
                   "unknown exception in psycles_engine_create");
        return nullptr;
    }
}

void psycles_engine_destroy(PsyclesEngine *engine) { delete engine; }

PsyclesScene *psycles_scene_compile_memory(
    PsyclesEngine *engine,
    const char *scene_json,
    std::size_t json_len,
    const void *geometry_bin,
    std::size_t geom_len,
    const char *asset_base_dir,
    PsyclesSceneInfo *out_info,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    try {
        if (engine == nullptr || !engine->impl) {
            write_cstr(err_buf, err_cap, "engine is null");
            return nullptr;
        }
        if (scene_json == nullptr || json_len == 0u ||
            geometry_bin == nullptr || geom_len == 0u) {
            write_cstr(err_buf, err_cap,
                       "scene_json and geometry_bin must be non-empty");
            return nullptr;
        }
        const std::filesystem::path base_dir =
            asset_base_dir != nullptr ? utf8_path(asset_base_dir)
                                      : std::filesystem::path{};
        std::string error;
        auto scene = compile_scene_impl(
            engine->impl,
            std::string_view{scene_json, json_len},
            std::span<const std::byte>{
                static_cast<const std::byte *>(geometry_bin), geom_len},
            base_dir,
            out_info,
            error);
        if (!scene) {
            write_cstr(err_buf, err_cap, error);
            return nullptr;
        }
        return new PsyclesScene{.impl = std::move(scene)};
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return nullptr;
    } catch (...) {
        write_cstr(err_buf, err_cap,
                   "unknown exception in psycles_scene_compile_memory");
        return nullptr;
    }
}

void psycles_scene_destroy(PsyclesScene *scene) { delete scene; }

PsyclesSession *psycles_session_create(
    PsyclesEngine *engine,
    PsyclesScene *scene,
    const PsyclesRenderParams *params,
    const PsyclesPassDesc *passes,
    std::size_t pass_count,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    try {
        if (engine == nullptr || !engine->impl ||
            scene == nullptr || !scene->impl) {
            write_cstr(err_buf, err_cap, "engine or scene is null");
            return nullptr;
        }
        std::string error;
        auto session = create_session_impl(
            scene->impl, params, parse_pass_descs(passes, pass_count), error);
        if (!session) {
            write_cstr(err_buf, err_cap, error);
            return nullptr;
        }
        return new PsyclesSession{.impl = std::move(session)};
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return nullptr;
    } catch (...) {
        write_cstr(err_buf, err_cap,
                   "unknown exception in psycles_session_create");
        return nullptr;
    }
}

int psycles_session_reconfigure(
    PsyclesSession *session,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    const PsyclesPassDesc *passes,
    std::size_t pass_count,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    try {
        if (session == nullptr || !session->impl || !session->impl->session ||
            !session->impl->scene) {
            write_cstr(err_buf, err_cap, "invalid session argument");
            return -1;
        }
        const auto impl = session->impl;
        std::scoped_lock lock{impl->op_mutex};
        impl->cancelled.store(false);
        if (passes != nullptr && pass_count > 0u) {
            impl->passes = parse_pass_descs(passes, pass_count);
        }
        const auto settings = make_render_settings(
            impl->scene->imported, params, impl->passes);
        const auto cam = camera != nullptr ? convert_camera(*camera)
                                           : impl->scene->default_camera;
        if (!impl->session->update_camera(cam, settings)) {
            write_cstr(err_buf, err_cap,
                       "session backend does not support reconfigure");
            return -1;
        }
        return 1;
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return -1;
    } catch (...) {
        write_cstr(err_buf, err_cap,
                   "unknown exception in psycles_session_reconfigure");
        return -1;
    }
}

int psycles_session_update_camera(
    PsyclesSession *session,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    char *err_buf,
    std::size_t err_cap) {
    if (camera == nullptr) {
        write_cstr(err_buf, err_cap, "invalid session or camera argument");
        return -1;
    }
    return psycles_session_reconfigure(
        session, camera, params, nullptr, 0u, err_buf, err_cap);
}

int psycles_session_render_samples(
    PsyclesSession *session,
    std::uint32_t sample_first,
    std::uint32_t sample_count,
    std::uint32_t total_samples,
    std::uint8_t flip_y,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    try {
        if (session == nullptr || !session->impl || !session->impl->session) {
            write_cstr(err_buf, err_cap, "session is null");
            return -1;
        }
        const auto impl = session->impl;
        std::scoped_lock lock{impl->op_mutex};
        impl->sink.set_flip_y(flip_y != 0u);
        const bool ok = impl->session->render_samples(
            psycles::contract::SampleRange{.first = sample_first,
                                           .count = sample_count,
                                           .offset = 0u,
                                           .total = total_samples},
            impl->sink);
        if (impl->cancelled.load() || impl->sink.cancelled()) {
            return 0;
        }
        if (!ok) {
            write_cstr(
                err_buf,
                err_cap,
                "render_samples returned false (incomplete coverage or failure)");
            return -1;
        }
        return 1;
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return -1;
    } catch (...) {
        write_cstr(err_buf, err_cap,
                   "unknown exception in psycles_session_render_samples");
        return -1;
    }
}

int psycles_session_get_pass_buffer(
    const PsyclesSession *session,
    std::size_t pass_index,
    const float **out_ptr,
    std::uint32_t *out_width,
    std::uint32_t *out_height,
    std::uint32_t *out_channels) {
    if (session == nullptr || !session->impl ||
        pass_index >= session->impl->sink.buffers().size()) {
        return 0;
    }
    const auto &buf = session->impl->sink.buffers()[pass_index];
    if (out_ptr != nullptr) *out_ptr = buf.pixels.data();
    if (out_width != nullptr) *out_width = buf.width;
    if (out_height != nullptr) *out_height = buf.height;
    if (out_channels != nullptr) *out_channels = buf.channels;
    return 1;
}

int psycles_session_get_last_timings(
    const PsyclesSession *session,
    double *out_integration_s,
    double *out_pass_output_s,
    double *out_render_call_s) {
    if (session == nullptr || !session->impl || !session->impl->session) {
        return 0;
    }
    const auto timings = session->impl->session->last_render_timings();
    if (!timings) {
        return 0;
    }
    if (out_integration_s != nullptr) {
        *out_integration_s = timings->integration_seconds;
    }
    if (out_pass_output_s != nullptr) {
        *out_pass_output_s = timings->pass_output_seconds;
    }
    if (out_render_call_s != nullptr) {
        *out_render_call_s = timings->render_call_seconds;
    }
    return 1;
}

void psycles_session_cancel(PsyclesSession *session) {
    if (session != nullptr && session->impl) {
        session->impl->cancelled.store(true);
        if (session->impl->session) {
            session->impl->session->cancel();
        }
    }
}

void psycles_session_destroy(PsyclesSession *session) { delete session; }

void psycles_shutdown_runtime() { psycles::shutdown_parallel_pool(); }

} // extern "C"

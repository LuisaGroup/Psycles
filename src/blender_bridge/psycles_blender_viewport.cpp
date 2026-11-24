#include "psycles_blender_internal.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace psycles::blender_bridge::detail {
namespace {

#ifdef _WIN32
[[nodiscard]] inline const char *read_test_step_delay_env(char (&buffer)[64]) noexcept {
    const auto len = GetEnvironmentVariableA(
        "PSYCLES_TEST_VIEWPORT_STEP_DELAY_MS", buffer, static_cast<DWORD>(sizeof(buffer)));
    return (len > 0u && len < sizeof(buffer)) ? buffer : nullptr;
}
#else
[[nodiscard]] inline const char *read_test_step_delay_env(char (&)[64]) noexcept {
    return std::getenv("PSYCLES_TEST_VIEWPORT_STEP_DELAY_MS");
}
#endif

[[nodiscard]] std::uint32_t read_test_step_delay_ms() noexcept {
    char delay_env_buf[64]{};
    if (const char *raw = read_test_step_delay_env(delay_env_buf);
        raw != nullptr && *raw != '\0') {
        const std::string_view sv{raw};
        std::uint32_t ms = 0u;
        const auto [ptr, ec] =
            std::from_chars(sv.data(), sv.data() + sv.size(), ms);
        if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
            return std::min<std::uint32_t>(ms, 10000u);
        }
    }
    return 0u;
}

struct ViewportTaskImpl {
    // Declared before scene_impl / session_impl so destruction order is strict.
    std::shared_ptr<EngineImpl> engine_impl;

    std::mutex state_mutex;
    std::shared_ptr<SceneImpl> scene_impl;
    std::shared_ptr<SessionImpl> session_impl;
    std::uint32_t state{PSYCLES_VIEWPORT_STATE_IDLE};
    std::uint32_t rendered_samples{0u};
    std::uint32_t target_samples{1u};
    std::uint32_t start_samples{1u};
    std::uint32_t width{0u};
    std::uint32_t height{0u};
    std::string error_message;

    PsyclesCameraDesc pending_camera{};
    bool has_pending_camera{false};
    PsyclesRenderParams pending_params{};
    bool has_pending_params{false};
    std::uint32_t pending_target_samples{1u};
    std::uint32_t pending_start_samples{1u};
    bool flip_y{true};

    std::atomic<std::uint64_t> camera_epoch{0u};
    std::atomic_bool stop_requested{false};
    std::atomic_bool finished{false};

    std::mutex cancel_mutex;
    std::shared_ptr<SessionImpl> active_session;

    std::mutex frame_mutex;
    std::vector<float> frame_rgba;
    std::uint32_t frame_width{0u};
    std::uint32_t frame_height{0u};
    std::uint64_t frame_generation{0u};

    std::uint32_t step_delay_ms{0u};

    // Compile payload when starting from raw JSON + binary.
    std::string compile_json;
    std::vector<std::byte> compile_geom;
    std::filesystem::path compile_base_dir;
    bool needs_compile{false};
    bool needs_session_create{false};

    void set_active_session(std::shared_ptr<SessionImpl> s) noexcept {
        std::scoped_lock lock{cancel_mutex};
        active_session = std::move(s);
    }

    void finish(std::uint32_t final_state, std::string err = {}) noexcept {
        set_active_session({});
        {
            std::scoped_lock lock{state_mutex};
            state = final_state;
            if (!err.empty()) {
                error_message = std::move(err);
            }
        }
        finished.store(true, std::memory_order_release);
    }
};

void run_session_render_loop(
    const std::shared_ptr<ViewportTaskImpl> &task,
    std::shared_ptr<SessionImpl> session,
    std::uint64_t applied_epoch) {
    task->set_active_session(session);
    // Serialize exclusive access to this SessionImpl while the worker executes.
    std::unique_lock<std::mutex> session_lock{session->op_mutex};

    std::uint32_t rendered = 0u;
    std::uint32_t target = 1u;
    std::uint32_t start_step = 1u;
    bool flip_y = true;
    {
        std::scoped_lock lock{task->state_mutex};
        target = std::max(task->pending_target_samples, 1u);
        start_step = std::max(task->pending_start_samples, 1u);
        flip_y = task->flip_y;
        task->target_samples = target;
        task->rendered_samples = 0u;
        task->state = PSYCLES_VIEWPORT_STATE_RENDERING;
    }
    std::uint32_t step = std::max(1u, std::min(start_step, target));

    for (;;) {
        if (task->stop_requested.load(std::memory_order_acquire)) {
            task->finish(PSYCLES_VIEWPORT_STATE_CANCELLED);
            return;
        }

        const auto desired_epoch =
            task->camera_epoch.load(std::memory_order_acquire);
        if (desired_epoch != applied_epoch) {
            PsyclesCameraDesc cam_desc{};
            bool has_cam = false;
            PsyclesRenderParams params_desc{};
            bool has_params = false;
            std::uint32_t new_target = 1u;
            std::uint32_t new_start = 1u;
            {
                std::scoped_lock lock{task->state_mutex};
                cam_desc = task->pending_camera;
                has_cam = task->has_pending_camera;
                params_desc = task->pending_params;
                has_params = task->has_pending_params;
                new_target = std::max(task->pending_target_samples, 1u);
                new_start = std::max(task->pending_start_samples, 1u);
            }

            session->cancelled.store(false, std::memory_order_release);
            const auto passes = std::vector<psycles::contract::PassRequest>{
                psycles::contract::PassRequest{
                    .kind = psycles::contract::PassKind::combined,
                    .name = "Combined",
                    .light_group = {},
                    .channels = 4u}};
            session->passes = passes;
            const auto settings = make_render_settings(
                session->scene->imported,
                has_params ? &params_desc : nullptr,
                passes);
            const auto cam = has_cam ? convert_camera(cam_desc)
                                     : session->scene->default_camera;
            if (!session->session->update_camera(cam, settings)) {
                task->finish(
                    PSYCLES_VIEWPORT_STATE_FAILED,
                    "session backend does not support update_camera");
                return;
            }
            const auto win_w = settings.window.width > 0u
                                   ? settings.window.width
                                   : settings.full_extent.width;
            const auto win_h = settings.window.height > 0u
                                   ? settings.window.height
                                   : settings.full_extent.height;
            {
                std::scoped_lock lock{task->state_mutex};
                target = new_target;
                start_step = new_start;
                task->target_samples = target;
                task->rendered_samples = 0u;
                task->width = win_w;
                task->height = win_h;
                task->state = PSYCLES_VIEWPORT_STATE_RENDERING;
            }
            rendered = 0u;
            step = std::max(1u, std::min(start_step, target));
            applied_epoch = desired_epoch;

            if (task->stop_requested.load(std::memory_order_acquire)) {
                task->finish(PSYCLES_VIEWPORT_STATE_CANCELLED);
                return;
            }
            if (task->camera_epoch.load(std::memory_order_acquire) !=
                applied_epoch) {
                continue;
            }
        }

        if (rendered >= target) {
            task->finish(PSYCLES_VIEWPORT_STATE_DONE);
            return;
        }

        const auto count = std::min(step, target - rendered);
        if (task->step_delay_ms > 0u) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(task->step_delay_ms));
        }

        session->cancelled.store(false, std::memory_order_release);
        session->sink.set_flip_y(flip_y);
        const bool ok = session->session->render_samples(
            psycles::contract::SampleRange{.first = rendered,
                                           .count = count,
                                           .offset = 0u,
                                           .total = target},
            session->sink);

        if (task->stop_requested.load(std::memory_order_acquire)) {
            task->finish(PSYCLES_VIEWPORT_STATE_CANCELLED);
            return;
        }
        if (task->camera_epoch.load(std::memory_order_acquire) !=
            applied_epoch) {
            // Camera changed mid-batch; keep looping to apply the latest camera.
            continue;
        }
        if (!ok) {
            if (session->cancelled.load(std::memory_order_acquire) ||
                session->sink.cancelled()) {
                task->finish(PSYCLES_VIEWPORT_STATE_CANCELLED);
            } else {
                task->finish(
                    PSYCLES_VIEWPORT_STATE_FAILED,
                    "viewport render_samples returned false");
            }
            return;
        }

        if (!session->sink.buffers().empty()) {
            const auto &buf = session->sink.buffers()[0u];
            std::scoped_lock frame_lock{task->frame_mutex};
            task->frame_rgba = buf.pixels;
            task->frame_width = buf.width;
            task->frame_height = buf.height;
            ++task->frame_generation;
        }

        rendered += count;
        {
            std::scoped_lock lock{task->state_mutex};
            task->rendered_samples = rendered;
        }
        step = std::min<std::uint32_t>(std::max<std::uint32_t>(step * 2u, 1u), 16u);
    }
}

void viewport_worker_entry(std::shared_ptr<ViewportTaskImpl> task) noexcept {
    try {
        std::shared_ptr<SceneImpl> scene;
        std::shared_ptr<SessionImpl> session;
        std::uint64_t applied_epoch = 0u;

        if (task->needs_compile) {
            if (task->step_delay_ms > 0u) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(task->step_delay_ms));
            }
            if (task->stop_requested.load(std::memory_order_acquire)) {
                task->finish(PSYCLES_VIEWPORT_STATE_CANCELLED);
                return;
            }
            std::string error;
            scene = compile_scene_impl(
                task->engine_impl,
                task->compile_json,
                task->compile_geom,
                task->compile_base_dir,
                nullptr,
                error);
            // Release raw buffers early regardless of outcome.
            {
                std::string{}.swap(task->compile_json);
                std::vector<std::byte>{}.swap(task->compile_geom);
            }
            if (!scene) {
                task->finish(
                    PSYCLES_VIEWPORT_STATE_FAILED,
                    error.empty() ? "scene compilation failed" : error);
                return;
            }
            {
                std::scoped_lock lock{task->state_mutex};
                task->scene_impl = scene;
            }
            if (task->stop_requested.load(std::memory_order_acquire)) {
                task->finish(PSYCLES_VIEWPORT_STATE_CANCELLED);
                return;
            }
        } else {
            std::scoped_lock lock{task->state_mutex};
            scene = task->scene_impl;
            session = task->session_impl;
        }

        if (task->needs_session_create) {
            PsyclesRenderParams params_desc{};
            bool has_params = false;
            {
                std::scoped_lock lock{task->state_mutex};
                params_desc = task->pending_params;
                has_params = task->has_pending_params;
            }
            const std::vector<psycles::contract::PassRequest> passes{
                psycles::contract::PassRequest{
                    .kind = psycles::contract::PassKind::combined,
                    .name = "Combined",
                    .light_group = {},
                    .channels = 4u}};
            std::string error;
            session = create_session_impl(
                scene, has_params ? &params_desc : nullptr, passes, error);
            if (!session) {
                task->finish(
                    PSYCLES_VIEWPORT_STATE_FAILED,
                    error.empty() ? "create_session failed" : error);
                return;
            }
            {
                std::scoped_lock lock{task->state_mutex};
                task->session_impl = session;
                const auto ext_w =
                    has_params && params_desc.window_width > 0u
                        ? params_desc.window_width
                        : (has_params && params_desc.full_width > 0u
                               ? params_desc.full_width
                               : scene->imported.width);
                const auto ext_h =
                    has_params && params_desc.window_height > 0u
                        ? params_desc.window_height
                        : (has_params && params_desc.full_height > 0u
                               ? params_desc.full_height
                               : scene->imported.height);
                task->width = std::max(ext_w, 1u);
                task->height = std::max(ext_h, 1u);
            }
            if (task->stop_requested.load(std::memory_order_acquire)) {
                task->finish(PSYCLES_VIEWPORT_STATE_CANCELLED);
                return;
            }
        }

        run_session_render_loop(task, std::move(session), applied_epoch);
    } catch (const std::exception &e) {
        task->finish(PSYCLES_VIEWPORT_STATE_FAILED, e.what());
    } catch (...) {
        task->finish(
            PSYCLES_VIEWPORT_STATE_FAILED,
            "unknown exception in viewport worker");
    }
}

[[nodiscard]] std::shared_ptr<ViewportTaskImpl> make_base_task(
    std::shared_ptr<EngineImpl> engine,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples,
    std::uint8_t flip_y) {
    auto task = std::make_shared<ViewportTaskImpl>();
    task->engine_impl = std::move(engine);
    task->pending_target_samples = std::max(target_samples, 1u);
    task->pending_start_samples = std::max(start_samples, 1u);
    task->target_samples = task->pending_target_samples;
    task->start_samples = task->pending_start_samples;
    task->flip_y = (flip_y != 0u);
    if (camera != nullptr) {
        task->pending_camera = *camera;
        task->has_pending_camera = true;
    }
    if (params != nullptr) {
        task->pending_params = *params;
        task->has_pending_params = true;
        task->width = std::max(params->window_width > 0u ? params->window_width
                                                         : params->full_width,
                               1u);
        task->height = std::max(
            params->window_height > 0u ? params->window_height
                                       : params->full_height,
            1u);
    }
    task->step_delay_ms = read_test_step_delay_ms();
    return task;
}

} // namespace
} // namespace psycles::blender_bridge::detail

struct PsyclesViewportTask {
    std::shared_ptr<psycles::blender_bridge::detail::ViewportTaskImpl> impl;
    std::thread worker;
};

using namespace psycles::blender_bridge::detail;

extern "C" {

PsyclesViewportTask *psycles_viewport_task_start_compile(
    PsyclesEngine *engine,
    const char *scene_json,
    std::size_t json_len,
    const void *geometry_bin,
    std::size_t geom_len,
    const char *asset_base_dir,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples,
    std::uint8_t flip_y,
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
        auto task = make_base_task(
            engine->impl, camera, params, target_samples, start_samples, flip_y);
        task->state = PSYCLES_VIEWPORT_STATE_COMPILING;
        task->needs_compile = true;
        task->needs_session_create = true;
        task->compile_json.assign(scene_json, json_len);
        const auto *geom_ptr = static_cast<const std::byte *>(geometry_bin);
        task->compile_geom.assign(geom_ptr, geom_ptr + geom_len);
        task->compile_base_dir =
            asset_base_dir != nullptr ? utf8_path(asset_base_dir)
                                      : std::filesystem::path{};

        auto wrapper = std::make_unique<PsyclesViewportTask>();
        wrapper->impl = task;
        wrapper->worker = std::thread([task]() noexcept {
            viewport_worker_entry(task);
        });
        return wrapper.release();
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return nullptr;
    } catch (...) {
        write_cstr(
            err_buf,
            err_cap,
            "unknown exception in psycles_viewport_task_start_compile");
        return nullptr;
    }
}

PsyclesViewportTask *psycles_viewport_task_start_scene(
    PsyclesEngine *engine,
    PsyclesScene *scene,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples,
    std::uint8_t flip_y,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    try {
        if (engine == nullptr || !engine->impl ||
            scene == nullptr || !scene->impl) {
            write_cstr(err_buf, err_cap, "engine or scene is null");
            return nullptr;
        }
        auto task = make_base_task(
            engine->impl, camera, params, target_samples, start_samples, flip_y);
        task->state = PSYCLES_VIEWPORT_STATE_COMPILING;
        task->scene_impl = scene->impl;
        task->needs_session_create = true;
        if (camera != nullptr) {
            task->camera_epoch.store(1u, std::memory_order_release);
        }

        auto wrapper = std::make_unique<PsyclesViewportTask>();
        wrapper->impl = task;
        wrapper->worker = std::thread([task]() noexcept {
            viewport_worker_entry(task);
        });
        return wrapper.release();
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return nullptr;
    } catch (...) {
        write_cstr(
            err_buf,
            err_cap,
            "unknown exception in psycles_viewport_task_start_scene");
        return nullptr;
    }
}

PsyclesViewportTask *psycles_viewport_task_start_session(
    PsyclesSession *session,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples,
    std::uint8_t flip_y,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    try {
        if (session == nullptr || !session->impl || !session->impl->scene ||
            !session->impl->scene->engine) {
            write_cstr(err_buf, err_cap, "session is null");
            return nullptr;
        }
        auto task = make_base_task(
            session->impl->scene->engine,
            camera,
            params,
            target_samples,
            start_samples,
            flip_y);
        task->state = PSYCLES_VIEWPORT_STATE_RENDERING;
        task->scene_impl = session->impl->scene;
        task->session_impl = session->impl;
        task->set_active_session(session->impl);
        task->camera_epoch.store(1u, std::memory_order_release);

        auto wrapper = std::make_unique<PsyclesViewportTask>();
        wrapper->impl = task;
        wrapper->worker = std::thread([task]() noexcept {
            viewport_worker_entry(task);
        });
        return wrapper.release();
    } catch (const std::exception &e) {
        write_cstr(err_buf, err_cap, e.what());
        return nullptr;
    } catch (...) {
        write_cstr(
            err_buf,
            err_cap,
            "unknown exception in psycles_viewport_task_start_session");
        return nullptr;
    }
}

int psycles_viewport_task_request_camera(
    PsyclesViewportTask *task,
    const PsyclesCameraDesc *camera,
    const PsyclesRenderParams *params,
    std::uint32_t target_samples,
    std::uint32_t start_samples) {
    if (task == nullptr || !task->impl || camera == nullptr) {
        return -1;
    }
    const auto &impl = task->impl;
    {
        std::scoped_lock lock{impl->state_mutex};
        if (impl->finished.load(std::memory_order_acquire) ||
            impl->stop_requested.load(std::memory_order_acquire)) {
            return 0;
        }
        impl->pending_camera = *camera;
        impl->has_pending_camera = true;
        if (params != nullptr) {
            impl->pending_params = *params;
            impl->has_pending_params = true;
        }
        impl->pending_target_samples = std::max(target_samples, 1u);
        impl->pending_start_samples = std::max(start_samples, 1u);
        impl->camera_epoch.fetch_add(1u, std::memory_order_acq_rel);
    }
    std::shared_ptr<SessionImpl> active;
    {
        std::scoped_lock lock{impl->cancel_mutex};
        active = impl->active_session;
    }
    if (active && active->session) {
        active->session->cancel();
    }
    return 1;
}

void psycles_viewport_task_cancel(PsyclesViewportTask *task) {
    if (task == nullptr || !task->impl) {
        return;
    }
    const auto &impl = task->impl;
    impl->stop_requested.store(true, std::memory_order_release);
    std::shared_ptr<SessionImpl> active;
    {
        std::scoped_lock lock{impl->cancel_mutex};
        active = impl->active_session;
    }
    if (active) {
        active->cancelled.store(true, std::memory_order_release);
        if (active->session) {
            active->session->cancel();
        }
    }
}

int psycles_viewport_task_poll(
    const PsyclesViewportTask *task,
    PsyclesViewportStatus *out_status,
    char *err_buf,
    std::size_t err_cap) {
    write_cstr(err_buf, err_cap, "");
    if (task == nullptr || !task->impl || out_status == nullptr) {
        return 0;
    }
    auto *impl = task->impl.get();
    {
        std::scoped_lock lock{impl->state_mutex};
        out_status->state = impl->state;
        out_status->rendered_samples = impl->rendered_samples;
        out_status->target_samples = impl->target_samples;
        out_status->width = impl->width;
        out_status->height = impl->height;
        out_status->is_finished =
            impl->finished.load(std::memory_order_acquire) ? 1u : 0u;
        out_status->has_compiled_scene = impl->scene_impl != nullptr ? 1u : 0u;
        out_status->has_session = impl->session_impl != nullptr ? 1u : 0u;
        out_status->reserved = 0u;
        write_cstr(err_buf, err_cap, impl->error_message);
    }
    {
        std::scoped_lock flock{impl->frame_mutex};
        out_status->frame_generation = impl->frame_generation;
    }
    return 1;
}

int psycles_viewport_task_copy_frame(
    const PsyclesViewportTask *task,
    float *out_rgba,
    std::size_t out_float_cap,
    std::uint32_t *out_width,
    std::uint32_t *out_height,
    std::uint64_t *out_generation) {
    if (task == nullptr || !task->impl) {
        return 0;
    }
    auto *impl = task->impl.get();
    std::scoped_lock flock{impl->frame_mutex};
    if (impl->frame_rgba.empty() || impl->frame_width == 0u ||
        impl->frame_height == 0u) {
        return 0;
    }
    if (out_width != nullptr) *out_width = impl->frame_width;
    if (out_height != nullptr) *out_height = impl->frame_height;
    if (out_generation != nullptr) *out_generation = impl->frame_generation;
    if (out_rgba == nullptr || out_float_cap < impl->frame_rgba.size()) {
        return -1;
    }
    std::memcpy(
        out_rgba,
        impl->frame_rgba.data(),
        impl->frame_rgba.size() * sizeof(float));
    return 1;
}

PsyclesScene *psycles_viewport_task_acquire_scene(PsyclesViewportTask *task) {
    if (task == nullptr || !task->impl ||
        !task->impl->finished.load(std::memory_order_acquire)) {
        return nullptr;
    }
    std::shared_ptr<SceneImpl> scene;
    {
        std::scoped_lock lock{task->impl->state_mutex};
        scene = task->impl->scene_impl;
    }
    if (!scene) {
        return nullptr;
    }
    return new PsyclesScene{.impl = std::move(scene)};
}

PsyclesSession *psycles_viewport_task_acquire_session(
    PsyclesViewportTask *task) {
    if (task == nullptr || !task->impl ||
        !task->impl->finished.load(std::memory_order_acquire)) {
        return nullptr;
    }
    std::shared_ptr<SessionImpl> session;
    {
        std::scoped_lock lock{task->impl->state_mutex};
        session = task->impl->session_impl;
    }
    if (!session) {
        return nullptr;
    }
    return new PsyclesSession{.impl = std::move(session)};
}

void psycles_viewport_task_wait(PsyclesViewportTask *task) {
    if (task != nullptr && task->worker.joinable()) {
        task->worker.join();
    }
}

void psycles_viewport_task_destroy(PsyclesViewportTask *task) {
    if (task == nullptr) {
        return;
    }
    psycles_viewport_task_cancel(task);
    if (task->worker.joinable()) {
        if (task->impl && task->impl->finished.load(std::memory_order_acquire)) {
            task->worker.join();
        } else {
            // Do not stall Blender's UI thread if an in-flight kernel takes
            // time to notice cancellation; the worker holds shared ownership of
            // ViewportTaskImpl -> SessionImpl -> SceneImpl -> EngineImpl.
            task->worker.detach();
        }
    }
    delete task;
}

} // extern "C"

#pragma once

#include <psycles/blender_bridge/psycles_blender_capi.h>

#include <psycles/adapter/blender_scene.h>
#include <psycles/contract/render.h>
#include <psycles/luisa/path_tracer.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace psycles::blender_bridge::detail {

void write_cstr(char *dst, std::size_t cap, std::string_view text) noexcept;

struct PassBuffer {
    psycles::contract::PassRequest request;
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t channels{};
    std::vector<float> pixels;
};

class BridgeOutputSink final : public psycles::contract::OutputSink {
private:
    bool _flip_y{true};
    bool _cancelled{false};
    psycles::contract::PixelWindow _window;
    std::vector<PassBuffer> _buffers;

public:
    void set_flip_y(bool flip_y) noexcept { _flip_y = flip_y; }
    [[nodiscard]] bool cancelled() const noexcept { return _cancelled; }
    [[nodiscard]] const std::vector<PassBuffer> &buffers() const noexcept {
        return _buffers;
    }

    void begin(const psycles::contract::RenderSettings &settings) override;
    void write(const psycles::contract::PassTile &tile) override;
    void end(bool cancelled) override { _cancelled = cancelled; }
};

struct EngineImpl {
    std::unique_ptr<luisa::compute::Context> context;
    std::unique_ptr<psycles::luisa_backend::LuisaPathTracerBackend> renderer;
    std::string backend_name;
};

struct SceneImpl {
    // Declared first so `compiled` always destroys before `engine`.
    std::shared_ptr<EngineImpl> engine;
    psycles::adapter::BlenderSceneImport imported;
    psycles::contract::CameraDesc default_camera{};
    std::unique_ptr<psycles::contract::CompiledScene> compiled;
};

struct SessionImpl {
    // Declared first so `session` always destroys before `scene` and `engine`.
    std::shared_ptr<SceneImpl> scene;
    std::mutex op_mutex;
    std::vector<psycles::contract::PassRequest> passes;
    BridgeOutputSink sink;
    std::atomic_bool cancelled{false};
    std::unique_ptr<psycles::contract::RenderSession> session;
};

[[nodiscard]] std::vector<psycles::contract::PassRequest> parse_pass_descs(
    const PsyclesPassDesc *passes,
    std::size_t pass_count);

[[nodiscard]] psycles::contract::RenderSettings make_render_settings(
    const psycles::adapter::BlenderSceneImport &imported,
    const PsyclesRenderParams *params,
    std::vector<psycles::contract::PassRequest> passes);

[[nodiscard]] psycles::contract::CameraDesc convert_camera(
    const PsyclesCameraDesc &src);

[[nodiscard]] std::shared_ptr<SceneImpl> compile_scene_impl(
    const std::shared_ptr<EngineImpl> &engine,
    std::string_view scene_json,
    std::span<const std::byte> geometry_bin,
    const std::filesystem::path &base_dir,
    PsyclesSceneInfo *out_info,
    std::string &out_error);

[[nodiscard]] std::shared_ptr<SessionImpl> create_session_impl(
    const std::shared_ptr<SceneImpl> &scene,
    const PsyclesRenderParams *params,
    std::vector<psycles::contract::PassRequest> passes,
    std::string &out_error);

} // namespace psycles::blender_bridge::detail

struct PsyclesEngine {
    std::shared_ptr<psycles::blender_bridge::detail::EngineImpl> impl;
};

struct PsyclesScene {
    std::shared_ptr<psycles::blender_bridge::detail::SceneImpl> impl;
};

struct PsyclesSession {
    std::shared_ptr<psycles::blender_bridge::detail::SessionImpl> impl;
};

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <psycles/contract/scene.h>

namespace psycles::contract {

enum class PassKind : std::uint16_t {
    combined,
    depth,
    normal,
    position,
    albedo,
    emission,
    diffuse_direct,
    diffuse_indirect,
    glossy_direct,
    glossy_indirect,
    transmission_direct,
    transmission_indirect,
    volume_direct,
    volume_indirect,
    mist,
    shadow_catcher,
    cryptomatte_object,
    cryptomatte_material,
    denoising_normal,
    denoising_albedo,
    sample_count,
    aov_color,
    aov_value
};

struct PassRequest {
    PassKind kind{PassKind::combined};
    std::string name;
    std::string light_group;
    std::uint32_t channels{4u};
};

struct ImageExtent {
    std::uint32_t width{};
    std::uint32_t height{};
};

struct PixelWindow {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t width{};
    std::uint32_t height{};
};

// Matches the reconstruction filters exposed by Cycles. The filter is part of
// the camera sampling contract: it changes primary-ray raster positions and
// therefore must be evaluated by the Luisa program, not applied as a host-side
// post-process.
enum class PixelFilter : std::uint8_t {
    box,
    gaussian,
    blackman_harris
};

struct RenderSettings {
    ImageExtent full_extent;
    PixelWindow window;
    std::uint32_t seed{};
    bool transparent_background{false};
    PixelFilter pixel_filter{PixelFilter::box};
    float filter_width{1.0f};
    std::vector<PassRequest> passes;
};

struct SampleRange {
    std::uint32_t first{};
    std::uint32_t count{};
    std::uint32_t offset{};
};

struct PassTile {
    PassRequest pass;
    PixelWindow window;
    ImageExtent full_extent;
    std::span<const float> pixels;
};

class OutputSink {

public:
    virtual ~OutputSink() noexcept = default;
    virtual void begin(const RenderSettings &settings) = 0;
    virtual void write(const PassTile &tile) = 0;
    virtual void end(bool cancelled) = 0;
};

struct RenderDiagnostic {
    std::string message;
};

class CompiledScene {

public:
    virtual ~CompiledScene() noexcept = default;
    [[nodiscard]] virtual std::uint64_t source_revision() const noexcept = 0;
};

struct SceneCompilation {
    std::unique_ptr<CompiledScene> scene;
    std::vector<RenderDiagnostic> diagnostics;

    [[nodiscard]] bool ok() const noexcept { return scene != nullptr; }
};

class RenderSession {

public:
    virtual ~RenderSession() noexcept = default;
    virtual void reset(const RenderSettings &settings) = 0;
    virtual bool render_samples(const SampleRange &samples, OutputSink &output) = 0;
    virtual void cancel() noexcept = 0;
};

class RendererBackend {

public:
    virtual ~RendererBackend() noexcept = default;
    [[nodiscard]] virtual SceneCompilation compile_scene(
        const SceneSnapshot &snapshot) = 0;
    [[nodiscard]] virtual std::unique_ptr<RenderSession> create_session(
        const CompiledScene &scene,
        const RenderSettings &settings) = 0;
};

}// namespace psycles::contract

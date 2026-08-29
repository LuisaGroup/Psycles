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
    aov_value,
    glossy_color,
    transmission_color,
    environment
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

enum class DirectLightSampling : std::uint8_t {
    multiple_importance_sampling,
    forward_path_tracing,
    next_event_estimation
};

// Transport controls are part of the render contract, not backend tuning.
// A scene imported from Blender must carry the same values that Cycles uses so
// a backend cannot silently render with convenient hard-coded limits.
struct PathIntegratorSettings {
    std::uint32_t max_bounces{12u};
    std::uint32_t min_bounces{};
    std::uint32_t diffuse_bounces{4u};
    std::uint32_t glossy_bounces{4u};
    std::uint32_t transmission_bounces{12u};
    std::uint32_t volume_bounces{};
    std::uint32_t transparent_min_bounces{};
    std::uint32_t transparent_max_bounces{8u};
    // Effective Cycles Fast GI replacement state. Blender's authored
    // use_fast_gi/method/world controls are resolved by the scene adapter so
    // every backend observes the same transport contract as
    // KernelIntegrator::ao_bounces rather than reinterpreting UI policy.
    std::uint32_t ambient_occlusion_bounces{};
    float ambient_occlusion_factor{};
    float ambient_occlusion_additive_factor{};
    float ambient_occlusion_distance{10.0f};
    float sample_clamp_direct{};
    float sample_clamp_indirect{};
    // Blender exposes this as `blur_glossy`; Cycles stores the same scene
    // value as Integrator::filter_glossy and converts it to a reciprocal
    // device threshold during scene synchronization. Zero disables filtering.
    float filter_glossy{};
    // Cycles uses film exposure when converting the UI light-sampling
    // threshold into the device-side shadow-ray roulette threshold.
    float film_exposure{1.0f};
    float light_sampling_threshold{0.01f};
    bool reflective_caustics{true};
    bool refractive_caustics{true};
    bool use_light_tree{false};
    DirectLightSampling direct_light_sampling{
        DirectLightSampling::multiple_importance_sampling};
};

struct RenderSettings {
    ImageExtent full_extent;
    PixelWindow window;
    std::uint32_t seed{};
    bool transparent_background{false};
    PixelFilter pixel_filter{PixelFilter::box};
    float filter_width{1.0f};
    float pass_alpha_threshold{0.5f};
    PathIntegratorSettings integrator;
    std::vector<PassRequest> passes;
};

struct SampleRange {
    std::uint32_t first{};
    std::uint32_t count{};
    std::uint32_t offset{};
    // Total AA samples for the render, not the size of this progressive
    // dispatch. Sampling tables and other whole-sequence state must be
    // derived from this value so chunking does not change the random stream.
    std::uint32_t total{};
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

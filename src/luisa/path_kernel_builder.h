#pragma once

#include "path_kernel_direct_light_task.h"
#include "path_kernel_scene_geometry_plan.h"
#include "path_kernel_stage_policy.h"
#include "path_kernel_volume_state.h"
#include "path_tracer_camera.h"
#include "path_tracer_environment.h"
#include "path_tracer_geometry.h"
#include "path_tracer_light_distribution.h"
#include "path_tracer_light_tree.h"
#include "path_tracer_lighting.h"
#include "path_tracer_surfaces.h"

#include <psycles/luisa/surface_ray.h>
#include <psycles/luisa/volume_guiding.h>
#include <psycles/sampling/light_distribution.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

namespace luisa::compute {
template <typename T> class Coroutine;
}// namespace luisa::compute

namespace psycles::luisa_backend::detail {

class CausticsTaskSink;
class DirectLightTraceRecorder;

using RenderKernelSignature = void(Buffer<luisa::float4>, Buffer<luisa::float4>,
                                   Buffer<luisa::float4>, Buffer<luisa::float4>,
                                   Buffer<luisa::uint>, Buffer<luisa::float4>,
                                   Buffer<luisa::uint>, Buffer<luisa::float4>,
                                   std::uint32_t, std::uint32_t,
                                   Buffer<luisa::float4>, Buffer<float>,
    RenderKernelParameters);

template <typename Signature> struct RenderProgramTypes;

template <typename... Args> struct RenderProgramTypes<void(Args...)> {
    using SerialKernel = luisa::compute::Kernel2D<Args...>;
    using SerialCompiledShader = luisa::compute::Shader2D<Args...>;
    using SampleKernel = luisa::compute::Kernel3D<Args...>;
    using SampleCompiledShader = luisa::compute::Shader3D<Args...>;
    using Coroutine = luisa::compute::Coroutine<void(Args...)>;
};

using RenderSerialKernel =
    RenderProgramTypes<RenderKernelSignature>::SerialKernel;
using RenderSerialCompiledShader =
    RenderProgramTypes<RenderKernelSignature>::SerialCompiledShader;
using RenderSampleKernel =
    RenderProgramTypes<RenderKernelSignature>::SampleKernel;
using RenderSampleCompiledShader =
    RenderProgramTypes<RenderKernelSignature>::SampleCompiledShader;
using RenderCoroutine = RenderProgramTypes<RenderKernelSignature>::Coroutine;

// All data in this object exists on the host while Luisa records the kernel
// AST. Components may use ordinary C++ dispatch over it; the device still sees
// one fused path-tracing kernel and no extra Callable boundary.
struct PathKernelConfig {
    std::shared_ptr<LuisaSceneData> scene;
    CameraProjection camera_projection;
    bool camera_depth_of_field{};
    std::uint32_t camera_aperture_blades{};
    float camera_aperture_rotation{};
    bool next_event_estimation{};
    bool use_light_tree{};
    bool reflective_caustics{};
    bool refractive_caustics{};
    bool has_filter_glossy{true};
    bool has_subsurface{};
    // Host/JIT capability only: the authored bounce count remains a kernel
    // parameter, while a scene with replacement Fast GI disabled records no
    // AO-bounce predicate, object-distance load, or endpoint transfer at all.
    bool ambient_occlusion_bounce_approximation{};
    bool path_trace_enabled{};
    // Host/JIT diagnostic policy. When false, no histogram operation or
    // device-side enable predicate is recorded. The base addresses a private
    // suffix of the existing diagnostic buffer and is independent of image
    // dimensions and dispatch chunking.
    bool surface_closure_count_histogram_enabled{};
    std::uint32_t surface_closure_count_histogram_base{};
    bool staged_surface_sorting{};
    // Graph/persistent compatibility paths retain the legacy export ABI.
    // Host wavefront uses the application's typed Extension/Handler instead.
    bool staged_surface_sort_annotation{};
  // Optional host/JIT execution policy. When present, the canonical direct-
  // light transport stage publishes its reduced visibility task instead of
  // evaluating it inline. The material/lighting algorithm remains the same
  // PathKernelPipeline in both cases.
  std::shared_ptr<const DirectLightTaskSink> direct_light_task_sink;
  std::shared_ptr<const CausticsTaskSink> caustics_task_sink;
    std::uint32_t volume_stack_size{};
    std::uint32_t transparent_max_bounces{~0u};
    std::uint32_t max_volume_shadow_steps{1024u};
    bool camera_may_be_inside_volume{};
  std::shared_ptr<const PathVolumeStateComponent> volume_state;
    LightTransportCallables light_transport;
    LightDistributionSampleCallable light_distribution_sample;
    LightTreeCallables light_tree;
    SurfaceCallables surfaces;
    std::shared_ptr<const BackgroundSamplingDistribution> background_sampling;
    std::shared_ptr<const VolumeMajorantRuntime> volume_majorants;
    // Returns the bounded hit batch as a complete value. Split kernels retain
    // that value across INTERSECT_SHADOW -> SHADE_SHADOW; synchronous consumers
    // keep it local. Storage policy is selected while recording the DSL.
    std::shared_ptr<const ShadowIntersectionComponent> intersect_shadow;
    EvaluateShadowSurfaceCallable shade_shadow_surface;
    TraceShadowCallable trace_shadow;
    const luisa::compute::Buffer<luisa::ulong> *megakernel_stage_profile_buffer{
        nullptr};
    bool enable_virtual_texture_cache{false};
    bool spectral_rendering{false};
    contract::SpectralMode spectral_mode{contract::SpectralMode::none};
    bool xyz3_force_collapse{false};
    contract::CausticsMethod caustics_method{contract::CausticsMethod::mnee};
    bool sms_unbiased{true};
    std::uint32_t sms_max_trials{8u};

    [[nodiscard]] contract::SpectralMode
    effective_spectral_mode() const noexcept {
        if (spectral_mode != contract::SpectralMode::none) {
            return spectral_mode;
        }
        return spectral_rendering ? contract::SpectralMode::hero4
                                  : contract::SpectralMode::none;
    }
    [[nodiscard]] bool is_hero4() const noexcept {
        return effective_spectral_mode() == contract::SpectralMode::hero4;
    }
    [[nodiscard]] bool is_xyz3() const noexcept {
        return effective_spectral_mode() == contract::SpectralMode::xyz3;
    }
    [[nodiscard]] bool xyz3_rough_dispersion() const noexcept {
        return is_xyz3() && scene && scene->has_dispersive_materials;
    }

    [[nodiscard]] bool caustics_active() const noexcept {
        return (caustics_method == contract::CausticsMethod::mnee ||
                caustics_method == contract::CausticsMethod::sms) &&
               scene && scene->scene_has_mnee_caustics();
    }
    [[nodiscard]] bool caustics_is_sms() const noexcept {
        return caustics_method == contract::CausticsMethod::sms &&
               scene && scene->scene_has_mnee_caustics() &&
               scene->caustics_caster_triangle_count > 0u;
    }
};

struct MegakernelStageProfileState {
    static constexpr std::uint32_t stage_setup_and_film = 0u;
    static constexpr std::uint32_t stage_closest_trace = 1u;
    static constexpr std::uint32_t stage_volume_and_background = 2u;
    static constexpr std::uint32_t stage_surface_geometry = 3u;
    static constexpr std::uint32_t stage_surface_svm_shading = 4u;
    static constexpr std::uint32_t stage_nee_light_and_bsdf_eval = 5u;
    static constexpr std::uint32_t stage_bsdf_scatter = 6u;
    static constexpr std::uint32_t stage_shadow_trace = 7u;
    static constexpr std::uint32_t stage_subsurface_trace = 8u;
    static constexpr std::uint32_t stage_shadow_light_nee = 9u;
    static constexpr std::uint32_t stage_count = 10u;

    static constexpr std::uint32_t count_closest_traces = 0u;
    static constexpr std::uint32_t count_surface_shades = 1u;
    static constexpr std::uint32_t count_nee_evals = 2u;
    static constexpr std::uint32_t count_shadow_traces = 3u;
    static constexpr std::uint32_t count_subsurface_traces = 4u;
    static constexpr std::uint32_t counter_count = 5u;
    static constexpr std::uint32_t stripe_stride = 16u;
    static constexpr std::uint32_t stripe_count = 64u;

    luisa::compute::ArrayVar<luisa::ulong, stage_count> cycles;
    luisa::compute::ArrayVar<luisa::ulong, counter_count> counts;
    luisa::compute::ULong t_last;
    luisa::compute::UInt current_stage;

    void enter(std::uint32_t next_stage) noexcept {
        const auto now = luisa::compute::device_clock();
        cycles[current_stage] += now - t_last;
        t_last = now;
        current_stage = next_stage;
    }

    void bump(std::uint32_t counter_index) noexcept {
        counts[counter_index] += 1ull;
    }
};

// A zero scene population is an exact proof that the corresponding emitter
// kind can never be returned by either the flat distribution or the light
// tree. Keep this as a host-stage plan: recording a device-side kind check for
// a provably absent emitter still retains the complete component AST and makes
// kernel size depend on unsupported alternatives rather than scene
// capabilities. Positive populations remain conservative because individual
// zero-weight emitters are intentionally not inferred from aggregate counts.
struct DirectLightingStagePlan {
    bool environment{};
    bool emissive_mesh{};
    bool analytic{};

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return static_cast<std::size_t>(environment) +
               static_cast<std::size_t>(emissive_mesh) +
               static_cast<std::size_t>(analytic);
    }

    // Proposal generation is specialized per reachable emitter kind, but
    // every accepted proposal has the same shadow-transport continuation.
    // The continuation is therefore recorded exactly once whenever at least
    // one proposal stage exists, independently of the number of kinds.
  [[nodiscard]] constexpr std::size_t transport_stage_count() const noexcept {
        return size() == 0u ? 0u : 1u;
    }
};

struct PathKernelSceneStagePlan {
    // Analytic lamps are transparent path endpoints in Cycles, independently
    // of whether next-event estimation is enabled. A zero uploaded population
    // proves that neither the software intersection loop nor forward-light
    // shading can be reached.
    bool analytic_light_endpoints{};
    SceneTraversalStagePlan traversal{};
    DirectLightingStagePlan direct_lighting{};
};

[[nodiscard]] constexpr DirectLightingStagePlan
make_direct_lighting_stage_plan(bool next_event_estimation,
    bool environment_in_distribution,
    std::uint32_t emissive_triangle_count,
    std::uint32_t analytic_light_count) noexcept {
  return {.environment = next_event_estimation && environment_in_distribution,
        .emissive_mesh =
              next_event_estimation && emissive_triangle_count != 0u,
          .analytic = next_event_estimation && analytic_light_count != 0u};
}

[[nodiscard]] constexpr PathKernelSceneStagePlan
make_path_kernel_scene_stage_plan(
    bool next_event_estimation, bool environment_in_distribution,
    std::uint32_t emissive_triangle_count, std::uint32_t analytic_light_count,
    std::size_t triangle_geometry_count,
    std::size_t curve_geometry_count,
    std::optional<std::uint32_t> intersectable_analytic_light_count =
        std::nullopt) noexcept {
  const auto endpoint_light_count =
      intersectable_analytic_light_count.value_or(analytic_light_count);
  return {.analytic_light_endpoints = endpoint_light_count != 0u,
          .traversal = make_scene_traversal_stage_plan(
              triangle_geometry_count, curve_geometry_count),
          .direct_lighting = make_direct_lighting_stage_plan(
              next_event_estimation, environment_in_distribution,
              emissive_triangle_count, analytic_light_count)};
}

// Exact host-stage reachability predicate for the surface -> shadow side
// queue. A registered producer must name a materialized continuation, and a
// queue with no possible proposal is dead work. Diagnostic path tracing stays
// inline because its ordered event schema observes the shadow synchronously.
[[nodiscard]] constexpr bool
can_stage_direct_light_queue(bool requested, bool path_trace_enabled,
                             const PathKernelSceneStagePlan &plan) noexcept {
  return requested && !path_trace_enabled &&
         !plan.traversal.primitives.empty() &&
         plan.direct_lighting.transport_stage_count() != 0u;
}

// Host AST-construction state, never a device-side branch. Serial film
// accumulation has one writer per pixel; per-sample 3D dispatch follows
// Cycles GPU and atomically contributes each sample to the shared film.
enum class PathFilmAccumulation : std::uint8_t {
    serial,
    atomic,
    warp_reduce,
};

[[nodiscard]] inline bool simd_sample_parallel_enabled(
    const PathKernelConfig &config) noexcept {
    if (!config.scene || !static_cast<bool>(config.scene->device)) {
        return false;
    }
    if (config.scene->device.backend_name() != "simd" ||
        config.scene->device.compute_warp_size() != 8u ||
        config.path_trace_enabled ||
        config.direct_light_task_sink != nullptr) {
        return false;
    }
    if (const char *ww = std::getenv("LUISA_SIMD_WARP_WIDTH");
        ww != nullptr && *ww != '\0' && std::string_view{ww} != "8") {
        return false;
    }
    if (const char *raw = std::getenv("PSYCLES_SIMD_SAMPLE_PARALLEL");
        raw != nullptr && *raw != '\0') {
        return std::string_view{raw} != "0" && std::string_view{raw} != "off";
    }
    return false;
}

struct PathKernelInvocation {
    const PathKernelConfig &config;
    PathFilmAccumulation film_accumulation;
    const BufferFloat4 &combined;
    const BufferFloat4 &normal;
    const BufferFloat4 &albedo;
    const BufferFloat4 &light_passes;
    const BufferUInt &sample_count;
    const BufferFloat4 &volume_guiding_raw;
    const BufferUInt &volume_guiding_denoised;
    const BufferFloat4 &path_trace;
    const UInt &sample_first;
    const BufferFloat4 &sobol_table;
    const BufferFloat &filter_table;
    const Var<RenderKernelParameters> &parameters;

    UInt pixel;
    UInt local_x;
    UInt local_y;
    UInt full_x;
    UInt full_y;
    Float4 combined_sum;
    Float4 normal_sum;
    Float4 albedo_sum;
    UInt light_pass_base;
    Float4 diffuse_direct_sum;
    Float4 diffuse_indirect_sum;
    Float4 glossy_direct_sum;
    Float4 glossy_indirect_sum;
    Float4 transmission_direct_sum;
    Float4 transmission_indirect_sum;
    Float4 volume_direct_sum;
    Float4 volume_indirect_sum;
    Float4 emission_sum;
    Float4 environment_sum;
    Float4 glossy_color_sum;
    Float4 transmission_color_sum;
    UInt completed;
    UInt volume_guiding_raw_base;
    Float4 volume_guiding_scatter_sum;
    Float4 volume_guiding_transmit_sum;
    Float4 volume_guiding_optical_depth_sum;
    Float3 volume_guiding_scattered_radiance;
    Float3 volume_guiding_transmitted_radiance;
    SurfaceQuery surface_query;
    std::shared_ptr<MegakernelStageProfileState> stage_profile{nullptr};

    [[nodiscard]] Float3 clamp_contribution(Float3 contribution,
                                            UInt depth) const noexcept;
    [[nodiscard]] Float3
  clamp_emission_contribution(Float3 contribution,
        UInt path_depth) const noexcept;
    [[nodiscard]] Float sample_light_roulette(Float3 unshadowed_contribution,
                                              Float random) const noexcept;
  [[nodiscard]] Float volume_guiding_majorant_optical_depth() const noexcept;
  [[nodiscard]] Float3 constant_environment() const noexcept;
  [[nodiscard]] Float4 constant_environment_spectral(
      const cycles_spectrum::SpectralContext &spectral) const noexcept;

    void write_film() noexcept;
};

[[nodiscard]] PathKernelInvocation begin_path_kernel(
    const PathKernelConfig &config, PathFilmAccumulation film_accumulation,
    UInt pixel, const BufferFloat4 &combined, const BufferFloat4 &normal,
    const BufferFloat4 &albedo, const BufferFloat4 &light_passes,
    const BufferUInt &sample_count, const BufferFloat4 &volume_guiding_raw,
    const BufferUInt &volume_guiding_denoised, const BufferFloat4 &path_trace,
    const UInt &sample_first, const BufferFloat4 &sobol_table,
                  const BufferFloat &filter_table,
                  const Var<RenderKernelParameters> &parameters) noexcept;

// A pending BSSRDF exit can only originate from a committed surface
// candidate. Its hit kind is therefore a type-level invariant rather than
// mutable path state; materialization restores the canonical Surface tag.
struct PendingSubsurfaceHit {
    UInt instance;
    UInt primitive;
    Float2 barycentric;
    Float committed_ray_t;

  void store_surface(UInt selected_instance, UInt selected_primitive,
                       Float2 selected_barycentric,
                       Float selected_ray_t) noexcept;
  void store_surface(const Var<luisa::compute::CommittedHit> &source) noexcept;
    [[nodiscard]] Var<luisa::compute::CommittedHit>
    materialize_surface() const noexcept;
};

struct PathSampleContext {
    PathKernelInvocation &invocation;
    UInt sample_index;
    UInt cycles_y;
    UInt rng_hash;
    Float2 filter_sample;
    Float3 lens_time_sample;
    Bool path_trace_active;
    Var<luisa::compute::Ray> ray;
    Float ray_dP;
    Float ray_dD;
    UInt ray_source_object;
    UInt ray_source_primitive;
    Float3 radiance;
    cycles_spectrum::Spectrum throughput;
    std::optional<Float> spectral_u_lambda;
    std::optional<Float3> spectral_pdf_ratio;
    Float3 sample_normal;
    Float3 sample_albedo;
    Float3 sample_glossy_color;
    Float3 sample_transmission_color;
    Float3 sample_diffuse_direct;
    Float3 sample_diffuse_indirect;
    Float3 sample_glossy_direct;
    Float3 sample_glossy_indirect;
    Float3 sample_transmission_direct;
    Float3 sample_transmission_indirect;
    Float3 sample_volume_direct;
    Float3 sample_volume_indirect;
    Float3 sample_emission;
    Float3 sample_environment;
    Float3 volume_guiding_scatter;
    Float3 volume_guiding_transmit;
    // Matches Cycles' render-buffer representation: the Combined fourth
    // component accumulates transparency, and alpha is derived only after
    // sample normalization.
    Float sample_transparency;
    Float previous_bsdf_pdf;
    Float3 previous_mis_origin_normal;
    // Cycles ray.previous_dt: full volume segment length associated with
    // mis_origin_n. It is not generally the distance to the sampled collision.
    Float previous_light_tree_dt;
    Float minimum_bsdf_pdf;
    // INTERSECT_CLOSEST owns Cycles' continuation decision. Later surface or
    // volume stages retain only its probability for survivor renormalization.
    Float continuation_probability;
    Float3 path_diffuse_weight;
    Float3 path_glossy_weight;
    UInt ray_events;
    UInt diffuse_depth;
    UInt glossy_depth;
    UInt transparent_depth;
    UInt transmission_depth;
    UInt portal_depth;
    UInt path_depth;
    UInt path_flags;
    UInt cycles_path_visibility;
    UInt cycles_rng_offset;
    PathVolumeState volume;
    UInt volume_bounce;
    UInt volume_bounds_bounce;
    Float optical_depth;
    // A successful BSSRDF traversal carries its exact selected intersection
    // into the next shading iteration. Re-tracing a short ray would change
    // the hit identity at shared edges and would not match Cycles.
    Bool pending_subsurface_exit;
    PendingSubsurfaceHit pending_subsurface_hit;
    UInt *vt_miss_page{nullptr};
    std::function<void(luisa::compute::Expr<luisa::uint>)> vt_wait_resident;

    // Traversal visibility is a pure projection of the canonical Cycles path
    // visibility. Re-materializing it at each use prevents a duplicate frame
    // field from drifting away from the state that defines it.
    [[nodiscard]] UInt traversal_ray_visibility() const noexcept;
    // Shader evaluation observes the canonical path classification. In
    // particular, TRANSMIT does not erase DIFFUSE/GLOSSY as it does for BVH
    // visibility; Cycles Light Path nodes consume this projection.
    [[nodiscard]] UInt shader_ray_visibility() const noexcept;
    // These predicates are projections of the canonical Cycles path flags,
    // not independently mutable path state. Keeping one source of truth also
    // preserves PATH_RAY_MIS_SKIP for transparent ray-portal transitions.
    [[nodiscard]] Bool single_pass_recorded() const noexcept;
    [[nodiscard]] Bool mis_competition_skipped() const noexcept;
    [[nodiscard]] Bool terminate_after_transparent_requested() const noexcept;
    [[nodiscard]] Bool terminate_on_next_surface_requested() const noexcept;
    // Pure Cycles Sobol lookup for PRNG_TERMINATE. Keeping it independent of
    // the light sampler prevents surface-only random state from crossing the
    // intersect-to-shade scheduling boundary.
    [[nodiscard]] Float continuation_terminate_sample() const noexcept;
    [[nodiscard]] Float3 surface_bsdf_sample() const noexcept;
    [[nodiscard]] Float3 trace_uint32(UInt value) const noexcept;
    void trace_write(UInt slot, Float3 value) const noexcept;
    void trace_write_global(path_trace_schema::GlobalSlot slot,
                            Float3 value) const noexcept;
  void trace_write_event(UInt event, path_trace_schema::EventSlot slot,
                           Float3 value) const noexcept;
  void trace_write_shadow_event(UInt event,
        path_trace_schema::ShadowEventSlot slot,
        Float3 value) const noexcept;
  void trace_write_forward_event(UInt event,
        path_trace_schema::ForwardEventSlot slot,
        Float3 value) const noexcept;
  void trace_write_closure(UInt event, std::uint32_t closure,
                           std::uint32_t field, Float3 value) const noexcept;
    void record_surface_closure_count(UInt count) const noexcept;
    // These are host-specialized film operations. Serial construction updates
    // the per-pixel local accumulators; per-sample construction emits direct
    // atomic pass writes at the contribution point. No device-side scheduler
    // or accumulation-mode branch is generated.
  void accumulate_light_pass(LightPassBuffer pass,
        Float3 contribution) noexcept;
  void accumulate_light_pass(UInt pass, Float3 contribution) noexcept;
    void accumulate_normal_pass(Float3 contribution) noexcept;
    void accumulate_albedo_pass(Float3 contribution) noexcept;
  void accumulate_scattered_light_at_state(Float3 contribution, UInt flags,
                                            Float3 diffuse, Float3 glossy,
                                            Bool direct) noexcept;
  void accumulate_emission_or_background(Float3 contribution,
                                          LightPassBuffer pass) noexcept;
  void
  accumulate_radiance(Float3 contribution,
                      Bool primary_volume_scatter_override = false) noexcept;
  // Accumulates an event using an explicitly captured path classification.
  // This is semantically identical to accumulate_radiance() at the capture
  // point, but permits later scheduling stages to run after the canonical
  // path state has advanced.
  void accumulate_radiance_at_state(
      Float3 contribution, UInt event_path_flags,
      UInt event_path_visibility, UInt event_path_depth,
      Bool primary_volume_scatter_override = false) noexcept;
  void accumulate_transparency(Float transparency) noexcept;
    [[nodiscard]] contract::SpectralMode spectral_mode() const noexcept;
    [[nodiscard]] bool is_spectral() const noexcept;
    [[nodiscard]] bool is_hero4() const noexcept;
    [[nodiscard]] bool is_xyz3() const noexcept;
    [[nodiscard]] Bool spectral_secondary_terminated() const noexcept;
    [[nodiscard]] Bool spectral_lane_terminated() const noexcept;
    [[nodiscard]] Expr<luisa::float4> throughput_eval4() const noexcept;
    [[nodiscard]] std::optional<cycles_spectrum::SampledWavelengths>
    sampled_wavelengths() const noexcept;
    [[nodiscard]] std::optional<cycles_spectrum::SpectralContext>
    spectral_context() const noexcept;
    [[nodiscard]] Float3 spectrum_to_rgb(
        const cycles_spectrum::Spectrum &value) const noexcept;
    [[nodiscard]] Float3 reflectance_to_rgb(
        const cycles_spectrum::Spectrum &reflectance) const noexcept;
    [[nodiscard]] Float throughput_average() const noexcept;
    [[nodiscard]] cycles_spectrum::Spectrum
    spectrum_from_eval(Expr<luisa::float4> eval4) const noexcept;
    [[nodiscard]] Float xyz3_mis_pdf(
        Expr<float> scalar_pdf,
        const std::optional<Float4> &lane_pdf) const noexcept;
    [[nodiscard]] Float apply_xyz3_bsdf_weight(
        Expr<luisa::float4> f_eval, Expr<float> scalar_pdf,
        const std::optional<Float4> &lane_pdf) noexcept;
    void collapse_xyz3(Expr<float> hero_bsdf_weight) noexcept;
    [[nodiscard]] Float xyz3_hero_caustic_factor() const noexcept;
    [[nodiscard]] Float3
  analytic_light_constant_shader(Var<LightGpu> light) const noexcept;
    [[nodiscard]] Float3
  analytic_light_shader(Var<LightGpu> light, UInt light_index,
                        Float3 light_position, Float3 light_normal,
                        Float2 light_uv, Float3 incoming,
                          Float light_distance) const noexcept;
    [[nodiscard]] cycles_spectrum::Spectrum
  analytic_light_constant_shader_spectrum(Var<LightGpu> light) const noexcept;
    [[nodiscard]] cycles_spectrum::Spectrum
  analytic_light_shader_spectrum(Var<LightGpu> light, UInt light_index,
                                 Float3 light_position, Float3 light_normal,
                                 Float2 light_uv, Float3 incoming,
                                 Float light_distance) const noexcept;
};

[[nodiscard]] PathSampleContext
begin_path_sample(PathKernelInvocation &invocation,
                  const UInt &sample_offset) noexcept;
void reset_path_sample(PathSampleContext &sample,
                       const UInt &sample_offset) noexcept;
void accumulate_path_sample(PathSampleContext &sample) noexcept;

struct PathBounceRandomState {
    Float3 light_sample;
    Var<LightDistributionGpu> selected_light;
    Float light_terminate_sample;
};

struct PathBounceContext {
    PathSampleContext &sample;
    const UInt &path_step;
    // Host/JIT link scoped to its consuming volume segment or active surface
    // NEE branch. It never crosses a coroutine boundary or becomes a device
    // ABI field. Surface tracing records pure random tuples independently.
    PathBounceRandomState *random_state;
    Var<luisa::compute::CommittedHit> hit;
    Bool subsurface_exit;

    [[nodiscard]] PathBounceRandomState &random() const noexcept {
        LUISA_ASSERT(random_state != nullptr, "Path random state has no active consumer.");
        return *random_state;
    }
};

// Exactly one of analytic_light, surface, and background is true. The
// distance is measured on the current ray and identifies the complete
// free-flight segment which must be resolved before the event itself.
struct ClosestPathEvent {
    PathBounceContext &bounce;
    Bool analytic_light;
    Bool surface;
    Bool background;
    Float distance;
    UInt light_index;
    Float3 light_position;
    Float3 light_normal;
    Float2 light_uv;
    Float light_pdf;
    Float light_evaluation_factor;
    // True only if no surface-emission or in-volume contribution remains
    // after a failed continuation decision.
    Bool terminated;
    // Potential endpoint emission and sampled-light participation are not the
    // same predicate: EmissionSampling::NONE remains visibly emissive but has
    // no competing light-sampling PDF.
    Bool surface_may_emit;
};

struct VolumeSegmentEvent {
    Bool scattered;
    Bool terminated;
};

struct SurfaceGeometryContext {
    PathBounceContext &bounce;
    // Host/JIT metadata; this field never crosses the device ABI.
    ScenePrimitiveStagePlan primitive_plan;
    UInt emission_sampling;
    Float3 wp0;
    Float3 wp1;
    Float3 wp2;
    Float3 geometric_normal;
    Float3 hit_position;
    Float differential_radius;
    Bool is_curve;
    UInt cycles_primitive_type;
    Bool cycles_transform_applied;
    Bool triangle_smooth;
    Float3 shadow_shading_normal;
    UInt surface_tag;
    UInt cycles_surface_shader;
    UInt cycles_object_index;
    UInt cycles_primitive_index;
    std::optional<UInt> cycles_shader_flags;
    std::optional<UInt> cycles_object_flags;
    VolumeStackEntry volume_stack_entry;
    Bool surface_has_volume;
    SurfacePoint point;
    SurfaceQuery path_surface_query;
    Float shadow_terminator_geometry_offset{0.0f};

    // World-space triangle vertices are needed only by forward-hit emission
    // MIS. Populate them at that consumer so non-emissive triangles do not
    // pay three object-to-world transforms during every surface event.
    [[nodiscard]] std::tuple<Float3, Float3, Float3>
    resolve_world_triangle_vertices() noexcept;
    [[nodiscard]] luisa::compute::Float4x4
    fetch_world_to_object() const noexcept;

    // Retain the one graph evaluation's ShaderData::N-equivalent for the
    // shadow-terminator geometry consumer without replaying the graph.
    void set_evaluated_shadow_shading_normal(Float3 normal) noexcept;
    [[nodiscard]] Float3 make_ray_origin(Float3 direction) const noexcept;
    [[nodiscard]] surface_ray::ShadowOrigin
    make_shadow_origin(Float3 direction) const noexcept;
    [[nodiscard]] surface_ray::ShadowOrigin
    make_shadow_terminator_origin(Float3 direction) const noexcept;
    [[nodiscard]] Float3 apply_shadow_certificate(
        Float3 position, Float3 direction, Bool skip_self) const noexcept;
};

struct SurfaceShadingState {
    UInt cycles_surface_runtime_flags;
    Float3 bsdf_sample;
    // Final graph-level shading normal. The raw SurfacePoint remains intact
    // for the legacy replay route, so it cannot apply Bump twice.
    Float3 shading_normal;
    // Host-only ownership of device-local populated closures. It lives across
    // the C++ AST-construction calls for NEE and scattering, but no pointer or
    // closure record crosses a device coroutine suspension.
    std::shared_ptr<PopulatedSurfaceShader> populated_surface;

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        PathKernelInvocation &invocation,
        UInt surface_tag,
        const SurfacePoint &point,
        Float3 outgoing,
        const SurfaceQuery &query,
        UInt shader_flags) const noexcept;
    [[nodiscard]] SurfaceSample sample(
        PathKernelInvocation &invocation,
        UInt surface_tag,
        const SurfacePoint &point,
        Float u_lobe,
        Float2 u_direction,
        const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfaceSampleTrace sample_trace(
        PathKernelInvocation &invocation,
        UInt surface_tag,
        const SurfacePoint &point,
        Float u_lobe,
        Float2 u_direction,
        const SurfaceQuery &query) const noexcept;
};

struct DirectLightingContext {
    PathBounceContext &bounce;
    SurfaceGeometryContext &surface;
    SurfaceShadingState &shading;
};

// Cycles' LightSample-equivalent state after one emitter-specific proposal
// has been accepted. This is a host/JIT aggregate of typed Luisa variables,
// not a device ABI. All proposal predicates refine distinct values of
// LightDistributionEmitterKind, so their writes are pairwise disjoint. That
// invariant permits the three emitter implementations to merge here before
// recording the receiving BSDF evaluator exactly once.
struct DirectLightSampleState {
    Float3 direction;
    Float3 target_position;
    Float3 light_normal;
    Float2 light_uv;
    Float2 barycentric;
    // Radiometric factors known without evaluating the emitter shader (for
    // example analytic-lamp strength and sampling normalization).
    Float3 radiometric_weight;
    // Constant emission is populated before receiving-BSDF evaluation;
    // non-constant emission is populated afterwards by the owning provider.
    Float3 light_shader;
    Float pdf;
    // Preserve the existing per-emitter normalization contract independently
    // from the PDF reported to MIS and tracing.
    Float normalization_pdf;
    Float distance;
    UInt emitter_kind;
    UInt emitter_index;
    UInt light_object;
    UInt light_primitive;
    UInt shader_flags;
    Bool apply_mis;
    Bool constant_light_shader;
    Bool distant;
    Bool valid;
    std::optional<Float4> spectral_radiometric_weight{};
    std::optional<Float4> spectral_light_shader{};

    [[nodiscard]] static DirectLightSampleState
    empty(contract::SpectralMode spectral_mode = contract::SpectralMode::none) noexcept;
    void accept(const DirectLightSampleState &proposal) noexcept;
};

// Canonical continuation state produced by one emitter-specific NEE proposal
// stage and consumed by the single common shadow-transport stage. These are
// typed Luisa DSL variables, not a packed device ABI or a weak float-register
// protocol. The light-distribution kind is a disjoint sum, so at most one
// proposal component can accept the state for a path bounce.
struct DirectLightTransportState {
    Float3 weighted_bsdf;
    Float3 light_shader;
    Float3 direction;
    Float3 target_position;
    Float3 bsdf;
    Float3 diffuse_bsdf;
    Float3 glossy_bsdf;
    UInt light_object;
    UInt light_primitive;
    UInt shader_flags;
    Float average_roughness_squared;
    Float light_terminate_sample;
    Bool constant_light_shader;
    Bool distant;
    Bool valid;
    std::optional<Float4> spectral_weighted_bsdf{};
    std::optional<Float4> spectral_light_shader{};
    std::optional<Float4> spectral_bsdf{};
    std::optional<Float4> spectral_diffuse_bsdf{};
    std::optional<Float4> spectral_glossy_bsdf{};

    [[nodiscard]] static DirectLightTransportState
    empty(contract::SpectralMode spectral_mode = contract::SpectralMode::none) noexcept;
    void accept(const DirectLightTransportState &other) noexcept;
    void accept(const SurfaceEvaluation &evaluation,
                Float3 proposal_weighted_bsdf, Float3 proposal_light_shader,
                Float3 proposal_direction, Float3 proposal_target_position,
                Bool proposal_distant, UInt proposal_light_object,
                UInt proposal_light_primitive, UInt proposal_shader_flags,
                Bool proposal_constant_light_shader) noexcept;
    void accept(const SurfaceEvaluation &evaluation,
                const cycles_spectrum::Spectrum &proposal_weighted_bsdf,
                const cycles_spectrum::Spectrum &proposal_light_shader,
                Float3 proposal_direction, Float3 proposal_target_position,
                Bool proposal_distant, UInt proposal_light_object,
                UInt proposal_light_primitive, UInt proposal_shader_flags,
                Bool proposal_constant_light_shader,
                const cycles_spectrum::SpectralContext *spectral = nullptr) noexcept;
};

// Per-emitter host/JIT protocol matching Cycles' light-sample state machine:
// sample geometry and constant emission, evaluate the receiving BSDF once at
// the common merge, then evaluate non-constant emitter emission only when the
// receiving response is non-zero. Implementations retain typed light state;
// no closure is baked or serialized through this interface.
class DirectLightProvider {

  public:
    virtual ~DirectLightProvider() noexcept = default;
    virtual void sample() const noexcept = 0;
};

class PathBounceSetupStage {

  public:
    virtual ~PathBounceSetupStage() noexcept = default;
    [[nodiscard]] virtual PathBounceContext
    emit(PathSampleContext &sample, const UInt &path_step) const noexcept = 0;
};

class PathBounceRandomStage {

  public:
    virtual ~PathBounceRandomStage() noexcept = default;
    [[nodiscard]] virtual PathBounceRandomState
    emit(PathSampleContext &sample) const noexcept = 0;

    // Host recording helper: the callback and its random state share the
    // lifetime of one consumer. No device call boundary is introduced.
    template<typename F>
    void with_active_state(PathSampleContext &sample, Bool active,
                           F &&consume) const noexcept {
        $if(active) {
            auto state = emit(sample);
            std::forward<F>(consume)(state);
        };
    }
};

class ClosestEventStage {

  public:
    virtual ~ClosestEventStage() noexcept = default;
    [[nodiscard]] virtual ClosestPathEvent
    emit(PathBounceContext &bounce,
         Expr<std::uint32_t> excluded_analytic_light) const noexcept = 0;
};

class ForwardLightStage {

  public:
    virtual ~ForwardLightStage() noexcept = default;
    // Returns whether Cycles' transparent-bounce limit terminates the path.
  [[nodiscard]] virtual Bool emit(ClosestPathEvent &event) const noexcept = 0;
};

class PathVolumeSegmentStage {

  public:
  virtual ~PathVolumeSegmentStage() noexcept = default;
    [[nodiscard]] virtual VolumeSegmentEvent
    emit(ClosestPathEvent &event) const noexcept = 0;
};

class BackgroundEventStage {

  public:
    virtual ~BackgroundEventStage() noexcept = default;
    virtual void emit(ClosestPathEvent &event) const noexcept = 0;
};

class SurfaceGeometryStage {

  public:
    virtual ~SurfaceGeometryStage() noexcept = default;
    [[nodiscard]] virtual SurfaceGeometryContext
    emit(PathBounceContext &bounce) const noexcept = 0;
};

class SurfaceShadingStage {

  public:
    virtual ~SurfaceShadingStage() noexcept = default;
    [[nodiscard]] virtual SurfaceShadingState
    emit(SurfaceGeometryContext &surface,
         Bool *sample_terminated_out = nullptr) const noexcept = 0;
};

class DirectLightingComponent {

  public:
    virtual ~DirectLightingComponent() noexcept = default;
    [[nodiscard]] virtual std::unique_ptr<DirectLightProvider>
    make_light_provider(DirectLightingContext &context,
                        DirectLightSampleState &sample) const = 0;
};

class DirectLightTransportStage {

  public:
    virtual ~DirectLightTransportStage() noexcept = default;

    // Reduces all surface/closure data to the exact shadow-path state. The
    // returned record is deliberately self-contained so the pipeline may
    // finish surface scattering before entering a coroutine shadow stage.
    [[nodiscard]] virtual DirectLightTransportPreparation
    prepare(DirectLightingContext &context,
            const DirectLightTransportState &transport) const noexcept = 0;

    // True only when emit() records coroutine transitions. Megakernel and
    // detached-queue construction stay before surface scattering to avoid
    // lengthening their live ranges.
    [[nodiscard]] virtual bool defer_until_after_surface_scatter(
        PathCoroutineCutPolicy cut_policy) const noexcept = 0;

    virtual void emit(PathBounceContext &bounce,
                      DirectLightTransportPreparation preparation,
                      PathCoroutineCutPolicy cut_policy) const noexcept = 0;
};

class SurfaceScatterStage {

  public:
    virtual ~SurfaceScatterStage() noexcept = default;
    struct Result {
        SurfaceSample sample;
        Bool subsurface;
        // Surface sampling cannot terminate its caller internally once NEE
        // shadow work is deferred: a failed BSDF sample still keeps the NEE
        // contribution produced at this vertex. The pipeline consumes this
        // explicit continuation predicate after shadow completion.
        Bool valid;
    };
    [[nodiscard]] virtual Result
    emit(DirectLightingContext &context) const noexcept = 0;
};

// Exact sufficient state at Cycles' SHADE_SURFACE -> INTERSECT_SUBSURFACE
// boundary. The entry ray and intersection live in PathSampleContext's
// canonical ray/pending-hit state; the BSSRDF contributes only the same 40
// logical bytes as Cycles' packed subsurface state. Method and back-facing
// tags use the canonical PathRayFlag bits instead of widening this payload.
struct SubsurfaceTransportState {
    Float3 albedo;
    Float3 radius;
    Float encoded_anisotropy;
    Float3 normal;
};

struct SubsurfaceTransportPreparation {
    SubsurfaceTransportState state;
    Bool valid;
};

class SubsurfaceTransportStage {

  public:
    virtual ~SubsurfaceTransportStage() noexcept = default;
    // Samples random-walk entry and canonicalizes ray/intersection state at
    // the last point where populated surface data is available. Invalid
    // entries terminate before they consume a scheduler queue slot.
    [[nodiscard]] virtual SubsurfaceTransportPreparation
    prepare(DirectLightingContext &context,
            const SurfaceSample &sample) const noexcept = 0;
    // Returns false when the sampled profile finds no valid same-object exit;
    // Cycles terminates that path rather than substituting another closure.
    [[nodiscard]] virtual Bool
    emit(PathSampleContext &path,
         const SubsurfaceTransportState &state) const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<PathBounceSetupStage>
make_path_bounce_setup_stage(SceneTraversalStagePlan plan, bool has_subsurface,
                             bool ambient_occlusion_bounce_approximation);
[[nodiscard]] std::unique_ptr<PathBounceRandomStage>
make_path_bounce_random_stage();
[[nodiscard]] std::unique_ptr<ClosestEventStage>
make_closest_event_stage(bool analytic_light_endpoints,
    ScenePrimitiveStagePlan primitive_plan,
    bool ambient_occlusion_bounce_approximation);
[[nodiscard]] std::unique_ptr<ForwardLightStage> make_forward_light_stage();
[[nodiscard]] std::unique_ptr<PathVolumeSegmentStage>
make_path_volume_segment_stage(const PathKernelConfig &config);
[[nodiscard]] std::unique_ptr<BackgroundEventStage>
make_background_event_stage(bool ambient_occlusion_bounce_approximation);
[[nodiscard]] std::unique_ptr<SurfaceGeometryStage>
make_surface_geometry_stage(ScenePrimitiveStagePlan plan);
class SceneTraversalComponent;

struct SurfaceShadingStageOptions {
    std::shared_ptr<const SurfacePopulationComponent> population;
    std::shared_ptr<const SceneTraversalComponent> traversal;
    bool may_have_emission{true};
    bool may_have_holdout{true};
};

[[nodiscard]] std::unique_ptr<SurfaceShadingStage>
make_surface_shading_stage(SurfaceShadingStageOptions options = {});
[[nodiscard]] std::unique_ptr<DirectLightingComponent>
make_environment_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace);
[[nodiscard]] std::unique_ptr<DirectLightingComponent>
make_emissive_mesh_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace);
[[nodiscard]] std::unique_ptr<DirectLightingComponent>
make_analytic_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace);
[[nodiscard]] std::unique_ptr<DirectLightTransportStage>
make_direct_light_transport_stage(
    std::shared_ptr<const DirectLightTraceRecorder> trace,
    DirectLightTaskEvaluator evaluator,
    std::shared_ptr<const DirectLightTaskSink> task_sink,
    bool path_trace_enabled);
[[nodiscard]] std::unique_ptr<SurfaceScatterStage> make_surface_scatter_stage();
[[nodiscard]] std::unique_ptr<SubsurfaceTransportStage>
make_subsurface_transport_stage();

class PathKernelPipeline {

  private:
    class Impl;
    std::unique_ptr<Impl> _impl;

  public:
  explicit PathKernelPipeline(const PathKernelConfig &config);
    ~PathKernelPipeline() noexcept;
    PathKernelPipeline(PathKernelPipeline &&) noexcept;
    PathKernelPipeline &operator=(PathKernelPipeline &&) noexcept;
    PathKernelPipeline(const PathKernelPipeline &) = delete;
    PathKernelPipeline &operator=(const PathKernelPipeline &) = delete;

    using SampleCompleteHandler =
        std::function<Bool(PathSampleContext &, UInt &)>;

    // `cut_policy` is a host-side AST-construction choice. It must never be a
    // device value: the megakernel contains neither dynamic scheduler branches
    // nor coroutine instructions, while coroutine variants annotate the same
    // semantic path program at the Cycles scheduling boundaries.
    void emit(PathSampleContext &sample,
              PathCoroutineCutPolicy cut_policy,
              const SampleCompleteHandler &on_sample_complete = {}) const noexcept;
};

[[nodiscard]] RenderSerialKernel
build_path_serial_kernel(const PathKernelConfig &config) noexcept;
[[nodiscard]] RenderSampleKernel
build_path_sample_kernel(const PathKernelConfig &config) noexcept;
[[nodiscard]] RenderCoroutine
build_path_coroutine(const PathKernelConfig &config,
                     PathCoroutineCutPolicy cut_policy);

} // namespace psycles::luisa_backend::detail

#pragma once

#include "path_kernel_scene_geometry_plan.h"
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
#include <memory>

namespace luisa::compute {
template<typename T>
class Coroutine;
}// namespace luisa::compute

namespace psycles::luisa_backend::detail {

class DirectLightTraceRecorder;

using RenderKernelSignature = void(
    Buffer<luisa::float4>,
    Buffer<luisa::float4>,
    Buffer<luisa::float4>,
    Buffer<luisa::float4>,
    Buffer<luisa::uint>,
    Buffer<luisa::float4>,
    Buffer<luisa::uint>,
    Buffer<luisa::float4>,
    std::uint32_t,
    std::uint32_t,
    Buffer<luisa::float4>,
    Buffer<float>,
    RenderKernelParameters);

template<typename Signature>
struct RenderProgramTypes;

template<typename... Args>
struct RenderProgramTypes<void(Args...)> {
    using SerialKernel = Kernel1D<Args...>;
    using SerialCompiledShader = Shader1D<Args...>;
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
using RenderCoroutine =
    RenderProgramTypes<RenderKernelSignature>::Coroutine;

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
    bool has_subsurface{};
    bool path_trace_enabled{};
    std::uint32_t volume_stack_size{};
    bool camera_may_be_inside_volume{};
    std::shared_ptr<const PathVolumeStateComponent>
        volume_state;
    LightTransportCallables light_transport;
    LightDistributionSampleCallable light_distribution_sample;
    LightTreeCallables light_tree;
    SurfaceCallables surfaces;
    EnvironmentCallables environment;
    TraceShadowCallable trace_shadow;
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
    [[nodiscard]] constexpr std::size_t
    transport_stage_count() const noexcept {
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
make_direct_lighting_stage_plan(
    bool next_event_estimation,
    bool environment_in_distribution,
    std::uint32_t emissive_triangle_count,
    std::uint32_t analytic_light_count) noexcept {
    return {
        .environment =
            next_event_estimation &&
            environment_in_distribution,
        .emissive_mesh =
            next_event_estimation &&
            emissive_triangle_count != 0u,
        .analytic =
            next_event_estimation &&
            analytic_light_count != 0u};
}

[[nodiscard]] constexpr PathKernelSceneStagePlan
make_path_kernel_scene_stage_plan(
    bool next_event_estimation,
    bool environment_in_distribution,
    std::uint32_t emissive_triangle_count,
    std::uint32_t analytic_light_count,
    std::size_t triangle_geometry_count,
    std::size_t curve_geometry_count,
    std::size_t completion_source_dense_count,
    std::size_t completion_source_sparse_count) noexcept {
    return {
        .analytic_light_endpoints =
            analytic_light_count != 0u,
        .traversal =
            make_scene_traversal_stage_plan(
                triangle_geometry_count,
                curve_geometry_count,
                completion_source_dense_count,
                completion_source_sparse_count),
        .direct_lighting =
            make_direct_lighting_stage_plan(
                next_event_estimation,
                environment_in_distribution,
                emissive_triangle_count,
                analytic_light_count)};
}

// Host AST-construction state, never a device-side branch. Serial film
// accumulation has one writer per pixel; per-sample 3D dispatch follows
// Cycles GPU and atomically contributes each sample to the shared film.
enum class PathFilmAccumulation : std::uint8_t {
    serial,
    atomic,
};

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

    [[nodiscard]] Float3 clamp_contribution(Float3 contribution,
                                            UInt depth) const noexcept;
    [[nodiscard]] Float3
    clamp_emission_contribution(
        Float3 contribution,
        UInt path_depth) const noexcept;
    [[nodiscard]] Float sample_light_roulette(Float3 unshadowed_contribution,
                                              Float random) const noexcept;
    [[nodiscard]] Float
    volume_guiding_majorant_optical_depth()
        const noexcept;
    [[nodiscard]] SurfaceEvaluation
    evaluate_light_surface(UInt surface_tag,
                           const SurfacePoint &point,
                           Float3 outgoing,
                           const SurfaceQuery &query,
                           UInt shader_flags) const noexcept;
    [[nodiscard]] SurfacePreparation
    prepare_surface(UInt surface_tag,
                    const SurfacePoint &point,
                    Float3 outgoing,
                    const SurfaceQuery &query,
                    Bool include_runtime_flags,
                    Bool include_aov) const noexcept;
    [[nodiscard]] Float3 surface_emission(UInt surface_tag,
                                          const SurfacePoint &point,
                                          Float3 outgoing) const noexcept;
    [[nodiscard]] Float3 constant_surface_emission(
        UInt surface_tag,
        UInt parameter_block) const noexcept;
    [[nodiscard]] SurfaceSample
    sample_surface(UInt surface_tag,
                   const SurfacePoint &point,
                   Float u_lobe,
                   Float2 u_direction,
                   const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfaceClosureTrace
    trace_surface_closure(UInt surface_tag,
                          const SurfacePoint &point,
                          UInt requested_index,
                          Bool reflective_caustics,
                          Bool refractive_caustics) const noexcept;
    [[nodiscard]] SurfaceSampleTrace
    trace_sample_surface(UInt surface_tag,
                         const SurfacePoint &point,
                         Float u_lobe,
                         Float2 u_direction,
                         const SurfaceQuery &query) const noexcept;
    [[nodiscard]] Float3
    surface_bssrdf_normal(UInt surface_tag,
                          const SurfacePoint &point,
                          Bool reflective_caustics,
                          Bool refractive_caustics) const noexcept;
    [[nodiscard]] Float3
    surface_shading_normal(UInt surface_tag,
                           const SurfacePoint &point) const noexcept;
    [[nodiscard]] Float3
    constant_environment() const noexcept;
    [[nodiscard]] Float3
    evaluate_environment(Float3 direction,
                         const cycles_path_state::ShaderEvaluationState
                             &shader_state) const noexcept;

    void write_film() noexcept;
};

[[nodiscard]] PathKernelInvocation
begin_path_kernel(const PathKernelConfig &config,
                  PathFilmAccumulation film_accumulation,
                  UInt pixel,
                  const BufferFloat4 &combined,
                  const BufferFloat4 &normal,
                  const BufferFloat4 &albedo,
                  const BufferFloat4 &light_passes,
                  const BufferUInt &sample_count,
                  const BufferFloat4 &volume_guiding_raw,
                  const BufferUInt &volume_guiding_denoised,
                  const BufferFloat4 &path_trace,
                  const UInt &sample_first,
                  const BufferFloat4 &sobol_table,
                  const BufferFloat &filter_table,
                  const Var<RenderKernelParameters> &parameters) noexcept;

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
    UInt ray_visibility;
    Float3 radiance;
    Float3 throughput;
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
    Bool primary_recorded;
    Float previous_bsdf_pdf;
    Float3 previous_mis_origin_normal;
    // Cycles ray.previous_dt: full volume segment length associated with
    // mis_origin_n. It is not generally the distance to the sampled collision.
    Float previous_light_tree_dt;
    Float minimum_bsdf_pdf;
    Bool previous_delta;
    Float continuation_probability;
    // The closest-event volume stage performs Cycles' roulette before
    // integrating the segment. An attenuated surface must reuse that exact
    // decision and divide only after recording surface emission.
    Bool continuation_decided_in_volume;
    Float3 path_diffuse_weight;
    Float3 path_glossy_weight;
    UInt ray_events;
    UInt diffuse_depth;
    UInt glossy_depth;
    UInt transparent_depth;
    UInt transmission_depth;
    UInt path_depth;
    UInt path_flags;
    UInt cycles_path_visibility;
    UInt cycles_rng_offset;
    PathVolumeState volume;
    UInt volume_bounce;
    UInt volume_bounds_bounce;
    Float optical_depth;
    Bool terminate_after_transparent;
    Bool terminate_on_next_surface;
    // A successful BSSRDF traversal carries its exact selected intersection
    // into the next shading iteration. Re-tracing a short ray would change
    // the hit identity at shared edges and would not match Cycles.
    Bool pending_subsurface_exit;
    Var<luisa::compute::CommittedHit> pending_subsurface_hit;

    [[nodiscard]] Float3 trace_uint32(UInt value) const noexcept;
    void trace_write(UInt slot, Float3 value) const noexcept;
    void trace_write_global(path_trace_schema::GlobalSlot slot,
                            Float3 value) const noexcept;
    void trace_write_event(UInt event,
                           path_trace_schema::EventSlot slot,
                           Float3 value) const noexcept;
    void trace_write_shadow_event(
        UInt event,
        path_trace_schema::ShadowEventSlot slot,
        Float3 value) const noexcept;
    void trace_write_closure(UInt event,
                             std::uint32_t closure,
                             std::uint32_t field,
                             Float3 value) const noexcept;
    void
    accumulate_light_pass(Var<LightPassContributionCall> contribution) noexcept;
    void accumulate_scattered_light(
        Float3 contribution) noexcept;
    void accumulate_radiance(
        Float3 contribution,
        Bool primary_volume_scatter_override =
            false) noexcept;
    void accumulate_transparency(
        Float transparency) noexcept;
    [[nodiscard]] Float3
    analytic_light_constant_shader(
        Var<LightGpu> light) const noexcept;
    [[nodiscard]] Float3
    analytic_light_shader(Var<LightGpu> light,
                          UInt light_index,
                          Float3 light_position,
                          Float3 light_normal,
                          Float2 light_uv,
                          Float3 incoming,
                          Float light_distance) const noexcept;
};

[[nodiscard]] PathSampleContext
begin_path_sample(PathKernelInvocation &invocation,
                  const UInt &sample_offset) noexcept;
void accumulate_path_sample(PathSampleContext &sample) noexcept;

struct PathBounceContext {
    PathSampleContext &sample;
    const UInt &path_step;
    Float terminate_sample;
    Float3 light_sample;
    Var<LightDistributionGpu> selected_light;
    Float light_terminate_sample;
    Float3 bsdf_sample;
    Var<luisa::compute::CommittedHit> hit;
    Float closest_surface_distance;
    Bool subsurface_exit;
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
    // Potential endpoint emission and sampled-light participation are not the
    // same predicate: EmissionSampling::NONE remains visibly emissive but has
    // no competing light-sampling PDF.
    Bool surface_may_emit;
    UInt surface_emission_sampling;
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
    Var<InstanceGpu> instance;
    Float3 p0;
    Float3 p1;
    Float3 p2;
    Float3 n0;
    Float3 n1;
    Float3 n2;
    luisa::compute::Float4x4 object_to_world;
    luisa::compute::Float4x4 world_to_object;
    Float3 wp0;
    Float3 wp1;
    Float3 wp2;
    Float3 geometric_normal;
    Float3 hit_position;
    Float3 object_hit_position;
    Float differential_radius;
    Bool is_curve;
    Bool cycles_transform_applied;
    Bool triangle_smooth;
    Float3 shadow_shading_normal;
    UInt surface_tag;
    UInt cycles_surface_shader;
    UInt cycles_object_index;
    UInt cycles_primitive_index;
    VolumeStackEntry volume_stack_entry;
    Bool surface_has_volume;
    SurfacePoint point;
    SurfaceQuery path_surface_query;

    [[nodiscard]] Float3 make_ray_origin(Float3 direction) const noexcept;
    [[nodiscard]] surface_ray::ShadowOrigin
    make_shadow_origin(Float3 direction) const noexcept;
};

struct SurfaceShadingState {
    UInt cycles_surface_runtime_flags;
};

struct DirectLightingContext {
    PathBounceContext &bounce;
    SurfaceGeometryContext &surface;
    SurfaceShadingState &shading;
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
    Bool distant;
    Bool valid;

    [[nodiscard]] static DirectLightTransportState empty() noexcept;
    void accept(const SurfaceEvaluation &evaluation,
                Float3 proposal_weighted_bsdf,
                Float3 proposal_light_shader,
                Float3 proposal_direction,
                Float3 proposal_target_position,
                Bool proposal_distant,
                UInt proposal_light_object,
                UInt proposal_light_primitive,
                UInt proposal_shader_flags) noexcept;
};

class PathBounceSetupStage {

  public:
    virtual ~PathBounceSetupStage() noexcept = default;
    [[nodiscard]] virtual PathBounceContext
    emit(PathSampleContext &sample, const UInt &path_step) const noexcept = 0;
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
    [[nodiscard]] virtual Bool
    emit(ClosestPathEvent &event) const noexcept = 0;
};

class PathVolumeSegmentStage {

  public:
    virtual ~PathVolumeSegmentStage() noexcept =
        default;
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
    emit(PathBounceContext &bounce,
         UInt emission_sampling) const noexcept = 0;
};

class SurfaceShadingStage {

  public:
    virtual ~SurfaceShadingStage() noexcept = default;
    [[nodiscard]] virtual SurfaceShadingState
    emit(SurfaceGeometryContext &surface) const noexcept = 0;
};

class DirectLightingComponent {

  public:
    virtual ~DirectLightingComponent() noexcept = default;
    virtual void prepare(
        DirectLightingContext &context,
        DirectLightTransportState &transport) const noexcept = 0;
};

class DirectLightTransportStage {

  public:
    virtual ~DirectLightTransportStage() noexcept = default;
    virtual void emit(
        DirectLightingContext &context,
        const DirectLightTransportState &transport) const noexcept = 0;
};

class SurfaceScatterStage {

  public:
    virtual ~SurfaceScatterStage() noexcept = default;
    struct Result {
        SurfaceSample sample;
        Bool subsurface;
    };
    [[nodiscard]] virtual Result
    emit(DirectLightingContext &context) const noexcept = 0;
};

class SubsurfaceTransportStage {

  public:
    virtual ~SubsurfaceTransportStage() noexcept = default;
    // Returns false when the sampled profile finds no valid same-object exit;
    // Cycles terminates that path rather than substituting another closure.
    [[nodiscard]] virtual Bool
    emit(DirectLightingContext &context,
         const SurfaceSample &sample) const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<PathBounceSetupStage>
make_path_bounce_setup_stage(
    SceneTraversalStagePlan plan);
[[nodiscard]] std::unique_ptr<ClosestEventStage>
make_closest_event_stage(
    bool analytic_light_endpoints,
    ScenePrimitiveStagePlan primitive_plan);
[[nodiscard]] std::unique_ptr<ForwardLightStage>
make_forward_light_stage();
[[nodiscard]] std::unique_ptr<PathVolumeSegmentStage>
make_path_volume_segment_stage(
    const PathKernelConfig &config);
[[nodiscard]] std::unique_ptr<BackgroundEventStage>
make_background_event_stage();
[[nodiscard]] std::unique_ptr<SurfaceGeometryStage>
make_surface_geometry_stage(
    ScenePrimitiveStagePlan plan);
[[nodiscard]] std::unique_ptr<SurfaceShadingStage> make_surface_shading_stage();
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
    std::shared_ptr<const DirectLightTraceRecorder> trace);
[[nodiscard]] std::unique_ptr<SurfaceScatterStage> make_surface_scatter_stage();
[[nodiscard]] std::unique_ptr<SubsurfaceTransportStage>
make_subsurface_transport_stage();

class PathKernelPipeline {

  private:
    class Impl;
    std::unique_ptr<Impl> _impl;

  public:
    explicit PathKernelPipeline(
        const PathKernelConfig &config);
    ~PathKernelPipeline() noexcept;
    PathKernelPipeline(PathKernelPipeline &&) noexcept;
    PathKernelPipeline &operator=(PathKernelPipeline &&) noexcept;
    PathKernelPipeline(const PathKernelPipeline &) = delete;
    PathKernelPipeline &operator=(const PathKernelPipeline &) = delete;

    // `is_coro` is a host-side AST-construction choice. It must never be a
    // device Bool: the megakernel contains neither dynamic scheduler branches
    // nor coroutine instructions, while the coroutine records suspension at
    // the same semantic path-event boundaries.
    void emit(PathSampleContext &sample, bool is_coro) const noexcept;
};

[[nodiscard]] RenderSerialKernel
build_path_serial_kernel(const PathKernelConfig &config) noexcept;
[[nodiscard]] RenderSampleKernel
build_path_sample_kernel(const PathKernelConfig &config) noexcept;
[[nodiscard]] RenderCoroutine
build_path_coroutine(const PathKernelConfig &config);

} // namespace psycles::luisa_backend::detail

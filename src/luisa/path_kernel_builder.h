#pragma once

#include "path_tracer_camera.h"
#include "path_tracer_environment.h"
#include "path_tracer_geometry.h"
#include "path_tracer_light_distribution.h"
#include "path_tracer_lighting.h"
#include "path_tracer_surfaces.h"

#include <psycles/luisa/surface_ray.h>
#include <psycles/sampling/light_distribution.h>

#include <memory>

namespace psycles::luisa_backend::detail {

using RenderKernel = Kernel1D<Buffer<luisa::float4>,
                              Buffer<luisa::float4>,
                              Buffer<luisa::float4>,
                              Buffer<luisa::float4>,
                              Buffer<luisa::uint>,
                              Buffer<luisa::float4>,
                              std::uint32_t,
                              std::uint32_t,
                              Buffer<luisa::float4>,
                              Buffer<float>,
                              RenderKernelParameters>;

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
    bool reflective_caustics{};
    bool refractive_caustics{};
    bool path_trace_enabled{};
    LightTransportCallables light_transport;
    EmissiveTrianglePdfCallable emissive_triangle_pdf;
    LightDistributionSampleCallable light_distribution_sample;
    SurfaceCallables surfaces;
    EnvironmentCallables environment;
    TraceShadowCallable trace_shadow;
};

struct PathKernelInvocation {
    const PathKernelConfig &config;
    const BufferFloat4 &combined;
    const BufferFloat4 &normal;
    const BufferFloat4 &albedo;
    const BufferFloat4 &light_passes;
    const BufferUInt &sample_count;
    const BufferFloat4 &path_trace;
    const UInt &sample_first;
    const UInt &samples;
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
    Float4 emission_sum;
    Float4 environment_sum;
    Float4 glossy_color_sum;
    Float4 transmission_color_sum;
    UInt completed;
    SurfaceQuery surface_query;

    [[nodiscard]] Float3 clamp_contribution(Float3 contribution,
                                            UInt depth) const noexcept;
    [[nodiscard]] Float sample_light_roulette(Float3 unshadowed_contribution,
                                              Float random) const noexcept;
    [[nodiscard]] SurfaceEvaluation
    evaluate_surface(UInt surface_tag,
                     const SurfacePoint &point,
                     Float3 outgoing,
                     const SurfaceQuery &query) const noexcept;
    [[nodiscard]] Float3 surface_emission(UInt surface_tag,
                                          const SurfacePoint &point,
                                          Float3 outgoing) const noexcept;
    [[nodiscard]] SurfaceSample
    sample_surface(UInt surface_tag,
                   const SurfacePoint &point,
                   Float u_lobe,
                   Float2 u_direction,
                   const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfaceClosureTrace
    trace_surface_closure(UInt surface_tag,
                          const SurfacePoint &point,
                          UInt requested_index) const noexcept;
    [[nodiscard]] SurfaceSampleTrace
    trace_sample_surface(UInt surface_tag,
                         const SurfacePoint &point,
                         Float u_lobe,
                         Float2 u_direction,
                         const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfaceAov
    surface_aov(UInt surface_tag, const SurfacePoint &point) const noexcept;
    [[nodiscard]] Float3
    surface_shading_normal(UInt surface_tag,
                           const SurfacePoint &point) const noexcept;
    [[nodiscard]] Float3
    evaluate_environment(Float3 direction,
                         const cycles_path_state::ShaderEvaluationState
                             &shader_state) const noexcept;

    void write_film() noexcept;
};

[[nodiscard]] PathKernelInvocation
begin_path_kernel(const PathKernelConfig &config,
                  const BufferFloat4 &combined,
                  const BufferFloat4 &normal,
                  const BufferFloat4 &albedo,
                  const BufferFloat4 &light_passes,
                  const BufferUInt &sample_count,
                  const BufferFloat4 &path_trace,
                  const UInt &sample_first,
                  const UInt &samples,
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
    UInt ray_source_instance;
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
    Float3 sample_emission;
    Float3 sample_environment;
    Float sample_alpha;
    Bool primary_recorded;
    Float previous_bsdf_pdf;
    Float3 previous_mis_origin_normal;
    Float minimum_bsdf_pdf;
    Bool previous_delta;
    Float continuation_probability;
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
    Bool terminate_after_transparent;
    Bool terminate_on_next_surface;

    [[nodiscard]] Float3 trace_uint32(UInt value) const noexcept;
    void trace_write(UInt slot, Float3 value) const noexcept;
    void trace_write_global(path_trace_schema::GlobalSlot slot,
                            Float3 value) const noexcept;
    void trace_write_event(UInt event,
                           path_trace_schema::EventSlot slot,
                           Float3 value) const noexcept;
    void trace_write_closure(UInt event,
                             std::uint32_t closure,
                             std::uint32_t field,
                             Float3 value) const noexcept;
    void
    accumulate_light_pass(Var<LightPassContributionCall> contribution) noexcept;
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
};

struct SurfaceGeometryContext {
    PathBounceContext &bounce;
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
    Bool triangle_smooth;
    Float3 shadow_shading_normal;
    UInt surface_tag;
    UInt cycles_surface_shader;
    UInt cycles_object_index;
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

class ClosestEventStage {

  public:
    virtual ~ClosestEventStage() noexcept = default;
    [[nodiscard]] virtual PathBounceContext
    emit(PathSampleContext &sample, const UInt &path_step) const noexcept = 0;
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
    emit(SurfaceGeometryContext &surface) const noexcept = 0;
};

class DirectLightingComponent {

  public:
    virtual ~DirectLightingComponent() noexcept = default;
    virtual void emit(DirectLightingContext &context) const noexcept = 0;
};

class SurfaceScatterStage {

  public:
    virtual ~SurfaceScatterStage() noexcept = default;
    virtual void emit(DirectLightingContext &context) const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<ClosestEventStage> make_closest_event_stage();
[[nodiscard]] std::unique_ptr<SurfaceGeometryStage>
make_surface_geometry_stage();
[[nodiscard]] std::unique_ptr<SurfaceShadingStage> make_surface_shading_stage();
[[nodiscard]] std::unique_ptr<DirectLightingComponent>
make_environment_lighting_component();
[[nodiscard]] std::unique_ptr<DirectLightingComponent>
make_emissive_mesh_lighting_component();
[[nodiscard]] std::unique_ptr<DirectLightingComponent>
make_analytic_lighting_component();
[[nodiscard]] std::unique_ptr<SurfaceScatterStage> make_surface_scatter_stage();

class PathKernelPipeline {

  private:
    class Impl;
    std::unique_ptr<Impl> _impl;

  public:
    PathKernelPipeline();
    ~PathKernelPipeline() noexcept;
    PathKernelPipeline(PathKernelPipeline &&) noexcept;
    PathKernelPipeline &operator=(PathKernelPipeline &&) noexcept;
    PathKernelPipeline(const PathKernelPipeline &) = delete;
    PathKernelPipeline &operator=(const PathKernelPipeline &) = delete;

    void emit(PathSampleContext &sample) const noexcept;
};

[[nodiscard]] RenderKernel
build_path_kernel(const PathKernelConfig &config) noexcept;

} // namespace psycles::luisa_backend::detail

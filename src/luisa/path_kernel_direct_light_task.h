#pragma once

#include "path_tracer_geometry.h"

#include <type_traits>

namespace psycles::luisa_backend::detail {

struct PathKernelConfig;

// Minimal state crossing the surface -> shadow-work boundary. Material
// closures and surface temporaries deliberately do not escape this record:
// proposal generation reduces them to an unshadowed contribution and two
// light-pass weights before publication.
struct DirectLightTaskCall {
  luisa::float3 ray_origin{};
  luisa::float3 ray_direction{};
  // Before SHADE_LIGHT_NEE this is the receiving BSDF/path factor. That
  // stage multiplies the deferred light shader and applies light roulette,
  // replacing the field with the complete unshadowed contribution.
  luisa::float3 unshadowed_contribution{};
  luisa::float3 light_shader{};
  // Running shadow-path throughput. INTERSECT_SHADOW never mutates it;
  // SHADE_SHADOW alone applies transparent closure extinction.
  luisa::float3 shadow_transmittance{};
  luisa::float3 diffuse_weight{};
  luisa::float3 glossy_weight{};
  luisa::float2 shadow_hit_barycentric{};
  float ray_minimum{};
  float ray_maximum{};
  float ray_dP{};
  float ray_dD{};
  float shadow_hit_distance{};
  float light_terminate_sample{};
  luisa::uint source_object{};
  luisa::uint source_primitive{};
  luisa::uint light_object{};
  luisa::uint light_primitive{};
  luisa::uint shadow_hit_instance{};
  luisa::uint shadow_hit_primitive{};
  luisa::uint shadow_hit_type{};
  luisa::uint constant_light_shader{};
  luisa::uint shader_flags{};
  luisa::uint pixel{};
  luisa::uint path_depth{};
  luisa::uint path_flags{};
  luisa::uint path_visibility{};
  luisa::uint diffuse_depth{};
  luisa::uint glossy_depth{};
  luisa::uint transparent_depth{};
  luisa::uint transmission_depth{};
};

static_assert(std::is_trivially_copyable_v<DirectLightTaskCall>);
static_assert(sizeof(DirectLightTaskCall) == 224u);

struct DirectLightTaskFilm {
  const BufferFloat4 &combined;
  const BufferFloat4 &light_passes;
  const BufferFloat4 &volume_guiding_raw;
};

// The SHADE_SHADOW transition is a total, disjoint state transition:
//
//   transparent hit -> continue_shadow
//   miss            -> visible
//   opaque hit      -> neither (occluded terminal)
//
// Keeping the two terminal outcomes explicit avoids encoding scheduler
// control in a renderer-specific sentinel or conflating a visible miss with
// an opaque termination.
struct DirectLightShadowStep {
  Bool continue_shadow;
  Bool visible;
};

struct DirectLightTaskEvaluator {
  IntersectShadowCallable intersect_shadow;
  EvaluateShadowSurfaceCallable shade_shadow_surface;
  TraceShadowCallable trace_shadow;
  LightSampleRouletteCallable light_sample_roulette;
  ClampLightContributionCallable clamp_contribution;
  SplitScatteredLightCallable split_scattered_light;
  bool volume_guiding{};

  [[nodiscard]] Var<ShadowTraceResultCall>
  trace(const Var<DirectLightTaskCall> &task,
        const Var<RenderKernelParameters> &parameters) const noexcept;
  [[nodiscard]] Bool
  shade_light_nee(Var<DirectLightTaskCall> &task,
                  const Var<RenderKernelParameters> &parameters) const noexcept;
  void intersect(Var<DirectLightTaskCall> &task) const noexcept;
  [[nodiscard]] DirectLightShadowStep
  shade_shadow(Var<DirectLightTaskCall> &task,
               const Var<RenderKernelParameters> &parameters) const noexcept;
  [[nodiscard]] Float3
  contribution(const Var<DirectLightTaskCall> &task, Float3 transmittance,
               const Var<RenderKernelParameters> &parameters) const noexcept;
  [[nodiscard]] Var<LightPassContributionCall>
  split(const Var<DirectLightTaskCall> &task,
        Float3 contribution) const noexcept;
  void
  emit_atomic(const Var<DirectLightTaskCall> &task,
              const DirectLightTaskFilm &film,
              const Var<RenderKernelParameters> &parameters) const noexcept;
};

[[nodiscard]] DirectLightTaskEvaluator
make_direct_light_task_evaluator(const PathKernelConfig &config) noexcept;

class DirectLightTaskSink {

public:
  virtual ~DirectLightTaskSink() noexcept = default;
  virtual void emit(
      Var<DirectLightTaskCall> task,
      Expr<std::uint32_t> runtime_capacity) const noexcept = 0;
};

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::DirectLightTaskCall, ray_origin,
             ray_direction, unshadowed_contribution, light_shader,
             shadow_transmittance, diffuse_weight, glossy_weight,
             shadow_hit_barycentric, ray_minimum, ray_maximum, ray_dP, ray_dD,
             shadow_hit_distance, light_terminate_sample, source_object,
             source_primitive, light_object, light_primitive,
             shadow_hit_instance, shadow_hit_primitive, shadow_hit_type,
             constant_light_shader, shader_flags, pixel, path_depth, path_flags,
             path_visibility, diffuse_depth, glossy_depth, transparent_depth,
             transmission_depth){};

namespace psycles::luisa_backend::detail {

// A surface vertex produces at most one direct-light task. `valid` is the
// device control predicate; the zero-initialized task makes the value total
// even on the no-proposal edge and therefore keeps XIR definite-assignment
// independent of later control-flow simplification.
struct DirectLightTransportPreparation {
  Var<DirectLightTaskCall> task;
  Bool valid;
};

} // namespace psycles::luisa_backend::detail

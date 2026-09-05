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
  // Before SHADE_LIGHT_NEE this is the receiving BSDF factor in the local
  // light-sample domain. That stage multiplies the deferred light shader,
  // applies roulette in that same domain, then replaces the field with the
  // complete path-space unshadowed contribution.
  luisa::float3 unshadowed_contribution{};
  // Path throughput is deliberately separate from the roulette domain.
  // Cycles evaluates light termination before multiplying by path throughput.
  luisa::float3 nee_path_throughput{};
  luisa::float3 light_shader{};
  // Running shadow-path throughput. INTERSECT_SHADOW never mutates it;
  // SHADE_SHADOW alone applies transparent closure extinction.
  luisa::float3 shadow_transmittance{};
  luisa::float3 diffuse_weight{};
  luisa::float3 glossy_weight{};
  float ray_minimum{};
  float ray_maximum{};
  float ray_dP{};
  float ray_dD{};
  float light_terminate_sample{};
  luisa::uint source_object{};
  luisa::uint source_primitive{};
  luisa::uint light_object{};
  luisa::uint light_primitive{};
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
// This is the invariant shadow-path state. The four-hit traversal batch is
// deliberately not a member: it is live only on the
// INTERSECT_SHADOW -> SHADE_SHADOW edge, while a fused shadow consumer never
// needs it at all. Keeping the types disjoint makes that lifetime true before
// optimization instead of relying on aggregate DCE.
static_assert(sizeof(DirectLightTaskCall) == 208u);

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

// Finalizes one light sample without changing its roulette measure. The local
// unshadowed value determines survival and inverse probability; path
// throughput is applied exactly once, after that decision.
[[nodiscard]] Float3 finalize_direct_light_sample(
    const LightSampleRouletteCallable &light_sample_roulette,
    Float3 local_unshadowed, Float3 path_throughput,
    Float light_terminate_sample, Float inverse_threshold) noexcept;

struct DirectLightTaskEvaluator {
  std::shared_ptr<const ShadowIntersectionComponent> intersect_shadow;
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
  [[nodiscard]] Var<ShadowIntersectionBatchCall>
  intersect(const Var<DirectLightTaskCall> &task,
            const Var<RenderKernelParameters> &parameters) const noexcept;
  [[nodiscard]] DirectLightShadowStep
  shade_shadow(Var<DirectLightTaskCall> &task,
               const Var<ShadowIntersectionBatchCall> &shadow_batch,
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
             ray_direction, unshadowed_contribution, nee_path_throughput,
             light_shader, shadow_transmittance, diffuse_weight, glossy_weight,
             ray_minimum, ray_maximum, ray_dP, ray_dD,
             light_terminate_sample, source_object, source_primitive,
             light_object, light_primitive, constant_light_shader, shader_flags,
             pixel, path_depth, path_flags, path_visibility, diffuse_depth,
             glossy_depth, transparent_depth, transmission_depth){};

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

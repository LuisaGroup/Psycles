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
  luisa::float3 unshadowed_contribution{};
  luisa::float3 diffuse_weight{};
  luisa::float3 glossy_weight{};
  float ray_minimum{};
  float ray_maximum{};
  float ray_dP{};
  float ray_dD{};
  luisa::uint source_object{};
  luisa::uint source_primitive{};
  luisa::uint light_object{};
  luisa::uint light_primitive{};
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
static_assert(sizeof(DirectLightTaskCall) == 144u);

struct DirectLightTaskFilm {
  const BufferFloat4 &combined;
  const BufferFloat4 &light_passes;
  const BufferFloat4 &volume_guiding_raw;
};

struct DirectLightTaskEvaluator {
  TraceShadowCallable trace_shadow;
  ClampLightContributionCallable clamp_contribution;
  SplitScatteredLightCallable split_scattered_light;
  bool volume_guiding{};

  [[nodiscard]] Var<ShadowTraceResultCall>
  trace(const Var<DirectLightTaskCall> &task,
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
  virtual void emit(Var<DirectLightTaskCall> task) const noexcept = 0;
};

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::DirectLightTaskCall, ray_origin,
             ray_direction, unshadowed_contribution, diffuse_weight,
             glossy_weight, ray_minimum, ray_maximum, ray_dP, ray_dD,
             source_object, source_primitive, light_object, light_primitive,
             pixel, path_depth, path_flags, path_visibility, diffuse_depth,
             glossy_depth, transparent_depth, transmission_depth){};

#pragma once

#include "path_tracer_geometry.h"
#include "path_kernel_direct_light_task.h"

#include <psycles/luisa/cycles_spectrum.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace psycles::luisa_backend::detail {

struct PathKernelConfig;
struct PathSampleContext;

// Self-contained work item handed off from an eligible receiver NEE bounce
// to the refractive MNEE manifold solver (either inline in megakernel or via
// the detached CausticsTaskQueue in wavefront-staged).
struct CausticTaskCall {
  luisa::float4 path_throughput{};
  luisa::float4 light_shader{};
  luisa::float3 receiver_p{};
  float ray_time{};
  luisa::float3 receiver_wi{};
  float ray_dP{};
  luisa::float3 receiver_ng{};
  float ray_dD{};
  luisa::float3 light_p{};
  float light_selection_pdf{};
  luisa::float3 light_dir{};
  float light_area_pdf{};
  luisa::float3 light_normal{};
  float light_distance{};
  luisa::float3 path_diffuse_weight{};
  float pad_d{};
  luisa::float3 path_glossy_weight{};
  float pad_g{};
  luisa::float2 receiver_bary{};
  luisa::float2 light_uv{};
  luisa::uint receiver_inst{};
  luisa::uint receiver_prim{};
  luisa::uint receiver_cycles_object{};
  luisa::uint receiver_cycles_shader{};
  luisa::uint light_index{};
  luisa::uint light_cycles_object{};
  luisa::uint light_shader_flags{};
  luisa::uint light_constant_shader{};
  luisa::uint pixel{};
  luisa::uint path_depth{};
  luisa::uint path_flags{};
  luisa::uint path_visibility{};
  luisa::uint diffuse_depth{};
  luisa::uint glossy_depth{};
  luisa::uint transmission_depth{};
  luisa::uint transparent_depth{};
  luisa::uint portal_depth{};
  luisa::uint sample_index{};
  luisa::uint rng_hash{};
  luisa::uint rng_offset{};
  float u_lambda{};
  float pad0{};
  luisa::uint pad1{};
  luisa::uint pad2{};
};

static_assert(std::is_trivially_copyable_v<CausticTaskCall>);
static_assert(sizeof(CausticTaskCall) % 16u == 0u);

class CausticsTaskSink {
public:
  virtual ~CausticsTaskSink() noexcept = default;
  virtual void emit(
      Var<CausticTaskCall> task,
      Expr<std::uint32_t> runtime_capacity) const noexcept = 0;
};

struct CausticsTaskEvaluator {
  std::shared_ptr<LuisaSceneData> scene;
  ClampLightContributionCallable clamp_contribution;
  LightComponentRatioCallable light_component_ratio;
  std::shared_ptr<const DirectLightEmissionComponent> light_emission;
  std::shared_ptr<const cycles_spectrum::SpectralTablesGpu> spectral_tables;
  CameraProjection camera_projection{CameraProjection::perspective};
  bool spectral_rendering{false};
  bool volume_guiding{false};
  bool reflective_caustics{true};
  bool refractive_caustics{true};

  [[nodiscard]] Float3 evaluate(
      const Var<CausticTaskCall> &task,
      const Var<RenderKernelParameters> &parameters,
      Bool &valid,
      Float3 *out_diffuse_weight = nullptr,
      Float3 *out_glossy_weight = nullptr) const noexcept;

  void evaluate_and_accumulate_atomic(
      const Var<CausticTaskCall> &task,
      const DirectLightTaskFilm &film,
      const Var<RenderKernelParameters> &parameters) const noexcept;

  void evaluate_and_accumulate_sample(
      PathSampleContext &sample,
      const Var<CausticTaskCall> &task,
      const Var<RenderKernelParameters> &parameters) const noexcept;
};

[[nodiscard]] CausticsTaskEvaluator
make_caustics_task_evaluator(const PathKernelConfig &config) noexcept;

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::CausticTaskCall,
             path_throughput, light_shader, receiver_p, ray_time,
             receiver_wi, ray_dP, receiver_ng, ray_dD, light_p,
             light_selection_pdf, light_dir, light_area_pdf, light_normal,
             light_distance, path_diffuse_weight, pad_d, path_glossy_weight,
             pad_g, receiver_bary, light_uv, receiver_inst,
             receiver_prim, receiver_cycles_object, receiver_cycles_shader,
             light_index, light_cycles_object, light_shader_flags,
             light_constant_shader, pixel, path_depth, path_flags,
             path_visibility, diffuse_depth, glossy_depth, transmission_depth,
             transparent_depth, portal_depth, sample_index, rng_hash,
             rng_offset, u_lambda, pad0, pad1, pad2){};

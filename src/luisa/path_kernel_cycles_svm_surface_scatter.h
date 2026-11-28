#pragma once

#include "path_kernel_builder.h"

namespace psycles::luisa_backend::detail {

struct PreparedCyclesSvmSurfaceScatter {
  Float3 wo;
  cycles_spectrum::Spectrum throughput_weight;
  Float3 diffuse_ratio;
  Float3 glossy_ratio;
  Float pdf;
  Float average_roughness_squared;
  Float3 shading_normal;
  Float dP;
  Float ray_length;
  UInt label;
  UInt sd_flag;
  SurfaceSample bssrdf_sample;
  Bool subsurface;
  Bool valid;
  std::optional<Bool> spectral_terminated_secondary{};
  std::optional<Bool> sms_caster_eligible{};
  std::optional<Float3> xyz3_next_pdf_ratio{};

  [[nodiscard]] static PreparedCyclesSvmSurfaceScatter
  empty(contract::SpectralMode spectral_mode = contract::SpectralMode::none,
        bool sms = false, bool xyz3_rough_dispersion = false) noexcept;
  [[nodiscard]] Bool needs_triangle_ray_origin() const noexcept;
  void accept(const PreparedCyclesSvmSurfaceScatter &other,
              bool include_bssrdf) noexcept;
};

// Samples the continuation closure from ShaderData/ClosurePool without mutating
// PathSampleContext so ClosurePool can die before NEE ray-offset / queue stores
// and continuation ray-offset geometry loads.
[[nodiscard]] PreparedCyclesSvmSurfaceScatter
sample_cycles_svm_surface_continuation(
    DirectLightingContext &context,
    const CyclesSvmSurfaceState &native,
    bool may_have_bssrdf) noexcept;

[[nodiscard]] SurfaceScatterStage::Result
commit_cycles_svm_surface_continuation(
    DirectLightingContext &context,
    const PreparedCyclesSvmSurfaceScatter &prepared,
    bool may_have_bssrdf) noexcept;

// Direct consumer of the one retained native ShaderData/closure population.
// Native labels stay native through continuation; SurfaceSample is populated
// only for the existing BSSRDF entry-parameter handoff.
[[nodiscard]] SurfaceScatterStage::Result emit_cycles_svm_surface_scatter(
    DirectLightingContext &context, const CyclesSvmSurfaceState &native) noexcept;

} // namespace psycles::luisa_backend::detail

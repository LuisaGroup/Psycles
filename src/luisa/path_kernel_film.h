#pragma once

#include "path_kernel_builder.h"

namespace psycles::luisa_backend::detail {

void atomic_accumulate_float4(const BufferFloat4 &buffer, const UInt &index,
                              const Float4 &value) noexcept;

void atomic_accumulate_float3(const BufferFloat4 &buffer, const UInt &index,
                              const Float3 &value) noexcept;

void atomic_accumulate_light_pass(const BufferFloat4 &light_passes,
                                  const UInt &light_pass_base,
                                  LightPassBuffer pass,
                                  const Float3 &value) noexcept;

void atomic_accumulate_light_passes(
    const BufferFloat4 &light_passes, const UInt &light_pass_base,
    const Var<LightPassContributionCall> &contribution) noexcept;

// Shared atomic Combined/volume-guiding sink for per-sample main paths and
// detached shadow work. All classification inputs are explicit, so moving an
// event between execution queues cannot silently change film semantics.
void atomic_accumulate_radiance(
    const BufferFloat4 &combined, const BufferFloat4 &volume_guiding_raw,
    const UInt &pixel, const UInt &volume_guiding_raw_base, bool volume_guiding,
    const UInt &path_flags, const UInt &path_visibility, const UInt &path_depth,
    const Float3 &contribution,
    Bool primary_volume_scatter_override = false) noexcept;

} // namespace psycles::luisa_backend::detail

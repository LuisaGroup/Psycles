#include "path_kernel_film.h"

namespace psycles::luisa_backend::detail {

void atomic_accumulate_float4(const BufferFloat4 &buffer, const UInt &index,
                              const Float4 &value) noexcept {
  auto destination = buffer.atomic(index);
  destination.x.fetch_add(value.x);
  destination.y.fetch_add(value.y);
  destination.z.fetch_add(value.z);
  destination.w.fetch_add(value.w);
}

void atomic_accumulate_float3(const BufferFloat4 &buffer, const UInt &index,
                              const Float3 &value) noexcept {
  auto destination = buffer.atomic(index);
  destination.x.fetch_add(value.x);
  destination.y.fetch_add(value.y);
  destination.z.fetch_add(value.z);
}

void atomic_accumulate_light_pass(const BufferFloat4 &light_passes,
                                  const UInt &light_pass_base,
                                  LightPassBuffer pass,
                                  const Float3 &value) noexcept {
  atomic_accumulate_float3(light_passes,
                           light_pass_base + light_pass_index(pass), value);
}

void atomic_accumulate_light_passes(
    const BufferFloat4 &light_passes, const UInt &light_pass_base,
    const Var<LightPassContributionCall> &contribution) noexcept {
  atomic_accumulate_light_pass(light_passes, light_pass_base,
                               LightPassBuffer::diffuse_direct,
                               contribution.diffuse_direct);
  atomic_accumulate_light_pass(light_passes, light_pass_base,
                               LightPassBuffer::diffuse_indirect,
                               contribution.diffuse_indirect);
  atomic_accumulate_light_pass(light_passes, light_pass_base,
                               LightPassBuffer::glossy_direct,
                               contribution.glossy_direct);
  atomic_accumulate_light_pass(light_passes, light_pass_base,
                               LightPassBuffer::glossy_indirect,
                               contribution.glossy_indirect);
  atomic_accumulate_light_pass(light_passes, light_pass_base,
                               LightPassBuffer::transmission_direct,
                               contribution.transmission_direct);
  atomic_accumulate_light_pass(light_passes, light_pass_base,
                               LightPassBuffer::transmission_indirect,
                               contribution.transmission_indirect);
}

void atomic_accumulate_radiance(
    const BufferFloat4 &combined, const BufferFloat4 &volume_guiding_raw,
    const UInt &pixel, const UInt &volume_guiding_raw_base, bool volume_guiding,
    const UInt &path_flags, const UInt &path_visibility, const UInt &path_depth,
    const Float3 &contribution, Bool primary_volume_scatter_override) noexcept {
  atomic_accumulate_float3(combined, pixel, contribution);
  if (!volume_guiding) {
    return;
  }

  const auto primary_volume_direct =
      primary_volume_scatter_override & (path_depth == 0u);
  const auto primary_transmit =
      ((path_flags & cycles_path_state::flag_volume_primary_transmit) != 0u) &
      !primary_volume_direct;
  const auto volume_scatter =
      !primary_transmit &
      (primary_volume_direct |
       ((path_visibility & cycles_path_state::visibility_volume_scatter) !=
        0u));
  $if(primary_transmit) {
    atomic_accumulate_float3(volume_guiding_raw,
                             volume_guiding_raw_base +
                                 volume_guiding::raw_transmit_slot,
                             contribution);
  };
  $if(volume_scatter) {
    atomic_accumulate_float3(volume_guiding_raw,
                             volume_guiding_raw_base +
                                 volume_guiding::raw_scatter_slot,
                             contribution);
  };
}

namespace {

// Enumerate the ordered local contribution of one serial path sample. The
// per-sample specialization writes each already-clamped contribution directly
// when it is produced, avoiding a path-lifetime film staging record.
template <typename Emit>
void emit_path_sample_film_contributions(PathSampleContext &sample,
                                         Emit &&emit) noexcept {
  auto &invocation = sample.invocation;
  emit(invocation.combined_sum, invocation.combined, invocation.pixel,
       make_float4(sample.radiance, sample.sample_transparency));
  emit(invocation.normal_sum, invocation.normal, invocation.pixel,
       make_float4(sample.sample_normal, 1.0f));
  emit(invocation.albedo_sum, invocation.albedo, invocation.pixel,
       make_float4(sample.sample_albedo, 1.0f));

  const auto emit_light_pass = [&](auto &sum, LightPassBuffer pass,
                                   const Float3 &value) noexcept {
    emit(sum, invocation.light_passes,
         invocation.light_pass_base + light_pass_index(pass),
         make_float4(value, 1.0f));
  };
  emit_light_pass(invocation.diffuse_direct_sum,
                  LightPassBuffer::diffuse_direct,
                  sample.sample_diffuse_direct);
  emit_light_pass(invocation.diffuse_indirect_sum,
                  LightPassBuffer::diffuse_indirect,
                  sample.sample_diffuse_indirect);
  emit_light_pass(invocation.glossy_direct_sum, LightPassBuffer::glossy_direct,
                  sample.sample_glossy_direct);
  emit_light_pass(invocation.glossy_indirect_sum,
                  LightPassBuffer::glossy_indirect,
                  sample.sample_glossy_indirect);
  emit_light_pass(invocation.transmission_direct_sum,
                  LightPassBuffer::transmission_direct,
                  sample.sample_transmission_direct);
  emit_light_pass(invocation.transmission_indirect_sum,
                  LightPassBuffer::transmission_indirect,
                  sample.sample_transmission_indirect);
  emit_light_pass(invocation.volume_direct_sum, LightPassBuffer::volume_direct,
                  sample.sample_volume_direct);
  emit_light_pass(invocation.volume_indirect_sum,
                  LightPassBuffer::volume_indirect,
                  sample.sample_volume_indirect);
  emit_light_pass(invocation.emission_sum, LightPassBuffer::emission,
                  sample.sample_emission);
  emit_light_pass(invocation.environment_sum, LightPassBuffer::environment,
                  sample.sample_environment);
  emit_light_pass(invocation.glossy_color_sum, LightPassBuffer::glossy_color,
                  sample.sample_glossy_color);
  emit_light_pass(invocation.transmission_color_sum,
                  LightPassBuffer::transmission_color,
                  sample.sample_transmission_color);

  if (invocation.config.volume_state) {
    const auto emit_volume_guiding = [&](auto &sum, std::uint32_t slot,
                                         const Float4 &value) noexcept {
      emit(sum, invocation.volume_guiding_raw,
           invocation.volume_guiding_raw_base + slot, value);
    };
    emit_volume_guiding(invocation.volume_guiding_scatter_sum,
                        volume_guiding::raw_scatter_slot,
                        make_float4(sample.volume_guiding_scatter, 0.0f));
    emit_volume_guiding(invocation.volume_guiding_transmit_sum,
                        volume_guiding::raw_transmit_slot,
                        make_float4(sample.volume_guiding_transmit, 0.0f));
    const auto primary_volume_transmit =
        (sample.path_flags & cycles_path_state::flag_volume_primary_transmit) !=
        0u;
    emit_volume_guiding(
        invocation.volume_guiding_optical_depth_sum,
        volume_guiding::optical_depth_slot,
        select(make_float4(0.0f),
               make_float4(sample.optical_depth, 1.0f, 0.0f, 0.0f),
               primary_volume_transmit));
  }
}

} // namespace

void PathSampleContext::accumulate_light_pass(LightPassBuffer pass,
                                              Float3 contribution) noexcept {
  auto &invocation = this->invocation;
  if (invocation.film_accumulation == PathFilmAccumulation::atomic) {
    atomic_accumulate_light_pass(invocation.light_passes,
                                 invocation.light_pass_base, pass,
                                 contribution);
    return;
  }
  switch (pass) {
  case LightPassBuffer::diffuse_direct:
    sample_diffuse_direct += contribution;
    break;
  case LightPassBuffer::diffuse_indirect:
    sample_diffuse_indirect += contribution;
    break;
  case LightPassBuffer::glossy_direct:
    sample_glossy_direct += contribution;
    break;
  case LightPassBuffer::glossy_indirect:
    sample_glossy_indirect += contribution;
    break;
  case LightPassBuffer::transmission_direct:
    sample_transmission_direct += contribution;
    break;
  case LightPassBuffer::transmission_indirect:
    sample_transmission_indirect += contribution;
    break;
  case LightPassBuffer::volume_direct:
    sample_volume_direct += contribution;
    break;
  case LightPassBuffer::volume_indirect:
    sample_volume_indirect += contribution;
    break;
  case LightPassBuffer::emission:
    sample_emission += contribution;
    break;
  case LightPassBuffer::environment:
    sample_environment += contribution;
    break;
  case LightPassBuffer::glossy_color:
    sample_glossy_color += contribution;
    break;
  case LightPassBuffer::transmission_color:
    sample_transmission_color += contribution;
    break;
  }
}

void PathSampleContext::accumulate_light_pass(
    Var<LightPassContributionCall> contribution) noexcept {
  accumulate_light_pass(LightPassBuffer::diffuse_direct,
                        contribution.diffuse_direct);
  accumulate_light_pass(LightPassBuffer::diffuse_indirect,
                        contribution.diffuse_indirect);
  accumulate_light_pass(LightPassBuffer::glossy_direct,
                        contribution.glossy_direct);
  accumulate_light_pass(LightPassBuffer::glossy_indirect,
                        contribution.glossy_indirect);
  accumulate_light_pass(LightPassBuffer::transmission_direct,
                        contribution.transmission_direct);
  accumulate_light_pass(LightPassBuffer::transmission_indirect,
                        contribution.transmission_indirect);
}

void PathSampleContext::accumulate_normal_pass(Float3 contribution) noexcept {
  auto &invocation = this->invocation;
  if (invocation.film_accumulation == PathFilmAccumulation::atomic) {
    atomic_accumulate_float3(invocation.normal, invocation.pixel, contribution);
    return;
  }
  sample_normal += contribution;
}

void PathSampleContext::accumulate_albedo_pass(Float3 contribution) noexcept {
  auto &invocation = this->invocation;
  if (invocation.film_accumulation == PathFilmAccumulation::atomic) {
    atomic_accumulate_float3(invocation.albedo, invocation.pixel, contribution);
    return;
  }
  sample_albedo += contribution;
}

void PathSampleContext::accumulate_scattered_light(
    Float3 contribution) noexcept {
  const auto surface_pass =
      (path_flags & cycles_path_state::flag_surface_pass) != 0u;
  const auto volume_pass =
      (path_flags & cycles_path_state::flag_volume_pass) != 0u;
  $if(surface_pass) {
    accumulate_light_pass(
        invocation.config.light_transport.split_scattered_light(
            contribution, path_diffuse_weight, path_glossy_weight,
            path_depth == 1u));
  };
  $if(volume_pass) {
    accumulate_light_pass(
        LightPassBuffer::volume_direct,
        select(make_float3(0.0f), contribution, path_depth == 1u));
    accumulate_light_pass(
        LightPassBuffer::volume_indirect,
        select(contribution, make_float3(0.0f), path_depth == 1u));
  };
}

void PathSampleContext::accumulate_radiance(
    Float3 contribution, Bool primary_volume_scatter_override) noexcept {
  auto &invocation = this->invocation;
  const auto atomic_film =
      invocation.film_accumulation == PathFilmAccumulation::atomic;
  if (atomic_film) {
    atomic_accumulate_radiance(invocation.combined,
                               invocation.volume_guiding_raw, invocation.pixel,
                               invocation.volume_guiding_raw_base,
                               invocation.config.volume_state != nullptr,
                               path_flags, cycles_path_visibility, path_depth,
                               contribution, primary_volume_scatter_override);
    return;
  } else {
    radiance += contribution;
  }
  if (!invocation.config.volume_state) {
    return;
  }

  // This is the exact priority used by
  // film_write_volume_scattering_guiding_pass(): primary transmission wins
  // over volume-scatter visibility. A primary volume NEE shadow path is the
  // one exception; Cycles clears PRIMARY_TRANSMIT and injects VOLUME_SCATTER
  // into the copied shadow state before writing Combined.
  const auto primary_volume_direct =
      primary_volume_scatter_override & (path_depth == 0u);
  const auto primary_transmit =
      ((path_flags & cycles_path_state::flag_volume_primary_transmit) != 0u) &
      !primary_volume_direct;
  const auto volume_scatter =
      !primary_transmit &
      (primary_volume_direct |
       ((cycles_path_visibility &
         cycles_path_state::visibility_volume_scatter) != 0u));
  volume_guiding_transmit +=
      select(make_float3(0.0f), contribution, primary_transmit);
  volume_guiding_scatter +=
      select(make_float3(0.0f), contribution, volume_scatter);
}

void PathSampleContext::accumulate_transparency(Float transparency) noexcept {
  auto &invocation = this->invocation;
  const auto atomic_film =
      invocation.film_accumulation == PathFilmAccumulation::atomic;
  if (atomic_film) {
    invocation.combined.atomic(invocation.pixel).w.fetch_add(transparency);
  } else {
    sample_transparency += transparency;
  }
  if (!invocation.config.volume_state) {
    return;
  }
  const auto primary_transmit =
      (path_flags & cycles_path_state::flag_volume_primary_transmit) != 0u;
  if (atomic_film) {
    $if(primary_transmit) {
      atomic_accumulate_float3(invocation.volume_guiding_raw,
                               invocation.volume_guiding_raw_base +
                                   volume_guiding::raw_transmit_slot,
                               make_float3(transparency));
    };
  } else {
    volume_guiding_transmit +=
        select(make_float3(0.0f), make_float3(transparency), primary_transmit);
  }
}

void accumulate_path_sample(PathSampleContext &sample) noexcept {
  auto &invocation = sample.invocation;

  if (invocation.film_accumulation == PathFilmAccumulation::atomic) {
    // Optical depth is path-terminal state in Cycles, not an additive
    // per-event film contribution. Keep this one scalar live and write it
    // exactly once when the path terminates.
    if (invocation.config.volume_state) {
      const auto primary_volume_transmit =
          (sample.path_flags &
           cycles_path_state::flag_volume_primary_transmit) != 0u;
      $if(primary_volume_transmit) {
        atomic_accumulate_float4(
            invocation.volume_guiding_raw,
            invocation.volume_guiding_raw_base +
                volume_guiding::optical_depth_slot,
            make_float4(sample.optical_depth, 1.0f, 0.0f, 0.0f));
      };
    }
    invocation.sample_count.atomic(invocation.pixel).fetch_add(1u);
    return;
  }

  auto &radiance = sample.radiance;
  radiance = select(radiance, make_float3(0.0f),
                    any(luisa::compute::dsl::isnan(radiance)));

  emit_path_sample_film_contributions(
      sample, [](auto &sum, const BufferFloat4 &, const UInt &,
                 const Float4 &value) noexcept { sum += value; });
  invocation.completed += 1u;
}

void PathKernelInvocation::write_film() noexcept {
  // Atomic per-sample film contributions are reduced at the end of the
  // sample itself, after its final suspension. There is no local film to
  // flush in this specialization.
  if (film_accumulation == PathFilmAccumulation::atomic) {
    return;
  }
  const auto write_float4 = [](const BufferFloat4 &buffer, const UInt &index,
                               const Float4 &value) noexcept {
    buffer.write(index, value);
  };

  write_float4(combined, pixel, combined_sum);
  write_float4(normal, pixel, normal_sum);
  write_float4(albedo, pixel, albedo_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::diffuse_direct),
               diffuse_direct_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::diffuse_indirect),
               diffuse_indirect_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::glossy_direct),
               glossy_direct_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::glossy_indirect),
               glossy_indirect_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::transmission_direct),
               transmission_direct_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::transmission_indirect),
               transmission_indirect_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::volume_direct),
               volume_direct_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::volume_indirect),
               volume_indirect_sum);
  write_float4(light_passes,
               light_pass_base + light_pass_index(LightPassBuffer::emission),
               emission_sum);
  write_float4(light_passes,
               light_pass_base + light_pass_index(LightPassBuffer::environment),
               environment_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::glossy_color),
               glossy_color_sum);
  write_float4(light_passes,
               light_pass_base +
                   light_pass_index(LightPassBuffer::transmission_color),
               transmission_color_sum);

  if (config.volume_state) {
    write_float4(volume_guiding_raw,
                 volume_guiding_raw_base + volume_guiding::raw_scatter_slot,
                 volume_guiding_scatter_sum);
    write_float4(volume_guiding_raw,
                 volume_guiding_raw_base + volume_guiding::raw_transmit_slot,
                 volume_guiding_transmit_sum);
    write_float4(volume_guiding_raw,
                 volume_guiding_raw_base + volume_guiding::optical_depth_slot,
                 volume_guiding_optical_depth_sum);
  }
  sample_count.write(pixel, completed);
}

} // namespace psycles::luisa_backend::detail

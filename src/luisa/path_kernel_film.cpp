#include "path_kernel_builder.h"

namespace psycles::luisa_backend::detail {
namespace {

void atomic_add_float4(const BufferFloat4 &buffer,
                       const UInt &index,
                       const Float4 &value) noexcept {
    auto destination = buffer.atomic(index);
    destination.x.fetch_add(value.x);
    destination.y.fetch_add(value.y);
    destination.z.fetch_add(value.z);
    destination.w.fetch_add(value.w);
}

// Enumerate the complete linear contribution of one path sample. The first
// argument passed to `emit` is the serial per-pixel accumulator corresponding
// to the buffer slot. Keeping the mapping in one place prevents the serial and
// per-sample paths from silently disagreeing about film-pass routing.
template<typename Emit>
void emit_path_sample_film_contributions(
    PathSampleContext &sample,
    Emit &&emit) noexcept {
    auto &invocation = sample.invocation;
    emit(invocation.combined_sum,
         invocation.combined,
         invocation.pixel,
         make_float4(
             sample.radiance,
             sample.sample_transparency));
    emit(invocation.normal_sum,
         invocation.normal,
         invocation.pixel,
         make_float4(sample.sample_normal, 1.0f));
    emit(invocation.albedo_sum,
         invocation.albedo,
         invocation.pixel,
         make_float4(sample.sample_albedo, 1.0f));

    const auto emit_light_pass =
        [&](auto &sum,
            LightPassBuffer pass,
            const Float3 &value) noexcept {
            emit(sum,
                 invocation.light_passes,
                 invocation.light_pass_base +
                     light_pass_index(pass),
                 make_float4(value, 1.0f));
        };
    emit_light_pass(
        invocation.diffuse_direct_sum,
        LightPassBuffer::diffuse_direct,
        sample.sample_diffuse_direct);
    emit_light_pass(
        invocation.diffuse_indirect_sum,
        LightPassBuffer::diffuse_indirect,
        sample.sample_diffuse_indirect);
    emit_light_pass(
        invocation.glossy_direct_sum,
        LightPassBuffer::glossy_direct,
        sample.sample_glossy_direct);
    emit_light_pass(
        invocation.glossy_indirect_sum,
        LightPassBuffer::glossy_indirect,
        sample.sample_glossy_indirect);
    emit_light_pass(
        invocation.transmission_direct_sum,
        LightPassBuffer::transmission_direct,
        sample.sample_transmission_direct);
    emit_light_pass(
        invocation.transmission_indirect_sum,
        LightPassBuffer::transmission_indirect,
        sample.sample_transmission_indirect);
    emit_light_pass(
        invocation.volume_direct_sum,
        LightPassBuffer::volume_direct,
        sample.sample_volume_direct);
    emit_light_pass(
        invocation.volume_indirect_sum,
        LightPassBuffer::volume_indirect,
        sample.sample_volume_indirect);
    emit_light_pass(
        invocation.emission_sum,
        LightPassBuffer::emission,
        sample.sample_emission);
    emit_light_pass(
        invocation.environment_sum,
        LightPassBuffer::environment,
        sample.sample_environment);
    emit_light_pass(
        invocation.glossy_color_sum,
        LightPassBuffer::glossy_color,
        sample.sample_glossy_color);
    emit_light_pass(
        invocation.transmission_color_sum,
        LightPassBuffer::transmission_color,
        sample.sample_transmission_color);

    if (invocation.config.volume_state) {
        const auto emit_volume_guiding =
            [&](auto &sum,
                std::uint32_t slot,
                const Float4 &value) noexcept {
                emit(sum,
                     invocation.volume_guiding_raw,
                     invocation.volume_guiding_raw_base + slot,
                     value);
            };
        emit_volume_guiding(
            invocation.volume_guiding_scatter_sum,
            volume_guiding::raw_scatter_slot,
            make_float4(
                sample.volume_guiding_scatter,
                0.0f));
        emit_volume_guiding(
            invocation.volume_guiding_transmit_sum,
            volume_guiding::raw_transmit_slot,
            make_float4(
                sample.volume_guiding_transmit,
                0.0f));
        const auto primary_volume_transmit =
            (sample.path_flags &
             cycles_path_state::flag_volume_primary_transmit) != 0u;
        emit_volume_guiding(
            invocation.volume_guiding_optical_depth_sum,
            volume_guiding::optical_depth_slot,
            select(
                make_float4(0.0f),
                make_float4(
                    sample.optical_depth,
                    1.0f,
                    0.0f,
                    0.0f),
                primary_volume_transmit));
    }
}

}// namespace

void accumulate_path_sample(PathSampleContext &sample) noexcept {
    auto &invocation = sample.invocation;
    auto &radiance = sample.radiance;
    radiance = select(
        radiance,
        make_float3(0.0f),
        any(luisa::compute::dsl::isnan(radiance)));

    if (invocation.film_accumulation ==
        PathFilmAccumulation::atomic) {
        // A per-sample dispatch instance contributes exactly one sample.
        // Applying the atomic film reduction here is equivalent to first
        // adding the contribution to a zero-valued local film and flushing
        // that local film immediately afterwards. The direct form avoids
        // keeping those identity-initialized pass accumulators live across
        // every coroutine suspension.
        emit_path_sample_film_contributions(
            sample,
            [](auto &,
               const BufferFloat4 &buffer,
               const UInt &index,
               const Float4 &value) noexcept {
                atomic_add_float4(
                    buffer, index, value);
            });
        invocation.sample_count
            .atomic(invocation.pixel)
            .fetch_add(1u);
        return;
    }

    emit_path_sample_film_contributions(
        sample,
        [](auto &sum,
           const BufferFloat4 &,
           const UInt &,
           const Float4 &value) noexcept {
            sum += value;
        });
    invocation.completed += 1u;
}

void PathKernelInvocation::write_film() noexcept {
    // Atomic per-sample film contributions are reduced at the end of the
    // sample itself, after its final suspension. There is no local film to
    // flush in this specialization.
    if (film_accumulation == PathFilmAccumulation::atomic) {
        return;
    }
    const auto write_float4 = [](
                                  const BufferFloat4 &buffer,
                                  const UInt &index,
                                  const Float4 &value) noexcept {
        buffer.write(index, value);
    };

    write_float4(combined, pixel, combined_sum);
    write_float4(normal, pixel, normal_sum);
    write_float4(albedo, pixel, albedo_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::diffuse_direct),
        diffuse_direct_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::diffuse_indirect),
        diffuse_indirect_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::glossy_direct),
        glossy_direct_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::glossy_indirect),
        glossy_indirect_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::transmission_direct),
        transmission_direct_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::transmission_indirect),
        transmission_indirect_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::volume_direct),
        volume_direct_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::volume_indirect),
        volume_indirect_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::emission),
        emission_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::environment),
        environment_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::glossy_color),
        glossy_color_sum);
    write_float4(
        light_passes,
        light_pass_base +
            light_pass_index(LightPassBuffer::transmission_color),
        transmission_color_sum);

    if (config.volume_state) {
        write_float4(
            volume_guiding_raw,
            volume_guiding_raw_base +
                volume_guiding::raw_scatter_slot,
            volume_guiding_scatter_sum);
        write_float4(
            volume_guiding_raw,
            volume_guiding_raw_base +
                volume_guiding::raw_transmit_slot,
            volume_guiding_transmit_sum);
        write_float4(
            volume_guiding_raw,
            volume_guiding_raw_base +
                volume_guiding::optical_depth_slot,
            volume_guiding_optical_depth_sum);
    }
    sample_count.write(pixel, completed);
}

}// namespace psycles::luisa_backend::detail

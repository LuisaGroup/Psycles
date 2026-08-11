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

}// namespace

void accumulate_path_sample(PathSampleContext &sample) noexcept {
    auto &invocation = sample.invocation;
    auto &radiance = sample.radiance;
    radiance = select(
        radiance,
        make_float3(0.0f),
        any(luisa::compute::dsl::isnan(radiance)));

    invocation.combined_sum += make_float4(
        radiance, sample.sample_transparency);
    invocation.normal_sum += make_float4(
        sample.sample_normal, 1.0f);
    invocation.albedo_sum += make_float4(
        sample.sample_albedo, 1.0f);
    invocation.glossy_color_sum += make_float4(
        sample.sample_glossy_color, 1.0f);
    invocation.transmission_color_sum += make_float4(
        sample.sample_transmission_color, 1.0f);
    invocation.diffuse_direct_sum += make_float4(
        sample.sample_diffuse_direct, 1.0f);
    invocation.diffuse_indirect_sum += make_float4(
        sample.sample_diffuse_indirect, 1.0f);
    invocation.glossy_direct_sum += make_float4(
        sample.sample_glossy_direct, 1.0f);
    invocation.glossy_indirect_sum += make_float4(
        sample.sample_glossy_indirect, 1.0f);
    invocation.transmission_direct_sum += make_float4(
        sample.sample_transmission_direct, 1.0f);
    invocation.transmission_indirect_sum += make_float4(
        sample.sample_transmission_indirect, 1.0f);
    invocation.volume_direct_sum += make_float4(
        sample.sample_volume_direct, 1.0f);
    invocation.volume_indirect_sum += make_float4(
        sample.sample_volume_indirect, 1.0f);
    invocation.emission_sum += make_float4(
        sample.sample_emission, 1.0f);
    invocation.environment_sum += make_float4(
        sample.sample_environment, 1.0f);

    if (invocation.config.volume_state) {
        invocation.volume_guiding_scatter_sum += make_float4(
            sample.volume_guiding_scatter, 0.0f);
        invocation.volume_guiding_transmit_sum += make_float4(
            sample.volume_guiding_transmit, 0.0f);
        const auto primary_volume_transmit =
            (sample.path_flags &
             cycles_path_state::flag_volume_primary_transmit) != 0u;
        invocation.volume_guiding_optical_depth_sum += select(
            make_float4(0.0f),
            make_float4(
                sample.optical_depth,
                1.0f,
                0.0f,
                0.0f),
            primary_volume_transmit);
    }
    invocation.completed += 1u;
}

void PathKernelInvocation::write_film() noexcept {
    const auto write_float4 = [this](
                                  const BufferFloat4 &buffer,
                                  const UInt &index,
                                  const Float4 &value) noexcept {
        if (film_accumulation == PathFilmAccumulation::atomic) {
            atomic_add_float4(buffer, index, value);
        } else {
            buffer.write(index, value);
        }
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
    if (film_accumulation == PathFilmAccumulation::atomic) {
        sample_count.atomic(pixel).fetch_add(completed);
    } else {
        sample_count.write(pixel, completed);
    }
}

}// namespace psycles::luisa_backend::detail

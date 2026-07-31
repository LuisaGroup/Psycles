#include <psycles/luisa/volume_majorant_prepass.h>

#include <limits>

#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_sampler.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

VolumeMajorantPrepass::VolumeMajorantPrepass(
    const SurfaceDispatch &surfaces,
    const VolumeStackEntryPointProvider &points) noexcept
    : _surfaces{surfaces},
      _points{points} {}

VolumeMajorantCellExtrema
VolumeMajorantPrepass::evaluate_cell(
    const VolumeStackEntry &entry,
    const ShaderServices &services,
    const VolumeMajorantGrid &grid,
    UInt cell_index) const noexcept {
    constexpr auto resolution =
        volume_majorant_grid_resolution;
    const auto xy = cell_index / resolution;
    const auto coordinate = make_uint3(
        cell_index % resolution,
        xy % resolution,
        xy / resolution);
    const auto cell_size =
        (grid.maximum - grid.minimum) /
        static_cast<float>(resolution);
    const auto cell_minimum =
        grid.minimum +
        make_float3(
            cast<float>(coordinate.x),
            cast<float>(coordinate.y),
            cast<float>(coordinate.z)) *
            cell_size;
    const auto padding =
        cell_size *
        volume_majorant_voxel_padding;
    const auto sample_minimum =
        cell_minimum - padding;
    const auto sample_size =
        cell_size + 2.0f * padding;

    Float minimum =
        std::numeric_limits<float>::max();
    Float maximum =
        -std::numeric_limits<float>::max();
    Float object_density = 1.0f;
    UInt sample = 0u;
    $while(sample <
           volume_majorant_samples_per_cell) {
        const auto sample_index =
            cell_index *
                volume_majorant_samples_per_cell +
            sample;
        const auto random =
            cycles_sampler::
                sobol_burley_sample_3d(
                    sample_index,
                    0u,
                    0u,
                    0xffffffffu);
        const auto object_position =
            sample_minimum +
            random * sample_size;
        const auto world_position =
            (grid.object_to_world *
             make_float4(
                 object_position, 1.0f))
                .xyz();
        const VolumeShadingState state{
            .position = world_position,
            .incoming = make_float3(0.0f),
            .ray_visibility =
                contract::visibility_bit(
                    contract::RayVisibility::
                        camera),
            .ray_events = 0u,
            .ray_depth = 0u,
            .diffuse_depth = 0u,
            .glossy_depth = 0u,
            .transparent_depth = 0u,
            .transmission_depth = 0u,
            .ray_length = 0.0f,
            .time = 0.5f};
        const auto shading =
            _points.emit(entry, state);
        const auto coefficients =
            _surfaces.volume_coefficients(
                entry.surface_tag,
                services,
                shading.point,
                VolumeQuery{
                    .object_density =
                        shading.object_density,
                    .evaluate_emission = true});
        const auto extinction =
            luisa::compute::max(
                coefficients.sigma_t.x,
                luisa::compute::max(
                    coefficients.sigma_t.y,
                    coefficients.sigma_t.z));
        const auto emission =
            luisa::compute::max(
                coefficients.emission.x,
                luisa::compute::max(
                    coefficients.emission.y,
                    coefficients.emission.z));
        const auto sigma =
            luisa::compute::max(
                extinction, emission);
        minimum =
            luisa::compute::min(
                minimum, sigma);
        maximum =
            luisa::compute::max(
                maximum, sigma);
        object_density =
            shading.object_density;
        sample += 1u;
    };
    return {
        .minimum = minimum / object_density,
        .maximum = maximum / object_density};
}

}// namespace psycles::luisa_backend

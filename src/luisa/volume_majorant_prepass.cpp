/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */
#include "path_tracer_cycles_svm_volume.h"

#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_sampler.h>
#include <limits>

namespace psycles::luisa_backend::detail {

VolumeMajorantCellExtrema evaluate_cycles_svm_volume_density_cell(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    const VolumeStackEntry &entry, const VolumeMajorantGrid &grid,
    UInt cell_index) noexcept {
    namespace abi = compiler::cycles_svm;
    const auto resolution = grid.resolution;
    const auto xy = cell_index / resolution;
    const auto coordinate = make_uint3(cell_index % resolution, xy % resolution, xy / resolution);
    const auto cell_size = (grid.maximum - grid.minimum) / cast<float>(resolution);
    const auto cell_minimum = grid.minimum +
        make_float3(cast<float>(coordinate.x), cast<float>(coordinate.y), cast<float>(coordinate.z)) * cell_size;
    const auto padding = cell_size * volume_majorant_voxel_padding;
    const auto sample_minimum = cell_minimum - padding;
    const auto sample_size = cell_size + 2.0f * padding;
    const cycles_svm::PathState state{cycles_svm::path_ray_visibility_camera, 0u};
    const PathCyclesSvmVolumeShader shader{
        scene, parameters, sample_minimum, make_float3(0.0f), 0.0f, 0.5f,
        entry.object, state,
        cycles_noise::hash_uint3(cell_index ^ 0x15b4f88du, 0u, 0u), false, true};
    const auto samples = select(volume_majorant_samples_per_cell, 1u, shader.homogeneous(entry));
    Bool transform = false;
    $if(entry.object != cycles_svm::object_none) {
        transform = (scene->cycles_svm->objects->object_flag_buffer->read(entry.object) &
                     static_cast<unsigned>(abi::SD_OBJECT_TRANSFORM_APPLIED)) == 0u;
    };
    const auto object_to_world = shader.object_transform(entry, false);
    Float minimum = std::numeric_limits<float>::max();
    Float maximum = -std::numeric_limits<float>::max();
    UInt sample = 0u;
    $while(sample < samples) {
        const auto random = cycles_sampler::sobol_burley_sample_3d(
            cell_index * samples + sample, 0u, 0u, 0xffffffffu);
        Float3 position = sample_minimum + random * sample_size;
        $if(transform) { position = (object_to_world * make_float4(position, 1.0f)).xyz(); };
        const auto coefficients = shader.evaluate_entry(
            entry, position, cycles_svm::kernel_feature_node_mask_volume &
                             ~cycles_svm::kernel_feature_node_light_path);
        const auto sigma = max(
            max(coefficients.sigma_t.x, max(coefficients.sigma_t.y, coefficients.sigma_t.z)),
            max(coefficients.emission.x, max(coefficients.emission.y, coefficients.emission.z)));
        minimum = min(minimum, sigma);
        maximum = max(maximum, sigma);
        sample += 1u;
    };
    const auto density = shader.object_density(entry);
    return {.minimum = minimum / density, .maximum = maximum / density};
}
} // namespace psycles::luisa_backend::detail

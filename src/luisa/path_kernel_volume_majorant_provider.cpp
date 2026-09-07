#include "path_kernel_volume_majorant_provider.h"

#include <limits>

namespace psycles::luisa_backend::detail {
namespace {
class SceneVolumeMajorantEntryProvider final : public VolumeMajorantEntryProvider {
    const PathCyclesSvmVolumeShader &_shader;
public:
    explicit SceneVolumeMajorantEntryProvider(const PathCyclesSvmVolumeShader &shader) noexcept
        : _shader{shader} {}

    VolumeMajorantEntrySpace entry_space(
        const VolumeStackEntry &entry, Float3 origin, Float3 direction) const noexcept override {
        const auto inverse = _shader.object_transform(entry, true);
        return {.ray_origin = (inverse * make_float4(origin, 1.0f)).xyz(),
                .ray_direction = (inverse * make_float4(direction, 0.0f)).xyz(),
                .object_density = _shader.object_density(entry)};
    }

    VolumeMajorantRuntimeExtrema extrema(
        const VolumeStackEntry &entry, const VolumeMajorantLeaf &leaf,
        Float object_density, Float shade_offset,
        Float3 world_ray_origin, Float3 world_ray_direction) const noexcept override {
        auto result = VolumeMajorantEntryProvider::extrema(
            entry, leaf, object_density, shade_offset, world_ray_origin, world_ray_direction);
        const auto flags = _shader.shader_flags(entry);
        $if(!_shader.camera_ray() &
            ((flags & static_cast<unsigned>(compiler::cycles_svm::SD_HAS_LIGHT_PATH_NODE)) != 0u)) {
            const auto heterogeneous = !_shader.homogeneous(entry);
            const auto samples = select(1u, 4u, heterogeneous);
            const auto offset = select(0.5f, shade_offset, heterogeneous);
            const auto step = (leaf.maximum - leaf.minimum) / cast<float>(samples);
            Float minimum = std::numeric_limits<float>::max();
            Float maximum = -std::numeric_limits<float>::max();
            UInt index = 0u;
            $while(index < samples) {
                const auto t = leaf.minimum + (offset + cast<float>(index)) * step;
                // Entry evaluation, including shadow visibility and zeroing
                // E/T, is exactly the native volume_estimate_extrema policy.
                const auto coefficients = _shader.evaluate_entry(
                    entry, world_ray_origin + world_ray_direction * t);
                const auto sigma = max(
                    max(coefficients.sigma_t.x, max(coefficients.sigma_t.y, coefficients.sigma_t.z)),
                    max(coefficients.emission.x, max(coefficients.emission.y, coefficients.emission.z)));
                minimum = min(minimum, sigma);
                maximum = max(maximum, sigma);
                index += 1u;
            };
            result.minimum = minimum;
            result.maximum = select(maximum, max(0.5f, maximum * 1.5f), heterogeneous);
        };
        return result;
    }
};
} // namespace

std::unique_ptr<VolumeMajorantEntryProvider>
make_scene_volume_majorant_entry_provider(const PathCyclesSvmVolumeShader &shader) {
    return std::make_unique<SceneVolumeMajorantEntryProvider>(shader);
}
} // namespace psycles::luisa_backend::detail

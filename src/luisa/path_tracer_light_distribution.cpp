#include "path_tracer_light_distribution.h"

#include <psycles/sampling/light_distribution.h>

namespace psycles::luisa_backend::detail {

LightDistributionSampleCallable
make_light_distribution_sample_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    using sampling::LightDistributionEmitterKind;

    return [scene](Float sample) noexcept {
        Var<LightDistributionGpu> result;
        if (scene->light_distribution_count == 0u) {
            result.cumulative = 0.0f;
            result.selection_pdf = 0.0f;
            result.kind = static_cast<std::uint32_t>(
                LightDistributionEmitterKind::sentinel);
            result.index = 0u;
            return result;
        }

        UInt first = 0u;
        UInt length =
            scene->light_distribution_count + 1u;
        $while (length > 0u) {
            UInt half_length = length >> 1u;
            UInt middle = first + half_length;
            Var<LightDistributionGpu> entry =
                scene->light_distribution_buffer->read(
                    middle);
            $if (sample < entry.cumulative) {
                length = half_length;
            }
            $else {
                first = middle + 1u;
                length =
                    length - half_length - 1u;
            };
        };

        UInt selected = 0u;
        $if (first > 0u) {
            selected = first - 1u;
        };
        selected = min(
            selected,
            scene->light_distribution_count - 1u);
        result =
            scene->light_distribution_buffer->read(selected);
        return result;
    };
}

}// namespace psycles::luisa_backend::detail

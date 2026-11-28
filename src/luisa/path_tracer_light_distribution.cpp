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

        const bool use_compact_cdf =
            static_cast<bool>(scene->light_distribution_cdf_buffer);
        const bool use_guide =
            use_compact_cdf &&
            scene->light_distribution_cdf_guide_bins != 0u &&
            static_cast<bool>(scene->light_distribution_cdf_guide_buffer);
        UInt first = 0u;
        UInt length =
            scene->light_distribution_count + 1u;
        if (use_guide) {
            const auto bins = scene->light_distribution_cdf_guide_bins;
            const auto bins_f = static_cast<float>(bins);
            UInt bucket = select(
                bins - 1u,
                select(
                    0u,
                    min(cast<luisa::uint>(sample * bins_f), bins - 1u),
                    sample >= 0.0f),
                sample < 1.0f);
            first = scene->light_distribution_cdf_guide_buffer->read(bucket);
            UInt last =
                scene->light_distribution_cdf_guide_buffer->read(bucket + 1u);
            length = last - first;
        }
        $while (length > 0u) {
            UInt half_length = length >> 1u;
            UInt middle = first + half_length;
            Float cdf =
                use_compact_cdf
                    ? scene->light_distribution_cdf_buffer->read(middle)
                    : scene->light_distribution_buffer->read(middle).cumulative;
            $if (sample < cdf) {
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
        if (use_compact_cdf &&
            scene->light_distribution_payload_buffer.has_value()) {
            const auto payload =
                (*scene->light_distribution_payload_buffer)->read(selected);
            result.cumulative = 0.0f;
            result.selection_pdf = payload.selection_pdf;
            result.kind = payload.kind;
            result.index = payload.index;
            result.emitter_id = payload.emitter_id;
        } else {
            result =
                scene->light_distribution_buffer->read(selected);
        }
        return result;
    };
}

}// namespace psycles::luisa_backend::detail

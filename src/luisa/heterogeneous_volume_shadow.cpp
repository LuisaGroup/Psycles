#include <psycles/luisa/cycles_spectrum.h>
#include <psycles/luisa/heterogeneous_volume_shadow.h>

#include <type_traits>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

template <bool Spectral, typename ThroughputT>
[[nodiscard]] auto emit_shadow_impl(
    VolumeMajorantSegmentSequence &segments,
    const HeterogeneousVolumeTrackingRandomSource &random,
    const HeterogeneousVolumeCollisionProvider &collisions,
    ThroughputT throughput,
    UInt tracking_rng_offset,
    Bool secondary_terminated) noexcept {
    const auto zero_color = []() noexcept -> ThroughputT {
        if constexpr (Spectral) { return make_float4(0.0f); } else { return make_float3(0.0f); }
    };
    const HeterogeneousVolumeTransmittance transmittance;
    UInt groups = 0u;
    UInt source_segments = 0u;
    UInt closure_evaluations = 0u;
    Bool traversal_exhausted = false;
    Bool lookup_complete = true;
    Bool throughput_terminated = false;
    Bool first_group = true;

    const auto initial = segments.current();
    Bool active = initial.valid;
    lookup_complete = initial.lookup_complete;
    $if(!lookup_complete) {
        throughput = zero_color();
        active = false;
    };

    $while(active) {
        tracking_rng_offset += heterogeneous_volume_tracking_rng_stride;

        Bool has_segment = true;
        $if(!first_group) {
            has_segment = segments.advance(
                random.shade_offset(tracking_rng_offset));
            traversal_exhausted |= !has_segment;
        };
        first_group = false;

        $if(has_segment) {
            const auto first = segments.current();
            Float group_minimum = first.minimum;
            Float group_maximum = first.maximum;
            Float sigma_minimum = first.sigma_minimum;
            Float sigma_maximum = first.sigma_maximum;
            UInt group_object = first.object;
            UInt group_shader = first.shader;
            UInt group_node = first.node;
            Bool group_no_overlap = first.no_overlap;
            Bool group_complete = first.lookup_complete;
            source_segments += 1u;
            lookup_complete &= group_complete;

            Bool collecting =
                group_complete &
                ((sigma_maximum - sigma_minimum) *
                     (group_maximum - group_minimum) <
                 1.0f);
            $while(collecting) {
                const auto advanced = segments.advance(
                    random.shade_offset(tracking_rng_offset));
                $if(advanced) {
                    const auto next = segments.current();
                    group_maximum = next.maximum;
                    sigma_minimum = min(sigma_minimum, next.sigma_minimum);
                    sigma_maximum = max(sigma_maximum, next.sigma_maximum);
                    group_complete &= next.lookup_complete;
                    lookup_complete &= next.lookup_complete;
                    source_segments += 1u;
                    collecting =
                        group_complete &
                        ((sigma_maximum - sigma_minimum) *
                             (group_maximum - group_minimum) <
                         1.0f);
                }
                $else {
                    traversal_exhausted = true;
                    collecting = false;
                };
            };

            $if(group_complete) {
                const VolumeMajorantSegment coalesced{
                    .minimum = group_minimum,
                    .maximum = group_maximum,
                    .sigma_minimum = sigma_minimum,
                    .sigma_maximum = sigma_maximum,
                    .object = group_object,
                    .shader = group_shader,
                    .node = group_node,
                    .valid = true,
                    .no_overlap = group_no_overlap,
                    .lookup_complete = group_complete};
                Float magnitude;
                if constexpr (Spectral) {
                    const auto estimate = transmittance.evaluate_spectral(
                        coalesced, group_minimum, group_maximum,
                        random, collisions, tracking_rng_offset);
                    throughput *= estimate.transmittance;
                    closure_evaluations += estimate.evaluations;
                    magnitude = cycles_spectrum::max_abs_active(
                        throughput, secondary_terminated);
                } else {
                    static_cast<void>(secondary_terminated);
                    const auto estimate = transmittance.evaluate(
                        coalesced, group_minimum, group_maximum,
                        random, collisions, tracking_rng_offset);
                    throughput *= estimate.transmittance;
                    closure_evaluations += estimate.evaluations;
                    magnitude = max(
                        abs(throughput.x),
                        max(abs(throughput.y), abs(throughput.z)));
                }
                groups += 1u;
                throughput_terminated =
                    magnitude < heterogeneous_volume_shadow_throughput_epsilon;
                active = !throughput_terminated;
            }
            $else {
                throughput = zero_color();
                active = false;
            };
        }
        $else {
            active = false;
        };
    };

    if constexpr (Spectral) {
        return HeterogeneousVolumeShadowResultSpectral{
            .throughput = std::move(throughput),
            .next_tracking_rng_offset = tracking_rng_offset,
            .groups = groups,
            .source_segments = source_segments,
            .closure_evaluations = closure_evaluations,
            .traversal_exhausted = traversal_exhausted,
            .lookup_complete = lookup_complete,
            .throughput_terminated = throughput_terminated};
    } else {
        return HeterogeneousVolumeShadowResult{
            .throughput = std::move(throughput),
            .next_tracking_rng_offset = tracking_rng_offset,
            .groups = groups,
            .source_segments = source_segments,
            .closure_evaluations = closure_evaluations,
            .traversal_exhausted = traversal_exhausted,
            .lookup_complete = lookup_complete,
            .throughput_terminated = throughput_terminated};
    }
}

class HeterogeneousVolumeShadowComponentImpl final
    : public HeterogeneousVolumeShadowComponent {

  public:
    HeterogeneousVolumeShadowResult emit(
        VolumeMajorantSegmentSequence &segments,
        const HeterogeneousVolumeTrackingRandomSource &random,
        const HeterogeneousVolumeCollisionProvider &collisions,
        Float3 throughput,
        UInt tracking_rng_offset) const noexcept override {
        return emit_shadow_impl<false>(
            segments, random, collisions, std::move(throughput),
            std::move(tracking_rng_offset), Bool{false});
    }

    HeterogeneousVolumeShadowResultSpectral emit_spectral(
        VolumeMajorantSegmentSequence &segments,
        const HeterogeneousVolumeTrackingRandomSource &random,
        const HeterogeneousVolumeCollisionProvider &collisions,
        Float4 throughput,
        UInt tracking_rng_offset,
        Bool secondary_terminated) const noexcept override {
        return emit_shadow_impl<true>(
            segments, random, collisions, std::move(throughput),
            std::move(tracking_rng_offset), std::move(secondary_terminated));
    }
};

}// namespace

std::unique_ptr<HeterogeneousVolumeShadowComponent>
make_heterogeneous_volume_shadow_component() {
    return std::make_unique<
        HeterogeneousVolumeShadowComponentImpl>();
}

}// namespace psycles::luisa_backend

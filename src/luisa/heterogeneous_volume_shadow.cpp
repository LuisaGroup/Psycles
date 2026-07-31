#include <psycles/luisa/heterogeneous_volume_shadow.h>

#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

class HeterogeneousVolumeShadowComponentImpl final
    : public HeterogeneousVolumeShadowComponent {

  public:
    HeterogeneousVolumeShadowResult emit(
        VolumeMajorantSegmentSequence &segments,
        const HeterogeneousVolumeTrackingRandomSource
            &random,
        const HeterogeneousVolumeCollisionProvider
            &collisions,
        Float3 throughput,
        UInt tracking_rng_offset)
        const noexcept override {
        const HeterogeneousVolumeTransmittance
            transmittance;
        UInt groups = 0u;
        UInt source_segments = 0u;
        UInt closure_evaluations = 0u;
        Bool traversal_exhausted = false;
        Bool lookup_complete = true;
        Bool throughput_terminated = false;
        Bool first_group = true;

        const auto initial = segments.current();
        Bool active = initial.valid;
        lookup_complete =
            initial.lookup_complete;
        $if(!lookup_complete) {
            throughput =
                make_float3(0.0f);
            active = false;
        };

        $while(active) {
            // volume_octree_advance_shadow() advances one complete Cycles
            // bounce block before every attempted coalesced interval,
            // including the final failed attempt after traversal exhaustion.
            tracking_rng_offset +=
                heterogeneous_volume_tracking_rng_stride;

            Bool has_segment = true;
            $if(!first_group) {
                has_segment =
                    segments.advance(
                        random.shade_offset(
                            tracking_rng_offset));
                traversal_exhausted |=
                    !has_segment;
            };
            first_group = false;

            $if(has_segment) {
                const auto first =
                    segments.current();
                Float group_minimum =
                    first.minimum;
                Float group_maximum =
                    first.maximum;
                Float sigma_minimum =
                    first.sigma_minimum;
                Float sigma_maximum =
                    first.sigma_maximum;
                UInt group_object =
                    first.object;
                UInt group_shader =
                    first.shader;
                UInt group_node =
                    first.node;
                Bool group_no_overlap =
                    first.no_overlap;
                Bool group_complete =
                    first.lookup_complete;
                source_segments += 1u;
                lookup_complete &=
                    group_complete;

                Bool collecting =
                    group_complete &
                    ((sigma_maximum -
                      sigma_minimum) *
                         (group_maximum -
                          group_minimum) <
                     1.0f);
                $while(collecting) {
                    const auto advanced =
                        segments.advance(
                            random.shade_offset(
                                tracking_rng_offset));
                    $if(advanced) {
                        const auto next =
                            segments.current();
                        group_maximum =
                            next.maximum;
                        sigma_minimum =
                            min(
                                sigma_minimum,
                                next.sigma_minimum);
                        sigma_maximum =
                            max(
                                sigma_maximum,
                                next.sigma_maximum);
                        group_complete &=
                            next.lookup_complete;
                        lookup_complete &=
                            next.lookup_complete;
                        source_segments += 1u;
                        collecting =
                            group_complete &
                            ((sigma_maximum -
                              sigma_minimum) *
                                 (group_maximum -
                                  group_minimum) <
                             1.0f);
                    }
                    $else {
                        traversal_exhausted =
                            true;
                        collecting = false;
                    };
                };

                $if(group_complete) {
                    const auto estimate =
                        transmittance.evaluate(
                            {.minimum =
                                 group_minimum,
                             .maximum =
                                 group_maximum,
                             .sigma_minimum =
                                 sigma_minimum,
                             .sigma_maximum =
                                 sigma_maximum,
                             .object =
                                 group_object,
                             .shader =
                                 group_shader,
                             .node = group_node,
                             .valid = true,
                             .no_overlap =
                                 group_no_overlap,
                             .lookup_complete =
                                 group_complete},
                            group_minimum,
                            group_maximum,
                            random,
                            collisions,
                            tracking_rng_offset);
                    throughput *=
                        estimate.transmittance;
                    closure_evaluations +=
                        estimate.evaluations;
                    groups += 1u;
                    const auto magnitude =
                        max(
                            abs(throughput.x),
                            max(
                                abs(throughput.y),
                                abs(throughput.z)));
                    throughput_terminated =
                        magnitude <
                        heterogeneous_volume_shadow_throughput_epsilon;
                    active =
                        !throughput_terminated;
                }
                $else {
                    throughput =
                        make_float3(0.0f);
                    active = false;
                };
            }
            $else {
                active = false;
            };
        };

        return {
            .throughput =
                std::move(throughput),
            .next_tracking_rng_offset =
                tracking_rng_offset,
            .groups = groups,
            .source_segments =
                source_segments,
            .closure_evaluations =
                closure_evaluations,
            .traversal_exhausted =
                traversal_exhausted,
            .lookup_complete =
                lookup_complete,
            .throughput_terminated =
                throughput_terminated};
    }
};

}// namespace

std::unique_ptr<HeterogeneousVolumeShadowComponent>
make_heterogeneous_volume_shadow_component() {
    return std::make_unique<
        HeterogeneousVolumeShadowComponentImpl>();
}

}// namespace psycles::luisa_backend

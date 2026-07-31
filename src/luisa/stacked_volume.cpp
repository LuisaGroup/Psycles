#include <psycles/luisa/stacked_volume.h>

namespace psycles::luisa_backend {

StackedVolumeEvaluator::StackedVolumeEvaluator(
    const SurfaceDispatch &surfaces,
    const VolumeStackEntryPointProvider &points) noexcept
    : _surfaces{surfaces},
      _points{points} {}

VolumeCoefficients StackedVolumeEvaluator::evaluate(
    const VolumeStack &stack,
    const ShaderServices &services,
    const VolumeShadingState &state,
    Bool evaluate_emission,
    VolumePhaseSet *phases) const noexcept {
    auto result = VolumeCoefficients::zero();
    UInt index = 0u;
    $while(index < stack.count()) {
        const auto entry = stack.entry(index);
        const auto should_evaluate =
            _points.should_evaluate(
                entry, state);
        $if(entry.valid & should_evaluate) {
            const auto shading =
                _points.emit(entry, state);
            const auto coefficients =
                _surfaces.evaluate_volume(
                    entry.surface_tag,
                    services,
                    shading.point,
                    VolumeQuery{
                        .object_density =
                            shading.object_density,
                        .evaluate_emission =
                            evaluate_emission},
                    phases);
            result.sigma_t +=
                coefficients.sigma_t;
            result.sigma_s +=
                coefficients.sigma_s;
            result.emission +=
                coefficients.emission;
            result.has_extinction =
                result.has_extinction |
                coefficients.has_extinction;
            result.has_scatter =
                result.has_scatter |
                coefficients.has_scatter;
            result.has_emission =
                result.has_emission |
                coefficients.has_emission;

            // Cycles skips volume_shader_merge_closures() for stack entry
            // zero and runs it after every later entry. This preserves raw
            // duplicate closures in a single medium while merging equal
            // phases once overlapping media have been accumulated.
            if (phases != nullptr) {
                $if(index > 0u) {
                    phases->merge_equal();
                };
            }
        };
        index += 1u;
    };
    if (phases != nullptr) {
        phases->truncate();
    }
    return result;
}

}// namespace psycles::luisa_backend

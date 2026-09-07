#include <psycles/luisa/heterogeneous_volume_collision.h>

namespace psycles::luisa_backend {
namespace {
class StackedHeterogeneousVolumeCollisionProvider final
    : public HeterogeneousVolumeCollisionProvider {
    const VolumeShaderEvaluator &_shader;
    const VolumeStack &_stack;
    Float3 _origin;
    Float3 _direction;
public:
    StackedHeterogeneousVolumeCollisionProvider(
        const VolumeShaderEvaluator &shader, const VolumeStack &stack,
        Float3 origin, Float3 direction) noexcept
        : _shader{shader}, _stack{stack}, _origin{origin}, _direction{direction} {}

    VolumeCoefficients evaluate(Float distance, Bool evaluate_emission,
                                VolumePhaseSet *phases) const noexcept override {
        // The consumer query does not specialize the SVM feature mask.
        // Cycles computes emission even when the caller only uses extinction.
        static_cast<void>(evaluate_emission);
        return _shader.evaluate(_stack, _origin + _direction * distance, phases);
    }
};
} // namespace

std::unique_ptr<HeterogeneousVolumeCollisionProvider>
make_stacked_heterogeneous_volume_collision_provider(
    const VolumeShaderEvaluator &shader, const VolumeStack &stack,
    Float3 ray_origin, Float3 ray_direction) {
    return std::make_unique<StackedHeterogeneousVolumeCollisionProvider>(
        shader, stack, ray_origin, ray_direction);
}
} // namespace psycles::luisa_backend

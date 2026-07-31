#include <psycles/luisa/heterogeneous_volume_collision.h>

#include <utility>

namespace psycles::luisa_backend {
namespace {

class StackedHeterogeneousVolumeCollisionProvider final
    : public HeterogeneousVolumeCollisionProvider {

  private:
    const SurfaceDispatch &_surfaces;
    std::shared_ptr<
        const VolumeStackEntryPointProvider>
        _points;
    const VolumeStack &_stack;
    const ShaderServices &_services;
    const VolumeShadingState &_base_state;
    Float3 _ray_origin;
    Float3 _ray_direction;

  public:
    StackedHeterogeneousVolumeCollisionProvider(
        const SurfaceDispatch &surfaces,
        std::shared_ptr<
            const VolumeStackEntryPointProvider>
            points,
        const VolumeStack &stack,
        const ShaderServices &services,
        const VolumeShadingState &base_state,
        Float3 ray_origin,
        Float3 ray_direction) noexcept
        : _surfaces{surfaces},
          _points{std::move(points)},
          _stack{stack},
          _services{services},
          _base_state{base_state},
          _ray_origin{std::move(ray_origin)},
          _ray_direction{
              std::move(ray_direction)} {}

    VolumeCoefficients evaluate(
        Float distance,
        Bool evaluate_emission,
        VolumePhaseSet *phases)
        const noexcept override {
        const VolumeShadingState state{
            .position =
                _ray_origin +
                _ray_direction * distance,
            .incoming =
                _base_state.incoming,
            .ray_visibility =
                _base_state.ray_visibility,
            .ray_events =
                _base_state.ray_events,
            .ray_depth =
                _base_state.ray_depth,
            .diffuse_depth =
                _base_state.diffuse_depth,
            .glossy_depth =
                _base_state.glossy_depth,
            .transparent_depth =
                _base_state
                    .transparent_depth,
            .transmission_depth =
                _base_state
                    .transmission_depth,
            // shader_setup_from_volume() deliberately initializes this to
            // zero in current Cycles.
            .ray_length =
                _base_state.ray_length,
            .time = _base_state.time};
        const StackedVolumeEvaluator evaluator{
            _surfaces, *_points};
        return evaluator.evaluate(
            _stack,
            _services,
            state,
            evaluate_emission,
            phases);
    }
};

}// namespace

std::unique_ptr<HeterogeneousVolumeCollisionProvider>
make_stacked_heterogeneous_volume_collision_provider(
    const SurfaceDispatch &surfaces,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    const VolumeStack &stack,
    const ShaderServices &services,
    const VolumeShadingState &base_state,
    Float3 ray_origin,
    Float3 ray_direction) {
    return std::make_unique<
        StackedHeterogeneousVolumeCollisionProvider>(
        surfaces,
        std::move(points),
        stack,
        services,
        base_state,
        std::move(ray_origin),
        std::move(ray_direction));
}

}// namespace psycles::luisa_backend

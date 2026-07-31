#include "path_kernel_volume_shadow.h"

#include "path_kernel_volume_boundary.h"
#include "path_kernel_volume_point.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/stacked_volume.h>
#include <psycles/luisa/surface_ray.h>

#include <algorithm>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class HomogeneousVolumeShadowComponentImpl final
    : public HomogeneousVolumeShadowComponent {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::size_t _stack_size;
    std::shared_ptr<
        const VolumeStackEntryPointProvider>
        _points;
    std::shared_ptr<
        const TriangleVolumeBoundaryComponent>
        _boundary;

  public:
    explicit HomogeneousVolumeShadowComponentImpl(
        const PathKernelConfig &config)
        : _scene{config.scene},
          _stack_size{
              std::max(
                  std::size_t{
                      config.volume_stack_size},
                  std::size_t{1u})},
          _points{
              make_scene_volume_stack_entry_point_provider(
                  _scene)},
          _boundary{
              make_triangle_volume_boundary_component()} {}

    Float3 emit(
        const PathSampleContext &sample,
        const VolumeStack &path_stack,
        Var<luisa::compute::Ray> shadow_ray,
        UInt light_instance,
        UInt light_primitive)
        const noexcept override {
        VolumeStack shadow_stack{
            _stack_size};
        shadow_stack.copy_from(path_stack);
        const StackedVolumeEvaluator evaluator{
            _scene->surfaces,
            *_points};
        BufferShaderServices services{
            _scene->parameter_buffer,
            _scene->cycles_bsdf_table_buffer,
            _scene->texture_heap,
            _scene->heap,
            _scene->attribute_binding_slot,
            _scene->attribute_range_slot,
            _scene->nishita_texture_bindings,
            _scene->shader_color_space};
        const auto shader_state =
            cycles_path_state::
                shadow_shader_state(
                    sample.path_depth,
                    sample.diffuse_depth,
                    sample.glossy_depth,
                    sample.transparent_depth,
                    sample.transmission_depth);

        const auto ray_origin =
            shadow_ray->origin();
        const auto ray_direction =
            shadow_ray->direction();
        const auto initial_minimum =
            shadow_ray->t_min();
        Float current_minimum =
            initial_minimum;
        Float3 transmittance =
            make_float3(1.0f);
        Bool active = true;

        // Reduce to the nearest boundary on every iteration. Luisa ray-query
        // candidate order is unspecified; an any-hit stack update would
        // therefore be non-formal and backend-dependent.
        $while(active) {
            const auto committed =
                surface_ray::
                    closest_shadow_intersection(
                        _scene->accel,
                        shadow_ray,
                        surface_ray::
                            invalid_primitive,
                        surface_ray::
                            invalid_primitive,
                        light_instance,
                        light_primitive,
                        shadow_visibility);
            Float segment_end =
                shadow_ray->t_max();
            $if(!committed->miss()) {
                segment_end =
                    committed
                        ->committed_ray_t;
            };
            const auto segment_length =
                max(
                    segment_end -
                        current_minimum,
                    0.0f);
            const VolumeShadingState state{
                .position =
                    ray_origin +
                    ray_direction *
                        current_minimum,
                .incoming =
                    -ray_direction,
                .ray_visibility =
                    shader_state
                        .ray_visibility,
                .ray_events =
                    shader_state.ray_events,
                .ray_depth =
                    shader_state.ray_depth,
                .diffuse_depth =
                    shader_state.diffuse_depth,
                .glossy_depth =
                    shader_state.glossy_depth,
                .transparent_depth =
                    shader_state
                        .transparent_depth,
                .transmission_depth =
                    shader_state
                        .transmission_depth,
                .ray_length =
                    current_minimum -
                    initial_minimum,
                .time = 0.0f};
            const auto coefficients =
                evaluator.evaluate(
                    shadow_stack,
                    services,
                    state,
                    false);
            transmittance *=
                exp(
                    -coefficients.sigma_t *
                    segment_length);

            $if(committed->miss()) {
                active = false;
            }
            $else {
                const auto boundary =
                    _boundary->resolve(
                        _scene,
                        committed->inst,
                        committed->prim,
                        ray_direction);
                shadow_stack.cross_boundary(
                    boundary.primitive
                        .volume_stack_entry(),
                    boundary.back_facing,
                    boundary.primitive
                        .has_volume,
                    true);
                current_minimum =
                    surface_ray::
                        intersection_t_offset(
                            committed
                                ->committed_ray_t);
                shadow_ray->set_t_min(
                    current_minimum);
            };
        };
        return transmittance;
    }
};

}// namespace

std::unique_ptr<
    HomogeneousVolumeShadowComponent>
make_homogeneous_volume_shadow_component(
    const PathKernelConfig &config) {
    return std::make_unique<
        HomogeneousVolumeShadowComponentImpl>(
        config);
}

}// namespace psycles::luisa_backend::detail

#include "path_kernel_builder.h"
#include "path_kernel_scene_traversal.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathBounceSetupStageImpl final
    : public PathBounceSetupStage {

  private:
    std::shared_ptr<const SceneTraversalComponent> _traversal;
    bool _has_subsurface;

  public:
    explicit PathBounceSetupStageImpl(
        SceneTraversalStagePlan plan,
        bool has_subsurface)
        : _traversal{
              make_scene_traversal_component(
                  plan)},
          _has_subsurface{has_subsurface} {}

    PathBounceContext
    emit(PathSampleContext &sample,
         const UInt &path_step) const noexcept override {
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &scene = config.scene;
        auto &continuation_probability =
            sample.continuation_probability;
        auto &ray = sample.ray;
        const auto ray_visibility =
            sample.contracted_ray_visibility();
        auto &ray_source_object =
            sample.ray_source_object;
        auto &ray_source_primitive =
            sample.ray_source_primitive;
        auto &pending_subsurface_exit =
            sample.pending_subsurface_exit;
        auto &pending_subsurface_hit =
            sample.pending_subsurface_hit;

        continuation_probability = 1.0f;
        // `has_subsurface` is proved over all reachable material graphs and
        // their bound parameters while constructing the immutable scene. If
        // false, no scatter can produce a BSSRDF exit, so the pending state is
        // inductively false and recording its state machine would only widen
        // every coroutine transition.
        Bool subsurface_exit = false;
        subsurface_exit.set_name(
            "path_bounce_subsurface_exit");
        Var<luisa::compute::CommittedHit> hit;
        hit.set_name("path_bounce_hit");
        if (_has_subsurface) {
            subsurface_exit = pending_subsurface_exit;
            $if(subsurface_exit) {
                // The local BSSRDF traversal has already selected the exact
                // intersection. Preserve it directly, as Cycles does between
                // INTERSECT_SUBSURFACE and SHADE_SURFACE.
                hit = pending_subsurface_hit.materialize_surface();
                pending_subsurface_exit = false;
            }
            $else {
                hit = _traversal->closest(
                    scene, ray, ray_visibility,
                    {.object = ray_source_object,
                     .primitive = ray_source_primitive});
            };
        } else {
            // Match Cycles' RaySelfPrimitives contract: the previous
            // committed primitive is rejected by identity during traversal.
            // This remains independent of the geometric origin offset.
            hit = _traversal->closest(
                scene, ray, ray_visibility,
                {.object = ray_source_object,
                 .primitive = ray_source_primitive});
        }

        return {
            .sample = sample,
            .path_step = path_step,
            .random_state = nullptr,
            .hit = std::move(hit),
            .subsurface_exit = std::move(subsurface_exit)};
    }
};

}// namespace

std::unique_ptr<PathBounceSetupStage>
make_path_bounce_setup_stage(
    SceneTraversalStagePlan plan,
    bool has_subsurface) {
    return std::make_unique<
        PathBounceSetupStageImpl>(
            plan,
            has_subsurface);
}

}// namespace psycles::luisa_backend::detail

#include "path_kernel_surface_queue.h"

#include "path_kernel_primitive_material.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceQueueKeyStageImpl final : public SurfaceQueueKeyStage {

  private:
    ScenePrimitiveStagePlan _plan;
    std::shared_ptr<const PrimitiveMaterialComponent> _material{
        make_primitive_material_component()};

  public:
    explicit SurfaceQueueKeyStageImpl(
        ScenePrimitiveStagePlan plan) noexcept
        : _plan{plan} {}

    [[nodiscard]] UInt
    emit(const PathBounceContext &bounce) const noexcept override {
        const auto &scene = bounce.sample.invocation.config.scene;
        const auto &hit = bounce.hit;
        const UInt instance_id = hit->inst;
        Var<InstanceGpu> instance =
            scene->instance_buffer->read(instance_id);
        Var<GeometryGpu> geometry =
            scene->geometry_buffer->read(instance.geometry_index);
        UInt material_slot = 0u;

        const auto resolve_triangle = [&] noexcept {
            material_slot = _material->triangle_material_slot(
                scene, geometry, hit->prim);
        };
        const auto resolve_curve = [&] noexcept {
            const auto segment =
                scene->heap
                    ->buffer<CurveSegmentGpu>(geometry.bindless_base)
                    .read(hit->prim);
            material_slot = _material->curve_material_slot(
                scene, geometry, segment);
        };

        if (_plan.mixed()) {
            $if(hit->is_procedural()) {
                resolve_curve();
            }
            $else {
                resolve_triangle();
            };
        } else if (_plan.curves) {
            resolve_curve();
        } else {
            resolve_triangle();
        }

        // Resolve only the shared binding relation. Cycles shader identity,
        // volume flags, and emission metadata are deliberately absent from
        // this scheduler-only continuation. SurfaceDispatch::surface_tag
        // groups exactly the graph implementations expanded by the JIT.
        auto binding = _material->resolve_binding(
            scene, instance, geometry, material_slot);
        return std::move(binding.surface_tag);
    }
};

}// namespace

std::unique_ptr<SurfaceQueueKeyStage>
make_surface_queue_key_stage(ScenePrimitiveStagePlan plan) {
    return std::make_unique<SurfaceQueueKeyStageImpl>(plan);
}

}// namespace psycles::luisa_backend::detail

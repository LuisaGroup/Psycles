#include "path_kernel_surface_queue.h"

#include "path_kernel_primitive_material.h"
#include "cycles_shader_identity.h"

#include <psycles/luisa/cycles_svm.h>

#include <limits>
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
    emit(const std::shared_ptr<LuisaSceneData> &scene,
         const Var<luisa::compute::CommittedHit> &hit) const noexcept override {
        const UInt instance_id = hit->inst;
        Var<InstanceGpu> instance =
            scene->instance_buffer->read(instance_id);
        Var<GeometryGpu> geometry =
            scene->geometry_buffer->read(instance.geometry_index);

        // Cycles intersection_get_shader_from_isect_prim(): the native
        // interpreter's identity is the shader-table index, not the deduped
        // graph topology. Material overrides are already resolved in this
        // geometry image. Do not reintroduce the legacy binding indirections.
        if (scene->native_cycles_svm_surface) {
            LUISA_ASSERT(scene->cycles_svm && scene->cycles_svm->geometry,
                         "Native surface sorting requires finalized Cycles geometry.");
            const auto &native = *scene->cycles_svm->geometry;
            UInt shader = 0u;
            const auto triangle_shader = [&] noexcept {
                shader = native.triangle_shader_buffer->read(
                    geometry.cycles_primitive_offset + hit->prim);
            };
            const auto curve_shader = [&] noexcept {
                const auto segment = scene->heap
                    ->buffer<CurveSegmentGpu>(geometry.bindless_base)
                    .read(hit->prim);
                shader = native.curve_buffer->read(segment.cycles_curve_index)
                             .shader_id.cast<std::uint32_t>();
            };
            if (_plan.mixed()) {
                $if(hit->is_procedural()) { curve_shader(); }
                $else { triangle_shader(); };
            } else if (_plan.curves) {
                curve_shader();
            } else {
                triangle_shader();
            }
            return shader & cycles_shader_identity::shader_mask;
        }
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

        // The legacy diagnostic evaluator still expands graph implementations
        // by SurfaceDispatch::surface_tag. Keep its own coherence key.
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

std::uint32_t surface_queue_key_range(const LuisaSceneData &scene) noexcept {
    LUISA_ASSERT(!scene.native_cycles_svm_surface || scene.cycles_svm,
                 "Native surface sorting requires a Cycles shader table.");
    const auto count = scene.native_cycles_svm_surface
                           ? scene.cycles_svm->compilation.kernel_shaders.size()
                           : scene.surfaces.size();
    LUISA_ASSERT(count <= std::numeric_limits<std::uint32_t>::max(),
                 "Surface queue key range {} exceeds the uint32 scheduler ABI.",
                 count);
    return static_cast<std::uint32_t>(count);
}

}// namespace psycles::luisa_backend::detail

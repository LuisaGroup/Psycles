#include "path_kernel_builder.h"

#include <utility>

namespace psycles::luisa_backend::detail {

void PendingSubsurfaceHit::store_surface(
    UInt selected_instance,
    UInt selected_primitive,
    Float2 selected_barycentric,
    Float selected_ray_t) noexcept {
    instance = std::move(selected_instance);
    primitive = std::move(selected_primitive);
    barycentric = std::move(selected_barycentric);
    committed_ray_t = std::move(selected_ray_t);
}

void PendingSubsurfaceHit::store_surface(
    const Var<luisa::compute::CommittedHit> &source) noexcept {
    store_surface(source->inst,
                  source->prim,
                  source->bary,
                  source->committed_ray_t);
}

Var<luisa::compute::CommittedHit>
PendingSubsurfaceHit::materialize_surface() const noexcept {
    Var<luisa::compute::CommittedHit> result;
    result.inst = instance;
    result.prim = primitive;
    result.bary = barycentric;
    result.hit_type = static_cast<std::uint32_t>(
        luisa::compute::HitType::Surface);
    result.committed_ray_t = committed_ray_t;
    return result;
}

}// namespace psycles::luisa_backend::detail

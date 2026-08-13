#include "path_kernel_subsurface_intersection.h"

namespace psycles::luisa_backend::detail {

UInt subsurface_primary_instance(
    const std::shared_ptr<LuisaSceneData> &scene,
    Expr<std::uint32_t> local_instance) noexcept {
    LUISA_ASSERT(
        scene->subsurface_accel.has_value(),
        "BSSRDF local-instance mapping requires a local acceleration "
        "structure.");
    return (*scene->subsurface_accel)->instance_user_id(local_instance);
}

}// namespace psycles::luisa_backend::detail
